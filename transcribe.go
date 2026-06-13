package whisper

import (
	"bytes"
	"compress/zlib"
	"context"
	"errors"
	"math"
	"strings"
	"time"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

// Transcribe runs speech recognition on PCM audio samples (16kHz, mono, float32).
// Handles audio of any length via an internal sliding window.
func (m *Model) Transcribe(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.infer(ctx, samples, cfg, tokenTranscribe)
}

// Translate runs speech recognition and translates the result into English.
// Handles audio of any length via an internal sliding window.
func (m *Model) Translate(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.infer(ctx, samples, cfg, tokenTranslate)
}

func (m *Model) infer(ctx context.Context, samples []float32, cfg TranscribeConfig, taskToken int32) (*Result, error) {
	if m == nil || m.bridge == nil {
		return nil, errors.New("model is closed")
	}
	if len(samples) == 0 {
		return nil, errors.New("samples are empty")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	cfg.applyDefaults()

	mel, totalFrames := computeMelSpectrogram(samples, m.nMels, m.melFilters)
	duration := time.Duration(float64(len(samples)) / whisperSampleRate * float64(time.Second))

	lang := cfg.Language
	var langProb float32

	if lang == "" && m.IsMultilingual() {
		var err error
		lang, langProb, err = m.detectLanguageFromMel(mel, totalFrames, 0)
		if err != nil {
			return nil, err
		}
	} else if lang == "" {
		lang = "en"
		langProb = 1.0
	} else {
		langProb = 1.0
	}

	suppressTokens := m.tokenizer.SuppressedTokens(cfg.SuppressTokens)

	seek := 0
	var allTokens []int32
	promptResetSince := 0
	segIdx := 0

	if cfg.InitialPrompt != "" {
		initialTokens := m.tokenizer.Encode(" " + strings.TrimSpace(cfg.InitialPrompt))
		allTokens = append(allTokens, initialTokens...)
	}

	var segments []Segment

	for seek < totalFrames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		segmentSize := min(whisperNFrames, totalFrames-seek)
		wr, err := m.processWindow(processWindowParams{
			mel:            mel,
			totalFrames:    totalFrames,
			seek:           seek,
			segmentSize:    segmentSize,
			lang:           lang,
			previousTokens: allTokens[promptResetSince:],
			cfg:            cfg,
			taskToken:      taskToken,
			suppressTokens: suppressTokens,
			segIdx:         segIdx,
		})
		if err != nil {
			return nil, err
		}

		seek = wr.seek
		if wr.lang != "" {
			lang = wr.lang
		}
		allTokens = append(allTokens, wr.tokens...)
		segIdx = wr.segIdx
		segments = append(segments, wr.segments...)

		if !*cfg.ConditionOnPreviousText || wr.temperature > *cfg.PromptResetOnTemperature {
			promptResetSince = len(allTokens)
		}
	}

	var textBuf strings.Builder
	for i, seg := range segments {
		if i > 0 {
			textBuf.WriteByte(' ')
		}
		textBuf.WriteString(seg.Text)
	}

	return &Result{
		Text:     textBuf.String(),
		Segments: segments,
		Info: TranscriptionInfo{
			Language:            lang,
			LanguageProbability: langProb,
			Duration:            duration,
		},
	}, nil
}

// DetectLanguage identifies the spoken language from the first 30 seconds of audio.
func (m *Model) DetectLanguage(ctx context.Context, samples []float32) (LanguageDetection, error) {
	if m == nil || m.bridge == nil {
		return LanguageDetection{}, errors.New("model is closed")
	}
	if len(samples) == 0 {
		return LanguageDetection{}, errors.New("samples are empty")
	}
	if err := ctx.Err(); err != nil {
		return LanguageDetection{}, err
	}

	audio := padOrTrim(samples, whisperSampleRate*whisperChunkLen)
	mel, totalFrames := computeMelSpectrogram(audio, m.nMels, m.melFilters)
	window := extractMelWindow(mel, totalFrames, m.nMels, 0, whisperNFrames)

	enc, err := m.bridge.Encode(window, m.nMels, whisperNFrames)
	if err != nil {
		return LanguageDetection{}, err
	}
	defer enc.Free()

	result, err := m.bridge.DetectLanguage(enc)
	if err != nil {
		return LanguageDetection{}, err
	}
	return LanguageDetection{
		Language:    result.Language,
		Probability: result.Probability,
	}, nil
}

func (m *Model) detectLanguageFromMel(mel []float32, totalFrames, seekOffset int) (string, float32, error) {
	window := extractMelWindow(mel, totalFrames, m.nMels, seekOffset, whisperNFrames)
	enc, err := m.bridge.Encode(window, m.nMels, whisperNFrames)
	if err != nil {
		return "", 0, err
	}
	defer enc.Free()

	result, err := m.bridge.DetectLanguage(enc)
	if err != nil {
		return "", 0, err
	}
	return result.Language, result.Probability, nil
}

type processWindowParams struct {
	mel            []float32
	totalFrames    int
	seek           int
	segmentSize    int
	lang           string
	previousTokens []int32
	cfg            TranscribeConfig
	taskToken      int32
	suppressTokens []int32
	segIdx         int
}

type windowResult struct {
	segments    []Segment
	tokens      []int32
	seek        int
	segIdx      int
	lang        string
	temperature float32
}

func (m *Model) processWindow(p processWindowParams) (windowResult, error) {
	segmentDuration := float64(p.segmentSize) * timePerFrame
	timeOffset := float64(p.seek) * timePerFrame

	window := extractMelWindow(p.mel, p.totalFrames, m.nMels, p.seek, p.segmentSize)
	enc, err := m.bridge.Encode(window, m.nMels, whisperNFrames)
	if err != nil {
		return windowResult{}, err
	}
	defer enc.Free()

	lang := p.lang
	if p.cfg.Multilingual && m.IsMultilingual() {
		if result, dlErr := m.bridge.DetectLanguage(enc); dlErr == nil {
			lang = result.Language
		}
	}

	prompt := m.buildPrompt(lang, p.previousTokens, p.cfg, p.taskToken)

	genResult, avgLogProb, temperature, compressionRatio, genErr := m.generateWithFallback(enc, prompt, p.cfg, p.suppressTokens)
	if genErr != nil {
		return windowResult{}, genErr
	}

	if shouldSkipSegment(genResult, avgLogProb, p.cfg) {
		return windowResult{
			seek:        p.seek + p.segmentSize,
			segIdx:      p.segIdx,
			lang:        lang,
			temperature: temperature,
		}, nil
	}

	split := m.tokenizer.SplitSegmentsByTimestamps(genResult.SequenceIDs, timeOffset, p.segmentSize, segmentDuration, p.seek)

	var segments []Segment
	var tokens []int32
	segIdx := p.segIdx

	for _, seg := range split.segments {
		text := strings.TrimSpace(m.tokenizer.Decode(seg.tokens))
		if text == "" || seg.start == seg.end {
			continue
		}

		tokens = append(tokens, seg.tokens...)
		segIdx++

		var words []Word
		if !p.cfg.DisableTimestamps && p.cfg.WordTimestamps {
			words, _ = m.extractWordTimestamps(enc, seg.tokens, lang, p.taskToken, p.segmentSize, seg.start)
		}

		s := Segment{
			ID:               segIdx,
			Start:            secToDuration(seg.start),
			End:              secToDuration(seg.end),
			Text:             text,
			Words:            words,
			Temperature:      temperature,
			AvgLogProb:       avgLogProb,
			CompressionRatio: compressionRatio,
			NoSpeechProb:     genResult.NoSpeechProb,
		}
		if p.cfg.DisableTimestamps {
			s.Start = 0
			s.End = 0
		}
		segments = append(segments, s)
	}

	return windowResult{
		segments:    segments,
		tokens:      tokens,
		seek:        split.seek,
		segIdx:      segIdx,
		lang:        lang,
		temperature: temperature,
	}, nil
}

func (m *Model) buildPrompt(lang string, previousTokens []int32, cfg TranscribeConfig, taskToken int32) []int32 {
	var prompt []int32

	if len(previousTokens) > 0 || cfg.Hotwords != "" {
		prompt = append(prompt, tokenSOTprev)

		if cfg.Hotwords != "" {
			hw := m.tokenizer.Encode(" " + strings.TrimSpace(cfg.Hotwords))
			maxHW := maxTokenLength / 2
			if len(hw) >= maxHW {
				hw = hw[:maxHW-1]
			}
			prompt = append(prompt, hw...)
		}

		if len(previousTokens) > 0 {
			maxPrev := maxTokenLength/2 - 1
			if len(previousTokens) > maxPrev {
				previousTokens = previousTokens[len(previousTokens)-maxPrev:]
			}
			prompt = append(prompt, previousTokens...)
		}
	}

	prompt = append(prompt, tokenSOT)

	if m.IsMultilingual() && lang != "" {
		langTok := m.tokenizer.LanguageToken(lang)
		if langTok >= 0 {
			prompt = append(prompt, langTok)
		}
		prompt = append(prompt, taskToken)
	}

	if cfg.DisableTimestamps {
		prompt = append(prompt, tokenNoTimestamps)
	}

	return prompt
}

type fallbackResult struct {
	genResult        ct2bridge.GenerateResult
	avgLogProb       float32
	temperature      float32
	compressionRatio float32
}

func (m *Model) generateWithFallback(
	enc *ct2bridge.EncoderOutput,
	prompt []int32,
	cfg TranscribeConfig,
	suppressTokens []int32,
) (ct2bridge.GenerateResult, float32, float32, float32, error) {
	var best *fallbackResult
	var bestBelowCR *fallbackResult

	maxInitialTSIdx := 0
	if !cfg.DisableTimestamps {
		maxInitialTSIdx = int(math.Round(float64(*cfg.MaxInitialTimestamp) / timePrecision))
	}

	maxLength := maxTokenLength
	if cfg.MaxNewTokens != nil {
		maxLength = len(prompt) + *cfg.MaxNewTokens
		if maxLength > maxTokenLength {
			maxLength = maxTokenLength
		}
	}

	for _, temp := range cfg.Temperature {
		opts := ct2bridge.GenerateOptions{
			BeamSize:                 cfg.BeamSize,
			BestOf:                   cfg.BestOf,
			Patience:                 *cfg.Patience,
			LengthPenalty:            *cfg.LengthPenalty,
			RepetitionPenalty:        *cfg.RepetitionPenalty,
			NoRepeatNgramSize:        cfg.NoRepeatNgramSize,
			MaxLength:                maxLength,
			SuppressBlank:            *cfg.SuppressBlank,
			SamplingTemperature:      temp,
			SuppressTokens:           suppressTokens,
			MaxInitialTimestampIndex: maxInitialTSIdx,
		}

		genResult, err := m.bridge.Generate(enc, prompt, opts)
		if err != nil {
			return ct2bridge.GenerateResult{}, 0, 0, 0, err
		}

		var avgLogProb float32
		if seqLen := len(genResult.SequenceIDs); seqLen > 0 {
			cumLogProb := genResult.Score * float32(math.Pow(float64(seqLen), float64(*cfg.LengthPenalty)))
			avgLogProb = cumLogProb / float32(seqLen+1)
		}

		text := m.tokenizer.Decode(genResult.SequenceIDs)
		compressionRatio := getCompressionRatio(text)

		fr := fallbackResult{
			genResult:        genResult,
			avgLogProb:       avgLogProb,
			temperature:      temp,
			compressionRatio: compressionRatio,
		}

		needsFallback := false

		if *cfg.CompressionRatioThreshold > 0 && compressionRatio > *cfg.CompressionRatioThreshold {
			needsFallback = true
		} else {
			if bestBelowCR == nil || fr.avgLogProb > bestBelowCR.avgLogProb {
				bestBelowCR = &fr
			}
		}

		if avgLogProb < *cfg.LogProbThreshold {
			isNoSpeech := *cfg.NoSpeechThreshold > 0 && genResult.NoSpeechProb > *cfg.NoSpeechThreshold
			if !isNoSpeech {
				needsFallback = true
			}
		}

		if best == nil || fr.avgLogProb > best.avgLogProb {
			best = &fr
		}

		if !needsFallback {
			return genResult, avgLogProb, temp, compressionRatio, nil
		}
	}

	pick := bestBelowCR
	if pick == nil {
		pick = best
	}
	return pick.genResult, pick.avgLogProb, pick.temperature, pick.compressionRatio, nil
}

func shouldSkipSegment(genResult ct2bridge.GenerateResult, avgLogProb float32, cfg TranscribeConfig) bool {
	if *cfg.NoSpeechThreshold <= 0 {
		return false
	}
	if genResult.NoSpeechProb <= *cfg.NoSpeechThreshold {
		return false
	}
	if avgLogProb > *cfg.LogProbThreshold {
		return false
	}
	return true
}

func getCompressionRatio(text string) float32 {
	raw := []byte(text)
	if len(raw) == 0 {
		return 0
	}
	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	w.Write(raw)
	w.Close()
	if buf.Len() == 0 {
		return 0
	}
	return float32(len(raw)) / float32(buf.Len())
}

func padOrTrim(samples []float32, length int) []float32 {
	if len(samples) >= length {
		return samples[:length]
	}
	out := make([]float32, length)
	copy(out, samples)
	return out
}

func secToDuration(sec float64) time.Duration {
	return time.Duration(sec * float64(time.Second))
}
