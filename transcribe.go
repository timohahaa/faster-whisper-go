package whisper

import (
	"bytes"
	"compress/flate"
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

	cfg = applyDefaults(cfg)

	mel, totalFrames := computeMelSpectrogram(samples, m.nMels)
	// Content frames: number of frames corresponding to actual audio content.
	// STFT produces one extra frame due to zero-padding, so subtract 1.
	contentFrames := totalFrames - 1
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

	for seek < contentFrames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		segmentSize := min(whisperNFrames, contentFrames-seek)
		segmentDuration := float64(segmentSize) * timePerFrame
		timeOffset := float64(seek) * timePerFrame

		window := extractMelWindow(mel, totalFrames, m.nMels, seek, segmentSize)
		enc, err := m.bridge.Encode(window, m.nMels, whisperNFrames)
		if err != nil {
			return nil, err
		}

		if cfg.Multilingual && m.IsMultilingual() {
			detectedLang, _, dlErr := m.detectLanguageFromEncoded(enc)
			if dlErr == nil {
				lang = detectedLang
			}
		}

		previousTokens := allTokens[promptResetSince:]
		prompt := m.buildPrompt(lang, previousTokens, cfg, taskToken)

		genResult, avgLogProb, temperature, compressionRatio, genErr := m.generateWithFallback(enc, prompt, cfg, suppressTokens)
		if genErr != nil {
			enc.Free()
			return nil, genErr
		}

		if shouldSkipSegment(genResult, avgLogProb, cfg) {
			enc.Free()
			seek += segmentSize
			continue
		}

		tokens := genResult.SequenceIDs

		split := m.tokenizer.SplitSegmentsByTimestamps(tokens, timeOffset, segmentSize, segmentDuration, seek)
		seek = split.seek

		for _, seg := range split.segments {
			text := strings.TrimSpace(m.tokenizer.Decode(seg.tokens))
			if text == "" || seg.start == seg.end {
				continue
			}

			allTokens = append(allTokens, seg.tokens...)
			segIdx++

			var words []Word
			if cfg.WordTimestamps {
				words, _ = m.extractWordTimestamps(enc, seg.tokens, lang, taskToken, segmentSize, seg.start)
			}

			segments = append(segments, Segment{
				ID:               segIdx,
				Start:            secToDuration(seg.start),
				End:              secToDuration(seg.end),
				Text:             text,
				Words:            words,
				Temperature:      temperature,
				AvgLogProb:       avgLogProb,
				CompressionRatio: compressionRatio,
				NoSpeechProb:     genResult.NoSpeechProb,
			})
		}

		enc.Free()

		if !cfg.ConditionOnPreviousText || temperature > cfg.PromptResetOnTemperature {
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
	mel, totalFrames := computeMelSpectrogram(audio, m.nMels)
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

	return m.detectLanguageFromEncoded(enc)
}

func (m *Model) detectLanguageFromEncoded(enc *ct2bridge.EncoderOutput) (string, float32, error) {
	result, err := m.bridge.DetectLanguage(enc)
	if err != nil {
		return "", 0, err
	}
	return result.Language, result.Probability, nil
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
	var allResults []fallbackResult
	var belowCRResults []fallbackResult

	maxInitialTSIdx := int(math.Round(float64(cfg.MaxInitialTimestamp) / timePrecision))

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
			Patience:                 1.0,
			LengthPenalty:            1.0,
			RepetitionPenalty:        1.0,
			NoRepeatNgramSize:        0,
			MaxLength:                maxLength,
			SuppressBlank:            cfg.SuppressBlank,
			ReturnScores:             true,
			SamplingTemperature:      temp,
			SuppressTokens:          suppressTokens,
			MaxInitialTimestampIndex: maxInitialTSIdx,
		}

		genResult, err := m.bridge.Generate(enc, prompt, opts)
		if err != nil {
			return ct2bridge.GenerateResult{}, 0, 0, 0, err
		}

		var avgLogProb float32
		if len(genResult.SequenceIDs) > 0 {
			avgLogProb = genResult.Score
		}

		text := m.tokenizer.Decode(genResult.SequenceIDs)
		compressionRatio := getCompressionRatio(text)

		fr := fallbackResult{
			genResult:        genResult,
			avgLogProb:       avgLogProb,
			temperature:      temp,
			compressionRatio: compressionRatio,
		}
		allResults = append(allResults, fr)

		needsFallback := false

		if cfg.CompressionRatioThreshold > 0 && compressionRatio > cfg.CompressionRatioThreshold {
			needsFallback = true
		} else {
			belowCRResults = append(belowCRResults, fr)
		}

		if cfg.LogProbThreshold != 0 && avgLogProb < cfg.LogProbThreshold {
			needsFallback = true
		}

		if cfg.NoSpeechThreshold > 0 &&
			genResult.NoSpeechProb > cfg.NoSpeechThreshold &&
			cfg.LogProbThreshold != 0 &&
			avgLogProb < cfg.LogProbThreshold {
			needsFallback = false
		}

		if !needsFallback {
			return genResult, avgLogProb, temp, compressionRatio, nil
		}
	}

	candidates := belowCRResults
	if len(candidates) == 0 {
		candidates = allResults
	}
	best := candidates[0]
	for _, c := range candidates[1:] {
		if c.avgLogProb > best.avgLogProb {
			best = c
		}
	}

	return best.genResult, best.avgLogProb, cfg.Temperature[len(cfg.Temperature)-1], best.compressionRatio, nil
}

func shouldSkipSegment(genResult ct2bridge.GenerateResult, avgLogProb float32, cfg TranscribeConfig) bool {
	if cfg.NoSpeechThreshold <= 0 {
		return false
	}
	shouldSkip := genResult.NoSpeechProb > cfg.NoSpeechThreshold
	if cfg.LogProbThreshold != 0 && avgLogProb > cfg.LogProbThreshold {
		shouldSkip = false
	}
	return shouldSkip
}

func getCompressionRatio(text string) float32 {
	raw := []byte(text)
	if len(raw) == 0 {
		return 0
	}
	var buf bytes.Buffer
	w, err := flate.NewWriter(&buf, flate.DefaultCompression)
	if err != nil {
		return 0
	}
	w.Write(raw)
	w.Close()
	compressed := buf.Len()
	if compressed == 0 {
		return 0
	}
	return float32(len(raw)) / float32(compressed)
}

func applyDefaults(cfg TranscribeConfig) TranscribeConfig {
	if cfg.BeamSize <= 0 {
		cfg.BeamSize = 5
	}
	if cfg.BestOf <= 0 {
		cfg.BestOf = 5
	}
	if len(cfg.Temperature) == 0 {
		cfg.Temperature = []float32{0, 0.2, 0.4, 0.6, 0.8, 1.0}
	}
	if cfg.CompressionRatioThreshold == 0 {
		cfg.CompressionRatioThreshold = 2.4
	}
	if cfg.LogProbThreshold == 0 {
		cfg.LogProbThreshold = -1.0
	}
	if cfg.NoSpeechThreshold == 0 {
		cfg.NoSpeechThreshold = 0.6
	}
	if cfg.PromptResetOnTemperature == 0 {
		cfg.PromptResetOnTemperature = 0.5
	}
	if cfg.MaxInitialTimestamp == 0 {
		cfg.MaxInitialTimestamp = 1.0
	}
	if cfg.SuppressTokens == nil {
		cfg.SuppressTokens = []int32{-1}
	}
	return cfg
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
