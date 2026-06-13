package whisper

import (
	"bytes"
	"compress/zlib"
	"context"
	"errors"
	"math"
	"strings"
	"sync"
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

	duration := time.Duration(float64(len(samples)) / whisperSampleRate * float64(time.Second))

	var speechChunks []SpeechChunk
	var durationAfterVad time.Duration
	if cfg.VadFilter {
		vadCfg := cfg.VadConfig
		if vadCfg == nil {
			vadCfg = &VadConfig{}
		}
		speechChunks = GetSpeechTimestamps(samples, *vadCfg)
		samples = collectChunks(samples, speechChunks)
		if len(samples) == 0 {
			return &Result{Info: TranscriptionInfo{
				Duration:         duration,
				DurationAfterVad: 0,
			}}, nil
		}
		durationAfterVad = time.Duration(float64(len(samples)) / whisperSampleRate * float64(time.Second))
	} else {
		durationAfterVad = duration
	}

	mel, totalFrames := computeMelSpectrogram(samples, m.nMels, m.sparseFilters)

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
	var lastSpeechTimestamp float64

	for seek < totalFrames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		segmentSize := min(whisperNFrames, totalFrames-seek)
		wr, err := m.processWindow(processWindowParams{
			mel:                 mel,
			totalFrames:         totalFrames,
			seek:                seek,
			segmentSize:         segmentSize,
			lang:                lang,
			previousTokens:      allTokens[promptResetSince:],
			cfg:                 cfg,
			taskToken:           taskToken,
			suppressTokens:      suppressTokens,
			segIdx:              segIdx,
			lastSpeechTimestamp: lastSpeechTimestamp,
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
		lastSpeechTimestamp = wr.lastSpeechTimestamp

		if !*cfg.ConditionOnPreviousText || wr.temperature > *cfg.PromptResetOnTemperature {
			promptResetSince = len(allTokens)
		}
	}

	if speechChunks != nil {
		tsMap := newSpeechTimestampsMap(speechChunks)
		for i := range segments {
			tsMap.restoreSegmentTimestamps(&segments[i])
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
			DurationAfterVad:    durationAfterVad,
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
	mel, totalFrames := computeMelSpectrogram(audio, m.nMels, m.sparseFilters)
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
	mel                 []float32
	totalFrames         int
	seek                int
	segmentSize         int
	lang                string
	previousTokens      []int32
	cfg                 TranscribeConfig
	taskToken           int32
	suppressTokens      []int32
	segIdx              int
	lastSpeechTimestamp float64
}

type windowResult struct {
	segments            []Segment
	tokens              []int32
	seek                int
	segIdx              int
	lang                string
	temperature         float32
	lastSpeechTimestamp float64
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

	prompt := m.buildPrompt(lang, p.previousTokens, p.cfg, p.taskToken, p.seek)

	genResult, avgLogProb, temperature, compressionRatio, genErr := m.generateWithFallback(enc, prompt, p.cfg, p.suppressTokens)
	if genErr != nil {
		return windowResult{}, genErr
	}

	if shouldSkipSegment(genResult, avgLogProb, p.cfg) {
		return windowResult{
			seek:                p.seek + p.segmentSize,
			segIdx:              p.segIdx,
			lang:                lang,
			temperature:         temperature,
			lastSpeechTimestamp: p.lastSpeechTimestamp,
		}, nil
	}

	split := m.tokenizer.SplitSegmentsByTimestamps(genResult.SequenceIDs, timeOffset, p.segmentSize, segmentDuration, p.seek)

	var segments []Segment
	var segmentTokens [][]int32
	var tokens []int32
	segIdx := p.segIdx

	for _, seg := range split.segments {
		text := strings.TrimSpace(m.tokenizer.Decode(seg.tokens))
		if text == "" || seg.start == seg.end {
			continue
		}

		tokens = append(tokens, seg.tokens...)
		segIdx++

		s := Segment{
			ID:               segIdx,
			Start:            secToDuration(seg.start),
			End:              secToDuration(seg.end),
			Text:             text,
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
		segmentTokens = append(segmentTokens, seg.tokens)
	}

	lastSpeechTS := p.lastSpeechTimestamp
	seek := split.seek

	if p.cfg.WordTimestamps && !p.cfg.DisableTimestamps && len(segments) > 0 {
		segments, lastSpeechTS = m.addWordTimestamps(
			enc, segments, segmentTokens,
			lang, p.taskToken, p.segmentSize, p.seek,
			p.cfg.PrependPunctuations, p.cfg.AppendPunctuations,
			lastSpeechTS,
		)

		// Adjust seek position by the last word end time when not a single-timestamp ending.
		if !split.singleTimestampEnding {
			if lastWordEnd := getLastWordEnd(segments); lastWordEnd > timeOffset {
				seek = int(math.Round(lastWordEnd * framesPerSecond))
			}
		}

		// Hallucination silence detection.
		if p.cfg.HallucinationSilenceThreshold != nil {
			threshold := float64(*p.cfg.HallucinationSilenceThreshold)
			windowEndTime := float64(p.seek+whisperNFrames) * timePerFrame

			// If first segment is anomalous and preceded by silence > threshold, skip it.
			if first := firstSegmentWithWords(segments); first >= 0 && isSegmentAnomaly(segments[first]) {
				gap := segments[first].Start.Seconds() - timeOffset
				if gap > threshold {
					seek = p.seek + int(math.Round(gap*framesPerSecond))
					return windowResult{
						seek:                seek,
						segIdx:              p.segIdx,
						lang:                lang,
						temperature:         temperature,
						lastSpeechTimestamp: lastSpeechTS,
					}, nil
				}
			}

			// Check each segment for hallucination surrounded by silence.
			halLastEnd := lastSpeechTS
			contentDuration := float64(p.totalFrames) * timePerFrame
			for si := range segments {
				if len(segments[si].Words) == 0 {
					continue
				}
				if isSegmentAnomaly(segments[si]) {
					segStart := segments[si].Start.Seconds()
					segEnd := segments[si].End.Seconds()

					halNextStart := timeOffset + segmentDuration
					if next := firstSegmentWithWords(segments[si+1:]); next >= 0 {
						halNextStart = segments[si+1+next].Words[0].Start.Seconds()
					}

					silenceBefore := segStart-halLastEnd > threshold ||
						segStart < threshold ||
						segStart-timeOffset < 2.0
					silenceAfter := halNextStart-segEnd > threshold ||
						(si+1 < len(segments) && isSegmentAnomaly(segments[si+1])) ||
						windowEndTime-segEnd < 2.0

					if silenceBefore && silenceAfter {
						seek = int(math.Round(math.Max(timeOffset+1, segStart) * framesPerSecond))
						if contentDuration-segEnd < threshold {
							seek = p.totalFrames
						}
						segments = segments[:si]
						break
					}
				}
				halLastEnd = segments[si].End.Seconds()
			}
		}

		// Update lastSpeechTimestamp from the last word end.
		if end := getLastWordEnd(segments); end > 0 {
			lastSpeechTS = end
		}
	}

	return windowResult{
		segments:            segments,
		tokens:              tokens,
		seek:                seek,
		segIdx:              segIdx,
		lang:                lang,
		temperature:         temperature,
		lastSpeechTimestamp: lastSpeechTS,
	}, nil
}

func (m *Model) buildPrompt(lang string, previousTokens []int32, cfg TranscribeConfig, taskToken int32, seek int) []int32 {
	var prompt []int32

	prefix := ""
	if seek == 0 {
		prefix = cfg.Prefix
	}

	if len(previousTokens) > 0 || (cfg.Hotwords != "" && prefix == "") {
		prompt = append(prompt, tokenSOTprev)

		if cfg.Hotwords != "" && prefix == "" {
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

	if prefix != "" {
		prefixTokens := m.tokenizer.Encode(" " + strings.TrimSpace(prefix))
		maxPrefix := maxTokenLength / 2
		if len(prefixTokens) >= maxPrefix {
			prefixTokens = prefixTokens[:maxPrefix-1]
		}
		if !cfg.DisableTimestamps {
			prompt = append(prompt, tokenTimestampBeg)
		}
		prompt = append(prompt, prefixTokens...)
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
			LengthPenalty:            *cfg.LengthPenalty,
			RepetitionPenalty:        *cfg.RepetitionPenalty,
			NoRepeatNgramSize:        cfg.NoRepeatNgramSize,
			MaxLength:                maxLength,
			SuppressBlank:            *cfg.SuppressBlank,
			SamplingTemperature:      temp,
			SuppressTokens:           suppressTokens,
			MaxInitialTimestampIndex: maxInitialTSIdx,
		}
		if temp > 0 {
			opts.BeamSize = 1
			opts.BestOf = cfg.BestOf
		} else {
			opts.BeamSize = cfg.BeamSize
			opts.Patience = *cfg.Patience
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
	// Use the last temperature from the chain (not pick's temperature) so that
	// prompt_reset_on_temperature triggers correctly when all attempts failed.
	lastTemp := cfg.Temperature[len(cfg.Temperature)-1]
	return pick.genResult, pick.avgLogProb, lastTemp, pick.compressionRatio, nil
}

const framesPerSecond = float64(whisperSampleRate) / float64(whisperHopLength)

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

// getLastWordEnd returns the end time (in seconds) of the last word across all segments.
func getLastWordEnd(segments []Segment) float64 {
	for i := len(segments) - 1; i >= 0; i-- {
		if len(segments[i].Words) > 0 {
			return segments[i].Words[len(segments[i].Words)-1].End.Seconds()
		}
	}
	if len(segments) > 0 {
		return segments[len(segments)-1].End.Seconds()
	}
	return 0
}

// firstSegmentWithWords returns the index of the first segment that has word timestamps,
// or -1 if none.
func firstSegmentWithWords(segments []Segment) int {
	for i, s := range segments {
		if len(s.Words) > 0 {
			return i
		}
	}
	return -1
}

// wordAnomalyScore computes an anomaly score for a single word based on
// its probability and duration (matching Python's word_anomaly_score).
func wordAnomalyScore(w Word) float64 {
	prob := float64(w.Probability)
	dur := w.End.Seconds() - w.Start.Seconds()
	score := 0.0
	if prob < 0.15 {
		score += 1.0
	}
	if dur < 0.133 {
		score += (0.133 - dur) * 15
	}
	if dur > 2.0 {
		score += dur - 2.0
	}
	return score
}

const anomalyPunctuation = `"'"¿([{-"'.。,，!！?？:：")]}、`

// isSegmentAnomaly checks if a segment is likely a hallucination based on
// anomaly scores of its words (matching Python's is_segment_anomaly).
// Punctuation-only words are filtered out before scoring.
func isSegmentAnomaly(seg Segment) bool {
	if len(seg.Words) == 0 {
		return false
	}
	var words []Word
	for _, w := range seg.Words {
		if !isAnomalyPunctuation(w.Word) {
			words = append(words, w)
		}
	}
	if len(words) == 0 {
		return false
	}
	if len(words) > 8 {
		words = words[:8]
	}
	var score float64
	for _, w := range words {
		score += wordAnomalyScore(w)
	}
	n := float64(len(words))
	return score >= 3 || score+0.01 >= n
}

func isAnomalyPunctuation(word string) bool {
	word = strings.TrimSpace(word)
	if word == "" {
		return true
	}
	for _, r := range anomalyPunctuation {
		if word == string(r) {
			return true
		}
	}
	return false
}

var zlibPool = sync.Pool{
	New: func() any {
		var buf bytes.Buffer
		w := zlib.NewWriter(&buf)
		return &zlibState{w: w, buf: &buf}
	},
}

type zlibState struct {
	w   *zlib.Writer
	buf *bytes.Buffer
}

func getCompressionRatio(text string) float32 {
	raw := []byte(text)
	if len(raw) == 0 {
		return 0
	}
	st := zlibPool.Get().(*zlibState)
	st.buf.Reset()
	st.w.Reset(st.buf)
	st.w.Write(raw)
	st.w.Close()
	compressed := st.buf.Len()
	zlibPool.Put(st)
	if compressed == 0 {
		return 0
	}
	return float32(len(raw)) / float32(compressed)
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
