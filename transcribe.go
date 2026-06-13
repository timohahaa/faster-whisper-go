package whisper

import (
	"context"
	"errors"
	"math"
	"strings"
	"time"
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




