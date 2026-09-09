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
	return m.infer(ctx, samples, cfg, m.tokenizer.transcribe)
}

// Translate runs speech recognition and translates the result into English.
// Handles audio of any length via an internal sliding window.
func (m *Model) Translate(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.infer(ctx, samples, cfg, m.tokenizer.translate)
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
		var err error
		speechChunks, err = GetSpeechTimestamps(m.vad, samples, *vadCfg)
		if err != nil {
			return nil, err
		}
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
	// faster-whisper drops the trailing feature frame when bounding the seek
	// loop: content_frames = features.shape[-1] - 1.
	contentFrames := totalFrames - 1
	if contentFrames < 0 {
		contentFrames = 0
	}

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

	suppressTokens := m.tokenizer.suppressedTokens(cfg.SuppressTokens)

	seek := 0
	var allTokens []int32
	promptResetSince := 0
	segIdx := 0

	if cfg.InitialPrompt != "" {
		initialTokens := m.tokenizer.encode(" " + strings.TrimSpace(cfg.InitialPrompt))
		allTokens = append(allTokens, initialTokens...)
	}

	var segments []Segment
	var lastSpeechTimestamp float64

	for seek < contentFrames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		segmentSize := min(whisperNFrames, contentFrames-seek)
		winRes, err := m.processWindow(processWindowParams{
			mel:                 mel,
			totalFrames:         totalFrames,
			contentFrames:       contentFrames,
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

		seek = winRes.seek
		if winRes.lang != "" {
			lang = winRes.lang
		}
		allTokens = append(allTokens, winRes.tokens...)
		segIdx = winRes.segIdx
		segments = append(segments, winRes.segments...)
		lastSpeechTimestamp = winRes.lastSpeechTimestamp

		if !*cfg.ConditionOnPreviousText || winRes.temperature > *cfg.PromptResetOnTemperature {
			promptResetSince = len(allTokens)
		}
	}

	if speechChunks != nil {
		tsMap := newSpeechTimestampsMap(speechChunks)
		for i := range segments {
			tsMap.restoreSegmentTimestamps(&segments[i])
		}
	}

	if cfg.FilterHallucinationPhrases {
		segments = filterHallucinationPhrases(segments, lang)
	}

	return &Result{
		Text:     joinSegmentsText(segments),
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

	lang, prob, err := m.detectLanguageFromMel(mel, totalFrames, 0)
	if err != nil {
		return LanguageDetection{}, err
	}
	return LanguageDetection{
		Language:    lang,
		Probability: prob,
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
	contentFrames       int
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

	prefix := ""
	if p.seek == 0 {
		prefix = p.cfg.Prefix
	}
	prompt := m.buildPrompt(promptParams{
		lang:              lang,
		previousTokens:    p.previousTokens,
		taskToken:         p.taskToken,
		hotwords:          p.cfg.Hotwords,
		prefix:            prefix,
		withoutTimestamps: p.cfg.DisableTimestamps,
	})

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

	split := m.tokenizer.splitSegmentsByTimestamps(genResult.SequenceIDs, timeOffset, p.segmentSize, segmentDuration, p.seek)

	var segments []Segment
	var segmentTokens [][]int32
	var tokens []int32
	segIdx := p.segIdx

	for _, seg := range split.segments {
		text := strings.TrimSpace(m.tokenizer.decode(seg.tokens))
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
			if wordEnd := lastWordEnd(segments); wordEnd > timeOffset {
				seek = int(math.Round(wordEnd * framesPerSecond))
			}
		}

		// Hallucination silence detection.
		if p.cfg.HallucinationSilenceThreshold != nil {
			var skipWindow bool
			segments, seek, skipWindow = m.applyHallucinationSilence(p, segments, seek, timeOffset, segmentDuration, lastSpeechTS)
			if skipWindow {
				return windowResult{
					seek:                seek,
					segIdx:              p.segIdx,
					lang:                lang,
					temperature:         temperature,
					lastSpeechTimestamp: lastSpeechTS,
				}, nil
			}
		}

		// Update lastSpeechTimestamp from the last word end.
		if end := lastWordEnd(segments); end > 0 {
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

// applyHallucinationSilence drops anomalous segments that are surrounded by
// silence (Whisper's hallucination filtering heuristic). It returns the
// possibly-truncated segments and the adjusted seek position. When skipWindow is
// true the caller should discard this window entirely and resume from seek.
func (m *Model) applyHallucinationSilence(
	p processWindowParams,
	segments []Segment,
	seek int,
	timeOffset, segmentDuration, lastSpeechTS float64,
) (result []Segment, newSeek int, skipWindow bool) {
	threshold := float64(*p.cfg.HallucinationSilenceThreshold)
	windowEndTime := float64(p.seek+whisperNFrames) * timePerFrame

	// If the first segment is anomalous and preceded by silence > threshold, skip it.
	if first := firstSegmentWithWords(segments); first >= 0 && isSegmentAnomaly(segments[first]) {
		gap := segments[first].Start.Seconds() - timeOffset
		if gap > threshold {
			return segments, p.seek + int(math.Round(gap*framesPerSecond)), true
		}
	}

	// Check each segment for a hallucination surrounded by silence.
	halLastEnd := lastSpeechTS
	contentDuration := float64(p.contentFrames) * timePerFrame
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
				segStart-timeOffset < hallucinationEdgeMarginS
			silenceAfter := halNextStart-segEnd > threshold ||
				(si+1 < len(segments) && isSegmentAnomaly(segments[si+1])) ||
				windowEndTime-segEnd < hallucinationEdgeMarginS

			if silenceBefore && silenceAfter {
				seek = int(math.Round(math.Max(timeOffset+1, segStart) * framesPerSecond))
				if contentDuration-segEnd < threshold {
					seek = p.contentFrames
				}
				segments = segments[:si]
				break
			}
		}
		halLastEnd = segments[si].End.Seconds()
	}
	return segments, seek, false
}

// halfPromptBudget is the maximum number of tokens reserved for each of the
// hotwords, previous-text and prefix segments of the prompt (Whisper's max_length/2).
const halfPromptBudget = maxTokenLength / 2

// promptParams holds the inputs for buildPrompt shared by the sequential and
// batched pipelines. Batched mode always sets prefix="" and withoutTimestamps=true.
type promptParams struct {
	lang              string
	previousTokens    []int32
	taskToken         int32
	hotwords          string
	prefix            string
	withoutTimestamps bool
}

// buildPrompt constructs the decoder prompt:
// [sot_prev, (hotwords), (previous_tokens), sot, (lang, task), (no_timestamps), (prefix)].
func (m *Model) buildPrompt(p promptParams) []int32 {
	var prompt []int32

	useHotwords := p.hotwords != "" && p.prefix == ""

	if len(p.previousTokens) > 0 || useHotwords {
		prompt = append(prompt, m.tokenizer.sotPrev)

		if useHotwords {
			hw := m.tokenizer.encode(" " + strings.TrimSpace(p.hotwords))
			if len(hw) >= halfPromptBudget {
				hw = hw[:halfPromptBudget-1]
			}
			prompt = append(prompt, hw...)
		}

		if len(p.previousTokens) > 0 {
			maxPrev := halfPromptBudget - 1
			prev := p.previousTokens
			if len(prev) > maxPrev {
				prev = prev[len(prev)-maxPrev:]
			}
			prompt = append(prompt, prev...)
		}
	}

	prompt = append(prompt, m.tokenizer.sot)

	if m.IsMultilingual() && p.lang != "" {
		langTok := m.tokenizer.languageToken(p.lang)
		if langTok >= 0 {
			prompt = append(prompt, langTok)
		}
		prompt = append(prompt, p.taskToken)
	}

	if p.withoutTimestamps {
		prompt = append(prompt, m.tokenizer.noTimestamps)
	}

	if p.prefix != "" {
		prefixTokens := m.tokenizer.encode(" " + strings.TrimSpace(p.prefix))
		if len(prefixTokens) >= halfPromptBudget {
			prefixTokens = prefixTokens[:halfPromptBudget-1]
		}
		if !p.withoutTimestamps {
			prompt = append(prompt, m.tokenizer.timestampBegin)
		}
		prompt = append(prompt, prefixTokens...)
	}

	return prompt
}
