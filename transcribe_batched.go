package whisper

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

const (
	defaultBatchSize    = 8
	defaultMelWorkers   = 4
	batchedMinSilenceMs = 160
)

// chunkSegments holds the per-chunk decoding output during batched post-processing.
type chunkSegments struct {
	segments      []Segment
	segmentTokens [][]int32
	segmentSize   int
}

// TranscribeBatched runs speech recognition by splitting audio into independent
// chunks via VAD and processing them in parallel batches through the encoder
// and decoder. This gives higher throughput than the sequential Transcribe
// method, at the cost of losing cross-window context.
func (m *Model) TranscribeBatched(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.inferBatched(ctx, samples, cfg, m.tokenizer.transcribe)
}

// inferBatched is the shared batched pipeline for transcription and translation;
// taskToken selects the decoding task (transcribe or translate).
func (m *Model) inferBatched(ctx context.Context, samples []float32, cfg TranscribeConfig, taskToken int32) (*Result, error) {
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
	if cfg.BatchSize == 0 {
		cfg.BatchSize = defaultBatchSize
	}
	if cfg.ChunkLength == 0 {
		cfg.ChunkLength = whisperChunkLen
	}
	if cfg.MelWorkers == 0 {
		cfg.MelWorkers = defaultMelWorkers
	}

	duration := time.Duration(float64(len(samples)) / whisperSampleRate * float64(time.Second))
	chunkLength := cfg.ChunkLength

	// Obtain speech regions.
	var speechChunks []SpeechChunk
	var audioChunks [][]float32
	var chunksMetadata []chunkMetadata
	clipTimestampsProvided := len(cfg.ClipTimestamps) > 0

	if clipTimestampsProvided {
		audioChunks = make([][]float32, len(cfg.ClipTimestamps))
		chunksMetadata = make([]chunkMetadata, len(cfg.ClipTimestamps))
		for i, clip := range cfg.ClipTimestamps {
			start, end := clampRange(clip.Start, clip.End, len(samples))
			audioChunks[i] = samples[start:end]
			clipDuration := float64(end-start) / whisperSampleRate
			chunksMetadata[i] = chunkMetadata{
				offset:   float64(clip.Start) / whisperSampleRate,
				duration: clipDuration,
			}
		}
		speechChunks = cfg.ClipTimestamps
	} else {
		vadCfg := cfg.VadConfig
		if vadCfg == nil {
			vadCfg = &VadConfig{}
		}
		vadCfg.MaxSpeechDurationS = float64(chunkLength)
		if vadCfg.MinSilenceDurationMs == 0 {
			vadCfg.MinSilenceDurationMs = batchedMinSilenceMs
		}
		vadCfg.applyDefaults()

		var err error
		speechChunks, err = GetSpeechTimestamps(samples, *vadCfg)
		if err != nil {
			return nil, err
		}

		if len(speechChunks) == 0 {
			durationSec := float64(len(samples)) / whisperSampleRate
			if durationSec < float64(chunkLength) {
				speechChunks = []SpeechChunk{{Start: 0, End: len(samples)}}
			}
		}

		audioChunks, chunksMetadata = collectChunksBatched(samples, speechChunks, float64(chunkLength))
	}

	durationAfterVad := time.Duration(0)
	for _, chunk := range speechChunks {
		durationAfterVad += time.Duration(float64(chunk.End-chunk.Start) / whisperSampleRate * float64(time.Second))
	}

	// Compute mel features for each chunk. Chunks are independent, so spread
	// the (pure-Go, CPU-bound) STFT + mel filterbank work across all cores.
	melChunks := make([][]float32, len(audioChunks))
	melWorkers := cfg.MelWorkers
	if melWorkers > len(audioChunks) {
		melWorkers = len(audioChunks)
	}
	if melWorkers < 1 {
		melWorkers = 1
	}
	var melWG sync.WaitGroup
	melIdx := make(chan int, len(audioChunks))
	for i := range audioChunks {
		melIdx <- i
	}
	close(melIdx)
	for w := 0; w < melWorkers; w++ {
		melWG.Add(1)
		go func() {
			defer melWG.Done()
			for i := range melIdx {
				if audio := audioChunks[i]; len(audio) > 0 {
					melChunks[i] = computeChunkMel(audio, m.nMels, m.sparseFilters)
				} else {
					melChunks[i] = make([]float32, m.nMels*whisperNFrames)
				}
			}
		}()
	}
	melWG.Wait()

	// Detect language if needed.
	lang := cfg.Language
	var langProb float32

	if lang == "" && m.IsMultilingual() {
		if len(melChunks) > 0 {
			enc, err := m.bridge.Encode(melChunks[0], m.nMels, whisperNFrames)
			if err != nil {
				return nil, err
			}
			result, err := m.bridge.DetectLanguage(enc)
			enc.Free()
			if err != nil {
				return nil, err
			}
			lang = result.Language
			langProb = result.Probability
		} else {
			lang = "en"
			langProb = 1.0
		}
	} else if lang == "" {
		lang = "en"
		langProb = 1.0
	} else {
		langProb = 1.0
	}

	suppressTokens := m.tokenizer.suppressedTokens(cfg.SuppressTokens)

	// Build the base prompt (same for all chunks in non-multilingual mode).
	var previousTokens []int32
	if cfg.InitialPrompt != "" {
		previousTokens = m.tokenizer.encode(" " + strings.TrimSpace(cfg.InitialPrompt))
	}

	// Batched mode always decodes without timestamp tokens, like the batched
	// inference pipeline (decoding without timestamps):
	// coarse segment boundaries come from the VAD chunks and word-level timing
	// from the separate alignment pass, so emitting timestamp tokens here only
	// lengthens the decoded sequence and slows generation for no benefit.
	// It also never conditions on previous-window text.
	basePrompt := m.buildPrompt(promptParams{
		lang:              lang,
		previousTokens:    previousTokens,
		taskToken:         taskToken,
		hotwords:          cfg.Hotwords,
		withoutTimestamps: true,
	})

	maxLength := maxTokenLength
	if cfg.MaxNewTokens != nil {
		maxLength = len(basePrompt) + *cfg.MaxNewTokens
		if maxLength > maxTokenLength {
			maxLength = maxTokenLength
		}
	}

	opts := ct2bridge.GenerateOptions{
		BeamSize:            cfg.BeamSize,
		BestOf:              cfg.BestOf,
		Patience:            *cfg.Patience,
		LengthPenalty:       *cfg.LengthPenalty,
		RepetitionPenalty:   *cfg.RepetitionPenalty,
		NoRepeatNgramSize:   cfg.NoRepeatNgramSize,
		MaxLength:           maxLength,
		SuppressBlank:       *cfg.SuppressBlank,
		SamplingTemperature: cfg.Temperature[0],
		SuppressTokens:      suppressTokens,
	}

	// Process batches.
	var allSegments []Segment
	segIdx := 0
	var lastSpeechTimestamp float64

	nChunks := len(melChunks)
	for batchStart := 0; batchStart < nChunks; batchStart += cfg.BatchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		batchEnd := batchStart + cfg.BatchSize
		if batchEnd > nChunks {
			batchEnd = nChunks
		}
		batchMels := melChunks[batchStart:batchEnd]
		batchMeta := chunksMetadata[batchStart:batchEnd]
		batchSize := len(batchMels)

		flatMel := stackMelBatch(batchMels)
		enc, err := m.bridge.EncodeBatch(flatMel, batchSize, m.nMels, whisperNFrames)
		if err != nil {
			return nil, err
		}

		// Build per-item prompts.
		prompts := make([][]int32, batchSize)
		for i := range batchSize {
			p := make([]int32, len(basePrompt))
			copy(p, basePrompt)
			prompts[i] = p
		}

		// In multilingual mode, detect language per chunk and patch the prompt.
		if cfg.Multilingual && m.IsMultilingual() {
			langTokenIdx := -1
			langTok := m.tokenizer.languageToken(lang)
			for j, tok := range basePrompt {
				if tok == langTok {
					langTokenIdx = j
					break
				}
			}
			if langTokenIdx >= 0 {
				for i := range batchSize {
					slicedEnc, sliceErr := enc.Slice(i)
					if sliceErr != nil {
						continue
					}
					dlResult, dlErr := m.bridge.DetectLanguage(slicedEnc)
					slicedEnc.Free()
					if dlErr != nil {
						continue
					}
					detectedLangTok := m.tokenizer.languageToken(dlResult.Language)
					if detectedLangTok >= 0 {
						prompts[i][langTokenIdx] = detectedLangTok
					}
				}
			}
		}

		batchResult, err := m.bridge.GenerateBatch(enc, prompts, opts)
		if err != nil {
			enc.Free()
			return nil, err
		}

		// Post-process each item: split by timestamps, build segments.
		chunkResults := make([]chunkSegments, batchSize)

		for i, genResult := range batchResult.Items {
			meta := batchMeta[i]
			dur := meta.duration
			segmentSize := int(math.Ceil(dur) * framesPerSecond)

			seqLen := len(genResult.SequenceIDs)
			if seqLen == 0 {
				continue
			}

			avgLogProb := recoverAvgLogProb(genResult.Score, seqLen, *cfg.LengthPenalty)

			split := m.tokenizer.splitSegmentsByTimestamps(
				genResult.SequenceIDs,
				meta.offset,
				segmentSize,
				dur,
				0,
			)

			var segments []Segment
			var segTokens [][]int32

			for _, seg := range split.segments {
				text := strings.TrimSpace(m.tokenizer.decode(seg.tokens))
				if text == "" || seg.start == seg.end {
					continue
				}

				segIdx++
				s := Segment{
					ID:               segIdx,
					Start:            secToDuration(seg.start),
					End:              secToDuration(seg.end),
					Text:             text,
					Temperature:      cfg.Temperature[0],
					AvgLogProb:       avgLogProb,
					CompressionRatio: compressionRatio(text),
					NoSpeechProb:     genResult.NoSpeechProb,
				}
				segments = append(segments, s)
				segTokens = append(segTokens, seg.tokens)
			}

			chunkResults[i] = chunkSegments{
				segments:      segments,
				segmentTokens: segTokens,
				segmentSize:   segmentSize,
			}
		}

		// Word timestamps: process per-chunk using sliced encoder output.
		if cfg.WordTimestamps && !cfg.DisableTimestamps {
			for i := range batchSize {
				cr := &chunkResults[i]
				if len(cr.segments) == 0 {
					continue
				}

				slicedEnc, sliceErr := enc.Slice(i)
				if sliceErr != nil {
					continue
				}

				seekVal := int(batchMeta[i].offset * framesPerSecond)
				cr.segments, lastSpeechTimestamp = m.addWordTimestamps(
					slicedEnc, cr.segments, cr.segmentTokens,
					lang, taskToken, cr.segmentSize, seekVal,
					cfg.PrependPunctuations, cfg.AppendPunctuations,
					lastSpeechTimestamp,
				)

				slicedEnc.Free()
			}
		}

		enc.Free()

		for _, cr := range chunkResults {
			allSegments = append(allSegments, cr.segments...)
		}
	}

	// Restore timestamps if VAD was used.
	if !clipTimestampsProvided && len(speechChunks) > 0 {
		tsMap := newSpeechTimestampsMap(speechChunks)
		for i := range allSegments {
			tsMap.restoreSegmentTimestamps(&allSegments[i])
		}
	}

	if cfg.FilterHallucinationPhrases {
		allSegments = filterHallucinationPhrases(allSegments, lang)
	}

	return &Result{
		Text:     joinSegmentsText(allSegments),
		Segments: allSegments,
		Info: TranscriptionInfo{
			Language:            lang,
			LanguageProbability: langProb,
			Duration:            duration,
			DurationAfterVad:    durationAfterVad,
		},
	}, nil
}

// TranslateBatched runs batched speech recognition and translates the result
// into English.
func (m *Model) TranslateBatched(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.inferBatched(ctx, samples, cfg, m.tokenizer.translate)
}
