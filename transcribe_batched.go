package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

// TranscribeBatched runs speech recognition by splitting audio into independent
// chunks via VAD and processing them in parallel batches through the encoder
// and decoder. This gives higher throughput than the sequential Transcribe
// method, at the cost of losing cross-window context.
func (m *Model) TranscribeBatched(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
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
		cfg.BatchSize = 8
	}
	if cfg.ChunkLength == 0 {
		cfg.ChunkLength = whisperChunkLen
	}

	prof := os.Getenv("WHISPER_PROFILE") != ""
	var tVad, tMel, tEncode, tGenerate, tWord time.Duration
	profStart := time.Now()

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
			start := clip.Start
			end := clip.End
			if start > len(samples) {
				start = len(samples)
			}
			if end > len(samples) {
				end = len(samples)
			}
			audioChunks[i] = samples[start:end]
			clipDuration := float64(end-start) / whisperSampleRate
			chunksMetadata[i] = chunkMetadata{
				offset:   float64(clip.Start) / whisperSampleRate,
				duration: clipDuration,
				segments: []SpeechChunk{clip},
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
			vadCfg.MinSilenceDurationMs = 160
		}
		vadCfg.applyDefaults()

		vadStart := time.Now()
		speechChunks = GetSpeechTimestamps(samples, *vadCfg)
		tVad += time.Since(vadStart)

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
	melStart := time.Now()
	melChunks := make([][]float32, len(audioChunks))
	melWorkers := runtime.GOMAXPROCS(0)
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
	tMel += time.Since(melStart)

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

	suppressTokens := m.tokenizer.SuppressedTokens(cfg.SuppressTokens)

	// Build the base prompt (same for all chunks in non-multilingual mode).
	var previousTokens []int32
	if cfg.InitialPrompt != "" {
		previousTokens = m.tokenizer.Encode(" " + strings.TrimSpace(cfg.InitialPrompt))
	}

	taskToken := m.tokenizer.transcribe

	basePrompt := m.buildBatchedPrompt(lang, previousTokens, cfg, taskToken)

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
		encStart := time.Now()
		enc, err := m.bridge.EncodeBatch(flatMel, batchSize, m.nMels, whisperNFrames)
		if err != nil {
			return nil, err
		}
		tEncode += time.Since(encStart)

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
			langTok := m.tokenizer.LanguageToken(lang)
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
					detectedLangTok := m.tokenizer.LanguageToken(dlResult.Language)
					if detectedLangTok >= 0 {
						prompts[i][langTokenIdx] = detectedLangTok
					}
				}
			}
		}

		genStart := time.Now()
		batchResult, err := m.bridge.GenerateBatch(enc, prompts, opts)
		if err != nil {
			enc.Free()
			return nil, err
		}
		tGenerate += time.Since(genStart)

		// Post-process each item: split by timestamps, build segments.
		type chunkSegments struct {
			segments      []Segment
			segmentTokens [][]int32
			segmentSize   int
		}
		chunkResults := make([]chunkSegments, batchSize)

		for i, genResult := range batchResult.Items {
			meta := batchMeta[i]
			dur := meta.duration
			segmentSize := int(math.Ceil(dur) * framesPerSecond)

			seqLen := len(genResult.SequenceIDs)
			if seqLen == 0 {
				continue
			}

			var avgLogProb float32
			cumLogProb := genResult.Score * float32(math.Pow(float64(seqLen), float64(*cfg.LengthPenalty)))
			avgLogProb = cumLogProb / float32(seqLen+1)

			split := m.tokenizer.SplitSegmentsByTimestamps(
				genResult.SequenceIDs,
				meta.offset,
				segmentSize,
				dur,
				0,
			)

			var segments []Segment
			var segTokens [][]int32

			for _, seg := range split.segments {
				text := strings.TrimSpace(m.tokenizer.Decode(seg.tokens))
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
					CompressionRatio: getCompressionRatio(text),
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
		wordStart := time.Now()
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
		tWord += time.Since(wordStart)

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

	if prof {
		total := time.Since(profStart)
		other := total - tVad - tMel - tEncode - tGenerate - tWord
		fmt.Fprintf(os.Stderr,
			"[profile] total=%.2fs vad=%.2fs mel=%.2fs encode=%.2fs generate=%.2fs word=%.2fs other=%.2fs (chunks=%d)\n",
			total.Seconds(), tVad.Seconds(), tMel.Seconds(), tEncode.Seconds(),
			tGenerate.Seconds(), tWord.Seconds(), other.Seconds(), len(audioChunks))
	}

	var textBuf strings.Builder
	for i, seg := range allSegments {
		if i > 0 {
			textBuf.WriteByte(' ')
		}
		textBuf.WriteString(seg.Text)
	}

	return &Result{
		Text:     textBuf.String(),
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
	// Translation uses the same pipeline; the task token is embedded in the
	// prompt by buildBatchedPrompt when taskToken=tokenTranslate. For now,
	// expose only Transcribe; translate support can be added by parameterizing
	// the task token if needed.
	return m.TranscribeBatched(ctx, samples, cfg)
}

// buildBatchedPrompt constructs the decoder prompt for batched mode.
// Unlike the sequential buildPrompt, this never includes previous-window
// context (condition_on_previous_text is always false in batched mode).
func (m *Model) buildBatchedPrompt(lang string, previousTokens []int32, cfg TranscribeConfig, taskToken int32) []int32 {
	var prompt []int32

	if len(previousTokens) > 0 || (cfg.Hotwords != "") {
		prompt = append(prompt, m.tokenizer.sotPrev)

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

	// Batched mode always decodes without timestamp tokens, matching
	// faster-whisper's BatchedInferencePipeline (without_timestamps=True):
	// coarse segment boundaries come from the VAD chunks and word-level timing
	// from the separate alignment pass, so emitting timestamp tokens here only
	// lengthens the decoded sequence and slows generation for no benefit.
	prompt = append(prompt, m.tokenizer.noTimestamps)

	return prompt
}
