package whisper

import (
	"bytes"
	"compress/zlib"
	"math"
	"strings"
	"sync"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

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

		avgLogProb := recoverAvgLogProb(genResult.Score, len(genResult.SequenceIDs), *cfg.LengthPenalty)

		text := strings.TrimSpace(m.tokenizer.decode(genResult.SequenceIDs))
		compRatio := compressionRatio(text)

		fb := fallbackResult{
			genResult:        genResult,
			avgLogProb:       avgLogProb,
			temperature:      temp,
			compressionRatio: compRatio,
		}

		needsFallback := false

		if *cfg.CompressionRatioThreshold > 0 && compRatio > *cfg.CompressionRatioThreshold {
			needsFallback = true
		} else {
			if bestBelowCR == nil || fb.avgLogProb > bestBelowCR.avgLogProb {
				bestBelowCR = &fb
			}
		}

		if avgLogProb < *cfg.LogProbThreshold {
			needsFallback = true
		}

		// silence overrides any fallback decision (including one triggered by a high compression ratio)
		if *cfg.NoSpeechThreshold > 0 && genResult.NoSpeechProb > *cfg.NoSpeechThreshold &&
			avgLogProb < *cfg.LogProbThreshold {
			needsFallback = false
		}

		if best == nil || fb.avgLogProb > best.avgLogProb {
			best = &fb
		}

		if !needsFallback {
			return genResult, avgLogProb, temp, compRatio, nil
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

// recoverAvgLogProb reconstructs the average log probability from the score
// returned by the decoder (which has the length penalty already applied).
func recoverAvgLogProb(score float32, seqLen int, lengthPenalty float32) float32 {
	if seqLen == 0 {
		return 0
	}
	cumLogProb := score * float32(math.Pow(float64(seqLen), float64(lengthPenalty)))
	return cumLogProb / float32(seqLen+1)
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

func compressionRatio(text string) float32 {
	raw := []byte(text)
	if len(raw) == 0 {
		return 0
	}
	zlibSt := zlibPool.Get().(*zlibState)
	zlibSt.buf.Reset()
	zlibSt.w.Reset(zlibSt.buf)
	zlibSt.w.Write(raw)
	zlibSt.w.Close()
	compressed := zlibSt.buf.Len()
	zlibPool.Put(zlibSt)
	if compressed == 0 {
		return 0
	}
	return float32(len(raw)) / float32(compressed)
}
