package whisper

import (
	"math"

	"github.com/timohahaa/faster-whisper-go/pyannotevad"
)

// pyannoteEngine is the vadEngine backed by the pyannote segmentation model.
// It reproduces the canonical pyannote VAD path: a Hamming-aggregated
// speech-activity curve, hysteresis binarization (onset/offset) with min-cut
// splitting of over-long regions, and speech-pad expansion.
type pyannoteEngine struct {
	vad *pyannotevad.VAD
}

func (e *pyannoteEngine) speechChunks(samples []float32, cfg VadConfig) ([]SpeechChunk, error) {
	cfg.applyDefaults()

	scores, err := e.vad.Scores(samples)
	if err != nil {
		return nil, err
	}

	maxDuration := cfg.MaxSpeechDurationS // +Inf when unset; batched sets it to chunk length
	speeches := binarizeVadScores(scores, pyannotevad.FrameStep, cfg.Onset, cfg.Offset, maxDuration, len(samples))

	speechPadSamples := int(float64(whisperSampleRate) * float64(cfg.SpeechPadMs) / 1000)
	applySpeechPadding(speeches, speechPadSamples, len(samples))
	return speeches, nil
}

func (e *pyannoteEngine) Close() {
	if e.vad != nil {
		e.vad.Close()
	}
}

// binarizeVadScores converts an aggregated per-frame speech-activity curve into
// speech regions (in sample indices) using hysteresis thresholding.
//
// It is a direct port of pyannote's BinarizeVadScores active-region extraction
// for the single already-max-reduced activity class. Active regions longer than
// maxDurationS are split at the lowest-scoring frame in their second half
// (min-cut). Padding (pad_onset/pad_offset) and min-duration filtering are left
// to the caller (defaults 0).
func binarizeVadScores(scores []float32, frameStep float64, onset, offset float32, maxDurationS float64, nSamples int) []SpeechChunk {
	n := len(scores)
	if n == 0 {
		return nil
	}

	// timestamps[i] is the middle time of frame i (seconds).
	timestamp := func(i int) float64 { return (float64(i) + 0.5) * frameStep }

	var regions []SpeechChunk
	emit := func(startSec, endSec float64) {
		start := int(math.Round(startSec * whisperSampleRate))
		end := int(math.Round(endSec * whisperSampleRate))
		start = max(0, min(start, nSamples))
		end = max(0, min(end, nSamples))
		if end > start {
			regions = append(regions, SpeechChunk{Start: start, End: end})
		}
	}

	start := timestamp(0)
	isActive := scores[0] > onset
	curScores := []float32{scores[0]}
	curTimes := []float64{timestamp(0)}
	t := timestamp(0)

	for i := 1; i < n; i++ {
		t = timestamp(i)
		y := scores[i]

		if isActive {
			curDuration := t - start
			if curDuration > maxDurationS {
				// Split at the lowest-scoring frame in the second half.
				searchAfter := len(curScores) / 2
				minIdx := searchAfter
				minVal := curScores[searchAfter]
				for k := searchAfter + 1; k < len(curScores); k++ {
					if curScores[k] < minVal {
						minVal = curScores[k]
						minIdx = k
					}
				}
				minT := curTimes[minIdx]
				emit(start, minT)
				start = curTimes[minIdx]
				curScores = append([]float32(nil), curScores[minIdx+1:]...)
				curTimes = append([]float64(nil), curTimes[minIdx+1:]...)
			} else if y < offset {
				// Switch from active to inactive.
				emit(start, t)
				start = t
				isActive = false
				curScores = curScores[:0]
				curTimes = curTimes[:0]
			}
			curScores = append(curScores, y)
			curTimes = append(curTimes, t)
		} else {
			// Switch from inactive to active.
			if y > onset {
				start = t
				isActive = true
			}
		}
	}

	if isActive {
		emit(start, t)
	}

	return regions
}

// applySpeechPadding expands speech regions by speechPadSamples on each side,
// splitting the padding across short gaps between neighbours (identical to the
// Silero get_speech_timestamps padding step). It mutates speeches in place.
func applySpeechPadding(speeches []SpeechChunk, speechPadSamples, audioLengthSamples int) {
	for i := range speeches {
		if i == 0 {
			speeches[i].Start = max(0, speeches[i].Start-speechPadSamples)
		}
		if i != len(speeches)-1 {
			silenceDuration := speeches[i+1].Start - speeches[i].End
			if silenceDuration < 2*speechPadSamples {
				speeches[i].End += silenceDuration / 2
				speeches[i+1].Start = max(0, speeches[i+1].Start-silenceDuration/2)
			} else {
				speeches[i].End = min(audioLengthSamples, speeches[i].End+speechPadSamples)
				speeches[i+1].Start = max(0, speeches[i+1].Start-speechPadSamples)
			}
		} else {
			speeches[i].End = min(audioLengthSamples, speeches[i].End+speechPadSamples)
		}
	}
}
