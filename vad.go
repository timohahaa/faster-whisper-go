package whisper

import (
	"math"
	"sort"
	"time"

	"github.com/zserge/govad"
)

// VadConfig controls Voice Activity Detection parameters.
// Zero-value fields are filled with sensible defaults
type VadConfig struct {
	// Threshold is the speech probability threshold. Probabilities above this
	// value are considered speech. Default: 0.5.
	Threshold float32
	// NegThreshold is the silence threshold for end-of-speech detection.
	// 0 means auto-compute as max(Threshold-0.15, 0.01).
	NegThreshold float32
	// MinSpeechDurationMs: speech chunks shorter than this are discarded. Default: 0.
	MinSpeechDurationMs int
	// MaxSpeechDurationS: chunks longer than this are split at the best silence point.
	// Default: +Inf (no limit).
	MaxSpeechDurationS float64
	// MinSilenceDurationMs: minimum silence duration to split speech chunks. Default: 2000.
	MinSilenceDurationMs int
	// SpeechPadMs: padding added to each side of a speech chunk. Default: 400.
	SpeechPadMs int
	// MinSilenceAtMaxSpeech: minimum silence (ms) used to find split points when
	// MaxSpeechDurationS is reached. Default: 98.
	MinSilenceAtMaxSpeech int
	// UseMaxPossSilAtMaxSpeech: when true, split at the longest silence found
	// (not just the last one). Default: true.
	UseMaxPossSilAtMaxSpeech *bool
}

func (c *VadConfig) applyDefaults() {
	if c.Threshold == 0 {
		c.Threshold = 0.5
	}
	if c.NegThreshold == 0 {
		c.NegThreshold = float32(math.Max(float64(c.Threshold)-0.15, 0.01))
	}
	if c.MaxSpeechDurationS == 0 {
		c.MaxSpeechDurationS = math.Inf(1)
	}
	if c.MinSilenceDurationMs == 0 {
		c.MinSilenceDurationMs = 2000
	}
	if c.SpeechPadMs == 0 {
		c.SpeechPadMs = 400
	}
	if c.MinSilenceAtMaxSpeech == 0 {
		c.MinSilenceAtMaxSpeech = 98
	}
	if c.UseMaxPossSilAtMaxSpeech == nil {
		c.UseMaxPossSilAtMaxSpeech = ptrBool(true)
	}
}

const vadWindowSize = 512 // samples per VAD frame (govad.SamplesPerFrame)

// SpeechChunk represents a contiguous region of speech in sample indices.
type SpeechChunk struct {
	Start int // first sample (inclusive)
	End   int // last sample (exclusive)
}

// GetSpeechTimestamps detects speech regions in 16kHz mono float32 audio.
// Returns a slice of SpeechChunk with start/end in sample indices.
// This is a direct port of Python faster-whisper's get_speech_timestamps.
func GetSpeechTimestamps(samples []float32, cfg VadConfig) []SpeechChunk {
	cfg.applyDefaults()

	samplingRate := float64(whisperSampleRate)
	minSpeechSamples := samplingRate * float64(cfg.MinSpeechDurationMs) / 1000
	speechPadSamples := int(samplingRate * float64(cfg.SpeechPadMs) / 1000)
	maxSpeechSamples := samplingRate*cfg.MaxSpeechDurationS -
		float64(vadWindowSize) - 2*float64(speechPadSamples)
	minSilenceSamples := samplingRate * float64(cfg.MinSilenceDurationMs) / 1000
	minSilenceSamplesAtMaxSpeech := samplingRate * float64(cfg.MinSilenceAtMaxSpeech) / 1000

	audioLengthSamples := len(samples)

	probs := getSpeechProbs(samples)

	triggered := false
	var speeches []SpeechChunk
	var currentSpeech SpeechChunk
	currentSpeechActive := false

	type possibleEnd struct {
		pos int
		dur int
	}
	var possibleEnds []possibleEnd

	negThreshold := cfg.NegThreshold
	tempEnd := 0
	prevEnd := 0
	nextStart := 0

	for i, speechProb := range probs {
		curSample := vadWindowSize * i

		if speechProb >= cfg.Threshold && tempEnd != 0 {
			silDur := curSample - tempEnd
			if silDur > int(minSilenceSamplesAtMaxSpeech) {
				possibleEnds = append(possibleEnds, possibleEnd{tempEnd, silDur})
			}
			tempEnd = 0
			if nextStart < prevEnd {
				nextStart = curSample
			}
		}

		if speechProb >= cfg.Threshold && !triggered {
			triggered = true
			currentSpeech = SpeechChunk{Start: curSample}
			currentSpeechActive = true
			continue
		}

		if triggered && currentSpeechActive && curSample-currentSpeech.Start > int(maxSpeechSamples) {
			if *cfg.UseMaxPossSilAtMaxSpeech && len(possibleEnds) > 0 {
				best := possibleEnds[0]
				for _, pe := range possibleEnds[1:] {
					if pe.dur > best.dur {
						best = pe
					}
				}
				prevEnd = best.pos
				currentSpeech.End = prevEnd
				speeches = append(speeches, currentSpeech)
				currentSpeechActive = false
				nextStart = prevEnd + best.dur

				if nextStart < prevEnd+curSample {
					currentSpeech = SpeechChunk{Start: nextStart}
					currentSpeechActive = true
				} else {
					triggered = false
				}
				prevEnd = 0
				nextStart = 0
				tempEnd = 0
				possibleEnds = nil
			} else {
				if prevEnd != 0 {
					currentSpeech.End = prevEnd
					speeches = append(speeches, currentSpeech)
					currentSpeechActive = false
					if nextStart < prevEnd {
						triggered = false
					} else {
						currentSpeech = SpeechChunk{Start: nextStart}
						currentSpeechActive = true
					}
					prevEnd = 0
					nextStart = 0
					tempEnd = 0
					possibleEnds = nil
				} else {
					currentSpeech.End = curSample
					speeches = append(speeches, currentSpeech)
					currentSpeechActive = false
					prevEnd = 0
					nextStart = 0
					tempEnd = 0
					triggered = false
					possibleEnds = nil
					continue
				}
			}
		}

		if speechProb < negThreshold && triggered {
			if tempEnd == 0 {
				tempEnd = curSample
			}
			silDurNow := curSample - tempEnd

			if !*cfg.UseMaxPossSilAtMaxSpeech && silDurNow > int(minSilenceSamplesAtMaxSpeech) {
				prevEnd = tempEnd
			}

			if float64(silDurNow) < minSilenceSamples {
				continue
			}

			currentSpeech.End = tempEnd
			if float64(currentSpeech.End-currentSpeech.Start) > minSpeechSamples {
				speeches = append(speeches, currentSpeech)
			}
			currentSpeechActive = false
			prevEnd = 0
			nextStart = 0
			tempEnd = 0
			triggered = false
			possibleEnds = nil
			continue
		}
	}

	if currentSpeechActive && float64(audioLengthSamples-currentSpeech.Start) > minSpeechSamples {
		currentSpeech.End = audioLengthSamples
		speeches = append(speeches, currentSpeech)
	}

	// Apply padding
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

	return speeches
}

// getSpeechProbs runs the Silero VAD model on audio and returns per-frame speech probabilities.
func getSpeechProbs(samples []float32) []float32 {
	v, err := govad.New()
	if err != nil {
		return nil
	}

	padLen := vadWindowSize - len(samples)%vadWindowSize
	if padLen < vadWindowSize {
		padded := make([]float32, len(samples)+padLen)
		copy(padded, samples)
		samples = padded
	}

	nFrames := len(samples) / vadWindowSize
	probs := make([]float32, nFrames)
	for i := range nFrames {
		probs[i] = v.Process(samples[i*vadWindowSize : (i+1)*vadWindowSize])
	}
	return probs
}

// chunkMetadata tracks the origin of a batched audio chunk for timestamp restoration.
type chunkMetadata struct {
	offset   float64       // seconds from start of original audio
	duration float64       // seconds of speech in this chunk
	segments []SpeechChunk // original VAD segments composing this chunk
}

// collectChunksBatched groups speech chunks into audio segments of at most
// maxDuration seconds. Returns the audio buffers and their metadata.
// This is a port of Python's collect_chunks with max_duration.
func collectChunksBatched(samples []float32, chunks []SpeechChunk, maxDuration float64) ([][]float32, []chunkMetadata) {
	if len(chunks) == 0 {
		return [][]float32{{}}, []chunkMetadata{{}}
	}

	maxSamples := maxDuration * whisperSampleRate

	var audioChunks [][]float32
	var metadata []chunkMetadata

	var currentAudio []float32
	var currentSegments []SpeechChunk
	var currentDuration float64
	var totalDuration float64

	for _, chunk := range chunks {
		chunkLen := float64(chunk.End - chunk.Start)

		if currentDuration+chunkLen > maxSamples {
			audioChunks = append(audioChunks, currentAudio)
			metadata = append(metadata, chunkMetadata{
				offset:   totalDuration / whisperSampleRate,
				duration: currentDuration / whisperSampleRate,
				segments: currentSegments,
			})
			totalDuration += currentDuration

			start := chunk.Start
			end := chunk.End
			if start > len(samples) {
				start = len(samples)
			}
			if end > len(samples) {
				end = len(samples)
			}
			currentAudio = append([]float32(nil), samples[start:end]...)
			currentSegments = []SpeechChunk{chunk}
			currentDuration = chunkLen
		} else {
			currentSegments = append(currentSegments, chunk)
			start := chunk.Start
			end := chunk.End
			if start > len(samples) {
				start = len(samples)
			}
			if end > len(samples) {
				end = len(samples)
			}
			currentAudio = append(currentAudio, samples[start:end]...)
			currentDuration += chunkLen
		}
	}

	audioChunks = append(audioChunks, currentAudio)
	metadata = append(metadata, chunkMetadata{
		offset:   totalDuration / whisperSampleRate,
		duration: currentDuration / whisperSampleRate,
		segments: currentSegments,
	})

	return audioChunks, metadata
}

// collectChunks concatenates the speech regions from the original audio into
// a single contiguous buffer.
func collectChunks(samples []float32, chunks []SpeechChunk) []float32 {
	if len(chunks) == 0 {
		return nil
	}

	totalLen := 0
	for _, c := range chunks {
		totalLen += c.End - c.Start
	}

	out := make([]float32, 0, totalLen)
	for _, c := range chunks {
		start := c.Start
		end := c.End
		if start > len(samples) {
			start = len(samples)
		}
		if end > len(samples) {
			end = len(samples)
		}
		out = append(out, samples[start:end]...)
	}
	return out
}

// speechTimestampsMap maps timestamps from VAD-compressed audio back to the
// original audio timeline by tracking cumulative silence removed before each chunk.
type speechTimestampsMap struct {
	chunkEndSample      []int
	totalSilenceBefore  []float64
	samplingRate        float64
}

func newSpeechTimestampsMap(chunks []SpeechChunk) *speechTimestampsMap {
	m := &speechTimestampsMap{
		chunkEndSample:     make([]int, len(chunks)),
		totalSilenceBefore: make([]float64, len(chunks)),
		samplingRate:       whisperSampleRate,
	}

	previousEnd := 0
	silentSamples := 0

	for i, chunk := range chunks {
		silentSamples += chunk.Start - previousEnd
		previousEnd = chunk.End

		m.chunkEndSample[i] = chunk.End - silentSamples
		m.totalSilenceBefore[i] = float64(silentSamples) / m.samplingRate
	}

	return m
}

func (m *speechTimestampsMap) getChunkIndex(t float64) int {
	sample := int(t * m.samplingRate)
	idx := sort.SearchInts(m.chunkEndSample, sample+1)
	if idx >= len(m.chunkEndSample) {
		idx = len(m.chunkEndSample) - 1
	}
	return idx
}

func (m *speechTimestampsMap) getOriginalTime(t float64, chunkIndex int) float64 {
	if chunkIndex < 0 || chunkIndex >= len(m.totalSilenceBefore) {
		return t
	}
	return m.totalSilenceBefore[chunkIndex] + t
}

// restoreSegmentTimestamps maps a segment's timestamps (and word timestamps)
// from compressed-audio time back to original-audio time.
func (m *speechTimestampsMap) restoreSegmentTimestamps(seg *Segment) {
	startSec := seg.Start.Seconds()
	endSec := seg.End.Seconds()

	startIdx := m.getChunkIndex(startSec)
	endIdx := m.getChunkIndex(endSec)

	seg.Start = time.Duration(m.getOriginalTime(startSec, startIdx) * float64(time.Second))
	seg.End = time.Duration(m.getOriginalTime(endSec, endIdx) * float64(time.Second))

	for i := range seg.Words {
		wStart := seg.Words[i].Start.Seconds()
		wEnd := seg.Words[i].End.Seconds()
		middle := (wStart + wEnd) / 2
		ci := m.getChunkIndex(middle)
		seg.Words[i].Start = time.Duration(m.getOriginalTime(wStart, ci) * float64(time.Second))
		seg.Words[i].End = time.Duration(m.getOriginalTime(wEnd, ci) * float64(time.Second))
	}
}
