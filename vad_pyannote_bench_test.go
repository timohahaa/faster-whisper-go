package whisper

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/timohahaa/faster-whisper-go/pyannotevad"
)

// benchScoreCurve builds a semi-realistic per-frame speech-activity curve:
// alternating high/low blocks (mean length blockSec) plus noise. Long high
// blocks (blockSec > max duration) exercise the min-cut split path.
func benchScoreCurve(seconds, frameStep, blockSec float64, seed int64) []float32 {
	n := int(seconds / frameStep)
	s := make([]float32, n)
	r := rand.New(rand.NewSource(seed))
	blockFrames := int(blockSec / frameStep)
	if blockFrames < 1 {
		blockFrames = 1
	}
	high := true
	next := 0
	base := float32(0.9)
	for i := 0; i < n; i++ {
		if i >= next {
			high = !high
			next = i + blockFrames/2 + r.Intn(blockFrames+1)
			if high {
				base = 0.9
			} else {
				base = 0.1
			}
		}
		v := base + float32(r.Float64()-0.5)*0.15
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		s[i] = v
	}
	return s
}

// benchChunks builds synthetic speech regions covering ~70% of the timeline with
// 1-5s speech segments separated by 0-3s gaps.
func benchChunks(seconds float64, seed int64) []SpeechChunk {
	r := rand.New(rand.NewSource(seed))
	var chunks []SpeechChunk
	pos := 0.0
	for pos < seconds {
		speechLen := 1 + r.Float64()*4
		gap := r.Float64() * 3
		start := int(pos * whisperSampleRate)
		end := int((pos + speechLen) * whisperSampleRate)
		if end > int(seconds*whisperSampleRate) {
			end = int(seconds * whisperSampleRate)
		}
		if end > start {
			chunks = append(chunks, SpeechChunk{Start: start, End: end})
		}
		pos += speechLen + gap
	}
	return chunks
}

func BenchmarkBinarizeVadScores(b *testing.B) {
	fs := pyannotevad.FrameStep
	cases := []struct {
		name         string
		secs, block  float64
		maxDurationS float64
	}{
		{"60s_normal", 60, 3, 30},
		{"1150s_normal", 1150, 3, 30},
		{"1150s_longruns_mincut", 1150, 90, 30},
	}
	for _, c := range cases {
		scores := benchScoreCurve(c.secs, fs, c.block, 7)
		nSamples := int(c.secs * whisperSampleRate)
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = binarizeVadScores(scores, fs, 0.5, 0.363, c.maxDurationS, nSamples)
			}
		})
	}
}

func BenchmarkApplySpeechPadding(b *testing.B) {
	for _, secs := range []float64{60, 1150} {
		base := benchChunks(secs, 8)
		nSamples := int(secs * whisperSampleRate)
		pad := int(0.4 * whisperSampleRate)
		// Pre-build fresh copies so the in-place mutation is measured cleanly
		// without per-iteration copy allocations polluting -benchmem.
		b.Run(fmt.Sprintf("%.0fs", secs), func(b *testing.B) {
			copies := make([][]SpeechChunk, b.N)
			for i := range copies {
				copies[i] = append([]SpeechChunk(nil), base...)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				applySpeechPadding(copies[i], pad, nSamples)
			}
		})
	}
}

func BenchmarkCollectChunksBatched(b *testing.B) {
	for _, secs := range []float64{60, 1150} {
		samples := make([]float32, int(secs*whisperSampleRate))
		chunks := benchChunks(secs, 9)
		b.Run(fmt.Sprintf("%.0fs", secs), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = collectChunksBatched(samples, chunks, 30)
			}
		})
	}
}

// BenchmarkPyannoteSpeechChunks measures the full VAD post-processing pipeline
// exposed to the transcription path: Scores + binarize + padding.
func BenchmarkPyannoteSpeechChunks(b *testing.B) {
	eng, err := newVadEngine(VadBackendPyannote, 4)
	if err != nil {
		b.Skipf("pyannote vad unavailable: %v", err)
	}
	defer eng.Close()
	for _, secs := range []float64{60, 300, 1150} {
		samples := make([]float32, int(secs*whisperSampleRate))
		for i := range samples {
			samples[i] = float32(math.Sin(float64(i)*0.01)) * 0.3
		}
		cfg := VadConfig{MaxSpeechDurationS: 30}
		cfg.applyDefaults()
		if _, err := eng.speechChunks(samples, cfg); err != nil { // warm-up
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("%.0fs", secs), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := eng.speechChunks(samples, cfg); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
