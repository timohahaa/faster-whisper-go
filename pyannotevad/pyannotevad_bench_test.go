package pyannotevad

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// benchSamples builds synthetic 16kHz mono audio of the requested length. The
// content is irrelevant to ONNX timing (the model runs the same ops regardless),
// but a non-trivial signal is used to be representative.
func benchSamples(seconds int) []float32 {
	n := seconds * sampleRate
	s := make([]float32, n)
	r := rand.New(rand.NewSource(1))
	for i := range s {
		s[i] = float32(math.Sin(float64(i)*0.01)) * 0.4 * float32(r.Float64())
	}
	return s
}

// BenchmarkScores measures the full VAD forward pass (ONNX sliding-window
// inference + Hamming aggregation) end to end. -benchmem surfaces the large
// overlapping-window input buffer allocated per call.
func BenchmarkScores(b *testing.B) {
	for _, sec := range []int{60, 300, 1150} {
		samples := benchSamples(sec)
		for _, threads := range []int{1, 4} {
			b.Run(fmt.Sprintf("%ds/threads=%d", sec, threads), func(b *testing.B) {
				v, err := NewWithThreads(threads)
				if err != nil {
					b.Skipf("pyannote vad unavailable: %v", err)
				}
				defer v.Close()
				if _, err := v.Scores(samples); err != nil { // warm-up
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := v.Scores(samples); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkAggregate isolates the pure-Go post-processing: Hamming-weighted
// overlap-add of per-window frame scores into a single curve.
func BenchmarkAggregate(b *testing.B) {
	for _, sec := range []int{60, 300, 1150} {
		nSamples := sec * sampleRate
		numChunks := 0
		if nSamples >= WindowSamples {
			numChunks = (nSamples-WindowSamples)/StepSamples + 1
		}
		if nSamples < WindowSamples || (nSamples-WindowSamples)%StepSamples > 0 {
			numChunks++
		}
		perChunkMax := make([]float32, numChunks*NumFramesPerChunk)
		r := rand.New(rand.NewSource(2))
		for i := range perChunkMax {
			perChunkMax[i] = float32(r.Float64())
		}
		b.Run(fmt.Sprintf("%ds", sec), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = aggregate(perChunkMax, numChunks, nSamples)
			}
		})
	}
}
