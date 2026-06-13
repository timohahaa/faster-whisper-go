package whisper

import "math"

func makeSineWave(freqHz float64, sampleRate int, durationSec float64) []float32 {
	n := int(float64(sampleRate) * durationSec)
	samples := make([]float32, n)
	for i := range n {
		samples[i] = float32(math.Sin(2 * math.Pi * freqHz * float64(i) / float64(sampleRate)))
	}
	return samples
}
