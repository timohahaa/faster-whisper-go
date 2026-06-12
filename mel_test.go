package whisper

import (
	"math"
	"testing"
)

func TestHzToMelRoundtrip(t *testing.T) {
	freqs := []float64{0, 100, 440, 1000, 4000, 8000}
	for _, hz := range freqs {
		mel := hzToMel(hz)
		got := melToHz(mel)
		if diff := math.Abs(got - hz); diff > 1e-6 {
			t.Errorf("roundtrip %f Hz: melToHz(hzToMel(%f)) = %f, diff = %e", hz, hz, got, diff)
		}
	}
}

func TestComputeMelFilterbank(t *testing.T) {
	nMels := 80
	nFFT := 400
	sr := 16000

	filters := computeMelFilterbank(nMels, nFFT, sr)
	freqBins := nFFT/2 + 1

	if len(filters) != nMels*freqBins {
		t.Fatalf("expected %d total filter values, got %d", nMels*freqBins, len(filters))
	}

	for i := range nMels {
		hasNonZero := false
		base := i * freqBins
		for j := range freqBins {
			if filters[base+j] > 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Errorf("filter[%d] is all zeros", i)
		}
	}

	for i := range nMels {
		base := i * freqBins
		for j := range freqBins {
			if filters[base+j] < 0 {
				t.Errorf("filter[%d][%d] = %f < 0", i, j, filters[base+j])
			}
		}
	}
}

func TestComputeMelSpectrogram(t *testing.T) {
	nSamples := whisperSampleRate
	samples := make([]float32, nSamples)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
	}

	nMels := 80
	mel := computeMelSpectrogram(samples, nMels)

	expectedLen := nMels * whisperNFrames
	if len(mel) != expectedLen {
		t.Fatalf("mel spectrogram length: got %d, want %d", len(mel), expectedLen)
	}

	for i, v := range mel {
		if v < -2 || v > 2 {
			t.Errorf("mel[%d] = %f, out of expected range [-2, 2]", i, v)
			break
		}
	}
}

func BenchmarkComputeMelSpectrogram(b *testing.B) {
	b.Run("1s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			computeMelSpectrogram(samples, 80)
		}
	})

	b.Run("30s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate*whisperChunkLen)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			computeMelSpectrogram(samples, 80)
		}
	})
}
