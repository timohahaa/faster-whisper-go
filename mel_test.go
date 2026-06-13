package whisper

import (
	"math"
	"testing"
)

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
	filters := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)
	mel, totalFrames := computeMelSpectrogram(samples, nMels, filters)

	expectedLen := nMels * totalFrames
	if len(mel) != expectedLen {
		t.Fatalf("mel spectrogram length: got %d, want %d (nMels=%d, frames=%d)", len(mel), expectedLen, nMels, totalFrames)
	}

	if totalFrames <= 0 {
		t.Fatalf("expected positive frame count, got %d", totalFrames)
	}

	for i, v := range mel {
		if v < -2 || v > 2 {
			t.Errorf("mel[%d] = %f, out of expected range [-2, 2]", i, v)
			break
		}
	}
}

func TestComputeMelSpectrogram30s(t *testing.T) {
	nSamples := whisperSampleRate * whisperChunkLen
	samples := make([]float32, nSamples)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
	}

	nMels := 80
	filters := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)
	mel, totalFrames := computeMelSpectrogram(samples, nMels, filters)

	if totalFrames < whisperNFrames {
		t.Errorf("30s audio should produce at least %d frames, got %d", whisperNFrames, totalFrames)
	}

	if len(mel) != nMels*totalFrames {
		t.Fatalf("mel length mismatch: got %d, want %d", len(mel), nMels*totalFrames)
	}
}

func TestExtractMelWindow(t *testing.T) {
	nMels := 2
	totalFrames := 10
	mel := make([]float32, nMels*totalFrames)
	for bin := range nMels {
		for f := range totalFrames {
			mel[bin*totalFrames+f] = float32(bin*100 + f)
		}
	}

	t.Run("FullWindow", func(t *testing.T) {
		window := extractMelWindow(mel, totalFrames, nMels, 0, 5)
		if len(window) != nMels*whisperNFrames {
			t.Fatalf("window length: got %d, want %d", len(window), nMels*whisperNFrames)
		}
		for bin := range nMels {
			for f := range 5 {
				got := window[bin*whisperNFrames+f]
				want := float32(bin*100 + f)
				if got != want {
					t.Errorf("window[bin=%d, frame=%d] = %f, want %f", bin, f, got, want)
				}
			}
			for f := 5; f < whisperNFrames; f++ {
				if window[bin*whisperNFrames+f] != 0 {
					t.Errorf("window[bin=%d, frame=%d] = %f, want 0 (zero-pad)", bin, f, window[bin*whisperNFrames+f])
					break
				}
			}
		}
	})

	t.Run("WithSeekOffset", func(t *testing.T) {
		window := extractMelWindow(mel, totalFrames, nMels, 3, 4)
		for bin := range nMels {
			for f := range 4 {
				got := window[bin*whisperNFrames+f]
				want := float32(bin*100 + 3 + f)
				if got != want {
					t.Errorf("window[bin=%d, frame=%d] = %f, want %f", bin, f, got, want)
				}
			}
		}
	})

	t.Run("ClampToEnd", func(t *testing.T) {
		window := extractMelWindow(mel, totalFrames, nMels, 8, 5)
		for bin := range nMels {
			for f := range 2 {
				got := window[bin*whisperNFrames+f]
				want := float32(bin*100 + 8 + f)
				if got != want {
					t.Errorf("window[bin=%d, frame=%d] = %f, want %f", bin, f, got, want)
				}
			}
			if window[bin*whisperNFrames+2] != 0 {
				t.Errorf("expected zero-pad at frame 2, got %f", window[bin*whisperNFrames+2])
			}
		}
	})
}

func BenchmarkComputeMelSpectrogram(b *testing.B) {
	filters := computeMelFilterbank(80, whisperNFFT, whisperSampleRate)

	b.Run("1s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			computeMelSpectrogram(samples, 80, filters)
		}
	})

	b.Run("30s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate*whisperChunkLen)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			computeMelSpectrogram(samples, 80, filters)
		}
	})
}
