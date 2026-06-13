package whisper

import "testing"

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
	nMels := 80
	filters := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)
	sparse := buildSparseFilters(filters, nMels, whisperFreqBins)

	t.Run("1s", func(t *testing.T) {
		mel, totalFrames := computeMelSpectrogram(makeSineWave(440, whisperSampleRate, 1.0), nMels, sparse)

		if totalFrames <= 0 {
			t.Fatalf("expected positive frame count, got %d", totalFrames)
		}
		if len(mel) != nMels*totalFrames {
			t.Fatalf("mel spectrogram length: got %d, want %d (nMels=%d, frames=%d)", len(mel), nMels*totalFrames, nMels, totalFrames)
		}
		for i, v := range mel {
			if v < -2 || v > 2 {
				t.Errorf("mel[%d] = %f, out of expected range [-2, 2]", i, v)
				break
			}
		}
	})

	t.Run("30s", func(t *testing.T) {
		mel, totalFrames := computeMelSpectrogram(makeSineWave(440, whisperSampleRate, float64(whisperChunkLen)), nMels, sparse)

		if totalFrames < whisperNFrames {
			t.Errorf("30s audio should produce at least %d frames, got %d", whisperNFrames, totalFrames)
		}
		if len(mel) != nMels*totalFrames {
			t.Fatalf("mel length mismatch: got %d, want %d", len(mel), nMels*totalFrames)
		}
	})
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
	sparse := buildSparseFilters(filters, 80, whisperFreqBins)

	b.Run("1s", func(b *testing.B) {
		samples := makeSineWave(440, whisperSampleRate, 1.0)
		b.ResetTimer()
		for b.Loop() {
			computeMelSpectrogram(samples, 80, sparse)
		}
	})

	b.Run("30s", func(b *testing.B) {
		samples := makeSineWave(440, whisperSampleRate, float64(whisperChunkLen))
		b.ResetTimer()
		for b.Loop() {
			computeMelSpectrogram(samples, 80, sparse)
		}
	})
}
