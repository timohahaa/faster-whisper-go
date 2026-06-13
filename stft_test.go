package whisper

import (
	"math"
	"testing"
)

func TestHannWindow(t *testing.T) {
	t.Run("Properties", func(t *testing.T) {
		n := 400
		w := makeHannWindow(n)
		if len(w) != n {
			t.Fatalf("hannWindow(%d) length = %d, want %d", n, len(w), n)
		}

		if w[0] != 0 {
			t.Errorf("w[0] = %f, want 0", w[0])
		}

		mid := n / 2
		if w[mid] < 0.99 {
			t.Errorf("w[%d] = %f, want ~1.0", mid, w[mid])
		}

		for i := 1; i < n/2; i++ {
			diff := math.Abs(w[i] - w[n-i])
			if diff > 1e-10 {
				t.Errorf("asymmetry at i=%d: w[%d]=%f, w[%d]=%f, diff=%e", i, i, w[i], n-i, w[n-i], diff)
				break
			}
		}

		for i, v := range w {
			if v < 0 || v > 1.0+1e-10 {
				t.Errorf("w[%d] = %f, out of [0, 1]", i, v)
				break
			}
		}
	})

	t.Run("ExactN4", func(t *testing.T) {
		w := makeHannWindow(4)
		if len(w) != 4 {
			t.Fatalf("length = %d, want 4", len(w))
		}
		expected := []float64{0, 0.5, 1.0, 0.5}
		for i, want := range expected {
			if diff := math.Abs(w[i] - want); diff > 1e-10 {
				t.Errorf("w[%d] = %f, want %f", i, w[i], want)
			}
		}
	})

	t.Run("Cached", func(t *testing.T) {
		w1 := whisperHannWindow
		w2 := whisperHannWindow
		if &w1[0] != &w2[0] {
			t.Error("whisperHannWindow should be a single shared slice")
		}
	})
}

func TestStft(t *testing.T) {
	t.Run("Silence", func(t *testing.T) {
		samples := make([]float32, whisperSampleRate)
		power, nFrames := stft(samples, whisperNFFT, whisperHopLength)

		if nFrames == 0 {
			t.Fatal("stft returned 0 frames")
		}

		freqBins := whisperNFFT/2 + 1
		if len(power) != nFrames*freqBins {
			t.Fatalf("power length: got %d, want %d", len(power), nFrames*freqBins)
		}

		for i := range nFrames {
			for j := range freqBins {
				v := power[i*freqBins+j]
				if v > 1e-10 {
					t.Errorf("silence: frame[%d][%d] = %e, want ~0", i, j, v)
					return
				}
			}
		}
	})

	t.Run("SineWavePeak", func(t *testing.T) {
		freq := 1000.0
		nSamples := whisperSampleRate
		samples := make([]float32, nSamples)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / float64(whisperSampleRate)))
		}

		power, nFrames := stft(samples, whisperNFFT, whisperHopLength)
		if nFrames == 0 {
			t.Fatal("stft returned 0 frames")
		}

		freqBins := whisperNFFT/2 + 1
		// Expected bin for 1000 Hz: bin = freq * nFFT / sampleRate = 1000 * 400 / 16000 = 25.
		expectedBin := 25
		maxBin := 0
		maxVal := 0.0
		frame5 := 5
		for j := range freqBins {
			v := power[frame5*freqBins+j]
			if v > maxVal {
				maxVal = v
				maxBin = j
			}
		}
		diff := maxBin - expectedBin
		if diff < -1 || diff > 1 {
			t.Errorf("peak at bin %d, expected near bin %d", maxBin, expectedBin)
		}
	})

	t.Run("FrameCount", func(t *testing.T) {
		nSamples := whisperSampleRate * whisperChunkLen
		samples := make([]float32, nSamples)
		_, nFrames := stft(samples, whisperNFFT, whisperHopLength)

		expected := 3001
		if nFrames != expected {
			t.Errorf("30s audio: got %d frames, want %d", nFrames, expected)
		}
	})
}

func BenchmarkStft(b *testing.B) {
	b.Run("1s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			stft(samples, whisperNFFT, whisperHopLength)
		}
	})

	b.Run("30s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate*whisperChunkLen)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			stft(samples, whisperNFFT, whisperHopLength)
		}
	})
}
