package whisper

import "math"

const (
	whisperSampleRate = 16000
	whisperNMels80    = 80
	whisperNMels128   = 128
	whisperChunkLen   = 30 // seconds
	whisperNFrames    = 3000
)

func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// computeMelFilterbank builds triangular mel filterbank of shape [nMels][nFFT/2+1].
func computeMelFilterbank(nMels, nFFT, sampleRate int) [][]float32 {
	freqBins := nFFT/2 + 1
	melMin := hzToMel(0)
	melMax := hzToMel(float64(sampleRate) / 2.0)

	melPoints := make([]float64, nMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(nMels+1)
	}

	hzPoints := make([]float64, len(melPoints))
	for i, m := range melPoints {
		hzPoints[i] = melToHz(m)
	}

	binPoints := make([]float64, len(hzPoints))
	for i, hz := range hzPoints {
		binPoints[i] = hz * float64(nFFT) / float64(sampleRate)
	}

	filters := make([][]float32, nMels)
	for i := range nMels {
		filters[i] = make([]float32, freqBins)
		left := binPoints[i]
		center := binPoints[i+1]
		right := binPoints[i+2]

		for j := range freqBins {
			freq := float64(j)
			if freq >= left && freq <= center && center > left {
				filters[i][j] = float32((freq - left) / (center - left))
			} else if freq > center && freq <= right && right > center {
				filters[i][j] = float32((right - freq) / (right - center))
			}
		}

		// Slaney normalization: divide by mel band width
		enorm := 2.0 / (hzPoints[i+2] - hzPoints[i])
		for j := range freqBins {
			filters[i][j] *= float32(enorm)
		}
	}
	return filters
}

// computeMelSpectrogram computes a log-mel spectrogram matching Whisper's preprocessing.
// Returns a flat []float32 of length nMels*whisperNFrames, row-major (mel bin major).
func computeMelSpectrogram(samples []float32, nMels int) []float32 {
	power := stft(samples, whisperNFFT, whisperHopLength)
	nFrames := len(power)
	freqBins := whisperNFFT/2 + 1

	filters := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)

	// Matmul: mel[i][t] = sum_j(filters[i][j] * power[t][j])
	mel := make([]float64, nMels*nFrames)
	for i := range nMels {
		for t := range nFrames {
			var sum float64
			for j := range freqBins {
				sum += float64(filters[i][j]) * power[t][j]
			}
			mel[i*nFrames+t] = sum
		}
	}

	// Log scale (matching whisper: log10, then *10 for dB-like, but whisper actually
	// uses log_spec = torch.clamp(mel_spec, min=1e-10).log10()
	// then log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
	// then (log_spec + 4.0) / 4.0
	maxVal := math.Inf(-1)
	for i := range mel {
		v := math.Log10(math.Max(mel[i], 1e-10))
		mel[i] = v
		if v > maxVal {
			maxVal = v
		}
	}

	for i := range mel {
		mel[i] = math.Max(mel[i], maxVal-8.0)
		mel[i] = (mel[i] + 4.0) / 4.0
	}

	// Pad or trim to whisperNFrames
	outFrames := whisperNFrames
	out := make([]float32, nMels*outFrames)
	for i := range nMels {
		for t := range outFrames {
			if t < nFrames {
				out[i*outFrames+t] = float32(mel[i*nFrames+t])
			}
			// else remains 0, which after the normalization above acts as silence
		}
	}
	return out
}
