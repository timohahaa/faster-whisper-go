package whisper

import "math"

const (
	whisperSampleRate = 16000
	whisperNMels80    = 80  // mel bins for Whisper v1/v2 models
	whisperNMels128   = 128 // mel bins for Whisper large-v3
	whisperChunkLen   = 30  // seconds
	// whisperNFrames is the number of STFT frames in one 30-second chunk:
	// sampleRate * chunkLen / hopLength = 16000 * 30 / 160 = 3000.
	whisperNFrames = 3000
)

// hzToMel converts frequency in Hz to the mel scale using the HTK formula
// mel = 2595 * log10(1 + hz/700).
// The mel scale models human nonlinear pitch perception: low frequencies
// are resolved in fine detail while high frequencies are compressed.
func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// melToHz is the inverse of hzToMel.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// computeMelFilterbank builds a bank of nMels triangular band-pass filters
// stored as a flat slice of length nMels*(nFFT/2+1), row-major (mel bin major).
// Center frequencies are evenly spaced on the mel scale.
func computeMelFilterbank(nMels, nFFT, sampleRate int) []float32 {
	freqBins := nFFT/2 + 1
	melMin := hzToMel(0)
	melMax := hzToMel(float64(sampleRate) / 2.0)

	nPoints := nMels + 2
	melPoints := make([]float64, nPoints)
	hzPoints := make([]float64, nPoints)
	binPoints := make([]float64, nPoints)

	melStep := (melMax - melMin) / float64(nMels+1)
	nFFTf := float64(nFFT)
	srf := float64(sampleRate)
	for i := range nPoints {
		m := melMin + float64(i)*melStep
		melPoints[i] = m
		hz := melToHz(m)
		hzPoints[i] = hz
		binPoints[i] = hz * nFFTf / srf
	}

	filters := make([]float32, nMels*freqBins)
	for i := range nMels {
		left := binPoints[i]
		center := binPoints[i+1]
		right := binPoints[i+2]
		base := i * freqBins

		for j := range freqBins {
			freq := float64(j)
			if freq >= left && freq <= center && center > left {
				filters[base+j] = float32((freq - left) / (center - left))
			} else if freq > center && freq <= right && right > center {
				filters[base+j] = float32((right - freq) / (right - center))
			}
		}

		// Slaney normalization: scale each triangle by 2/(f_right - f_left) so its
		// area is constant. This ensures narrow low-frequency bands and wide
		// high-frequency bands contribute comparable energy values.
		enorm := float32(2.0 / (hzPoints[i+2] - hzPoints[i]))
		for j := range freqBins {
			filters[base+j] *= enorm
		}
	}
	return filters
}

// computeMelSpectrogram computes a log-mel spectrogram matching Whisper's preprocessing.
// Returns a flat []float32 of length nMels*whisperNFrames, row-major (mel bin major).
func computeMelSpectrogram(samples []float32, nMels int) []float32 {
	power, stftFrames := stft(samples, whisperNFFT, whisperHopLength)
	freqBins := whisperFreqBins

	filters := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)

	outFrames := whisperNFrames
	nFrames := min(stftFrames, outFrames)

	// Fused matmul + log-normalization into the output buffer directly.
	// mel[i][t] = sum_j(filters[i][j] * power[t][j])
	out := make([]float32, nMels*outFrames)

	maxVal := math.Inf(-1)
	for i := range nMels {
		filterBase := i * freqBins
		outBase := i * outFrames
		for t := range nFrames {
			var sum float64
			powerBase := t * freqBins
			for j := range freqBins {
				sum += float64(filters[filterBase+j]) * power[powerBase+j]
			}
			v := math.Log10(math.Max(sum, 1e-10))
			out[outBase+t] = float32(v)
			if v > maxVal {
				maxVal = v
			}
		}
	}

	// Log-normalization matching faster-whisper's Python preprocessing:
	//
	// 1) log10(clamp(x, min=1e-10)) — already done above.
	//
	// 2) max(x, global_max - 8.0) — floor the dynamic range to ~80 dB below
	//    the loudest bin.
	//
	// 3) (x + 4.0) / 4.0 — shift and scale into roughly [-1, 1].
	floor := float32(maxVal - 8.0)
	for i := range out {
		v := out[i]
		if v < floor {
			v = floor
		}
		out[i] = (v + 4.0) / 4.0
	}

	return out
}
