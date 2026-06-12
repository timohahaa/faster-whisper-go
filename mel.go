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

	framesPerSecond = whisperSampleRate / whisperHopLength // 100
	timePerFrame    = float64(whisperHopLength) / float64(whisperSampleRate)
	// inputStride is the encoder's temporal downsampling factor (conv layers reduce frames by 2x).
	inputStride = 2
)

// hzToMel converts frequency in Hz to the mel scale using the HTK formula
// mel = 2595 * log10(1 + hz/700).
func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// melToHz is the inverse of hzToMel.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// computeMelFilterbank builds a bank of nMels triangular band-pass filters
// stored as a flat slice of length nMels*(nFFT/2+1), row-major (mel bin major).
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

		// Slaney normalization
		enorm := float32(2.0 / (hzPoints[i+2] - hzPoints[i]))
		for j := range freqBins {
			filters[base+j] *= enorm
		}
	}
	return filters
}

// computeMelSpectrogram computes a log-mel spectrogram for audio of any length.
// Returns a flat []float32 of length nMels*totalFrames (row-major, mel-bin-major)
// and the total number of STFT frames.
func computeMelSpectrogram(samples []float32, nMels int) ([]float32, int) {
	power, stftFrames := stft(samples, whisperNFFT, whisperHopLength)
	freqBins := whisperFreqBins

	filters := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)

	out := make([]float32, nMels*stftFrames)

	maxVal := math.Inf(-1)
	for i := range nMels {
		filterBase := i * freqBins
		outBase := i * stftFrames
		for t := range stftFrames {
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

	floor := float32(maxVal - 8.0)
	for i := range out {
		v := out[i]
		if v < floor {
			v = floor
		}
		out[i] = (v + 4.0) / 4.0
	}

	return out, stftFrames
}

// extractMelWindow copies frames [seek, seek+size) from the full mel spectrogram
// into a contiguous buffer of exactly nMels*whisperNFrames elements, zero-padding
// if fewer than whisperNFrames frames are available.
func extractMelWindow(mel []float32, totalFrames, nMels, seek, size int) []float32 {
	if size > whisperNFrames {
		size = whisperNFrames
	}
	if seek+size > totalFrames {
		size = totalFrames - seek
	}

	out := make([]float32, nMels*whisperNFrames)
	for bin := range nMels {
		srcBase := bin * totalFrames
		dstBase := bin * whisperNFrames
		copy(out[dstBase:dstBase+size], mel[srcBase+seek:srcBase+seek+size])
	}
	return out
}
