package whisper

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	whisperNFFT      = 400
	whisperHopLength = 160
)

func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// stft computes the Short-Time Fourier Transform.
// Returns magnitudes squared (power spectrum) as [n_frames][n_fft/2+1].
func stft(samples []float32, nFFT, hopLength int) [][]float64 {
	padLen := nFFT / 2
	padded := make([]float64, padLen+len(samples)+padLen)
	for i := 0; i < padLen; i++ {
		// reflect padding
		padded[padLen-1-i] = float64(samples[min(i+1, len(samples)-1)])
		padded[padLen+len(samples)+i] = float64(samples[max(len(samples)-2-i, 0)])
	}
	for i, s := range samples {
		padded[padLen+i] = float64(s)
	}

	window := hannWindow(nFFT)
	nFrames := (len(padded) - nFFT) / hopLength + 1
	freqBins := nFFT/2 + 1

	fft := fourier.NewFFT(nFFT)
	result := make([][]float64, nFrames)

	frame := make([]float64, nFFT)
	for i := range nFrames {
		offset := i * hopLength
		for j := range nFFT {
			frame[j] = padded[offset+j] * window[j]
		}
		coeffs := fft.Coefficients(nil, frame)
		power := make([]float64, freqBins)
		for j := range freqBins {
			re := real(coeffs[j])
			im := imag(coeffs[j])
			power[j] = re*re + im*im
		}
		result[i] = power
	}
	return result
}
