package whisper

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	// whisperNFFT is the FFT window size: 400 samples = 25 ms at 16 kHz.
	whisperNFFT = 400
	// whisperHopLength is the stride between frames: 160 samples = 10 ms at 16 kHz.
	whisperHopLength = 160
)

// hannWindow returns a periodic Hann window of length n.
// The window suppresses spectral leakage at frame boundaries: without it,
// abrupt signal cutoffs at frame edges produce spurious high-frequency components.
// Formula: w(i) = 0.5 * (1 - cos(2*pi*i / n)).
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
	// Reflect-pad the signal by nFFT/2 on each side so the first and last STFT
	// frames are centered on the start and end of the original signal.
	// This mirrors np.pad(mode='reflect') from the Python Whisper implementation.
	padLen := nFFT / 2
	padded := make([]float64, padLen+len(samples)+padLen)
	for i := range padLen {
		padded[padLen-1-i] = float64(samples[min(i+1, len(samples)-1)])
		padded[padLen+len(samples)+i] = float64(samples[max(len(samples)-2-i, 0)])
	}
	for i, s := range samples {
		padded[padLen+i] = float64(s)
	}

	window := hannWindow(nFFT)
	// Number of positions where a window of width nFFT fits into the padded signal
	// when sliding by hopLength.
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
