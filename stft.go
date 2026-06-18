package whisper

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	// whisperNFFT is the FFT window size: 400 samples = 25 ms at 16 kHz.
	whisperNFFT = 400
	// whisperHopLength is the stride between frames: 160 samples = 10 ms at 16 kHz.
	whisperHopLength = 160
	// whisperFreqBins is nFFT/2+1 — the number of unique frequency bins in an FFT.
	whisperFreqBins = whisperNFFT/2 + 1
)

var whisperHannWindow = makeHannWindow(whisperNFFT)

func makeHannWindow(n int) []float64 {
	w := make([]float64, n)
	coeff := 2 * math.Pi / float64(n)
	for i := range w {
		w[i] = 0.5 * (1 - math.Cos(coeff*float64(i)))
	}
	return w
}

type fftState struct {
	fft      *fourier.FFT
	coeffBuf []complex128
	frame    []float64
}

var fftPool = sync.Pool{
	New: func() any {
		return &fftState{
			fft:      fourier.NewFFT(whisperNFFT),
			coeffBuf: make([]complex128, whisperFreqBins),
			frame:    make([]float64, whisperNFFT),
		}
	},
}

// stft computes the Short-Time Fourier Transform.
// Returns magnitudes squared (power spectrum) as a flat float32 slice of length
// nFrames * freqBins (row-major, frame-major), plus the frame count.
// extraPadRight zero-samples are appended to the signal before reflect-padding.
func stft(samples []float32, nFFT, hopLength, extraPadRight int) (power []float32, nFrames int) {
	padLen := nFFT / 2
	signalLen := len(samples) + extraPadRight
	paddedLen := padLen + signalLen + padLen
	padded := make([]float64, paddedLen)
	for i := range padLen {
		padded[padLen-1-i] = float64(samples[min(i+1, len(samples)-1)])
	}
	// Right reflect-pad uses the extended signal (zeros beyond samples).
	for i := range padLen {
		idx := signalLen - 2 - i
		if idx >= 0 && idx < len(samples) {
			padded[padLen+signalLen+i] = float64(samples[idx])
		}
	}
	for i, s := range samples {
		padded[padLen+i] = float64(s)
	}

	window := whisperHannWindow
	freqBins := nFFT/2 + 1
	nFrames = (paddedLen-nFFT)/hopLength + 1

	st := fftPool.Get().(*fftState)
	coeffBuf := st.coeffBuf
	frame := st.frame
	power = make([]float32, nFrames*freqBins)

	for i := range nFrames {
		offset := i * hopLength
		seg := padded[offset : offset+nFFT]
		for j, w := range window {
			frame[j] = seg[j] * w
		}
		st.fft.Coefficients(coeffBuf, frame)
		base := i * freqBins
		for j, c := range coeffBuf {
			re := real(c)
			im := imag(c)
			power[base+j] = float32(re*re + im*im)
		}
	}

	fftPool.Put(st)
	return power, nFrames
}
