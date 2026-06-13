package whisper

import "math"

const (
	whisperSampleRate = 16000
	whisperChunkLen   = 30 // seconds
	// whisperNFrames is the number of STFT frames in one 30-second chunk:
	// sampleRate * chunkLen / hopLength = 16000 * 30 / 160 = 3000.
	whisperNFrames = 3000

	timePerFrame      = float64(whisperHopLength) / float64(whisperSampleRate)
	framesPerSecond   = float64(whisperSampleRate) / float64(whisperHopLength)
	// inputStride is the encoder's temporal downsampling factor (conv layers reduce frames by 2x).
	inputStride = 2
)

// melFilterSpan represents a non-zero range within a single mel filter row.
// Triangular mel filters are mostly zeros; storing only the non-zero span
// avoids iterating over ~95% zero values during the matmul.
type melFilterSpan struct {
	start   int       // first non-zero freq bin index
	weights []float32 // non-zero filter values (length = end-start)
}

// computeMelFilterbank builds a bank of nMels triangular band-pass filters
// using the Slaney mel scale (linear below 1000 Hz, logarithmic above).
// Returns a flat slice of length nMels*(nFFT/2+1), row-major (mel bin major).
//
// This matches the Python faster-whisper FeatureExtractor.get_mel_filters exactly.
func computeMelFilterbank(nMels, nFFT, sampleRate int) []float32 {
	freqBins := nFFT/2 + 1

	fftfreqs := make([]float64, freqBins)
	for i := range freqBins {
		fftfreqs[i] = float64(i) * float64(sampleRate) / float64(nFFT)
	}

	const (
		minMel    = 0.0
		maxMel    = 45.245640471924965
		fMin      = 0.0
		fSp       = 200.0 / 3.0
		minLogHz  = 1000.0
		minLogMel = (minLogHz - fMin) / fSp
	)
	logstep := math.Log(6.4) / 27.0

	nPoints := nMels + 2
	freqs := make([]float64, nPoints)
	for i := range nPoints {
		m := minMel + float64(i)*(maxMel-minMel)/float64(nPoints-1)
		if m < minLogMel {
			freqs[i] = fMin + fSp*m
		} else {
			freqs[i] = minLogHz * math.Exp(logstep*(m-minLogMel))
		}
	}

	fdiff := make([]float64, nPoints-1)
	for i := range fdiff {
		fdiff[i] = freqs[i+1] - freqs[i]
	}

	filters := make([]float32, nMels*freqBins)
	for i := range nMels {
		base := i * freqBins
		for j := range freqBins {
			lower := (fftfreqs[j] - freqs[i]) / fdiff[i]
			upper := (freqs[i+2] - fftfreqs[j]) / fdiff[i+1]
			v := math.Min(lower, upper)
			if v > 0 {
				filters[base+j] = float32(v)
			}
		}

		enorm := float32(2.0 / (freqs[i+2] - freqs[i]))
		for j := range freqBins {
			filters[base+j] *= enorm
		}
	}
	return filters
}

// buildSparseFilters extracts the non-zero spans from the dense filterbank.
func buildSparseFilters(filters []float32, nMels, freqBins int) []melFilterSpan {
	spans := make([]melFilterSpan, nMels)
	for i := range nMels {
		base := i * freqBins
		start := -1
		end := 0
		for j := range freqBins {
			if filters[base+j] != 0 {
				if start < 0 {
					start = j
				}
				end = j + 1
			}
		}
		if start < 0 {
			start = 0
		}
		spans[i] = melFilterSpan{
			start:   start,
			weights: make([]float32, end-start),
		}
		copy(spans[i].weights, filters[base+start:base+end])
	}
	return spans
}

// computeMelSpectrogram computes a log-mel spectrogram for audio of any length.
// sparse is a precomputed sparse mel filterbank from buildSparseFilters.
// Returns a flat []float32 of length nMels*totalFrames (row-major, mel-bin-major)
// and the total number of STFT frames.
func computeMelSpectrogram(samples []float32, nMels int, sparse []melFilterSpan) ([]float32, int) {
	power, rawFrames := stft(samples, whisperNFFT, whisperHopLength, whisperHopLength)
	stftFrames := rawFrames - 1
	freqBins := whisperFreqBins

	out := make([]float32, nMels*stftFrames)

	// Frame-outer loop: reads each frame of power once (good L2 cache behavior),
	// applies all mel filters to it. Sparse filters skip zero regions.
	var maxVal float32 = -1e30
	for t := range stftFrames {
		powerSlice := power[t*freqBins : t*freqBins+freqBins]
		for i, sp := range sparse {
			var sum float32
			start := sp.start
			for k, w := range sp.weights {
				sum += w * powerSlice[start+k]
			}
			if sum < 1e-10 {
				sum = 1e-10
			}
			v := float32(math.Log10(float64(sum)))
			out[i*stftFrames+t] = v
			if v > maxVal {
				maxVal = v
			}
		}
	}

	floor := maxVal - 8.0
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
