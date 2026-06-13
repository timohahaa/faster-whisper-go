package whisper

import "math"

const (
	whisperSampleRate = 16000
	whisperChunkLen   = 30 // seconds
	// whisperNFrames is the number of STFT frames in one 30-second chunk:
	// sampleRate * chunkLen / hopLength = 16000 * 30 / 160 = 3000.
	whisperNFrames = 3000

	framesPerSecond = whisperSampleRate / whisperHopLength // 100
	timePerFrame    = float64(whisperHopLength) / float64(whisperSampleRate)
	// inputStride is the encoder's temporal downsampling factor (conv layers reduce frames by 2x).
	inputStride = 2
)

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
	mels := make([]float64, nPoints)
	freqs := make([]float64, nPoints)
	for i := range nPoints {
		m := minMel + float64(i)*(maxMel-minMel)/float64(nPoints-1)
		mels[i] = m
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

// computeMelSpectrogram computes a log-mel spectrogram for audio of any length.
// filters is a precomputed mel filterbank from computeMelFilterbank.
// Returns a flat []float32 of length nMels*totalFrames (row-major, mel-bin-major)
// and the total number of STFT frames.
func computeMelSpectrogram(samples []float32, nMels int, filters []float32) ([]float32, int) {
	padded := make([]float32, len(samples)+whisperHopLength)
	copy(padded, samples)

	power, rawFrames := stft(padded, whisperNFFT, whisperHopLength)
	stftFrames := rawFrames - 1
	freqBins := whisperFreqBins

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
