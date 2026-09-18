// Package pyannotevad runs the official pyannote/segmentation (2022) VAD model
// via onnxruntime to compute a per-frame speech-activity curve.
//
// The model is a frame-level, multi-speaker segmentation network: it takes a
// fixed 5s window (80000 samples @ 16kHz) and outputs per-frame activity for
// three speaker slots. Following pyannote's VoiceActivityDetection pipeline,
// the three slots are collapsed with a per-frame max (pre-aggregation hook) and
// the sliding windows are combined with Hamming-weighted overlap-add
// aggregation into a single speech-activity curve for the whole audio.
//
// It always runs on CPU. The ONNX weights are embedded in the binary, mirroring
// the silerovad package. The onnxruntime environment is a process-wide resource
// shared with silerovad (initialized once per process).
package pyannotevad

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

//go:embed pyannote_vad.onnx
var modelData []byte

const (
	// WindowSamples is the model's fixed input window: 5.0s @ 16kHz.
	WindowSamples = 80000
	// StepSamples is the hop between consecutive windows: 0.5s @ 16kHz
	// (pyannote Inference default step = 10% of the 5s window duration).
	StepSamples = 8000
	// NumFramesPerChunk is the number of output frames per 5s window.
	NumFramesPerChunk = 293
	// NumClasses is the number of speaker slots the segmentation model emits.
	NumClasses = 3
	// FrameStep is the per-frame resolution in seconds, as reported by the
	// model's example output (≈ 5.0/293). Used for overlap-add frame indexing.
	FrameStep = 0.017064846416382253

	sampleRate = 16000
	// batchWindows is the max number of 5s windows fed to the model per Run.
	batchWindows = 32
)

// candidateLibPaths lists common locations of the onnxruntime shared library,
// tried in order when ONNXRUNTIME_SHARED_LIBRARY_PATH is not set.
func candidateLibPaths() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
	default:
		return []string{
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/lib/x86_64-linux-gnu/libonnxruntime.so",
		}
	}
}

var (
	envOnce sync.Once
	envErr  error
)

// ensureEnv initializes the process-wide onnxruntime environment once. The
// environment is shared with the silerovad package; if it is already
// initialized (by either package) this is a no-op.
func ensureEnv() error {
	envOnce.Do(func() {
		if ort.IsInitialized() {
			return
		}
		if path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); path != "" {
			ort.SetSharedLibraryPath(path)
		} else {
			for _, p := range candidateLibPaths() {
				if _, err := os.Stat(p); err == nil {
					ort.SetSharedLibraryPath(p)
					break
				}
			}
		}
		if err := ort.InitializeEnvironment(); err != nil {
			envErr = fmt.Errorf("initialize onnxruntime environment: %w", err)
		}
	})
	return envErr
}

// VAD is a single pyannote VAD instance owning an onnxruntime session. It is
// safe for concurrent use; calls to Scores are serialized by an internal mutex.
type VAD struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
}

// New creates a pyannote VAD instance backed by the embedded ONNX model.
func New() (_ *VAD, err error) {
	if err := ensureEnv(); err != nil {
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()

	_ = opts.SetIntraOpNumThreads(1)
	_ = opts.SetInterOpNumThreads(1)

	v := &VAD{}
	defer func() {
		if err != nil {
			v.Close()
		}
	}()

	v.session, err = ort.NewDynamicAdvancedSessionWithONNXData(
		modelData,
		[]string{"input"},
		[]string{"output"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("create pyannote vad session: %w", err)
	}

	return v, nil
}

// Scores runs the segmentation model over the whole audio with a sliding 5s
// window and returns the aggregated per-frame speech-activity curve (values in
// [0,1]). The curve resolution is FrameStep seconds per sample. The audio must
// be 16kHz mono float32.
func (v *VAD) Scores(samples []float32) ([]float32, error) {
	if v == nil || v.session == nil {
		return nil, fmt.Errorf("pyannotevad: VAD is closed")
	}

	n := len(samples)
	if n == 0 {
		return nil, nil
	}

	// Number of complete 5s windows plus a final zero-padded partial window,
	// matching pyannote Inference.slide.
	numComplete := 0
	if n >= WindowSamples {
		numComplete = (n-WindowSamples)/StepSamples + 1
	}
	hasLast := n < WindowSamples || (n-WindowSamples)%StepSamples > 0
	numChunks := numComplete
	if hasLast {
		numChunks++
	}
	if numChunks == 0 {
		return nil, nil
	}

	// Assemble the (numChunks, WindowSamples) input buffer. The trailing partial
	// window is zero-padded to WindowSamples.
	input := make([]float32, numChunks*WindowSamples)
	for c := 0; c < numComplete; c++ {
		off := c * StepSamples
		copy(input[c*WindowSamples:(c+1)*WindowSamples], samples[off:off+WindowSamples])
	}
	if hasLast {
		off := numComplete * StepSamples
		copy(input[numComplete*WindowSamples:], samples[off:n])
	}

	// Per-window, per-frame max over the speaker slots (pre-aggregation hook).
	perChunkMax := make([]float32, numChunks*NumFramesPerChunk)

	v.mu.Lock()
	defer v.mu.Unlock()

	for start := 0; start < numChunks; start += batchWindows {
		end := min(start+batchWindows, numChunks)
		bs := end - start

		inTensor, err := ort.NewTensor(
			ort.NewShape(int64(bs), 1, int64(WindowSamples)),
			input[start*WindowSamples:end*WindowSamples],
		)
		if err != nil {
			return nil, fmt.Errorf("create input tensor: %w", err)
		}
		outTensor, err := ort.NewEmptyTensor[float32](
			ort.NewShape(int64(bs), int64(NumFramesPerChunk), int64(NumClasses)),
		)
		if err != nil {
			inTensor.Destroy()
			return nil, fmt.Errorf("create output tensor: %w", err)
		}

		if err := v.session.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
			inTensor.Destroy()
			outTensor.Destroy()
			return nil, fmt.Errorf("run pyannote vad: %w", err)
		}

		od := outTensor.GetData()
		for b := 0; b < bs; b++ {
			for f := 0; f < NumFramesPerChunk; f++ {
				base := (b*NumFramesPerChunk + f) * NumClasses
				m := od[base]
				for k := 1; k < NumClasses; k++ {
					if od[base+k] > m {
						m = od[base+k]
					}
				}
				perChunkMax[(start+b)*NumFramesPerChunk+f] = m
			}
		}

		inTensor.Destroy()
		outTensor.Destroy()
	}

	return aggregate(perChunkMax, numChunks, n), nil
}

// aggregate combines the per-window frame scores into a single curve using
// Hamming-weighted overlap-add, then crops to the true audio length. This
// mirrors pyannote's Inference.aggregate (hamming=True, warm_up=(0,0),
// missing=0) followed by the "loose" crop to the waveform duration.
func aggregate(perChunkMax []float32, numChunks, nSamples int) []float32 {
	ham := make([]float64, NumFramesPerChunk)
	for i := range ham {
		ham[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(NumFramesPerChunk-1))
	}

	chunkStep := float64(StepSamples) / sampleRate  // 0.5s
	chunkDur := float64(WindowSamples) / sampleRate // 5.0s

	tEnd := chunkDur + float64(numChunks-1)*chunkStep
	numFrames := closestFrame(tEnd) + 1
	if numFrames < 1 {
		numFrames = 1
	}

	agg := make([]float64, numFrames)
	cnt := make([]float64, numFrames)

	for c := 0; c < numChunks; c++ {
		sf := closestFrame(float64(c) * chunkStep)
		for j := 0; j < NumFramesPerChunk; j++ {
			idx := sf + j
			if idx < 0 || idx >= numFrames {
				continue
			}
			agg[idx] += float64(perChunkMax[c*NumFramesPerChunk+j]) * ham[j]
			cnt[idx] += ham[j]
		}
	}

	const eps = 1e-12
	out := make([]float32, numFrames)
	for i := range out {
		d := cnt[i]
		if d < eps {
			d = eps
		}
		out[i] = float32(agg[i] / d)
	}

	// Crop (loose) to the real audio duration, dropping frames produced only by
	// the zero-padded tail of the last window.
	nOut := int(math.Floor(float64(nSamples)/sampleRate/FrameStep)) + 1
	if nOut >= 0 && nOut < len(out) {
		out = out[:nOut]
	}
	return out
}

// closestFrame maps a time (seconds) to the nearest frame index of the output
// resolution, matching pyannote.core SlidingWindow.closest_frame for a window
// with start=0 and duration=step=FrameStep: round((t - 0.5*step)/step).
// numpy's rint rounds halves to even, so RoundToEven is used.
func closestFrame(t float64) int {
	return int(math.RoundToEven((t - 0.5*FrameStep) / FrameStep))
}

// Close releases the onnxruntime session.
func (v *VAD) Close() error {
	if v == nil {
		return nil
	}
	if v.session != nil {
		v.session.Destroy()
		v.session = nil
	}
	return nil
}
