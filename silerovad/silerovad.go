// Package silerovad runs the Silero VAD v6 model via onnxruntime to compute
// per-window speech probabilities. It feeds all windows through the model in a
// single batched call (chunked) rather than one window at a time.
//
// A VAD is an owned instance (onnxruntime session + its own scratch tensors),
// intended to be held one-per-model. The onnxruntime environment itself is a
// process-wide resource initialized once on first New and kept alive for the
// lifetime of the process.
package silerovad

import (
	_ "embed"
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

//go:embed silero_vad_v6.onnx
var modelData []byte

const (
	// windowSamples is the number of audio samples per VAD window.
	windowSamples = 512
	// contextSamples is the number of trailing samples from the previous window
	// prepended as context to each window.
	contextSamples = 64
	// stateLen is the size of the LSTM hidden/cell state vectors.
	stateLen = 128
	// encoderBatch is the maximum number of windows fed to the model per Run.
	encoderBatch = 10000
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
	// envOnce guards process-wide onnxruntime environment initialization.
	// The environment is genuinely process-global (one per process) and is not
	// torn down per-instance, so it stays alive until the process exits.
	envOnce sync.Once
	envErr  error
)

// ensureEnv initializes the process-wide onnxruntime environment once:
// it resolves the shared library path and initializes the environment.
func ensureEnv() error {
	envOnce.Do(func() {
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

		if !ort.IsInitialized() {
			if err := ort.InitializeEnvironment(); err != nil {
				envErr = fmt.Errorf("initialize onnxruntime environment: %w", err)
			}
		}
	})
	return envErr
}

// VAD is a single Silero VAD instance: it owns an onnxruntime session and its
// own reusable LSTM state tensors. A VAD is safe for concurrent use; calls to
// Probs are serialized by an internal mutex.
//
// Ownership is one VAD per model instance. Create with New and release with
// Close.
type VAD struct {
	session *ort.DynamicAdvancedSession

	mu sync.Mutex

	// reusable LSTM state tensors (dimensions are independent of batch size).
	hIn, cIn, hOut, cOut *ort.Tensor[float32]
}

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
		[]string{"input", "h", "c"},
		[]string{"speech_probs", "hn", "cn"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("create silero vad session: %w", err)
	}

	stateShape := ort.NewShape(1, 1, stateLen)
	if v.hIn, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc h tensor: %w", err)
	}
	if v.cIn, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc c tensor: %w", err)
	}
	if v.hOut, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc hn tensor: %w", err)
	}
	if v.cOut, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc cn tensor: %w", err)
	}

	return v, nil
}

// Probs returns one speech probability per 512-sample window. The input is
// zero-padded to a multiple of the window size, so the result length is
// ceil(len(samples)/512).
func (v *VAD) Probs(samples []float32) ([]float32, error) {
	if v == nil || v.session == nil {
		return nil, fmt.Errorf("silerovad: VAD is closed")
	}

	// Pad to a whole number of windows.
	padded := samples
	if rem := len(samples) % windowSamples; rem != 0 {
		padded = make([]float32, len(samples)+windowSamples-rem)
		copy(padded, samples)
	}
	n := len(padded) / windowSamples
	if n == 0 {
		return nil, nil
	}

	// Build the (n, contextSamples+windowSamples) batched input: each row is the
	// trailing context of the previous window followed by the current window.
	// Row 0's context is zero; subsequent rows reuse the previous window's tail.
	rowLen := contextSamples + windowSamples
	batched := make([]float32, n*rowLen)
	for i := 0; i < n; i++ {
		dst := batched[i*rowLen:]
		if i > 0 {
			prev := padded[(i-1)*windowSamples:]
			copy(dst[:contextSamples], prev[windowSamples-contextSamples:windowSamples])
		}
		copy(dst[contextSamples:rowLen], padded[i*windowSamples:(i+1)*windowSamples])
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	// Reset LSTM state to zeros for this call.
	clearFloat32(v.hIn.GetData())
	clearFloat32(v.cIn.GetData())

	probs := make([]float32, n)

	for start := 0; start < n; start += encoderBatch {
		end := start + encoderBatch
		end = min(end, n)
		bs := end - start

		inTensor, err := ort.NewTensor(ort.NewShape(int64(bs), int64(rowLen)), batched[start*rowLen:end*rowLen])
		if err != nil {
			return nil, fmt.Errorf("create input tensor: %w", err)
		}
		outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(bs)))
		if err != nil {
			inTensor.Destroy()
			return nil, fmt.Errorf("create output tensor: %w", err)
		}

		err = v.session.Run(
			[]ort.Value{inTensor, v.hIn, v.cIn},
			[]ort.Value{outTensor, v.hOut, v.cOut},
		)
		if err != nil {
			inTensor.Destroy()
			outTensor.Destroy()
			return nil, fmt.Errorf("run silero vad: %w", err)
		}

		copy(probs[start:end], outTensor.GetData())
		// Carry LSTM state to the next chunk.
		copy(v.hIn.GetData(), v.hOut.GetData())
		copy(v.cIn.GetData(), v.cOut.GetData())

		inTensor.Destroy()
		outTensor.Destroy()
	}

	return probs, nil
}

func (v *VAD) Close() error {
	if v == nil {
		return nil
	}
	if v.hIn != nil {
		v.hIn.Destroy()
		v.hIn = nil
	}
	if v.cIn != nil {
		v.cIn.Destroy()
		v.cIn = nil
	}
	if v.hOut != nil {
		v.hOut.Destroy()
		v.hOut = nil
	}
	if v.cOut != nil {
		v.cOut.Destroy()
		v.cOut = nil
	}
	if v.session != nil {
		v.session.Destroy()
		v.session = nil
	}
	return nil
}

func clearFloat32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}
