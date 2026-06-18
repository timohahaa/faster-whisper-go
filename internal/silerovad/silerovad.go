// Package silerovad runs the Silero VAD v6 model via onnxruntime to compute
// per-window speech probabilities. It feeds all windows through the model in a
// single batched call (chunked) rather than one window at a time.
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
	initOnce sync.Once
	initErr  error

	session *ort.DynamicAdvancedSession
	// runMu serializes access to the shared onnxruntime session.
	runMu sync.Mutex

	// reusable LSTM state tensors (dimensions are independent of batch size).
	hIn, cIn, hOut, cOut *ort.Tensor[float32]
)

func setup() {
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
			initErr = fmt.Errorf("initialize onnxruntime environment: %w", err)
			return
		}
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		initErr = fmt.Errorf("create session options: %w", err)
		return
	}
	defer opts.Destroy()
	// Silero VAD is tiny; keep it on CPU. A single batched Run already lets
	// onnxruntime parallelize internally across intra-op threads.
	_ = opts.SetIntraOpNumThreads(0) // 0 = onnxruntime default (all cores)
	_ = opts.SetInterOpNumThreads(1)

	session, err = ort.NewDynamicAdvancedSessionWithONNXData(
		modelData,
		[]string{"input", "h", "c"},
		[]string{"speech_probs", "hn", "cn"},
		opts,
	)
	if err != nil {
		initErr = fmt.Errorf("create silero vad session: %w", err)
		return
	}

	stateShape := ort.NewShape(1, 1, stateLen)
	if hIn, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		initErr = fmt.Errorf("alloc h tensor: %w", err)
		return
	}
	if cIn, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		initErr = fmt.Errorf("alloc c tensor: %w", err)
		return
	}
	if hOut, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		initErr = fmt.Errorf("alloc hn tensor: %w", err)
		return
	}
	if cOut, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		initErr = fmt.Errorf("alloc cn tensor: %w", err)
		return
	}
}

// Probs returns one speech probability per 512-sample window. The input is
// zero-padded to a multiple of the window size, so the result length is
// ceil(len(samples)/512).
func Probs(samples []float32) ([]float32, error) {
	initOnce.Do(setup)
	if initErr != nil {
		return nil, initErr
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

	runMu.Lock()
	defer runMu.Unlock()

	// Reset LSTM state to zeros for this call.
	clearFloat32(hIn.GetData())
	clearFloat32(cIn.GetData())

	probs := make([]float32, n)

	for start := 0; start < n; start += encoderBatch {
		end := start + encoderBatch
		if end > n {
			end = n
		}
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

		err = session.Run(
			[]ort.Value{inTensor, hIn, cIn},
			[]ort.Value{outTensor, hOut, cOut},
		)
		if err != nil {
			inTensor.Destroy()
			outTensor.Destroy()
			return nil, fmt.Errorf("run silero vad: %w", err)
		}

		copy(probs[start:end], outTensor.GetData())
		// Carry LSTM state to the next chunk.
		copy(hIn.GetData(), hOut.GetData())
		copy(cIn.GetData(), cOut.GetData())

		inTensor.Destroy()
		outTensor.Destroy()
	}

	return probs, nil
}

func clearFloat32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}
