package whisper

import (
	"fmt"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
	"github.com/timohahaa/faster-whisper-go/pyannotevad"
	"github.com/timohahaa/faster-whisper-go/silerovad"
)

// VAD backend identifiers for ModelConfig.VadBackend.
const (
	// VadBackendSilero uses the embedded Silero VAD (default).
	VadBackendSilero = "silero"
	// VadBackendPyannote uses the embedded pyannote segmentation VAD (the
	// official pyannote/segmentation 2022 model).
	VadBackendPyannote = "pyannote"
)

// vadEngine abstracts a Voice Activity Detection backend that turns raw audio
// into speech regions. Implementations are held one-per-model.
type vadEngine interface {
	speechChunks(samples []float32, cfg VadConfig) ([]SpeechChunk, error)
	batchedDefaults() VadConfig
	Close()
}

// Batched-path VAD defaults per backend.
var (
	sileroBatchedDefaults = VadConfig{
		Threshold:            0.20,
		NegThreshold:         0.10,
		MinSilenceDurationMs: 500,
		SpeechPadMs:          800,
	}
	pyannoteBatchedDefaults = VadConfig{
		MinSilenceDurationMs: 160,
	}
)

// sileroEngine is the vadEngine backed by the Silero VAD.
type sileroEngine struct {
	vad      *silerovad.VAD
	defaults VadConfig
}

func (e *sileroEngine) speechChunks(samples []float32, cfg VadConfig) ([]SpeechChunk, error) {
	return GetSpeechTimestamps(e.vad, samples, cfg)
}

func (e *sileroEngine) batchedDefaults() VadConfig {
	return e.defaults
}

func (e *sileroEngine) Close() {
	if e.vad != nil {
		e.vad.Close()
	}
}

// Model is a loaded Whisper model ready for transcription.
type Model struct {
	bridge         *ct2bridge.Model
	vad            vadEngine
	tokenizer      *tokenizer
	nMels          int
	sparseFilters  []melFilterSpan
	isMultilingual bool
}

// Load opens a Whisper model. modelSizeOrPath is either a known model size
// (e.g. "tiny", "large-v3", "turbo") or a path to a local CTranslate2 model
// directory. When a size name is given, the model is downloaded and cached
// locally.
func Load(modelSizeOrPath string, cfg ModelConfig) (*Model, error) {
	modelDir, err := resolveModelPath(modelSizeOrPath, cfg)
	if err != nil {
		return nil, err
	}

	device := cfg.Device
	if device == "" {
		device = "cpu"
	}
	computeType := cfg.ComputeType
	if computeType == "" {
		computeType = "default"
	}

	bridge, err := ct2bridge.Load(
		modelDir,
		device,
		computeType,
		cfg.DeviceIndex,
		cfg.CPUThreads,
		cfg.NumWorkers,
	)
	if err != nil {
		return nil, fmt.Errorf("load ct2 model: %w", err)
	}

	tok, err := loadTokenizer(modelDir)
	if err != nil {
		bridge.Close()
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	nMels := bridge.NMels()
	if nMels == 0 {
		bridge.Close()
		return nil, fmt.Errorf("model reports 0 mel frequency bins")
	}

	dense := computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate)
	sparse := buildSparseFilters(dense, nMels, whisperFreqBins)

	vad, err := newVadEngine(cfg.VadBackend, cfg.VadCPUThreads)
	if err != nil {
		bridge.Close()
		return nil, err
	}

	return &Model{
		bridge:         bridge,
		vad:            vad,
		tokenizer:      tok,
		nMels:          nMels,
		sparseFilters:  sparse,
		isMultilingual: bridge.IsMultilingual(),
	}, nil
}

// Close releases all resources held by the model.
func (m *Model) Close() {
	if m == nil {
		return
	}
	if m.bridge != nil {
		m.bridge.Close()
		m.bridge = nil
	}
	if m.vad != nil {
		m.vad.Close()
		m.vad = nil
	}
}

// newVadEngine constructs the VAD backend selected by backend
// (VadBackendSilero by default, or VadBackendPyannote). vadCPUThreads sets the
// pyannote onnxruntime intra-op thread count (0 = autoscaled default); it is
// ignored by the silero backend.
func newVadEngine(backend string, vadCPUThreads int) (vadEngine, error) {
	switch backend {
	case VadBackendPyannote:
		threads := vadCPUThreads
		if threads <= 0 {
			threads = pyannotevad.DefaultIntraOpThreads
		}
		vad, err := pyannotevad.NewWithThreads(threads)
		if err != nil {
			return nil, fmt.Errorf("init pyannote vad: %w", err)
		}
		return &pyannoteEngine{vad: vad, defaults: pyannoteBatchedDefaults}, nil
	case "", VadBackendSilero:
		vad, err := silerovad.New()
		if err != nil {
			return nil, fmt.Errorf("init silero vad: %w", err)
		}
		return &sileroEngine{vad: vad, defaults: sileroBatchedDefaults}, nil
	default:
		return nil, fmt.Errorf("unknown VAD backend %q", backend)
	}
}

// IsMultilingual reports whether the model supports multiple languages.
func (m *Model) IsMultilingual() bool {
	if m == nil {
		return false
	}
	return m.isMultilingual
}
