package whisper

import (
	"fmt"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

// Model is a loaded Whisper model ready for transcription.
type Model struct {
	bridge        *ct2bridge.Model
	tokenizer     *tokenizer
	nMels         int
	sparseFilters []melFilterSpan
}

// Load opens a Whisper model. modelSizeOrPath is either a known model size
// (e.g. "tiny", "large-v3", "turbo") or a path to a local CTranslate2 model
// directory. When a size name is given, the model is downloaded from HuggingFace
// Hub and cached locally.
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

	return &Model{
		bridge:        bridge,
		tokenizer:     tok,
		nMels:         nMels,
		sparseFilters: sparse,
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
}

// IsMultilingual reports whether the model supports multiple languages.
func (m *Model) IsMultilingual() bool {
	if m == nil || m.bridge == nil {
		return false
	}
	return m.bridge.IsMultilingual()
}
