package whisper

import (
	"fmt"
	"os"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

// Model is a loaded Whisper model ready for transcription.
type Model struct {
	bridge     *ct2bridge.Model
	tokenizer  *tokenizer
	nMels      int
	melFilters []float32
}

// Load opens a Whisper model. The model argument can be:
//   - a local directory path (e.g. "/path/to/model" or "./models/large-v3")
//   - a short model name (e.g. "large-v3") mapped to a Hugging Face repo
//   - a full Hugging Face repo ID (e.g. "Systran/faster-whisper-large-v3")
//
// Remote models are downloaded and cached locally before loading.
func Load(model string, cfg ModelConfig) (*Model, error) {
	modelDir := model

	// Not a local directory — treat as a model name and download from Hugging Face.
	if info, err := os.Stat(model); err != nil || !info.IsDir() {
		repoID, err := resolveModelName(model)
		if err != nil {
			return nil, err
		}
		modelDir, err = downloadModel(repoID, cfg)
		if err != nil {
			return nil, fmt.Errorf("download model: %w", err)
		}
	}

	if err := validateModelDir(modelDir); err != nil {
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

	bridge, err := ct2bridge.Load(modelDir, device, computeType)
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

	return &Model{
		bridge:     bridge,
		tokenizer:  tok,
		nMels:      nMels,
		melFilters: computeMelFilterbank(nMels, whisperNFFT, whisperSampleRate),
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
