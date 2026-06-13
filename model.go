package whisper

import (
	"fmt"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

// Model is a loaded Whisper model ready for transcription.
type Model struct {
	bridge     *ct2bridge.Model
	tokenizer  *tokenizer
	nMels      int
	melFilters []float32
}

// Load opens a Whisper model from a CTranslate2 model directory.
func Load(modelDir string, cfg ModelConfig) (*Model, error) {
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
		nMels = 80
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
