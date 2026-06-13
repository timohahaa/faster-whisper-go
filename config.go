package whisper

// ModelConfig controls model loading parameters.
type ModelConfig struct {
	Device      string // "cpu" or "cuda"
	ComputeType string // "int8", "float16", "float32", "default"
}

// DefaultModelConfig returns sensible defaults for CPU inference.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		Device:      "cpu",
		ComputeType: "default",
	}
}

// TranscribeConfig controls inference parameters.
// Zero-value / nil fields are filled with sensible defaults automatically,
// so callers can set only the fields they care about:
//
//	cfg := whisper.TranscribeConfig{Language: "en", WordTimestamps: true}
//
// DefaultTranscribeConfig() is still available for explicit construction.
type TranscribeConfig struct {
	Language string // spoken language (e.g. "en", "ru"); empty = auto-detect
	BeamSize int    // beam search width; 0 uses default (5)
	BestOf   int    // number of candidates when sampling (temperature > 0); 0 uses default (5)

	// WordTimestamps enables word-level timestamps via cross-attention alignment.
	WordTimestamps bool

	// DisableTimestamps tells the decoder to skip timestamp token generation entirely.
	// When true, segment Start/End will be zero and word timestamps are not extracted.
	DisableTimestamps bool

	// Temperature is the fallback chain of sampling temperatures.
	// When decode quality is below thresholds, the next temperature is tried.
	// nil uses the default chain [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
	Temperature []float32

	CompressionRatioThreshold *float32 // nil uses default (2.4); 0 disables the check
	LogProbThreshold          *float32 // nil uses default (-1.0)
	NoSpeechThreshold         *float32 // nil uses default (0.6); 0 disables the check

	// InitialPrompt is optional text to provide as context for the first window.
	InitialPrompt string
	// Hotwords are hint phrases for the model. Ignored if prefix is set.
	Hotwords string
	// ConditionOnPreviousText feeds the previous window output as prompt for the next.
	// nil uses default (true).
	ConditionOnPreviousText *bool
	// PromptResetOnTemperature resets the prompt context if the fallback temperature
	// exceeds this value. Only effective when ConditionOnPreviousText is true.
	// nil uses default (0.5); 0 means never reset.
	PromptResetOnTemperature *float32

	// SuppressBlank suppresses blank outputs at the beginning of sampling.
	// nil uses default (true).
	SuppressBlank *bool
	// SuppressTokens lists token IDs to suppress. [-1] expands to the default
	// non-speech symbol set (symbols, speaker tags, etc.). nil uses default ([-1]).
	SuppressTokens []int32

	// Patience controls beam search patience factor; nil uses default (1.0).
	Patience *float32
	// LengthPenalty exponential length penalty for beam search; nil uses default (1.0).
	LengthPenalty *float32
	// RepetitionPenalty penalizes repeated tokens; nil uses default (1.0).
	RepetitionPenalty *float32
	// NoRepeatNgramSize prevents n-gram repetitions of this size; 0 = disabled.
	NoRepeatNgramSize int

	// MaxInitialTimestamp is the latest allowed first timestamp; nil uses default (1.0).
	// 0 restricts the first timestamp to position 0.
	MaxInitialTimestamp *float32
	// MaxNewTokens limits tokens per chunk. nil uses the default (448).
	MaxNewTokens *int
	// Multilingual performs language detection on every segment.
	Multilingual bool
}

// applyDefaults fills nil/zero fields with sensible defaults.
// After this call every pointer field is guaranteed non-nil.
func (c *TranscribeConfig) applyDefaults() {
	if c.BeamSize == 0 {
		c.BeamSize = 5
	}
	if c.BestOf == 0 {
		c.BestOf = 5
	}
	if c.Temperature == nil {
		c.Temperature = []float32{0, 0.2, 0.4, 0.6, 0.8, 1.0}
	}
	if c.CompressionRatioThreshold == nil {
		c.CompressionRatioThreshold = ptrFloat32(2.4)
	}
	if c.LogProbThreshold == nil {
		c.LogProbThreshold = ptrFloat32(-1.0)
	}
	if c.NoSpeechThreshold == nil {
		c.NoSpeechThreshold = ptrFloat32(0.6)
	}
	if c.ConditionOnPreviousText == nil {
		c.ConditionOnPreviousText = ptrBool(true)
	}
	if c.PromptResetOnTemperature == nil {
		c.PromptResetOnTemperature = ptrFloat32(0.5)
	}
	if c.SuppressBlank == nil {
		c.SuppressBlank = ptrBool(true)
	}
	if c.SuppressTokens == nil {
		c.SuppressTokens = []int32{-1}
	}
	if c.Patience == nil {
		c.Patience = ptrFloat32(1.0)
	}
	if c.LengthPenalty == nil {
		c.LengthPenalty = ptrFloat32(1.0)
	}
	if c.RepetitionPenalty == nil {
		c.RepetitionPenalty = ptrFloat32(1.0)
	}
	if c.MaxInitialTimestamp == nil {
		c.MaxInitialTimestamp = ptrFloat32(1.0)
	}
}

// DefaultTranscribeConfig returns sensible defaults for transcription.
// Using this function is optional -- unset fields are filled automatically.
func DefaultTranscribeConfig() TranscribeConfig {
	return TranscribeConfig{
		BeamSize:                  5,
		BestOf:                    5,
		Temperature:               []float32{0, 0.2, 0.4, 0.6, 0.8, 1.0},
		CompressionRatioThreshold: ptrFloat32(2.4),
		LogProbThreshold:          ptrFloat32(-1.0),
		NoSpeechThreshold:         ptrFloat32(0.6),
		PromptResetOnTemperature:  ptrFloat32(0.5),
		SuppressBlank:             ptrBool(true),
		SuppressTokens:            []int32{-1},
		Patience:                  ptrFloat32(1.0),
		LengthPenalty:             ptrFloat32(1.0),
		RepetitionPenalty:         ptrFloat32(1.0),
		NoRepeatNgramSize:         0,
		MaxInitialTimestamp:       ptrFloat32(1.0),
		ConditionOnPreviousText:   ptrBool(true),
	}
}

func ptrFloat32(v float32) *float32 { return &v }
func ptrBool(v bool) *bool          { return &v }
