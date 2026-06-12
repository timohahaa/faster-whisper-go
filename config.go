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
type TranscribeConfig struct {
	Language string // spoken language (e.g. "en", "ru"); empty = auto-detect
	BeamSize int    // beam search width; default 5
	BestOf   int    // number of candidates when sampling (temperature > 0); default 5

	// Timestamps controls whether segment-level timestamps are extracted.
	Timestamps bool

	// Temperature is the fallback chain of sampling temperatures.
	// When decode quality is below thresholds, the next temperature is tried.
	// nil uses the default chain [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
	Temperature []float32

	// Quality thresholds that trigger temperature fallback.
	// Zero values disable the respective check.
	CompressionRatioThreshold float32 // default 2.4
	LogProbThreshold          float32 // default -1.0
	NoSpeechThreshold         float32 // default 0.6

	// InitialPrompt is optional text to provide as context for the first window.
	InitialPrompt string
	// Hotwords are hint phrases for the model. Ignored if prefix is set.
	Hotwords string
	// ConditionOnPreviousText feeds the previous window output as prompt for the next.
	ConditionOnPreviousText bool
	// PromptResetOnTemperature resets the prompt context if the fallback temperature
	// exceeds this value. Only effective when ConditionOnPreviousText is true.
	PromptResetOnTemperature float32 // default 0.5

	// SuppressBlank suppresses blank outputs at the beginning of sampling.
	SuppressBlank bool
	// SuppressTokens lists token IDs to suppress. [-1] expands to the default
	// non-speech symbol set (symbols, speaker tags, etc.).
	SuppressTokens []int32

	// WithoutTimestamps generates only text tokens (no timestamp tokens).
	WithoutTimestamps bool
	// MaxInitialTimestamp is the latest allowed first timestamp; default 1.0.
	MaxInitialTimestamp float32
	// MaxNewTokens limits tokens per chunk. nil uses the default (448).
	MaxNewTokens *int
	// Multilingual performs language detection on every segment.
	Multilingual bool
}

// DefaultTranscribeConfig returns sensible defaults for transcription.
func DefaultTranscribeConfig() TranscribeConfig {
	return TranscribeConfig{
		BeamSize:                  5,
		BestOf:                    5,
		Timestamps:                true,
		Temperature:               []float32{0, 0.2, 0.4, 0.6, 0.8, 1.0},
		CompressionRatioThreshold: 2.4,
		LogProbThreshold:          -1.0,
		NoSpeechThreshold:         0.6,
		PromptResetOnTemperature:  0.5,
		SuppressBlank:             true,
		SuppressTokens:            []int32{-1},
		MaxInitialTimestamp:       1.0,
		ConditionOnPreviousText:   true,
	}
}
