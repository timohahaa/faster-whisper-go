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
// Use DefaultTranscribeConfig() as a starting point and override the fields you need.
// Zero values are NOT treated as "use default" — they mean exactly zero.
type TranscribeConfig struct {
	Language string // spoken language (e.g. "en", "ru"); empty = auto-detect
	BeamSize int    // beam search width; default 5
	BestOf   int    // number of candidates when sampling (temperature > 0); default 5

	// WordTimestamps enables word-level timestamps via cross-attention alignment.
	// Segment-level timestamps are always produced.
	WordTimestamps bool

	// Temperature is the fallback chain of sampling temperatures.
	// When decode quality is below thresholds, the next temperature is tried.
	// nil uses the default chain [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
	Temperature []float32

	CompressionRatioThreshold float32   // default 2.4
	LogProbThreshold          *float32 // default -1.0; nil disables the check
	NoSpeechThreshold         float32  // default 0.6

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

	// Patience controls beam search patience factor; default 1.0.
	Patience float32
	// LengthPenalty exponential length penalty for beam search; default 1.0.
	LengthPenalty float32
	// RepetitionPenalty penalizes repeated tokens; default 1.0.
	RepetitionPenalty float32
	// NoRepeatNgramSize prevents n-gram repetitions of this size; 0 = disabled.
	NoRepeatNgramSize int

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
		Temperature:               []float32{0, 0.2, 0.4, 0.6, 0.8, 1.0},
		CompressionRatioThreshold: 2.4,
		LogProbThreshold:          ptrFloat32(-1.0),
		NoSpeechThreshold:         0.6,
		PromptResetOnTemperature:  0.5,
		SuppressBlank:             true,
		SuppressTokens:            []int32{-1},
		Patience:                  1.0,
		LengthPenalty:             1.0,
		RepetitionPenalty:         1.0,
		NoRepeatNgramSize:         0,
		MaxInitialTimestamp:       1.0,
		ConditionOnPreviousText:   true,
	}
}

func ptrFloat32(v float32) *float32 { return &v }
