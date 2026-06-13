package whisper

import "math"

// VadConfig controls Voice Activity Detection parameters.
// Zero-value fields are filled with sensible defaults
type VadConfig struct {
	// Threshold is the speech probability threshold. Probabilities above this
	// value are considered speech. Default: 0.5.
	Threshold float32
	// NegThreshold is the silence threshold for end-of-speech detection.
	// 0 means auto-compute as max(Threshold-0.15, 0.01).
	NegThreshold float32
	// MinSpeechDurationMs: speech chunks shorter than this are discarded. Default: 0.
	MinSpeechDurationMs int
	// MaxSpeechDurationS: chunks longer than this are split at the best silence point.
	// Default: +Inf (no limit).
	MaxSpeechDurationS float64
	// MinSilenceDurationMs: minimum silence duration to split speech chunks. Default: 2000.
	MinSilenceDurationMs int
	// SpeechPadMs: padding added to each side of a speech chunk. Default: 400.
	SpeechPadMs int
	// MinSilenceAtMaxSpeech: minimum silence (ms) used to find split points when
	// MaxSpeechDurationS is reached. Default: 98.
	MinSilenceAtMaxSpeech int
	// UseMaxPossSilAtMaxSpeech: when true, split at the longest silence found
	// (not just the last one). Default: true.
	UseMaxPossSilAtMaxSpeech *bool
}

func (c *VadConfig) applyDefaults() {
	if c.Threshold == 0 {
		c.Threshold = 0.5
	}
	if c.NegThreshold == 0 {
		c.NegThreshold = float32(math.Max(float64(c.Threshold)-0.15, 0.01))
	}
	if c.MaxSpeechDurationS == 0 {
		c.MaxSpeechDurationS = math.Inf(1)
	}
	if c.MinSilenceDurationMs == 0 {
		c.MinSilenceDurationMs = 2000
	}
	if c.SpeechPadMs == 0 {
		c.SpeechPadMs = 400
	}
	if c.MinSilenceAtMaxSpeech == 0 {
		c.MinSilenceAtMaxSpeech = 98
	}
	if c.UseMaxPossSilAtMaxSpeech == nil {
		c.UseMaxPossSilAtMaxSpeech = ptrBool(true)
	}
}

// ModelConfig controls model loading parameters.
type ModelConfig struct {
	Device      string // "cpu" or "cuda"
	ComputeType string // "int8", "float16", "float32", "default"
	DeviceIndex []int  // GPU device IDs to use; nil/empty defaults to [0]
	CPUThreads  int    // threads per replica (intra_threads); 0 = CTranslate2 default (4)
	NumWorkers  int    // number of replicas (inter_threads); 0 = 1

	// CacheDir is the directory for caching downloaded models.
	// Empty uses $XDG_CACHE_HOME/faster-whisper-go/ or ~/.cache/faster-whisper-go/.
	CacheDir string
	// LocalFilesOnly disables network downloads; only cached models are used.
	LocalFilesOnly bool
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
	// Prefix is optional text to provide as a prefix at the beginning of the
	// first window. When set, Hotwords are ignored.
	Prefix string
	// Hotwords are hint phrases for the model. Ignored if Prefix is set.
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

	// HallucinationSilenceThreshold skips silent periods longer than this (in seconds)
	// when a possible hallucination is detected during word timestamp extraction.
	// nil means disabled. Only effective when WordTimestamps is true.
	HallucinationSilenceThreshold *float32

	// PrependPunctuations lists punctuation symbols that should be merged with
	// the following word during word timestamp extraction.
	// Empty uses default: "\"'"¿([{-"
	PrependPunctuations string
	// AppendPunctuations lists punctuation symbols that should be merged with
	// the preceding word during word timestamp extraction.
	// Empty uses default: "\"'.。,，!！?？:：")]}、"
	AppendPunctuations string

	// VadFilter enables Voice Activity Detection to filter out silence before
	// transcription. Reduces processing time and can improve quality.
	VadFilter bool
	// VadConfig controls VAD parameters. nil uses defaults.
	VadConfig *VadConfig
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
	if c.PrependPunctuations == "" {
		c.PrependPunctuations = "\"\u2018\u201c\u00bf([{-"
	}
	if c.AppendPunctuations == "" {
		c.AppendPunctuations = "\"\u2019.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]\u007d\u3001"
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
