package whisper

// ModelConfig controls model loading parameters
type ModelConfig struct {
	Device      string  // "cpu" or "cuda"
	ComputeType string  // "int8", "float16", "float32", "default"
	CacheDir    *string // override cache directory; nil = default ($XDG_CACHE_HOME/faster-whisper-go or ~/.cache/faster-whisper-go)
	Token       *string // Hugging Face token; nil = read from HF_TOKEN env
}

// DefaultModelConfig returns sensible defaults for CPU inference.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		Device:      "cpu",
		ComputeType: "default",
	}
}

// TranscribeConfig controls inference parameters
type TranscribeConfig struct {
	Language    string  // spoken language (e.g. "en", "ru"); empty = auto-detect
	BeamSize    int     // beam search width
	BestOf      int     // number of candidates when sampling (temperature > 0)
	Temperature float32 // sampling temperature
	Timestamps  bool    // extract per-segment timestamps
}

// DefaultTranscribeConfig returns sensible defaults for transcription.
func DefaultTranscribeConfig() TranscribeConfig {
	return TranscribeConfig{
		BeamSize:   5,
		BestOf:     1,
		Timestamps: true,
	}
}
