package whisper

// ModelConfig controls model loading parameters
type ModelConfig struct {
	Device      string // "cpu" or "cuda"
	ComputeType string // "int8", "float16", "float32", "default"
}

// DefaultModelConfig provides sensible defaults for CPU inference
var DefaultModelConfig = ModelConfig{
	Device:      "cpu",
	ComputeType: "default",
}

// TranscribeConfig controls inference parameters
type TranscribeConfig struct {
	Language    string  // spoken language (e.g. "en", "ru"); empty = auto-detect
	BeamSize    int     // beam search width
	BestOf      int     // number of candidates when sampling (temperature > 0)
	Temperature float32 // sampling temperature
	Timestamps  bool    // extract per-segment timestamps
}

// DefaultTranscribeConfig provides sensible defaults for transcription.
var DefaultTranscribeConfig = TranscribeConfig{
	BeamSize:   5,
	BestOf:     1,
	Timestamps: true,
}
