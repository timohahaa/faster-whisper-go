package whisper

import "time"

// Result holds the output of a transcription.
type Result struct {
	Text     string
	Segments []Segment
	Info     TranscriptionInfo
}

// Segment represents a transcribed segment with timing and quality information.
type Segment struct {
	ID               int
	Start            time.Duration
	End              time.Duration
	Text             string
	Tokens           []int32
	Temperature      float32
	AvgLogProb       float32
	CompressionRatio float32
	NoSpeechProb     float32
}

// TranscriptionInfo holds metadata about the transcription.
type TranscriptionInfo struct {
	Language            string
	LanguageProbability float32
	Duration            time.Duration
	AllLanguageProbs    []LanguageProb
}

// LanguageProb pairs a language code with its detection probability.
type LanguageProb struct {
	Language    string
	Probability float32
}

// LanguageDetection holds the result of language identification.
type LanguageDetection struct {
	Language    string
	Probability float32
}
