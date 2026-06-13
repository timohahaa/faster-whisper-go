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
	Words            []Word
	Temperature      float32
	AvgLogProb       float32
	CompressionRatio float32
	NoSpeechProb     float32
}

// Word represents a single word with timing information, available when
// WordTimestamps is enabled in TranscribeConfig.
type Word struct {
	Start       time.Duration
	End         time.Duration
	Word        string
	Probability float32
}

// TranscriptionInfo holds metadata about the transcription.
type TranscriptionInfo struct {
	Language            string
	LanguageProbability float32
	Duration            time.Duration
	DurationAfterVad    time.Duration // duration of audio after VAD filtering (equals Duration if VAD is off)
}

// LanguageDetection holds the result of language identification.
type LanguageDetection struct {
	Language    string
	Probability float32
}
