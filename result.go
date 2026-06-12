package whisper

import "time"

// Result holds the output of a transcription.
type Result struct {
	Text     string
	Segments []Segment
	Language string
}

// Segment represents a transcribed segment with timing information.
type Segment struct {
	Start time.Duration
	End   time.Duration
	Text  string
}
