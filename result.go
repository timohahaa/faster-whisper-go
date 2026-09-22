package whisper

import "time"

// Result holds the output of a transcription.
type Result struct {
	Text     string
	Segments []Segment
	Info     TranscriptionInfo
	Timings  Timings
}

// Timings is the per-phase wall-clock breakdown of one batched inference call.
// CPU phases: VAD, Mel. GPU phases: LangDetect, Encode, Decode, Align. Encode/
// Decode/Align are summed across all batches
type Timings struct {
	VAD        time.Duration // speech detection; 0 when ClipTimestamps is supplied
	Mel        time.Duration // STFT + mel filterbank across all chunks (CPU)
	LangDetect time.Duration // language-detection encode+detect; ~0 when Language is set
	Encode     time.Duration // Σ EncodeBatch over all batches (GPU)
	Decode     time.Duration // Σ GenerateBatch over all batches (GPU)
	Align      time.Duration // Σ AlignBatch over all batches (GPU); 0 without word timestamps
	Total      time.Duration // whole inferBatched call, wall-clock
	NumChunks  int           // number of VAD/clip chunks fed to the encoder
	NumBatches int           // number of encode/generate batches
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
