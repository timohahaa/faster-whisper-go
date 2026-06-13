package whisper

import (
	"math"
	"testing"
	"time"
)

func TestGetSpeechTimestamps_AllSilence(t *testing.T) {
	samples := make([]float32, 16000*5) // 5 seconds of silence
	chunks := GetSpeechTimestamps(samples, VadConfig{})
	if len(chunks) != 0 {
		t.Errorf("expected 0 speech chunks for silence, got %d", len(chunks))
	}
}

func TestGetSpeechTimestamps_AllSpeech(t *testing.T) {
	// Generate a 1kHz sine wave to simulate speech-like audio
	samples := makeSineWave(1000, 16000, 2.0) // 2 seconds of 1kHz tone
	chunks := GetSpeechTimestamps(samples, VadConfig{})
	// A continuous tone should produce at least one chunk
	if len(chunks) == 0 {
		t.Skip("VAD did not detect sine wave as speech (model-dependent)")
	}
	for _, c := range chunks {
		if c.End <= c.Start {
			t.Errorf("invalid chunk: start=%d end=%d", c.Start, c.End)
		}
	}
}

func TestGetSpeechTimestamps_SpeechWithSilence(t *testing.T) {
	sr := 16000
	// 1s speech + 3s silence + 1s speech
	speech1 := makeSineWave(440, sr, 1.0)
	silence := make([]float32, sr*3)
	speech2 := makeSineWave(440, sr, 1.0)

	samples := make([]float32, 0, len(speech1)+len(silence)+len(speech2))
	samples = append(samples, speech1...)
	samples = append(samples, silence...)
	samples = append(samples, speech2...)

	chunks := GetSpeechTimestamps(samples, VadConfig{
		MinSilenceDurationMs: 500,
	})
	// We expect the model to potentially detect the sine wave segments as speech
	// and the silence gap to cause a split. This is model-dependent.
	if len(chunks) == 0 {
		t.Skip("VAD did not detect sine waves as speech (model-dependent)")
	}
	for _, c := range chunks {
		if c.Start < 0 || c.End > len(samples) {
			t.Errorf("chunk out of bounds: start=%d end=%d len=%d", c.Start, c.End, len(samples))
		}
	}
}

func TestCollectChunks_Basic(t *testing.T) {
	samples := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	chunks := []SpeechChunk{
		{Start: 1, End: 3},
		{Start: 7, End: 9},
	}
	result := collectChunks(samples, chunks)
	expected := []float32{1, 2, 7, 8}
	if len(result) != len(expected) {
		t.Fatalf("expected len %d, got %d", len(expected), len(result))
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("at %d: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func TestCollectChunks_Empty(t *testing.T) {
	samples := []float32{1, 2, 3}
	result := collectChunks(samples, nil)
	if result != nil {
		t.Errorf("expected nil for empty chunks, got %v", result)
	}
}

func TestCollectChunks_FullRange(t *testing.T) {
	samples := []float32{0, 1, 2, 3, 4}
	chunks := []SpeechChunk{{Start: 0, End: 5}}
	result := collectChunks(samples, chunks)
	if len(result) != 5 {
		t.Fatalf("expected 5, got %d", len(result))
	}
}

func TestSpeechTimestampsMap_Simple(t *testing.T) {
	// Two speech chunks: [0, 16000) and [32000, 48000)
	// Silence gap: samples 16000-32000 (1 second of silence)
	chunks := []SpeechChunk{
		{Start: 0, End: 16000},
		{Start: 32000, End: 48000},
	}
	m := newSpeechTimestampsMap(chunks)

	// In the compressed audio, time 0.5s is in chunk 0 (no silence before)
	idx := m.getChunkIndex(0.5)
	original := m.getOriginalTime(0.5, idx)
	if math.Abs(original-0.5) > 0.01 {
		t.Errorf("expected ~0.5, got %f", original)
	}

	// In the compressed audio, time 1.5s is in chunk 1 (1s of silence was removed)
	idx = m.getChunkIndex(1.5)
	original = m.getOriginalTime(1.5, idx)
	// Should be 1.5 + 1.0 (silence) = 2.5 in original time
	if math.Abs(original-2.5) > 0.01 {
		t.Errorf("expected ~2.5, got %f", original)
	}
}

func TestSpeechTimestampsMap_RestoreSegment(t *testing.T) {
	chunks := []SpeechChunk{
		{Start: 0, End: 16000},     // 0-1s
		{Start: 48000, End: 64000}, // 3-4s in original (2s silence removed)
	}
	m := newSpeechTimestampsMap(chunks)

	seg := Segment{
		Start: time.Duration(1.2 * float64(time.Second)), // in compressed time, this is in chunk 1
		End:   time.Duration(1.8 * float64(time.Second)),
		Words: []Word{
			{
				Start: time.Duration(1.3 * float64(time.Second)),
				End:   time.Duration(1.5 * float64(time.Second)),
				Word:  "test",
			},
		},
	}

	m.restoreSegmentTimestamps(&seg)

	// Chunk 1 starts at compressed time 1.0s (16000 samples of chunk 0).
	// Total silence before chunk 1 = 48000 - 16000 = 32000 samples = 2.0s
	// So compressed 1.2s -> original 1.2 + 2.0 = 3.2s
	expectedStart := 3.2
	if math.Abs(seg.Start.Seconds()-expectedStart) > 0.05 {
		t.Errorf("segment start: expected ~%.1f, got %.3f", expectedStart, seg.Start.Seconds())
	}

	expectedEnd := 3.8
	if math.Abs(seg.End.Seconds()-expectedEnd) > 0.05 {
		t.Errorf("segment end: expected ~%.1f, got %.3f", expectedEnd, seg.End.Seconds())
	}

	if len(seg.Words) != 1 {
		t.Fatalf("expected 1 word, got %d", len(seg.Words))
	}
	expectedWordStart := 3.3
	if math.Abs(seg.Words[0].Start.Seconds()-expectedWordStart) > 0.05 {
		t.Errorf("word start: expected ~%.1f, got %.3f", expectedWordStart, seg.Words[0].Start.Seconds())
	}
}

func TestVadConfig_ApplyDefaults(t *testing.T) {
	cfg := VadConfig{}
	cfg.applyDefaults()

	if cfg.Threshold != 0.5 {
		t.Errorf("Threshold: expected 0.5, got %f", cfg.Threshold)
	}
	if cfg.NegThreshold != 0.35 {
		t.Errorf("NegThreshold: expected 0.35, got %f", cfg.NegThreshold)
	}
	if cfg.MinSpeechDurationMs != 0 {
		t.Errorf("MinSpeechDurationMs: expected 0, got %d", cfg.MinSpeechDurationMs)
	}
	if !math.IsInf(cfg.MaxSpeechDurationS, 1) {
		t.Errorf("MaxSpeechDurationS: expected +Inf, got %f", cfg.MaxSpeechDurationS)
	}
	if cfg.MinSilenceDurationMs != 2000 {
		t.Errorf("MinSilenceDurationMs: expected 2000, got %d", cfg.MinSilenceDurationMs)
	}
	if cfg.SpeechPadMs != 400 {
		t.Errorf("SpeechPadMs: expected 400, got %d", cfg.SpeechPadMs)
	}
	if cfg.MinSilenceAtMaxSpeech != 98 {
		t.Errorf("MinSilenceAtMaxSpeech: expected 98, got %d", cfg.MinSilenceAtMaxSpeech)
	}
	if cfg.UseMaxPossSilAtMaxSpeech == nil || !*cfg.UseMaxPossSilAtMaxSpeech {
		t.Error("UseMaxPossSilAtMaxSpeech: expected true")
	}
}

func TestGetSpeechProbs_ReturnsCorrectLength(t *testing.T) {
	// 1 second of audio = 16000 samples, should produce 16000/512 = 31.25 -> padded to 32 frames
	samples := make([]float32, 16000)
	probs := getSpeechProbs(samples)
	expectedFrames := (16000 + 511) / 512 // ceiling division
	if len(probs) != expectedFrames {
		t.Errorf("expected %d frames, got %d", expectedFrames, len(probs))
	}
	for _, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probability out of range: %f", p)
		}
	}
}

func makeSineWave(freqHz float64, sampleRate int, durationSec float64) []float32 {
	n := int(float64(sampleRate) * durationSec)
	samples := make([]float32, n)
	for i := range n {
		samples[i] = float32(math.Sin(2 * math.Pi * freqHz * float64(i) / float64(sampleRate)))
	}
	return samples
}
