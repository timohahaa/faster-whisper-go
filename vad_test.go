package whisper

import (
	"math"
	"testing"
	"time"

	"github.com/timohahaa/faster-whisper-go/internal/silerovad"
)

func TestGetSpeechTimestamps(t *testing.T) {
	t.Run("AllSilence", func(t *testing.T) {
		samples := make([]float32, 16000*5)
		chunks, err := GetSpeechTimestamps(samples, VadConfig{})
		if err != nil {
			t.Fatalf("GetSpeechTimestamps: %v", err)
		}
		if len(chunks) != 0 {
			t.Errorf("expected 0 speech chunks for silence, got %d", len(chunks))
		}
	})

	t.Run("AllSpeech", func(t *testing.T) {
		samples := makeSineWave(1000, 16000, 2.0)
		chunks, err := GetSpeechTimestamps(samples, VadConfig{})
		if err != nil {
			t.Fatalf("GetSpeechTimestamps: %v", err)
		}
		if len(chunks) == 0 {
			t.Skip("VAD did not detect sine wave as speech (model-dependent)")
		}
		for _, c := range chunks {
			if c.End <= c.Start {
				t.Errorf("invalid chunk: start=%d end=%d", c.Start, c.End)
			}
		}
	})

	t.Run("SpeechWithSilence", func(t *testing.T) {
		sr := 16000
		speech1 := makeSineWave(440, sr, 1.0)
		silence := make([]float32, sr*3)
		speech2 := makeSineWave(440, sr, 1.0)

		samples := make([]float32, 0, len(speech1)+len(silence)+len(speech2))
		samples = append(samples, speech1...)
		samples = append(samples, silence...)
		samples = append(samples, speech2...)

		chunks, err := GetSpeechTimestamps(samples, VadConfig{
			MinSilenceDurationMs: 500,
		})
		if err != nil {
			t.Fatalf("GetSpeechTimestamps: %v", err)
		}
		if len(chunks) == 0 {
			t.Skip("VAD did not detect sine waves as speech (model-dependent)")
		}
		for _, c := range chunks {
			if c.Start < 0 || c.End > len(samples) {
				t.Errorf("chunk out of bounds: start=%d end=%d len=%d", c.Start, c.End, len(samples))
			}
		}
	})
}

func TestCollectChunks(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
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
	})

	t.Run("Empty", func(t *testing.T) {
		samples := []float32{1, 2, 3}
		result := collectChunks(samples, nil)
		if result != nil {
			t.Errorf("expected nil for empty chunks, got %v", result)
		}
	})

	t.Run("FullRange", func(t *testing.T) {
		samples := []float32{0, 1, 2, 3, 4}
		chunks := []SpeechChunk{{Start: 0, End: 5}}
		result := collectChunks(samples, chunks)
		if len(result) != 5 {
			t.Fatalf("expected 5, got %d", len(result))
		}
	})
}

func TestSpeechTimestampsMap(t *testing.T) {
	t.Run("Simple", func(t *testing.T) {
		chunks := []SpeechChunk{
			{Start: 0, End: 16000},
			{Start: 32000, End: 48000},
		}
		m := newSpeechTimestampsMap(chunks)

		idx := m.chunkIndex(0.5)
		original := m.originalTime(0.5, idx)
		if math.Abs(original-0.5) > 0.01 {
			t.Errorf("expected ~0.5, got %f", original)
		}

		idx = m.chunkIndex(1.5)
		original = m.originalTime(1.5, idx)
		if math.Abs(original-2.5) > 0.01 {
			t.Errorf("expected ~2.5, got %f", original)
		}
	})

	t.Run("RestoreSegment", func(t *testing.T) {
		chunks := []SpeechChunk{
			{Start: 0, End: 16000},
			{Start: 48000, End: 64000},
		}
		m := newSpeechTimestampsMap(chunks)

		seg := Segment{
			Start: time.Duration(1.2 * float64(time.Second)),
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
	})
}

func TestApplyVadDefaults(t *testing.T) {
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
	if cfg.UseMaxPossSilAtMaxSpeech {
		t.Error("UseMaxPossSilAtMaxSpeech: expected false")
	}
}

func TestGetSpeechProbs(t *testing.T) {
	samples := make([]float32, 16000)
	probs, err := silerovad.Probs(samples)
	if err != nil {
		t.Fatalf("silerovad.Probs: %v", err)
	}
	expectedFrames := (16000 + 511) / 512
	if len(probs) != expectedFrames {
		t.Errorf("expected %d frames, got %d", expectedFrames, len(probs))
	}
	for _, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probability out of range: %f", p)
		}
	}
}

func BenchmarkGetSpeechProbs(b *testing.B) {
	b.Run("5s", func(b *testing.B) {
		samples := makeSineWave(440, whisperSampleRate, 5.0)
		b.ResetTimer()
		for b.Loop() {
			_, _ = silerovad.Probs(samples)
		}
	})
}
