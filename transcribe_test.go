package whisper

import (
	"testing"
)

func TestPadOrTrim(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		length int
		want   []float32
	}{
		{"Shorter", []float32{1, 2, 3}, 5, []float32{1, 2, 3, 0, 0}},
		{"Exact", []float32{1, 2, 3}, 3, []float32{1, 2, 3}},
		{"Longer", []float32{1, 2, 3, 4, 5}, 3, []float32{1, 2, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := padOrTrim(tt.input, tt.length)
			if len(out) != len(tt.want) {
				t.Fatalf("len = %d, want %d", len(out), len(tt.want))
			}
			for i := range tt.want {
				if out[i] != tt.want[i] {
					t.Errorf("out[%d] = %f, want %f", i, out[i], tt.want[i])
				}
			}
		})
	}

	t.Run("NoAlloc", func(t *testing.T) {
		samples := []float32{1, 2, 3, 4, 5}

		out := padOrTrim(samples, 5)
		if &out[0] != &samples[0] {
			t.Error("exact length should return original slice")
		}

		out = padOrTrim(samples, 3)
		if &out[0] != &samples[0] {
			t.Error("trim should return sub-slice of original")
		}
	})
}

func TestDefaultTranscribeConfig(t *testing.T) {
	cfg := DefaultTranscribeConfig()

	if cfg.BeamSize != 5 {
		t.Errorf("BeamSize: got %d, want 5", cfg.BeamSize)
	}
	if cfg.BestOf != 5 {
		t.Errorf("BestOf: got %d, want 5", cfg.BestOf)
	}
	if len(cfg.Temperature) != 6 {
		t.Errorf("Temperature: got %d entries, want 6", len(cfg.Temperature))
	}
	if cfg.CompressionRatioThreshold == nil || *cfg.CompressionRatioThreshold != 2.4 {
		t.Errorf("CompressionRatioThreshold: got %v, want 2.4", cfg.CompressionRatioThreshold)
	}
	if cfg.LogProbThreshold == nil || *cfg.LogProbThreshold != -1.0 {
		t.Errorf("LogProbThreshold: got %v, want -1.0", cfg.LogProbThreshold)
	}
	if cfg.NoSpeechThreshold == nil || *cfg.NoSpeechThreshold != 0.6 {
		t.Errorf("NoSpeechThreshold: got %v, want 0.6", cfg.NoSpeechThreshold)
	}
	if cfg.PromptResetOnTemperature == nil || *cfg.PromptResetOnTemperature != 0.5 {
		t.Errorf("PromptResetOnTemperature: got %v, want 0.5", cfg.PromptResetOnTemperature)
	}
	if cfg.MaxInitialTimestamp == nil || *cfg.MaxInitialTimestamp != 1.0 {
		t.Errorf("MaxInitialTimestamp: got %v, want 1.0", cfg.MaxInitialTimestamp)
	}
	if len(cfg.SuppressTokens) != 1 || cfg.SuppressTokens[0] != -1 {
		t.Errorf("SuppressTokens: got %v, want [-1]", cfg.SuppressTokens)
	}
	if cfg.Patience == nil || *cfg.Patience != 1.0 {
		t.Errorf("Patience: got %v, want 1.0", cfg.Patience)
	}
	if cfg.LengthPenalty == nil || *cfg.LengthPenalty != 1.0 {
		t.Errorf("LengthPenalty: got %v, want 1.0", cfg.LengthPenalty)
	}
	if cfg.RepetitionPenalty == nil || *cfg.RepetitionPenalty != 1.0 {
		t.Errorf("RepetitionPenalty: got %v, want 1.0", cfg.RepetitionPenalty)
	}
	if cfg.NoRepeatNgramSize != 0 {
		t.Errorf("NoRepeatNgramSize: got %d, want 0", cfg.NoRepeatNgramSize)
	}
	if cfg.SuppressBlank == nil || !*cfg.SuppressBlank {
		t.Error("SuppressBlank: got nil or false, want true")
	}
	if cfg.ConditionOnPreviousText == nil || !*cfg.ConditionOnPreviousText {
		t.Error("ConditionOnPreviousText: got nil or false, want true")
	}
}

func TestApplyDefaults(t *testing.T) {
	cfg := TranscribeConfig{Language: "en"}
	cfg.applyDefaults()

	if cfg.BeamSize != 5 {
		t.Errorf("BeamSize: got %d, want 5", cfg.BeamSize)
	}
	if cfg.BestOf != 5 {
		t.Errorf("BestOf: got %d, want 5", cfg.BestOf)
	}
	if cfg.CompressionRatioThreshold == nil || *cfg.CompressionRatioThreshold != 2.4 {
		t.Errorf("CompressionRatioThreshold: got %v, want 2.4", cfg.CompressionRatioThreshold)
	}
	if cfg.SuppressBlank == nil || !*cfg.SuppressBlank {
		t.Error("SuppressBlank: got nil or false, want true")
	}
	if cfg.ConditionOnPreviousText == nil || !*cfg.ConditionOnPreviousText {
		t.Error("ConditionOnPreviousText: got nil or false, want true")
	}

	t.Run("PreserveExplicitZero", func(t *testing.T) {
		zero := float32(0)
		cfg := TranscribeConfig{
			CompressionRatioThreshold: &zero,
			SuppressBlank:             ptrBool(false),
		}
		cfg.applyDefaults()

		if *cfg.CompressionRatioThreshold != 0 {
			t.Errorf("CompressionRatioThreshold: got %f, want 0 (explicitly set)", *cfg.CompressionRatioThreshold)
		}
		if *cfg.SuppressBlank {
			t.Error("SuppressBlank: got true, want false (explicitly set)")
		}
	})
}

func TestSecToDuration(t *testing.T) {
	d := secToDuration(1.5)
	if d.Seconds() != 1.5 {
		t.Errorf("secToDuration(1.5) = %v, want 1.5s", d)
	}
}
