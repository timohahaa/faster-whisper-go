package whisper

import (
	"testing"
)

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
