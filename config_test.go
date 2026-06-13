package whisper

import (
	"testing"
)

func TestApplyTranscribeDefaults(t *testing.T) {
	t.Run("FillsZeroFields", func(t *testing.T) {
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
	})

	t.Run("PreservesExplicitZero", func(t *testing.T) {
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
