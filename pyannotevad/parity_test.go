package pyannotevad

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// TestScoresParity compares the Go pyannote aggregated speech-activity curve
// against a reference curve produced by the Python pyannote pipeline on the
// same audio. It is skipped unless both fixtures are provided via env:
//
//	PYANNOTE_PARITY_CLIP    raw little-endian float32 mono 16kHz samples
//	PYANNOTE_PARITY_SCORES  raw little-endian float32 reference curve
//
// Generate fixtures with tools/export_pyannote_vad.py (see repo docs).
func TestScoresParity(t *testing.T) {
	clipPath := os.Getenv("PYANNOTE_PARITY_CLIP")
	scoresPath := os.Getenv("PYANNOTE_PARITY_SCORES")
	if clipPath == "" || scoresPath == "" {
		t.Skip("set PYANNOTE_PARITY_CLIP and PYANNOTE_PARITY_SCORES to run parity test")
	}

	samples := readF32(t, clipPath)
	ref := readF32(t, scoresPath)

	vad, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer vad.Close()

	got, err := vad.Scores(samples)
	if err != nil {
		t.Fatalf("Scores: %v", err)
	}

	if len(got) != len(ref) {
		t.Fatalf("length mismatch: got %d ref %d", len(got), len(ref))
	}

	var maxAbs, sumAbs float64
	var disagree int
	for i := range got {
		d := math.Abs(float64(got[i] - ref[i]))
		sumAbs += d
		if d > maxAbs {
			maxAbs = d
		}
		if (got[i] > 0.5) != (ref[i] > 0.5) {
			disagree++
		}
	}
	meanAbs := sumAbs / float64(len(got))
	disagreeFrac := float64(disagree) / float64(len(got))

	t.Logf("frames=%d maxAbs=%.6g meanAbs=%.6g onsetDisagree=%d (%.4f%%)",
		len(got), maxAbs, meanAbs, disagree, disagreeFrac*100)

	if meanAbs > 1e-3 {
		t.Errorf("mean abs diff too high: %.6g", meanAbs)
	}
	if maxAbs > 5e-2 {
		t.Errorf("max abs diff too high: %.6g", maxAbs)
	}
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("%s: length %d not a multiple of 4", path, len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}
