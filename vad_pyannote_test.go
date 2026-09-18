package whisper

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/timohahaa/faster-whisper-go/pyannotevad"
)

// TestBinarizeParity feeds the Python reference speech-activity curve into the
// Go binarizer and checks that the resulting speech regions match the Python
// pyannote BinarizeVadScores output (onset=0.5, offset=0.363,
// max_duration=30, no padding). Skipped unless fixtures are provided:
//
//	PYANNOTE_PARITY_SCORES   raw little-endian float32 reference curve
//	PYANNOTE_PARITY_REGIONS  ref_out.json produced by tools/export_pyannote_vad.py
//	PYANNOTE_PARITY_CLIP     raw float32 samples (used only for its length)
func TestBinarizeParity(t *testing.T) {
	scoresPath := os.Getenv("PYANNOTE_PARITY_SCORES")
	regionsPath := os.Getenv("PYANNOTE_PARITY_REGIONS")
	clipPath := os.Getenv("PYANNOTE_PARITY_CLIP")
	if scoresPath == "" || regionsPath == "" || clipPath == "" {
		t.Skip("set PYANNOTE_PARITY_SCORES, PYANNOTE_PARITY_REGIONS and PYANNOTE_PARITY_CLIP to run")
	}

	scores := readRawF32(t, scoresPath)
	nSamples := len(readRawF32(t, clipPath))

	var ref struct {
		Regions [][2]float64 `json:"regions"`
	}
	raw, err := os.ReadFile(regionsPath)
	if err != nil {
		t.Fatalf("read regions: %v", err)
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatalf("parse regions: %v", err)
	}

	got := binarizeVadScores(scores, pyannotevad.FrameStep, 0.5, 0.363, 30.0, nSamples)

	if len(got) != len(ref.Regions) {
		t.Fatalf("region count mismatch: got %d ref %d\ngot=%v\nref=%v",
			len(got), len(ref.Regions), toSec(got), ref.Regions)
	}

	const tol = 0.03 // ~2 frames
	var maxDiff float64
	for i, g := range got {
		gs := float64(g.Start) / whisperSampleRate
		ge := float64(g.End) / whisperSampleRate
		ds := math.Abs(gs - ref.Regions[i][0])
		de := math.Abs(ge - ref.Regions[i][1])
		maxDiff = math.Max(maxDiff, math.Max(ds, de))
		if ds > tol || de > tol {
			t.Errorf("region %d mismatch: got [%.3f,%.3f] ref [%.3f,%.3f]",
				i, gs, ge, ref.Regions[i][0], ref.Regions[i][1])
		}
	}
	t.Logf("regions=%d maxBoundaryDiff=%.4fs", len(got), maxDiff)
}

func toSec(cs []SpeechChunk) [][2]float64 {
	out := make([][2]float64, len(cs))
	for i, c := range cs {
		out[i] = [2]float64{float64(c.Start) / whisperSampleRate, float64(c.End) / whisperSampleRate}
	}
	return out
}

func readRawF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}
