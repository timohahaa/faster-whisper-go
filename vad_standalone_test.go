package whisper

import "testing"

// TestMergeSpeechChunksParityWithCollectBatched guards the single-source-of-truth
// invariant: the boundaries mergeSpeechChunks produces (used by the standalone
// BatchedSpeechChunks) must match exactly the windows collectChunksBatched builds
// internally, so externally-computed ClipTimestamps stay identical to the
// internal batched path.
func TestMergeSpeechChunksParityWithCollectBatched(t *testing.T) {
	const maxDur = 30.0
	maxSamples := int(maxDur * whisperSampleRate)
	samples := make([]float32, 40*whisperSampleRate) // large enough that no index clamps

	// Regions chosen so the 30s span forces a window cut before the third region.
	chunks := []SpeechChunk{
		{Start: 0, End: 5 * whisperSampleRate},
		{Start: 6 * whisperSampleRate, End: 10 * whisperSampleRate},
		{Start: 12 * whisperSampleRate, End: 31 * whisperSampleRate},
		{Start: 32 * whisperSampleRate, End: 38 * whisperSampleRate},
	}

	merged := mergeSpeechChunks(chunks, maxSamples)
	audioChunks, meta := collectChunksBatched(samples, chunks, maxDur)

	if len(merged) != len(meta) || len(merged) != len(audioChunks) {
		t.Fatalf("length mismatch: merged=%d meta=%d audio=%d", len(merged), len(meta), len(audioChunks))
	}

	for i, m := range merged {
		if wantOffset := float64(m.Start) / whisperSampleRate; meta[i].offset != wantOffset {
			t.Errorf("chunk %d offset: merged %.6f, collectBatched %.6f", i, wantOffset, meta[i].offset)
		}
		if wantDur := float64(m.End-m.Start) / whisperSampleRate; meta[i].duration != wantDur {
			t.Errorf("chunk %d duration: merged %.6f, collectBatched %.6f", i, wantDur, meta[i].duration)
		}
		start, end := clampRange(m.Start, m.End, len(samples))
		if end < start {
			end = start
		}
		if len(audioChunks[i]) != end-start {
			t.Errorf("chunk %d audio len: got %d, want %d", i, len(audioChunks[i]), end-start)
		}
	}
}

func TestMergeSpeechChunksEmpty(t *testing.T) {
	if got := mergeSpeechChunks(nil, 100); got != nil {
		t.Errorf("expected nil for no chunks, got %v", got)
	}
}
