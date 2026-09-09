package silerovad

import (
	"sync"
	"testing"
)

// buildSpeechSignal returns silence + sine (speech-like) + silence at 16kHz.
func buildSpeechSignal() []float32 {
	pre := make([]float32, streamSampleRate/2)  // 0.5s silence
	speech := makeSine(220, streamSampleRate, 1.5)
	post := make([]float32, streamSampleRate) // 1.0s silence (> min_silence)
	out := make([]float32, 0, len(pre)+len(speech)+len(post))
	out = append(out, pre...)
	out = append(out, speech...)
	out = append(out, post...)
	return out
}

func countTypes(events []Event) (starts, ends int) {
	for _, e := range events {
		switch e.Type {
		case EventStartOfSpeech:
			starts++
		case EventEndOfSpeech:
			ends++
		}
	}
	return
}

func TestStreamSegmentation(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	s, err := v.NewStream(StreamConfig{})
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer s.Close()

	input := buildSpeechSignal()
	events, err := s.Push(input)
	if err != nil {
		t.Fatalf("Push: %v", err)
	}
	fev, err := s.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	events = append(events, fev...)

	starts, ends := countTypes(events)
	if starts == 0 {
		t.Skip("VAD did not detect sine as speech (model-dependent)")
	}
	if starts != ends {
		t.Errorf("unbalanced events: %d starts, %d ends", starts, ends)
	}
	for _, e := range events {
		if e.Type == EventEndOfSpeech {
			if len(e.Samples) == 0 {
				t.Error("EndOfSpeech has empty Samples")
			}
			if e.EndTime < e.StartTime {
				t.Errorf("EndTime %v < StartTime %v", e.EndTime, e.StartTime)
			}
		}
	}
}

// TestStreamFramingInvariance verifies that the event sequence is independent of
// how the input is chunked across Push calls. This is deterministic regardless
// of what the model predicts.
func TestStreamFramingInvariance(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	input := buildSpeechSignal()

	run := func(chunk int) []Event {
		s, err := v.NewStream(StreamConfig{})
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		defer s.Close()

		var all []Event
		for i := 0; i < len(input); i += chunk {
			end := i + chunk
			if end > len(input) {
				end = len(input)
			}
			ev, err := s.Push(input[i:end])
			if err != nil {
				t.Fatalf("Push: %v", err)
			}
			all = append(all, ev...)
		}
		fev, err := s.Flush()
		if err != nil {
			t.Fatalf("Flush: %v", err)
		}
		return append(all, fev...)
	}

	ref := run(windowSamples * 3) // window-aligned
	for _, chunk := range []int{100, 512, 999, 4096} {
		got := run(chunk)
		if len(got) != len(ref) {
			t.Fatalf("chunk=%d: %d events, want %d", chunk, len(got), len(ref))
		}
		for i := range ref {
			if got[i].Type != ref[i].Type ||
				len(got[i].Samples) != len(ref[i].Samples) ||
				got[i].StartTime != ref[i].StartTime ||
				got[i].EndTime != ref[i].EndTime {
				t.Errorf("chunk=%d event %d mismatch: got %+v want %+v",
					chunk, i, got[i], ref[i])
			}
		}
	}
}

// TestStreamFlushWhileSpeaking checks that Flush emits a pending EndOfSpeech.
func TestStreamFlushWhileSpeaking(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	s, err := v.NewStream(StreamConfig{})
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer s.Close()

	// Speech with no trailing silence, so the segment stays open until Flush.
	speech := makeSine(220, streamSampleRate, 1.5)
	events, err := s.Push(speech)
	if err != nil {
		t.Fatalf("Push: %v", err)
	}
	starts, ends := countTypes(events)
	if starts == 0 {
		t.Skip("VAD did not detect sine as speech (model-dependent)")
	}
	if ends != 0 {
		t.Fatalf("unexpected EndOfSpeech before Flush")
	}

	fev, err := s.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if len(fev) != 1 || fev[0].Type != EventEndOfSpeech {
		t.Fatalf("Flush did not emit EndOfSpeech: %+v", fev)
	}
	if len(fev[0].Samples) == 0 {
		t.Error("flushed segment has empty Samples")
	}

	// Second flush is a no-op.
	fev2, err := s.Flush()
	if err != nil {
		t.Fatalf("Flush(2): %v", err)
	}
	if len(fev2) != 0 {
		t.Errorf("expected no events on second Flush, got %d", len(fev2))
	}
}

// TestStreamConcurrentStreams runs many streams over one shared VAD session in
// parallel to exercise the shared session + mutex under -race.
func TestStreamConcurrentStreams(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	input := buildSpeechSignal()

	const streams = 8
	var wg sync.WaitGroup
	wg.Add(streams)
	for g := 0; g < streams; g++ {
		go func() {
			defer wg.Done()
			s, err := v.NewStream(StreamConfig{})
			if err != nil {
				t.Errorf("NewStream: %v", err)
				return
			}
			defer s.Close()
			// Push in small chunks to interleave inference calls.
			for i := 0; i < len(input); i += 320 {
				end := i + 320
				if end > len(input) {
					end = len(input)
				}
				if _, err := s.Push(input[i:end]); err != nil {
					t.Errorf("Push: %v", err)
					return
				}
			}
			if _, err := s.Flush(); err != nil {
				t.Errorf("Flush: %v", err)
			}
		}()
	}
	wg.Wait()
}
