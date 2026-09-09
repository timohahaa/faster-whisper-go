package silerovad

import (
	"math"
	"sync"
	"testing"
)

// makeSine builds a mono 16kHz float32 sine wave of the given frequency/seconds.
func makeSine(freq float64, sampleRate int, seconds float64) []float32 {
	n := int(float64(sampleRate) * seconds)
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / float64(sampleRate)))
	}
	return out
}

func floatsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestVADResetsStatePerCall verifies that Probs resets LSTM state per call, so
// repeated calls on the same instance are deterministic.
func TestVADResetsStatePerCall(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	samples := makeSine(440, 16000, 2.0)

	first, err := v.Probs(samples)
	if err != nil {
		t.Fatalf("Probs: %v", err)
	}
	for i := 0; i < 5; i++ {
		got, err := v.Probs(samples)
		if err != nil {
			t.Fatalf("Probs: %v", err)
		}
		if !floatsEqual(first, got) {
			t.Fatalf("non-deterministic Probs on call %d", i)
		}
	}
}

// TestVADConcurrentSameInstance runs many goroutines against a single VAD to
// exercise the internal mutex under -race.
func TestVADConcurrentSameInstance(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	samples := makeSine(440, 16000, 1.0)
	want, err := v.Probs(samples)
	if err != nil {
		t.Fatalf("Probs: %v", err)
	}

	const goroutines = 16
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				got, err := v.Probs(samples)
				if err != nil {
					t.Errorf("Probs: %v", err)
					return
				}
				if !floatsEqual(want, got) {
					t.Errorf("concurrent Probs mismatch")
					return
				}
			}
		}()
	}
	wg.Wait()
}

// TestVADConcurrentIndependentInstances runs multiple independent VAD instances
// in parallel to ensure they do not share mutable state.
func TestVADConcurrentIndependentInstances(t *testing.T) {
	samples := makeSine(440, 16000, 1.0)

	// Reference probabilities from a throwaway instance.
	ref, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want, err := ref.Probs(samples)
	ref.Close()
	if err != nil {
		t.Fatalf("Probs: %v", err)
	}

	const instances = 8
	var wg sync.WaitGroup
	wg.Add(instances)
	for g := 0; g < instances; g++ {
		go func() {
			defer wg.Done()
			v, err := New()
			if err != nil {
				t.Errorf("New: %v", err)
				return
			}
			defer v.Close()
			for i := 0; i < 10; i++ {
				got, err := v.Probs(samples)
				if err != nil {
					t.Errorf("Probs: %v", err)
					return
				}
				if !floatsEqual(want, got) {
					t.Errorf("independent instance Probs mismatch")
					return
				}
			}
		}()
	}
	wg.Wait()
}
