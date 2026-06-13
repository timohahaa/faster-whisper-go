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

func TestSecToDuration(t *testing.T) {
	d := secToDuration(1.5)
	if d.Seconds() != 1.5 {
		t.Errorf("secToDuration(1.5) = %v, want 1.5s", d)
	}
}
