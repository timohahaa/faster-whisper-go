package whisper

import (
	"strings"
	"time"
)

func padOrTrim(samples []float32, length int) []float32 {
	if len(samples) >= length {
		return samples[:length]
	}
	out := make([]float32, length)
	copy(out, samples)
	return out
}

func secToDuration(sec float64) time.Duration {
	return time.Duration(sec * float64(time.Second))
}

// clampRange clamps start and end to at most n (the available sample count).
func clampRange(start, end, n int) (int, int) {
	if start > n {
		start = n
	}
	if end > n {
		end = n
	}
	return start, end
}

// containsRune reports whether the trimmed string s equals one of the single
// runes in chars. Empty strings are never matched.
func containsRune(chars, s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	for _, r := range chars {
		if s == string(r) {
			return true
		}
	}
	return false
}

// joinSegmentsText concatenates segment texts separated by single spaces.
func joinSegmentsText(segments []Segment) string {
	var buf strings.Builder
	for i, seg := range segments {
		if i > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteString(seg.Text)
	}
	return buf.String()
}

func ptrFloat32(v float32) *float32 { return &v }
func ptrBool(v bool) *bool          { return &v }
