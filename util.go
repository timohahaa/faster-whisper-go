package whisper

import "time"

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

func ptrFloat32(v float32) *float32 { return &v }
func ptrBool(v bool) *bool          { return &v }
