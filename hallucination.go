package whisper

import "strings"

const anomalyPunctuation = `"'"¿([{-"'.。,，!！?？:：")]}、`

// wordAnomalyScore computes an anomaly score for a single word based on
// its probability and duration (matching Python's word_anomaly_score).
func wordAnomalyScore(w Word) float64 {
	prob := float64(w.Probability)
	dur := w.End.Seconds() - w.Start.Seconds()
	score := 0.0
	if prob < 0.15 {
		score += 1.0
	}
	if dur < 0.133 {
		score += (0.133 - dur) * 15
	}
	if dur > 2.0 {
		score += dur - 2.0
	}
	return score
}

// isSegmentAnomaly checks if a segment is likely a hallucination based on
// anomaly scores of its words (matching Python's is_segment_anomaly).
// Punctuation-only words are filtered out before scoring.
func isSegmentAnomaly(seg Segment) bool {
	if len(seg.Words) == 0 {
		return false
	}
	var words []Word
	for _, w := range seg.Words {
		if !isAnomalyPunctuation(w.Word) {
			words = append(words, w)
		}
	}
	if len(words) == 0 {
		return false
	}
	if len(words) > 8 {
		words = words[:8]
	}
	var score float64
	for _, w := range words {
		score += wordAnomalyScore(w)
	}
	n := float64(len(words))
	return score >= 3 || score+0.01 >= n
}

func isAnomalyPunctuation(word string) bool {
	word = strings.TrimSpace(word)
	if word == "" {
		return true
	}
	for _, r := range anomalyPunctuation {
		if word == string(r) {
			return true
		}
	}
	return false
}

// getLastWordEnd returns the end time (in seconds) of the last word across all segments.
func getLastWordEnd(segments []Segment) float64 {
	for i := len(segments) - 1; i >= 0; i-- {
		if len(segments[i].Words) > 0 {
			return segments[i].Words[len(segments[i].Words)-1].End.Seconds()
		}
	}
	if len(segments) > 0 {
		return segments[len(segments)-1].End.Seconds()
	}
	return 0
}

// firstSegmentWithWords returns the index of the first segment that has word timestamps,
// or -1 if none.
func firstSegmentWithWords(segments []Segment) int {
	for i, s := range segments {
		if len(s.Words) > 0 {
			return i
		}
	}
	return -1
}
