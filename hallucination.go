package whisper

import "strings"

const anomalyPunctuation = `"'"¿([{-"'.。,，!！?？:：")]}、`

const (
	// Anomaly scoring thresholds (match Python's word_anomaly_score).
	anomalyLowProb           = 0.15  // words below this probability are suspicious
	anomalyShortDurationS    = 0.133 // words shorter than this are penalized
	anomalyShortPenaltyScale = 15.0  // penalty scale for too-short words
	anomalyLongDurationS     = 2.0   // words longer than this are penalized
	anomalyMaxWords          = 8     // only the first N words of a segment are scored
	anomalyScoreThreshold    = 3.0   // total score at/above which a segment is anomalous

	// hallucinationEdgeMarginS is the silence margin (seconds) around a segment
	// used when deciding whether a possible hallucination is surrounded by silence.
	hallucinationEdgeMarginS = 2.0
)

// wordAnomalyScore computes an anomaly score for a single word based on
// its probability and duration (matching Python's word_anomaly_score).
func wordAnomalyScore(w Word) float64 {
	prob := float64(w.Probability)
	dur := w.End.Seconds() - w.Start.Seconds()
	score := 0.0
	if prob < anomalyLowProb {
		score += 1.0
	}
	if dur < anomalyShortDurationS {
		score += (anomalyShortDurationS - dur) * anomalyShortPenaltyScale
	}
	if dur > anomalyLongDurationS {
		score += dur - anomalyLongDurationS
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
	if len(words) > anomalyMaxWords {
		words = words[:anomalyMaxWords]
	}
	var score float64
	for _, w := range words {
		score += wordAnomalyScore(w)
	}
	n := float64(len(words))
	return score >= anomalyScoreThreshold || score+0.01 >= n
}

func isAnomalyPunctuation(word string) bool {
	return strings.TrimSpace(word) == "" || containsRune(anomalyPunctuation, word)
}

// lastWordEnd returns the end time (in seconds) of the last word across all segments.
func lastWordEnd(segments []Segment) float64 {
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
