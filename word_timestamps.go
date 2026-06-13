package whisper

import (
	"math"
	"sort"
	"strings"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

const defaultMedianFilterWidth = 7

const tokensPerSecond = float64(whisperSampleRate) / float64(whisperHopLength*inputStride)

const sentenceEndMarks = ".。!！?？"

// alignmentWord holds per-word alignment data produced by findAlignment,
// before it is converted to public Word structs.
type alignmentWord struct {
	word        string
	tokens      []int32
	start       float64
	end         float64
	probability float32
}

// findAlignment computes word-level timing and probability from CTranslate2's
// align output. This is a direct port of Python's find_alignment.
func (m *Model) findAlignment(
	enc *ct2bridge.EncoderOutput,
	textTokens []int32,
	lang string,
	taskToken int32,
	numFrames int,
) ([]alignmentWord, error) {
	if len(textTokens) == 0 {
		return nil, nil
	}

	startSeq := m.buildAlignStartSequence(lang, taskToken)

	alignment, err := m.bridge.Align(enc, startSeq, textTokens, numFrames, defaultMedianFilterWidth)
	if err != nil {
		return nil, err
	}

	if alignment.NumTokens == 0 || len(alignment.TextIndices) == 0 {
		return nil, nil
	}

	textIndices := alignment.TextIndices
	timeIndices := alignment.TimeIndices
	textTokenProbs := alignment.TextTokenProbs

	tokensWithEOT := make([]int32, len(textTokens)+1)
	copy(tokensWithEOT, textTokens)
	tokensWithEOT[len(textTokens)] = tokenEOT

	words, wordTokens := m.tokenizer.splitToWordTokens(tokensWithEOT, lang)
	if len(wordTokens) <= 1 {
		return nil, nil
	}

	wordBoundaries := make([]int, len(wordTokens))
	wordBoundaries[0] = 0
	cum := 0
	for i := 0; i < len(wordTokens)-1; i++ {
		cum += len(wordTokens[i])
		wordBoundaries[i+1] = cum
	}

	// jumps: positions where text_index changes (or first position).
	jumps := make([]bool, len(textIndices))
	jumps[0] = true
	for i := 1; i < len(textIndices); i++ {
		if textIndices[i] != textIndices[i-1] {
			jumps[i] = true
		}
	}

	// jumpTimes: time_index values at jump positions, divided by tokensPerSecond.
	var jumpTimes []float64
	for i, isJump := range jumps {
		if isJump {
			jumpTimes = append(jumpTimes, float64(timeIndices[i])/tokensPerSecond)
		}
	}

	// Exclude the last word (EOT) from output.
	nWords := len(wordTokens) - 1
	result := make([]alignmentWord, 0, nWords)

	for wi := 0; wi < nWords; wi++ {
		startBound := wordBoundaries[wi]
		endBound := wordBoundaries[wi+1]

		var startTime, endTime float64
		if startBound < len(jumpTimes) {
			startTime = jumpTimes[startBound]
		}
		if endBound < len(jumpTimes) {
			endTime = jumpTimes[endBound]
		} else if len(jumpTimes) > 0 {
			endTime = jumpTimes[len(jumpTimes)-1]
		}

		// Compute word probability as mean of text_token_probs in [startBound, endBound).
		var prob float32
		count := 0
		for ti := startBound; ti < endBound && ti < len(textTokenProbs); ti++ {
			prob += textTokenProbs[ti]
			count++
		}
		if count > 0 {
			prob /= float32(count)
		}

		result = append(result, alignmentWord{
			word:        words[wi],
			tokens:      wordTokens[wi],
			start:       startTime,
			end:         endTime,
			probability: prob,
		})
	}

	return result, nil
}

// addWordTimestamps computes word-level timestamps for segments, applies
// duration truncation hacks, merge_punctuations, and segment boundary
// adjustment. This is a port of Python's add_word_timestamps.
//
// It modifies segments in place (setting Words, and adjusting Start/End)
// and returns the updated lastSpeechTimestamp.
func (m *Model) addWordTimestamps(
	enc *ct2bridge.EncoderOutput,
	segments []Segment,
	segmentTokens [][]int32,
	lang string,
	taskToken int32,
	numFrames int,
	seek int,
	prependPunct, appendPunct string,
	lastSpeechTimestamp float64,
) ([]Segment, float64) {
	if len(segments) == 0 {
		return segments, lastSpeechTimestamp
	}

	// Collect all text tokens across subsegments for a single alignment call.
	var allTextTokens []int32
	for _, toks := range segmentTokens {
		filtered := filterTextTokens(toks)
		allTextTokens = append(allTextTokens, filtered...)
	}

	alignment, err := m.findAlignment(enc, allTextTokens, lang, taskToken, numFrames)
	if err != nil || len(alignment) == 0 {
		return segments, lastSpeechTimestamp
	}

	// Compute median and max duration for truncation hacks.
	var durations []float64
	for _, w := range alignment {
		d := w.end - w.start
		if d > 0 {
			durations = append(durations, d)
		}
	}
	medianDuration := 0.0
	if len(durations) > 0 {
		medianDuration = median(durations)
	}
	medianDuration = math.Min(0.7, medianDuration)
	maxDuration := medianDuration * 2

	// Truncate long words at sentence boundaries.
	if len(durations) > 0 {
		for i := 1; i < len(alignment); i++ {
			if alignment[i].end-alignment[i].start > maxDuration {
				if isSentenceEndMark(alignment[i].word) {
					alignment[i].end = alignment[i].start + maxDuration
				} else if i > 0 && isSentenceEndMark(alignment[i-1].word) {
					alignment[i].start = alignment[i].end - maxDuration
				}
			}
		}
	}

	// Merge punctuations on the alignment words.
	mergePunctuationsOnAlignment(alignment, prependPunct, appendPunct)

	// Distribute aligned words to segments and apply timing hacks.
	timeOffset := float64(seek) / framesPerSecond
	wordIndex := 0

	for si := range segments {
		nSegTokens := len(filterTextTokens(segmentTokens[si]))
		savedTokens := 0
		var words []Word

		for wordIndex < len(alignment) && savedTokens < nSegTokens {
			aw := alignment[wordIndex]

			if aw.word != "" {
				words = append(words, Word{
					Start:       secToDuration(math.Round((timeOffset+aw.start)*100) / 100),
					End:         secToDuration(math.Round((timeOffset+aw.end)*100) / 100),
					Word:        strings.TrimSpace(aw.word),
					Probability: aw.probability,
				})
			}

			savedTokens += len(aw.tokens)
			wordIndex++
		}

		if len(words) > 0 {
			// Pause handling: truncate first word after a long pause.
			if words[0].End.Seconds()-lastSpeechTimestamp > medianDuration*4 &&
				(words[0].End.Seconds()-words[0].Start.Seconds() > maxDuration ||
					(len(words) > 1 && words[1].End.Seconds()-words[0].Start.Seconds() > maxDuration*2)) {
				if len(words) > 1 && words[1].End.Seconds()-words[1].Start.Seconds() > maxDuration {
					boundary := math.Max(words[1].End.Seconds()/2, words[1].End.Seconds()-maxDuration)
					words[0].End = secToDuration(boundary)
					words[1].Start = secToDuration(boundary)
				}
				words[0].Start = secToDuration(math.Max(0, words[0].End.Seconds()-maxDuration))
			}

			segStart := segments[si].Start.Seconds()
			segEnd := segments[si].End.Seconds()

			// Prefer segment-level start timestamp if the first word is too long.
			if segStart < words[0].End.Seconds() && segStart-0.5 > words[0].Start.Seconds() {
				words[0].Start = secToDuration(math.Max(0, math.Min(words[0].End.Seconds()-medianDuration, segStart)))
			} else {
				segments[si].Start = words[0].Start
			}

			// Prefer segment-level end timestamp if the last word is too long.
			last := len(words) - 1
			if segEnd > words[last].Start.Seconds() && segEnd+0.5 < words[last].End.Seconds() {
				words[last].End = secToDuration(math.Max(words[last].Start.Seconds()+medianDuration, segEnd))
			} else {
				segments[si].End = words[last].End
			}

			lastSpeechTimestamp = segments[si].End.Seconds()
		}

		segments[si].Words = words
	}

	return segments, lastSpeechTimestamp
}

// filterTextTokens returns only non-special, non-timestamp tokens.
func filterTextTokens(tokens []int32) []int32 {
	// Fast path: if all tokens are text tokens, return the input slice directly.
	allText := true
	for _, id := range tokens {
		if id >= tokenEOT {
			allText = false
			break
		}
	}
	if allText {
		return tokens
	}

	out := make([]int32, 0, len(tokens))
	for _, id := range tokens {
		if id < tokenEOT {
			out = append(out, id)
		}
	}
	return out
}

// buildAlignStartSequence constructs the start sequence for alignment:
// [SOT, lang_token, task_token]
func (m *Model) buildAlignStartSequence(lang string, taskToken int32) []int32 {
	seq := []int32{tokenSOT}
	if m.IsMultilingual() && lang != "" {
		langTok := m.tokenizer.LanguageToken(lang)
		if langTok >= 0 {
			seq = append(seq, langTok)
		}
		seq = append(seq, taskToken)
	}
	return seq
}

// mergePunctuationsOnAlignment merges punctuation-only alignment words with
// their neighbors, matching Python's merge_punctuations. Words at this stage
// still have their BPE-decoded leading spaces.
func mergePunctuationsOnAlignment(alignment []alignmentWord, prependChars, appendChars string) {
	if len(alignment) <= 1 {
		return
	}

	// Pass 1: merge prepend punctuations (right to left).
	i := len(alignment) - 2
	j := len(alignment) - 1
	for i >= 0 {
		w := alignment[i].word
		if strings.HasPrefix(w, " ") && containsRune(prependChars, strings.TrimSpace(w)) {
			alignment[j].word = alignment[i].word + alignment[j].word
			alignment[j].tokens = append(alignment[i].tokens, alignment[j].tokens...)
			alignment[i].word = ""
			alignment[i].tokens = nil
		} else {
			j = i
		}
		i--
	}

	// Pass 2: merge append punctuations (left to right).
	i = 0
	j = 1
	for j < len(alignment) {
		prev := alignment[i].word
		following := alignment[j].word
		if prev != "" && !strings.HasSuffix(prev, " ") && containsRune(appendChars, following) {
			alignment[i].word = alignment[i].word + alignment[j].word
			alignment[i].tokens = append(alignment[i].tokens, alignment[j].tokens...)
			alignment[j].word = ""
			alignment[j].tokens = nil
		} else {
			i = j
		}
		j++
	}
}

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

func isSentenceEndMark(word string) bool {
	w := strings.TrimSpace(word)
	if w == "" {
		return false
	}
	return strings.ContainsAny(w, sentenceEndMarks)
}

func median(data []float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}
