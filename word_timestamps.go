package whisper

import (
	"strings"
	"time"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

const defaultMedianFilterWidth = 7

// extractWordTimestamps computes word-level timestamps for a segment using
// cross-attention alignment from CTranslate2's align API.
func (m *Model) extractWordTimestamps(
	enc *ct2bridge.EncoderOutput,
	tokens []int32,
	lang string,
	taskToken int32,
	numFrames int,
	timeOffset float64,
	prepend, append string,
) ([]Word, error) {
	textTokens := filterTextTokens(tokens)
	if len(textTokens) == 0 {
		return nil, nil
	}

	startSeq := m.buildAlignStartSequence(lang, taskToken)

	alignment, err := m.bridge.Align(enc, startSeq, textTokens, numFrames, defaultMedianFilterWidth)
	if err != nil {
		return nil, err
	}

	if alignment.NumTokens == 0 || alignment.NumFrames == 0 {
		return nil, nil
	}

	wordBounds := m.splitTokensIntoWords(textTokens)
	words := assignWordTimestamps(wordBounds, alignment, timeOffset)
	words = mergePunctuations(words, prepend, append)

	return words, nil
}

// filterTextTokens returns only non-special, non-timestamp tokens.
func filterTextTokens(tokens []int32) []int32 {
	var out []int32
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

// wordBound represents a contiguous group of tokens forming a word.
type wordBound struct {
	tokens []int32
	text   string
}

// splitTokensIntoWords groups text tokens into words.
// A new word starts when the decoded token begins with a space.
func (m *Model) splitTokensIntoWords(tokens []int32) []wordBound {
	var words []wordBound
	var currentTokens []int32
	var currentText strings.Builder

	for _, id := range tokens {
		tok, ok := m.tokenizer.idToToken[id]
		if !ok {
			continue
		}

		var decoded strings.Builder
		m.tokenizer.decodeTokenInto(&decoded, tok)
		text := decoded.String()

		if strings.HasPrefix(text, " ") && currentText.Len() > 0 {
			words = append(words, wordBound{
				tokens: currentTokens,
				text:   currentText.String(),
			})
			currentTokens = nil
			currentText.Reset()
		}

		currentTokens = append(currentTokens, id)
		currentText.WriteString(text)
	}

	if currentText.Len() > 0 {
		words = append(words, wordBound{
			tokens: currentTokens,
			text:   currentText.String(),
		})
	}

	return words
}

// assignWordTimestamps uses alignment weights to assign start/end times to words.
// Each word's timing is determined by the frames where its tokens have peak attention.
func assignWordTimestamps(
	words []wordBound,
	alignment ct2bridge.AlignResult,
	timeOffset float64,
) []Word {
	if len(words) == 0 || alignment.NumTokens == 0 {
		return nil
	}

	nFrames := alignment.NumFrames

	tokenIdx := 0
	result := make([]Word, 0, len(words))

	for _, wb := range words {
		nToks := len(wb.tokens)
		if tokenIdx+nToks > alignment.NumTokens {
			break
		}

		startFrame, endFrame := findWordFrameBounds(
			alignment.Weights, tokenIdx, nToks, nFrames,
		)
		tokenIdx += nToks

		startTime := timeOffset + float64(startFrame)*timePerFrame
		endTime := timeOffset + float64(endFrame+1)*timePerFrame

		prob := computeWordProbability(alignment.Weights, tokenIdx-nToks, nToks, nFrames, startFrame, endFrame)

		result = append(result, Word{
			Start:       time.Duration(startTime * float64(time.Second)),
			End:         time.Duration(endTime * float64(time.Second)),
			Word:        strings.TrimSpace(wb.text),
			Probability: prob,
		})
	}

	return result
}

// findWordFrameBounds finds the start and end frame for a group of tokens
// by finding the frame with maximum total attention weight across the token group.
func findWordFrameBounds(weights []float32, tokenStart, nToks, nFrames int) (int, int) {
	startFrame := nFrames - 1
	endFrame := 0

	for t := tokenStart; t < tokenStart+nToks; t++ {
		maxWeight := float32(-1)
		maxFrame := 0
		rowBase := t * nFrames
		for f := 0; f < nFrames; f++ {
			if weights[rowBase+f] > maxWeight {
				maxWeight = weights[rowBase+f]
				maxFrame = f
			}
		}
		if maxFrame < startFrame {
			startFrame = maxFrame
		}
		if maxFrame > endFrame {
			endFrame = maxFrame
		}
	}

	return startFrame, endFrame
}

// computeWordProbability computes the average attention probability for a word
// over its assigned frame range.
func computeWordProbability(weights []float32, tokenStart, nToks, nFrames, startFrame, endFrame int) float32 {
	if endFrame < startFrame || nToks == 0 {
		return 0
	}

	frameSpan := endFrame - startFrame + 1
	var sum float32
	for t := tokenStart; t < tokenStart+nToks; t++ {
		rowBase := t * nFrames
		for f := startFrame; f <= endFrame; f++ {
			sum += weights[rowBase+f]
		}
	}

	return sum / float32(nToks*frameSpan)
}

// mergePunctuations merges punctuation-only words with their neighbors:
// - Prepend punctuations (e.g. opening quotes/brackets) merge with the next word.
// - Append punctuations (e.g. periods, commas) merge with the previous word.
// This matches the Python merge_punctuations behavior.
func mergePunctuations(words []Word, prependChars, appendChars string) []Word {
	if len(words) <= 1 {
		return words
	}

	isPrepend := func(w string) bool {
		w = strings.TrimSpace(w)
		if w == "" {
			return false
		}
		for _, r := range prependChars {
			if w == string(r) {
				return true
			}
		}
		return false
	}

	isAppend := func(w string) bool {
		w = strings.TrimSpace(w)
		if w == "" {
			return false
		}
		for _, r := range appendChars {
			if w == string(r) {
				return true
			}
		}
		return false
	}

	// Pass 1: merge prepend punctuations (right to left).
	// If word[i] is a prepend punctuation, merge it into word[j] (the next non-empty word).
	i := len(words) - 2
	j := len(words) - 1
	for i >= 0 {
		if isPrepend(words[i].Word) {
			words[j].Word = words[i].Word + words[j].Word
			words[j].Start = words[i].Start
			words[j].Probability = (words[i].Probability + words[j].Probability) / 2
			words[i].Word = ""
		} else {
			j = i
		}
		i--
	}

	// Pass 2: merge append punctuations (left to right).
	i = 0
	j = 1
	for j < len(words) {
		if words[i].Word == "" {
			i = j
			j++
			continue
		}
		if isAppend(words[j].Word) {
			words[i].Word = words[i].Word + words[j].Word
			words[i].End = words[j].End
			words[i].Probability = (words[i].Probability + words[j].Probability) / 2
			words[j].Word = ""
		} else {
			i = j
		}
		j++
	}

	// Filter out empty words.
	result := make([]Word, 0, len(words))
	for _, w := range words {
		if w.Word != "" {
			result = append(result, w)
		}
	}
	return result
}
