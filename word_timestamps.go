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
