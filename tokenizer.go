package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Well-known Whisper special token offsets (relative to base vocab size of 50257).
const (
	tokenEOT          int32 = 50257
	tokenSOT          int32 = 50258
	tokenSOTprev      int32 = 50361
	tokenNoSpeech     int32 = 50362
	tokenNoTimestamps int32 = 50363
	tokenTimestampBeg int32 = 50364
)

// Task tokens.
const (
	tokenTranslate  int32 = 50358
	tokenTranscribe int32 = 50359
	tokenSOTlm      int32 = 50360
)

const maxTokenLength = 448

// tokenizer decodes/encodes Whisper token IDs.
type tokenizer struct {
	idToToken       map[int32]string
	tokenToID       map[string]int32
	langToToken     map[string]int32
	merges          []bpeMerge
	mergeRank       map[string]int
	byteDecoder     [512]byte
	byteDecoderHigh map[rune]byte
	byteEncoder     map[byte]rune
	nonSpeechTokens []int32
}

type bpeMerge struct {
	a, b string
}

// loadTokenizer parses tokenizer.json from a model directory.
func loadTokenizer(modelDir string) (*tokenizer, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}

	var raw struct {
		Model struct {
			Vocab  map[string]int32 `json:"vocab"`
			Merges []string         `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	t := &tokenizer{
		idToToken:   make(map[int32]string, len(raw.Model.Vocab)+len(raw.AddedTokens)),
		tokenToID:   make(map[string]int32, len(raw.Model.Vocab)+len(raw.AddedTokens)),
		langToToken: make(map[string]int32),
	}
	buildByteDecoderInto(t)
	t.byteEncoder = buildByteEncoder(t)

	for token, id := range raw.Model.Vocab {
		t.idToToken[id] = token
		t.tokenToID[token] = id
	}
	for _, added := range raw.AddedTokens {
		t.idToToken[added.ID] = added.Content
		t.tokenToID[added.Content] = added.ID
		if isLangToken(added.Content) {
			lang := added.Content[2 : len(added.Content)-2]
			t.langToToken[lang] = added.ID
		}
	}

	t.merges = make([]bpeMerge, 0, len(raw.Model.Merges))
	for _, m := range raw.Model.Merges {
		parts := strings.SplitN(m, " ", 2)
		if len(parts) == 2 {
			t.merges = append(t.merges, bpeMerge{a: parts[0], b: parts[1]})
		}
	}

	t.mergeRank = make(map[string]int, len(t.merges))
	for i, m := range t.merges {
		t.mergeRank[m.a+" "+m.b] = i
	}

	return t, nil
}

// Decode converts token IDs into text, skipping special tokens.
func (t *tokenizer) Decode(ids []int32) string {
	var buf strings.Builder
	for _, id := range ids {
		if id >= tokenEOT {
			continue
		}
		tok, ok := t.idToToken[id]
		if !ok {
			continue
		}
		t.decodeTokenInto(&buf, tok)
	}
	return buf.String()
}

// Encode converts text into token IDs using GPT-2 BPE.
// Note: this is a simplified implementation that treats the entire input as one
// BPE word (no regex-based pre-tokenization). Suitable for short texts like
// prompts and hotwords.
func (t *tokenizer) Encode(text string) []int32 {
	if text == "" {
		return nil
	}

	bpeText := t.textToBPEString(text)

	words := strings.Split(bpeText, " ")
	var ids []int32
	for _, word := range words {
		if word == "" {
			continue
		}
		wordTokens := t.bpeEncode(word)
		ids = append(ids, wordTokens...)
	}
	return ids
}

// textToBPEString converts raw text to the GPT-2 byte-level BPE representation.
func (t *tokenizer) textToBPEString(text string) string {
	var buf strings.Builder
	for i := 0; i < len(text); i++ {
		r, ok := t.byteEncoder[text[i]]
		if ok {
			buf.WriteRune(r)
		} else {
			buf.WriteByte(text[i])
		}
	}
	return buf.String()
}

// bpeEncode applies BPE merges to a single word (already in BPE byte space).
func (t *tokenizer) bpeEncode(word string) []int32 {
	if len(word) == 0 {
		return nil
	}

	var symbols []string
	for _, r := range word {
		symbols = append(symbols, string(r))
	}

	for len(symbols) > 1 {
		bestIdx := -1
		bestRank := len(t.merges)

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.mergeRank[pair]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			break
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	var ids []int32
	for _, sym := range symbols {
		if id, ok := t.tokenToID[sym]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// NonSpeechTokens returns the standard set of token IDs to suppress
// to avoid non-speech annotations like ♪♪♪, (SPEAKING FOREIGN LANGUAGE), [DAVID], etc.
// The result is cached after the first computation.
func (t *tokenizer) NonSpeechTokens() []int32 {
	if t.nonSpeechTokens != nil {
		return t.nonSpeechTokens
	}

	symbols := []string{
		`"`, "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@",
		"[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
		"\u300c", "\u300d", "\u300e", "\u300f",
	}
	multiSymbols := strings.Split(
		"<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} \u266a\u266a \u266a\u266a\u266a",
		" ",
	)

	miscellaneous := "\u2669\u266a\u266b\u266c\u266d\u266e\u266f"

	resultSet := make(map[int32]bool)

	dashTokens := t.Encode(" -")
	if len(dashTokens) > 0 {
		resultSet[dashTokens[0]] = true
	}
	quoteTokens := t.Encode(" '")
	if len(quoteTokens) > 0 {
		resultSet[quoteTokens[0]] = true
	}

	allSymbols := append(symbols, multiSymbols...)
	for _, r := range miscellaneous {
		allSymbols = append(allSymbols, string(r))
	}

	isMisc := func(s string) bool {
		return len(s) > 0 && strings.ContainsRune(miscellaneous, []rune(s)[0])
	}

	for _, sym := range allSymbols {
		for _, variant := range []string{sym, " " + sym} {
			tokens := t.Encode(variant)
			if len(tokens) == 1 || isMisc(sym) {
				if len(tokens) > 0 {
					resultSet[tokens[0]] = true
				}
			}
		}
	}

	result := make([]int32, 0, len(resultSet))
	for id := range resultSet {
		result = append(result, id)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	t.nonSpeechTokens = result
	return result
}

// SuppressedTokens expands the suppress list: -1 becomes the default non-speech set,
// and always-suppressed special tokens are appended.
func (t *tokenizer) SuppressedTokens(suppress []int32) []int32 {
	set := make(map[int32]bool)

	hasDefault := false
	for _, tok := range suppress {
		if tok == -1 {
			hasDefault = true
		} else if tok >= 0 {
			set[tok] = true
		}
	}

	if hasDefault {
		for _, tok := range t.NonSpeechTokens() {
			set[tok] = true
		}
	}

	for _, tok := range []int32{
		tokenTranscribe, tokenTranslate,
		tokenSOT, tokenSOTprev, tokenSOTlm,
		tokenNoSpeech,
	} {
		set[tok] = true
	}

	result := make([]int32, 0, len(set))
	for id := range set {
		result = append(result, id)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result
}

// segmentSplitResult holds the output of SplitSegmentsByTimestamps.
type segmentSplitResult struct {
	segments []rawSegment
	seek     int
}

// SplitSegmentsByTimestamps parses timestamp tokens to determine segment boundaries
// and how far to advance the seek position, matching the Python _split_segments_by_timestamps.
func (t *tokenizer) SplitSegmentsByTimestamps(
	tokens []int32,
	timeOffset float64,
	segmentSize int,
	segmentDuration float64,
	seek int,
) segmentSplitResult {
	singleTimestampEnding := len(tokens) >= 2 &&
		tokens[len(tokens)-2] < tokenTimestampBeg &&
		tokens[len(tokens)-1] >= tokenTimestampBeg

	var consecutiveTimestamps []int
	for i := 1; i < len(tokens); i++ {
		if tokens[i] >= tokenTimestampBeg && tokens[i-1] >= tokenTimestampBeg {
			consecutiveTimestamps = append(consecutiveTimestamps, i)
		}
	}

	var segments []rawSegment

	if len(consecutiveTimestamps) > 0 {
		slices := append([]int(nil), consecutiveTimestamps...)
		if singleTimestampEnding {
			slices = append(slices, len(tokens))
		}

		lastSlice := 0
		for _, currentSlice := range slices {
			slicedTokens := tokens[lastSlice:currentSlice]
			startPos := slicedTokens[0] - tokenTimestampBeg
			endPos := slicedTokens[len(slicedTokens)-1] - tokenTimestampBeg
			startTime := timeOffset + float64(startPos)*timePrecision
			endTime := timeOffset + float64(endPos)*timePrecision

			segments = append(segments, rawSegment{
				start:  startTime,
				end:    endTime,
				tokens: copyTokens(slicedTokens),
			})
			lastSlice = currentSlice
		}

		if singleTimestampEnding {
			seek += segmentSize
		} else {
			lastTSPos := tokens[lastSlice-1] - tokenTimestampBeg
			seek += int(lastTSPos) * inputStride
		}
	} else {
		duration := segmentDuration
		var timestamps []int32
		for _, tok := range tokens {
			if tok >= tokenTimestampBeg {
				timestamps = append(timestamps, tok)
			}
		}
		if len(timestamps) > 0 && timestamps[len(timestamps)-1] != tokenTimestampBeg {
			lastTSPos := timestamps[len(timestamps)-1] - tokenTimestampBeg
			duration = float64(lastTSPos) * timePrecision
		}

		segments = append(segments, rawSegment{
			start:  timeOffset,
			end:    timeOffset + duration,
			tokens: copyTokens(tokens),
		})
		seek += segmentSize
	}

	return segmentSplitResult{
		segments: segments,
		seek:     seek,
	}
}

const timePrecision = 0.02

// IsTimestamp reports whether a token ID is a timestamp token.
func (t *tokenizer) IsTimestamp(id int32) bool {
	return id >= tokenTimestampBeg
}

// TimestampValue returns the time in seconds for a timestamp token.
func (t *tokenizer) TimestampValue(id int32) float64 {
	return float64(id-tokenTimestampBeg) * timePrecision
}

// IsSpecial reports whether a token ID is a special (non-text) token.
func (t *tokenizer) IsSpecial(id int32) bool {
	return id >= tokenEOT
}

// LanguageToken returns the token ID for a language code, or -1 if not found.
func (t *tokenizer) LanguageToken(lang string) int32 {
	if id, ok := t.langToToken[lang]; ok {
		return id
	}
	return -1
}

// rawSegment is an intermediate segment before converting to public Segment type.
type rawSegment struct {
	start  float64
	end    float64
	tokens []int32
}

func copyTokens(src []int32) []int32 {
	out := make([]int32, len(src))
	copy(out, src)
	return out
}

func isLangToken(s string) bool {
	if len(s) < 6 || len(s) > 7 {
		return false
	}
	if s[0] != '<' || s[1] != '|' || s[len(s)-2] != '|' || s[len(s)-1] != '>' {
		return false
	}
	inner := s[2 : len(s)-2]
	for _, r := range inner {
		if r < 'a' || r > 'z' {
			return false
		}
	}
	return true
}

// decodeTokenInto writes the decoded bytes of a GPT-2 BPE token directly into the provided Builder.
func (t *tokenizer) decodeTokenInto(buf *strings.Builder, token string) {
	for _, r := range token {
		if int(r) < len(t.byteDecoder) {
			b := t.byteDecoder[r]
			if b != 0 || r == 0 {
				buf.WriteByte(b)
				continue
			}
		}
		if b, ok := t.byteDecoderHigh[r]; ok {
			buf.WriteByte(b)
		} else {
			var tmp [utf8.UTFMax]byte
			n := utf8.EncodeRune(tmp[:], r)
			buf.Write(tmp[:n])
		}
	}
}

// buildByteEncoder creates the forward mapping byte -> rune for BPE encoding.
func buildByteEncoder(t *tokenizer) map[byte]rune {
	enc := make(map[byte]rune, 256)
	for i := range 512 {
		b := t.byteDecoder[i]
		if b != 0 || i == 0 {
			enc[b] = rune(i)
		}
	}
	for r, b := range t.byteDecoderHigh {
		enc[b] = r
	}
	return enc
}

// buildByteDecoder builds a byte decoder and returns it as a map (for tests).
func buildByteDecoder() map[rune]byte {
	t := &tokenizer{}
	buildByteDecoderInto(t)
	result := make(map[rune]byte, 256)
	for i, b := range t.byteDecoder {
		if b != 0 || i == 0 {
			result[rune(i)] = b
		}
	}
	for r, b := range t.byteDecoderHigh {
		result[r] = b
	}
	return result
}

// buildByteDecoderInto populates the tokenizer's byte decoder fields.
//
// GPT-2 BPE represents every byte (0-255) as a printable Unicode character so
// the vocabulary never contains invisible control chars or whitespace.
//
// The 188 "safe" bytes (printable ASCII '!'-'~', Latin-1 Supplement '¡'-'¬'
// and '®'-'ÿ') map to the same Unicode codepoint (identity). The remaining
// 68 bytes (space, tab, newline, DEL, other control chars) are assigned to
// codepoints starting at U+0100 (Ā, ā, Ă, …).
func buildByteDecoderInto(t *tokenizer) {
	var isSafe [256]bool

	for i := int('!'); i <= int('~'); i++ {
		isSafe[i] = true
		t.byteDecoder[i] = byte(i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		isSafe[i] = true
		t.byteDecoder[i] = byte(i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		isSafe[i] = true
		t.byteDecoder[i] = byte(i)
	}

	t.byteDecoderHigh = make(map[rune]byte)
	n := 0
	for i := range 256 {
		if !isSafe[i] {
			r := rune(256 + n)
			if int(r) < len(t.byteDecoder) {
				t.byteDecoder[r] = byte(i)
			} else {
				t.byteDecoderHigh[r] = byte(i)
			}
			n++
		}
	}
}
