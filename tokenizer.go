package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// gpt2PreTokenizer is the GPT-2 pre-tokenization regex that splits text into
// chunks before BPE encoding. Each chunk is BPE-encoded independently, preventing
// merges from crossing word boundaries.
//
// Matches (in priority order): English contractions, optional-space + letters,
// optional-space + digits, optional-space + punctuation/symbols, whitespace runs.
var gpt2PreTokenizer = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`)

// Well-known Whisper special token IDs. tokenEOT and tokenSOT sit immediately
// after the 50257-entry base vocabulary and are the same for every Whisper
// model. The remaining special tokens come after the language-token block,
// whose size differs between models (99 languages for large-v2, 100 for
// large-v3/turbo), so their concrete IDs are resolved per-model from the
// tokenizer in initSpecialTokens. The constants below are the large-v2 layout
// and serve only as a fallback when a model is missing the named tokens.
const (
	tokenEOT          int32 = 50257
	tokenSOT          int32 = 50258
	tokenSOTprev      int32 = 50361
	tokenNoSpeech     int32 = 50362
	tokenNoTimestamps int32 = 50363
	tokenTimestampBeg int32 = 50364
)

// Task tokens (large-v2 fallback layout, see note above).
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
	mergeRank       map[string]int
	byteDecoder     [512]byte
	byteEncoder     map[byte]rune
	nonSpeechTokens []int32

	// Per-model special token IDs, resolved by initSpecialTokens. These vary
	// with the language-block size (e.g. large-v3 shifts them by +1 vs v2).
	transcribe     int32
	translate      int32
	sotLm          int32
	sotPrev        int32
	noSpeech       int32
	noTimestamps   int32
	timestampBegin int32
}

// initSpecialTokens resolves model-specific special token IDs from the loaded
// vocabulary, falling back to the large-v2 constants when a token is absent.
// This keeps timestamp parsing and prompt construction correct across model
// variants whose language-token block differs in size.
func (t *tokenizer) initSpecialTokens() {
	resolve := func(fallback int32, contents ...string) int32 {
		for _, c := range contents {
			if id, ok := t.tokenToID[c]; ok {
				return id
			}
		}
		return fallback
	}
	t.transcribe = resolve(tokenTranscribe, "<|transcribe|>")
	t.translate = resolve(tokenTranslate, "<|translate|>")
	t.sotLm = resolve(tokenSOTlm, "<|startoflm|>")
	t.sotPrev = resolve(tokenSOTprev, "<|startofprev|>")
	t.noSpeech = resolve(tokenNoSpeech, "<|nospeech|>", "<|nocaptions|>")
	t.noTimestamps = resolve(tokenNoTimestamps, "<|notimestamps|>")
	t.timestampBegin = resolve(tokenTimestampBeg, "<|0.00|>")
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

	t.mergeRank = make(map[string]int, len(raw.Model.Merges))
	for i, m := range raw.Model.Merges {
		if _, ok := t.mergeRank[m]; !ok {
			t.mergeRank[m] = i
		}
	}

	t.initSpecialTokens()

	return t, nil
}

// Decode converts token IDs into text, skipping special tokens.
func (t *tokenizer) Decode(ids []int32) string {
	var buf strings.Builder
	buf.Grow(len(ids) * 4)
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

// decodeWithTimestamps converts token IDs into text, rendering timestamp tokens
// as "<|0.00|>"-style markers instead of skipping them.
func (t *tokenizer) decodeWithTimestamps(ids []int32) string {
	var buf strings.Builder
	for _, id := range ids {
		if id >= t.timestampBegin {
			ts := float64(id-t.timestampBegin) * timePrecision
			fmt.Fprintf(&buf, "<|%.2f|>", ts)
			continue
		}
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

var cjkLanguages = map[string]bool{
	"zh": true, "ja": true, "th": true,
	"lo": true, "my": true, "yue": true,
}

// splitToWordTokens groups tokens into words. For CJK languages the split is
// character-based (unicode boundaries); for others it is space-based.
// Returns (words, wordTokens) where words retain leading spaces from BPE
// (matching Python's split_to_word_tokens).
func (t *tokenizer) splitToWordTokens(tokens []int32, lang string) ([]string, [][]int32) {
	if cjkLanguages[lang] {
		return t.splitTokensOnUnicode(tokens)
	}
	return t.splitTokensOnSpaces(tokens)
}

// splitTokensOnUnicode splits tokens at valid unicode decode boundaries.
// Each token that decodes without replacement chars forms its own word.
func (t *tokenizer) splitTokensOnUnicode(tokens []int32) ([]string, [][]int32) {
	decodedFull := t.decodeWithTimestamps(tokens)
	const replacementChar = '\ufffd'

	var words []string
	var wordTokens [][]int32
	var currentTokens []int32
	unicodeOffset := 0

	for _, id := range tokens {
		currentTokens = append(currentTokens, id)
		decoded := t.decodeWithTimestamps(currentTokens)

		replacementCharIndex := -1
		for i, r := range decoded {
			if r == replacementChar {
				replacementCharIndex = i + unicodeOffset
				break
			}
		}

		skip := false
		if replacementCharIndex >= 0 {
			fullRunes := []rune(decodedFull)
			if replacementCharIndex < len(fullRunes) && fullRunes[replacementCharIndex] != replacementChar {
				skip = true
			}
		}

		if replacementCharIndex < 0 || !skip {
			words = append(words, decoded)
			wordTokens = append(wordTokens, currentTokens)
			currentTokens = nil
			unicodeOffset += len([]rune(decoded))
		}
	}

	return words, wordTokens
}

// splitTokensOnSpaces splits tokens into words by spaces, using unicode
// split as a base, then merging subwords that don't start with a space.
func (t *tokenizer) splitTokensOnSpaces(tokens []int32) ([]string, [][]int32) {
	subwords, subwordTokensList := t.splitTokensOnUnicode(tokens)

	var words []string
	var wordTokens [][]int32

	for i, subword := range subwords {
		subToks := subwordTokensList[i]
		isSpecial := len(subToks) > 0 && subToks[0] >= tokenEOT
		withSpace := strings.HasPrefix(subword, " ")
		isPunct := len(strings.TrimSpace(subword)) == 1 && isPunctuationChar(strings.TrimSpace(subword))

		if isSpecial || withSpace || isPunct || len(words) == 0 {
			words = append(words, subword)
			wordTokens = append(wordTokens, subToks)
		} else {
			words[len(words)-1] += subword
			wordTokens[len(wordTokens)-1] = append(wordTokens[len(wordTokens)-1], subToks...)
		}
	}

	return words, wordTokens
}

func isPunctuationChar(s string) bool {
	if len(s) != 1 {
		return false
	}
	const punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
	return strings.ContainsAny(s, punctuation)
}

// Encode converts text into token IDs using GPT-2 BPE with regex pre-tokenization.
// The input is first split into chunks by the GPT-2 regex (contractions, words,
// numbers, punctuation, whitespace), then each chunk is byte-level BPE-encoded
// independently — matching the HuggingFace/Python tokenizer behavior.
func (t *tokenizer) Encode(text string) []int32 {
	if text == "" {
		return nil
	}
	chunks := gpt2PreTokenizer.FindAllString(text, -1)
	ids := make([]int32, 0, len(chunks)*2)
	for _, chunk := range chunks {
		bpeWord := t.textToBPEString(chunk)
		ids = append(ids, t.bpeEncode(bpeWord)...)
	}
	return ids
}

// textToBPEString converts raw text to the GPT-2 byte-level BPE representation.
func (t *tokenizer) textToBPEString(text string) string {
	var buf strings.Builder
	buf.Grow(len(text))
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

	runes := []rune(word)
	n := len(runes)
	if n == 1 {
		if id, ok := t.tokenToID[word]; ok {
			return []int32{id}
		}
		return nil
	}

	symbols := make([]string, n)
	for i, r := range runes {
		symbols[i] = string(r)
	}

	// Pre-allocate a buffer for building pair keys to avoid per-iteration allocation.
	var pairKey []byte

	for len(symbols) > 1 {
		bestIdx := -1
		bestRank := len(t.mergeRank)

		for i := 0; i < len(symbols)-1; i++ {
			pairKey = pairKey[:0]
			pairKey = append(pairKey, symbols[i]...)
			pairKey = append(pairKey, ' ')
			pairKey = append(pairKey, symbols[i+1]...)
			if rank, ok := t.mergeRank[string(pairKey)]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			break
		}

		symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
	}

	ids := make([]int32, 0, len(symbols))
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

	allSymbols := make([]string, 0, len(symbols)+len(multiSymbols)+len(miscellaneous))
	allSymbols = append(allSymbols, symbols...)
	allSymbols = append(allSymbols, multiSymbols...)
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
		t.transcribe, t.translate,
		tokenSOT, t.sotPrev, t.sotLm,
		t.noSpeech,
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


// LanguageToken returns the token ID for a language code, or -1 if not found.
func (t *tokenizer) LanguageToken(lang string) int32 {
	if id, ok := t.langToToken[lang]; ok {
		return id
	}
	return -1
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
		var tmp [utf8.UTFMax]byte
		n := utf8.EncodeRune(tmp[:], r)
		buf.Write(tmp[:n])
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
	return enc
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

	n := 0
	for i := range 256 {
		if !isSafe[i] {
			t.byteDecoder[256+n] = byte(i)
			n++
		}
	}
}
