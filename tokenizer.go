package whisper

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// gpt2PreTokenizer is the GPT-2 pre-tokenization regex that splits text into
// chunks before BPE encoding. Each chunk is BPE-encoded independently, preventing
// merges from crossing word boundaries.
//
// Matches (in priority order): English contractions, optional-space + letters,
// optional-space + digits, optional-space + punctuation/symbols, whitespace runs.
var gpt2PreTokenizer = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`)

const maxTokenLength = 448

// tokenizer decodes/encodes Whisper token IDs.
type tokenizer struct {
	idToToken      map[int32]string
	tokenToID      map[string]int32
	langToToken    map[string]int32
	mergeRank      map[string]int
	byteEncoder    [256]rune  // byte -> printable rune
	byteDecoder    [512]int16 // printable rune -> byte, -1 if unmapped
	nonSpeechCache []int32

	// Per-model special token IDs, all resolved from the loaded vocabulary by
	// initSpecialTokens. Concrete IDs vary with the language-block size (e.g.
	// large-v3 shifts the post-language tokens by +1 vs v2), so they are read
	// from the model rather than hardcoded.
	eot            int32
	sot            int32
	transcribe     int32
	translate      int32
	sotLm          int32
	sotPrev        int32
	noSpeech       int32
	noTimestamps   int32
	timestampBegin int32
}

// initSpecialTokens resolves every special token ID from the loaded vocabulary
// using token-to-ID lookups. Concrete IDs vary
// across model variants whose language-token block differs in size, so nothing
// is hardcoded. Returns an error listing any tokens the model is missing.
func (t *tokenizer) initSpecialTokens() error {
	var missing []string
	resolve := func(contents ...string) int32 {
		for _, c := range contents {
			if id, ok := t.tokenToID[c]; ok {
				return id
			}
		}
		missing = append(missing, contents[0])
		return 0
	}
	t.eot = resolve("<|endoftext|>")
	t.sot = resolve("<|startoftranscript|>")
	t.transcribe = resolve("<|transcribe|>")
	t.translate = resolve("<|translate|>")
	t.sotLm = resolve("<|startoflm|>")
	t.sotPrev = resolve("<|startofprev|>")
	t.noSpeech = resolve("<|nospeech|>", "<|nocaptions|>")
	t.noTimestamps = resolve("<|notimestamps|>")
	if len(missing) > 0 {
		return fmt.Errorf("tokenizer.json missing special tokens: %s", strings.Join(missing, ", "))
	}
	// Timestamp tokens are not named entries in the vocabulary; the first one
	// (<|0.00|>) immediately follows <|notimestamps|>, so timestampBegin =
	// noTimestamps + 1.
	t.timestampBegin = t.noTimestamps + 1
	return nil
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
	buildByteMaps(t)

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

	if err := t.initSpecialTokens(); err != nil {
		return nil, err
	}

	return t, nil
}

// decode converts token IDs into text, skipping special tokens.
func (t *tokenizer) decode(ids []int32) string {
	var buf strings.Builder
	buf.Grow(len(ids) * 4)
	for _, id := range ids {
		if id >= t.eot {
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
		if id >= t.eot {
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
// Returns (words, wordTokens) where words retain leading spaces from BPE.
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
	fullRunes := []rune(decodedFull)
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
		isSpecial := len(subToks) > 0 && subToks[0] >= t.eot
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

// encode converts text into token IDs using GPT-2 BPE with regex pre-tokenization.
// The input is first split into chunks by the GPT-2 regex (contractions, words,
// numbers, punctuation, whitespace), then each chunk is byte-level BPE-encoded
// independently — standard GPT-2 byte-level BPE tokenization.
func (t *tokenizer) encode(text string) []int32 {
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
		buf.WriteRune(t.byteEncoder[text[i]])
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

// nonSpeechTokens returns the standard set of token IDs to suppress
// to avoid non-speech annotations like ♪♪♪, (SPEAKING FOREIGN LANGUAGE), [DAVID], etc.
// The result is cached after the first computation.
func (t *tokenizer) nonSpeechTokens() []int32 {
	if t.nonSpeechCache != nil {
		return t.nonSpeechCache
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

	dashTokens := t.encode(" -")
	if len(dashTokens) > 0 {
		resultSet[dashTokens[0]] = true
	}
	quoteTokens := t.encode(" '")
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
			tokens := t.encode(variant)
			if len(tokens) == 1 || isMisc(sym) {
				if len(tokens) > 0 {
					resultSet[tokens[0]] = true
				}
			}
		}
	}

	result := slices.Sorted(maps.Keys(resultSet))
	t.nonSpeechCache = result
	return result
}

// suppressedTokens expands the suppress list: -1 becomes the default non-speech set,
// and always-suppressed special tokens are appended.
func (t *tokenizer) suppressedTokens(suppress []int32) []int32 {
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
		for _, tok := range t.nonSpeechTokens() {
			set[tok] = true
		}
	}

	for _, tok := range []int32{
		t.transcribe, t.translate,
		t.sot, t.sotPrev, t.sotLm,
		t.noSpeech,
	} {
		set[tok] = true
	}

	return slices.Sorted(maps.Keys(set))
}

// languageToken returns the token ID for a language code, or -1 if not found.
func (t *tokenizer) languageToken(lang string) int32 {
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
		if int(r) < len(t.byteDecoder) && t.byteDecoder[r] >= 0 {
			buf.WriteByte(byte(t.byteDecoder[r]))
		} else {
			buf.WriteRune(r)
		}
	}
}

// buildByteMaps populates the tokenizer's byte<->rune tables in a single pass.
//
// GPT-2 BPE represents every byte (0-255) as a printable Unicode character so
// the vocabulary never contains invisible control chars or whitespace. The 188
// "safe" bytes (printable ASCII '!'-'~', Latin-1 Supplement '¡'-'¬' and
// '®'-'ÿ') map to the same codepoint (identity); the remaining 68 bytes (space,
// tab, newline, DEL, other control chars) are assigned to codepoints starting
// at U+0100 (Ā, ā, Ă, …), in increasing byte order.
//
// byteEncoder maps byte -> rune; byteDecoder maps rune -> byte (-1 = unmapped).
func buildByteMaps(t *tokenizer) {
	for i := range t.byteDecoder {
		t.byteDecoder[i] = -1
	}
	n := 0
	for b := 0; b < 256; b++ {
		var r rune
		switch {
		case b >= '!' && b <= '~', b >= '\u00a1' && b <= '\u00ac', b >= '\u00ae' && b <= '\u00ff':
			r = rune(b)
		default:
			r = rune(256 + n)
			n++
		}
		t.byteEncoder[b] = r
		t.byteDecoder[r] = int16(b)
	}
}
