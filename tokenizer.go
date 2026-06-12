package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
)

// Tokenizer decodes Whisper token IDs to text.
type Tokenizer struct {
	idToToken   map[int32]string
	langToToken map[string]int32
	byteDecoder map[rune]byte
}

// LoadTokenizer parses tokenizer.json from a model directory.
func LoadTokenizer(modelDir string) (*Tokenizer, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}

	var raw struct {
		Model struct {
			Vocab map[string]int32 `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	t := &Tokenizer{
		idToToken:   make(map[int32]string, len(raw.Model.Vocab)+len(raw.AddedTokens)),
		langToToken: make(map[string]int32),
		byteDecoder: buildByteDecoder(),
	}

	for token, id := range raw.Model.Vocab {
		t.idToToken[id] = token
	}
	for _, added := range raw.AddedTokens {
		t.idToToken[added.ID] = added.Content
		if isLangToken(added.Content) {
			lang := added.Content[2 : len(added.Content)-2] // strip <| and |>
			t.langToToken[lang] = added.ID
		}
	}

	return t, nil
}

// Decode converts token IDs into text, skipping special tokens.
func (t *Tokenizer) Decode(ids []int32) string {
	var buf strings.Builder
	for _, id := range ids {
		if t.IsSpecial(id) {
			continue
		}
		tok, ok := t.idToToken[id]
		if !ok {
			continue
		}
		buf.WriteString(t.decodeToken(tok))
	}
	return buf.String()
}

// DecodeSegmentTokens extracts text segments with timestamp boundaries.
// It groups tokens between consecutive timestamp token pairs.
func (t *Tokenizer) DecodeSegmentTokens(ids []int32) []rawSegment {
	var segments []rawSegment
	var current rawSegment
	var textTokens []int32
	inSegment := false

	for _, id := range ids {
		if id == tokenEOT {
			break
		}
		if t.IsTimestamp(id) {
			ts := t.TimestampValue(id)
			if !inSegment {
				current.start = ts
				inSegment = true
			} else {
				current.end = ts
				current.text = t.Decode(textTokens)
				segments = append(segments, current)
				textTokens = textTokens[:0]
				current = rawSegment{}
				inSegment = false
			}
			continue
		}
		if inSegment {
			textTokens = append(textTokens, id)
		}
	}

	// If we have leftover text without closing timestamp, emit it anyway
	if inSegment && len(textTokens) > 0 {
		current.text = t.Decode(textTokens)
		segments = append(segments, current)
	}

	return segments
}

// IsTimestamp reports whether a token ID is a timestamp token.
func (t *Tokenizer) IsTimestamp(id int32) bool {
	return id >= tokenTimestampBeg
}

// TimestampValue returns the time in seconds for a timestamp token.
func (t *Tokenizer) TimestampValue(id int32) float64 {
	return float64(id-tokenTimestampBeg) * 0.02
}

// IsSpecial reports whether a token ID is a special (non-text) token.
func (t *Tokenizer) IsSpecial(id int32) bool {
	return id >= tokenEOT
}

// LanguageToken returns the token ID for a language code, or -1 if not found.
func (t *Tokenizer) LanguageToken(lang string) int32 {
	if id, ok := t.langToToken[lang]; ok {
		return id
	}
	return -1
}

// rawSegment is an intermediate segment before converting to public Segment type.
type rawSegment struct {
	start float64
	end   float64
	text  string
}

func isLangToken(s string) bool {
	return strings.HasPrefix(s, "<|") && strings.HasSuffix(s, "|>") &&
		len(s) == 6 // <|xx|> — 2-char language code
}

// decodeToken converts a GPT-2 BPE token string to its UTF-8 representation.
// GPT-2 uses a byte-level encoding where certain unicode chars map to bytes.
func (t *Tokenizer) decodeToken(token string) string {
	var buf []byte
	for _, r := range token {
		if b, ok := t.byteDecoder[r]; ok {
			buf = append(buf, b)
		} else {
			buf = append(buf, []byte(string(r))...)
		}
	}
	return string(buf)
}

// buildByteDecoder builds the inverse of GPT-2's bytes_to_unicode mapping.
func buildByteDecoder() map[rune]byte {
	// GPT-2 maps bytes 0-255 to specific unicode characters to avoid
	// whitespace/control characters in the vocab.
	bs := make([]int, 0, 256)
	cs := make([]int, 0, 256)

	// printable byte ranges that map to themselves
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
		cs = append(cs, i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		bs = append(bs, i)
		cs = append(cs, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bs = append(bs, i)
		cs = append(cs, i)
	}

	n := 0
	for i := range 256 {
		found := false
		for _, b := range bs {
			if b == i {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, i)
			cs = append(cs, 256+n)
			n++
		}
	}

	decoder := make(map[rune]byte, 256)
	for i, c := range cs {
		decoder[rune(c)] = byte(bs[i])
	}
	return decoder
}

// Special token accessors used for prompt construction.
func sotToken() int32          { return tokenSOT }
func eotToken() int32          { return tokenEOT }
func transcribeToken() int32   { return tokenTranscribe }
func translateToken() int32    { return tokenTranslate }
func noTimestampsToken() int32 { return tokenNoTimestamps }

// FormatTimestamp formats seconds as "MM:SS.mmm" or "HH:MM:SS.mmm".
func FormatTimestamp(seconds float64) string {
	h := int(seconds) / 3600
	m := (int(seconds) % 3600) / 60
	s := seconds - float64(h*3600+m*60)
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%s", h, m, formatSec(s))
	}
	return fmt.Sprintf("%02d:%s", m, formatSec(s))
}

func formatSec(s float64) string {
	whole := int(s)
	frac := int((s - float64(whole)) * 1000)
	return strconv.Itoa(whole) + "." + fmt.Sprintf("%03d", frac)
}
