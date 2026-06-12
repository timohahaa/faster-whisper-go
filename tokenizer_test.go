package whisper

import (
	"testing"
)

func makeTestTokenizer() *tokenizer {
	t := &tokenizer{
		idToToken:   make(map[int32]string),
		langToToken: make(map[string]int32),
	}
	buildByteDecoderInto(t)
	t.idToToken[0] = "Hello"
	t.idToToken[1] = "Ġworld"
	t.idToToken[2] = "Ġfoo"
	t.idToToken[3] = "bar"

	t.idToToken[tokenEOT] = "<|endoftext|>"
	t.idToToken[tokenSOT] = "<|startoftranscript|>"
	t.idToToken[tokenTimestampBeg] = "<|0.00|>"
	t.idToToken[tokenTimestampBeg+1] = "<|0.02|>"
	t.idToToken[tokenTimestampBeg+50] = "<|1.00|>"
	t.idToToken[tokenTimestampBeg+150] = "<|3.00|>"

	t.idToToken[50259] = "<|en|>"
	t.idToToken[50260] = "<|ru|>"
	t.langToToken["en"] = 50259
	t.langToToken["ru"] = 50260

	return t
}

func TestBuildByteDecoder(t *testing.T) {
	dec := buildByteDecoder()
	if len(dec) != 256 {
		t.Fatalf("byteDecoder should have 256 entries, got %d", len(dec))
	}

	for b := byte('!'); b <= byte('~'); b++ {
		got, ok := dec[rune(b)]
		if !ok {
			t.Errorf("missing identity mapping for byte %d (char %c)", b, b)
		}
		if got != b {
			t.Errorf("identity mapping: rune %d → byte %d, want %d", b, got, b)
		}
	}

	if _, ok := dec[' ']; ok {
		t.Error("space should not have identity mapping in GPT-2 byte decoder")
	}

	bytesSeen := make(map[byte]bool)
	for _, b := range dec {
		bytesSeen[b] = true
	}
	if len(bytesSeen) != 256 {
		t.Errorf("expected 256 unique byte values, got %d", len(bytesSeen))
	}
}

func TestIsLangToken(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"<|en|>", true},
		{"<|ru|>", true},
		{"<|zh|>", true},
		{"<|endoftext|>", false},
		{"<|eng|>", false},
		{"<||>", false},
		{"hello", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isLangToken(tt.input); got != tt.want {
			t.Errorf("isLangToken(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestTokenizerTimestampValue(t *testing.T) {
	tok := makeTestTokenizer()
	tests := []struct {
		id   int32
		want float64
	}{
		{tokenTimestampBeg, 0.0},
		{tokenTimestampBeg + 1, 0.02},
		{tokenTimestampBeg + 50, 1.0},
		{tokenTimestampBeg + 1500, 30.0},
	}
	for _, tt := range tests {
		got := tok.TimestampValue(tt.id)
		if diff := got - tt.want; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("TimestampValue(%d) = %f, want %f", tt.id, got, tt.want)
		}
	}
}

func TestDecodeToken(t *testing.T) {
	tok := makeTestTokenizer()

	if got := tok.decodeToken("Hello"); got != "Hello" {
		t.Errorf("decodeToken(Hello) = %q, want %q", got, "Hello")
	}

	// Ġ (U+0120) is GPT-2's encoding of space (0x20).
	got := tok.decodeToken("Ġworld")
	if got != " world" {
		t.Errorf("decodeToken(Ġworld) = %q, want %q", got, " world")
	}
}

func TestDecode(t *testing.T) {
	tok := makeTestTokenizer()

	got := tok.Decode([]int32{0, 1})
	if got != "Hello world" {
		t.Errorf("Decode([0,1]) = %q, want %q", got, "Hello world")
	}

	got = tok.Decode([]int32{0, tokenEOT, 1})
	if got != "Hello world" {
		t.Errorf("Decode with special = %q, want %q", got, "Hello world")
	}

	got = tok.Decode([]int32{0, 99999, 1})
	if got != "Hello world" {
		t.Errorf("Decode with unknown = %q, want %q", got, "Hello world")
	}

	got = tok.Decode(nil)
	if got != "" {
		t.Errorf("Decode(nil) = %q, want empty", got)
	}
}

func TestDecodeSegmentTokens(t *testing.T) {
	tok := makeTestTokenizer()

	t.Run("Complete", func(t *testing.T) {
		ts0 := tokenTimestampBeg
		ts1 := tokenTimestampBeg + 50
		ts3 := tokenTimestampBeg + 150

		ids := []int32{ts0, 0, 1, ts1, ts1, 2, 3, ts3, tokenEOT}
		segs := tok.DecodeSegmentTokens(ids)

		if len(segs) != 2 {
			t.Fatalf("expected 2 segments, got %d", len(segs))
		}

		if segs[0].start != 0.0 || segs[0].end != 1.0 {
			t.Errorf("seg[0] time: got [%f, %f], want [0.0, 1.0]", segs[0].start, segs[0].end)
		}
		if segs[0].text != "Hello world" {
			t.Errorf("seg[0] text: got %q, want %q", segs[0].text, "Hello world")
		}

		if segs[1].start != 1.0 || segs[1].end != 3.0 {
			t.Errorf("seg[1] time: got [%f, %f], want [1.0, 3.0]", segs[1].start, segs[1].end)
		}
		if segs[1].text != " foobar" {
			t.Errorf("seg[1] text: got %q, want %q", segs[1].text, " foobar")
		}
	})

	t.Run("Incomplete", func(t *testing.T) {
		ts0 := tokenTimestampBeg
		ids := []int32{ts0, 0, 1}
		segs := tok.DecodeSegmentTokens(ids)

		if len(segs) != 1 {
			t.Fatalf("expected 1 incomplete segment, got %d", len(segs))
		}
		if segs[0].text != "Hello world" {
			t.Errorf("incomplete seg text: got %q, want %q", segs[0].text, "Hello world")
		}
	})
}

func BenchmarkDecode(b *testing.B) {
	tok := makeTestTokenizer()
	ids := []int32{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}
	b.ResetTimer()
	for b.Loop() {
		tok.Decode(ids)
	}
}

func BenchmarkDecodeSegmentTokens(b *testing.B) {
	tok := makeTestTokenizer()
	ts0 := tokenTimestampBeg
	ts1 := tokenTimestampBeg + 50
	ts3 := tokenTimestampBeg + 150
	ids := []int32{ts0, 0, 1, ts1, ts1, 2, 3, ts3, tokenEOT}
	b.ResetTimer()
	for b.Loop() {
		tok.DecodeSegmentTokens(ids)
	}
}
