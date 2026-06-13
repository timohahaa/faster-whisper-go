package whisper

import (
	"reflect"
	"strings"
	"testing"
)

func makeTestTokenizer() *tokenizer {
	t := &tokenizer{
		idToToken:   make(map[int32]string),
		tokenToID:   make(map[string]int32),
		langToToken: make(map[string]int32),
		mergeRank:   make(map[string]int),
	}
	buildByteDecoderInto(t)
	t.byteEncoder = buildByteEncoder(t)

	addToken := func(id int32, tok string) {
		t.idToToken[id] = tok
		t.tokenToID[tok] = id
	}

	addToken(0, "Hello")
	addToken(1, "Ġworld")
	addToken(2, "Ġfoo")
	addToken(3, "bar")

	addToken(tokenEOT, "<|endoftext|>")
	addToken(tokenSOT, "<|startoftranscript|>")
	addToken(tokenTimestampBeg, "<|0.00|>")
	addToken(tokenTimestampBeg+1, "<|0.02|>")
	addToken(tokenTimestampBeg+50, "<|1.00|>")
	addToken(tokenTimestampBeg+150, "<|3.00|>")

	addToken(50259, "<|en|>")
	addToken(50260, "<|ru|>")
	t.langToToken["en"] = 50259
	t.langToToken["ru"] = 50260

	addToken(tokenTranscribe, "<|transcribe|>")
	addToken(tokenTranslate, "<|translate|>")
	addToken(tokenSOTprev, "<|startofprev|>")
	addToken(tokenSOTlm, "<|startoflm|>")
	addToken(tokenNoSpeech, "<|nospeech|>")

	return t
}

func buildByteDecoder() map[rune]byte {
	t := &tokenizer{}
	buildByteDecoderInto(t)
	result := make(map[rune]byte, 256)
	for i, b := range t.byteDecoder {
		if b != 0 || i == 0 {
			result[rune(i)] = b
		}
	}
	return result
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
		{"<|haw|>", true},
		{"<|yue|>", true},
		{"<|endoftext|>", false},
		{"<|transcribe|>", false},
		{"<|startoftranscript|>", false},
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

func TestDecodeTokenInto(t *testing.T) {
	tok := makeTestTokenizer()

	decodeOne := func(token string) string {
		var buf strings.Builder
		tok.decodeTokenInto(&buf, token)
		return buf.String()
	}

	if got := decodeOne("Hello"); got != "Hello" {
		t.Errorf("decodeTokenInto(Hello) = %q, want %q", got, "Hello")
	}

	got := decodeOne("Ġworld")
	if got != " world" {
		t.Errorf("decodeTokenInto(Ġworld) = %q, want %q", got, " world")
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

func TestSplitSegmentsByTimestamps(t *testing.T) {
	tok := makeTestTokenizer()

	t.Run("ConsecutiveWithEOT", func(t *testing.T) {
		ts0 := tokenTimestampBeg         // 0.00
		ts50 := tokenTimestampBeg + 50   // 1.00
		ts150 := tokenTimestampBeg + 150 // 3.00

		tokens := []int32{ts0, 0, 1, ts50, ts50, 2, 3, ts150, tokenEOT}
		result := tok.SplitSegmentsByTimestamps(tokens, 0.0, 3000, 30.0, 0)

		if len(result.segments) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(result.segments))
		}

		seg0 := result.segments[0]
		if seg0.start != 0.0 || seg0.end != 1.0 {
			t.Errorf("seg[0] time: got [%f, %f], want [0.0, 1.0]", seg0.start, seg0.end)
		}
	})

	t.Run("ConsecutiveWithSingleEnding", func(t *testing.T) {
		ts0 := tokenTimestampBeg         // 0.00
		ts50 := tokenTimestampBeg + 50   // 1.00
		ts150 := tokenTimestampBeg + 150 // 3.00

		tokens := []int32{ts0, 0, 1, ts50, ts50, 2, 3, ts150}
		result := tok.SplitSegmentsByTimestamps(tokens, 0.0, 3000, 30.0, 0)

		if len(result.segments) != 2 {
			t.Fatalf("expected 2 segments, got %d", len(result.segments))
		}

		seg0 := result.segments[0]
		if seg0.start != 0.0 || seg0.end != 1.0 {
			t.Errorf("seg[0] time: got [%f, %f], want [0.0, 1.0]", seg0.start, seg0.end)
		}

		seg1 := result.segments[1]
		if seg1.start != 1.0 || seg1.end != 3.0 {
			t.Errorf("seg[1] time: got [%f, %f], want [1.0, 3.0]", seg1.start, seg1.end)
		}

		if result.seek != 3000 {
			t.Errorf("seek: got %d, want 3000", result.seek)
		}
	})

	t.Run("SingleTimestampEndingNoConsecutive", func(t *testing.T) {
		ts50 := tokenTimestampBeg + 50

		tokens := []int32{0, 1, ts50}
		result := tok.SplitSegmentsByTimestamps(tokens, 0.0, 3000, 30.0, 0)

		if len(result.segments) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(result.segments))
		}
		if result.segments[0].end != 1.0 {
			t.Errorf("end: got %f, want 1.0", result.segments[0].end)
		}
	})

	t.Run("NoTimestamps", func(t *testing.T) {
		tokens := []int32{0, 1, 2, 3}
		result := tok.SplitSegmentsByTimestamps(tokens, 5.0, 3000, 30.0, 500)

		if len(result.segments) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(result.segments))
		}
		if result.segments[0].start != 5.0 {
			t.Errorf("start: got %f, want 5.0", result.segments[0].start)
		}
		if result.seek != 500+3000 {
			t.Errorf("seek: got %d, want %d", result.seek, 500+3000)
		}
	})

	t.Run("WithTimeOffset", func(t *testing.T) {
		ts0 := tokenTimestampBeg
		ts50 := tokenTimestampBeg + 50

		tokens := []int32{ts0, 0, ts50}
		result := tok.SplitSegmentsByTimestamps(tokens, 10.0, 3000, 30.0, 1000)

		if len(result.segments) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(result.segments))
		}
		if result.segments[0].start != 10.0 {
			t.Errorf("start: got %f, want 10.0", result.segments[0].start)
		}
		if result.segments[0].end != 11.0 {
			t.Errorf("end: got %f, want 11.0", result.segments[0].end)
		}
	})
}

func TestSuppressedTokens(t *testing.T) {
	tok := makeTestTokenizer()

	t.Run("DefaultExpansion", func(t *testing.T) {
		result := tok.SuppressedTokens([]int32{-1})

		hasSpecial := func(id int32) bool {
			for _, tok := range result {
				if tok == id {
					return true
				}
			}
			return false
		}

		for _, special := range []int32{tokenTranscribe, tokenTranslate, tokenSOT, tokenSOTprev, tokenSOTlm, tokenNoSpeech} {
			if !hasSpecial(special) {
				t.Errorf("expected special token %d to be suppressed", special)
			}
		}
	})

	t.Run("ExplicitTokens", func(t *testing.T) {
		result := tok.SuppressedTokens([]int32{42, 99})

		has42 := false
		has99 := false
		for _, id := range result {
			if id == 42 {
				has42 = true
			}
			if id == 99 {
				has99 = true
			}
		}
		if !has42 || !has99 {
			t.Error("expected explicit tokens 42 and 99 to be present")
		}
	})

	t.Run("Sorted", func(t *testing.T) {
		result := tok.SuppressedTokens([]int32{-1, 100, 50})
		for i := 1; i < len(result); i++ {
			if result[i] <= result[i-1] {
				t.Errorf("result not sorted at index %d: %d <= %d", i, result[i], result[i-1])
				break
			}
		}
	})
}

func TestGetCompressionRatio(t *testing.T) {
	t.Run("Empty", func(t *testing.T) {
		if r := getCompressionRatio(""); r != 0 {
			t.Errorf("empty: got %f, want 0", r)
		}
	})

	t.Run("Normal", func(t *testing.T) {
		r := getCompressionRatio("Hello, this is a normal sentence with some text.")
		if r <= 0 {
			t.Errorf("normal text: got %f, want > 0", r)
		}
	})

	t.Run("Repetitive", func(t *testing.T) {
		normal := getCompressionRatio("Hello, this is a normal sentence.")
		repeated := getCompressionRatio("Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello")
		if repeated <= normal {
			t.Errorf("repetitive text (%f) should have higher ratio than normal (%f)", repeated, normal)
		}
	})
}

func TestGPT2PreTokenizer(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "SimpleWords",
			input: "Hello world",
			want:  []string{"Hello", " world"},
		},
		{
			name:  "Contractions",
			input: "I'm don't they'll we've you're",
			want:  []string{"I", "'m", " don", "'t", " they", "'ll", " we", "'ve", " you", "'re"},
		},
		{
			name:  "MixedContent",
			input: "Hello, world! 123",
			want:  []string{"Hello", ",", " world", "!", " 123"},
		},
		{
			name:  "Cyrillic",
			input: "Привет мир",
			want:  []string{"Привет", " мир"},
		},
		{
			name:  "PromptLike",
			input: " Transcribe the following audio:",
			want:  []string{" Transcribe", " the", " following", " audio", ":"},
		},
		{
			name:  "PunctuationCluster",
			input: `"Hello"`,
			want:  []string{`"`, "Hello", `"`},
		},
		{
			name:  "NumbersAndLetters",
			input: "test123 456abc",
			want:  []string{"test", "123", " 456", "abc"},
		},
		{
			name:  "LeadingSpace",
			input: " hello",
			want:  []string{" hello"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gpt2PreTokenizer.FindAllString(tt.input, -1)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("gpt2PreTokenizer.FindAllString(%q) =\n  %q\nwant:\n  %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestEncodePreTokenization(t *testing.T) {
	tok := &tokenizer{
		idToToken:   make(map[int32]string),
		tokenToID:   make(map[string]int32),
		langToToken: make(map[string]int32),
		mergeRank:   make(map[string]int),
	}
	buildByteDecoderInto(tok)
	tok.byteEncoder = buildByteEncoder(tok)

	toBPE := func(raw string) string { return tok.textToBPEString(raw) }

	addToken := func(id int32, raw string) {
		bpe := toBPE(raw)
		tok.idToToken[id] = bpe
		tok.tokenToID[bpe] = id
	}

	// Build merges so that multi-rune BPE strings can be formed.
	// Each raw string is converted to its BPE rune sequence, and we add
	// successive pair merges so bpeEncode produces a single token.
	addMergesForWord := func(raw string) {
		bpe := toBPE(raw)
		runes := []rune(bpe)
		if len(runes) <= 1 {
			return
		}
		merged := string(runes[0])
		for i := 1; i < len(runes); i++ {
			next := string(runes[i])
			tok.mergeRank[merged+" "+next] = len(tok.mergeRank)
			merged = merged + next
		}
	}

	addToken(10, "I")
	addToken(11, "'m")
	addToken(12, " don")
	addToken(13, "'t")
	addToken(14, " hello")
	addToken(15, ",")
	addToken(16, " world")
	addToken(17, "!")

	for _, raw := range []string{"'m", " don", "'t", " hello", ",", " world", "!"} {
		addMergesForWord(raw)
	}

	t.Run("Contractions", func(t *testing.T) {
		ids := tok.Encode("I'm don't")
		want := []int32{10, 11, 12, 13}
		if !reflect.DeepEqual(ids, want) {
			t.Errorf("Encode(\"I'm don't\") = %v, want %v", ids, want)
		}
	})

	t.Run("MixedPunctuation", func(t *testing.T) {
		ids := tok.Encode(" hello, world!")
		want := []int32{14, 15, 16, 17}
		if !reflect.DeepEqual(ids, want) {
			t.Errorf("Encode(\" hello, world!\") = %v, want %v", ids, want)
		}
	})

	t.Run("RoundTrip", func(t *testing.T) {
		text := " hello, world!"
		ids := tok.Encode(text)
		decoded := tok.Decode(ids)
		if decoded != text {
			t.Errorf("roundtrip: Encode+Decode(%q) = %q", text, decoded)
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

func BenchmarkSplitSegmentsByTimestamps(b *testing.B) {
	tok := makeTestTokenizer()
	ts0 := tokenTimestampBeg
	ts1 := tokenTimestampBeg + 50
	ts3 := tokenTimestampBeg + 150
	ids := []int32{ts0, 0, 1, ts1, ts1, 2, 3, ts3, tokenEOT}
	b.ResetTimer()
	for b.Loop() {
		tok.SplitSegmentsByTimestamps(ids, 0.0, 3000, 30.0, 0)
	}
}
