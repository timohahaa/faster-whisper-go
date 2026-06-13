package whisper

import (
	"math"
	"testing"
)

func BenchmarkGetSpeechProbs(b *testing.B) {
	b.Run("5s", func(b *testing.B) {
		samples := make([]float32, whisperSampleRate*5)
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(whisperSampleRate)))
		}
		b.ResetTimer()
		for b.Loop() {
			getSpeechProbs(samples)
		}
	})
}

func BenchmarkCollectChunks(b *testing.B) {
	samples := make([]float32, whisperSampleRate*30)
	chunks := []SpeechChunk{
		{Start: 0, End: 16000},
		{Start: 32000, End: 80000},
		{Start: 160000, End: 320000},
		{Start: 400000, End: 480000},
	}
	b.ResetTimer()
	for b.Loop() {
		collectChunks(samples, chunks)
	}
}

func BenchmarkEncode(b *testing.B) {
	tok := makeTestTokenizerForBench()

	b.Run("Short", func(b *testing.B) {
		for b.Loop() {
			tok.Encode(" hello, world!")
		}
	})

	b.Run("Long", func(b *testing.B) {
		text := " This is a longer sentence that would be typical of transcription output from a whisper model."
		for b.Loop() {
			tok.Encode(text)
		}
	})
}

func BenchmarkBPEEncode(b *testing.B) {
	tok := makeTestTokenizerForBench()
	bpeWord := tok.textToBPEString(" hello")
	b.ResetTimer()
	for b.Loop() {
		tok.bpeEncode(bpeWord)
	}
}

func BenchmarkGetCompressionRatio(b *testing.B) {
	b.Run("Short", func(b *testing.B) {
		text := "Hello, this is a normal sentence."
		for b.Loop() {
			getCompressionRatio(text)
		}
	})

	b.Run("Typical", func(b *testing.B) {
		text := "This is a longer piece of text that would be typical of what comes out of a Whisper transcription segment, perhaps about 200 characters or so in length."
		for b.Loop() {
			getCompressionRatio(text)
		}
	})
}

func BenchmarkFilterTextTokens(b *testing.B) {
	tokens := []int32{
		tokenTimestampBeg, 100, 200, 300, 400, 500, 600, 700, 800,
		tokenTimestampBeg + 50, tokenEOT,
	}
	b.ResetTimer()
	for b.Loop() {
		filterTextTokens(tokens)
	}
}

func BenchmarkExtractMelWindow(b *testing.B) {
	nMels := 80
	totalFrames := 3000
	mel := make([]float32, nMels*totalFrames)
	for i := range mel {
		mel[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for b.Loop() {
		extractMelWindow(mel, totalFrames, nMels, 0, whisperNFrames)
	}
}

func makeTestTokenizerForBench() *tokenizer {
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

	words := []struct {
		id  int32
		raw string
	}{
		{10, "I"}, {11, "'m"}, {12, " don"}, {13, "'t"},
		{14, " hello"}, {15, ","}, {16, " world"}, {17, "!"},
		{18, " This"}, {19, " is"}, {20, " a"}, {21, " longer"},
		{22, " sentence"}, {23, " that"}, {24, " would"}, {25, " be"},
		{26, " typical"}, {27, " of"}, {28, " transcription"},
		{29, " output"}, {30, " from"}, {31, " whisper"}, {32, " model"},
		{33, "."}, {34, " the"}, {35, " following"}, {36, " audio"},
		{37, ":"}, {38, " The"}, {39, " quick"}, {40, " brown"},
		{41, " fox"}, {42, " jumps"}, {43, " over"}, {44, " lazy"},
		{45, " dog"},
	}

	for _, w := range words {
		addToken(w.id, w.raw)
		addMergesForWord(w.raw)
	}

	return tok
}
