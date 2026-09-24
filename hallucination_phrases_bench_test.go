package whisper

import (
	"strings"
	"testing"
	"time"
)

const benchSpeech = "Так, ну что, поехали. Внимание в центр головы. Найдем сейчас какую-нибудь " +
	"свою любимую болячку, которая давно с вами. Чувствуем свое физическое тело целиком, " +
	"дышим спокойно и ровно. Вся вселенная состоит из энергии, и я чувствую эту энергию. " +
	"Делаем шаг вперед, принимаем себя таким, какой есть, и ничего не отвергаем. "

// benchWords returns n words of ordinary Russian speech with one-second timings.
func benchWords(n int) []Word {
	source := strings.Fields(benchSpeech)
	words := make([]Word, n)
	for index := range words {
		words[index] = Word{
			Start: time.Duration(index) * time.Second,
			End:   time.Duration(index+1) * time.Second,
			Word:  source[index%len(source)],
		}
	}
	return words
}

// benchSegment builds a segment from words with the given text inserted at
// word position insertAt (-1 means no insertion).
func benchSegment(n int, insert string, insertAt int) Segment {
	words := benchWords(n)
	if insertAt >= 0 {
		var inserted []Word
		for _, text := range strings.Fields(insert) {
			inserted = append(inserted, Word{Word: text})
		}
		words = append(words[:insertAt], append(inserted, words[insertAt:]...)...)
	}
	texts := make([]string, len(words))
	for index, word := range words {
		texts[index] = word.Word
	}
	return Segment{Text: strings.Join(texts, " "), Words: words, End: time.Duration(len(words)) * time.Second}
}

func benchFilter(b *testing.B, template []Segment) {
	b.Helper()
	buffer := make([]Segment, len(template))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(buffer, template)
		filterHallucinationPhrases(buffer, "ru")
	}
}

func BenchmarkFilterHallucinationPhrases(b *testing.B) {
	b.Run("clean_window_80w", func(b *testing.B) {
		benchFilter(b, []Segment{benchSegment(80, "", -1)})
	})
	b.Run("phrase_in_middle", func(b *testing.B) {
		benchFilter(b, []Segment{benchSegment(80, "Субтитры сделал DimaTorzok", 40)})
	})
	b.Run("repeat_at_end", func(b *testing.B) {
		benchFilter(b, []Segment{benchSegment(80, "Продолжение следует... Продолжение следует...", 80)})
	})
	b.Run("whole_repeat", func(b *testing.B) {
		benchFilter(b, []Segment{benchSegment(0, "Субтитры создавал DimaTorzok Субтитры создавал DimaTorzok", 0)})
	})
	b.Run("hour_120_windows", func(b *testing.B) {
		segments := make([]Segment, 120)
		for index := range segments {
			segments[index] = benchSegment(80, "", -1)
		}
		segments[10] = benchSegment(80, "Субтитры сделал DimaTorzok", 40)
		segments[60] = benchSegment(80, "Продолжение следует...", 80)
		segments[119] = benchSegment(0, "Продолжение следует... Продолжение следует...", 0)
		benchFilter(b, segments)
	})
}

func BenchmarkNormalizePhraseWords(b *testing.B) {
	text := benchSegment(80, "", -1).Text
	b.ReportAllocs()
	for b.Loop() {
		normalizePhraseWords(text)
	}
}

func BenchmarkLoadHallucinationPhrases(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		mustLoadHallucinationPhrases()
	}
}
