package whisper

import (
	"strings"
	"testing"
	"time"
)

func TestIsHallucinationPhrase(t *testing.T) {
	tests := []struct {
		name string
		text string
		lang string
		want bool
	}{
		{"ExactEN", "*Amara.org*", "en", true},
		{"CaseInsensitive", "*AMARA.ORG*", "en", true},
		{"Trimmed", "  [www.mooji.org]  ", "en", true},
		{"RU", "[музыка]", "ru", true},
		{"WrongLang", "*Amara.org*", "ru", false},
		{"UnknownLang", "*Amara.org*", "zz", false},
		{"NotAHallucination", "hello world", "en", false},
		{"PartialNoMatch", "see *Amara.org* now", "en", false},
		{"IgnoresPunctuation", "продолжение следует", "ru", true},
		{"YoAsYe", "(веселая музыка)", "ru", true},
		{"RepeatIsNotWhole", "Продолжение следует... Продолжение следует...", "ru", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isHallucinationPhrase(tt.text, tt.lang); got != tt.want {
				t.Errorf("isHallucinationPhrase(%q, %q) = %v, want %v", tt.text, tt.lang, got, tt.want)
			}
		})
	}
}

func TestFilterHallucinationPhrases(t *testing.T) {
	segments := []Segment{
		{ID: 1, Text: "Hello there."},
		{ID: 2, Text: "[музыка]"},
		{ID: 3, Text: "How are you?"},
		{ID: 4, Text: "(аплодисменты)"},
	}

	out := filterHallucinationPhrases(segments, "ru")

	if len(out) != 2 {
		t.Fatalf("len = %d, want 2", len(out))
	}
	if out[0].Text != "Hello there." || out[1].Text != "How are you?" {
		t.Errorf("unexpected kept segments: %q, %q", out[0].Text, out[1].Text)
	}
	if out[0].ID != 1 || out[1].ID != 2 {
		t.Errorf("IDs not renumbered contiguously: got %d, %d", out[0].ID, out[1].ID)
	}
}

// wordSegment builds a segment of one-second words from space-separated text.
func wordSegment(text string) Segment {
	var words []Word
	for index, word := range strings.Fields(text) {
		words = append(words, Word{
			Start: time.Duration(index) * time.Second,
			End:   time.Duration(index+1) * time.Second,
			Word:  word,
		})
	}
	return Segment{Text: text, Start: 0, End: time.Duration(len(words)) * time.Second, Words: words}
}

func TestFilterHallucinationPhrasesInText(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		wantText  string // "" means the segment is dropped
		wantStart time.Duration
		wantEnd   time.Duration
	}{
		{
			name: "RepeatedLongPhraseDropsSegment",
			text: "Субтитры создавал DimaTorzok Субтитры создавал DimaTorzok",
		},
		{
			name:     "LongPhraseInsideText",
			text:     "Спасибо. Субтитры сделал DimaTorzok Пока.",
			wantText: "Спасибо. Пока.", wantStart: 0, wantEnd: 5 * time.Second,
		},
		{
			name:     "RepeatedShortPhraseAtEnd",
			text:     "Итак, продолжаем. Продолжение следует... Продолжение следует...",
			wantText: "Итак, продолжаем.", wantStart: 0, wantEnd: 2 * time.Second,
		},
		{
			name:     "ShortPhraseAsSentenceAtEnd",
			text:     "Мы увидим свет. Продолжение следует...",
			wantText: "Мы увидим свет.", wantStart: 0, wantEnd: 3 * time.Second,
		},
		{
			name:     "ShortPhraseAsSentenceAtStart",
			text:     "Продолжение следует... Итак, начнем.",
			wantText: "Итак, начнем.", wantStart: 2 * time.Second, wantEnd: 4 * time.Second,
		},
		{
			name:     "ShortPhraseInsideSentenceKept",
			text:     "И продолжение следует завтра.",
			wantText: "И продолжение следует завтра.", wantStart: 0, wantEnd: 4 * time.Second,
		},
		{
			name:     "ShortPhraseCapitalizedMidSentence",
			text:     "Это проблема же в шаге, Продолжение следует... по-моему, да?",
			wantText: "Это проблема же в шаге, по-моему, да?", wantStart: 0, wantEnd: 9 * time.Second,
		},
		{
			name:     "ShortPhraseBeforeNewSentence",
			text:     "Босфор. Музыка Музыка Какой год?",
			wantText: "Босфор. Какой год?", wantStart: 0, wantEnd: 5 * time.Second,
		},
		{
			name:     "ShortPhraseStartingSentenceKept",
			text:     "Продолжение следует завтра.",
			wantText: "Продолжение следует завтра.", wantStart: 0, wantEnd: 3 * time.Second,
		},
		{
			name:     "SingleWordPhraseInsideTextKept",
			text:     "Мне нравится музыка.",
			wantText: "Мне нравится музыка.", wantStart: 0, wantEnd: 3 * time.Second,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := filterHallucinationPhrases([]Segment{wordSegment(tt.text)}, "ru")
			if tt.wantText == "" {
				if len(out) != 0 {
					t.Fatalf("expected segment to be dropped, got %q", out[0].Text)
				}
				return
			}
			if len(out) != 1 {
				t.Fatalf("expected one segment, got %d", len(out))
			}
			got := out[0]
			if got.Text != tt.wantText {
				t.Errorf("Text = %q, want %q", got.Text, tt.wantText)
			}
			var joined []string
			for _, word := range got.Words {
				joined = append(joined, word.Word)
			}
			if strings.Join(joined, " ") != tt.wantText {
				t.Errorf("Words = %q, want %q", strings.Join(joined, " "), tt.wantText)
			}
			if got.Start != tt.wantStart || got.End != tt.wantEnd {
				t.Errorf("bounds = %v-%v, want %v-%v", got.Start, got.End, tt.wantStart, tt.wantEnd)
			}
		})
	}
}

func TestFilterHallucinationPhrasesWithoutWords(t *testing.T) {
	segments := []Segment{
		{ID: 1, Text: "Продолжение следует... Продолжение следует..."},
		{ID: 2, Text: "Спасибо. Субтитры сделал DimaTorzok"},
	}
	out := filterHallucinationPhrases(segments, "ru")
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	if out[0].Text != "Спасибо." || out[0].ID != 1 {
		t.Errorf("got ID %d %q, want ID 1 %q", out[0].ID, out[0].Text, "Спасибо.")
	}
}

func TestPhraseScratchTokenizeMatchesNormalize(t *testing.T) {
	texts := []string{
		"Продолжение следует...",
		"ЁЖИК, ёлка и «Её» — 5G-тавры!",
		"Don't STOP 3.14 me-now",
		"日本語のテキスト、字幕",
		"émoji 🎵 Ünïcödé",
		"  ...  ",
	}
	for _, text := range texts {
		units := strings.Fields(text)
		var scratch phraseScratch
		scratch.tokenize(len(units), func(index int) string { return units[index] })
		var got []string
		for _, word := range scratch.words {
			got = append(got, string(scratch.text[word.start:word.end]))
		}
		want := normalizePhraseWords(text)
		if strings.Join(got, "|") != strings.Join(want, "|") {
			t.Errorf("tokenize(%q) = %q, want %q", text, got, want)
		}
	}
}

func TestFilterHallucinationPhrasesWordless(t *testing.T) {
	segments := []Segment{
		{ID: 1, Text: "!.."},
		{ID: 2, Text: "..."},
	}
	out := filterHallucinationPhrases(segments, "af")
	if len(out) != 1 || out[0].Text != "..." {
		t.Fatalf("got %+v, want only %q kept", out, "...")
	}
}

func TestHallucinationPhrasesLoaded(t *testing.T) {
	if len(hallucinationPhrases()) == 0 {
		t.Fatal("embedded hallucination filter loaded no languages")
	}
	if _, ok := hallucinationPhrases()["en"]; !ok {
		t.Error("expected 'en' language in hallucination filter")
	}
}
