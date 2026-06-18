package whisper

import "testing"

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

func TestHallucinationPhrasesLoaded(t *testing.T) {
	if len(hallucinationPhrases) == 0 {
		t.Fatal("embedded hallucination filter loaded no languages")
	}
	if _, ok := hallucinationPhrases["en"]; !ok {
		t.Error("expected 'en' language in hallucination filter")
	}
}
