package whisper

import (
	_ "embed"
	"encoding/json"
	"strings"
)

// hallucinationFilterJSON is the embedded per-language blocklist of known
// Whisper hallucination phrases (e.g. "subtitles by ...", channel sign-offs,
// music markers). The structure is {"<lang>": ["phrase", ...], ...}.
//
//go:embed data/hallucination_filter.json
var hallucinationFilterJSON []byte

// hallucinationPhrases maps a language code to the set of lowercased phrases
// that should be treated as hallucinations and dropped from the output.
var hallucinationPhrases = mustLoadHallucinationPhrases()

func mustLoadHallucinationPhrases() map[string]map[string]struct{} {
	var raw map[string][]string
	if err := json.Unmarshal(hallucinationFilterJSON, &raw); err != nil {
		panic("whisper: invalid embedded hallucination filter: " + err.Error())
	}
	m := make(map[string]map[string]struct{}, len(raw))
	for lang, phrases := range raw {
		set := make(map[string]struct{}, len(phrases))
		for _, p := range phrases {
			set[strings.ToLower(p)] = struct{}{}
		}
		m[lang] = set
	}
	return m
}

// isHallucinationPhrase reports whether text exactly matches (case-insensitively,
// after trimming) a known hallucination phrase for the given language.
func isHallucinationPhrase(text, lang string) bool {
	set, ok := hallucinationPhrases[lang]
	if !ok {
		return false
	}
	_, hit := set[strings.ToLower(strings.TrimSpace(text))]
	return hit
}

// filterHallucinationPhrases drops segments whose full text matches a known
// hallucination phrase for lang, then renumbers the remaining segment IDs so
// they stay contiguous (1-based). The input slice is reused.
func filterHallucinationPhrases(segments []Segment, lang string) []Segment {
	out := segments[:0]
	for _, seg := range segments {
		if isHallucinationPhrase(seg.Text, lang) {
			continue
		}
		out = append(out, seg)
	}
	for i := range out {
		out[i].ID = i + 1
	}
	return out
}
