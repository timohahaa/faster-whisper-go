package whisper

import (
	_ "embed"
	"encoding/json"
	"slices"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// hallucinationFilterJSON is the embedded per-language blocklist of known
// Whisper hallucination phrases (e.g. "subtitles by ...", channel sign-offs,
// music markers). The structure is {"<lang>": ["phrase", ...], ...}.
//
//go:embed data/hallucination_filter.json
var hallucinationFilterJSON []byte

// hallucinationPhrases returns the hallucination phrases by language code,
// loading them on first use.
var hallucinationPhrases = sync.OnceValue(mustLoadHallucinationPhrases)

// Phrases of at least this many words are removed wherever they occur.
// Shorter ones are removed only as a whole segment, a standalone sentence or
// a repeat, so that a real word matching a short phrase is kept.
const minAnywherePhraseWords = 3

// phraseSet holds the hallucination phrases of one language.
type phraseSet struct {
	// wordless holds lowercased, trimmed phrases that have no words (e.g.
	// "!.."); they can only match a whole text exactly.
	wordless map[string]struct{}
	// byFirstWord holds normalized phrases indexed by their first word,
	// longest first.
	byFirstWord map[string][][]string
}

func mustLoadHallucinationPhrases() map[string]*phraseSet {
	var raw map[string][]string
	if err := json.Unmarshal(hallucinationFilterJSON, &raw); err != nil {
		panic("whisper: invalid embedded hallucination filter: " + err.Error())
	}
	sets := make(map[string]*phraseSet, len(raw))
	for lang, phrases := range raw {
		set := &phraseSet{
			wordless:    map[string]struct{}{},
			byFirstWord: make(map[string][][]string, len(phrases)),
		}
		for _, phrase := range phrases {
			words := normalizePhraseWords(phrase)
			if len(words) == 0 {
				set.wordless[strings.ToLower(strings.TrimSpace(phrase))] = struct{}{}
				continue
			}
			set.byFirstWord[words[0]] = append(set.byFirstWord[words[0]], words)
		}
		for _, candidates := range set.byFirstWord {
			sort.SliceStable(candidates, func(i, j int) bool { return len(candidates[i]) > len(candidates[j]) })
		}
		sets[lang] = set
	}
	return sets
}

// normalizePhraseWords lowercases text, maps "ё" to "е" and splits it into
// words, dropping punctuation. phraseScratch.tokenize must stay equivalent.
func normalizePhraseWords(text string) []string {
	text = strings.ReplaceAll(strings.ToLower(text), "ё", "е")
	return strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}

// isHallucinationPhrase reports whether the whole text is a known
// hallucination phrase for lang, ignoring case and punctuation.
func isHallucinationPhrase(text, lang string) bool {
	set, ok := hallucinationPhrases()[lang]
	if !ok {
		return false
	}
	words := normalizePhraseWords(text)
	if len(words) == 0 {
		_, hit := set.wordless[strings.ToLower(strings.TrimSpace(text))]
		return hit
	}
	for _, candidate := range set.byFirstWord[words[0]] {
		if slices.Equal(candidate, words) {
			return true
		}
	}
	return false
}

// wordSpan is one normalized word: text[start:end] of its scratch, found in
// unit number unit.
type wordSpan struct {
	start, end, unit int
}

// phraseMatch is one occurrence of a phrase: its words are words[first:after]
// of the scratch, covering units firstUnit..lastUnit.
type phraseMatch struct {
	phrase              []string
	first, after        int
	firstUnit, lastUnit int
}

// phraseScratch holds the reusable buffers of one keepMask call.
type phraseScratch struct {
	text      []byte
	words     []wordSpan
	unitWords []int
	matches   []phraseMatch
}

var phraseScratchPool = sync.Pool{New: func() any { return new(phraseScratch) }}

// tokenize splits units into normalized words the same way as
// normalizePhraseWords, writing them into the scratch buffers.
func (s *phraseScratch) tokenize(count int, unit func(int) string) {
	s.text, s.words, s.matches = s.text[:0], s.words[:0], s.matches[:0]
	s.unitWords = slices.Grow(s.unitWords[:0], count)[:count]
	for index := range count {
		start := -1
		for _, r := range unit(index) {
			if unicode.IsLetter(r) || unicode.IsNumber(r) {
				if start < 0 {
					start = len(s.text)
				}
				r = unicode.ToLower(r)
				if r == 'ё' {
					r = 'е'
				}
				s.text = utf8.AppendRune(s.text, r)
				continue
			}
			if start >= 0 {
				s.words = append(s.words, wordSpan{start: start, end: len(s.text), unit: index})
				start = -1
			}
		}
		if start >= 0 {
			s.words = append(s.words, wordSpan{start: start, end: len(s.text), unit: index})
		}
	}
	clear(s.unitWords)
	for _, word := range s.words {
		s.unitWords[word.unit]++
	}
}

// matchAt returns the longest phrase starting at word index that starts and
// ends on unit boundaries.
func (s *phraseScratch) matchAt(set *phraseSet, index int) []string {
	if index > 0 && s.words[index-1].unit == s.words[index].unit {
		return nil
	}
	candidates := set.byFirstWord[string(s.text[s.words[index].start:s.words[index].end])]
	for _, candidate := range candidates {
		after := index + len(candidate)
		if after > len(s.words) {
			continue
		}
		if after < len(s.words) && s.words[after].unit == s.words[after-1].unit {
			continue
		}
		matched := true
		for offset := 1; offset < len(candidate); offset++ {
			word := s.words[index+offset]
			if string(s.text[word.start:word.end]) != candidate[offset] {
				matched = false
				break
			}
		}
		if matched {
			return candidate
		}
	}
	return nil
}

// keepMask decides which of count units (words as transcribed, see unit)
// survive phrase removal. A phrase occurrence must start and end on unit
// boundaries. It is removed when phrases cover the whole segment, when the
// phrase has at least minAnywherePhraseWords words, or, for two-word phrases,
// when it is repeated back to back or stands apart from the text around it
// (see standsApart). Units without words follow the preceding unit.
//
// keep is nil when nothing is removed; hasWords reports whether any kept unit
// has words.
func (set *phraseSet) keepMask(count int, unit func(int) string) (keep []bool, hasWords bool) {
	scratch := phraseScratchPool.Get().(*phraseScratch)
	defer phraseScratchPool.Put(scratch)
	scratch.tokenize(count, unit)
	if len(scratch.words) == 0 {
		return nil, false
	}

	covered := 0
	for index := 0; index < len(scratch.words); {
		phrase := scratch.matchAt(set, index)
		if phrase == nil {
			index++
			continue
		}
		after := index + len(phrase)
		scratch.matches = append(scratch.matches, phraseMatch{
			phrase:    phrase,
			first:     index,
			after:     after,
			firstUnit: scratch.words[index].unit,
			lastUnit:  scratch.words[after-1].unit,
		})
		covered += len(phrase)
		index = after
	}
	if len(scratch.matches) == 0 {
		return nil, true
	}
	if covered == len(scratch.words) {
		return make([]bool, count), false
	}

	keep = make([]bool, count)
	for index := range keep {
		keep[index] = true
	}
	removed := false
	for index, match := range scratch.matches {
		remove := len(match.phrase) >= minAnywherePhraseWords
		if !remove && len(match.phrase) == 2 {
			previous, next := index-1, index+1
			repeated := previous >= 0 && scratch.matches[previous].after == match.first &&
				slices.Equal(scratch.matches[previous].phrase, match.phrase) ||
				next < len(scratch.matches) && scratch.matches[next].first == match.after &&
					slices.Equal(scratch.matches[next].phrase, match.phrase)
			remove = repeated || scratch.standsApart(count, unit, match.firstUnit, match.lastUnit)
		}
		if remove {
			removed = true
			for position := match.firstUnit; position <= match.lastUnit; position++ {
				keep[position] = false
			}
		}
	}
	if !removed {
		return nil, true
	}

	for index := range count {
		if scratch.unitWords[index] == 0 && index > 0 {
			keep[index] = keep[index-1]
		}
		if keep[index] && scratch.unitWords[index] > 0 {
			hasWords = true
		}
	}
	return keep, hasWords
}

// standsApart reports whether units first..last are cut off from the text
// around them. On the left: the text starts there, the previous unit ends a
// sentence, or the phrase is capitalized mid-sentence. On the right: the text
// ends there, the phrase ends a sentence (possibly in a following
// punctuation-only unit), or the next word is capitalized.
func (s *phraseScratch) standsApart(count int, unit func(int) string, first, last int) bool {
	startsApart := first == 0 || endsSentence(unit(first-1)) || startsUpper(unit(first))
	if !startsApart {
		return false
	}
	if last == count-1 || endsSentence(unit(last)) {
		return true
	}
	for next := last + 1; next < count; next++ {
		if s.unitWords[next] > 0 {
			return startsUpper(unit(next))
		}
		if endsSentence(unit(next)) {
			return true
		}
	}
	return true
}

func endsSentence(unit string) bool {
	unit = strings.TrimRight(strings.TrimSpace(unit), `"'»”)]`)
	return strings.HasSuffix(unit, ".") || strings.HasSuffix(unit, "!") ||
		strings.HasSuffix(unit, "?") || strings.HasSuffix(unit, "…")
}

// startsUpper reports whether the first letter of unit is uppercase.
func startsUpper(unit string) bool {
	for _, r := range unit {
		if unicode.IsLetter(r) {
			return unicode.IsUpper(r)
		}
	}
	return false
}

// cleanSegment removes hallucination phrases from seg. With word timestamps
// it removes the words and rebuilds Text and the segment bounds from the rest;
// otherwise it works on the whitespace-separated words of Text. It returns
// false when the segment should be dropped.
func (set *phraseSet) cleanSegment(seg Segment) (Segment, bool) {
	if len(seg.Words) > 0 {
		keep, hasWords := set.keepMask(len(seg.Words), func(index int) string { return seg.Words[index].Word })
		if keep == nil {
			return seg, hasWords || !set.isWordless(seg.Text)
		}
		if !hasWords {
			return seg, false
		}
		kept := make([]Word, 0, len(seg.Words))
		var text strings.Builder
		text.Grow(len(seg.Text))
		for index, word := range seg.Words {
			if !keep[index] {
				continue
			}
			if len(kept) > 0 {
				text.WriteByte(' ')
			}
			text.WriteString(strings.TrimSpace(word.Word))
			kept = append(kept, word)
		}
		if !keep[0] {
			seg.Start = kept[0].Start
		}
		if !keep[len(keep)-1] {
			seg.End = kept[len(kept)-1].End
		}
		seg.Words = kept
		seg.Text = text.String()
		return seg, true
	}

	units := strings.Fields(seg.Text)
	keep, hasWords := set.keepMask(len(units), func(index int) string { return units[index] })
	if keep == nil {
		return seg, hasWords || !set.isWordless(seg.Text)
	}
	if !hasWords {
		return seg, false
	}
	var text strings.Builder
	text.Grow(len(seg.Text))
	for index, unit := range units {
		if !keep[index] {
			continue
		}
		if text.Len() > 0 {
			text.WriteByte(' ')
		}
		text.WriteString(unit)
	}
	seg.Text = text.String()
	return seg, true
}

func (set *phraseSet) isWordless(text string) bool {
	if len(set.wordless) == 0 {
		return false
	}
	_, hit := set.wordless[strings.ToLower(strings.TrimSpace(text))]
	return hit
}

// filterHallucinationPhrases removes known hallucination phrases for lang
// from the segments (see keepMask), drops segments left empty and renumbers
// the rest so IDs stay contiguous (1-based). The input slice is reused.
func filterHallucinationPhrases(segments []Segment, lang string) []Segment {
	set, ok := hallucinationPhrases()[lang]
	out := segments[:0]
	for _, seg := range segments {
		if ok {
			var keep bool
			if seg, keep = set.cleanSegment(seg); !keep {
				continue
			}
		}
		out = append(out, seg)
	}
	for i := range out {
		out[i].ID = i + 1
	}
	return out
}
