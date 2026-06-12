package whisper

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/timohahaa/faster-whisper-go/internal/ct2bridge"
)

const maxSamples = whisperSampleRate * whisperChunkLen // 480000

// Transcribe runs speech recognition on PCM audio samples (16kHz, mono, float32).
// Output text is in the original spoken language.
// For audio longer than 30 seconds, only the first 30 seconds are processed.
func (m *Model) Transcribe(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.infer(ctx, samples, cfg, tokenTranscribe)
}

// Translate runs speech recognition and translates the result into English.
// For audio longer than 30 seconds, only the first 30 seconds are processed.
func (m *Model) Translate(ctx context.Context, samples []float32, cfg TranscribeConfig) (*Result, error) {
	return m.infer(ctx, samples, cfg, tokenTranslate)
}

func (m *Model) infer(ctx context.Context, samples []float32, cfg TranscribeConfig, taskToken int32) (*Result, error) {
	if m == nil || m.bridge == nil {
		return nil, errors.New("model is closed")
	}
	if len(samples) == 0 {
		return nil, errors.New("samples are empty")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	audio := padOrTrim(samples, maxSamples)
	mel := computeMelSpectrogram(audio, m.nMels)

	lang := cfg.Language
	if lang == "" && m.IsMultilingual() {
		detected, err := m.bridge.DetectLanguage(mel, m.nMels, whisperNFrames)
		if err != nil {
			return nil, err
		}
		lang = detected.Language
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	prompt := m.buildPrompt(lang, cfg, taskToken)

	genOpts := ct2bridge.GenerateOptions{
		BeamSize:          cfg.BeamSize,
		BestOf:            cfg.BestOf,
		Patience:          1.0,
		LengthPenalty:     1.0,
		RepetitionPenalty: 1.0,
		NoRepeatNgramSize: 0,
		MaxLength:         448,
		SuppressBlank:     true,
		ReturnScores:      true,
	}

	genResult, err := m.bridge.Generate(mel, m.nMels, whisperNFrames, prompt, genOpts)
	if err != nil {
		return nil, err
	}

	result := &Result{
		Language: lang,
	}

	if cfg.Timestamps {
		segments := m.tokenizer.DecodeSegmentTokens(genResult.SequenceIDs)
		for _, seg := range segments {
			result.Segments = append(result.Segments, Segment{
				Start: time.Duration(seg.start * float64(time.Second)),
				End:   time.Duration(seg.end * float64(time.Second)),
				Text:  strings.TrimSpace(seg.text),
			})
		}
		var textBuf strings.Builder
		for i, seg := range result.Segments {
			if i > 0 {
				textBuf.WriteByte(' ')
			}
			textBuf.WriteString(seg.Text)
		}
		result.Text = textBuf.String()
	} else {
		result.Text = strings.TrimSpace(m.tokenizer.Decode(genResult.SequenceIDs))
	}

	return result, nil
}

// DetectLanguage identifies the spoken language from the first 30 seconds of audio.
func (m *Model) DetectLanguage(ctx context.Context, samples []float32) (LanguageDetection, error) {
	if m == nil || m.bridge == nil {
		return LanguageDetection{}, errors.New("model is closed")
	}
	if len(samples) == 0 {
		return LanguageDetection{}, errors.New("samples are empty")
	}
	if err := ctx.Err(); err != nil {
		return LanguageDetection{}, err
	}

	audio := padOrTrim(samples, maxSamples)
	mel := computeMelSpectrogram(audio, m.nMels)

	result, err := m.bridge.DetectLanguage(mel, m.nMels, whisperNFrames)
	if err != nil {
		return LanguageDetection{}, err
	}
	return LanguageDetection{
		Language:    result.Language,
		Probability: result.Probability,
	}, nil
}

func (m *Model) buildPrompt(lang string, cfg TranscribeConfig, taskToken int32) []int32 {
	prompt := []int32{tokenSOT}

	if m.IsMultilingual() && lang != "" {
		langTok := m.tokenizer.LanguageToken(lang)
		if langTok >= 0 {
			prompt = append(prompt, langTok)
		}
		prompt = append(prompt, taskToken)
	}

	if !cfg.Timestamps {
		prompt = append(prompt, tokenNoTimestamps)
	}

	return prompt
}

func padOrTrim(samples []float32, length int) []float32 {
	if len(samples) >= length {
		return samples[:length]
	}
	out := make([]float32, length)
	copy(out, samples)
	return out
}
