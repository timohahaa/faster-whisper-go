package ct2bridge

/*
#cgo pkg-config: ctranslate2
#cgo CXXFLAGS: -std=c++17
#include "bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Model wraps a loaded CTranslate2 Whisper model.
type Model struct {
	ptr *C.ct2_model
}

// Load opens a Whisper model from a CTranslate2 model directory.
func Load(path, device, computeType string) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))

	cComputeType := C.CString(computeType)
	defer C.free(unsafe.Pointer(cComputeType))

	ptr := C.ct2_model_load(cPath, cDevice, cComputeType)
	if ptr == nil {
		return nil, errors.New(C.GoString(C.ct2_last_error()))
	}

	return &Model{ptr: ptr}, nil
}

// Close releases model resources.
func (m *Model) Close() {
	if m == nil || m.ptr == nil {
		return
	}
	C.ct2_model_free(m.ptr)
	m.ptr = nil
}

// IsMultilingual reports whether the model supports multiple languages.
func (m *Model) IsMultilingual() bool {
	if m == nil || m.ptr == nil {
		return false
	}
	return bool(C.ct2_model_is_multilingual(m.ptr))
}

// NMels returns the number of mel bins expected by the model.
func (m *Model) NMels() int {
	if m == nil || m.ptr == nil {
		return 0
	}
	return int(C.ct2_model_n_mels(m.ptr))
}

// EncoderOutput holds the encoder output from a single encode pass.
// Must be freed after use.
type EncoderOutput struct {
	ptr *C.ct2_encoder_output
}

// Free releases the encoder output resources.
func (e *EncoderOutput) Free() {
	if e == nil || e.ptr == nil {
		return
	}
	C.ct2_encoder_output_free(e.ptr)
	e.ptr = nil
}

// Encode runs the Whisper encoder on a mel spectrogram window.
func (m *Model) Encode(mel []float32, nMels, nFrames int) (*EncoderOutput, error) {
	if m == nil || m.ptr == nil {
		return nil, errors.New("model is closed")
	}
	if len(mel) == 0 {
		return nil, errors.New("mel spectrogram is required")
	}

	ptr := C.ct2_encode(
		m.ptr,
		(*C.float)(unsafe.Pointer(&mel[0])),
		C.size_t(nMels),
		C.size_t(nFrames),
	)
	if ptr == nil {
		return nil, errors.New(C.GoString(C.ct2_last_error()))
	}

	return &EncoderOutput{ptr: ptr}, nil
}

// GenerateOptions controls Whisper decoding through the C bridge.
type GenerateOptions struct {
	BeamSize                 int
	BestOf                   int
	Patience                 float32
	LengthPenalty            float32
	RepetitionPenalty        float32
	NoRepeatNgramSize        int
	MaxLength                int
	SuppressBlank            bool
	ReturnScores             bool
	SamplingTemperature      float32
	SuppressTokens           []int32
	MaxInitialTimestampIndex int
}

// GenerateResult holds token IDs returned by the C bridge.
type GenerateResult struct {
	SequenceIDs  []int32
	Score        float32
	NoSpeechProb float32
}

// Generate runs Whisper decoding on a previously computed encoder output.
func (m *Model) Generate(enc *EncoderOutput, prompt []int32, opts GenerateOptions) (GenerateResult, error) {
	if m == nil || m.ptr == nil {
		return GenerateResult{}, errors.New("model is closed")
	}
	if enc == nil || enc.ptr == nil {
		return GenerateResult{}, errors.New("encoder output is required")
	}
	if len(prompt) == 0 {
		return GenerateResult{}, errors.New("prompt tokens are required")
	}

	promptPtr := (*C.int32_t)(unsafe.Pointer(&prompt[0]))

	var suppressPtr *C.int32_t
	suppressCount := len(opts.SuppressTokens)
	if suppressCount > 0 {
		suppressPtr = (*C.int32_t)(unsafe.Pointer(&opts.SuppressTokens[0]))
	}

	result := C.ct2_generate(
		m.ptr,
		enc.ptr,
		promptPtr,
		C.size_t(len(prompt)),
		C.int(opts.BeamSize),
		C.int(opts.BestOf),
		C.float(opts.Patience),
		C.float(opts.LengthPenalty),
		C.float(opts.RepetitionPenalty),
		C.int(opts.NoRepeatNgramSize),
		C.int(opts.MaxLength),
		C.bool(opts.SuppressBlank),
		C.bool(opts.ReturnScores),
		C.float(opts.SamplingTemperature),
		suppressPtr,
		C.size_t(suppressCount),
		C.int(opts.MaxInitialTimestampIndex),
	)
	defer C.ct2_generate_result_free(&result)

	if result.error != nil {
		return GenerateResult{}, errors.New(C.GoString(result.error))
	}

	out := GenerateResult{
		Score:        float32(result.score),
		NoSpeechProb: float32(result.no_speech_prob),
	}
	if result.sequences_count > 0 {
		out.SequenceIDs = cInt32Slice(result.sequences_ids, int(result.sequences_count))
	}
	return out, nil
}

// DetectLanguageResult holds the top detected language.
type DetectLanguageResult struct {
	Language    string
	Probability float32
}

// DetectLanguage detects the most likely spoken language from encoder output.
func (m *Model) DetectLanguage(enc *EncoderOutput) (DetectLanguageResult, error) {
	if m == nil || m.ptr == nil {
		return DetectLanguageResult{}, errors.New("model is closed")
	}
	if enc == nil || enc.ptr == nil {
		return DetectLanguageResult{}, errors.New("encoder output is required")
	}

	result := C.ct2_detect_language(m.ptr, enc.ptr)
	defer C.ct2_detect_result_free(&result)

	if result.error != nil {
		return DetectLanguageResult{}, errors.New(C.GoString(result.error))
	}

	return DetectLanguageResult{
		Language:    C.GoString(result.language),
		Probability: float32(result.probability),
	}, nil
}

func cInt32Slice(ptr *C.int32_t, count int) []int32 {
	if ptr == nil || count == 0 {
		return nil
	}
	out := make([]int32, count)
	copy(out, unsafe.Slice((*int32)(unsafe.Pointer(ptr)), count))
	return out
}
