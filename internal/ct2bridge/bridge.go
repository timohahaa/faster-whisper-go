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
	"fmt"
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

// GenerateOptions controls Whisper decoding through the C bridge.
type GenerateOptions struct {
	BeamSize          int
	BestOf            int
	Patience          float32
	LengthPenalty     float32
	RepetitionPenalty float32
	NoRepeatNgramSize int
	MaxLength         int
	SuppressBlank     bool
	ReturnScores      bool
}

// GenerateResult holds token IDs returned by the C bridge.
type GenerateResult struct {
	SequenceIDs  []int32
	Score        float32
	NoSpeechProb float32
}

// Generate runs Whisper decoding on a mel spectrogram and prompt tokens.
func (m *Model) Generate(mel []float32, nMels, nFrames int, prompt []int32, opts GenerateOptions) (GenerateResult, error) {
	if m == nil || m.ptr == nil {
		return GenerateResult{}, errors.New("model is closed")
	}
	if len(mel) == 0 {
		return GenerateResult{}, errors.New("mel spectrogram is required")
	}
	if len(prompt) == 0 {
		return GenerateResult{}, errors.New("prompt tokens are required")
	}

	var promptPtr *C.int32_t
	if len(prompt) > 0 {
		promptPtr = (*C.int32_t)(unsafe.Pointer(&prompt[0]))
	}

	result := C.ct2_generate(
		m.ptr,
		(*C.float)(unsafe.Pointer(&mel[0])),
		C.size_t(nMels),
		C.size_t(nFrames),
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

// EncodeResult holds encoder output from the C bridge.
type EncodeResult struct {
	Data  []float32
	Shape []int
}

// Encode runs the Whisper encoder on a mel spectrogram.
func (m *Model) Encode(mel []float32, nMels, nFrames int) (EncodeResult, error) {
	if m == nil || m.ptr == nil {
		return EncodeResult{}, errors.New("model is closed")
	}
	if len(mel) == 0 {
		return EncodeResult{}, errors.New("mel spectrogram is required")
	}

	result := C.ct2_encode(
		m.ptr,
		(*C.float)(unsafe.Pointer(&mel[0])),
		C.size_t(nMels),
		C.size_t(nFrames),
	)
	defer C.ct2_encode_result_free(&result)

	if result.error != nil {
		return EncodeResult{}, errors.New(C.GoString(result.error))
	}

	out := EncodeResult{
		Shape: cSizeSlice(result.shape, int(result.shape_len)),
	}
	if result.data != nil {
		count := 1
		for _, dim := range out.Shape {
			count *= dim
		}
		out.Data = cFloatSlice(result.data, count)
	}
	return out, nil
}

// DetectLanguageResult holds the top detected language.
type DetectLanguageResult struct {
	Language    string
	Probability float32
}

// DetectLanguage detects the most likely spoken language from a mel spectrogram.
func (m *Model) DetectLanguage(mel []float32, nMels, nFrames int) (DetectLanguageResult, error) {
	if m == nil || m.ptr == nil {
		return DetectLanguageResult{}, errors.New("model is closed")
	}
	if len(mel) == 0 {
		return DetectLanguageResult{}, errors.New("mel spectrogram is required")
	}

	result := C.ct2_detect_language(
		m.ptr,
		(*C.float)(unsafe.Pointer(&mel[0])),
		C.size_t(nMels),
		C.size_t(nFrames),
	)
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

func cFloatSlice(ptr *C.float, count int) []float32 {
	if ptr == nil || count == 0 {
		return nil
	}
	out := make([]float32, count)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count))
	return out
}

func cSizeSlice(ptr *C.size_t, count int) []int {
	if ptr == nil || count == 0 {
		return nil
	}
	raw := unsafe.Slice(ptr, count)
	out := make([]int, count)
	for i, value := range raw {
		out[i] = int(value)
	}
	return out
}

// String returns a short debug representation of the model handle.
func (m *Model) String() string {
	if m == nil || m.ptr == nil {
		return "ct2bridge.Model(nil)"
	}
	return fmt.Sprintf("ct2bridge.Model(%p)", m.ptr)
}
