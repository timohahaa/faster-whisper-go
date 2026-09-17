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
func Load(path, device, computeType string, deviceIndex []int, intraThreads, interThreads int) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))

	cComputeType := C.CString(computeType)
	defer C.free(unsafe.Pointer(cComputeType))

	var devIdxPtr *C.int
	var cDevIdx []C.int
	if len(deviceIndex) > 0 {
		cDevIdx = make([]C.int, len(deviceIndex))
		for i, v := range deviceIndex {
			cDevIdx[i] = C.int(v)
		}
		devIdxPtr = &cDevIdx[0]
	}

	var cErr *C.char
	ptr := C.ct2_model_load(cPath, cDevice, cComputeType,
		devIdxPtr, C.size_t(len(deviceIndex)),
		C.int(intraThreads), C.int(interThreads),
		&cErr)
	if ptr == nil {
		err := errors.New(C.GoString(cErr))
		C.free(unsafe.Pointer(cErr))
		return nil, err
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

	var cErr *C.char
	ptr := C.ct2_encode(
		m.ptr,
		(*C.float)(unsafe.Pointer(&mel[0])),
		C.size_t(nMels),
		C.size_t(nFrames),
		&cErr,
	)
	if ptr == nil {
		err := errors.New(C.GoString(cErr))
		C.free(unsafe.Pointer(cErr))
		return nil, err
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

// BatchDetectLanguageResult holds per-item language detection results.
type BatchDetectLanguageResult struct {
	Items []DetectLanguageResult
}

// DetectLanguageBatch detects the most likely spoken language for every item of
// a batched encoder output in a single native call. The number of results
// equals the encoder output's batch dimension.
func (m *Model) DetectLanguageBatch(enc *EncoderOutput) (BatchDetectLanguageResult, error) {
	if m == nil || m.ptr == nil {
		return BatchDetectLanguageResult{}, errors.New("model is closed")
	}
	if enc == nil || enc.ptr == nil {
		return BatchDetectLanguageResult{}, errors.New("encoder output is required")
	}

	result := C.ct2_detect_language_batch(m.ptr, enc.ptr)
	defer C.ct2_batch_detect_result_free(&result)

	if result.error != nil {
		return BatchDetectLanguageResult{}, errors.New(C.GoString(result.error))
	}

	batchSize := int(result.batch_size)
	langs := unsafe.Slice(result.languages, batchSize)
	probs := unsafe.Slice(result.probabilities, batchSize)

	items := make([]DetectLanguageResult, batchSize)
	for b := 0; b < batchSize; b++ {
		lang := ""
		if langs[b] != nil {
			lang = C.GoString(langs[b])
		}
		items[b] = DetectLanguageResult{
			Language:    lang,
			Probability: float32(probs[b]),
		}
	}

	return BatchDetectLanguageResult{Items: items}, nil
}

// AlignResult holds the raw DTW alignment output from CTranslate2's align pass.
type AlignResult struct {
	TextTokenProbs []float32 // per-token probability [NumTokens]
	TextIndices    []int32   // text index for each alignment pair [NumAlignments]
	TimeIndices    []int32   // time index for each alignment pair [NumAlignments]
	NumTokens      int
}

// Align computes cross-attention alignment between text tokens and audio frames.
// Used for word-level timestamp extraction.
func (m *Model) Align(enc *EncoderOutput, startSeq, textTokens []int32, numFrames, medianFilterWidth int) (AlignResult, error) {
	if m == nil || m.ptr == nil {
		return AlignResult{}, errors.New("model is closed")
	}
	if enc == nil || enc.ptr == nil {
		return AlignResult{}, errors.New("encoder output is required")
	}
	if len(textTokens) == 0 {
		return AlignResult{}, errors.New("text tokens are required")
	}

	var startPtr *C.int32_t
	if len(startSeq) > 0 {
		startPtr = (*C.int32_t)(unsafe.Pointer(&startSeq[0]))
	}
	textPtr := (*C.int32_t)(unsafe.Pointer(&textTokens[0]))

	result := C.ct2_align(
		m.ptr,
		enc.ptr,
		startPtr,
		C.size_t(len(startSeq)),
		textPtr,
		C.size_t(len(textTokens)),
		C.size_t(numFrames),
		C.int(medianFilterWidth),
	)
	defer C.ct2_align_result_free(&result)

	if result.error != nil {
		return AlignResult{}, errors.New(C.GoString(result.error))
	}

	nTokens := int(result.num_tokens)
	nAlign := int(result.num_alignments)

	probs := make([]float32, nTokens)
	if nTokens > 0 && result.text_token_probs != nil {
		copy(probs, unsafe.Slice((*float32)(unsafe.Pointer(result.text_token_probs)), nTokens))
	}

	textIdx := make([]int32, nAlign)
	timeIdx := make([]int32, nAlign)
	if nAlign > 0 {
		if result.text_indices != nil {
			copy(textIdx, unsafe.Slice((*int32)(unsafe.Pointer(result.text_indices)), nAlign))
		}
		if result.time_indices != nil {
			copy(timeIdx, unsafe.Slice((*int32)(unsafe.Pointer(result.time_indices)), nAlign))
		}
	}

	return AlignResult{
		TextTokenProbs: probs,
		TextIndices:    textIdx,
		TimeIndices:    timeIdx,
		NumTokens:      nTokens,
	}, nil
}

// BatchAlignResult holds per-item alignment results from a batched align call.
type BatchAlignResult struct {
	Items []AlignResult
}

// AlignBatch computes cross-attention alignment for a whole batched encoder
// output in a single native call. textTokens and numFrames are per batch item
// and must match the encoder output's batch dimension. Per-item token slices
// may be empty; the corresponding result is returned with zero
// tokens/alignments.
func (m *Model) AlignBatch(
	enc *EncoderOutput,
	startSeq []int32,
	textTokens [][]int32,
	numFrames []int,
	medianFilterWidth int,
) (BatchAlignResult, error) {
	if m == nil || m.ptr == nil {
		return BatchAlignResult{}, errors.New("model is closed")
	}
	if enc == nil || enc.ptr == nil {
		return BatchAlignResult{}, errors.New("encoder output is required")
	}
	batchSize := len(textTokens)
	if batchSize == 0 {
		return BatchAlignResult{}, errors.New("text tokens are required")
	}
	if len(numFrames) != batchSize {
		return BatchAlignResult{}, errors.New("num frames count mismatch")
	}

	var startPtr *C.int32_t
	if len(startSeq) > 0 {
		startPtr = (*C.int32_t)(unsafe.Pointer(&startSeq[0]))
	}

	// Flatten per-item tokens into a single C-allocated buffer to satisfy cgo
	// pointer rules (Go pointers containing Go pointers cannot be passed to C).
	totalTokens := 0
	for _, t := range textTokens {
		totalTokens += len(t)
	}
	// Allocate at least one element so the base pointer is always valid.
	cTokensBuf := (*C.int32_t)(C.malloc(C.size_t(totalTokens+1) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
	defer C.free(unsafe.Pointer(cTokensBuf))
	cTokPtrs := (**C.int32_t)(C.malloc(C.size_t(batchSize) * C.size_t(unsafe.Sizeof(cTokensBuf))))
	defer C.free(unsafe.Pointer(cTokPtrs))
	cTokCounts := (*C.size_t)(C.malloc(C.size_t(batchSize) * C.size_t(unsafe.Sizeof(C.size_t(0)))))
	defer C.free(unsafe.Pointer(cTokCounts))
	cNumFrames := (*C.size_t)(C.malloc(C.size_t(batchSize) * C.size_t(unsafe.Sizeof(C.size_t(0)))))
	defer C.free(unsafe.Pointer(cNumFrames))

	tokPtrsSlice := unsafe.Slice(cTokPtrs, batchSize)
	tokCountsSlice := unsafe.Slice(cTokCounts, batchSize)
	numFramesSlice := unsafe.Slice(cNumFrames, batchSize)
	tokensBufSlice := unsafe.Slice(cTokensBuf, totalTokens+1)

	offset := 0
	for i, t := range textTokens {
		tokCountsSlice[i] = C.size_t(len(t))
		numFramesSlice[i] = C.size_t(numFrames[i])
		if len(t) > 0 {
			tokPtrsSlice[i] = &tokensBufSlice[offset]
			for j, tok := range t {
				tokensBufSlice[offset+j] = C.int32_t(tok)
			}
			offset += len(t)
		} else {
			tokPtrsSlice[i] = nil
		}
	}

	result := C.ct2_align_batch(
		m.ptr,
		enc.ptr,
		startPtr,
		C.size_t(len(startSeq)),
		cTokPtrs,
		cTokCounts,
		cNumFrames,
		C.size_t(batchSize),
		C.int(medianFilterWidth),
	)
	defer C.ct2_batch_align_result_free(&result)

	if result.error != nil {
		return BatchAlignResult{}, errors.New(C.GoString(result.error))
	}

	numTokens := unsafe.Slice(result.num_tokens, batchSize)
	probsPtrs := unsafe.Slice(result.text_token_probs, batchSize)
	numAligns := unsafe.Slice(result.num_alignments, batchSize)
	textIdxPtrs := unsafe.Slice(result.text_indices, batchSize)
	timeIdxPtrs := unsafe.Slice(result.time_indices, batchSize)

	items := make([]AlignResult, batchSize)
	for b := 0; b < batchSize; b++ {
		nTok := int(numTokens[b])
		nAlign := int(numAligns[b])

		probs := make([]float32, nTok)
		if nTok > 0 && probsPtrs[b] != nil {
			copy(probs, unsafe.Slice((*float32)(unsafe.Pointer(probsPtrs[b])), nTok))
		}
		textIdx := make([]int32, nAlign)
		timeIdx := make([]int32, nAlign)
		if nAlign > 0 {
			if textIdxPtrs[b] != nil {
				copy(textIdx, unsafe.Slice((*int32)(unsafe.Pointer(textIdxPtrs[b])), nAlign))
			}
			if timeIdxPtrs[b] != nil {
				copy(timeIdx, unsafe.Slice((*int32)(unsafe.Pointer(timeIdxPtrs[b])), nAlign))
			}
		}

		items[b] = AlignResult{
			TextTokenProbs: probs,
			TextIndices:    textIdx,
			TimeIndices:    timeIdx,
			NumTokens:      nTok,
		}
	}

	return BatchAlignResult{Items: items}, nil
}

// EncodeBatch runs the Whisper encoder on a batch of mel spectrograms.
// mel must be a flat buffer of shape [batchSize, nMels, nFrames].
func (m *Model) EncodeBatch(mel []float32, batchSize, nMels, nFrames int) (*EncoderOutput, error) {
	if m == nil || m.ptr == nil {
		return nil, errors.New("model is closed")
	}
	if len(mel) == 0 || batchSize == 0 {
		return nil, errors.New("mel spectrogram batch is required")
	}

	var cErr *C.char
	ptr := C.ct2_encode_batch(
		m.ptr,
		(*C.float)(unsafe.Pointer(&mel[0])),
		C.size_t(batchSize),
		C.size_t(nMels),
		C.size_t(nFrames),
		&cErr,
	)
	if ptr == nil {
		err := errors.New(C.GoString(cErr))
		C.free(unsafe.Pointer(cErr))
		return nil, err
	}

	return &EncoderOutput{ptr: ptr}, nil
}

// BatchGenerateResult holds per-item results from a batched generate call.
type BatchGenerateResult struct {
	Items []GenerateResult
}

// GenerateBatch runs Whisper decoding on a batched encoder output with one
// prompt per batch item.
func (m *Model) GenerateBatch(enc *EncoderOutput, prompts [][]int32, opts GenerateOptions) (BatchGenerateResult, error) {
	if m == nil || m.ptr == nil {
		return BatchGenerateResult{}, errors.New("model is closed")
	}
	if enc == nil || enc.ptr == nil {
		return BatchGenerateResult{}, errors.New("encoder output is required")
	}
	batchSize := len(prompts)
	if batchSize == 0 {
		return BatchGenerateResult{}, errors.New("prompts are required")
	}

	// Flatten all prompts into a single C-allocated buffer to satisfy cgo
	// pointer rules (Go pointers containing Go pointers cannot be passed to C).
	totalTokens := 0
	for _, p := range prompts {
		totalTokens += len(p)
	}

	cTokensBuf := (*C.int32_t)(C.malloc(C.size_t(totalTokens) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
	defer C.free(unsafe.Pointer(cTokensBuf))
	cPromptPtrs := (**C.int32_t)(C.malloc(C.size_t(batchSize) * C.size_t(unsafe.Sizeof(cTokensBuf))))
	defer C.free(unsafe.Pointer(cPromptPtrs))
	cPromptCounts := (*C.size_t)(C.malloc(C.size_t(batchSize) * C.size_t(unsafe.Sizeof(C.size_t(0)))))
	defer C.free(unsafe.Pointer(cPromptCounts))

	promptPtrsSlice := unsafe.Slice(cPromptPtrs, batchSize)
	promptCountsSlice := unsafe.Slice(cPromptCounts, batchSize)
	tokensBufSlice := unsafe.Slice(cTokensBuf, totalTokens)

	offset := 0
	for i, p := range prompts {
		promptCountsSlice[i] = C.size_t(len(p))
		if len(p) > 0 {
			promptPtrsSlice[i] = &tokensBufSlice[offset]
			for j, tok := range p {
				tokensBufSlice[offset+j] = C.int32_t(tok)
			}
			offset += len(p)
		}
	}

	var suppressPtr *C.int32_t
	suppressCount := len(opts.SuppressTokens)
	if suppressCount > 0 {
		suppressPtr = (*C.int32_t)(unsafe.Pointer(&opts.SuppressTokens[0]))
	}

	result := C.ct2_generate_batch(
		m.ptr,
		enc.ptr,
		cPromptPtrs,
		cPromptCounts,
		C.size_t(batchSize),
		C.int(opts.BeamSize),
		C.int(opts.BestOf),
		C.float(opts.Patience),
		C.float(opts.LengthPenalty),
		C.float(opts.RepetitionPenalty),
		C.int(opts.NoRepeatNgramSize),
		C.int(opts.MaxLength),
		C.bool(opts.SuppressBlank),
		C.float(opts.SamplingTemperature),
		suppressPtr,
		C.size_t(suppressCount),
	)
	defer C.ct2_batch_generate_result_free(&result)

	if result.error != nil {
		return BatchGenerateResult{}, errors.New(C.GoString(result.error))
	}

	items := make([]GenerateResult, batchSize)
	seqIDs := unsafe.Slice(result.sequences_ids, batchSize)
	seqCounts := unsafe.Slice(result.sequences_counts, batchSize)
	scores := unsafe.Slice(result.scores, batchSize)
	noSpeechProbs := unsafe.Slice(result.no_speech_probs, batchSize)

	for i := 0; i < batchSize; i++ {
		items[i] = GenerateResult{
			Score:        float32(scores[i]),
			NoSpeechProb: float32(noSpeechProbs[i]),
		}
		count := int(seqCounts[i])
		if count > 0 && seqIDs[i] != nil {
			items[i].SequenceIDs = cInt32Slice(seqIDs[i], count)
		}
	}

	return BatchGenerateResult{Items: items}, nil
}

func cInt32Slice(ptr *C.int32_t, count int) []int32 {
	if ptr == nil || count == 0 {
		return nil
	}
	out := make([]int32, count)
	copy(out, unsafe.Slice((*int32)(unsafe.Pointer(ptr)), count))
	return out
}
