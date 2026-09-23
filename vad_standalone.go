package whisper

// A VAD owns its own onnxruntime session and serializes inference internally
// (mutex per underlying engine); create one instance per parallel worker for
// throughput. It is safe to use alongside a loaded Model's own VAD engine — they
// share the process-wide onnxruntime environment, initialized once.
type VAD struct {
	engine vadEngine
}

// NewVAD builds a standalone VAD for the given backend: VadBackendSilero
// ("silero") or "" for the default, or VadBackendPyannote ("pyannote").
// cpuThreads sets the pyannote onnxruntime intra-op thread count (0 = autoscaled
// default); it is ignored by the silero backend.
func NewVAD(backend string, cpuThreads int) (*VAD, error) {
	engine, err := newVadEngine(backend, cpuThreads)
	if err != nil {
		return nil, err
	}
	return &VAD{engine: engine}, nil
}

// BatchedSpeechChunks reproduces the VAD + windowing that TranscribeBatched runs
// internally, returning window boundaries ready to be passed back as
// TranscribeConfig.ClipTimestamps for the SAME samples slice. It builds the
// batched VadConfig (the engine's batched defaults, MaxSpeechDurationS =
// chunkLength) and merges the detected regions into windows of at most
// chunkLength seconds via the shared mergeSpeechChunks helper.
//
// chunkLength == 0 uses the default 30s chunk length. The empty-result handling
// mirrors TranscribeBatched: a clip shorter than chunkLength with no detected
// speech becomes a single full-length chunk (so short clips are still
// transcribed); otherwise the result is nil — no speech, and the caller should
// skip transcription of this window rather than pass empty ClipTimestamps (which
// would trigger the internal VAD fallback).
func (v *VAD) BatchedSpeechChunks(samples []float32, chunkLength int) ([]SpeechChunk, error) {
	if chunkLength == 0 {
		chunkLength = whisperChunkLen
	}

	// Same config the batched path builds before calling m.vad.speechChunks
	// (see transcribe_batched.go). applyDefaults fills the rest; it is
	// idempotent, so the engine calling it again internally is harmless.
	vadCfg := v.engine.batchedDefaults()
	vadCfg.MaxSpeechDurationS = float64(chunkLength)
	vadCfg.applyDefaults()

	chunks, err := v.engine.speechChunks(samples, vadCfg)
	if err != nil {
		return nil, err
	}
	if len(chunks) == 0 {
		if float64(len(samples))/whisperSampleRate < float64(chunkLength) {
			return []SpeechChunk{{Start: 0, End: len(samples)}}, nil
		}
		return nil, nil
	}

	return mergeSpeechChunks(chunks, int(float64(chunkLength)*whisperSampleRate)), nil
}

// SpeechChunks returns the raw detected speech regions (before batched
// windowing), using the given VadConfig. This mirrors Model.SpeechTimestamps but
// without a loaded whisper Model. Most callers want BatchedSpeechChunks.
func (v *VAD) SpeechChunks(samples []float32, cfg VadConfig) ([]SpeechChunk, error) {
	return v.engine.speechChunks(samples, cfg)
}

// Close releases the VAD's onnxruntime session. The process-wide onnxruntime
// environment is not torn down (it is shared and kept alive for the process).
func (v *VAD) Close() {
	if v.engine != nil {
		v.engine.Close()
	}
}
