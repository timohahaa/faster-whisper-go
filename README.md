# faster-whisper-go

Go port of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — speech-to-text on [CTranslate2](https://github.com/OpenNMT/CTranslate2).

Inference runs through a thin cgo bridge to the CTranslate2 C++ API. VAD uses [Silero VAD v6](https://github.com/snakers4/silero-vad) via onnxruntime. No Python needed.

## Features

- Sequential and batched transcription pipelines
- Word-level timestamps (cross-attention alignment + DTW)
- Silero VAD for silence filtering
- Auto-download of models from HuggingFace (or use local CTranslate2 model directories)
- Language detection, translation to English
- Hallucination phrase filtering
- CPU and CUDA

## Requirements

- Go 1.25+
- CTranslate2 shared library, discoverable via `pkg-config` (see [Building CTranslate2](#building-ctranslate2))
- onnxruntime shared library for VAD (`libonnxruntime.so` / `libonnxruntime.dylib` in system lib paths, or set `ONNXRUNTIME_SHARED_LIBRARY_PATH`)

## Quick start

```go
package main

import (
	"context"
	"fmt"
	"log"

	whisper "github.com/timohahaa/faster-whisper-go"
)

func main() {
	model, err := whisper.Load("large-v3", whisper.ModelConfig{
		Device:      "cuda",
		ComputeType: "int8",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	samples := readYourAudio() // []float32, 16 kHz mono, normalized to [-1, 1]

	result, err := model.TranscribeBatched(context.Background(), samples, whisper.TranscribeConfig{
		Language:       "en",
		VadFilter:      true,
		WordTimestamps: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, seg := range result.Segments {
		fmt.Printf("[%s -> %s] %s\n", seg.Start, seg.End, seg.Text)
	}
}
```

Audio input is `[]float32` samples at 16 kHz, mono, normalized to [-1, 1]. WAV decoding, ffmpeg, etc. is your responsibility.

## Available models

Pass a model name to `Load` — it downloads and caches automatically:

`tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`,
`large-v1`, `large-v2`, `large-v3` (alias `large`), `turbo` (alias `large-v3-turbo`),
`distil-small.en`, `distil-medium.en`, `distil-large-v2`, `distil-large-v3`

Or pass a path to a local CTranslate2 model directory.

Cache location: `$XDG_CACHE_HOME/faster-whisper-go/` (default `~/.cache/faster-whisper-go/`).

## API

### Model loading

```go
model, err := whisper.Load(modelSizeOrPath, whisper.ModelConfig{
    Device:      "cpu",    // "cpu" or "cuda"
    ComputeType: "int8",   // "int8", "float16", "float32", "default"
    DeviceIndex: []int{0}, // GPU IDs
    CPUThreads:  4,        // intra_threads per replica
    NumWorkers:  1,        // inter_threads (replicas)
})
defer model.Close()
```

### Transcription

Two pipelines:

- **`Transcribe`** — sequential sliding window, carries context between windows. Better for short/medium audio or when cross-window coherence matters.
- **`TranscribeBatched`** — splits audio via VAD, processes chunks in parallel batches. Higher throughput on long audio.

Both return `*Result` with `.Text`, `.Segments` and `.Info`.

```go
result, err := model.Transcribe(ctx, samples, whisper.TranscribeConfig{...})
result, err := model.TranscribeBatched(ctx, samples, whisper.TranscribeConfig{...})
```

### Translation

Translate audio to English. Available in both sequential and batched variants:

```go
result, err := model.Translate(ctx, samples, cfg)
result, err := model.TranslateBatched(ctx, samples, cfg)
```

### Language detection

```go
det, err := model.DetectLanguage(ctx, samples)
fmt.Println(det.Language, det.Probability) // "en" 0.98
```

### Voice Activity Detection

Each `Model` owns its own Silero VAD instance (a dedicated onnxruntime session
plus LSTM scratch state), created on `Load` and released on `Close` — a strict
1:1 relationship between a model and its VAD. This keeps VAD concurrency tied to
model concurrency: run one model per GPU/worker and its VAD is used by that
worker's transcription calls.

Detect speech regions directly with the model's VAD:

```go
chunks, err := model.SpeechTimestamps(samples, whisper.VadConfig{})
for _, c := range chunks {
    // c.Start / c.End are sample indices at 16kHz
}
```

For standalone use (without a model), create a VAD from the public `silerovad`
package. A `*silerovad.VAD` is safe for concurrent use — calls are serialized by
an internal mutex, so share one instance per model rather than per goroutine:

```go
vad, err := silerovad.New()
defer vad.Close()

chunks, err := whisper.GetSpeechTimestamps(vad, samples, whisper.VadConfig{})
```

The process-wide onnxruntime environment is initialized once on the first
`silerovad.New` / `whisper.Load` and stays alive for the lifetime of the process.

#### Streaming (real-time endpointing)

For live audio, use a `silerovad.Stream`: it detects speech start/end on the fly
(hysteresis + min speech/silence durations + prefix padding) and emits the
speech segment on `EndOfSpeech`, ready to feed a transcriber. One shared `VAD`
(session) backs many `Stream`s — create one `Stream` per source (e.g. per audio
track). A `Stream` is single-goroutine; different `Stream`s may run concurrently.

```go
vad, _ := silerovad.New()
defer vad.Close()

stream, _ := vad.NewStream(silerovad.StreamConfig{})
defer stream.Close()

// Feed 16kHz mono float32 chunks (resample upstream). On EndOfSpeech, transcribe
// the segment with the model's own VAD disabled — it's already segmented:
for chunk := range audio16k {
    events, _ := stream.Push(chunk)
    for _, e := range events {
        if e.Type == silerovad.EventEndOfSpeech {
            res, _ := model.Transcribe(ctx, e.Samples, whisper.TranscribeConfig{VadFilter: false})
            _ = res
        }
    }
}
events, _ := stream.Flush() // force EndOfSpeech when the source ends
_ = events
```

### Key config fields

`TranscribeConfig` fields are optional — zero values get sensible defaults.

| Field                        | Default             | Notes                              |
|------------------------------|---------------------|------------------------------------|
| `Language`                   | auto-detect         | `"en"`, `"ru"`, etc.               |
| `BeamSize`                   | 5                   |                                    |
| `BestOf`                     | 5                   | Candidates when sampling           |
| `Temperature`                | [0, 0.2, ..., 1.0]  | Fallback chain                     |
| `WordTimestamps`             | false               | Word-level timing via cross-attention |
| `VadFilter`                  | false               | Silero VAD preprocessing           |
| `BatchSize`                  | 8                   | Batched mode only                  |
| `InitialPrompt`              | ""                  | Context for first window           |
| `Hotwords`                   | ""                  | Hint phrases                       |
| `FilterHallucinationPhrases` | false               | Drop known hallucination segments  |

See `config.go` for the full list.

## Building CTranslate2

The package links against CTranslate2 via cgo (`pkg-config: ctranslate2`). Build scripts are in `scripts/`:

```bash
# Linux with CUDA
./scripts/build-ct2-linux-gpu.sh --cuda-version 12.9 --cuda-arch 86

# macOS (CPU, Accelerate)
./scripts/build-ct2-macos.sh
```

Both scripts clone CTranslate2 v4.8.0, build, install to `/usr/local`, and generate a `ctranslate2.pc` for pkg-config.

After installing, make sure pkg-config can find it:

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH  # Linux
```

## Building and testing

```bash
make build
make test
```

## Benchmarks

Head-to-head comparison with the reference Python `faster-whisper` on identical audio and decode parameters. See [`benchmarks/`](benchmarks/) for the full harness.

Large benchmark files are excluded from repo.

### Setup

- **Model:** `large-v3`, int8
- **GPU:** NVIDIA RTX 3060 12GB, CUDA 12.9
- **VAD:** Silero, `vad_filter=true`
- **Decode:** beam_size=5, best_of=5, temperature fallback, word timestamps

### Speed

`go/py` = Go time / Python time (< 1.00 means Go is faster).

#### Batched

| File                      | Audio  | Python  | Go      | go/py |
|---------------------------|-------:|--------:|--------:|------:|
| test.wav                  |    82s |   2.31s |   2.25s | 0.97x |
| anthropic_workshop_en.wav | 4540s  | 111.45s | 109.34s | 0.98x |
| postgres_interview_ru.wav | 7260s  | 204.00s | 199.96s | 0.98x |

#### Sequential

| File                      | Audio  | Python  | Go      | go/py |
|---------------------------|-------:|--------:|--------:|------:|
| test.wav                  |    82s |   3.54s |   3.66s | 1.04x |
| anthropic_workshop_en.wav | 4540s  | 420.94s | 521.52s | 1.24x |
| postgres_interview_ru.wav | 7260s  | 703.93s | 745.55s | 1.06x |

### Accuracy (Go vs Python reference)

#### Batched

| File                      |   WER |   CER |
|---------------------------|------:|------:|
| test.wav                  | 1.97% | 2.33% |
| anthropic_workshop_en.wav | 0.49% | 0.35% |
| postgres_interview_ru.wav | 1.04% | 0.60% |

#### Sequential

| File                      |   WER |   CER |
|---------------------------|------:|------:|
| test.wav                  | 1.27% | 0.96% |
| anthropic_workshop_en.wav | 7.51% | 5.70% |
| postgres_interview_ru.wav | 6.71% | 4.48% |

### Takeaways

- **Batched:** Go matches Python on speed (within 2-3%, slightly faster) and output quality (WER <= 2%, identical segment counts).
- **Sequential:** close on short audio; on long audio transcripts diverge more (WER ~7%) due to different segment boundaries and different FFT implementations.
