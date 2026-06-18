# Benchmarks: Python `faster-whisper` vs Go port

This directory compares the Go port against the reference Python
`faster-whisper` implementation on the audio files in `../.testdata`, measuring
speed and verifying transcript correctness.

Both sides:

- read the same canonical WAV files with the same `int16 / 32768` conversion, so
  they transcribe identical samples (see below);
- read all parameters from the same `config.json` and pass every decode and VAD
  option explicitly, instead of relying on each library's defaults;
- emit the same JSON schema, so the outputs diff directly.

Two pipelines are exercised:

- batched: `BatchedInferencePipeline` / `Model.TranscribeBatched`
- sequential: `WhisperModel.transcribe` / `Model.Transcribe`

## Layout

| File | Purpose |
|------|---------|
| `config.json` | shared run parameters (model, device, decode + VAD options, files) |
| `python/bench.py` | runs the Python baseline, writes `results/python_<mode>.json` |
| `go/bench.go` | runs the Go port, writes `results/go_<mode>.json` |
| `compare.py` | diffs speed + accuracy (WER/CER) between the two |
| `run.sh` | sets everything up and runs all of the above |

`results/` is generated and git-ignored.

## Audio input requirement

The benchmark expects each input to already be 16 kHz, mono, 16-bit PCM WAV.
Both implementations read it with the same minimal reader (no ffmpeg, no
resampling), which guarantees identical model input. Convert beforehand if
needed, e.g.:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 -c:a pcm_s16le output.wav
```

## Requirements

- Go toolchain with the CTranslate2 library installed (see `../scripts/`), so the
  cgo bridge links. If CTranslate2 lives outside the default prefix, point to it:

  ```bash
  export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  ```

- The onnxruntime shared library (`libonnxruntime.so`) for the Go Silero VAD.
  It is discovered automatically in common system paths; otherwise:

  ```bash
  export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so
  ```

- Python 3.10+ for the baseline (`run.sh` creates a local `.venv`).

## Quick start

```bash
cd benchmarks
./run.sh            # python baseline + Go port + comparison, both pipelines
```

Targeted runs:

```bash
MODE=batched ./run.sh python    # only the Python baseline, batched only
MODE=batched ./run.sh go        # only the Go port, batched only
./run.sh compare                # re-print the comparison from existing results
```

Run a benchmark directly without `run.sh`:

```bash
# Python
python python/bench.py --mode batched --files test.wav

# Go
cd go && go run . --mode batched --files test.wav
```

## Configuration

All parameters live in `config.json` and are applied identically on both sides.

| Key | Meaning | Default |
|-----|---------|---------|
| `model` | model name or path | `large-v3` |
| `device` / `device_index` | `cpu` or `cuda`, and which GPU | `cuda` / `0` |
| `compute_type` | `int8`, `float16`, `float32`, ... | `int8` |
| `cpu_threads` / `num_workers` | CTranslate2 intra/inter threads (CPU only) | `4` / `2` |
| `batch_size` | chunks per batch (batched mode) | `8` |
| `word_timestamps` | emit word-level timestamps | `true` |
| `vad_filter` | filter silence with Silero VAD | `true` |
| `warmup` | run one untimed warmup pass per mode | `true` |
| `beam_size` / `best_of` | beam width / sampling candidates | `5` / `5` |
| `temperature` | fallback temperature schedule | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` |
| `compression_ratio_threshold` | gzip ratio fallback threshold | `2.4` |
| `log_prob_threshold` | avg log-prob fallback threshold | `-1.0` |
| `no_speech_threshold` | no-speech probability threshold | `0.6` |
| `condition_on_previous_text` | feed previous text as prompt | `true` |
| `vad.threshold` | speech probability threshold | `0.5` |
| `vad.min_speech_duration_ms` | drop speech chunks shorter than this | `0` |
| `vad.max_speech_duration_s` | split chunks longer than this (`0` = no limit) | `30` |
| `vad.min_silence_duration_ms` | silence before splitting a chunk | `2000` |
| `vad.speech_pad_ms` | padding added to each side of a chunk | `400` |
| `datadir` | directory of the input WAV files | `../.testdata` |
| `files` | list of `{ path, language }` (empty language = auto-detect) | — |

In batched mode each VAD speech chunk is capped at the 30s `chunk_length`: the Go
port does this unconditionally, faster-whisper only when `vad_parameters` is `None`
or a `dict`. This benchmark passes a `VadOptions` object, so faster-whisper skips
the cap; with `0` (`+Inf`) it then emits chunks longer than 30s and transcribes
only the first 30s of each, dropping the rest. Use `30` here. In sequential mode
the value only shifts VAD segment boundaries and never drops audio.

## Reading the output

`compare.py` prints two tables per mode:

- SPEED: audio duration, Python vs Go transcription time, the realtime
  multiplier (`x`) of each, and the Go/Python time ratio (`< 1.00x` means Go is
  faster).
- ACCURACY: Word Error Rate and Character Error Rate of the Go transcript
  against the Python transcript (the reference), plus the detected language and
  segment counts on both sides.

Timing excludes file reading and the warmup pass.
