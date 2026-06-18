# faster-whisper-go

Go port of [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).

## Benchmarks

The Go port is benchmarked head-to-head against the reference Python
`faster-whisper` on identical audio, identical decode/VAD parameters, and the
same JSON output schema. Speed is reported as transcription time and realtime
multiplier; accuracy is the Word/Character Error Rate of the Go transcript
against the Python transcript (the reference). See [`benchmarks/`](benchmarks/)
for the full harness and methodology.

### Setup

| | |
|---|---|
| Model | `large-v3` |
| Compute type | `int8` |
| Device | CUDA, NVIDIA GeForce RTX 3060 (12 GB), driver 575.64.03 / CUDA 12.9 |
| Python baseline | `faster-whisper` + CTranslate2 4.7.0 |
| Batch size (batched mode) | 8 |
| VAD | Silero, `vad_filter=true` |
| Decode | beam_size 5, best_of 5, temperature fallback `[0.0..1.0]`, word timestamps on |

Two pipelines are exercised:

- **batched** — `BatchedInferencePipeline` / `Model.TranscribeBatched`
- **sequential** — `WhisperModel.transcribe` / `Model.Transcribe`

Each implementation ran on a dedicated GPU in parallel (Python on GPU 0, Go on
GPU 1). Timing excludes file reading and a warmup pass.

### Speed

`x` = realtime multiplier (higher is faster). `go/py` = Go time ÷ Python time
(`< 1.00` means Go is faster).

#### Batched

| File | Audio | Python | Go | Python x | Go x | go/py |
|------|------:|-------:|---:|---------:|-----:|------:|
| `test.wav` | 82.0 s | 2.31 s | 2.25 s | 35.4x | 36.4x | 0.97x |
| `anthropic_workshop_en.wav` | 4539.7 s | 111.45 s | 109.34 s | 40.7x | 41.5x | 0.98x |
| `postgres_interview_ru.wav` | 7259.5 s | 204.00 s | 199.96 s | 35.6x | 36.3x | 0.98x |

#### Sequential

| File | Audio | Python | Go | Python x | Go x | go/py |
|------|------:|-------:|---:|---------:|-----:|------:|
| `test.wav` | 82.0 s | 3.54 s | 3.66 s | 23.2x | 22.4x | 1.04x |
| `anthropic_workshop_en.wav` | 4539.7 s | 420.94 s | 521.52 s | 10.8x | 8.7x | 1.24x |
| `postgres_interview_ru.wav` | 7259.5 s | 703.93 s | 745.55 s | 10.3x | 9.7x | 1.06x |

### Accuracy (Go vs Python reference)

#### Batched

| File | WER | CER | lang py/go | seg py/go |
|------|----:|----:|:----------:|:---------:|
| `test.wav` | 1.97% | 2.33% | en/en | 4/4 |
| `anthropic_workshop_en.wav` | 0.49% | 0.35% | en/en | 174/174 |
| `postgres_interview_ru.wav` | 1.04% | 0.60% | ru/ru | 265/265 |

#### Sequential

| File | WER | CER | lang py/go | seg py/go |
|------|----:|----:|:----------:|:---------:|
| `test.wav` | 1.27% | 0.96% | en/en | 7/7 |
| `anthropic_workshop_en.wav` | 7.51% | 5.70% | en/en | 1553/1480 |
| `postgres_interview_ru.wav` | 6.71% | 4.48% | ru/ru | 3006/3374 |

### Takeaways

- **Batched**: the Go port matches the Python baseline on speed (within ~2–3%,
  consistently slightly faster) and stays nearly identical on output (WER ≤ 2%,
  identical segment counts and detected languages).
- **Sequential**: output is close on short audio; on long audio the transcripts
  diverge more (WER ~7%), driven mainly by different segment boundaries
  (e.g. 1553 vs 1480 segments). Go is competitive on the shorter/Russian files
  and ~24% slower on the long English file.
- Batched is the faster pipeline overall (~36–42x realtime vs ~9–23x for
  sequential) and the recommended path for throughput.

Reproduce with:

```bash
cd benchmarks
./run.sh
```
