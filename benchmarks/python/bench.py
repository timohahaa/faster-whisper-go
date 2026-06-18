#!/usr/bin/env python3
"""Benchmark the reference Python faster-whisper implementation.

Two pipelines: "batched" (BatchedInferencePipeline) and "sequential"
(WhisperModel.transcribe). Every decode and VAD parameter is read from the
shared config.json and passed explicitly, so this baseline and the Go port (see
../go/bench.go) run with identical settings. Results are written to JSON for
comparison with compare.py.
"""

import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.vad import VadOptions

SAMPLE_RATE = 16000


def rnd(v: float, places: int) -> float:
    """Round half away from zero, identical to the Go port's round()."""
    p = 10**places
    return math.copysign(math.floor(abs(v) * p + 0.5), v) / p


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_datadir(cfg: dict, config_path: Path) -> Path:
    datadir = Path(cfg.get("datadir", "../.testdata"))
    if not datadir.is_absolute():
        datadir = (config_path.parent / datadir).resolve()
    return datadir


def read_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit PCM WAV into float32 samples in [-1, 1)."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate {w.getframerate()} != {SAMPLE_RATE}")
        if w.getnchannels() != 1:
            raise ValueError(f"{path}: channels {w.getnchannels()} != 1 (mono)")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: sample width {w.getsampwidth()*8} bits != 16")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def vad_parameters(cfg: dict) -> VadOptions:
    vad = cfg.get("vad", {})
    max_speech = float(vad.get("max_speech_duration_s", 0))
    return VadOptions(
        threshold=float(vad.get("threshold", 0.5)),
        min_speech_duration_ms=int(vad.get("min_speech_duration_ms", 0)),
        max_speech_duration_s=max_speech if max_speech > 0 else float("inf"),
        min_silence_duration_ms=int(vad.get("min_silence_duration_ms", 2000)),
        speech_pad_ms=int(vad.get("speech_pad_ms", 400)),
    )


def segments_to_dicts(segments) -> list[dict]:
    # Consuming the generator is what triggers the actual transcription work.
    out = []
    for seg in segments:
        words = None
        if seg.words:
            words = [
                {
                    "start": rnd(w.start, 3),
                    "end": rnd(w.end, 3),
                    "word": w.word,
                    "probability": rnd(w.probability, 4),
                }
                for w in seg.words
            ]
        out.append(
            {
                "id": seg.id,
                "start": rnd(seg.start, 3),
                "end": rnd(seg.end, 3),
                "text": seg.text,
                "words": words,
            }
        )
    return out


def run_one(model, batched, mode: str, audio, language: str, cfg: dict) -> dict:
    common = dict(
        language=language or None,
        task="transcribe",
        beam_size=int(cfg.get("beam_size", 5)),
        best_of=int(cfg.get("best_of", 5)),
        temperature=cfg.get("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        compression_ratio_threshold=cfg.get("compression_ratio_threshold", 2.4),
        log_prob_threshold=cfg.get("log_prob_threshold", -1.0),
        no_speech_threshold=cfg.get("no_speech_threshold", 0.6),
        condition_on_previous_text=bool(cfg.get("condition_on_previous_text", True)),
        word_timestamps=bool(cfg.get("word_timestamps", True)),
        vad_filter=bool(cfg.get("vad_filter", True)),
        vad_parameters=vad_parameters(cfg),
    )

    audio_duration = len(audio) / SAMPLE_RATE

    start = time.perf_counter()
    if mode == "batched":
        segments, info = batched.transcribe(
            audio, batch_size=int(cfg.get("batch_size", 8)), **common
        )
    else:
        segments, info = model.transcribe(audio, **common)
    seg_dicts = segments_to_dicts(segments)
    elapsed = time.perf_counter() - start

    text = " ".join(s["text"].strip() for s in seg_dicts).strip()
    num_words = sum(len(s["words"]) for s in seg_dicts if s["words"])

    return {
        "language_detected": info.language,
        "language_probability": rnd(info.language_probability, 4),
        "audio_duration_sec": rnd(audio_duration, 3),
        "transcribe_time_sec": rnd(elapsed, 3),
        "rtf": rnd(elapsed / audio_duration, 4) if audio_duration else 0.0,
        "speedup_vs_realtime": rnd(audio_duration / elapsed, 2) if elapsed else 0.0,
        "num_segments": len(seg_dicts),
        "num_words": num_words,
        "text": text,
        "segments": seg_dicts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.json"))
    ap.add_argument("--mode", choices=["batched", "sequential", "both"], default="both")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "results"))
    ap.add_argument("--files", nargs="*", help="only run these file names (basename match)")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    datadir = resolve_datadir(cfg, config_path)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = cfg["files"]
    if args.files:
        wanted = set(args.files)
        files = [f for f in files if Path(f["path"]).name in wanted or f["path"] in wanted]

    modes = ["batched", "sequential"] if args.mode == "both" else [args.mode]

    print(
        f"loading model={cfg['model']} device={cfg['device']} "
        f"compute_type={cfg['compute_type']} ..."
    )
    load_start = time.perf_counter()
    model = WhisperModel(
        cfg["model"],
        device=cfg.get("device", "cpu"),
        device_index=cfg.get("device_index", 0),
        compute_type=cfg.get("compute_type", "int8"),
        cpu_threads=int(cfg.get("cpu_threads", 0)),
        num_workers=int(cfg.get("num_workers", 1)),
    )
    batched = BatchedInferencePipeline(model=model)
    model_load_sec = time.perf_counter() - load_start
    print(f"model loaded in {model_load_sec:.2f}s")

    # Read every WAV once up front so file IO is excluded from the timed region.
    decoded: dict[str, np.ndarray] = {}
    for f in files:
        p = datadir / f["path"]
        if not p.exists():
            print(f"  WARN: missing {p}, skipping")
            continue
        decoded[f["path"]] = read_wav(p)

    for mode in modes:
        if cfg.get("warmup") and decoded:
            first = next(iter(decoded.values()))
            warm = first[: SAMPLE_RATE * 30]
            print(f"[{mode}] warmup ...")
            run_one(model, batched, mode, warm, "en", cfg)

        results = []
        for f in files:
            if f["path"] not in decoded:
                continue
            audio = decoded[f["path"]]
            print(f"[{mode}] transcribing {f['path']} ({len(audio)/SAMPLE_RATE:.1f}s) ...")
            r = run_one(model, batched, mode, audio, f.get("language", ""), cfg)
            r["file"] = f["path"]
            r["language_requested"] = f.get("language", "")
            results.append(r)
            print(
                f"    -> {r['transcribe_time_sec']:.2f}s "
                f"({r['speedup_vs_realtime']:.1f}x realtime, "
                f"lang={r['language_detected']}, segments={r['num_segments']})"
            )

        payload = {
            "implementation": "python",
            "mode": mode,
            "model": cfg["model"],
            "device": cfg.get("device"),
            "device_index": cfg.get("device_index", 0),
            "compute_type": cfg.get("compute_type"),
            "batch_size": cfg.get("batch_size", 8),
            "word_timestamps": cfg.get("word_timestamps", True),
            "vad_filter": cfg.get("vad_filter", True),
            "model_load_sec": rnd(model_load_sec, 3),
            "results": results,
        }
        out_path = out_dir / f"python_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[{mode}] wrote {out_path}")


if __name__ == "__main__":
    main()
