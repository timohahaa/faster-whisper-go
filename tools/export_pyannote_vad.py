#!/usr/bin/env python3
"""Export the official pyannote/segmentation (2022) VAD model to ONNX and
(optionally) dump parity fixtures for the Go pyannotevad package.

The embedded ``pyannotevad/pyannote_vad.onnx`` is produced by this script. The
weights come from the official pyannote model:

    https://huggingface.co/pyannote/segmentation

That model is gated: accept its conditions on Hugging Face and create an access
token (hf.co/settings/tokens). Then either pass ``--hf-token`` (the script loads
``pyannote/segmentation`` directly) or download the checkpoint yourself and pass
its local path via ``--model``.

Environment (heavy; matches what pyannote.audio 3.1.1 needs):

    python3 -m venv env && . env/bin/activate
    pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    pip install "pyannote.audio==3.1.1" "numpy<2" onnx onnxruntime soundfile

Export the ONNX (from the gated HF model):

    python tools/export_pyannote_vad.py --hf-token hf_xxx --out pyannotevad/pyannote_vad.onnx

...or from a local checkpoint:

    python tools/export_pyannote_vad.py --model ./pytorch_model.bin --out pyannotevad/pyannote_vad.onnx

Dump parity fixtures (raw float32) for the Go tests, from a 16kHz mono wav:

    python tools/export_pyannote_vad.py --hf-token hf_xxx --clip clip.wav \
        --dump-clip clip.f32 --dump-scores ref_scores.f32 --dump-regions ref_out.json

Then run the Go parity tests:

    PYANNOTE_PARITY_CLIP=clip.f32 PYANNOTE_PARITY_SCORES=ref_scores.f32 \
        go test ./pyannotevad/ -run TestScoresParity -v
    PYANNOTE_PARITY_SCORES=ref_scores.f32 PYANNOTE_PARITY_REGIONS=ref_out.json \
        PYANNOTE_PARITY_CLIP=clip.f32 go test . -run TestBinarizeParity -v
"""

import argparse
import json

import numpy as np
import torch
from pyannote.audio import Model, Inference
from pyannote.core import Annotation, Segment, SlidingWindowFeature

# Default Hugging Face id of the official 2022 segmentation model.
DEFAULT_MODEL = "pyannote/segmentation"

# VAD hysteresis thresholds and merge chunk size used by the Go backend.
ONSET = 0.5
OFFSET = 0.363
CHUNK_SIZE = 30.0
SAMPLE_RATE = 16000


def load_model(model_ref: str, hf_token: str | None) -> Model:
    # model_ref may be a local checkpoint path or a Hugging Face repo id.
    return Model.from_pretrained(model_ref, use_auth_token=hf_token)


def export_onnx(model: Model, out_path: str) -> None:
    dur = model.specifications.duration
    win = model.audio.get_num_samples(dur)
    x = torch.zeros(1, 1, win)
    model.eval()
    with torch.inference_mode():
        y = model(x).cpu().numpy()
    print(
        f"duration={dur}s window_samples={win} "
        f"frames_per_chunk={y.shape[1]} num_classes={y.shape[2]} "
        f"warm_up={tuple(model.specifications.warm_up)} powerset={model.specifications.powerset}"
    )
    if model.specifications.powerset:
        raise SystemExit(
            "this model is powerset (e.g. segmentation-3.0); the Go backend "
            "expects the multilabel 2022 pyannote/segmentation model"
        )
    torch.onnx.export(
        model, x, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=16, do_constant_folding=True,
    )
    print(f"wrote {out_path}")


def segmentation_curve(model: Model, wav: np.ndarray) -> SlidingWindowFeature:
    # Voice-activity curve: per-frame max over the speaker slots, then Hamming
    # overlap-add aggregation. This is pyannote's canonical VAD pre-aggregation.
    inf = Inference(
        model,
        pre_aggregation_hook=lambda s: np.max(s, axis=-1, keepdims=True),
    )
    return inf({"waveform": torch.from_numpy(wav)[None, :], "sample_rate": SAMPLE_RATE})


def binarize(scores: SlidingWindowFeature, onset, offset, max_duration):
    # Direct reference for the Go binarizeVadScores port (single activity class).
    num_frames, num_classes = scores.data.shape
    frames = scores.sliding_window
    ts = [frames[i].middle for i in range(num_frames)]
    active = Annotation()
    for k, ks in enumerate(scores.data.T):
        start = ts[0]
        is_active = ks[0] > onset
        cur_s, cur_t, t = [ks[0]], [ts[0]], ts[0]
        for t, y in zip(ts[1:], ks[1:]):
            if is_active:
                if t - start > max_duration:
                    sa = len(cur_s) // 2
                    mi = sa + int(np.argmin(cur_s[sa:]))
                    active[Segment(start, cur_t[mi]), k] = k
                    start = cur_t[mi]
                    cur_s, cur_t = cur_s[mi + 1:], cur_t[mi + 1:]
                elif y < offset:
                    active[Segment(start, t), k] = k
                    start, is_active, cur_s, cur_t = t, False, [], []
                cur_s.append(y)
                cur_t.append(t)
            else:
                if y > onset:
                    start, is_active = t, True
        if is_active:
            active[Segment(start, t), k] = k
    return [[float(s.start), float(s.end)] for s in active.get_timeline()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="local checkpoint path or HF repo id (default: pyannote/segmentation)")
    ap.add_argument("--hf-token", default=None, help="Hugging Face access token for the gated model")
    ap.add_argument("--out", help="output ONNX path")
    ap.add_argument("--clip", help="16kHz mono wav for parity fixtures")
    ap.add_argument("--dump-clip", help="write raw float32 samples here")
    ap.add_argument("--dump-scores", help="write raw float32 reference curve here")
    ap.add_argument("--dump-regions", help="write ref regions JSON here")
    args = ap.parse_args()

    model = load_model(args.model, args.hf_token)

    if args.out:
        export_onnx(model, args.out)

    if args.clip:
        import soundfile as sf

        wav, sr = sf.read(args.clip, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == SAMPLE_RATE, f"expected 16kHz, got {sr}"
        wav = wav.astype(np.float32)

        seg = segmentation_curve(model, wav)
        scores = np.asarray(seg.data, dtype=np.float32).reshape(-1)
        regions = binarize(seg, ONSET, OFFSET, CHUNK_SIZE)
        print(f"clip samples={len(wav)} frames={len(scores)} regions={len(regions)}")

        if args.dump_clip:
            wav.tofile(args.dump_clip)
        if args.dump_scores:
            scores.tofile(args.dump_scores)
        if args.dump_regions:
            json.dump(
                {"frames_step": float(seg.sliding_window.step), "regions": regions},
                open(args.dump_regions, "w"),
                indent=2,
            )


if __name__ == "__main__":
    main()
