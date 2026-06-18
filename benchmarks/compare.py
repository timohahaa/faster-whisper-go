#!/usr/bin/env python3
"""Compare Python baseline vs Go port benchmark results.

Reads the JSON files produced by python/bench.py and go/bench.go and reports,
per mode and per file, transcription time + realtime factor (speed) and the
WER/CER of the Go transcript against the Python reference (accuracy). WER/CER
use an exact Levenshtein distance.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

MODES = ["batched", "sequential"]


def load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein_bounded(a, b, max_k: int) -> int:
    """Exact Levenshtein distance if it is <= max_k, otherwise max_k + 1.

    Only the diagonal band |i - j| <= max_k is computed, so the cost is
    O(len(a) * max_k) instead of O(len(a) * len(b)).
    """
    n, m = len(a), len(b)
    if abs(n - m) > max_k:
        return max_k + 1

    inf = max_k + 1
    prev = [inf] * (m + 1)
    cur = [inf] * (m + 1)
    for j in range(0, min(max_k, m) + 1):
        prev[j] = j

    for i in range(1, n + 1):
        lo = max(0, i - max_k)
        hi = min(m, i + max_k)
        if lo == 0:
            cur[0] = i
        elif lo - 1 >= 0:
            cur[lo - 1] = inf  # left guard so cur[lo-1] never wins
        ai = a[i - 1]
        for j in range(max(1, lo), hi + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = prev[j - 1] + cost
            d = prev[j] + 1
            if d < v:
                v = d
            d = cur[j - 1] + 1
            if d < v:
                v = d
            cur[j] = v
        if hi + 1 <= m:
            cur[hi + 1] = inf  # right guard for the next row's prev[hi+1]
        prev, cur = cur, prev

    d = prev[m]
    return d if d <= max_k else max_k + 1


def levenshtein(a, b) -> int:
    """Exact Levenshtein distance, fast when the inputs are similar."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    k = max(1, abs(n - m))
    while True:
        d = _levenshtein_bounded(a, b, k)
        if d <= k:
            return d
        k *= 2


def wer(ref: str, hyp: str) -> float:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = normalize(ref).replace(" ", "")
    h = normalize(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def index_results(payload) -> dict:
    return {r["file"]: r for r in (payload.get("results") or [])}


def compare_mode(mode: str, results_dir: Path) -> bool:
    py = load(results_dir / f"python_{mode}.json")
    go = load(results_dir / f"go_{mode}.json")
    if py is None or go is None:
        missing = []
        if py is None:
            missing.append(f"python_{mode}.json")
        if go is None:
            missing.append(f"go_{mode}.json")
        print(f"\n## mode={mode}: skipped (missing {', '.join(missing)})")
        return False

    print(f"\n{'='*78}")
    print(f"MODE: {mode}")
    print(f"  python: {py['model']} / {py['device']} / {py['compute_type']} "
          f"(load {py.get('model_load_sec', 0):.1f}s)")
    print(f"  go:     {go['model']} / {go['device']} / {go['compute_type']} "
          f"(load {go.get('model_load_sec', 0):.1f}s)")
    print('=' * 78)

    py_idx = index_results(py)
    go_idx = index_results(go)
    files = [f for f in py_idx if f in go_idx]

    print("\n-- SPEED --")
    header = f"{'file':<30} {'audio':>9} {'py(s)':>9} {'go(s)':>9} {'py x':>7} {'go x':>7} {'go/py':>7}"
    print(header)
    print("-" * len(header))
    for f in files:
        p = py_idx[f]
        g = go_idx[f]
        ratio = (g["transcribe_time_sec"] / p["transcribe_time_sec"]
                 if p["transcribe_time_sec"] else 0.0)
        print(f"{f[:30]:<30} {p['audio_duration_sec']:>8.1f}s "
              f"{p['transcribe_time_sec']:>9.2f} {g['transcribe_time_sec']:>9.2f} "
              f"{p['speedup_vs_realtime']:>6.1f}x {g['speedup_vs_realtime']:>6.1f}x "
              f"{ratio:>6.2f}x")

    print("\n-- ACCURACY (Go vs Python reference) --")
    header = f"{'file':<30} {'WER':>8} {'CER':>8} {'lang py/go':>14} {'seg py/go':>12}"
    print(header)
    print("-" * len(header))
    for f in files:
        p = py_idx[f]
        g = go_idx[f]
        w = wer(p["text"], g["text"])
        c = cer(p["text"], g["text"])
        lang = f"{p['language_detected']}/{g['language_detected']}"
        seg = f"{p['num_segments']}/{g['num_segments']}"
        print(f"{f[:30]:<30} {w*100:>7.2f}% {c*100:>7.2f}% {lang:>14} {seg:>12}")

    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--mode", choices=MODES + ["both"], default="both")
    args = ap.parse_args()

    results_dir = Path(args.results).resolve()
    modes = MODES if args.mode == "both" else [args.mode]

    any_done = False
    for mode in modes:
        any_done |= compare_mode(mode, results_dir)

    if not any_done:
        print("\nNo result pairs found. Run python/bench.py and go/bench.go first.")


if __name__ == "__main__":
    main()
