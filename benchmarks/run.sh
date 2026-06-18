#!/usr/bin/env bash
# Run the Python-vs-Go benchmark:
#   1. python -> set up a venv, run the reference faster-whisper baseline
#   2. go     -> build and run the Go port
#   3. compare -> diff speed and accuracy
#
# Usage:
#   ./run.sh            # python, go, compare
#   ./run.sh python     # only the Python baseline
#   ./run.sh go         # only the Go port
#   ./run.sh compare    # only re-run the comparison on existing results
#
# Environment overrides:
#   MODE=batched|sequential|both   pipeline(s) to run        (default: both)
#   CT2_PREFIX=/usr/local          CTranslate2 install prefix (for the cgo link)
#   PYTHON=python3                 interpreter used for the venv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$HERE/results"
VENV_DIR="$HERE/.venv"

MODE="${MODE:-both}"
CT2_PREFIX="${CT2_PREFIX:-/usr/local}"
PYTHON="${PYTHON:-python3}"
TARGET="${1:-all}"

export PKG_CONFIG_PATH="$CT2_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$CT2_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RESULTS_DIR"

run_python() {
    echo "==> Python baseline (mode=$MODE)"
    [[ -d "$VENV_DIR" ]] || "$PYTHON" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$HERE/python/requirements.txt"
    python "$HERE/python/bench.py" --mode "$MODE" --out "$RESULTS_DIR"
    deactivate
}

run_go() {
    echo "==> Go port (mode=$MODE)"
    ( cd "$HERE/go" && go run . --mode "$MODE" --out "$RESULTS_DIR" )
}

run_compare() {
    echo "==> Comparison"
    "$PYTHON" "$HERE/compare.py" --results "$RESULTS_DIR" --mode "$MODE"
}

case "$TARGET" in
    all)     run_python; run_go; run_compare ;;
    python)  run_python ;;
    go)      run_go ;;
    compare) run_compare ;;
    *) echo "unknown target: $TARGET (use: all|python|go|compare)"; exit 1 ;;
esac
