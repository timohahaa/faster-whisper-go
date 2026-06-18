#!/usr/bin/env bash
set -euo pipefail

# Pinned versions. Bump deliberately; keep in sync with build-ct2-macos.sh where it makes sense.
CT2_VERSION="v4.8.0"
# CUDA toolkit version to build against. Must match an installed /usr/local/cuda-<ver>.
CUDA_VERSION="12.9"
PREFIX="/usr/local"
# CUDA compute capabilities to generate code for. "86" == NVIDIA RTX 3060 (Ampere).
# Use a semicolon-separated list to target multiple GPUs, e.g. "75;80;86;89".
CUDA_ARCH="86"
JOBS="$(nproc)"

# This script only builds CTranslate2. Prerequisites are assumed to be installed
# by the user beforehand: a CUDA toolkit, cuDNN (dev), a CPU BLAS (OpenBLAS dev),
# plus build-essential, cmake, git and pkg-config.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)       PREFIX="$2"; shift 2 ;;
        --version)      CT2_VERSION="$2"; shift 2 ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --cuda-arch)    CUDA_ARCH="$2"; shift 2 ;;
        --jobs)         JOBS="$2"; shift 2 ;;
        *) echo "Usage: $0 [--prefix <dir>] [--version <tag>] [--cuda-version <ver>] [--cuda-arch <list>] [--jobs <n>]"; exit 1 ;;
    esac
done

CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"

echo "==> Building CTranslate2 $CT2_VERSION for Linux/GPU"
echo "    prefix:    $PREFIX"
echo "    cuda:      $CUDA_VERSION ($CUDA_HOME)"
echo "    cuda arch: $CUDA_ARCH"
echo "    jobs:      $JOBS"

if [[ ! -d "$CUDA_HOME" ]]; then
    echo "ERROR: CUDA $CUDA_VERSION not found at $CUDA_HOME" >&2
    echo "       Installed toolkits:" >&2
    ls -d /usr/local/cuda-* 2>/dev/null >&2 || echo "       (none)" >&2
    exit 1
fi

# Verify required tools are present (the user is expected to have installed them).
missing=()
for tool in cmake git pkg-config; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required tools: ${missing[*]}" >&2
    echo "       Install them and re-run, e.g.:" >&2
    echo "       sudo apt-get install -y build-essential cmake git pkg-config libopenblas-dev libcudnn8-dev" >&2
    exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

git clone --recursive --branch "$CT2_VERSION" --depth 1 https://github.com/OpenNMT/CTranslate2.git "$WORKDIR/CTranslate2"
cd "$WORKDIR/CTranslate2"
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

make -j"$JOBS"

if [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    sudo make install
    sudo ldconfig
else
    make install
fi

echo "==> Generating pkg-config file..."
PC_DIR="$PREFIX/lib/pkgconfig"
write_pc() {
    cat <<PCEOF
prefix=$PREFIX
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: CTranslate2
Description: Fast inference engine for Transformer models
Version: ${CT2_VERSION#v}
Libs: -L\${libdir} -Wl,-rpath,\${libdir} -lctranslate2
Cflags: -I\${includedir}
PCEOF
}
if [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    sudo mkdir -p "$PC_DIR"
    write_pc | sudo tee "$PC_DIR/ctranslate2.pc" > /dev/null
else
    mkdir -p "$PC_DIR"
    write_pc > "$PC_DIR/ctranslate2.pc"
fi

echo "==> Verifying installation..."
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
pkg-config --exists ctranslate2 && echo "==> OK: pkg-config finds ctranslate2 $(pkg-config --modversion ctranslate2)" || echo "==> WARN: pkg-config cannot find ctranslate2"

echo
echo "==> Done. To build/run the Go package against this CTranslate2:"
echo "    export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
echo "    export LD_LIBRARY_PATH=$PREFIX/lib:\${LD_LIBRARY_PATH:-}"
