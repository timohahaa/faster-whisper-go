#!/usr/bin/env bash
set -euo pipefail

CT2_VERSION="v4.8.0"
PREFIX="/usr/local"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)  PREFIX="$2"; shift 2 ;;
        --version) CT2_VERSION="$2"; shift 2 ;;
        *) echo "Usage: $0 [--prefix <dir>] [--version <tag>]"; exit 1 ;;
    esac
done

echo "==> Building CTranslate2 $CT2_VERSION for macOS (prefix: $PREFIX)"

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

git clone --recursive --branch "$CT2_VERSION" --depth 1 https://github.com/OpenNMT/CTranslate2.git "$WORKDIR/CTranslate2"
cd "$WORKDIR/CTranslate2"
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DWITH_MKL=OFF \
  -DWITH_ACCELERATE=ON \
  -DWITH_CUDA=OFF \
  -DOPENMP_RUNTIME=NONE \
  -DCMAKE_BUILD_TYPE=Release

make -j"$(sysctl -n hw.logicalcpu)"

if [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    sudo make install
else
    make install
fi

echo "==> Generating pkg-config file..."
PC_DIR="$PREFIX/lib/pkgconfig"
if [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    sudo mkdir -p "$PC_DIR"
    sudo tee "$PC_DIR/ctranslate2.pc" > /dev/null <<PCEOF
prefix=$PREFIX
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: CTranslate2
Description: Fast inference engine for Transformer models
Version: ${CT2_VERSION#v}
Libs: -L\${libdir} -Wl,-rpath,\${libdir} -lctranslate2
Cflags: -I\${includedir}
PCEOF
else
    mkdir -p "$PC_DIR"
    cat > "$PC_DIR/ctranslate2.pc" <<PCEOF
prefix=$PREFIX
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: CTranslate2
Description: Fast inference engine for Transformer models
Version: ${CT2_VERSION#v}
Libs: -L\${libdir} -Wl,-rpath,\${libdir} -lctranslate2
Cflags: -I\${includedir}
PCEOF
fi

echo "==> Verifying installation..."
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
pkg-config --exists ctranslate2 && echo "==> OK: pkg-config finds ctranslate2" || echo "==> WARN: pkg-config cannot find ctranslate2"
