#!/bin/bash
# Stage the minimal NVMM libraries needed for the JP6 Docker build.
#
# docker build does not use --runtime nvidia, so the NVIDIA Container Runtime
# does not inject Tegra libraries during build. This script copies only the
# libraries and headers needed by the meson configure step into the build
# context. The staged files are NOT committed to git.
#
# Run from the repo root before docker build:
#   bash scripts/stage-nvmm-libs.sh
#   docker build --network host -f docker/Dockerfile.jetson-jp6 -t gst-nvmm-cpp:jp6 .
#
# Must be run on a Jetson with JP6 (L4T R36.x).

set -euo pipefail

STAGE="$(cd "$(dirname "$0")/.." && pwd)/docker/nvmm-stage"
NV_DIR="/usr/lib/aarch64-linux-gnu/nvidia"
INC_DIR="/usr/src/jetson_multimedia_api/include"

mkdir -p "$STAGE/lib" "$STAGE/include"

# libnvbufsurface + libnvbufsurftransform and their transitive deps.
# Identified via:
#   ldd /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so
#   ldd /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurftransform.so
for stem in libnvbufsurface libnvbufsurftransform \
            libnvrm_mem libnvrm_surface libnvrm_chip \
            libnvrm_gpu libnvrm_sync libnvrm_host1x libnvrm_stream \
            libnvos libnvbuf_fdmap libnvsciipc libnvsocsys libnvtegrahv \
            libnvvic libnvcolorutil libcuda; do
    cp "$NV_DIR"/${stem}* "$STAGE/lib/" 2>/dev/null || true
done

cp "$INC_DIR/nvbufsurface.h" "$STAGE/include/"
cp "$INC_DIR/nvbufsurftransform.h" "$STAGE/include/" 2>/dev/null || true

echo "Staged $(ls "$STAGE/lib/" | wc -l) libs and $(ls "$STAGE/include/" | wc -l) headers to $STAGE/"
echo "Now run: docker build --network host -f docker/Dockerfile.jetson-jp6 -t gst-nvmm-cpp:jp6 ."
