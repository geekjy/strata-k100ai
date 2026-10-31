#!/usr/bin/env bash
# Build strata-dsv41 (DeepSeek-V4.1-Flash) for K100AI / gfx928 on the wave64 HIP backend.
# Run inside the pinned DTK image, with the patched upstream tree and a llama.cpp checkout under /work.
#
#   STRATA_SRC=/work/strata  LLAMA_SRC=/work/llama  BUILD=/work/build-dsv41 \
#     docker run --rm --network none --cpus=64 -v /path/to/work:/work -w /work --entrypoint bash \
#     dcu-qwen3-5:perfect-0.0.1 /work/build-dsv41-gfx928.sh
set -euo pipefail

STRATA_SRC="${STRATA_SRC:-/work/strata}"
LLAMA_SRC="${LLAMA_SRC:-/work/llama}"
DTK_PATH="${DTK_PATH:-/opt/dtk}"
BUILD="${BUILD:-/work/build-dsv41}"
ARCH="${ARCH:-gfx928}"
JOBS="${JOBS:-48}"
NATIVE_EXPERTS="${NATIVE_EXPERTS:-ON}"

[[ -f "$STRATA_SRC/CMakeLists.txt" ]] || { echo "STRATA_SRC does not look like the strata tree: $STRATA_SRC" >&2; exit 1; }
[[ -f "$STRATA_SRC/src/dsv41/ds4_main.cpp" ]] || { echo "STRATA_SRC has no src/dsv41 (is this upstream before the DeepSeek engine?): $STRATA_SRC" >&2; exit 1; }
[[ -f "$LLAMA_SRC/ggml/CMakeLists.txt" ]] || { echo "LLAMA_SRC does not look like llama.cpp: $LLAMA_SRC" >&2; exit 1; }
[[ -x "$DTK_PATH/llvm/bin/clang++" ]] || { echo "no HIP compiler at $DTK_PATH/llvm/bin/clang++" >&2; exit 1; }

# DTK keeps the HSA headers under .hyhal/include/hsa; the hip-lang CMake package looks for them at
# include/hsa. This is the same fix the c4 replay recipe applies inside its build container.
if [[ ! -e "$DTK_PATH/include/hsa" && -d "$DTK_PATH/.hyhal/include/hsa" ]]; then
  ln -sfn "$DTK_PATH/.hyhal/include/hsa" "$DTK_PATH/include/hsa"
  echo "linked $DTK_PATH/include/hsa -> $DTK_PATH/.hyhal/include/hsa"
fi

cmake -S "$STRATA_SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS="$NATIVE_EXPERTS" \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_COMPILER="$DTK_PATH/llvm/bin/clang++" \
  -DCMAKE_C_COMPILER="$DTK_PATH/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$DTK_PATH/llvm/bin/clang++" \
  "-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false" \
  "-DCMAKE_PREFIX_PATH=$DTK_PATH/.hyhal;$DTK_PATH"

cmake --build "$BUILD" -j "$JOBS" --target strata-dsv41 ds4_iq_test -- -k 0

sha256sum "$BUILD/strata-dsv41" "$BUILD/ds4_iq_test"
echo "built: $BUILD/strata-dsv41"
