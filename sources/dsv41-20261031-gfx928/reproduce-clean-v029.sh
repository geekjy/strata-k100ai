#!/usr/bin/env bash
# Clean v0.2.9 (c62d5e3a) -> the exact production tree that was built and tested on the K100_AI box.
#
#   ./reproduce-clean-v029.sh /path/to/v029.tar.gz
#   ./reproduce-clean-v029.sh /path/to/v029.tar.gz --dir /4T-02/strata-work/repro
#   ./reproduce-clean-v029.sh /path/to/v029.tar.gz --build      # docker build + ds4_mfma_test
#
# --build needs the dcu-qwen3-5:perfect-0.0.1 image and a llama.cpp tree (LLAMA_DIR, default
# /4T-02/strata-work/llama).  The source tree it produces is hash-identical to build-all.
set -euo pipefail

SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SELF_DIR/../.." && pwd)
V029=""
TARGET=""
DO_BUILD=0
while (($#)); do
    case "$1" in
        --dir) TARGET=${2:?--dir needs a path}; shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) V029=$1; shift ;;
    esac
done
if [[ -z "$V029" ]]; then
    echo "usage: $0 /path/to/v029.tar.gz [--dir TARGET] [--build]" >&2
    exit 2
fi
[[ -f "$V029" ]] || { echo "no v029 tarball at $V029" >&2; exit 2; }
TARGET=${TARGET:-$PWD/strata-clean-v029}
[[ -e "$TARGET" ]] && { echo "target exists: $TARGET (remove it first)" >&2; exit 2; }

P=$REPO/sources/dsv41-20261031-uncensored-fp8
G=$REPO/sources/dsv41-20261031-gfx928
PATCHES=(
    "$P/tools/patches-wip/baseline-gfx928-port.patch"
    "$P/patches/0002-ds4-launch-probe.patch"
    "$P/patches/0003-ds4-stage-markers.patch"
    "$P/patches/0004-allreduce-rs2.patch"
    "$P/tools/patches-wip/probes-base.patch"
    "$P/tools/patches-wip/expert-index-tp.patch"
    "$G/patches/0006-mfma-attention.patch"
)
for f in "${PATCHES[@]}"; do [[ -f "$f" ]] || { echo "missing patch: $f" >&2; exit 2; }; done

mkdir -p "$TARGET"
TARGET=$(cd -- "$TARGET" && pwd)
tar -xzf "$V029" -C "$TARGET"
SRC=$(find "$TARGET" -maxdepth 1 -mindepth 1 -type d | head -1)
[[ -n "$SRC" ]] || { echo "the tarball did not contain a directory" >&2; exit 2; }
cd "$SRC"
for f in "${PATCHES[@]}"; do
    echo "=== apply $(basename "$f") ==="
    patch -p1 --forward < "$f"
done
if find . -name '*.rej' | grep -q .; then
    echo "rejects found:" >&2
    find . -name '*.rej' >&2
    exit 1
fi

echo "=== verify the four changed files and the production kernel ==="
sha256sum -c <<'EOF'
d19122f293c9b69843324ba03b3126aeb133cfeaa574bddee2fbefcbefd1a3a5  src/dsv41/ds4_model.cu
c4e85e2a059658ac52cea545a85f2e47f156d9e0b83a1381223fd4b4535e042b  src/dsv41/ds4_kernels.cu
d88d7efd63de73fefbf986e35f8ace4dfd5848b0e884446fecc94f2bb3642057  src/dsv41/ds4_kernels.cuh
d60adb1d1229d738f25673b2f94a63544fc7130f207e1ac3f9d4948ea4f48c83  src/dsv41/ds4_mfma.cuh
8e9903d49f254e98d58a63a3caca57aed7c1c24b5fab97404d76f86ecb2bb133  tests/dsv41/mfma_layout_test.cpp
e48158c6c41ed588c317078dc1f07054aa7c09f698a01760b793b5f2925565d8  CMakeLists.txt
EOF

echo
echo "clean tree: $SRC"
echo "This is hash-identical to the build-all tree that was benchmarked."
echo
echo "Next (on the K100_AI box):"
echo "  docker run --rm --privileged --device /dev/kfd --device /dev/dri \\"
echo "    -v $TARGET:/work -v /opt/hyhal:/opt/hyhal:ro -v /usr/local/hyhal:/usr/local/hyhal:ro \\"
echo "    --entrypoint bash dcu-qwen3-5:perfect-0.0.1 -lc 'source /opt/dtk/env.sh; \\"
echo "      cmake -S /work/$(basename "$SRC") -B /work/build -DCMAKE_BUILD_TYPE=Release \\"
echo "        -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_HIP_ARCHITECTURES=gfx928 \\"
echo "        -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=/work/llama \\"
echo "        -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \\"
echo "        -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \\"
echo "        -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal;/opt/dtk && \\"
echo "      cmake --build /work/build -j 40 --target strata-dsv41 ds4_mfma_test'"
echo "  # then run /work/build/ds4_mfma_test on the card (expects: matrix-core layout test: ALL OK)"

if ((DO_BUILD)); then
    IMAGE=${DCU_IMAGE:-dcu-qwen3-5:perfect-0.0.1}
    LLAMA=${LLAMA_DIR:-/4T-02/strata-work/llama}
    command -v docker >/dev/null || { echo "--build needs docker" >&2; exit 2; }
    [[ -d "$LLAMA" ]] || { echo "no llama.cpp tree at $LLAMA (set LLAMA_DIR)" >&2; exit 2; }
    SRC_NAME=$(basename "$SRC")
    cat > "$TARGET/build.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/dtk/env.sh 2>/dev/null
cmake -S "/work/$SRC_NAME" -B /work/build \\
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF \\
  -DCMAKE_HIP_ARCHITECTURES=gfx928 -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON \\
  -DSTRATA_GGML_DIR=/work/llama -DSTRATA_BUILD_TESTS=OFF \\
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \\
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \\
  "-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false" \\
  "-DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal;/opt/dtk"
cmake --build /work/build -j 40 --target strata-dsv41 ds4_mfma_test -- -k 0
EOF
    chmod +x "$TARGET/build.sh"
    echo "=== build with $IMAGE ==="
    docker run --rm -v "$TARGET:/work" -v "$LLAMA:/work/llama:ro" --cpus=48 \
        --entrypoint bash "$IMAGE" /work/build.sh
    echo "built: $TARGET/build/strata-dsv41 and ds4_mfma_test"
fi
