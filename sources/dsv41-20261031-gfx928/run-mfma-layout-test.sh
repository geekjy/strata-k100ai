#!/usr/bin/env bash
# Run the matrix-core layout gate (ds4_mfma_test) on a K100_AI card.  No model weights are needed: it compares one
# V_MMAC_F32_16X16X16_F16 step (gfx928's matrix core; CDNA would be MFMA), the two-tile mma_scores and the two-tile
# mma_pv against a CPU product of exactly representable integers.  Build it first with build-dsv41-gfx928.sh after
# applying patches/0006-mfma-attention.patch.
#
#   WORK=/path/to/work WORK_IMAGE=dcu-qwen3-5:perfect-0.0.1 bash run-mfma-layout-test.sh
#
# The card is only visible with the host HyHAL libraries and the whole /dev/dri (same set as run-numerics-gate.sh).
set -euo pipefail

WORK="${WORK:-/4T-02/strata-work}"
WORK_IMAGE="${WORK_IMAGE:-dcu-qwen3-5:perfect-0.0.1}"
BUILD="${BUILD:-$WORK/build-dsv41}"

[[ -x "$BUILD/ds4_mfma_test" ]] || {
  echo "no ds4_mfma_test under $BUILD (apply 0006-mfma-attention.patch and build it with build-dsv41-gfx928.sh)" >&2
  exit 1
}

set +e
docker run --rm --privileged \
  --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$WORK:/work" -w /work \
  --entrypoint bash "$WORK_IMAGE" -lc '
    set -o pipefail
    source /opt/dtk/env.sh 2>/dev/null || true
    export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
    echo "agent: $(rocminfo 2>/dev/null | grep -m1 -o "gfx[0-9]*" || echo unknown)"
    ./'"${BUILD#$WORK/}"'/ds4_mfma_test
  ' | tee /tmp/ds4_mfma_test.out
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -eq 77 ]]; then
  echo "matrix-core layout test skipped: this build/device has no MMAC/MFMA matrix core" >&2
  exit 77
fi
[[ $rc -eq 0 ]] || { echo "matrix-core layout test failed (exit $rc)" >&2; exit 1; }
grep -q 'matrix-core layout test: ALL OK' /tmp/ds4_mfma_test.out || { echo "layout test did not report ALL OK" >&2; exit 1; }
echo "matrix-core layout gate: ALL OK"
