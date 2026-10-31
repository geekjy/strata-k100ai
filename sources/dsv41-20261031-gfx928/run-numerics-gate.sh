#!/usr/bin/env bash
# Run the DeepSeek engine's H2 numerics gate (ds4_iq_test) on a K100_AI card.
# No model weights are needed: the test compares the device i-quant expert kernels against ggml's own
# decode of the same bytes plus a double reference.
#
#   WORK=/path/to/work WORK_IMAGE=dcu-qwen3-5:perfect-0.0.1 ./run-numerics-gate.sh
#
# The card is only visible with the host HyHAL libraries and the whole /dev/dri: with just renderD128 it
# comes up as "KONGMING" and HIP answers "no ROCm-capable device".
set -euo pipefail

WORK="${WORK:-/4T-02/strata-work}"
WORK_IMAGE="${WORK_IMAGE:-dcu-qwen3-5:perfect-0.0.1}"
BUILD="${BUILD:-$WORK/build-dsv41-full}"

[[ -x "$BUILD/ds4_iq_test" ]] || { echo "no ds4_iq_test under $BUILD (build it with build-dsv41-gfx928.sh)" >&2; exit 1; }

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
    ./'"${BUILD#$WORK/}"'/ds4_iq_test
  ' | tee /tmp/ds4_iq_test.out

grep -q '^ALL OK' /tmp/ds4_iq_test.out || { echo "numerics gate did not report ALL OK" >&2; exit 1; }
echo "numerics gate: ALL OK"
