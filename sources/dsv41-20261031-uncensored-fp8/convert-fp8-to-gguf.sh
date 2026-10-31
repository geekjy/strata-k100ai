#!/usr/bin/env bash
# Convert a DeepSeek-V4.1-Flash FP8 checkpoint (e4m3 linears with 32x32 ue8m0 blocks, FP4 experts,
# Engram tables and DSpark/mlp inside) into the three GGUFs strata-dsv41's pack_gguf.py reads.
#
#   MODEL=/models/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8  OUT=/models/gguf  ./convert-fp8-to-gguf.sh
#
# Needs llama.cpp master's converter (conversion/deepseek41.py, merged 2026-10-25, ggml-org/llama.cpp#31086)
# with patches/0001-deepseek41-mxfp8-engram.patch applied, plus torch, transformers, tokenizers, safetensors
# and numpy.  Runs on the CPU; no GPU is used.
#
# Two things the checkpoint itself does not carry and this script checks for:
#   * chat_template.jinja next to the weights  (tokenizer_config.json has no chat_template; without the file
#     the GGUF gets no template and pack_gguf.py refuses - #299)
#   * nothing else: the Engram tables, the DSpark drafter and the vision tower are all in the checkpoint
set -euo pipefail

MODEL="${MODEL:?Set MODEL to the checkpoint directory}"
OUT="${OUT:?Set OUT to the directory for the GGUFs}"
CONVERTER="${CONVERTER:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llamacpp-min}"
NAME="${NAME:-DSV41-UNC}"
DTK="${DTK_PATH:-/opt/dtk}"

[[ -f "$MODEL/config.json" ]] || { echo "no config.json under $MODEL" >&2; exit 1; }
[[ -f "$CONVERTER/convert_hf_to_gguf.py" ]] || { echo "no convert_hf_to_gguf.py under $CONVERTER" >&2; exit 1; }
[[ -f "$CONVERTER/conversion/deepseek41.py" ]] || { echo "no conversion/deepseek41.py under $CONVERTER (llama.cpp before the V4.1 merge?)" >&2; exit 1; }
if [[ ! -f "$MODEL/chat_template.jinja" ]]; then
  echo "warning: $MODEL/chat_template.jinja is missing - the GGUF will carry no chat template and" >&2
  echo "         pack_gguf.py will refuse to pack it (#299).  Copy the model family's chat template there." >&2
fi
grep -q "_MXFP8_GGML_ID" "$CONVERTER/conversion/deepseek41.py" || {
  echo "warning: conversion/deepseek41.py has no MXFP8 Engram write - the pack will hold BF16 Engram tables," >&2
  echo "         which the engine's Engram reader rejects.  Apply patches/0001-deepseek41-mxfp8-engram.patch" >&2
  echo "         and re-run." >&2
}

export PYTHONPATH="$CONVERTER/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DTK/lib:$DTK/hip/lib:/usr/local/hyhal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$OUT"

echo "=== main model (Engram tables inline, DSpark excluded) ==="
python3 -u "$CONVERTER/convert_hf_to_gguf.py" "$MODEL" \
  --outfile "$OUT/$NAME-v3.gguf" --outtype bf16

echo "=== DSpark drafter (mtp.0-2 as a DFLASH GGUF) ==="
python3 -u "$CONVERTER/convert_hf_to_gguf.py" "$MODEL" \
  --outfile "$OUT/dspark-$NAME-bf16.gguf" --outtype bf16 --dspark

echo "=== vision tower (mmproj, projector deepseek41v) ==="
python3 -u "$CONVERTER/convert_hf_to_gguf.py" "$MODEL" \
  --outfile "$OUT/mmproj-$NAME-bf16.gguf" --outtype bf16 --mmproj

ls -l "$OUT"/*.gguf
sha256sum "$OUT"/*.gguf
echo "next: python3 tools/dsv41/pack_gguf.py --model $OUT --shards $NAME-v3.gguf \\"
echo "        --dspark-gguf $OUT/dspark-$NAME-bf16.gguf --mmproj $OUT/mmproj-$NAME-bf16.gguf \\"
echo "        --dspark-noise-token 128799 --out $OUT/pack"
