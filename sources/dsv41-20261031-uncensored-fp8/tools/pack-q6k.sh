set -u
source /opt/dtk/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
C=/4T-02/dsv41-conv
BIN=/4T-02/strata-work/build-dsv41-full/strata-dsv41

echo "=== 1. repack around the Q6_K dense + IQ2_XXS experts shard ==="
cd /4T-02/strata-work/strata || exit 1
rm -rf "$C/pack6"
python3 -u tools/dsv41/pack_gguf.py \
  --model "$C" --shards DSV41-UNC-v5-q6k.gguf \
  --dspark-gguf "$C/dspark-UNC-bf16.gguf" \
  --mmproj "$C/mmproj-UNC-bf16.gguf" \
  --dspark-noise-token 128799 \
  --out "$C/pack6"
echo "pack exit=$?"

echo "=== 2. pack-info ==="
"$BIN" --pack "$C/pack6" --pack-info 2>&1 | grep -aE "all experts|dense types|no GPU kernel|engram layers|files:|dspark:"
echo "[pack-info exit: $?]"

echo "=== 3. smoke (must contain Paris) ==="
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
start=$(date +%s)
python3 -u tools/dsv41/smoke.py --exe "$BIN" --pack "$C/pack6" --gpus 0,1 --tokens 16 --timeout 3000
echo "[smoke exit: $? after $(( $(date +%s) - start ))s]"
