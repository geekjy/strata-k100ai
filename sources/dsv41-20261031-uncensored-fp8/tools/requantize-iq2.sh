set -u
export PYTHONPATH="/4T-02/strata-work/llamacpp/gguf-py:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/4T-02/strata-work/llamacpp/build/bin:${LD_LIBRARY_PATH:-}"
cd /4T-02/strata-work || exit 1

mkdir -p /4T-02/dsv41-conv/iq2-tmp
rm -f /4T-02/dsv41-conv/iq2-tmp/*.bin
df -h /4T-02 | tail -1
free -g | head -2

echo "=== quantize 120 expert tensors to IQ2_XXS, 120 at a time ==="
start=$(date +%s)
python3 -u /4T-02/strata-work/ds4_requant_gguf.py \
  --src /4T-02/dsv41-conv/DSV41-UNC-v3.gguf \
  --dst /4T-02/dsv41-conv/DSV41-UNC-v4-iq2.gguf \
  --jobs 120 2>&1 | tail -40
echo "[exit $? after $(( $(date +%s) - start ))s]"
ls -l /4T-02/dsv41-conv/DSV41-UNC-v4-iq2.gguf 2>/dev/null | awk '{print $5, "bytes"}'
