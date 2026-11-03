set +e
# DeepSeek-V4.1-Flash on the expert-index tensor-parallel build: both cards compute EVERY layer.
# DS4_EXPERT_TP is a flag in the engine (any non-empty value enables it).
source /opt/dtk/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
cd /4T-02/strata-work/strata || exit 1

docker stop -t 60 dsv41-serve >/dev/null 2>&1
for i in $(seq 1 30); do docker ps --format '{{.Names}}' | grep -qx dsv41-serve || break; sleep 2; done
docker rm -f dsv41-serve >/dev/null 2>&1
sleep 2

docker run -d --name dsv41-serve --restart unless-stopped --privileged \
  --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro -v /opt/hyhal:/opt/hyhal:ro \
  -v /4T-02:/4T-02 -p 8095:8095 \
  -e DS4_AR_RS2=1 -e DS4_EXPERT_TP=1 -e DS4_ATTN_SPLIT=1 \
  -w /4T-02/strata-work/strata \
  --entrypoint bash dcu-qwen3-5:perfect-0.0.1 -lc '
    source /opt/dtk/env.sh 2>/dev/null
    export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
    exec python3 -m serve.server --engine strata \
      --config /4T-02/strata-work/dsv41-serve-p7tp.json \
      --tokenizer /4T-02/dsv41-conv/pack4/tokenizer \
      --port 8095 --host 0.0.0.0' | tail -1

echo "=== wait for /health ==="
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8095/health 2>/dev/null)
  [ "$code" = "200" ] && { echo "health 200 after ~$((i*10))s"; break; }
  sleep 10
done
curl -s -m 10 http://127.0.0.1:8095/health | head -c 250; echo
