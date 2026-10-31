set +e
source /opt/dtk/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
cd /4T-02/strata-work/strata || exit 1

echo "=== start the OpenAI/Anthropic server over the dsv41 engine ==="
docker rm -f dsv41-serve 2>/dev/null | head -1
docker run -d --name dsv41-serve --privileged \
  --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro -v /opt/hyhal:/opt/hyhal:ro \
  -v /4T-02:/4T-02 -p 8095:8095 \
  -w /4T-02/strata-work/strata \
  --entrypoint bash dcu-qwen3-5:perfect-0.0.1 -lc '
    source /opt/dtk/env.sh 2>/dev/null
    exec python3 -m serve.server --engine strata \
      --config /4T-02/strata-work/dsv41-serve.json \
      --tokenizer /4T-02/dsv41-conv/pack4/tokenizer \
      --port 8095 --host 0.0.0.0' | tail -1

echo "=== wait for /health (the engine loads first; up to 10 min) ==="
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8095/health 2>/dev/null)
  if [ "$code" = "200" ]; then echo "health 200 after ~$((i*10))s"; break; fi
  sleep 10
done
curl -s -m 10 http://127.0.0.1:8095/health | head -c 400; echo
echo "=== /v1/models ==="
curl -s -m 10 http://127.0.0.1:8095/v1/models | head -c 400; echo
