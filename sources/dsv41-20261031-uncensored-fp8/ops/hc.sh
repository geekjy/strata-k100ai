set -u
echo "=== container / engine ==="
docker ps --format '{{.Names}}\t{{.Status}}' | grep dsv41-serve
docker exec dsv41-serve bash -lc 'ps -eo etime,rss,args | grep -m1 strata-dsv41 | cut -c1-140'
echo
echo "=== health ==="
curl -s -m 10 -o /dev/null -w "  http %{http_code}\n" http://127.0.0.1:8095/health
curl -s -m 10 http://127.0.0.1:8095/health | head -c 200; echo
echo
echo "=== a real request ==="
curl -s -m 300 http://127.0.0.1:8095/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"What is the capital of France? One sentence."}]}' \
  | python3 -c "import json,sys;print('  ->', json.load(sys.stdin)['choices'][0]['message']['content'][:150])" 2>&1
echo
echo "=== disk ==="
df -h /4T-02 | tail -1
