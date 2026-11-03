# Operations runbook — DeepSeek-V4.1-Flash-1017 on 2x K100_AI

Everything needed to start, check, roll back and re-tune the deployment, plus the list of switches
that must not be touched and the traps already paid for. Written 2026-11-03.

## 1. The box

| item | value |
|---|---|
| GPUs | 2x Hygon K100_AI, `gfx928` (CDNA, wave64), 65,520 MiB VRAM each |
| CPU / RAM | Hygon C86-4G, 128 cores, 503 GB (the container is pinned to NUMA nodes 1,3,5,7) |
| OS / driver | Ubuntu 22.04 host, hyhal mounted from the host at `/opt/hyhal` and `/usr/local/hyhal` |
| toolkit | DTK 26.04 **inside the image**, clang 17 (`/opt/dtk`), gdb 15.2 as `hipgdb` |
| container image | `dcu-qwen3-5:perfect-0.0.1` (the only image used throughout) |
| disk | `/4T-02` (3.5 T, the model and all scripts live there) |

There is **no docker CLI inside the container** — every `docker` command runs on the host.

The container recipe used everywhere (note the device and driver mounts):

```bash
docker run --rm --privileged --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro -v /opt/hyhal:/opt/hyhal:ro \
  -v /4T-02:/4T-02 -v /4T-02/strata-work:/work -w /work \
  --entrypoint bash dcu-qwen3-5:perfect-0.0.1 <script-or--lc "...">
```

Inside every container: `source /opt/dtk/env.sh; export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH`.

## 2. What is deployed

**Production = expert-index tensor parallelism**: both cards compute every layer, each holding its
own whole experts.

| piece | path |
|---|---|
| engine | `/4T-02/strata-work/repro-build/build/strata-dsv41` (clean v0.2.9 reproduction; previous: `build-exp`) |
| model pack | `/4T-02/dsv41-conv/pack8` (IQ1_S experts, 106.2 GB at 1.56 bpw) over `/4T-02/dsv41-conv/DSV41-UNC-v3-iq1s.gguf` (325.8 GB); native dense/Engram, quantized experts only. Previous: native pack4 (v3, 508 GB) and quantized pack7 (IQ2/Q4_K, 350 GB) |
| serve config | `/4T-02/strata-work/dsv41-serve-p7tp.json` |
| start script | `/4T-02/strata-work/serve_pack7_tp.sh` (container `dsv41-serve`, port 8095; name is historical) |
| fallback | `/4T-02/strata-work/serve_pack7.sh` (baseline engine, layer split, pack7) |

Effective engine line and environment:

```
repro-build/build/strata-dsv41 --serve --pack /4T-02/dsv41-conv/pack8 --gpus 0,1 --tp 2 --dspark on --mmap-experts on
DS4_EXPERT_TP=1  DS4_AR_RS2=1  DS4_ATTN_SPLIT=1  DS4_FAST_ATTN=1   # production experiment since 21:43
```

`--restart unless-stopped` is set, but a `docker rm` still takes the service down — always
re-run the start script after removing the container.

Production was switched on 2026-11-03 21:32 to the clean v0.2.9 reproduction build
(`repro-build/build/strata-dsv41`), at 21:43 it was restarted with `DS4_FAST_ATTN=1` (MMAC path,
explicit production experiment), at 21:58 the model pack was switched from the quantized pack7
(IQ2/Q4_K, v7-q4k) to the **native MXFP4 pack4** (v3, 508 GB), and at 23:57 to **pack8, the IQ1_S
expert quant** (`DSV41-UNC-v3-iq1s.gguf`, 325.8 GB; experts 106.2 GB/1.56 bpw, all in VRAM; native
dense/Engram). Config backups:
`/4T-02/strata-work/dsv41-serve-p7tp.json.bak-20261103-213218` (pre-clean-build),
`...bak-pack7-20261103-215758` (pack7), and `...bak-pack4-20261103-235735` (pack4). The pre-fast-attn
start script is at `/4T-02/strata-work/serve_pack7_tp.sh.bak-20261103-214315`. Restore the desired
config and re-run the start script to roll back. Health and a small inference request were verified
after each
change.

## 3. Day-to-day commands (on the host)

```bash
# start / restart the production
bash /4T-02/strata-work/serve_pack7_tp.sh          # waits for /health, ~190-200 s on pack4

# health + a real answer  (the standard check)
bash /4T-02/strata-work/hc.sh

# raw checks
curl -s http://127.0.0.1:8095/health
curl -s http://127.0.0.1:8095/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"hi"}]}'

# what the engine is doing
docker logs dsv41-serve --tail 50
docker exec dsv41-serve bash -lc 'ps -eo args | grep -m1 strata-dsv41'
hy-smi                    # both cards should show work while a prompt is running

# roll back to the baseline engine (layer split, no cross-card sync)
bash /4T-02/strata-work/serve_pack7.sh
```

The API is OpenAI-compatible on port 8095 (`/v1/chat/completions`, `/health`). The model name is
`DeepSeek-V4.1-Flash-1017`.

## 4. Baseline performance (use these to detect regressions)

Measured with `--dspark on`, pack7, expert-TP unless noted. Single runs vary by +/-20 % — compare
medians of at least three runs, never single samples.

| workload | prefill | decode |
|---|---|---|
| expert-TP, 4080-token prompt | ~138 tok/s | 24-27 tok/s |
| expert-TP, 10200-token prompt | ~183 tok/s | — |
| baseline engine (layer split), same packs | ~134 tok/s | ~27 tok/s |

The engine prints its own numbers into the log, use those rather than wall-clock timings:

```
[strata] done: 128 tokens in 6 s (23.1 tok/s) ...
prompt: 4080 tokens in 29517 ms (138.23 tok/s, windows of 8192)
decode: 64 tokens in 2357 ms (26.73 tok/s); CPU lane: 1436 of 19320 experts
```

## 5. Switches

### 5.1 In use

| switch | effect |
|---|---|
| `DS4_EXPERT_TP=1` | the expert-index TP itself (any non-empty value enables it) |
| `--tp 2` | two members; required for the above |
| `DS4_AR_RS2=1` | the all-reduce variant that measured best here |
| `DS4_ATTN_SPLIT=1` | split-KV decode attention; A/B here 23.39 -> 25.21 tok/s (+7.8 %) |
| `--dspark on` | the speculative drafter; worth ~+28 % decode in earlier measurements |
| `--mmap-experts on` | read the CPU-lane experts in place (do not copy them into RAM) |

### 5.2 Not enabled here

| switch | why |
|---|---|
| `DS4_FAST_ATTN=1` | **enabled in production since 2026-11-03 21:43 as an explicit experiment**. Without `patches/0006-mfma-attention.patch` it selects the WMMA kernel; `ds4_wmma.cuh` has no gfx9 branch and executes `__builtin_trap()`. With 0006 on gfx928 it uses **MMAC** (not MFMA); layout gate and 4K/10K ids pass, but a controlled 4K A/B/A/B showed **no prefill gain** and 10K only about **+2.8 %**; the all-prompt-position KL mean is 0.0255 (last-position 0.00131). Roll back by restoring `serve_pack7_tp.sh.bak-20261103-214315` and re-running. See `docs/mfma-attention-port.md` section 10. |

### 5.3 Measured and rejected (do not re-litigate without new data)

| change | result |
|---|---|
| `--prefill-chunk 8192` | prefill 109 vs 138 tok/s (worse) |
| `DS4_FAT_ROWS=64` | 136 vs 138 tok/s (no gain) |
| `--max-context 12288/8192` | expert count 5803 -> 5809 (+0.1 %); the model's KV cache is tiny |
| restructuring the chunk all-reduce | it is not the bottleneck (3.17 ms/token of ~40) |

### 5.4 Debug-only (leave unset in production)

`DS4_PROF`, `DS4_PROF_CHUNK`, `DS4_TLOG`, `DS4_TLOG2`, `DS4_PRELOG`, `DS4_LAUNCH_CHECK`,
`DS4_LAUNCH_SYNC`, plus the probes added with this fork: `DS4_GEMMDBG`, `DS4_BUFDBG`, `DS4_SETDBG`,
`DS4_SETCAP`, `DS4_RANGECHK`, `DS4_PTRCHECK`, `DS4_YGUARD`, `DS4_SERIALIZE`, `DS4_NO_CHUNK_AR`.

## 6. The failure modes already paid for

### 6.1 Disk filled with core dumps
Each crash of the engine under a faulting kernel wrote a ~140 GB `core.*` file. Sixteen of them
filled `/4T-02` to 100 %, which also made every SFTP upload fail with "size mismatch 0 != N".
`kernel.core_pattern` is now `/dev/null` on the host; if it is ever reset, expect this again.

### 6.2 Service holds the GPUs
Any test that starts the engine directly must stop `dsv41-serve` first, otherwise the new engine
fails with `cudaMalloc ... failed` / "no CUDA device". The test scripts here all do
`docker stop dsv41-serve; docker rm -f dsv41-serve; sleep 3` first and restore afterwards.

### 6.3 The chunk-path crash (fixed 2026-11-03)
Symptom: VMFault for any prompt longer than ~2048 tokens, fine below. Root cause: under expert-TP
each member computes **whole** experts (and the lead the whole shared expert, width `g.ff` = 2304),
but four scratch buffers (`c.sg`, `c.su`, `c.sh16`, `c.h16`) were still sized with the **per-member
half** FFN width from `split128(g.ff, nm, ...)` = 1152. The shared-expert GEMM wrote 2304 columns
into a 1152-wide buffer as soon as `T > C/2 = 2048`.

Regression guard: in `ds4_model.cu`, the allocation of those four buffers must use the full width
when `expert_tp_on()` (`const int FFs = (nm > 1 && expert_tp_on()) ? g.ff : FF;`). If someone
re-derives the buffers from `split128` again, the crash comes back and it will look like a random
VMFault in a rocBLAS GEMM — that is why this note exists.

### 6.4 How the diagnosis was done (repeat if a similar fault appears)
1. Make the reproducer cheap: drive the engine through the API server instead of one process per
   prompt (15 min -> 13 s).
2. `hipgdb -q -batch -ex "handle SIGSEGV stop" -ex run -ex "bt 3" -ex "info line *\$pc"` names the
   faulting kernel; host-side variables are not readable, registers are.
3. Print the shapes at the suspect call site (`DS4_GEMMDBG`) and the buffer addresses/sizes
   (`DS4_BUFDBG`), then compare the fault address with them — "240 KB past `c.sg`" is what settled
   it.

## 7. Rebuilding

Sources live under `/4T-02/strata-work/`:

| tree | what |
|---|---|
| `strata` | the baseline tree the production baseline binary is built from |
| `strata-exp` | the previous experimental tree (expert-TP + the probes); this is what `build-exp` compiles |
| `repro-build` | **current production**: the clean v0.2.9 reproduction (all seven patches + 0006); built by `sources/dsv41-20261031-gfx928/reproduce-clean-v029.sh --build` |

The one-command production build (replaces the manual route below):

```bash
sources/dsv41-20261031-gfx928/reproduce-clean-v029.sh /path/to/v029.tar.gz \
  --dir /4T-02/strata-work/repro-build --build
```

The manual `strata-exp` → `build-exp` route is kept for reference:

```bash
# inside a container with /work mounted, with the DTK environment sourced:
cmake -S /work/strata-exp -B /work/build-exp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx928 \
  -DCMAKE_PREFIX_PATH="/opt/dtk/.hyhal;/opt/dtk" \
  -DCMAKE_HIP_FLAGS="--gpu-max-threads-per-block=1024 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false" \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_PLE_FIXTURE_DIR=/work/strata-exp/bench/micro \
  -DSTRATA_GGML_DIR=/work/llama
ninja -C /work/build-exp -j 48 strata-dsv41
```

(the option list above is the one the working cache `/4T-02/strata-work/build-exp/CMakeCache.txt`
actually holds — check it with
`grep -E '^STRATA_' /4T-02/strata-work/build-exp/CMakeCache.txt` if a fresh configure is needed.)
```

The expert-TP change is exported as a patch: `sources/dsv41-20261031-uncensored-fp8/tools/patches-wip/expert-index-tp.patch`
(`diff -u` against the baseline trees; 56 KB).

## 8. Quants and packs

| pack | dense | experts | gguf |
|---|---|---|---|
| **pack8** | **native (BF16/MXFP4/MXFP8)** | **IQ1_S** (106.2 GB, 1.56 bpw) | `DSV41-UNC-v3-iq1s.gguf`, 325.8 GB — **current production since 23:57** |
| pack4 | MXFP4 (native) | MXFP4 | `DSV41-UNC-v3.gguf`, 508.4 GB — previous native production |
| pack5 | Q6_K | IQ2_XXS | `DSV41-UNC-v4-iq2.gguf`, 359.7 GB |
| pack6 | Q6_K | IQ2_XXS | `DSV41-UNC-v5-q6k.gguf`, 352.0 GB |
| pack7 | Q4_K | IQ2_XXS | `DSV41-UNC-v7-q4k.gguf`, 350.3 GB — previous quantized production |

Production switched to pack8 at 23:57: IQ1_S experts (106.2 GB) fit entirely in the two cards' VRAM;
native dense/Engram stay on CPU. Quality is below pack4; the fast-attn experiment remains on. A controlled
decode A/B (2026-11-04, same 45-token prompt, 256 generated tokens, DSpark on) measured:
**pack8 IQ1_S 26.2 / 26.7 / 26.6 tok/s** vs **pack7 IQ2_XXS 35.2 / 36.3 / 36.6 tok/s** (prompt
16.6–17.4 vs 21.4–23.1 tok/s). IQ1_S does **not** win: it fills VRAM but its dequant/matvec is slower,
and DSpark accepts fewer drafts (167/220 vs 184/224 in that test). The 12.9 tok/s first short request
was a cold 14-token sample, not the stable decode rate.

### IQ1_S pack build (2026-11-03)

From the native v3 GGUF (experts MXFP4), with the IQ1_S support added to `tools/ds4_requant_gguf2.py`
and `tools/requant_gguf_tensor.cpp` (rebuild `requant_all` against llama.cpp's ggml):

```bash
# 120 expert tensors, 288.8 GB MXFP4 -> 106.2 GB IQ1_S; 66.6 min on this box with --jobs 64
python3 ds4_requant_gguf2.py --src DSV41-UNC-v3.gguf --dst DSV41-UNC-v3-iq1s.gguf \
  --mode experts --target IQ1_S --jobs 64 --requant requant_all

# pack only (the engine reads weights from the GGUF; no experts.bin)
python3 tools/dsv41/pack_gguf.py --model /4T-02/dsv41-conv \
  --shards DSV41-UNC-v3-iq1s.gguf \
  --dspark-gguf dspark-UNC-bf16.gguf --dspark-noise-token 128799 --dspark-markov-rank 256 \
  --mmproj mmproj-UNC-bf16.gguf --out /4T-02/dsv41-conv/pack8

# routing profile/slots are quant-independent
cp -a pack4/expert-profile.bin pack4/slots pack8/
```

`pack8` is 4.1 MB of metadata; the 325.8 GB GGUF holds dense/Engram + IQ1_S experts. VRAM after load:
60.5 + 55.5 GB of 2x64 GB; RAM ~146 GB.

## 9. Known gaps

- vision + expert-TP is untested;
- the parallel build has been exercised up to 10k-token prompts (12 mixed API requests, 0 failures);
  beyond that nothing has been validated;
- the matrix-core attention port is **implemented and validated on the box**: `sources/dsv41-20261031-gfx928/patches/0006-mfma-attention.patch`
  uses MMAC on gfx928 (DTK has no MFMA there), the layout gate is ALL OK and 4K/10K ids match. It is **not a win**:
  a controlled 4K A/B/A/B showed no gain and 10K only about +2.8 %; the all-prompt KL mean is 0.0255
  (last-position 0.00131). Keep it off; 32K is still unmeasured: `docs/mfma-attention-port.md` section 10.
