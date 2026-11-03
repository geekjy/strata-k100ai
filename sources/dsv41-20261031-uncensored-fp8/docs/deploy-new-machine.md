# Bringing this deployment up on another machine

Two routes: **copy the working set** (fast, exact, ~360 GB to move) or **rebuild from sources**
(small transfer, needs the matching upstream). Both end with the same command line.

## 0. Prerequisites on the new machine

| requirement | detail |
|---|---|
| DCUs | 2x Hygon K100_AI (`gfx928`) — the binaries here are compiled for that arch only. A different DCU needs its own `-DCMAKE_HIP_ARCHITECTURES`. |
| GPU driver | the Hygon driver (`hyhal`) installed **on the host**, exposed to the container as `/opt/hyhal` and `/usr/local/hyhal`. Check with `hy-smi`. |
| RAM | >= 200 GB free for the RAM expert store (this deployment uses 140 GB there; the box has 503 GB) |
| disk | >= 1 TB free for the model (pack7 GGUF 350 GB + dspark 8.5 + mmproj 1.0 + margin) |
| docker | with `--device /dev/kfd --device /dev/dri` working |
| toolkit | DTK 26.04 **inside the container image** (clang 17, `hipgdb`), not on the host |
| kernel | `kernel.core_pattern=/dev/null` (see the operations runbook, section 6.1) |

## 1. The container image

The image carries the whole toolkit (DTK 26.04, hipBLAS/Tensile, python, the serve layer). It is
26.2 GB, so move it as an image, not by rebuilding:

```bash
# on the old machine
docker save dcu-qwen3-5:perfect-0.0.1 | gzip -1 > dcu-qwen3-5.tar.gz
# on the new machine
gunzip -c dcu-qwen3-5.tar.gz | docker load
docker images | grep dcu-qwen3-5        # expect perfect-0.0.1
```

If the image cannot be copied, it is built with the DTK *Container Toolkit* from a stock Ubuntu
22.04 base + the DTK 26.04 tarball from `https://download.sourcefind.cn:65024/file/1/DTK-26.10/`
(or `DTK-26.04/`). That is a separate exercise — prefer `docker save`.

## 2. The model (route A: copy) — ~360 GB

Only pack7 and its files are needed; the other GGUFs in `dsv41-conv` (v3/v4/v5 = 1.2 TB) are
alternatives and can stay behind.

```bash
rsync -av --progress /4T-02/dsv41-conv/pack7/                 NEW:/4T-02/dsv41-conv/pack7/
rsync -av --progress /4T-02/dsv41-conv/DSV41-UNC-v7-q4k.gguf  NEW:/4T-02/dsv41-conv/
rsync -av --progress /4T-02/dsv41-conv/dspark-UNC-bf16.gguf   NEW:/4T-02/dsv41-conv/
rsync -av --progress /4T-02/dsv41-conv/mmproj-UNC-bf16.gguf   NEW:/4T-02/dsv41-conv/
```

`pack7/` (403 MB) is the config (`ds4.txt`) + tensor index (`tensors.txt`) + `engram.bin` +
`slots/` (394 MB) + `expert-profile.bin` + its own `tokenizer/`. **Do not omit
`expert-profile.bin`** — without it the VRAM expert plan degrades badly (measured: 635 instead of
7440 CPU-lane picks, i.e. the experts land in the wrong tier). The working start script points the
serve layer at `/4T-02/dsv41-conv/pack4/tokenizer`; `pack7/tokenizer` holds the same files, either
works as long as it exists.

## 3. The sources (route A: copy, 196 MB)

```bash
rsync -av /4T-02/strata-work/strata/      NEW:/4T-02/strata-work/strata/       # baseline tree
rsync -av /4T-02/strata-work/strata-exp/  NEW:/4T-02/strata-work/strata-exp/   # expert-TP tree
```

Also copy the scripts and configs (all in `/4T-02/strata-work/`):

```
serve_pack7_tp.sh  dsv41-serve-p7tp.json     # the production start (expert-TP)
serve_pack7.sh     dsv41-serve-p7.json       # the fallback (baseline engine)
hc.sh                                        # health + one real answer
ids17.txt ids_4k.txt ids_10k.txt             # the test prompts used for measurements
```

The two build trees (`build-dsv41-full`, `build-exp`, ~200 MB each) are optional — copy them only if
the new machine must run **without** a rebuild. They contain absolute paths to `/4T-02`, so they only
work at the same location.

## 4. The sources (route B: rebuild from upstream)

The trees here are **not git checkouts**, so the exact upstream commit is not recorded. The patches
in `sources/dsv41-20261031-uncensored-fp8/` were made against the upstream tree dated 2026-10-31:

| patch | what it does |
|---|---|
| `/4T-02/strata-work/k100ai-dsv41-gfx928.patch` (18 KB) | upstream -> the **baseline** tree (the gfx928 port) |
| `/4T-02/strata-work/exptp-wip.patch` (56 KB) | baseline -> the **expert-TP** tree |

```bash
git clone https://github.com/Niko1221/Strata upstream        # pick the release closest to 2026-10-31
cd upstream && git checkout <that release>
patch -p1 < k100ai-dsv41-gfx928.patch                        # -> baseline
cp -a . ../strata-exp && cd ../strata-exp
patch -p1 < exptp-wip.patch                                  # -> expert-TP
```

If either patch does not apply cleanly, stop and use route A — the trees are only 98 MB each and
route A guarantees the exact code that was measured.

## 5. Build

Both trees build the same way; only the directory differs. Inside a container with `/work` mounted
and the DTK environment sourced:

```bash
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

- `/work/llama` is the pinned ggml the engine links against; it must exist next to the trees (the
  build tree here uses it via `STRATA_GGML_DIR`).
- First build ~11 min at `-j 48`; the result is `/work/build-exp/strata-dsv41`.
- The baseline binary comes from the same command with `strata` / `build-dsv41-full`.
- If `/opt/dtk/include/hsa` is missing inside the container, link it once:
  `ln -sfn /opt/dtk/.hyhal/include/hsa /opt/dtk/include/hsa` (the runbook's builds do this).

## 6. Serve

```bash
# config (already in the copy): /4T-02/strata-work/dsv41-serve-p7tp.json
#   {"exe":"/4T-02/strata-work/build-exp/strata-dsv41",
#    "args":["--pack","/4T-02/dsv41-conv/pack7","--gpus","0,1","--tp","2","--dspark","on","--mmap-experts","on"],
#    "cwd":"/4T-02/strata-work","model_name":"DeepSeek-V4.1-Flash-1017"}

bash /4T-02/strata-work/serve_pack7_tp.sh        # starts, waits for /health, ~150-170 s
bash /4T-02/strata-work/hc.sh                    # health + one real answer
```

The start script runs `python3 -m serve.server --engine strata --config .../dsv41-serve-p7tp.json
--tokenizer /4T-02/dsv41-conv/pack4/tokenizer --port 8095 --host 0.0.0.0` in the container, with

```
-e DS4_EXPERT_TP=1 -e DS4_AR_RS2=1 -e DS4_ATTN_SPLIT=1
```

and `--restart unless-stopped`.

## 7. Acceptance on the new box

| check | expected |
|---|---|
| `curl -s localhost:8095/health` | `{"status":"ok", ... "loaded":true}` |
| a one-sentence prompt | a real answer (not empty) |
| engine log `prompt:` line at 4080 tokens | ~130-140 tok/s (expert-TP) |
| engine log `decode:` line | 24-27 tok/s with `--dspark on` |
| a 10k-token prompt | completes; ~180 tok/s prefill on the reference box |
| `hy-smi` during a long prompt | **both** cards busy (that is the point of expert-TP) |

If the numbers are much lower, check in this order: `DS4_ATTN_SPLIT` set, `--dspark on`, both cards
visible (`--gpus 0,1`), `expert-profile.bin` present in the pack, and `--tp 2` actually taking effect
(the log must say `tensor parallel 2 by expert index`).

## 8. What NOT to carry over blindly

- **do not** enable `DS4_FAST_ATTN` (traps on gfx928, see `docs/mfma-attention-port.md`);
- the debug probes (`DS4_GEMMDBG`, `DS4_BUFDBG`, `DS4_SET*`, `DS4_RANGE*`, `DS4_PTRCHECK`,
  `DS4_YGUARD`, `DS4_SERIALIZE`) are compiled in but off unless set — leave them off;
- `build-exp` contains the probes from the crash hunt; they cost nothing when unset, but a clean
  rebuild from `strata-exp` gives the same binary without them;
- if the new machine has a **different** DCU model, both the arch flag and possibly the attention
  kernels change — validate the token ids against this box before trusting the speed numbers.
