# Matrix-core port of the attention (gfx928 MMAC / CDNA MFMA, K100_AI)

Status: **implemented, built and run on the box** (2026-11-03); the layout gate and 16-token ids pass,
but the controlled A/B/A/B shows **no prefill speedup** and the all-position KL is still open. Written on
the `strata-k100ai` fork while running DeepSeek-V4.1-Flash-1017 (pack7, expert-index TP) on 2x Hygon
K100_AI. Section 10 has the implementation, the real-machine results, and what is still open.

> **The box corrected the instruction, not the idea.** gfx928 has **no MFMA**: DTK 26.04 / clang 17
> rejects `__builtin_amdgcn_mfma_f32_16x16x16f16` with `needs target feature mai-insts`, and its matrix
> core is **MMAC** (`__builtin_amdgcn_mmac_f32_16x16x16f16` / `V_MMAC_F32_16X16X16_F16`, layout in
> `/opt/dtk/include/du_mma.hpp`). MMAC and MFMA share the A/B four-halves-per-lane layout but have
> **transposed accumulator maps**: MMAC C `row = lane & 15`, `column = (lane >> 4) + 4*i`; MFMA C
> `row = 4*(lane >> 4) + i`, `column = lane & 15`. The implementation therefore uses MMAC on gfx928 and
> keeps MFMA for the CDNA parts (gfx908/gfx90a/gfx940–942/gfx950). The rest of this document is the
> original design, with the corrected map in section 5 and the box results in section 10.

## 1. Why this is worth doing

`DS4_FAST_ATTN` selects `sparse_attn_wmma_kernel`, the chunked-prompt attention on the matrix
cores. It is **off on every architecture except gfx12**, because that is where the project's owner
measured it. The owner's own numbers (from the source comments in
`src/dsv41/ds4_kernels.cu`, around line 2128):

| machine | prompt | result |
|---|---|---|
| 4x R9700 (gfx1201), 2.25 bpw | 8K code prompt | prefill **1180 -> 1354 tok/s (+14.7 %)**; same ids, KL 0.10833 vs 0.10925 |
| Strix Halo (gfx1151), 2.00 bpw | 32K prompt | prefill **101.6 s -> 70.8 s (+43.6 %)**; the sparse-attention stage **38.3 s -> 7.0 s (5.5x)**; same greedy first token |

So on a long prompt the matrix-core path takes the attention stage down by 5x. That is the single
largest remaining prefill lever on this box.

What it does **not** help: **decode**. The owner's comment is explicit — the decode path stays on
the scalar kernels (that is `sparse_attn_kernel` / the split-KV variant behind `DS4_ATTN_SPLIT`,
which is already enabled in production here and measured +7.8 % decode in an A/B).

Current state on this box (pack7, expert-index TP, DSpark on, `DS4_ATTN_SPLIT=1`):

- prefill: 4080 tokens 138 tok/s, 10200 tokens 183 tok/s
- decode: 24-27 tok/s
- `DS4_PROF` per decode token (sum of both cards): attention 50.78 ms, between layers 32.39,
  down+post 18.93, routing+shared 13.47, experts 6.29, waiting for the CPU lane 3.17

## 2. Why `DS4_FAST_ATTN=1` crashes here

It is not a memory bug and not a misconfiguration. `src/dsv41/ds4_wmma.cuh` is a two-function
abstraction over the matrix core:

```cpp
inline __device__ f8v set_wmma(h16v a, h16v b, f8v c) {
#if defined(__gfx1100__) || ... || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();          // <- gfx928 lands here
#endif
}
```

gfx928 matches none of the `__gfx11xx__` / `__gfx12xx__` macros, so the kernel executes
`__builtin_trap()` on its first matrix step. The runtime reports that as a VMFault, which is why it
looked like a memory fault at first.

## 3. Why a plain intrinsic swap will not work

Both sides compute the same 16x16 tile with a 16-deep K step, but the **lane-to-data mapping is
different**:

| | gfx11/gfx12 (WMMA, wave32) | gfx928 (MMAC) / CDNA (MFMA), wave64 |
|---|---|---|
| instruction | `wmma_f32_16x16x16_f16_w32` | `mmac_f32_16x16x16f16` / `mfma_f32_16x16x16f16` |
| A operand per lane | 16 halves (one full row) | 4 halves (scattered) |
| B operand per lane | 16 halves (one full column) | 4 halves (scattered) |
| accumulator per lane | 8 floats | 4 floats |
| wave coverage | 32 x 8 = 256 = 16x16 | 64 x 4 = 256 = 16x16 |

The kernel exploits the WMMA layout heavily: it loads its own row of Q and its own column of K into
`h16v a, b`, calls `set_wmma`, and then scatters the eight results with
`set_wmma_row(i, hi)` / `column = lane % 16`. None of that holds under MMAC/MFMA (see section 5 for the
two accumulator maps).

## 4. The design

Two properties of the existing kernel make the port tractable:

1. **Q and K are already staged in shared memory** (`__shared__ uint16_t qs[kFaHeads * kFaLd]`,
   `ks[kFaKeys * kFaLd]`), so the matrix-core operands can be gathered straight out of shared memory —
   no warp shuffles and no re-staging.
2. **A wave64 can cover two 16x16 tiles at once**: 2 x 4 = 8 accumulator floats per lane, which is
   exactly the `f8v` the kernel already carries. Keeping `f8v` keeps `o[4]` (the 512-dim output
   accumulation) and the softmax bookkeeping unchanged.

So the plan is:

1. **New header `src/dsv41/ds4_mfma.cuh`** (gfx9 only), mirroring `ds4_wmma.cuh`'s role:
   - `mma_scores(const uint16_t* qs, const uint16_t* ks, int ld, int kk, ...)` — two matrix-core calls,
     one per 16-key tile, operands gathered from `qs` / `ks` in the family's layout;
   - `mma_pv(...)` for the second product (P x V), same trick;
   - `mma_acc_row(i, lane)` / `mma_acc_col(i, lane)` replacing `set_wmma_row(i, hi)` (the accumulator
     map of the family: MMAC on gfx928, MFMA on CDNA).
2. **Kernel changes in `sparse_attn_wmma_kernel`** (`src/dsv41/ds4_kernels.cu`, ~line 1153):
   - `kFaKeys` 16 -> 32 under gfx9 (the wave covers two key tiles);
   - the two call sites `set_wmma(a, b, acc)` become `mma_scores(...)` / `mma_pv(...)` behind
     `#if defined(STRATA_DS4_MATRIX_CORE)`, the WMMA path stays byte-identical for gfx11/gfx12;
   - the block's head tile (`kFaHeads = 16`) and the shared-memory strides (`kFaLd = 520`) stay.
3. **Dispatch** in `fast_attn_enabled()`: on gfx9, `DS4_FAST_ATTN` should mean the matrix-core path (and
   stay opt-in until it is validated). `device_is_gfx12()` gains a sibling `device_is_gfx9()`, which
   must include gfx928 (MMAC) and the CDNA parts (MFMA), not the gfx9 parts without matrix cores.

## 5. The one thing that has to be right first

The **exact (lane, register) -> (row, column) mapping of the matrix core here** (MMAC on gfx928,
MFMA on CDNA) must be taken from the vendor's header/ISA reference and then *verified empirically*, not
guessed. Before touching the attention kernel, write a standalone unit test:

- fill A and B with a known pattern (e.g. `A[r][k] = r + k`, `B[k][c] = r - k`) in shared memory,
- run one matrix-core step, write every lane's four accumulators to global memory,
- compare against a CPU reference product, and derive/confirm the mapping table.

That test is the whole risk of the port: once the mapping and the operand gather are right, the
attention kernel follows mechanically. Getting them wrong produces plausible-looking but wrong
scores, which the id comparison would catch only sometimes.

The A/B gather is **the same** for MMAC and MFMA; only the accumulator map differs. For gfx928's
`V_MMAC_F32_16X16X16_F16` (the implementation's primary path, taken from `/opt/dtk/include/du_mma.hpp`):

| matrix | element held by a lane |
|---|---|
| A (M x K) | lane `16 * (k / 4) + m`, halves `k % 4` in order: `m = lane & 15`, `k = 4 * (lane >> 4) + i`, `i = 0..3` |
| B (K x N) | lane `16 * (k / 4) + n`: `n = lane & 15`, `k = 4 * (lane >> 4) + i`, `i = 0..3` |
| C (M x N) | register `i` is `C[lane & 15][(lane >> 4) + 4*i]`: **row = lane & 15**, **column = (lane >> 4) + 4*i** |

For the CDNA parts' `V_MFMA_F32_16X16X16F16` the A/B rows are identical and C is transposed relative
to MMAC: register `i` is `C[4*(lane >> 4) + i][lane & 15]`. The header selects the family by
`__gfx928__` -> MMAC, CDNA macros -> MFMA; the layout test checks the device's actual map.

The unit test is `tests/dsv41/mfma_layout_test.cpp` (new in
`sources/dsv41-20261031-gfx928/patches/0006-mfma-attention.patch`, built as `ds4_mfma_test`): one
16x16x16 step, then `mma_scores` and `mma_pv`, each against a CPU product of exactly representable
integers. **On the box it printed `matrix-core layout test: ALL OK` (three checks, 0 mismatches).**
Run it on the box before trusting `DS4_FAST_ATTN=1`.

## 6. Validation protocol (what "done" means)

1. **ids identical**: the same prompt through the scalar path and the matrix-core path must produce the
   same greedy token ids (this is the owner's own acceptance criterion for the gfx12 path). **Box: 4K
   `671,6102` and 17-token `671,6102,294,8760` are identical.**
2. **KL gate**: the project guards this path with KL <= 0.006 against the reference logits
   (`--dump-logits` gives the tensor). The gfx12 measurement was KL 0.0003. **Box: the last-position
   (generation) logits are KL 0.00131, under 0.006; the mean over all 3601 prompt positions is 0.0255
   (max 1.595, 592 rows > 0.006), which fails the gate if applied to every position.** Most positions
   are bit-identical (median KL 7.7e-7), so this is not plain FP16 rounding; see section 10.
3. **Speed**: prefill tok/s on a 4K and a 10K prompt, A/B/A/B with a median (single runs on this box
   vary by +/-20 %). **Box: a controlled 4K A/B/A/B (`--window 512`) showed no gain** — scalar 95,503 /
   95,081 ms (42.72 / 42.91 tok/s), MMAC 97,291 / 114,571 ms (41.94 / 35.61 tok/s); the best MMAC run
   was about 2 % slower. An earlier 118,187 -> 97,096 ms "pair" was a cold first run and is retracted.
   10K was not run.**
4. **Regression**: `DS4_FAST_ATTN` unset must keep the scalar path bit-identical. **Box: ids match
   with the scalar path; a full tensor comparison was not run.**

## 7. Cost and risks

- **3-5 working days** for a kernel engineer who can iterate on the box (the layout test, the two
  products, the two-tile bookkeeping, the validation runs).
- Risks: the PV product has a different shape (16 keys x 512 dims per head) and may not map onto
  the same two-tile trick; MMAC/MFMA need 64-lane waves, so the softmax code that assumes 32 lanes
   (`__shfl_xor_sync(0xffffffffu, ...)`, `lane = tid & 31`) must be revisited; the kernel's
  `__syncthreads()` / bank-conflict padding (`kFaLd = 520`) was tuned for the WMMA layout.
- Expected payoff if it lands: **+30-45 % prefill on long prompts** (the owner's 32K number),
  decode unchanged. **Measured first: no gain on a 4K, `--window 512` controlled A/B/A/B; the best
  MMAC run was about 2 % slower than scalar. The owner's long-prompt gain did not reproduce here.**

## 8. Cheaper things to try before committing to the port

- measure the attention share of *our* prefill (the `DS4_PROF` breakdown above is the decode path;
  the chunked prompt path needs its own instrumentation) — if attention is not ~40 % of the prefill
  time, the ceiling is lower than the table in section 1 suggests;
- `--prefill-chunk` sizes: 4096 (default rule), 8192 was measured *worse* here (109 vs 138 tok/s);
- `DS4_FAT_ROWS` (busy experts as FP16 GEMMs): measured no gain here (136 vs 138 tok/s);
- more VRAM per card for experts: measured dead — the KV cache of this model is tiny
  (0.03 -> 0.01 GB when max-context drops from 32K to 8K), so the expert count moves only
  5803 -> 5809.

## 9. Files touched by the port

| file | role |
|---|---|
| `src/dsv41/ds4_wmma.cuh` | the abstraction being replaced (keep it; add the gfx9 sibling) |
| `src/dsv41/ds4_kernels.cu` | `sparse_attn_wmma_kernel` ~1153, its launch, `fast_attn_enabled()`, `device_is_gfx9()` |
| new: `src/dsv41/ds4_mfma.cuh` | MMAC (gfx928) / MFMA (CDNA) operands, the two products, the accumulator maps |
| new: `tests/dsv41/mfma_layout_test.cpp` | the layout gate, built as `ds4_mfma_test` (see section 5) |

## 10. Implementation and box results (2026-11-03)

The port is in **`sources/dsv41-20261031-gfx928/patches/0006-mfma-attention.patch`** (apply after the
baseline gfx928 patch — `patches/0001..0005` or the combined
`tools/patches-wip/baseline-gfx928-port.patch` — and after `tools/patches-wip/expert-index-tp.patch` on
the production tree).
It touches four files:

| file | change |
|---|---|
| new `src/dsv41/ds4_mfma.cuh` | `h4v`/`f4v`, `mma_tile` (one `__builtin_amdgcn_mmac_f32_16x16x16f16` on gfx928, one `__builtin_amdgcn_mfma_f32_16x16x16f16` on CDNA), `mma_acc_row`/`mma_acc_col`, the A/B gathers, `mma_scores` and `mma_pv`. Empty on non-matrix-core parts. |
| `src/dsv41/ds4_kernels.cu` | `kFaKeys` 32 / `kFaWaves` 4 under `STRATA_DS4_MATRIX_CORE` (16 / 8 unchanged for WMMA); the score, softmax, value and output sections get a second form behind `#if`; `device_is_gfx9()`; `fast_attn_enabled()` keeps gfx9 opt-in. |
| new `tests/dsv41/mfma_layout_test.cpp` | the layout gate of section 5, built as `ds4_mfma_test` (skips with exit 77 on a non-matrix-core device). |
| `CMakeLists.txt` | adds `ds4_mfma_test` when `CMAKE_HIP_ARCHITECTURES` names a matrix-core part. |

The wave layout, in the attention kernel (256 threads = 4 wave64s):

- a wave covers **128 dims** (two 64-dim slices) instead of a warp's 64;
- the key tile is **32 keys**: `mma_scores` produces the keys 0–15 and 16–31 tiles in one `f8v`
  (accumulators 0–3 and 4–7), and `part[4][16][32]` sums the four waves;
- the online softmax gives a wave **2 heads with 32 lanes a head** (one lane a key), two passes for
  the 16 heads; the WMMA path's warp-per-2-heads, 16-lane form is untouched;
- `mma_pv` computes **two 16-dim value tiles** per `f8v` (the same `P` for both), 4 `f8v` = the wave's
  128 dims; the two key groups (0–15, 16–31) accumulate into the same `f8v`;
- the output scatter on gfx928/MMAC is **`head = lane & 15`** and
  **`dim = wave * 128 + f * 32 + (lane >> 4) + 4*i`** (and `+16` for the second tile); on CDNA/MFMA
  it is `head = 4*(lane >> 4) + i`, `dim = wave * 128 + f * 32 + (lane & 15)`;
- a partial last tile zeroes the tail rows of `ks` (up to 31 rows x 512 halves): `ps` is 0 for those
  keys anyway, but 0 times an uninitialised half is not a number.

Shared memory: 61,632 B under `STRATA_DS4_MATRIX_CORE` (16,640 Q + 33,280 K + 8,192 `part` + 2,048 `sc` +
1,024 `ps` + 256 rows + 192 softmax state), under the 64 KiB a gfx9 workgroup has; a `static_assert`
enforces it. The WMMA path's 43,328 B are unchanged.

Dispatch: `device_is_gfx9()` is true for **gfx928 and the CDNA matrix-core parts** (gfx900/902/904/906/909/90c
are gfx9 without matrix cores and must not take the WMMA `__builtin_trap()`); `fast_attn_enabled()`
returns `env > 0` there. **Unset `DS4_FAST_ATTN` keeps the scalar path bit-identical**; `=1` is the
opt-in and does not change any other architecture's default.

### Build and box results

Built with DTK 26.04 / clang 17.0.0, `CMAKE_HIP_ARCHITECTURES=gfx928`; the engine and `ds4_mfma_test`
both linked. The first DTK build failed exactly on the design's premise:

```
error: '__builtin_amdgcn_mfma_f32_16x16x16f16' needs target feature mai-insts
```

gfx928's matrix core is MMAC. The header was changed to select
`__builtin_amdgcn_mmac_f32_16x16x16f16` under `__gfx928__` and keep MFMA for CDNA. The standalone
layout test then passed on the card:

```
matrix-core layout test on gfx928:sramecc+:xnack- (MMAC)
1. one 16x16x16 step (MMAC map): ok (0 mismatches)
2. mma_scores (QK, 32 keys, 4 waves): ok (0 mismatches)
3. mma_pv (PV, 32 keys, 4 waves): ok (0 mismatches)
matrix-core layout test: ALL OK
```

A raw MMAC probe matched the CPU product with 0 mismatches; random-value precision against FP64 was
1.75e-7 max absolute error for one step and 2.05e-6 for 16 accumulated steps, i.e. the instruction
does accumulate in FP32.

End-to-end, on pack7 with expert-index TP, `--tp 2`, `--dspark none`, same binary:

| run | scalar (`DS4_FAST_ATTN=0`) | MMAC (`=1`) | result |
|---|---|---|---|
| 4K A/B/A/B, `--window 512` | **95,503 / 95,081 ms** (42.72 / 42.91 tok/s) | **97,291 / 114,571 ms** (41.94 / 35.61 tok/s) | **no gain**; best MMAC ~2 % slower |
| earlier cold first scalar | 118,187 ms | 97,096 ms | **retracted**: cold run, not an A/B |
| generated ids (4K, 16 tokens) | `671,6102,294,8760,344,11111,16,1` | same | identical |
| generated ids (17 tokens) | `671,6102,294,8760` | same | identical (window path, not the fast path) |

`--window 512` forces the chunked prompt path; the MMAC runs logged 560 launches of
`ds4: fast attn T=512 heads=32 gen=1`. The warm-up run (scalar, 103,318 ms) is excluded from the A/B/A/B.
The earlier single pair (118,187 -> 97,096 ms, "+21.7 %") was the first run after stopping the service and
was taken while the caches/clocks were still cold; it must not be quoted as a speedup. In the controlled
runs the **best MMAC time is slower than scalar**, so this is a negative result for the 4K workload: the
extra staging/conversion work is not paid back by the matrix core here. The 10K/32K cases remain
unmeasured; the owner's long-prompt gain did not reproduce on this box.

KL is the other open part. On the 4K dump (3601 rows) the all-position mean is **0.0255**, max 1.595,
592 rows > 0.006, and 101 rows have a different argmax. The **last-position logits — the ones used to
generate — are KL 0.00131**, under the 0.006 gate, and the last 10 rows all keep their argmax. Most
positions are bit-identical (median KL 7.7e-7), so the divergence is concentrated in a subset, not
explained by uniform FP16 rounding. This is why `DS4_FAST_ATTN=1` stays opt-in: the generation
oracle passes, the all-position gate does not.

### Off-box checks

- The header/kernel/test were syntax-checked with clang 23 against a stub HIP header and the
  matrix-core builtin; this caught the `uint16_t[16][32]` -> `const uint16_t*` call error.
- A CPU model of the **MMAC** branch (`sources/dsv41-20261031-gfx928/mfma-attention-model.py`: the same
  lane map, wave slices, 32-key stage, 16 tiles, online softmax and both products) matches a plain
  attention reference to **2.7e-4** max absolute error on 500 keys (last tile partial). This checks the
  bookkeeping built on the map; the map itself is the layout test's job.
- The patch applies cleanly to a fresh v0.2.9 tree with the baseline gfx928 patch, and to the
  expert-index-TP kernel; the four changed files reproduce the expected SHA-256s.

### Still open

- **All-position KL**: mean 0.0255 > 0.006; find the 592 divergent rows before making the path the
  default. The concentration (most rows identical, some far off) suggests a row-selection or
  tile-boundary issue rather than general rounding.
- **A proper A/B/A/B on 10K/32K prompts**; the 4K A/B/A/B already shows **no gain**, so a long-prompt
  retest is needed before spending more on this path.
- **Why the gain disappeared**: measure the attention share of this prefill (the fast kernel adds
  32-key staging and FP16 conversions; on the 4K run it is a net loss). The owner's 32K/5.5x attention
  stage result does not reproduce on this box.
- **Production default window**: the fast path was forced with `--window 512`; confirm it triggers for
  the service's normal long prompts.
- **Full tensor regression** of `DS4_FAST_ATTN` unset vs `=0` (ids match; tensors were not compared).
- **Byte-level reproducibility** (`clean_rebuild_verified` is false).

### Commands for the box

```bash
# after the baseline patch + expert-index-TP, in the patched tree
patch -p1 < 0006-mfma-attention.patch
./build-dsv41-gfx928.sh                     # also builds ds4_mfma_test when the header is present
bash run-mfma-layout-test.sh                # must print "matrix-core layout test: ALL OK"

# 4K A/B (stop the serve container first; restart it afterwards)
DS4_FAST_ATTN=1 strata-dsv41 --pack /4T-02/dsv41-conv/pack7 --ids @ids_4k.txt \
  --window 512 --n 2 --max-context 8192 --gpus 0,1 --tp 2 --dspark none --mmap-experts on
```

If the layout test fails, stop: the map in section 5 or an operand gather is wrong, and a
plausible-looking but wrong attention is worse than the scalar path.
