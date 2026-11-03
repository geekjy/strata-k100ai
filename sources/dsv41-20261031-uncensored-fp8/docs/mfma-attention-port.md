# MFMA port of the matrix-core attention (gfx928 / K100_AI)

Status: **design only — not implemented**. Written 2026-11-03 on the `strata-k100ai` fork while
running DeepSeek-V4.1-Flash-1017 (pack7, expert-index TP) on 2x Hygon K100_AI.

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

| | gfx11/gfx12 (WMMA, wave32) | gfx9 (MFMA, wave64) |
|---|---|---|
| instruction | `wmma_f32_16x16x16_f16_w32` | `mfma_f32_16x16x16f16` |
| A operand per lane | 16 halves (one full row) | 4 halves (scattered) |
| B operand per lane | 16 halves (one full column) | 4 halves (scattered) |
| accumulator per lane | 8 floats | 4 floats |
| wave coverage | 32 x 8 = 256 = 16x16 | 64 x 4 = 256 = 16x16 |

The kernel exploits the WMMA layout heavily: it loads its own row of Q and its own column of K into
`h16v a, b`, calls `set_wmma`, and then scatters the eight results with
`set_wmma_row(i, hi)` / `column = lane % 16`. None of that holds under MFMA.

## 4. The design

Two properties of the existing kernel make the port tractable:

1. **Q and K are already staged in shared memory** (`__shared__ uint16_t qs[kFaHeads * kFaLd]`,
   `ks[kFaKeys * kFaLd]`), so the MFMA operands can be gathered straight out of shared memory —
   no warp shuffles and no re-staging.
2. **A wave64 can cover two 16x16 tiles at once**: 2 x 4 = 8 accumulator floats per lane, which is
   exactly the `f8v` the kernel already carries. Keeping `f8v` keeps `o[4]` (the 512-dim output
   accumulation) and the softmax bookkeeping unchanged.

So the plan is:

1. **New header `src/dsv41/ds4_mfma.cuh`** (gfx9 only), mirroring `ds4_wmma.cuh`'s role:
   - `mfma_scores(const uint16_t* qs, const uint16_t* ks, int l16, int kk, f8v& acc)`
     — two `__builtin_amdgcn_mfma_f32_16x16x16f16` calls, one per 16-key tile, operands gathered
     from `qs` / `ks` in the MFMA layout;
   - `mfma_pv(...)` for the second product (P x V), same trick;
   - `mfma_row(int i)` replacing `set_wmma_row(i, hi)` (the mapping of accumulator index to result
     row for the two-tile case).
2. **Kernel changes in `sparse_attn_wmma_kernel`** (`src/dsv41/ds4_kernels.cu`, ~line 1153):
   - `kFaKeys` 16 -> 32 under gfx9 (the wave covers two key tiles);
   - the two call sites `set_wmma(a, b, acc)` become `mfma_scores(...)` / `mfma_pv(...)` behind
     `#if defined(__gfx9__)`, the WMMA path stays byte-identical for gfx11/gfx12;
   - the block's head tile (`kFaHeads = 16`) and the shared-memory strides (`kFaLd = 520`) stay.
3. **Dispatch** in `fast_attn_enabled()` (~line 2171): on gfx9, `DS4_FAST_ATTN` should mean the
   MFMA path (and stay opt-in until it is validated). `device_is_gfx12()` (~2139) gains a sibling
   `device_is_gfx9()`.

## 5. The one thing that has to be right first

The **exact (lane, register) -> (row, column) mapping of `V_MFMA_F32_16X16X16F16`** must be taken
from the AMDGPU ISA reference and then *verified empirically*, not guessed. Before touching the
attention kernel, write a standalone unit test:

- fill A and B with a known pattern (e.g. `A[r][k] = r + k`, `B[k][c] = r - k`) in shared memory,
- run one MFMA step, write every lane's four accumulators to global memory,
- compare against a CPU reference product, and derive/confirm the mapping table.

That test is the whole risk of the port: once the mapping and the operand gather are right, the
attention kernel follows mechanically. Getting them wrong produces plausible-looking but wrong
scores, which the id comparison would catch only sometimes.

## 6. Validation protocol (what "done" means)

1. **ids identical**: the same prompt through the scalar path and the MFMA path must produce the
   same greedy token ids (this is the owner's own acceptance criterion for the gfx12 path).
2. **KL gate**: the project guards this path with KL <= 0.006 against the reference logits
   (`--dump-logits` gives the tensor). The gfx12 measurement was KL 0.0003.
3. **Speed**: prefill tok/s on a 4K and a 10K prompt, A/B/A/B with a median (single runs on this box
   vary by +/-20 %).
4. **Regression**: `DS4_FAST_ATTN` unset must keep the scalar path bit-identical.

## 7. Cost and risks

- **3-5 working days** for a kernel engineer who can iterate on the box (the layout test, the two
  products, the two-tile bookkeeping, the validation runs).
- Risks: the PV product has a different shape (16 keys x 512 dims per head) and may not map onto
  the same two-tile trick; MFMA needs 64-lane waves, so the softmax code that assumes 32 lanes
   (`__shfl_xor_sync(0xffffffffu, ...)`, `lane = tid & 31`) must be revisited; the kernel's
  `__syncthreads()` / bank-conflict padding (`kFaLd = 520`) was tuned for the WMMA layout.
- Expected payoff if it lands: **+30-45 % prefill on long prompts** (the owner's 32K number),
  decode unchanged.

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
| `src/dsv41/ds4_kernels.cu` | `sparse_attn_wmma_kernel` ~1153, its launch ~2623, `fast_attn_enabled()` ~2171, `device_is_gfx12()` ~2139 |
| new: `src/dsv41/ds4_mfma.cuh` | MFMA operands, the two products, the row mapping |
| new: a unit test for the layout | see section 5 |
