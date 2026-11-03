# Patch series for the strata engine on gfx928

Applied **in order** to a clean checkout of `https://github.com/Niko1221/Strata` (the release closest
to 2026-10-31). After step 1 you have the baseline tree; after step 2 the tree that runs the
expert-index tensor parallelism used in production here.

| order | file | size | what it does |
|---|---|---|---|
| 1 | `baseline-gfx928-port.patch` | 18 KB | upstream -> baseline: the five changes that make the `strata-dsv41` engine build at all on the wave64 (gfx906-compatible) HIP backend — the wave32 backend rejects `gfx928` (`cmake/hip_backend.cmake:31`) and the wave64 backend excludes `STRATA_BUILD_DSV41` (`CMakeLists.txt:910`). |
| 2 | `expert-index-tp.patch` | 56 KB | baseline -> expert-TP: each member holds its own **whole** experts, the router table gets three states (pointer / CPU lane / peer), only the lead computes the shared expert, plus the four scratch buffers sized with the whole FFN width when `expert_tp_on()` — the fix for the `T > 2048` crash. Also carries the debug probes (`DS4_GEMMDBG`, `DS4_BUFDBG`, `DS4_SET*`, `DS4_RANGE*`, `DS4_PTRCHECK`, `DS4_YGUARD`, `DS4_SERIALIZE`), which are inert unless set. |
| — | `0004-allreduce-rs2.patch` | 1.2 KB | the `DS4_AR_RS2` all-reduce variant (used by production; also already in `../../patches/`). |

```bash
git clone https://github.com/Niko1221/Strata upstream
cd upstream && git checkout <release closest to 2026-10-31>
patch -p1 < .../baseline-gfx928-port.patch        # -> baseline tree
cp -a . ../strata-exp && cd ../strata-exp
patch -p1 < .../expert-index-tp.patch            # -> expert-TP tree
```

If either patch does not apply cleanly the upstream has moved: do not force it. Copy the two working
trees instead (98 MB each) — that is what `../../docs/deploy-new-machine.md` recommends, and it
guarantees the code that was actually measured.

The conversion patches (`../../patches/0001…0003`) are **not** engine patches: they belong to the
llama.cpp side of the FP8 -> GGUF pipeline described in `../../README.md`.

The opt-in matrix-core attention port is separate: apply
`sources/dsv41-20261031-gfx928/patches/0006-mfma-attention.patch` **after step 2** if you want to test
it. On gfx928 it uses **MMAC** (not MFMA); the layout gate and 16-token ids pass, but a controlled 4K
A/B/A/B shows **no prefill gain** (best MMAC ~2 % slower) and the all-prompt KL is still open — keep
`DS4_FAST_ATTN` unset in production. Details: `sources/dsv41-20261031-gfx928/README.md`.
