# Patch series for the strata engine on gfx928

Applied **in order** to a clean checkout of `https://github.com/Niko1221/Strata` (v0.2.9,
`c62d5e3a`, the release closest to 2026-10-31). The one-command reproduction is in the gfx928 package:

```bash
sources/dsv41-20261031-gfx928/reproduce-clean-v029.sh /path/to/v029.tar.gz [--build]
```

It was verified on the bench box: the resulting tree is **hash-identical** to the `build-all` tree
that was benchmarked (full `diff -qr` count 0).

| order | file | size | what it does |
|---|---|---|---|
| 1 | `baseline-gfx928-port.patch` | 18 KB | upstream -> baseline: the five changes that make the `strata-dsv41` engine build at all on the wave64 (gfx906-compatible) HIP backend — the wave32 backend rejects `gfx928` (`cmake/hip_backend.cmake:31`) and the wave64 backend excludes `STRATA_BUILD_DSV41` (`CMakeLists.txt:910`). |
| 2 | `../../patches/0002-ds4-launch-probe.patch`, `../../patches/0003-ds4-stage-markers.patch`, `../../patches/0004-allreduce-rs2.patch` | 2.3 / 7.4 / 1.2 KB | the local launch/stage probes and the `DS4_AR_RS2` all-reduce variant (used by production). |
| 2b | `probes-base.patch` | 19 KB | the local probe base for `ds4_model.cu`, `ds4_kernels.cu` and `ds4_kernels.cuh`. `expert-index-tp.patch` needs it; without it the `ds4_model.cu` probe hunk rejects on a clean tree. |
| 3 | `expert-index-tp.patch` | 56 KB | baseline -> expert-TP: each member holds its own **whole** experts, the router table gets three states (pointer / CPU lane / peer), only the lead computes the shared expert, plus the four scratch buffers sized with the whole FFN width when `expert_tp_on()` — the fix for the `T > 2048` crash. |
| 4 | `../../../dsv41-20261031-gfx928/patches/0006-mfma-attention.patch` | 36 KB | optional matrix-core attention (MMAC on gfx928, MFMA on CDNA); **no production speed gain**, keep `DS4_FAST_ATTN` unset. |

```bash
git clone https://github.com/Niko1221/Strata upstream
cd upstream && git checkout <release closest to 2026-10-31>
patch -p1 < .../baseline-gfx928-port.patch        # -> baseline tree
patch -p1 < .../patches/0002-ds4-launch-probe.patch
patch -p1 < .../patches/0003-ds4-stage-markers.patch
patch -p1 < .../patches/0004-allreduce-rs2.patch
patch -p1 < .../probes-base.patch                 # -> the probe base expert expects
patch -p1 < .../expert-index-tp.patch            # -> expert-TP tree
patch -p1 < .../0006-mfma-attention.patch         # optional
```

If a patch does not apply cleanly the upstream has moved: do not force it. Copy the two working
trees instead (98 MB each) — that is what `../../docs/deploy-new-machine.md` recommends, and it
guarantees the code that was actually measured.

The conversion patches (`../../patches/0001…0003`) are **not** engine patches: they belong to the
llama.cpp side of the FP8 -> GGUF pipeline described in `../../README.md`. (The `0002`/`0003` used
above are the engine probe patches, not conversion patches.)
