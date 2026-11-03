# Operational scripts for this source snapshot

These are the exact scripts that run on the bench box (`/4T-02/strata-work/`). They are copied
here so the deployment can be reproduced without the machine; paths inside them are absolute and
assume the layout described in `../docs/operations.md`.

| file | purpose |
|---|---|
| `serve_pack7_tp.sh` | **production**: expert-index TP (both cards per layer), **pack8 (IQ1_S experts, 1.56 bpw)** + DSpark, container `dsv41-serve`, port 8095, `--restart unless-stopped` (the script name is historical) |
| `dsv41-serve-p7tp.json` | the serve config it uses (engine path, args, model name; currently `--pack pack8`) |
| `serve_pack7.sh` | fallback: the baseline engine (layer split, no cross-card all-reduce) on pack7 |
| `dsv41-serve-p7.json` | the fallback's config |
| `hc.sh` | health check: container state, `/health`, one real answer, disk |
| `ids17.txt`, `ids_2100.txt`, `ids_3100.txt`, `ids_4k.txt`, `ids_10k.txt` | the token-id prompt files used for the measurements (17 / 2100 / 3100 / 4080 / 10200 tokens) |

Both serve scripts assume:

- the container image `dcu-qwen3-5:perfect-0.0.1`,
- `/4T-02/dsv41-conv/pack8` (+ `DSV41-UNC-v3-iq1s.gguf`, 325.8 GB, `dspark-UNC-bf16.gguf`,
  `mmproj-UNC-bf16.gguf`). Pack8 keeps the native dense/Engram and quantizes only the routed experts
  to IQ1_S (106.2 GB at 1.56 bpw), so all experts fit in the two cards' VRAM; quality is below the
  native pack4. The `expert-profile.bin`/`slots` are copied from pack4 (same routing),
- the engine binary at `/4T-02/strata-work/repro-build/build/strata-dsv41` (the clean v0.2.9
  reproduction; production since 2026-11-03, `DS4_FAST_ATTN=1` since 21:43), or
  `/4T-02/strata-work/build-exp/strata-dsv41` (the previous production binary) /
  `/4T-02/strata-work/build-dsv41-full/strata-dsv41` (baseline),
- the Python serve layer inside the image (`python3 -m serve.server`), which spawns the engine as a
  subprocess and restarts it if it dies.

Environment that matters (see `../docs/operations.md` section 5):

```
DS4_EXPERT_TP=1  DS4_AR_RS2=1  DS4_ATTN_SPLIT=1  DS4_FAST_ATTN=1   # production since 2026-11-03 21:43 (explicit experiment)
```

`DS4_FAST_ATTN` was unset before 2026-11-03 21:43. It is now enabled on the clean v0.2.9
reproduction (`repro-build/build/strata-dsv41`), so gfx928 takes the **MMAC** path. Caveat: the
controlled 4K A/B/A/B measured **no prefill gain**, 10K only about **+2.8 %**, and the all-prompt KL
mean is 0.0255 (last-position 0.00131). To roll back to scalar, restore
`/4T-02/strata-work/serve_pack7_tp.sh.bak-20261103-214315` and re-run it (or remove the
`-e DS4_FAST_ATTN=1` line and re-run). Details: `../docs/mfma-attention-port.md` section 10.
