# Operational scripts for this source snapshot

These are the exact scripts that run on the bench box (`/4T-02/strata-work/`). They are copied
here so the deployment can be reproduced without the machine; paths inside them are absolute and
assume the layout described in `../docs/operations.md`.

| file | purpose |
|---|---|
| `serve_pack7_tp.sh` | **production**: expert-index TP (both cards per layer), pack7 + DSpark, container `dsv41-serve`, port 8095, `--restart unless-stopped` |
| `dsv41-serve-p7tp.json` | the serve config it uses (engine path, args, model name) |
| `serve_pack7.sh` | fallback: the baseline engine (layer split, no cross-card all-reduce) on the same pack |
| `dsv41-serve-p7.json` | the fallback's config |
| `hc.sh` | health check: container state, `/health`, one real answer, disk |
| `ids17.txt`, `ids_2100.txt`, `ids_3100.txt`, `ids_4k.txt`, `ids_10k.txt` | the token-id prompt files used for the measurements (17 / 2100 / 3100 / 4080 / 10200 tokens) |

Both serve scripts assume:

- the container image `dcu-qwen3-5:perfect-0.0.1`,
- `/4T-02/dsv41-conv/pack7` (+ `DSV41-UNC-v7-q4k.gguf`, `dspark-UNC-bf16.gguf`, `mmproj-UNC-bf16.gguf`),
- the engine binary at `/4T-02/strata-work/build-exp/strata-dsv41` (expert-TP) or
  `/4T-02/strata-work/build-dsv41-full/strata-dsv41` (baseline),
- the Python serve layer inside the image (`python3 -m serve.server`), which spawns the engine as a
  subprocess and restarts it if it dies.

Environment that matters (see `../docs/operations.md` section 5):

```
DS4_EXPERT_TP=1  DS4_AR_RS2=1  DS4_ATTN_SPLIT=1     # production
```

`DS4_FAST_ATTN` must stay unset here: without `0006-mfma-attention.patch` it traps on gfx928; with the
patch gfx928 uses **MMAC** (not MFMA), the layout gate and 4K/10K ids pass, but the controlled 4K
A/B/A/B measured **no prefill gain** and 10K only about **+2.8 %**; the all-prompt KL mean is 0.0255
(last-position 0.00131). Details: `../docs/mfma-attention-port.md` section 10.
