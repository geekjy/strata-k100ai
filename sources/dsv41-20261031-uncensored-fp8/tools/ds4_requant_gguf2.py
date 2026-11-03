#!/usr/bin/env python3
"""Requantize one class of tensors of a GGUF, byte-copying everything else.

Two modes, both leaving every other tensor untouched:
  --mode experts --target IQ2_XXS   the routed expert stacks (MXFP4 -> IQ2_XXS)
  --mode dense   --target Q6_K      the BF16 matrices (BF16 -> Q6_K), token_embd kept as it is

The arithmetic is ggml's own (requant_gguf_tensor), one process per tensor.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

GGUF_MAGIC = 0x46554747
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
ALIGN = 32
# ggml type ids and (block, bytes) — this gguf-py predates type 43 (the engine's MXFP8)
MXFP4, BF16, IQ2_XXS, Q6_K, MXFP8 = 39, 30, 16, 14, 43
IQ1_S = 19
# the matrices that carry the per-token bandwidth: the attention projections and the shared expert
DENSE_SUFFIXES = ("attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
                  "attn_output_a.weight", "attn_output_b.weight",
                  "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                  "attn_compressor_kv.weight", "attn_compressor_gate.weight")
TYPES = {
    0:  ("F32",     (1, 4)),
    1:  ("F16",     (1, 2)),
    30: ("BF16",    (1, 2)),
    39: ("MXFP4",   (32, 17)),
    43: ("MXFP8",   (32, 33)),
    19: ("IQ1_S", (256, 50)),
    16: ("IQ2_XXS", (256, 66)),
    14: ("Q6_K",    (256, 210)),
    12: ("Q4_K",    (256, 144)),
}


def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise RuntimeError("short read (%d < %d)" % (len(b), n))
    return b


def parse_header(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", read_exact(f, 8))
        if magic != GGUF_MAGIC:
            raise RuntimeError("not a GGUF file")
        if version != 3:
            raise RuntimeError("expected GGUF v3, got %d" % version)
        n_tensors, n_kv = struct.unpack("<QQ", read_exact(f, 16))

        def skip_value(vtype):
            if vtype in SCALAR:
                f.seek(SCALAR[vtype], 1)
            elif vtype == 8:
                (n,) = struct.unpack("<Q", read_exact(f, 8))
                f.seek(n, 1)
            elif vtype == 9:
                atype, n = struct.unpack("<IQ", read_exact(f, 12))
                for _ in range(n):
                    skip_value(atype)
            else:
                raise RuntimeError("unknown metadata value type %d" % vtype)

        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", read_exact(f, 8))
            f.seek(klen, 1)
            (vtype,) = struct.unpack("<I", read_exact(f, 4))
            skip_value(vtype)
        kv_end = f.tell()

        entries = []
        for _ in range(n_tensors):
            start = f.tell()
            (nlen,) = struct.unpack("<Q", read_exact(f, 8))
            name = read_exact(f, nlen).decode("utf-8")
            (n_dims,) = struct.unpack("<I", read_exact(f, 4))
            dims = struct.unpack("<%dQ" % n_dims, read_exact(f, 8 * n_dims))
            type_pos = f.tell()
            (typ,) = struct.unpack("<I", read_exact(f, 4))
            off_pos = f.tell()
            (off,) = struct.unpack("<Q", read_exact(f, 8))
            entries.append({"name": name, "dims": dims, "type": typ, "offset": off,
                            "start": start, "end": f.tell(), "type_pos": type_pos, "off_pos": off_pos})
        info_end = f.tell()
        f.seek(0)
        head = f.read(info_end)
    return kv_end, entries, info_end, head


def size_of(typ, n):
    _, (blck, tsize) = TYPES[typ]
    if n % blck:
        raise RuntimeError("type %s: %d values not a multiple of %d" % (TYPES[typ][0], n, blck))
    return n // blck * tsize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=["experts", "dense"], required=True)
    ap.add_argument("--target", required=True, choices=["IQ2_XXS", "IQ1_S", "Q6_K", "Q4_K"])
    ap.add_argument("--requant", default="/4T-02/strata-work/requant_all")
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    tgt = {"IQ2_XXS": IQ2_XXS, "IQ1_S": IQ1_S, "Q6_K": Q6_K, "Q4_K": 12}[args.target]
    src_type = MXFP4 if args.mode == "experts" else BF16
    tmp = Path(args.tmpdir or ("/4T-02/dsv41-conv/req-%s" % args.target.lower()))

    kv_end, entries, info_end, head = parse_header(args.src)
    data_start = info_end + ((ALIGN - info_end % ALIGN) % ALIGN)
    for e in entries:
        e["n_values"] = 1
        for d in e["dims"]:
            e["n_values"] *= d
        e["old_size"] = size_of(e["type"], e["n_values"])
        if args.mode == "experts":
            e["do"] = e["name"].endswith("_exps.weight") and e["type"] == MXFP4
        else:
            # only the bandwidth-carrying matrices: the engine requires F16 for indexer.weights_proj,
            # and the head / embedding stay BF16 (a few ms a token, not worth the risk)
            e["do"] = (e["type"] == BF16 and e["dims"][0] % 256 == 0
                       and e["name"].endswith(DENSE_SUFFIXES))
        e["new_type"] = tgt if e["do"] else e["type"]
        e["new_size"] = size_of(tgt, e["n_values"]) if e["do"] else e["old_size"]

    work = [e for e in entries if e["do"]]
    for e in work:
        if e["type"] != src_type:
            raise RuntimeError("%s: type %d, expected %d" % (e["name"], e["type"], src_type))
    off = 0
    for e in entries:
        e["new_offset"] = off
        off += e["new_size"]
    old_total = sum(e["old_size"] for e in entries)
    print("tensors: %d, converting %d (%s of %.2f GB -> %.2f GB), data %.2f -> %.2f GB"
          % (len(entries), len(work), args.target, sum(e["old_size"] for e in work) / 1e9,
             sum(e["new_size"] for e in work) / 1e9, old_total / 1e9, off / 1e9))
    if args.dry_run:
        for e in work[:4]:
            print("  %-40s %s -> %s  %.3f -> %.3f GB" % (e["name"], TYPES[e["type"]][0],
                                                          TYPES[e["new_type"]][0],
                                                          e["old_size"] / 1e9, e["new_size"] / 1e9))
        return 0

    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = [(i, e) for i, e in enumerate(work)
            if not (args.reuse and (tmp / ("t%04d.bin" % i)).exists()
                    and (tmp / ("t%04d.bin" % i)).stat().st_size == e["new_size"])]
    if args.reuse:
        print("  reusing %d of %d blobs" % (len(work) - len(todo), len(work)))
    running = []
    done = 0
    while todo or running:
        while todo and len(running) < args.jobs:
            i, e = todo.pop(0)
            out = tmp / ("t%04d.bin" % i)
            cmd = [args.requant, args.src, str(data_start + e["offset"]), str(e["n_values"]),
                   str(e["dims"][0]), TYPES[src_type][0], TYPES[tgt][0], str(out)]
            running.append((e, out, subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)))
        time.sleep(0.3)
        for item in list(running):
            e, out, p = item
            if p.poll() is not None:
                running.remove(item)
                if p.returncode != 0:
                    raise SystemExit("requant failed: %s" % e["name"])
                if out.stat().st_size != e["new_size"]:
                    raise SystemExit("%s: %d bytes, expected %d" % (e["name"], out.stat().st_size, e["new_size"]))
                done += 1
                if done % 32 == 0 or done == len(work):
                    el = time.time() - t0
                    print("  %d/%d, %.0f s, eta %.1f min" % (done, len(work), el,
                          (len(work) - done) * el / max(done, 1) / 60), flush=True)

    print("assembling %s ..." % args.dst, flush=True)
    index = {e["name"]: i for i, e in enumerate(work)}
    CH = 64 << 20
    with open(args.src, "rb") as fi, open(args.dst, "wb") as fo:
        fo.write(head[:kv_end])
        for e in entries:
            raw = bytearray(head[e["start"]:e["end"]])
            rt = e["type_pos"] - e["start"]
            ro = e["off_pos"] - e["start"]
            raw[rt:rt + 4] = struct.pack("<I", e["new_type"])
            raw[ro:ro + 8] = struct.pack("<Q", e["new_offset"])
            fo.write(raw)
        fo.write(b"\x00" * ((ALIGN - fo.tell() % ALIGN) % ALIGN))
        if fo.tell() != data_start:
            raise SystemExit("header size mismatch: %d vs %d" % (fo.tell(), data_start))
        for e in entries:
            if e["do"]:
                with open(tmp / ("t%04d.bin" % index[e["name"]]), "rb") as ft:
                    while True:
                        b = ft.read(CH)
                        if not b:
                            break
                        fo.write(b)
            else:
                fi.seek(data_start + e["offset"])
                left = e["old_size"]
                while left:
                    b = fi.read(min(CH, left))
                    if not b:
                        raise SystemExit("short read copying %s" % e["name"])
                    fo.write(b)
                    left -= len(b)
    print("done: %s (%.2f GB) in %.1f min"
          % (args.dst, os.path.getsize(args.dst) / 1e9, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
