#!/usr/bin/env python3
"""Requantize only the expert tensors of a GGUF, byte-copying everything else.

The engine's engram table (ggml type 43, MXFP8) is never re-encoded: this tool only touches
`*_exps.weight`, so the pack keeps its 8.25 bpw knowledge table while the experts drop to IQ2_XXS.
The arithmetic is ggml's own (requant_mxfp4_to_iq2xxs), one process per tensor.

    PYTHONPATH=$LLAMACPP/gguf-py python3 ds4_requant_gguf.py --src in.gguf --dst out.gguf
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

GGUF_MAGIC = 0x46554747
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
ALIGN = 32
MXFP4, IQ2_XXS = 39, 16      # ggml/engine: IQ2_XXS = 16, IQ4_NL = 20


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


# this gguf-py predates type 43 (the engine's MXFP8: 32 values in 33 bytes), so keep the sizes we use here
LOCAL_SIZES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 39: (32, 17), 43: (32, 33), 16: (256, 66)}


def size_of(typ, n):
    if typ in LOCAL_SIZES:
        blck, tsize = LOCAL_SIZES[typ]
    else:
        blck, tsize = GGML_QUANT_SIZES[GGMLQuantizationType(typ)]
    if n % blck:
        raise RuntimeError("type %d: %d values not a multiple of %d" % (typ, n, blck))
    return n // blck * tsize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--requant", default="/4T-02/strata-work/requant")
    ap.add_argument("--tmpdir", default="/4T-02/dsv41-conv/iq2-tmp")
    ap.add_argument("--jobs", type=int, default=96)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="keep the temp blobs from a previous run")
    args = ap.parse_args()

    kv_end, entries, info_end, head = parse_header(args.src)
    data_start = info_end + ((ALIGN - info_end % ALIGN) % ALIGN)
    src_size = os.path.getsize(args.src)

    for e in entries:
        e["n_values"] = 1
        for d in e["dims"]:
            e["n_values"] *= d
        e["old_size"] = size_of(e["type"], e["n_values"])
        e["is_expert"] = e["name"].endswith("_exps.weight")
        e["new_type"] = IQ2_XXS if e["is_expert"] else e["type"]
        e["new_size"] = size_of(IQ2_XXS, e["n_values"]) if e["is_expert"] else e["old_size"]

    experts = [e for e in entries if e["is_expert"]]
    for e in experts:
        if e["type"] != MXFP4:
            raise RuntimeError("%s is type %d, expected MXFP4 (%d)" % (e["name"], e["type"], MXFP4))
        if e["n_values"] % 256 or e["dims"][0] % 256:
            raise RuntimeError("%s: %d values, row %d (a multiple of 256 is required)"
                               % (e["name"], e["n_values"], e["dims"][0]))

    off = 0
    for e in entries:
        e["new_offset"] = off
        off += e["new_size"]
    old_total = sum(e["old_size"] for e in entries)

    print("header: kv_end %d, info_end %d, data_start %d" % (kv_end, info_end, data_start))
    if data_start + old_total != src_size:
        print("WARNING: tensors add to %d bytes, file holds %d (padding?)"
              % (data_start + old_total, src_size))
    print("tensors: %d (%d experts, %d copied)" % (len(entries), len(experts), len(entries) - len(experts)))
    print("data: %.2f GB -> %.2f GB" % (old_total / 1e9, off / 1e9))

    if args.dry_run:
        for e in entries[:2] + experts[:2]:
            print("  %-38s type %2d->%2d  %7.2f -> %7.2f GB  off %d -> %d"
                  % (e["name"], e["type"], e["new_type"], e["old_size"] / 1e9, e["new_size"] / 1e9,
                     e["offset"], e["new_offset"]))
        return 0

    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- stage 1: quantize the experts, --jobs at a time ----
    todo = [(i, e) for i, e in enumerate(experts)
            if not (args.reuse and (tmp / ("e%03d.bin" % i)).exists()
                    and (tmp / ("e%03d.bin" % i)).stat().st_size == e["new_size"])]
    if args.reuse:
        print("  reusing %d of %d expert blobs" % (len(experts) - len(todo), len(experts)))
    running = []
    done = 0
    while todo or running:
        while todo and len(running) < args.jobs:
            i, e = todo.pop(0)
            out = tmp / ("e%03d.bin" % i)
            cmd = [args.requant, args.src, str(data_start + e["offset"]), str(e["n_values"]),
                   str(e["dims"][0]), str(out)]
            running.append((e, out, subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)))
        time.sleep(0.5)
        for item in list(running):
            e, out, p = item
            if p.poll() is not None:
                running.remove(item)
                if p.returncode != 0:
                    raise SystemExit("requant failed: %s" % e["name"])
                got = out.stat().st_size
                if got != e["new_size"]:
                    raise SystemExit("%s: %d bytes, expected %d" % (e["name"], got, e["new_size"]))
                done += 1
                if done % 8 == 0 or done == len(experts):
                    el = time.time() - t0
                    print("  %d/%d experts, %.0f s elapsed, eta %.0f min"
                          % (done, len(experts), el, (len(experts) - done) * el / max(done, 1) / 60),
                          flush=True)

    # ---- stage 2: assemble ----
    print("assembling %s ..." % args.dst, flush=True)
    exp_index = {e["name"]: i for i, e in enumerate(experts)}
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
            if e["is_expert"]:
                with open(tmp / ("e%03d.bin" % exp_index[e["name"]]), "rb") as ft:
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
