#!/usr/bin/env python3
"""CPU model of the attention kernel's gfx928 MMAC branch (ds4_kernels.cu / ds4_mfma.cuh).

It does not replace the on-box layout test: it models the ISA map the layout test confirms, from /opt/dtk/include/du_mma.hpp
(A/B four halves a lane, C row = lane&15, column = (lane>>4)+4i; CDNA's MFMA is the same A/B with the transposed C map).
What it checks is the bookkeeping built on top of that map: the two 16x16 tiles a wave carries, the 4 waves' 128-dim
slices, the 32-key stage across many tiles (with a partial last tile), the online softmax, the output rescale and the
value product, against a plain attention reference.
"""
import numpy as np

HEADS, D, LD = 16, 512, 520
KEYS = 32
TOTAL = 500          # many tiles, last one partial (20 keys)
SCALE = 0.125
SINK = -0.75

rng = np.random.default_rng(7)
Q = rng.standard_normal((HEADS, D)).astype(np.float16)
K = rng.standard_normal((TOTAL, D)).astype(np.float16)
V = K   # the DeepSeek attention stores one shared K = V row per token: the key tile is the value tile

# Q is staged once; the key tile is re-staged for every tile (rows past TOTAL stay zero, as the kernel zeroes them)
qs = np.zeros((HEADS, LD), np.float16); qs[:, :D] = Q


def mfma(A, B):
    """one V_MFMA_F32_16X16X16F16: A[m][k], B[k][n] -> C[m][n]"""
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)


def gather_a(x, row, kk, quad):
    return x[row, kk + 4 * quad: kk + 4 * quad + 4]


def gather_b(x, col, kk, quad):
    return x[col, kk + 4 * quad: kk + 4 * quad + 4]


def tile_ks(t0):
    """the staged 32-key tile: keys t0..t0+31, rows past TOTAL zeroed"""
    ks = np.zeros((KEYS, LD), np.float16)
    n = min(KEYS, TOTAL - t0)
    ks[:n, :D] = K[t0:t0 + n]
    return ks


def mfma_scores(ks, kk):
    """the two 16-key score tiles from one wave's 16-dim k step; returns C0/C1 as [[m][n]]"""
    A = np.zeros((16, 16), np.float16)
    B0 = np.zeros((16, 16), np.float16)
    B1 = np.zeros((16, 16), np.float16)
    for lane in range(64):
        m = lane & 15
        quad = lane >> 4
        A[m, 4 * quad: 4 * quad + 4] = gather_a(qs, m, kk, quad)
        B0[4 * quad: 4 * quad + 4, m] = gather_b(ks, m, kk, quad)              # B[k][n], n = m
        B1[4 * quad: 4 * quad + 4, m] = gather_b(ks, 16 + m, kk, quad)
    return mfma(A, B0), mfma(A, B1)


def wave_scores(ks, wv):
    acc = np.zeros((2, 16, 16), np.float32)
    for sub in range(2):
        for kk in range(0, 64, 16):
            c0, c1 = mfma_scores(ks, kk + sub * 64 + wv * 128)
            acc[0] += c0
            acc[1] += c1
    return acc


def part_scores(ks):
    """part[4][16][32] as the kernel builds it, then sc (with the masked tail)"""
    part = np.zeros((4, HEADS, KEYS), np.float32)
    for wv in range(4):
        a = wave_scores(ks, wv)
        for lane in range(64):
            quad, m = lane >> 4, lane & 15
            for i in range(4):
                part[wv, m, quad + 4 * i] = a[0][m, quad + 4 * i]
                part[wv, m, 16 + quad + 4 * i] = a[1][m, quad + 4 * i]
    return part.sum(axis=0)


def pv_add(o, ps, ks, dim0):
    """o (16x512) += P x V for this 32-key tile, exactly the kernel's two-tile gathers"""
    for f in range(4):
        dm = dim0 + f * 32
        acc = np.zeros((2, 16, 16), np.float32)
        for kg in (0, 16):
            A = np.zeros((16, 16), np.float16)
            B0 = np.zeros((16, 16), np.float16)
            B1 = np.zeros((16, 16), np.float16)
            for lane in range(64):
                l16, quad = lane & 15, lane >> 4
                A[l16, 4 * quad: 4 * quad + 4] = ps[l16, kg + 4 * quad: kg + 4 * quad + 4]
                for i in range(4):
                    key = kg + 4 * quad + i
                    B0[4 * quad + i, l16] = ks[key, dm + l16]
                    B1[4 * quad + i, l16] = ks[key, dm + 16 + l16]
            acc[0] += mfma(A, B0)
            acc[1] += mfma(A, B1)
        for lane in range(64):
            quad, l16 = lane >> 4, lane & 15
            for i in range(4):
                o[l16, dm + quad + 4 * i] += acc[0][l16, quad + 4 * i]
                o[l16, dm + 16 + quad + 4 * i] += acc[1][l16, quad + 4 * i]


m_run = np.full(HEADS, -1e30, np.float32)
den = np.zeros(HEADS, np.float32)
o = np.zeros((HEADS, D), np.float32)
for t0 in range(0, TOTAL, KEYS):
    ks = tile_ks(t0)
    s = part_scores(ks) * SCALE
    nt = min(KEYS, TOTAL - t0)
    sc = np.full((HEADS, KEYS), -np.inf, np.float32)
    sc[:, :nt] = s[:, :nt]
    # the online softmax (one lane a key on the real kernel; the per-head state is the same)
    p = np.zeros((HEADS, KEYS), np.float32)
    ts = np.zeros(HEADS, np.float32)
    for h in range(HEADS):
        mo = m_run[h]
        mn = max(mo, sc[h].max())
        p[h] = np.exp(sc[h] - mn)
        p[h][~np.isfinite(sc[h])] = 0.0
        ts[h] = np.exp(mo - mn)
        den[h] = den[h] * ts[h] + p[h].sum(dtype=np.float32)
        m_run[h] = mn
    ps = p.astype(np.float16)   # the kernel stores P as FP16 before the value product
    for dim0 in range(0, D, 128):   # the kernel rescales every wave's 128 dims, then adds this tile's P x V
        o[:, dim0:dim0 + 128] *= ts[:, None]
        pv_add(o, ps, ks, dim0)

for h in range(HEADS):
    o[h] /= den[h] + np.exp(SINK - m_run[h])

# --- the reference: softmax over the packed keys plus the learned sink, FP32 sums ---
s_ref = (Q.astype(np.float32) @ K.astype(np.float32).T) * SCALE
e = np.exp(s_ref).astype(np.float64)
ref = (e @ V.astype(np.float64)) / (e.sum(axis=1) + np.exp(SINK))[:, None]

err = np.abs(o - ref)
rel = err / (np.abs(ref) + 1e-6)
print('keys used:', TOTAL, ' (16 tiles; the last one partial)')
print('max abs err %.3e   median rel err %.3e   mean abs err %.3e' % (err.max(), np.median(rel), err.mean()))
print('rows/cols covered:', np.isfinite(o).all(), o.shape)
print('OK' if err.max() < 5e-3 else 'MISMATCH')
