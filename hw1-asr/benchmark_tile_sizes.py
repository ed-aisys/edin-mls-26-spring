"""
Tile / block size tuning — HW1 optimization 1 ONLY (adjust tile/block sizes).

This script does not cover fusion (opt 2) or FlashAttention (opt 3). See TILE_TUNING.md
for the step-by-step workflow scoped to tiling: BLOCK_M/N/K, num_warps, num_stages.

Benchmarks mirror glm_asr_triton_template/layers.py:
  linear_kernel_tf32, linear_gelu_kernel, swiglu_fused_kernel, gelu_kernel.

TFLOPS use padded tensor sizes so comparisons across BLOCK_* are fair; pick winners by time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

# Repo layout: hw1-asr/benchmark_tile_sizes.py → import template layers
_HW1 = Path(__file__).resolve().parent
if str(_HW1) not in sys.path:
    sys.path.insert(0, str(_HW1))

from glm_asr_triton_template.layers import (  # noqa: E402
    EncoderMLP,
    GELU_BLOCK_SIZE,
    GELU_NUM_WARPS,
    Linear,
    MLP,
)

device = torch.device("cuda")


# --------------------------------------------------------------------------- #
#  Matmul (same structure as linear_kernel_tf32)
# --------------------------------------------------------------------------- #


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)


# --------------------------------------------------------------------------- #
#  Linear + GELU fused (same structure as linear_gelu_kernel)
# --------------------------------------------------------------------------- #


@triton.jit
def linear_gelu_kernel_bench(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# --------------------------------------------------------------------------- #
#  GELU element-wise (same as gelu_kernel in layers.py)
# --------------------------------------------------------------------------- #


@triton.jit
def gelu_kernel_bench(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    inner = tl.sqrt(2.0 / 3.14159265358979) * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
    y = 0.5 * x_f32 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)


# --------------------------------------------------------------------------- #
#  SwiGLU fused (same structure as swiglu_fused_kernel)
# --------------------------------------------------------------------------- #


@triton.jit
def swiglu_kernel_bench(
    a_ptr,
    gate_ptr,
    up_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_gk,
    stride_gn,
    stride_uk,
    stride_un,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        gw = tl.load(
            gate_ptr + (k + offs_k[:, None]) * stride_gk + offs_n[None, :] * stride_gn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        uw = tl.load(
            up_ptr + (k + offs_k[:, None]) * stride_uk + offs_n[None, :] * stride_un,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        gate_acc += tl.dot(a, gw)
        up_acc += tl.dot(a, uw)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate_acc))
    out = gate_acc * sigmoid * up_acc

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def pad(x: int, mult: int) -> int:
    return ((x + mult - 1) // mult) * mult


def bench_matmul(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    num_stages,
    warmup: int,
    iters: int,
):
    M_p = pad(M, BLOCK_M)
    N_p = pad(N, BLOCK_N)
    K_p = pad(K, BLOCK_K)
    a = torch.randn(M_p, K_p, device=device, dtype=torch.float32)
    b = torch.randn(K_p, N_p, device=device, dtype=torch.float32)
    c = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)

    grid = (triton.cdiv(M_p, BLOCK_M), triton.cdiv(N_p, BLOCK_N))

    for _ in range(warmup):
        matmul_kernel[grid](
            a,
            b,
            c,
            M_p,
            N_p,
            K_p,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        matmul_kernel[grid](
            a,
            b,
            c,
            M_p,
            N_p,
            K_p,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    # Throughput matches actual tensor ops (padded), comparable across tile configs.
    tflops = 2 * M_p * N_p * K_p / elapsed / 1e12
    return elapsed * 1e3, tflops


def bench_linear_gelu(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    num_stages,
    warmup: int,
    iters: int,
):
    M_p = pad(M, BLOCK_M)
    N_p = pad(N, BLOCK_N)
    K_p = pad(K, BLOCK_K)
    a = torch.randn(M_p, K_p, device=device, dtype=torch.float32)
    b = torch.randn(K_p, N_p, device=device, dtype=torch.float32)
    c = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)

    grid = (triton.cdiv(M_p, BLOCK_M), triton.cdiv(N_p, BLOCK_N))

    for _ in range(warmup):
        linear_gelu_kernel_bench[grid](
            a,
            b,
            c,
            M_p,
            N_p,
            K_p,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        linear_gelu_kernel_bench[grid](
            a,
            b,
            c,
            M_p,
            N_p,
            K_p,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    tflops = 2 * M_p * N_p * K_p / elapsed / 1e12
    return elapsed * 1e3, tflops


def bench_swiglu(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    num_stages,
    warmup: int,
    iters: int,
):
    M_p = pad(M, BLOCK_M)
    N_p = pad(N, BLOCK_N)
    K_p = pad(K, BLOCK_K)
    a = torch.randn(M_p, K_p, device=device, dtype=torch.float32)
    gw = torch.randn(K_p, N_p, device=device, dtype=torch.float32)
    uw = torch.randn(K_p, N_p, device=device, dtype=torch.float32)
    c = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)

    grid = (triton.cdiv(M_p, BLOCK_M), triton.cdiv(N_p, BLOCK_N))

    for _ in range(warmup):
        swiglu_kernel_bench[grid](
            a,
            gw,
            uw,
            c,
            M_p,
            N_p,
            K_p,
            a.stride(0),
            a.stride(1),
            gw.stride(0),
            gw.stride(1),
            uw.stride(0),
            uw.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        swiglu_kernel_bench[grid](
            a,
            gw,
            uw,
            c,
            M_p,
            N_p,
            K_p,
            a.stride(0),
            a.stride(1),
            gw.stride(0),
            gw.stride(1),
            uw.stride(0),
            uw.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    # Two matmuls (gate + up) → ~4 * M*N*K multiply-adds (same convention as matmul TFLOPS scale)
    tflops = 4 * M_p * N_p * K_p / elapsed / 1e12
    return elapsed * 1e3, tflops


def bench_gelu(n_elements, BLOCK_SIZE, num_warps, warmup: int, iters: int):
    x = torch.randn(n_elements, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    for _ in range(warmup):
        gelu_kernel_bench[grid](
            x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        gelu_kernel_bench[grid](
            x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed * 1e3


def build_matmul_configs(quick: bool):
    """Template defaults first, then extra search points; dedupe by (BM, BN, BK, nw, ns)."""
    template_rows = [
        (
            Linear.TILE_M,
            Linear.TILE_N,
            Linear.TILE_K,
            Linear.NUM_WARPS,
            Linear.NUM_STAGES,
            f"template Linear: {Linear.TILE_M}x{Linear.TILE_N}x{Linear.TILE_K} "
            f"w{Linear.NUM_WARPS} s{Linear.NUM_STAGES}",
        ),
        (
            MLP.TILE_M,
            MLP.TILE_N,
            MLP.TILE_K,
            MLP.NUM_WARPS,
            MLP.NUM_STAGES,
            f"template MLP (SwiGLU): {MLP.TILE_M}x{MLP.TILE_N}x{MLP.TILE_K} "
            f"w{MLP.NUM_WARPS} s{MLP.NUM_STAGES}",
        ),
        (
            EncoderMLP.TILE_M,
            EncoderMLP.TILE_N,
            EncoderMLP.TILE_K,
            EncoderMLP.NUM_WARPS,
            EncoderMLP.NUM_STAGES,
            f"template EncoderMLP (fused GELU): {EncoderMLP.TILE_M}x{EncoderMLP.TILE_N}x{EncoderMLP.TILE_K} "
            f"w{EncoderMLP.NUM_WARPS} s{EncoderMLP.NUM_STAGES}",
        ),
    ]
    extra = [
        (128, 64, 32, 4, 2, "128x64x32 w4 s2"),
        (64, 128, 32, 4, 2, "64x128x32 w4 s2"),
        (128, 128, 32, 8, 2, "128x128x32 w8 s2"),
        (64, 64, 64, 4, 2, "64x64x64 w4 s2"),
        (128, 64, 64, 4, 2, "128x64x64 w4 s2"),
        (128, 128, 64, 8, 2, "128x128x64 w8 s2"),
        (128, 64, 32, 8, 2, "128x64x32 w8 s2"),
        (64, 64, 32, 2, 2, "64x64x32 w2 s2"),
    ]
    if not quick:
        extra += [
            (64, 64, 32, 4, 2, "64x64x32 w4 s2"),
            (64, 64, 32, 4, 4, "64x64x32 w4 s4"),
        ]

    seen = set()
    out = []
    for row in template_rows + extra:
        key = row[:5]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description="Triton tile-size microbenchmarks for HW1 (tile tuning).")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations per config")
    p.add_argument("--iters", type=int, default=20, help="Timed iterations per config")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer alternative configs (still includes template defaults)",
    )
    p.add_argument("--skip-gelu", action="store_true", help="Skip element-wise GELU sweep")
    p.add_argument("--skip-linear-gelu", action="store_true", help="Skip fused Linear+GELU sweep")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required. Activate the Triton environment and run on a GPU machine.")
        sys.exit(1)

    warmup, iters = args.warmup, args.iters
    matmul_configs = build_matmul_configs(args.quick)

    print("=" * 80)
    print("TILE / BLOCK SIZE TUNING (Triton)")
    print("GPU:", torch.cuda.get_device_name())
    print("TFLOPS use padded M×N×K (actual work); pick best primarily by Time (ms).")
    print("-" * 80)
    print("Template defaults in glm_asr_triton_template/layers.py:")
    print(
        f"  Linear:      {Linear.TILE_M}x{Linear.TILE_N}x{Linear.TILE_K}  "
        f"warps={Linear.NUM_WARPS}  stages={Linear.NUM_STAGES}"
    )
    print(
        f"  MLP (SwiGLU): {MLP.TILE_M}x{MLP.TILE_N}x{MLP.TILE_K}  "
        f"warps={MLP.NUM_WARPS}  stages={MLP.NUM_STAGES}"
    )
    print(
        f"  EncoderMLP:  {EncoderMLP.TILE_M}x{EncoderMLP.TILE_N}x{EncoderMLP.TILE_K}  "
        f"warps={EncoderMLP.NUM_WARPS}  stages={EncoderMLP.NUM_STAGES}"
    )
    print(f"  GELU (elem):  BLOCK_SIZE={GELU_BLOCK_SIZE}  warps={GELU_NUM_WARPS}")
    print("=" * 80)

    matmul_shapes = [
        (375, 5120, 1280, "encoder_fc1"),
        (375, 1280, 5120, "encoder_fc2"),
        (13, 8192, 2048, "decoder_gate"),
        (13, 2048, 8192, "decoder_down"),
    ]

    print("\n" + "=" * 80)
    print("MATMUL (linear_kernel_tf32 pattern)")
    print("=" * 80)

    for M, N, K, shape_name in matmul_shapes:
        print(f"\n--- {shape_name}  (logical M×N×K = {M}×{N}×{K}) ---")
        print(f"{'Config':<48s} {'Time (ms)':>10s} {'TFLOPS':>10s}")
        print("-" * 72)
        best_time = float("inf")
        best_label = ""
        for BM, BN, BK, nw, ns, label in matmul_configs:
            try:
                t, tflops = bench_matmul(M, N, K, BM, BN, BK, nw, ns, warmup, iters)
                marker = ""
                if t < best_time:
                    best_time = t
                    best_label = label
                    marker = "  <-- best"
                print(f"  {label:<46s} {t:>10.4f} {tflops:>10.2f}{marker}")
            except Exception as e:
                print(f"  {label:<46s}  FAILED: {e}")
        print(f"  >> BEST: {best_label} ({best_time:.4f} ms)")

    if not args.skip_linear_gelu:
        print("\n" + "=" * 80)
        print("FUSED LINEAR + GELU (linear_gelu_kernel — encoder MLP fc1 path)")
        print("=" * 80)
        linear_gelu_shapes = [(375, 5120, 1280, "encoder_fc1 → GELU")]
        for M, N, K, shape_name in linear_gelu_shapes:
            print(f"\n--- {shape_name}  ({M}×{N}×{K}) ---")
            print(f"{'Config':<48s} {'Time (ms)':>10s} {'TFLOPS':>10s}")
            print("-" * 72)
            best_time = float("inf")
            best_label = ""
            for BM, BN, BK, nw, ns, label in matmul_configs:
                try:
                    t, tflops = bench_linear_gelu(M, N, K, BM, BN, BK, nw, ns, warmup, iters)
                    marker = ""
                    if t < best_time:
                        best_time = t
                        best_label = label
                        marker = "  <-- best"
                    print(f"  {label:<46s} {t:>10.4f} {tflops:>10.2f}{marker}")
                except Exception as e:
                    print(f"  {label:<46s}  FAILED: {e}")
            print(f"  >> BEST: {best_label} ({best_time:.4f} ms)")

    print("\n" + "=" * 80)
    print("SWIGLU FUSED (swiglu_fused_kernel — decoder MLP)")
    print("TFLOPS column ≈ 2× matmul (gate + up); comparable across configs.")
    print("=" * 80)

    swiglu_shapes = [(13, 8192, 2048, "decoder SwiGLU")]
    for M, N, K, shape_name in swiglu_shapes:
        print(f"\n--- {shape_name}  ({M}×{N}×{K}) ---")
        print(f"{'Config':<48s} {'Time (ms)':>10s} {'TFLOPS':>10s}")
        print("-" * 72)
        best_time = float("inf")
        best_label = ""
        for BM, BN, BK, nw, ns, label in matmul_configs:
            try:
                t, tflops = bench_swiglu(M, N, K, BM, BN, BK, nw, ns, warmup, iters)
                marker = ""
                if t < best_time:
                    best_time = t
                    best_label = label
                    marker = "  <-- best"
                print(f"  {label:<46s} {t:>10.4f} {tflops:>10.2f}{marker}")
            except Exception as e:
                print(f"  {label:<46s}  FAILED: {e}")
        print(f"  >> BEST: {best_label} ({best_time:.4f} ms)")

    if not args.skip_gelu:
        print("\n" + "=" * 80)
        print("ELEMENT-WISE GELU (gelu_kernel)")
        print("=" * 80)
        gelu_sizes = [375 * 5120, 13 * 8192]
        gelu_configs = [
            (256, 4, "BS=256  w4"),
            (512, 4, "BS=512  w4"),
            (1024, 4, "BS=1024 w4"),
            (2048, 4, "BS=2048 w4"),
            (256, 2, "BS=256  w2"),
            (512, 2, "BS=512  w2"),
            (1024, 2, "BS=1024 w2"),
            (256, 8, "BS=256  w8"),
            (512, 8, "BS=512  w8"),
            (1024, 8, "BS=1024 w8"),
        ]
        tmpl = (GELU_BLOCK_SIZE, GELU_NUM_WARPS, f"template layers: BS={GELU_BLOCK_SIZE} w{GELU_NUM_WARPS}")
        if tmpl[:2] not in {c[:2] for c in gelu_configs}:
            gelu_configs.insert(0, tmpl)

        for n in gelu_sizes:
            print(f"\n--- n_elements = {n} ---")
            print(f"{'Config':<40s} {'Time (ms)':>10s}")
            print("-" * 55)
            best_time = float("inf")
            best_label = ""
            for bs, nw, label in gelu_configs:
                try:
                    t = bench_gelu(n, bs, nw, warmup, iters)
                    marker = ""
                    if t < best_time:
                        best_time = t
                        best_label = label
                        marker = "  <-- best"
                    print(f"  {label:<38s} {t:>10.4f}{marker}")
                except Exception as e:
                    print(f"  {label:<38s}  FAILED: {e}")
            print(f"  >> BEST: {best_label} ({best_time:.4f} ms)")
            print(
                f"  >> Set GELU_BLOCK_SIZE / GELU_NUM_WARPS in layers.py to match the winning row."
            )

    print("\n" + "=" * 80)
    print("DONE (optimization 1: tile/block sizes only) — copy winners into layers.py;")
    print("       see TILE_TUNING.md for step-by-step. Fusion / FlashAttention are later steps.")
    print("=" * 80)


if __name__ == "__main__":
    main()
