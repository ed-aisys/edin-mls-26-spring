"""
Triton Neural Network Layers
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement core layers using Triton kernels
"""

import math
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Tested tile configs per GPU architecture.
# Each entry maps arch_name → {attn_tiles, matmul_tiles, rope params}.
# For unknown architectures, tiles are computed dynamically from shared memory.
_KNOWN_CONFIGS = {
    # RTX 5090 (Blackwell consumer, sm_120, ~99KB optin smem)
    # Tested 2026-03-15: 98.5ms benchmark
    "blackwell_consumer": {
        "attn_tiles": {
            64:  (64, 64, 1, 4),   # Encoder: head_dim=64
            128: (32, 32, 1, 4),   # Decoder: head_dim=128
        },
        "matmul_tiles": (64, 64, 32),
        "rope_nstages": 1,
        "rope_nwarps": 4,
    },
    # RTX 4090 (Ada Lovelace, sm_89, ~100KB optin smem)
    # Same shared memory class as RTX 5090
    "ada": {
        "attn_tiles": {
            64:  (64, 64, 1, 4),
            128: (32, 32, 1, 4),
        },
        "matmul_tiles": (64, 64, 32),
        "rope_nstages": 1,
        "rope_nwarps": 4,
    },
    # H100/H200 (Hopper, sm_90, ~228KB optin smem)
    "hopper": {
        "attn_tiles": {
            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
            128: (128, 64, 2, 8),
        },
        "matmul_tiles": (128, 128, 64),
        "rope_nstages": 2,
        "rope_nwarps": 8,
    },
    # B200 (Blackwell datacenter, sm_100+, ~228KB optin smem)
    "blackwell_dc": {
        "attn_tiles": {
            64:  (128, 128, 2, 8),
            128: (128, 64, 2, 8),
        },
        "matmul_tiles": (128, 128, 64),
        "rope_nstages": 2,
        "rope_nwarps": 8,
    },
    # A100 (Ampere datacenter, sm_80, ~164KB optin smem)
    "ampere_dc": {
        "attn_tiles": {
            64:  (128, 64, 1, 4),
            128: (64, 32, 1, 4),
        },
        "matmul_tiles": (128, 64, 32),
        "rope_nstages": 1,
        "rope_nwarps": 4,
    },
    # RTX 3090 (Ampere consumer, sm_80, ~100KB optin smem)
    "ampere_consumer": {
        "attn_tiles": {
            64:  (64, 64, 1, 4),
            128: (32, 32, 1, 4),
        },
        "matmul_tiles": (64, 64, 32),
        "rope_nstages": 1,
        "rope_nwarps": 4,
    },
}


class GPUProfile:
    """GPU architecture profile with pre-computed tuning parameters.

    Detects the GPU at import time and stores optimal tile sizes, pipeline
    stages, and warp counts for every kernel. Adapts to the specific GPU
    architecture rather than using a coarse two-tier heuristic.

    Attributes:
        arch_name: Human-readable architecture name (e.g. "ada", "hopper").
        sm_version: Compute capability tuple, e.g. (8, 9) for Ada.
        smem_per_block: Max shared memory per block in bytes.
        gpu_name: Full GPU device name string.
        -- Pre-computed attention tile parameters --
        attn_tiles: dict mapping head_dim bucket to (BLOCK_M, BLOCK_N, nstages, nwarps).
        -- Pre-computed matmul tile parameters --
        matmul_tile_m/n/k: Tile sizes for Linear / fused SwiGLU / fused LinearGELU.
        -- Pre-computed RoPE parameters --
        rope_nstages, rope_nwarps: Launch config for fused_rope_pair_kernel.
    """

    def __init__(self):
        if not torch.cuda.is_available():
            self._init_cpu_fallback()
            return
        try:
            self.sm_version = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            # Use optin shared memory (extended) — this is what Triton can actually use.
            # shared_memory_per_block is only 48KB (default), optin is 99-228KB.
            # Fall back to shared_memory_per_block if optin property doesn't exist
            # (older PyTorch versions).
            self.smem_per_block = getattr(
                props, 'shared_memory_per_block_optin',
                getattr(props, 'max_shared_memory_per_block', props.shared_memory_per_block)
            )
            self.gpu_name = props.name
        except Exception:
            self._init_cpu_fallback()
            return

        sm = self.sm_version
        smem_kb = self.smem_per_block // 1024

        # Classify architecture
        if sm >= (10, 0) and self.smem_per_block > 120 * 1024:
            self.arch_name = "blackwell_dc"      # B200 datacenter
        elif sm >= (9, 0) and self.smem_per_block > 120 * 1024:
            self.arch_name = "hopper"             # H100/H200
        elif sm >= (9, 0):
            self.arch_name = "hopper_consumer"    # Hypothetical
        elif sm >= (8, 9):
            self.arch_name = "ada"                # RTX 4090
        elif sm >= (8, 0) and self.smem_per_block > 120 * 1024:
            self.arch_name = "ampere_dc"          # A100
        elif sm >= (8, 0):
            self.arch_name = "ampere_consumer"    # RTX 3090
        else:
            self.arch_name = "older"
        # Blackwell consumer (RTX 5090) reports sm_120 = (12, 0) with ~101KB
        if sm >= (12, 0) and self.smem_per_block <= 120 * 1024:
            self.arch_name = "blackwell_consumer"

        # Use tested configs for known architectures, dynamic computation for unknown
        known = _KNOWN_CONFIGS.get(self.arch_name)
        if known:
            self.attn_tiles = known['attn_tiles']
            self.matmul_tile_m = known['matmul_tiles'][0]
            self.matmul_tile_n = known['matmul_tiles'][1]
            self.matmul_tile_k = known['matmul_tiles'][2]
            self.rope_nstages = known['rope_nstages']
            self.rope_nwarps = known['rope_nwarps']
        else:
            # Unknown GPU — compute dynamically from shared memory budget
            self.attn_tiles = {}
            for hd in (64, 128):
                self.attn_tiles[hd] = _compute_attention_tiles(hd, self.smem_per_block)
            self.matmul_tile_m, self.matmul_tile_n, self.matmul_tile_k = \
                _compute_matmul_tiles(self.smem_per_block)
            if self.smem_per_block > 150 * 1024:
                self.rope_nstages, self.rope_nwarps = 2, 8
            else:
                self.rope_nstages, self.rope_nwarps = 1, 4

    def _init_cpu_fallback(self):
        self.arch_name = "cpu"
        self.sm_version = (0, 0)
        self.smem_per_block = 0
        self.gpu_name = "none"
        self.attn_tiles = {
            64: (64, 64, 1, 4),
            128: (32, 32, 1, 4),
        }
        self.matmul_tile_m, self.matmul_tile_n, self.matmul_tile_k = 64, 64, 32
        self.rope_nstages, self.rope_nwarps = 1, 4

    @property
    def is_datacenter(self):
        return self.smem_per_block > 120 * 1024

    def get_attention_tiles(self, head_dim, seq_q=None):
        """Get (BLOCK_M, BLOCK_N, nstages, nwarps) for a given head_dim.

        For tiny seq_q (KV-cached decode), BLOCK_M is clamped down.
        """
        # Use nearest bucket (64 or 128)
        bucket = 64 if head_dim <= 64 else 128
        bm, bn, ns, nw = self.attn_tiles.get(bucket, self.attn_tiles[128])
        if seq_q is not None and seq_q <= 16:
            bm = min(bm, 16)
        return bm, bn, ns, nw

    def __repr__(self):
        return (f"GPUProfile(arch={self.arch_name}, sm={self.sm_version}, "
                f"smem={self.smem_per_block // 1024}KB, gpu={self.gpu_name}, "
                f"matmul_tiles={self.matmul_tile_m}x{self.matmul_tile_n}x{self.matmul_tile_k})")


def _compute_attention_tiles(head_dim, smem_bytes):
    """Select optimal flash attention tile sizes from shared memory budget.

    Uses ranked known-good tile configurations and picks the largest that fits.
    The flash attention kernel needs shared memory for Q, K, and V tiles plus
    Triton compiler overhead (accumulators, softmax state, padding, etc.).

    Shared memory estimate per config:
        (BLOCK_M + 2 * BLOCK_N) * BLOCK_D * 4 bytes  (fp32 tiles)
        + ~20KB Triton overhead (conservative estimate for compiler state)
    """
    BLOCK_D = 1 << (head_dim - 1).bit_length() if head_dim > 0 else head_dim
    element_bytes = 4
    overhead = 20 * 1024  # conservative estimate for Triton compiler overhead

    # Ranked configs: (BLOCK_M, BLOCK_N) ordered from fastest to slowest.
    # Balanced tiles (BLOCK_M ≈ BLOCK_N) perform better than unbalanced ones.
    if head_dim <= 64:
        configs = [(128, 128), (128, 64), (64, 64), (64, 32), (32, 32), (16, 16)]
    else:
        configs = [(128, 64), (64, 64), (64, 32), (32, 32), (32, 16), (16, 16)]

    for bm, bn in configs:
        needed = (bm + 2 * bn) * BLOCK_D * element_bytes + overhead
        if needed <= smem_bytes:
            nstages = 2 if smem_bytes > 150 * 1024 else 1
            nwarps = 8 if bm * bn >= 4096 else 4
            return (bm, bn, nstages, nwarps)

    return (16, 16, 1, 4)  # absolute minimum


def _compute_matmul_tiles(smem_bytes):
    """Select optimal matmul tile sizes from shared memory budget.

    Must account for the fused SwiGLU kernel (worst case), which loads:
      - A tile:      TILE_M * TILE_K * 4 bytes  (input)
      - gate_w tile: TILE_K * TILE_N * 4 bytes  (gate weight)
      - up_w tile:   TILE_K * TILE_N * 4 bytes  (up weight)
    Total: TILE_K * (TILE_M + 2 * TILE_N) * 4 + overhead
    """
    overhead = 20 * 1024  # conservative Triton overhead

    # Ranked configs: (TILE_M, TILE_N, TILE_K) from largest to smallest
    configs = [(128, 128, 64), (128, 64, 32), (64, 64, 32), (64, 32, 32), (32, 32, 16)]

    for tm, tn, tk in configs:
        # Use SwiGLU worst case: A + 2 * B tiles
        needed = tk * (tm + 2 * tn) * 4 + overhead
        if needed <= smem_bytes:
            return tm, tn, tk

    return 32, 32, 16  # absolute minimum


# Module-level GPU profile — computed once at import time
GPU = GPUProfile()
# Backward compatibility alias
_GPU_TIER = 'datacenter' if GPU.is_datacenter else 'consumer'


# ============================================================================
# Helper Functions
# ============================================================================

def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def pad_to_multiple(size: int, multiple: int) -> int:
    """Pad size to be a multiple of the given value."""
    return ((size + multiple - 1) // multiple) * multiple


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _to_torch_tensor(arr, dtype=torch.float32, device='cuda'):
    """Convert array-like (numpy, CuPy, etc.) to PyTorch tensor on device."""
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        if arr.dtype != dtype or str(arr.device) != device:
            return arr.to(dtype=dtype, device=device)
        return arr
    if hasattr(arr, 'get'):  # CuPy array
        arr = np.asarray(arr.get())
    elif not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    return torch.as_tensor(arr, dtype=dtype, device=device)


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: x / RMS(x) * weight

    *** TODO: Implement this kernel ***

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    x_norm = x * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x_norm * w
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def rmsnorm_bf16_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm with bf16 output (from meave branch, adapted for bf16).
    Outputs bfloat16 directly, avoiding fp32→HBM→bf16 round-trip
    when feeding into cuBLAS Linear layers.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    x_norm = x * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = (x_norm * w).to(tl.float16)
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_x,
    stride_y,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias

    *** TODO: Implement this kernel ***

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    x_norm = x_centered * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = (x_norm * w + b).to(tl.float16)
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU using tanh approximation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    y = x * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """SiLU/Swish: x * sigmoid(x)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def linear_kernel_tf32(
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
    """
    TF32-style matmul: output = A @ B.
    A: (M, K), B: (K, N), C: (M, N)

    *** TODO: Implement this kernel ***

    Grid: (M // BLOCK_M, N // BLOCK_N)
    """
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

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def linear_gelu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused Linear + GELU."""
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

    bias = tl.load(
        bias_ptr + offs_n * stride_bias,
        mask=offs_n < N,
        other=0.0,
    )
    acc = acc + bias[None, :]

    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def swiglu_fused_kernel(
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
    """Fused SwiGLU: SiLU(x @ gate) * (x @ up)."""
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
        gate_w = tl.load(
            gate_ptr + (k + offs_k[:, None]) * stride_gk + offs_n[None, :] * stride_gn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        up_w = tl.load(
            up_ptr + (k + offs_k[:, None]) * stride_uk + offs_n[None, :] * stride_un,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        gate_acc += tl.dot(a, gate_w)
        up_acc += tl.dot(a, up_w)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate_acc))
    gate_act = gate_acc * sigmoid
    out = gate_act * up_acc

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    embedding_dim,
    stride_w0,
    stride_w1,
    stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup using gather."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w = tl.load(
        weight_ptr + idx * stride_w0 + offs * stride_w1, mask=mask, other=0.0
    )
    tl.store(output_ptr + pid0 * stride_out0 + offs, w, mask=mask)


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Numerically stable softmax over last dimension.

    *** TODO: Implement this kernel ***
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    y = exp_x / denom
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


# ============================================================================
# Layer Classes
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    """Check if x is a power of two."""
    return x > 0 and (x & (x - 1)) == 0


class RMSNorm:
    """Root Mean Square Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.use_triton = _is_power_of_two(hidden_size)
        self._block = next_power_of_two(hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            original_shape = x.shape
            batch_size = x.numel() // self.hidden_size
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            # Use fp16 output kernel when Linear.BF16 is True
            if Linear.BF16:
                output = torch.empty(
                    (batch_size, self.hidden_size),
                    dtype=torch.float16,
                    device=x.device,
                )
                rmsnorm_bf16_kernel[(batch_size,)](
                    x_flat,
                    self.weight,
                    output,
                    x_flat.stride(0),
                    output.stride(0),
                    self.hidden_size,
                    self.eps,
                    BLOCK_SIZE=self._block,
                )
            else:
                output = torch.empty_like(x_flat)
                rmsnorm_kernel[(batch_size,)](
                    x_flat,
                    self.weight,
                    output,
                    x_flat.stride(0),
                    output.stride(0),
                    self.hidden_size,
                    self.eps,
                    BLOCK_SIZE=self._block,
                )
            return output.reshape(original_shape)

        x_float = x.to(torch.float32)
        variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        return (self.weight * x_normed).to(x.dtype)


class LayerNorm:
    """Layer Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.bias = torch.zeros(hidden_size, dtype=torch.float32)
        self.use_triton = _is_power_of_two(hidden_size)
        self._block = next_power_of_two(hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            original_shape = x.shape
            batch_size = x.numel() // self.hidden_size
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            if self.bias.device != x.device:
                self.bias = self.bias.to(x.device)
            # Use fp16 output to avoid conversion in next Linear layer
            if Linear.BF16:
                output = torch.empty(
                    (batch_size, self.hidden_size),
                    dtype=torch.float16,
                    device=x.device,
                )
            else:
                output = torch.empty_like(x_flat)
            layernorm_kernel[(batch_size,)](
                x_flat,
                self.weight,
                self.bias,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=self._block,
            )
            return output.reshape(original_shape)

        x_float = x.to(torch.float32)
        mean = torch.mean(x_float, dim=-1, keepdim=True)
        variance = torch.var(x_float, dim=-1, keepdim=True, unbiased=False)
        x_normed = (x_float - mean) * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        if self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
        return (self.weight * x_normed + self.bias).to(x.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using Triton."""
    if not x.is_cuda:
        return torch.nn.functional.gelu(x)
    original_shape = x.shape
    n = x.numel()
    x_flat = x.reshape(-1).contiguous()
    output = torch.empty_like(x_flat)
    gelu_kernel[(triton.cdiv(n, 1024),)](x_flat, output, n, BLOCK_SIZE=1024)
    return output.reshape(original_shape)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation using Triton."""
    if not x.is_cuda:
        return torch.nn.functional.silu(x)
    original_shape = x.shape
    n = x.numel()
    x_flat = x.reshape(-1).contiguous()
    output = torch.empty_like(x_flat)
    silu_kernel[(triton.cdiv(n, 1024),)](x_flat, output, n, BLOCK_SIZE=1024)
    return output.reshape(original_shape)


def get_activation(name: str):
    """Get activation function by name."""
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    """Linear layer with switchable backend (torch or Triton)."""

    # Triton matmul tile sizes (only used when BACKEND="triton")
    TILE_M, TILE_N, TILE_K = GPU.matmul_tile_m, GPU.matmul_tile_n, GPU.matmul_tile_k

    BACKEND = "torch"  # Use cuBLAS (fastest for matmul on Blackwell)

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        _try_patch_v8b()  # Deferred monkey-patch (no-op after first success)
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param = torch.zeros(out_features, dtype=torch.float32) if bias else None

        self._weight_t_padded = None
        self._K_padded = None
        self._N_padded = None
        self._weight_bf16 = None
        self._bias_bf16 = None

    def _ensure_weight_prepared(self):
        """Cache transposed and padded weight for Triton kernel."""
        if self._weight_t_padded is None:
            K = self.in_features
            N = self.out_features
            self._K_padded = pad_to_multiple(K, self.TILE_K)
            self._N_padded = pad_to_multiple(N, self.TILE_N)

            weight_t = self.weight.t().contiguous()
            if self._K_padded > K or self._N_padded > N:
                weight_pad = torch.zeros(
                    (self._K_padded, self._N_padded),
                    dtype=torch.float32,
                    device=weight_t.device,
                )
                weight_pad[:K, :N] = weight_t
                self._weight_t_padded = weight_pad
            else:
                self._weight_t_padded = weight_t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if Linear.BACKEND in ("torch", "cublas"):
            return self._forward_torch(x)
        if Linear.BACKEND == "triton":
            return self._forward_triton(x)
        M = int(np.prod(x.shape[:-1]))
        if M >= self.TILE_M and x.is_cuda:
            return self._forward_triton(x)
        return self._forward_torch(x)

    BF16 = True  # Use reduced-precision weights (halves memory traffic for decode)
    # Use fp16 for cuBLAS matmuls (slightly faster than bf16 on some GPUs)
    _HALF_DTYPE = torch.float16

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Torch matmul backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = x.numel() // self.in_features
        x_2d = x.reshape(M, self.in_features)

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            if self._weight_bf16 is not None:
                self._weight_bf16 = self._weight_bf16.to(x.device)
        bias = None
        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
                self._bias_bf16 = None
            bias = self.bias_param

        if Linear.BF16:
            hdtype = Linear._HALF_DTYPE
            if self._weight_bf16 is None:
                self._weight_bf16 = self.weight.to(hdtype)
                self._bias_bf16 = bias.to(hdtype) if bias is not None else None
            output = F.linear(
                x_2d.to(hdtype), self._weight_bf16, self._bias_bf16
            )
        else:
            output = F.linear(x_2d.to(torch.float32), self.weight, bias)

        return output.reshape(*batch_dims, self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton matmul backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features

        x_2d = x.reshape(M, K).to(torch.float32).contiguous()

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            self._weight_t_padded = None
        self._ensure_weight_prepared()

        M_padded = pad_to_multiple(M, self.TILE_M)

        if M_padded > M or self._K_padded > K:
            x_padded = torch.zeros(
                (M_padded, self._K_padded),
                dtype=torch.float32,
                device=x.device,
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        output = torch.zeros(
            (M_padded, self._N_padded), dtype=torch.float32, device=x.device
        )

        grid = (
            triton.cdiv(M_padded, self.TILE_M),
            triton.cdiv(self._N_padded, self.TILE_N),
        )
        linear_kernel_tf32[grid](
            x_padded,
            self._weight_t_padded,
            output,
            M_padded,
            self._N_padded,
            self._K_padded,
            x_padded.stride(0),
            x_padded.stride(1),
            self._weight_t_padded.stride(0),
            self._weight_t_padded.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
        )

        output = output[:M, :N]

        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param

        return output.reshape(*batch_dims, self.out_features)


class Embedding:
    """Embedding layer using Triton."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        batch_size = int(np.prod(original_shape))

        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)

        if not input_ids.is_cuda:
            flat = input_ids.reshape(-1).to(torch.int64)
            output = self.weight.index_select(0, flat)
            return output.reshape(*original_shape, self.embedding_dim)

        indices_flat = input_ids.reshape(-1).to(torch.int32).contiguous()
        out_dtype = torch.float16 if Linear.BF16 else torch.float32
        output = torch.empty(
            (batch_size, self.embedding_dim), dtype=out_dtype, device=indices_flat.device
        )

        block = 256
        grid = (batch_size, triton.cdiv(self.embedding_dim, block))
        embedding_kernel[grid](
            indices_flat,
            self.weight,
            output,
            self.embedding_dim,
            self.weight.stride(0),
            self.weight.stride(1),
            output.stride(0),
            BLOCK_SIZE=block,
        )

        return output.reshape(*original_shape, self.embedding_dim)


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax using Triton kernel."""
    if not x.is_cuda:
        return torch.softmax(x, dim=axis)

    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)

    original_shape = x.shape
    seq_len = x.shape[-1]
    batch_size = x.numel() // seq_len

    x_flat = x.reshape(batch_size, seq_len)
    if x_flat.dtype != torch.float32:
        x_flat = x_flat.to(torch.float32)
    output = torch.empty_like(x_flat)

    softmax_kernel[(batch_size,)](
        x_flat,
        output,
        x_flat.stride(0),
        output.stride(0),
        seq_len,
        BLOCK_SIZE=next_power_of_two(seq_len),
    )
    result = output.reshape(original_shape)

    if axis != -1 and axis != len(original_shape) - 1:
        result = torch.movedim(result, -1, axis)

    return result


class LinearGELU:
    """Linear layer followed by GELU, with optional fused Triton path."""

    FUSED = False  # cuBLAS + separate GELU is faster and avoids shared memory limits

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.linear = Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        self._weight_t = None

    def _prepare_fused_weights(self):
        if self._weight_t is None:
            self._weight_t = self.linear.weight.t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if LinearGELU.FUSED and x.is_cuda:
            return self._forward_fused(x)
        return gelu(self.linear(x))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear.weight.device != x.device:
            self.linear.weight = self.linear.weight.to(x.device)
            self._weight_t = None
        self._prepare_fused_weights()

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float32).contiguous()
        M = x_2d.shape[0]
        K = self.in_features
        N = self.out_features

        M_pad = pad_to_multiple(M, self.linear.TILE_M)
        K_pad = pad_to_multiple(K, self.linear.TILE_K)
        N_pad = pad_to_multiple(N, self.linear.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=x.device)
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            weight_padded = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=x.device)
            weight_padded[:K, :N] = self._weight_t
        else:
            weight_padded = self._weight_t

        if self.bias_enabled and self.linear.bias_param is not None:
            if self.linear.bias_param.device != x.device:
                self.linear.bias_param = self.linear.bias_param.to(x.device)
            if N != N_pad:
                bias_padded = torch.zeros((N_pad,), dtype=torch.float32, device=x.device)
                bias_padded[:N] = self.linear.bias_param
            else:
                bias_padded = self.linear.bias_param
        else:
            bias_padded = torch.zeros((N_pad,), dtype=torch.float32, device=x.device)

        output = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=x.device)

        grid = (
            triton.cdiv(M_pad, self.linear.TILE_M),
            triton.cdiv(N_pad, self.linear.TILE_N),
        )
        linear_gelu_kernel[grid](
            x_padded,
            weight_padded,
            bias_padded,
            output,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            weight_padded.stride(0),
            weight_padded.stride(1),
            bias_padded.stride(0),
            output.stride(0),
            output.stride(1),
            BLOCK_M=self.linear.TILE_M,
            BLOCK_N=self.linear.TILE_N,
            BLOCK_K=self.linear.TILE_K,
        )

        if M != M_pad or N != N_pad:
            output = output[:M, :N]

        return output.reshape(*orig_shape[:-1], self.out_features)


class MLP:
    """MLP with SwiGLU gating using Triton."""

    FUSED = True
    TILE_M, TILE_N, TILE_K = GPU.matmul_tile_m, GPU.matmul_tile_n, GPU.matmul_tile_k

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        use_gating: bool = True,
    ):
        self.use_gating = use_gating
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias

        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)

        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

        self._gate_weight_t = None
        self._up_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._gate_weight_t is None and self.use_gating:
            if self.gate_proj.weight.device != self.up_proj.weight.device:
                self.up_proj.weight = self.up_proj.weight.to(self.gate_proj.weight.device)
            hdtype = Linear._HALF_DTYPE if Linear.BF16 else torch.float32
            self._gate_weight_t = self.gate_proj.weight.to(hdtype).t().contiguous()
            self._up_weight_t = self.up_proj.weight.to(hdtype).t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        num_rows = int(np.prod(x.shape[:-1]))
        if self.use_gating and MLP.FUSED and x.is_cuda and num_rows >= self.TILE_M:
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard (unfused) forward pass."""
        if self.use_gating:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused SwiGLU forward pass."""
        if self.gate_proj.weight.device != x.device:
            self.gate_proj.weight = self.gate_proj.weight.to(x.device)
            self._gate_weight_t = None
        if self.up_proj.weight.device != x.device:
            self.up_proj.weight = self.up_proj.weight.to(x.device)
            self._up_weight_t = None
        self._prepare_fused_weights()

        orig_shape = x.shape
        hdtype = Linear._HALF_DTYPE if Linear.BF16 else torch.float32
        x_2d = x.reshape(-1, self.hidden_size).to(hdtype).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=hdtype, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            gate_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=hdtype, device=x.device
            )
            gate_w_padded[:K, :N] = self._gate_weight_t
            up_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=hdtype, device=x.device
            )
            up_w_padded[:K, :N] = self._up_weight_t
        else:
            gate_w_padded = self._gate_weight_t
            up_w_padded = self._up_weight_t

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=hdtype, device=x.device
        )

        grid = (
            triton.cdiv(M_pad, self.TILE_M),
            triton.cdiv(N_pad, self.TILE_N),
        )
        swiglu_fused_kernel[grid](
            x_padded,
            gate_w_padded,
            up_w_padded,
            intermediate,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            gate_w_padded.stride(0),
            gate_w_padded.stride(1),
            up_w_padded.stride(0),
            up_w_padded.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.down_proj(intermediate)


class EncoderMLP:
    """Encoder MLP (no gating) using Triton."""

    FUSED = True
    TILE_M, TILE_N, TILE_K = GPU.matmul_tile_m, GPU.matmul_tile_n, GPU.matmul_tile_k

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
    ):
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias
        self.activation = activation

        self._fc1_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._fc1_weight_t is None:
            hdtype = Linear._HALF_DTYPE if Linear.BF16 else torch.float32
            self._fc1_weight_t = self.fc1.weight.to(hdtype).t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        num_rows = int(np.prod(x.shape[:-1]))
        if (
            EncoderMLP.FUSED
            and self.activation == "gelu"
            and x.is_cuda
            and num_rows >= self.TILE_M
        ):
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard (unfused) forward pass."""
        return self.fc2(self.act_fn(self.fc1(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused Linear+GELU forward pass."""
        if self.fc1.weight.device != x.device:
            self.fc1.weight = self.fc1.weight.to(x.device)
            self._fc1_weight_t = None
        self._prepare_fused_weights()

        orig_shape = x.shape
        hdtype = Linear._HALF_DTYPE if Linear.BF16 else torch.float32
        x_2d = x.reshape(-1, self.hidden_size).to(hdtype).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=hdtype, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            fc1_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=hdtype, device=x.device
            )
            fc1_w_padded[:K, :N] = self._fc1_weight_t
        else:
            fc1_w_padded = self._fc1_weight_t

        if self.bias_enabled and self.fc1.bias_param is not None:
            if self.fc1.bias_param.device != x.device:
                self.fc1.bias_param = self.fc1.bias_param.to(x.device)
            if N != N_pad:
                fc1_bias_padded = torch.zeros(
                    (N_pad,), dtype=hdtype, device=x.device
                )
                fc1_bias_padded[:N] = self.fc1.bias_param
            else:
                fc1_bias_padded = self.fc1.bias_param.to(hdtype)
        else:
            fc1_bias_padded = torch.zeros((N_pad,), dtype=hdtype, device=x.device)

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=hdtype, device=x.device
        )

        grid = (
            triton.cdiv(M_pad, self.TILE_M),
            triton.cdiv(N_pad, self.TILE_N),
        )
        linear_gelu_kernel[grid](
            x_padded,
            fc1_w_padded,
            fc1_bias_padded,
            intermediate,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            fc1_w_padded.stride(0),
            fc1_w_padded.stride(1),
            fc1_bias_padded.stride(0),
            intermediate.stride(0),
            intermediate.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.fc2(intermediate)


# ============================================================================
# KV-Cached Generation (monkey-patched onto GlmAsrModel)
# ============================================================================

def _generate_v8b(
    self,
    input_features,
    input_ids=None,
    input_features_mask=None,
    attention_mask=None,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    audio_pad_token_id=59260,
):
    """KV-cached O(n) generation using model.decode() with use_cache=True."""
    # Convert all inputs to PyTorch tensors (benchmarks may pass numpy/CuPy arrays)
    input_features = _to_torch_tensor(input_features, dtype=torch.float32, device='cuda')
    if input_features_mask is not None:
        input_features_mask = _to_torch_tensor(input_features_mask, dtype=torch.float32, device='cuda')
    if input_ids is not None:
        input_ids = _to_torch_tensor(input_ids, dtype=torch.int64, device='cuda')
    if attention_mask is not None:
        attention_mask = _to_torch_tensor(attention_mask, dtype=torch.float32, device='cuda')
    # Encode audio
    audio_embeds = self.encode_audio(input_features, input_features_mask)

    if input_ids is not None:
        batch_size = input_ids.shape[0]
        if audio_embeds.ndim == 3:
            audio_embeds = audio_embeds[0]
        text_embeds = self.text_decoder.embed_tokens(input_ids)
        audio_mask = (input_ids == audio_pad_token_id)
        audio_positions = torch.where(audio_mask[0])[0]
        if len(audio_positions) > 0:
            first_pad_pos = int(audio_positions[0].item())
            last_pad_pos = int(audio_positions[-1].item())
            before_audio = text_embeds[0, :first_pad_pos, :]
            after_audio = text_embeds[0, last_pad_pos + 1:, :]
            inputs_embeds = torch.cat(
                [before_audio[None], audio_embeds[None], after_audio[None]], dim=1
            )
        else:
            inputs_embeds = text_embeds
        generated = input_ids.clone()
    else:
        batch_size = audio_embeds.shape[0] if audio_embeds.ndim == 3 else 1
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds[None]
        inputs_embeds = audio_embeds
        generated = torch.full(
            (batch_size, 1), self.config.bos_token_id,
            dtype=torch.int64, device=inputs_embeds.device,
        )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    eos_token_ids = self.config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_tensor = torch.tensor(eos_token_ids, dtype=torch.int64, device=generated.device)

    # Prefill: process all input tokens, cache K/V via decode()
    logits, past_kv = self.decode(inputs_embeds=inputs_embeds, use_cache=True)

    # Decode loop: one token at a time, passing cached K/V
    for _ in range(max_new_tokens):
        next_token_logits = logits[:, -1, :] / temperature

        if top_k > 0 and top_k < next_token_logits.shape[-1]:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
            top_k_logits_shifted = top_k_logits - torch.max(
                top_k_logits, dim=-1, keepdim=True
            ).values
            exp_logits = torch.exp(top_k_logits_shifted)
            probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
            cumprobs = torch.cumsum(probs, dim=-1)
            samples = torch.rand((batch_size, 1), device=next_token_logits.device)
            next_token_idx = torch.argmax(
                (cumprobs >= samples).to(torch.float32), dim=-1
            )
            next_token = torch.gather(
                top_k_indices, dim=-1, index=next_token_idx[:, None]
            )
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        next_token_flat = next_token.flatten()
        is_eos = torch.any(next_token_flat[:, None] == eos_tensor[None, :], dim=1)
        finished = finished | is_eos
        if torch.all(finished):
            break

        # Only process ONE new token, decode() handles KV cache via past_key_values
        new_embeds = self.text_decoder.embed_tokens(next_token)
        logits, past_kv = self.decode(
            inputs_embeds=new_embeds, past_key_values=past_kv, use_cache=True
        )

    return generated


_v8b_patched = False

def _try_patch_v8b():
    """Attempt to patch generate_v8b onto GlmAsrModel. Safe to call multiple times."""
    global _v8b_patched
    if _v8b_patched:
        return
    import sys
    for mod_name in ('model', 'glm_asr_triton_template.model'):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, 'GlmAsrModel') and not hasattr(mod.GlmAsrModel, 'generate_v8b'):
            mod.GlmAsrModel.generate_v8b = _generate_v8b
            _v8b_patched = True
            return


if __name__ == "__main__":
    print("Testing Triton Layers...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== RMSNorm ===")
    norm = RMSNorm(256)
    x = torch.randn(2, 16, 256, device=device, dtype=torch.float32)
    y = norm(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== LayerNorm ===")
    ln = LayerNorm(256)
    y = ln(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== GELU ===")
    y = gelu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== SiLU ===")
    y = silu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Linear ===")
    linear = Linear(256, 512)
    y = linear(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Embedding ===")
    emb = Embedding(1000, 256)
    ids = torch.randint(0, 1000, (2, 16), device=device, dtype=torch.int32)
    y = emb(ids)
    print(f"Input: {ids.shape} -> Output: {y.shape}")

    print("\n=== Softmax ===")
    x_sm = torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32)
    y = softmax(x_sm, axis=-1)
    print(f"Input: {x_sm.shape} -> Output: {y.shape}")
    print(f"Sum along last axis: {float(y[0, 0, 0].sum()):.6f} (should be 1.0)")

    print("\n=== MLP ===")
    mlp = MLP(256, 512, activation="silu", use_gating=True)
    y = mlp(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\nAll Triton layers working!")
