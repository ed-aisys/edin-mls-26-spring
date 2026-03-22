"""
Triton Neural Network Layers
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement core layers using Triton kernels
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

# GELU element-wise kernel launch (tune with ../benchmark_tile_sizes.py, then set here)
GELU_BLOCK_SIZE = 1024
GELU_NUM_WARPS = 4


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

    # ============================================================================
    # TODO: Implement RMSNorm kernel
    # ============================================================================
    #
    # Step 1: Load input row and weight
    # Step 2: Compute variance = mean(x^2)
    # Step 3: Normalize: x / sqrt(variance + eps)
    # Step 4: Apply weight and store
    # YOUR CODE HERE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    variance = tl.sum(x_f32 * x_f32, axis=0) / hidden_size
    rnorm = x_f32 * tl.rsqrt(variance + eps)
    y = rnorm * w
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

    # ============================================================================
    # TODO: Implement LayerNorm kernel
    # ============================================================================
    #
    # Step 1: Load input, weight, and bias
    # Step 2: Compute mean
    # Step 3: Center the data
    # Step 4: Compute variance = mean((x - mean)^2)
    # Step 5: Normalize and apply affine transform

    # YOUR CODE HERE
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask = mask, other = 0.0)
    w = tl.load(w_ptr + offs, mask = mask, other = 0.0)
    b = tl.load(b_ptr + offs, mask = mask, other = 0.0)
    x_f32 = x.to(tl.float32)
    x_mean = tl.sum(x_f32,axis=0) / hidden_size
    x_centered = x_f32 - x_mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    x_norm = x_centered * tl.rsqrt(x_var + eps)
    y = x_norm * w + b
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    GELU using tanh approximation.

    *** TODO: Implement this kernel ***
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement GELU kernel
    # ============================================================================
    #
    # Step 1: Load input tile
    # Step 2: Compute tanh approximation
    # Step 3: Store output
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    #sqrt(2/pi) =0.7978845608
    inner = tl.sqrt(2.0 / math.pi) * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
    # tl.libdevice.tanh(inner) didn't work so change it to tl.extra.cuda.libdevice.tanh(inner) like in the working example
    y = 0.5 * x_f32 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)
    



@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    SiLU/Swish: x * sigmoid(x)

    *** TODO: Implement this kernel ***
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement SiLU kernel
    # ============================================================================
    #
    # Step 1: Load input tile
    # Step 2: Compute sigmoid
    # Step 3: Multiply and store

    offs = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask = mask, other = 0.0)
    x_f32 = x.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sigmoid
    tl.store(y_ptr + offs, y, mask = mask)
    


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

    # ============================================================================
    # TODO: Implement tiled matrix multiplication
    # ============================================================================
    #
    # Step 1: Initialize accumulator
    # Step 2: Loop over K tiles and accumulate tl.dot
    # Step 3: Store the result

    # YOUR CODE HERE
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


@triton.jit
def linear_gelu_kernel(
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

    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + tl.libdevice.tanh(inner))

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

    # ============================================================================
    # TODO: Implement softmax kernel
    # ============================================================================
    #
    # Step 1: Load row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store output
    # YOUR CODE HERE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
    x -= tl.max(x, axis=0) # For numerical stability
    exp_x = tl.exp(x)
    y = exp_x / tl.sum(exp_x, axis=0)
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention scores: Q @ K^T * scale."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def attention_output_kernel(
    weights_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention output: weights @ V."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        weights_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """Apply causal mask to attention scores."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


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
        self.use_triton = _is_power_of_two(hidden_size) # This flag will force a fallback to a PyTorch implementation of the kernels when the hidden_size is not a power of 2.

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

       
        if self.use_triton and x.is_cuda:  # remove self.use_triton flag from this if-statement in case you want to always run your Triton kernel regardless of whether hidden_size is a power of 2.
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            x_flat = x_flat.to(torch.float32)
            output = torch.empty_like(x_flat)

            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)

            block = next_power_of_two(self.hidden_size)
            rmsnorm_kernel[(batch_size,)](
                x_flat,
                self.weight,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=block,
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
        self.use_triton = _is_power_of_two(hidden_size)  # This flag will force a fallback to a PyTorch implementation of the kernels when the hidden_size is not a power of 2.

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if self.use_triton and x.is_cuda:  # remove self.use_triton flag from this if-statement in case you want to always run your Triton kernel regardless of whether hidden_size is a power of 2.
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            x_flat = x_flat.to(torch.float32)
            output = torch.empty_like(x_flat)

            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            if self.bias.device != x.device:
                self.bias = self.bias.to(x.device)

            block = next_power_of_two(self.hidden_size)
            layernorm_kernel[(batch_size,)](
                x_flat,
                self.weight,
                self.bias,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=block,
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
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = GELU_BLOCK_SIZE

    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)

    if x.is_cuda:
        gelu_kernel[grid](
            x_flat,
            output,
            total,
            BLOCK_SIZE=block,
            num_warps=GELU_NUM_WARPS,
        )
        return output[:total].reshape(original_shape).to(x.dtype)

    return torch.nn.functional.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation using Triton."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = 1024

    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)

    if x.is_cuda:
        silu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape).to(x.dtype)

    return torch.nn.functional.silu(x)


def get_activation(name: str):
    """Get activation function by name."""
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    """Linear layer with switchable backend (torch or Triton)."""

    TILE_M = 64
    TILE_N = 64
    TILE_K = 32
    NUM_WARPS = 4
    NUM_STAGES = 3

    BACKEND = "torch"

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param = torch.zeros(out_features, dtype=torch.float32) if bias else None

        self._weight_t_padded = None
        self._K_padded = None
        self._N_padded = None

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

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Torch matmul backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        x_2d = x.reshape(M, self.in_features).to(torch.float32)

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        output = x_2d @ self.weight.t()

        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param

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
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
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
        output = torch.empty(
            (batch_size, self.embedding_dim), dtype=torch.float32, device=indices_flat.device
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
    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)

    original_shape = x.shape
    batch_size = int(np.prod(x.shape[:-1]))
    seq_len = x.shape[-1]

    x_flat = x.reshape(batch_size, seq_len).to(torch.float32).contiguous()
    output = torch.empty_like(x_flat)

    if x.is_cuda:
        block = next_power_of_two(seq_len)
        softmax_kernel[(batch_size,)](
            x_flat,
            output,
            x_flat.stride(0),
            output.stride(0),
            seq_len,
            BLOCK_SIZE=block,
        )
        result = output.reshape(original_shape)
    else:
        result = torch.softmax(x, dim=-1)

    if axis != -1 and axis != len(original_shape) - 1:
        result = torch.movedim(result, -1, axis)

    return result


class MLP:
    """MLP with SwiGLU gating using Triton."""

    FUSED = True
    TILE_M, TILE_N, TILE_K = 32, 64, 32
    NUM_WARPS = 4
    NUM_STAGES = 2

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
            self._gate_weight_t = self.gate_proj.weight.t().contiguous()
            self._up_weight_t = self.up_proj.weight.t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gating and MLP.FUSED and x.is_cuda:
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
        x_2d = x.reshape(-1, self.hidden_size).to(torch.float32).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=torch.float32, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            gate_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=torch.float32, device=x.device
            )
            gate_w_padded[:K, :N] = self._gate_weight_t
            up_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=torch.float32, device=x.device
            )
            up_w_padded[:K, :N] = self._up_weight_t
        else:
            gate_w_padded = self._gate_weight_t
            up_w_padded = self._up_weight_t

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=torch.float32, device=x.device
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
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.down_proj(intermediate)


class EncoderMLP:
    """Encoder MLP (no gating) using Triton."""

    FUSED = True
    TILE_M, TILE_N, TILE_K = 128, 64, 32
    NUM_WARPS = 4
    NUM_STAGES = 3

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
            self._fc1_weight_t = self.fc1.weight.t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if EncoderMLP.FUSED and self.activation == "gelu" and x.is_cuda:
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
        x_2d = x.reshape(-1, self.hidden_size).to(torch.float32).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=torch.float32, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            fc1_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=torch.float32, device=x.device
            )
            fc1_w_padded[:K, :N] = self._fc1_weight_t
        else:
            fc1_w_padded = self._fc1_weight_t

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=torch.float32, device=x.device
        )

        grid = (
            triton.cdiv(M_pad, self.TILE_M),
            triton.cdiv(N_pad, self.TILE_N),
        )
        linear_gelu_kernel[grid](
            x_padded,
            fc1_w_padded,
            intermediate,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            fc1_w_padded.stride(0),
            fc1_w_padded.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        if self.bias_enabled and self.fc1.bias_param is not None:
            if self.fc1.bias_param.device != x.device:
                self.fc1.bias_param = self.fc1.bias_param.to(x.device)
            intermediate = intermediate + self.fc1.bias_param

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.fc2(intermediate)


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
