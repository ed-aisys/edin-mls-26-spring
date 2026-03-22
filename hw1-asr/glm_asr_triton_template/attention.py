"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

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
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    # YOUR CODE HERE

    col_idx = tl.arange(0, BLOCK_D)
    key_idx = tl.arange(0, BLOCK_K)

    q_base = q_ptr + pid_bh * stride_q0 + pid_q * stride_q1
    q_vec = tl.load(q_base + col_idx * stride_q2, mask=col_idx < head_dim, other=0.0)

    k_base = k_ptr + pid_bh * stride_k0
    k_mat = tl.load(
        k_base + key_idx[:, None] * stride_k1 + col_idx[None, :] * stride_k2,
        mask=(key_idx[:, None] < seq_k) & (col_idx[None, :] < head_dim),
        other=0.0,
    )

    dot = tl.sum(k_mat * q_vec[None, :], axis=1)
    s_base = scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1
    tl.store(s_base + key_idx * stride_s2, dot * scale, mask=key_idx < seq_k)


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store back

    # YOUR CODE HERE

    col_idx = tl.arange(0, BLOCK_SIZE)
    valid = col_idx < seq_k
    row_ptr = scores_ptr + row * stride_s
    x = tl.load(row_ptr + col_idx, mask=valid, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    tl.store(row_ptr + col_idx, numerator / tl.sum(numerator, axis=0), mask=valid)


@triton.jit
def attention_output_kernel(
    attn_ptr,
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
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights for this query
    # Step 2: Load all values for this batch_head
    # Step 3: Compute weighted sum
    # Step 4: Store output

    # YOUR CODE HERE

    key_idx = tl.arange(0, BLOCK_K)
    col_idx = tl.arange(0, BLOCK_D)

    w_base = attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1
    weights = tl.load(w_base + key_idx * stride_w2, mask=key_idx < seq_k, other=0.0)

    v_base = v_ptr + pid_bh * stride_v0
    val_mat = tl.load(
        v_base + key_idx[:, None] * stride_v1 + col_idx[None, :] * stride_v2,
        mask=(key_idx[:, None] < seq_k) & (col_idx[None, :] < head_dim),
        other=0.0,
    )

    result = tl.sum(val_mat * weights[:, None], axis=0)
    o_base = output_ptr + pid_bh * stride_o0 + pid_q * stride_o1
    tl.store(o_base + col_idx * stride_o2, result, mask=col_idx < head_dim)


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
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
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
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat,
            k_flat,
            scores,
            float(scale),
            seq_k_padded,
            head_dim_padded,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(
                    batch * num_heads, seq_q, seq_k
                )
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        block = seq_k_padded
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=block
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores,
            v_flat,
            output,
            seq_k_padded,
            head_dim_padded,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
