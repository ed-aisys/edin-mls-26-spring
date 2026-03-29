"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple


from layers import GPU


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Fused Flash Attention Kernel (online softmax, single kernel launch)
# Old 3-kernel pipeline (attention_scores, softmax_inplace, attention_output,
# causal_mask) removed — superseded by flash_attention_kernel below.
# ============================================================================

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    mask_ptr,
    scale,
    seq_q, seq_k, head_dim: tl.constexpr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_mb, stride_mq, stride_mk,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused Flash Attention with online softmax.
    Grid: (cdiv(seq_q, BLOCK_M), batch_heads)

    Processes Q in tiles of BLOCK_M rows, iterating over K/V in BLOCK_N chunks.
    Uses online softmax to avoid materializing the full attention matrix.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Offsets for this Q tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                      # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_D)                      # [BLOCK_D]

    # Base pointers for this batch/head
    q_base = q_ptr + pid_bh * stride_qb
    k_base = k_ptr + pid_bh * stride_kb
    v_base = v_ptr + pid_bh * stride_vb
    if HAS_MASK:
        m_base = mask_ptr + pid_bh * stride_mb

    # Load Q tile: [BLOCK_M, BLOCK_D]
    q_mask = (offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )
    q = (q * scale).to(tl.float32)

    # Running softmax state
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)  # running max
    l_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)            # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)              # accumulator

    # Determine K/V iteration range
    if IS_CAUSAL:
        # For causal, only need to go up to the last row in this Q block
        kv_len = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M)
    else:
        kv_len = seq_k

    # Iterate over K/V blocks
    for start_n in range(0, kv_len, BLOCK_N):
        cur_offs_n = start_n + offs_n  # [BLOCK_N]

        # Load K block: [BLOCK_N, BLOCK_D]
        k_mask = (cur_offs_n[:, None] < seq_k) & (offs_d[None, :] < head_dim)
        k = tl.load(
            k_base + cur_offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        # S = Q @ K^T: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k))

        # Mask out-of-bounds keys
        s = tl.where(cur_offs_n[None, :] < seq_k, s, -float("inf"))

        # Apply causal mask
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= cur_offs_n[None, :], s, -float("inf"))

        # Apply attention mask bias
        if HAS_MASK:
            am = tl.load(
                m_base + offs_m[:, None] * stride_mq + cur_offs_n[None, :] * stride_mk,
                mask=(offs_m[:, None] < seq_q) & (cur_offs_n[None, :] < seq_k),
                other=0.0,
            )
            s = s + am

        # Online softmax update
        m_ij = tl.max(s, axis=1)                        # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)                   # [BLOCK_M]
        alpha = tl.exp(m_i - m_new)                      # rescale old
        p = tl.exp(s - m_new[:, None])                   # new weights [BLOCK_M, BLOCK_N]

        # Update running sum and rescale accumulator
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc

        # Load V block: [BLOCK_N, BLOCK_D]
        v_mask = (cur_offs_n[:, None] < seq_k) & (offs_d[None, :] < head_dim)
        v = tl.load(
            v_base + cur_offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        # Accumulate: acc += P @ V
        acc += tl.dot(p, v)

        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output: [BLOCK_M, BLOCK_D]
    o_base = o_ptr + pid_bh * stride_ob
    o_mask = (offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim)
    tl.store(
        o_base + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
        acc,
        mask=o_mask,
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
ATTENTION_MODE_ENV = "GLM_ASR_ATTENTION_MODE"
VALID_ATTENTION_MODES = {"auto", "three_kernel", "sdpa_all"}


def _get_attention_mode() -> str:
    """Return the requested attention backend for benchmarking and ablations."""
    mode = os.environ.get(ATTENTION_MODE_ENV, "auto").strip().lower().replace("-", "_")
    if mode not in VALID_ATTENTION_MODES:
        raise ValueError(
            f"Unsupported {ATTENTION_MODE_ENV}={mode!r}. "
            f"Expected one of: {', '.join(sorted(VALID_ATTENTION_MODES))}."
        )
    return mode


def _expand_kv_heads(
    x: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:
    """Expand grouped KV heads to query-head count for non-SDPA fallbacks."""
    batch, num_kv_heads, seq_len, head_dim = x.shape
    num_repeats = num_query_heads // num_kv_heads
    x_expanded = x[:, :, None, :, :].expand(
        batch, num_kv_heads, num_repeats, seq_len, head_dim
    )
    return x_expanded.reshape(batch, num_query_heads, seq_len, head_dim)


def _materialized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """
    Original score-materializing attention path.

    On small problems we keep the historical Triton 3-kernel implementation.
    For larger shapes we fall back to explicit Torch matmul/softmax/matmul,
    which still materializes the full attention matrix in DRAM.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

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
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k)
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
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=seq_k_padded
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


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Default mode uses the fused Triton flash-attention kernel for seq_q > 4
    and PyTorch SDPA for tiny KV-cached decode steps. Benchmark mode
    GLM_ASR_ATTENTION_MODE=three_kernel resurrects the historical score-
    materializing path without changing the rest of the codebase.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape
    use_gqa = num_kv_heads != num_heads
    attention_mode = _get_attention_mode()

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    if use_gqa:
        k = _expand_kv_heads(k, num_heads)
        v = _expand_kv_heads(v, num_heads)

    if attention_mode == "sdpa_all" and q.is_cuda:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=scale
        )

    if attention_mode == "three_kernel":
        return _materialized_attention(q, k, v, attention_mask, is_causal, scale)

    head_dim_padded = next_power_of_two(head_dim)

    if q.is_cuda and seq_q <= 4:
        # For very short queries (KV-cached decode), use PyTorch SDPA
        # which avoids Triton kernel launch overhead for tiny problems
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=scale
        )

    if q.is_cuda:
        # Fused Flash Attention kernel — single kernel launch with online softmax.
        # No materialization of the full scores matrix in DRAM.
        BH = batch * num_heads
        q_flat = q.reshape(BH, seq_q, head_dim).contiguous()
        k_flat = k.reshape(BH, seq_k, head_dim).contiguous()
        v_flat = v.reshape(BH, seq_k, head_dim).contiguous()

        # Prepare attention mask for kernel (flatten batch*heads dimension)
        has_mask = attention_mask is not None
        if has_mask:
            # attention_mask is (batch, 1, seq_q, seq_k) or (batch, num_heads, seq_q, seq_k)
            mask_flat = attention_mask.expand(batch, num_heads, seq_q, seq_k)
            mask_flat = mask_flat.reshape(BH, seq_q, seq_k).contiguous().to(torch.float32)
            mask_strides = (mask_flat.stride(0), mask_flat.stride(1), mask_flat.stride(2))
        else:
            mask_flat = q_flat  # dummy, not accessed
            mask_strides = (0, 0, 0)

        output = torch.empty(
            (BH, seq_q, head_dim), dtype=torch.float32, device=q.device
        )

        BLOCK_D = head_dim_padded
        BLOCK_M, BLOCK_N, nstages, nwarps = GPU.get_attention_tiles(head_dim, seq_q)

        grid = (triton.cdiv(seq_q, BLOCK_M), BH)
        flash_attention_kernel[grid](
            q_flat, k_flat, v_flat, output,
            mask_flat,
            float(scale),
            seq_q, seq_k, head_dim,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            mask_strides[0], mask_strides[1], mask_strides[2],
            IS_CAUSAL=is_causal,
            HAS_MASK=has_mask,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_stages=nstages,
            num_warps=nwarps,
        )

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    return _materialized_attention(q, k, v, attention_mask, is_causal, scale)


def _reference_attention(q, k, v, attention_mask=None, is_causal=False, scale=None):
    """Pure PyTorch reference for numerical parity testing."""
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])
    scores = torch.einsum("bnqd,bnkd->bnqk", q.float(), k.float()) * scale
    if is_causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        causal = torch.triu(torch.ones(seq_q, seq_k, device=q.device), diagonal=1) * -1e9
        scores = scores + causal[None, None]
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    scores = scores - scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", weights, v.float()).to(q.dtype)


def _make_test_mask(kind: str, batch: int, num_heads: int, seq_q: int, seq_k: int, device):
    """Create a deterministic additive attention mask for parity tests."""
    if kind == "none":
        return None

    mask_heads = 1 if kind == "b1" else num_heads
    mask = torch.zeros((batch, mask_heads, seq_q, seq_k), device=device, dtype=torch.float32)

    # Mask a suffix so we exercise large blocked-out regions.
    mask[..., seq_k // 2 :] = -1e9

    # Also mask a small band in the middle to catch indexing bugs.
    if seq_k > 8:
        mid_start = seq_k // 3
        mid_end = min(seq_k, mid_start + 2)
        mask[..., : max(1, seq_q // 2), mid_start:mid_end] = -1e9

    return mask


if __name__ == "__main__":
    print("Testing Triton Flash Attention with numerical parity assertions...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("Warning: CUDA is unavailable, so only the CPU fallback path is being tested.")

    atol = 1e-2  # flash attention fp32 accumulation tolerance

    def run_case(name, seed, batch, num_heads, num_kv_heads, seq_q, seq_k, head_dim, *, is_causal=False, mask_kind="none"):
        torch.manual_seed(seed)

        q = torch.randn(batch, num_heads, seq_q, head_dim, device=device)
        k = torch.randn(batch, num_kv_heads, seq_k, head_dim, device=device)
        v = torch.randn(batch, num_kv_heads, seq_k, head_dim, device=device)
        mask = _make_test_mask(mask_kind, batch, num_heads, seq_q, seq_k, device)

        got = scaled_dot_product_attention(
            q, k, v, attention_mask=mask, is_causal=is_causal
        )

        if num_kv_heads != num_heads:
            k_ref = _expand_kv_heads(k, num_heads)
            v_ref = _expand_kv_heads(v, num_heads)
        else:
            k_ref = k
            v_ref = v

        if mask is not None and mask.shape[1] == 1:
            mask_ref = mask.expand(batch, num_heads, seq_q, seq_k)
        else:
            mask_ref = mask

        ref = _reference_attention(
            q, k_ref, v_ref, attention_mask=mask_ref, is_causal=is_causal
        )

        diff = (got.float() - ref.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        status = "PASS" if max_diff < atol else "FAIL"
        print(f"  [{status}] {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        assert max_diff < atol, f"{name} exceeded tolerance: {max_diff} > {atol}"

    gpu_cases = [
        ("basic_hd64_seq32", 1000, 2, 4, 4, 32, 32, 64, False, "none"),
        ("causal_hd64_seq32", 1001, 2, 4, 4, 32, 32, 64, True, "none"),
        ("masked_b1_hd64_seq32", 1002, 2, 4, 4, 32, 32, 64, False, "b1"),
        ("gqa_hd64_seq32", 1003, 2, 4, 2, 32, 32, 64, False, "none"),
        ("encoder_ragged_175", 1004, 1, 20, 20, 175, 175, 64, False, "none"),
        ("encoder_ragged_175_mask1", 1005, 1, 20, 20, 175, 175, 64, False, "b1"),
        ("encoder_short_47_maskh", 1006, 2, 20, 20, 47, 47, 64, False, "bh"),
        ("basic_hd128_seq32", 1007, 2, 4, 4, 32, 32, 128, False, "none"),
        ("causal_hd128_seq32", 1008, 2, 4, 4, 32, 32, 128, True, "none"),
        ("causal_mask_hd128_seq32", 1009, 2, 4, 4, 32, 32, 128, True, "b1"),
        ("decoder_prefill_93", 1010, 1, 16, 16, 93, 93, 128, True, "none"),
        ("decoder_prefill_93_mask1", 1011, 1, 16, 16, 93, 93, 128, True, "b1"),
        ("decoder_prefill_gqa_93", 1012, 1, 16, 4, 93, 93, 128, True, "none"),
        ("decode_step_1x64", 1013, 2, 4, 4, 1, 64, 128, False, "none"),
        ("decode_step_causal_mask_1x64", 1014, 2, 4, 4, 1, 64, 128, True, "b1"),
        ("decode_step_gqa_1x93", 1015, 2, 16, 4, 1, 93, 128, False, "bh"),
        ("decoder_nonpow2_17x61", 1016, 2, 16, 4, 17, 61, 128, False, "b1"),
    ]
    cpu_cases = [
        ("basic_hd64_seq32", 1000, 2, 4, 4, 32, 32, 64, False, "none"),
        ("causal_hd64_seq32", 1001, 2, 4, 4, 32, 32, 64, True, "none"),
        ("masked_b1_hd64_seq32", 1002, 2, 4, 4, 32, 32, 64, False, "b1"),
        ("gqa_hd64_seq32", 1003, 2, 4, 2, 32, 32, 64, False, "none"),
        ("decode_step_causal_mask_1x64", 1014, 2, 4, 4, 1, 64, 128, True, "b1"),
    ]

    cases = gpu_cases if device.type == "cuda" else cpu_cases
    for idx, case in enumerate(cases, start=1):
        name, seed, batch, num_heads, num_kv_heads, seq_q, seq_k, head_dim, is_causal, mask_kind = case
        print(f"\n{idx}. {name}:")
        run_case(
            name,
            seed,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_k,
            head_dim,
            is_causal=is_causal,
            mask_kind=mask_kind,
        )

    print("\nAll parity tests passed!")
