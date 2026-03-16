"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple


from layers import GPU

# Warmup autotune cache: maps shape signature → best (BLOCK_M, BLOCK_N, nstages, nwarps)
_AUTOTUNE_CACHE = {}


def warmup_attention_tiles(seq_q, seq_k, head_dim, num_batch_heads, is_causal=False, n_trials=3):
    """One-time warmup autotune for fixed-shape attention (encoder/prefill).

    Tests candidate tile configurations on small random tensors and caches the fastest.
    Only useful for shapes that repeat (encoder, prefill) — NOT for decode where seq_k changes.
    """
    if not torch.cuda.is_available():
        return

    from layers import _compute_attention_tiles, next_power_of_two

    head_dim_padded = next_power_of_two(head_dim)
    tag = f"{head_dim_padded}_{seq_q}_{seq_k}_{is_causal}"
    if tag in _AUTOTUNE_CACHE:
        return

    # Generate candidate tile configs from the dynamic computation + variations
    base = _compute_attention_tiles(head_dim, GPU.smem_per_block)
    candidates = [base]

    # Add smaller/larger variations that fit in shared memory
    overhead = 8 * 1024
    usable = max(GPU.smem_per_block - overhead, 16 * 1024)
    max_budget = usable // (head_dim_padded * 4)

    for bm in (128, 64, 32, 16):
        for bn in (128, 64, 32, 16):
            if bm + 2 * bn > max_budget:
                continue
            if bm > seq_q:
                continue
            for ns in (1, 2):
                if ns == 2 and GPU.smem_per_block <= 150 * 1024:
                    continue
                nw = 8 if bm * bn >= 4096 else 4
                cfg = (bm, bn, ns, nw)
                if cfg not in candidates:
                    candidates.append(cfg)

    # Benchmark each candidate
    q = torch.randn(num_batch_heads, seq_q, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(num_batch_heads, seq_k, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(num_batch_heads, seq_k, head_dim, device='cuda', dtype=torch.float32)
    scale = 1.0 / (head_dim ** 0.5)

    best_time = float('inf')
    best_cfg = base

    for bm, bn, ns, nw in candidates:
        output = torch.empty_like(q)
        grid = (triton.cdiv(seq_q, bm), num_batch_heads)
        try:
            # Warmup
            for _ in range(2):
                flash_attention_kernel[grid](
                    q, k, v, output, q,  # dummy mask
                    float(scale), seq_q, seq_k, head_dim,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    output.stride(0), output.stride(1), output.stride(2),
                    0, 0, 0,
                    IS_CAUSAL=is_causal, HAS_MASK=False,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=head_dim_padded,
                    num_stages=ns, num_warps=nw,
                )
            torch.cuda.synchronize()

            # Timed runs
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_trials):
                flash_attention_kernel[grid](
                    q, k, v, output, q,
                    float(scale), seq_q, seq_k, head_dim,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    output.stride(0), output.stride(1), output.stride(2),
                    0, 0, 0,
                    IS_CAUSAL=is_causal, HAS_MASK=False,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=head_dim_padded,
                    num_stages=ns, num_warps=nw,
                )
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / n_trials

            if elapsed < best_time:
                best_time = elapsed
                best_cfg = (bm, bn, ns, nw)
        except Exception:
            continue  # Config doesn't work (OOM, invalid config, etc.)

    _AUTOTUNE_CACHE[tag] = best_cfg


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

    Uses the fused Triton flash-attention kernel on CUDA and a simple Torch
    reference path on CPU. For GQA, KV heads are expanded explicitly so the
    fused kernel always operates on matching query/key head counts.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape
    use_gqa = num_kv_heads != num_heads

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    if use_gqa:
        k = _expand_kv_heads(k, num_heads)
        v = _expand_kv_heads(v, num_heads)

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

        # Choose tile sizes: autotuned cache (opt-in) > GPU profile (default)
        # Use warmup_attention_tiles() to pre-tune for known fixed shapes.
        # Decode shapes (seq_q <= 4) already use SDPA fallback above.
        BLOCK_D = head_dim_padded
        attn_tag = f"{head_dim_padded}_{seq_q}_{seq_k}_{is_causal}"
        if attn_tag in _AUTOTUNE_CACHE:
            BLOCK_M, BLOCK_N, nstages, nwarps = _AUTOTUNE_CACHE[attn_tag]
        else:
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
