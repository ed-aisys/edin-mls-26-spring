"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement RoPE using Triton kernels
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from layers import _GPU_TIER


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.

    *** TODO: Implement this kernel ***

    Grid: (seq_len,)
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    pos = tl.load(positions_ptr + pid * stride_pos)
    inv = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv

    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_half, mask=mask)
    tl.store(
        cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1,
        cos_half,
        mask=mask,
    )
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(
        sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1,
        sin_half,
        mask=mask,
    )


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


# ============================================================================
# Fused RoPE Pair Kernel (from meave branch — single launch for Q+K)
# ============================================================================

@triton.jit
def fused_rope_pair_kernel(
    q_ptr,          # query  (B*Hq, S, D)
    k_ptr,          # key    (B*Hk, S, D)
    cos_ptr,        # cos    (S, half_dim)
    sin_ptr,        # sin    (S, half_dim)
    qo_ptr,         # q out  (B*Hq, S, D)
    ko_ptr,         # k out  (B*Hk, S, D)
    half_dim,
    head_dim,
    seq_len,
    total_qh,       # B * num_q_heads
    total_kh,       # B * num_kv_heads
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_cs, stride_cd,
    stride_qos, stride_qod,
    stride_kos, stride_kod,
    BLOCK_HD: tl.constexpr,
):
    """
    Fused RoPE kernel that processes BOTH Q and K in a single grid launch.
    Grid: ((total_qh + total_kh) * seq_len,)
    Programs 0..total_qh*seq_len-1 handle Q, the rest handle K.
    """
    pid = tl.program_id(0)
    total_q_programs = total_qh * seq_len

    is_q = pid < total_q_programs

    if is_q:
        bh = pid // seq_len
        s = pid % seq_len
        x_ptr = q_ptr
        o_ptr = qo_ptr
        stride_s = stride_qs
        stride_d = stride_qd
        stride_os_val = stride_qos
        stride_od_val = stride_qod
        stride_bh = stride_qs * seq_len
        stride_obh = stride_qos * seq_len
    else:
        local_pid = pid - total_q_programs
        bh = local_pid // seq_len
        s = local_pid % seq_len
        x_ptr = k_ptr
        o_ptr = ko_ptr
        stride_s = stride_ks
        stride_d = stride_kd
        stride_os_val = stride_kos
        stride_od_val = stride_kod
        stride_bh = stride_ks * seq_len
        stride_obh = stride_kos * seq_len

    offs_half = tl.arange(0, BLOCK_HD)
    mask_half = offs_half < half_dim

    base = bh * stride_bh + s * stride_s
    x1 = tl.load(x_ptr + base + offs_half * stride_d, mask=mask_half, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs_half + half_dim) * stride_d, mask=mask_half, other=0.0).to(tl.float32)

    cos_val = tl.load(cos_ptr + s * stride_cs + offs_half * stride_cd, mask=mask_half, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + s * stride_cs + offs_half * stride_cd, mask=mask_half, other=0.0).to(tl.float32)

    out1 = x1 * cos_val - x2 * sin_val
    out2 = x2 * cos_val + x1 * sin_val

    obase = bh * stride_obh + s * stride_os_val
    tl.store(o_ptr + obase + offs_half * stride_od_val, out1, mask=mask_half)
    tl.store(o_ptr + obase + (offs_half + half_dim) * stride_od_val, out2, mask=mask_half)

    # Copy passthrough dimensions (for partial RoPE in audio encoder)
    remaining = head_dim - 2 * half_dim
    if remaining > 0:
        offs_rest = tl.arange(0, BLOCK_HD)
        mask_rest = offs_rest < remaining
        rest_in = tl.load(x_ptr + base + (2 * half_dim + offs_rest) * stride_d, mask=mask_rest, other=0.0)
        tl.store(o_ptr + obase + (2 * half_dim + offs_rest) * stride_od_val, rest_in, mask=mask_rest)


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch (CPU fallback)."""
    batch, num_heads, seq_len, _ = x.shape

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]

    cos_expanded = cos[None, None, :, :]
    sin_expanded = sin[None, None, :, :]

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x2 * cos_expanded + x1 * sin_expanded

    if head_dim > half_dim * 2:
        x_pass = x[..., half_dim * 2 :]
        return torch.cat([x1_rot, x2_rot, x_pass], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings.
    On CUDA, uses a single fused Triton kernel launch for both Q and K (from meave branch).
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if sin.dtype != torch.float32:
        sin = sin.to(torch.float32)
    if not sin.is_contiguous():
        sin = sin.contiguous()

    # CUDA fast path: single fused kernel for both Q and K (from meave branch)
    if q.is_cuda:
        total_qh = batch * num_q_heads
        total_kh = batch * num_kv_heads
        BLOCK_HD = next_power_of_two(max(half_dim, head_dim - 2 * half_dim, 1))

        q_flat = q.reshape(total_qh, seq_len, head_dim)
        if not q_flat.is_contiguous():
            q_flat = q_flat.contiguous()
        k_flat = k.reshape(total_kh, seq_len, head_dim)
        if not k_flat.is_contiguous():
            k_flat = k_flat.contiguous()
        qo_flat = torch.empty_like(q_flat)
        ko_flat = torch.empty_like(k_flat)

        cos_half = cos[:seq_len]
        if not cos_half.is_contiguous():
            cos_half = cos_half.contiguous()
        sin_half = sin[:seq_len]
        if not sin_half.is_contiguous():
            sin_half = sin_half.contiguous()

        total_programs = (total_qh + total_kh) * seq_len
        fused_rope_pair_kernel[(total_programs,)](
            q_flat, k_flat,
            cos_half, sin_half,
            qo_flat, ko_flat,
            half_dim, head_dim, seq_len,
            total_qh, total_kh,
            q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(1), k_flat.stride(2),
            cos_half.stride(0), cos_half.stride(1),
            qo_flat.stride(1), qo_flat.stride(2),
            ko_flat.stride(1), ko_flat.stride(2),
            BLOCK_HD=BLOCK_HD,
            num_stages=2 if _GPU_TIER == 'datacenter' else 1,
            num_warps=8 if _GPU_TIER == 'datacenter' else 4,
        )

        q_out = qo_flat.reshape(batch, num_q_heads, seq_len, head_dim)
        k_out = ko_flat.reshape(batch, num_kv_heads, seq_len, head_dim)
        return q_out.to(q.dtype), k_out.to(k.dtype)

    # CPU fallback
    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)

    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    print("\nTriton RoPE working!")
