#!/usr/bin/env python3
"""
Recompute arithmetic-intensity-related quantities when encoder sequence length N changes.

Default N=1500 matches audio_encoder output seq len for test_audio.wav
(mel shape (1,128,3000) -> conv -> 1500).

Usage:
  python compute_ai_encoder_n.py
  python compute_ai_encoder_n.py --n 750
  python compute_ai_encoder_n.py --n 1500 --block-m 128
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderFlashAttention:
    n: int
    d_k: int
    block_m: int
    bytes_elem_fp16: int = 2

    @property
    def flops_qkt(self) -> int:
        return 2 * self.n * self.n * self.d_k

    @property
    def flops_pv(self) -> int:
        return 2 * self.n * self.n * self.d_k

    @property
    def flops_softmax(self) -> int:
        return 5 * self.n * self.n

    @property
    def flops_per_head(self) -> int:
        return self.flops_qkt + self.flops_pv + self.flops_softmax

    @property
    def num_q_tiles(self) -> int:
        return (self.n + self.block_m - 1) // self.block_m

    @property
    def dram_bytes_flash_tiled_model(self) -> int:
        """
        Same model as docs/ai_manual_calculation.md:
        Q, O: one full (n x d_k) footprint; K and V each reloaded once per Q-tile.
        """
        qo = self.n * self.d_k * self.bytes_elem_fp16
        kv_per_pass = self.n * self.d_k * self.bytes_elem_fp16
        return qo + self.num_q_tiles * kv_per_pass + self.num_q_tiles * kv_per_pass + qo

    @property
    def ai_flash(self) -> float:
        return self.flops_per_head / self.dram_bytes_flash_tiled_model

    @property
    def dram_bytes_three_kernel_fp32_scores(self) -> int:
        """Materialized score matrix S in fp32: n*n*4; same traffic pattern as manual doc."""
        q_bytes = self.n * self.d_k * self.bytes_elem_fp16
        k_bytes = self.n * self.d_k * self.bytes_elem_fp16
        v_bytes = self.n * self.d_k * self.bytes_elem_fp16
        o_bytes = self.n * self.d_k * self.bytes_elem_fp16
        s_bytes = self.n * self.n * 4
        return (
            q_bytes
            + k_bytes
            + s_bytes
            + s_bytes
            + s_bytes
            + s_bytes
            + v_bytes
            + o_bytes
        )

    @property
    def ai_three_kernel(self) -> float:
        return self.flops_per_head / self.dram_bytes_three_kernel_fp32_scores


def gelu_example(n_seq: int, hidden: int = 1280, flops_per_elem: int = 10) -> dict:
    n_el = n_seq * hidden
    flops = flops_per_elem * n_el
    bytes_total = n_el * 2 + n_el * 2  # fp16 load + store
    return {
        "elements": n_el,
        "flops": flops,
        "bytes": bytes_total,
        "ai": flops / bytes_total,
    }


def linear_encoder_q_proj(n_seq: int, hidden: int = 1280) -> dict:
    m, k, n = n_seq, hidden, hidden
    flops = 2 * m * k * n
    bytes_ = m * k * 2 + k * n * 2 + m * n * 2
    return {"flops": flops, "bytes": bytes_, "ai": flops / bytes_}


def rope_encoder_layer(n_seq: int, num_heads: int = 20, rotary_pairs: int = 32, flops_per_pair: int = 6) -> dict:
    """Same FLOP count style as ai_manual_calculation.md (20×N×32×6 for the listed rotary work)."""
    flops = num_heads * n_seq * rotary_pairs * flops_per_pair
    bytes_750 = 5_856_000
    bytes_n = int(bytes_750 * (n_seq / 750))
    return {"flops": flops, "bytes_est": bytes_n, "ai": flops / bytes_n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500, help="Encoder self-attention sequence length")
    p.add_argument("--d-k", type=int, default=64, dest="d_k", help="Encoder head dim")
    p.add_argument("--block-m", type=int, default=128, help="Q-tile rows (for ceil(N/BLOCK_M) count)")
    p.add_argument("--three-kernel", action="store_true", help="Also print 3-kernel fp32-score AI")
    args = p.parse_args()

    fa = EncoderFlashAttention(n=args.n, d_k=args.d_k, block_m=args.block_m)

    print("=== Encoder flash attention (fused, tiled) ===")
    print(f"N={fa.n}, d_k={fa.d_k}, BLOCK_M={fa.block_m} -> num_Q_tiles={fa.num_q_tiles}")
    print(f"FLOPs/head:  QK^T={fa.flops_qkt:,}  PV={fa.flops_pv:,}  softmax~={fa.flops_softmax:,}  total={fa.flops_per_head:,}")
    print(f"DRAM bytes/head (tiled model): {fa.dram_bytes_flash_tiled_model:,} ({fa.dram_bytes_flash_tiled_model/1e6:.3f} MB)")
    print(f"AI (flash):  {fa.ai_flash:.2f} FLOP/byte")
    print()

    if args.three_kernel:
        print("=== 3-kernel attention (fp32 score materialized) ===")
        print(f"DRAM bytes/head: {fa.dram_bytes_three_kernel_fp32_scores:,} ({fa.dram_bytes_three_kernel_fp32_scores/1e6:.3f} MB)")
        print(f"AI (3-kernel): {fa.ai_three_kernel:.2f} FLOP/byte")
        print()

    g = gelu_example(args.n)
    print("=== GELU (one encoder MLP activation, hidden=1280) ===")
    print(f"N_elem = {args.n}*1280 = {g['elements']:,}")
    print(f"FLOPs ~10/elem = {g['flops']:,}, bytes = {g['bytes']:,}, AI = {g['ai']:.2f}")
    print()

    lin = linear_encoder_q_proj(args.n)
    print("=== Linear Q projection x(N×1280) @ W(1280×1280) ===")
    print(f"FLOPs = {lin['flops']:,}, bytes = {lin['bytes']:,}, AI = {lin['ai']:.1f}")
    print()

    r = rope_encoder_layer(args.n)
    print("=== RoPE fused pair (encoder layer, scaled byte est from N=750 baseline) ===")
    print(f"FLOPs ≈ {r['flops']:,}, bytes_est ≈ {r['bytes_est']:,}, AI ≈ {r['ai']:.2f}")
    print()

    print("Ridge (H200 MIG): ~12.7 FLOP/byte  |  Ridge (RTX 5090 ref): ~58.5 FLOP/byte")


if __name__ == "__main__":
    main()