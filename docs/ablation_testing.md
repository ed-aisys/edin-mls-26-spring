# Ablation Testing: GLM-ASR Triton Kernel Optimizations

**Platform:** NVIDIA H200 MIG 3g.71gb (Hopper, 60 SMs, 70GB HBM3e, ~228KB shared memory)
**Cluster:** Edinburgh teaching cluster, `saxa.inf.ed.ac.uk`
**Date:** 2026-03-18
**Test audio:** 3.5s WAV, expected output: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"

---

## Test Matrix

Each test modifies a single variable from the optimized baseline, clears the Triton cache, and runs 2 warmup + 5 timed iterations via `benchmark_student.py`.

### Category 1: Precision Pipeline

| Test | Config | Expected Impact | Result |
|------|--------|-----------------|--------|
| `baseline_optimized` | fp16 pipeline (all opts on) | Reference | *pending* |
| `precision_fp32` | `Linear.BF16 = False` (full fp32) | Slower (2x bandwidth) | *pending* |

### Category 2: Kernel Fusion

| Test | Config | Expected Impact | Result |
|------|--------|-----------------|--------|
| `fusion_off_mlp` | `MLP.FUSED = False`, `EncoderMLP.FUSED = False` | Slower (+kernel launches, +VRAM round-trips) | *pending* |
| `fusion_off_rope` | Disable fused Q+K RoPE (force separate launches) | Slower (+launch overhead) | *pending* |

### Category 3: Backend

| Test | Config | Expected Impact | Result |
|------|--------|-----------------|--------|
| `backend_triton` | `Linear.BACKEND = "triton"` instead of cuBLAS | Slower for large GEMMs | *pending* |

### Category 4: SDPA Fallback Threshold

| Test | Config | Expected Impact | Result |
|------|--------|-----------------|--------|
| `sdpa_off` | Disable SDPA fallback entirely | Slower (custom kernel launch overhead for seq_q=1) | *pending* |
| `sdpa_threshold_1` | `seq_q <= 1` | Slightly slower | *pending* |
| `sdpa_threshold_8` | `seq_q <= 8` | Similar or slightly faster | *pending* |
| `sdpa_threshold_16` | `seq_q <= 16` | Unknown | *pending* |

### Category 5: Attention Tile Sizes (Encoder, head_dim=64)

Baseline on H200: (128, 128, nstages=2, nwarps=8)

| Test | BLOCK_M x BLOCK_N | nstages | nwarps | smem (est.) | Result |
|------|-------------------|---------|--------|-------------|--------|
| `attn_enc_128x128` (baseline) | 128 x 128 | 2 | 8 | ~116KB | *pending* |
| `attn_enc_128x64` | 128 x 64 | 2 | 8 | ~84KB | *pending* |
| `attn_enc_64x64` | 64 x 64 | 1 | 4 | ~68KB | *pending* |
| `attn_enc_64x32` | 64 x 32 | 1 | 4 | ~52KB | *pending* |

### Category 6: Attention Tile Sizes (Decoder, head_dim=128)

Baseline on H200: (128, 64, nstages=2, nwarps=8)

| Test | BLOCK_M x BLOCK_N | nstages | nwarps | smem (est.) | Result |
|------|-------------------|---------|--------|-------------|--------|
| `attn_dec_128x64` (baseline) | 128 x 64 | 2 | 8 | ~151KB | *pending* |
| `attn_dec_64x64` | 64 x 64 | 2 | 8 | ~131KB | *pending* |
| `attn_dec_64x32` | 64 x 32 | 2 | 8 | ~99KB | *pending* |
| `attn_dec_32x32` | 32 x 32 | 1 | 4 | ~83KB | *pending* |

### Category 7: num_stages Ablation (Encoder Attention)

All with 128x128 tiles, num_warps=8

| Test | num_stages | Expected Impact | Result |
|------|-----------|-----------------|--------|
| `enc_nstages_1` | 1 (no double-buffering) | Slower (no load/compute overlap) | *pending* |
| `enc_nstages_2` (baseline) | 2 (double-buffering) | Reference | *pending* |
| `enc_nstages_3` | 3 (triple-buffering) | Unknown (more smem, more overlap) | *pending* |

### Category 8: num_warps Ablation (Encoder Attention)

All with 128x128 tiles, num_stages=2

| Test | num_warps | Threads/block | Expected Impact | Result |
|------|----------|---------------|-----------------|--------|
| `enc_nwarps_4` | 4 | 128 | Slower (less latency hiding) | *pending* |
| `enc_nwarps_8` (baseline) | 8 | 256 | Reference | *pending* |
| `enc_nwarps_16` | 16 | 512 | Unknown (more parallelism vs register pressure) | *pending* |

### Category 9: Matmul Tile Sizes

| Test | TILE_M x TILE_N x TILE_K | Expected Impact | Result |
|------|-------------------------|-----------------|--------|
| `matmul_128x128x64` (baseline) | 128 x 128 x 64 | Reference | *pending* |
| `matmul_128x128x32` | 128 x 128 x 32 | Slower (more inner loop iterations) | *pending* |
| `matmul_128x64x32` | 128 x 64 x 32 | Slower (smaller tiles) | *pending* |
| `matmul_64x64x32` | 64 x 64 x 32 | Slower (consumer-GPU config on H200) | *pending* |

---

## Nsight Systems Profiling

*Pending — nsys profiling to be run on the optimized baseline configuration.*

Planned metrics:
- Per-kernel wall-clock timing
- SM occupancy per kernel
- Memory throughput (GB/s achieved vs peak)
- Kernel launch overhead distribution
- CUDA API call breakdown

---

## Results Summary

*Will be populated after ablation tests complete.*

### Key Findings

*pending*

### Recommendations

*pending*
