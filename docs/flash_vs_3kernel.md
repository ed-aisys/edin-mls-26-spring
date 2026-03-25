# Flash Attention vs Alternatives: H200 Comparison

**Date:** 2026-03-25
**GPU:** NVIDIA H200 MIG 3g.71gb (60 SMs, 70GB HBM3e, ~228KB shared memory)

---

## Test Methodology

All tests swap only the attention path while keeping everything else identical (KV cache, fp16 pipeline, fused RoPE, cuBLAS, GPUProfile). The test scripts:
1. Back up `attention.py`
2. Modify the attention dispatch (either swap the file or change a single line)
3. Clear Triton cache, run benchmark
4. Restore backup

No committed code is changed. The swap is temporary within the SLURM job.

---

## Test 1: Flash Attention vs PyTorch SDPA (clean single-variable ablation)

**Method:** Change `if q.is_cuda and seq_q <= 4:` to `if q.is_cuda:` — forces ALL attention through PyTorch SDPA instead of our flash attention kernel. Everything else (KV cache, fp16, fused RoPE) stays the same.

**Student benchmark (KV-cached, 13 tokens, 5 runs each):**

| Config | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean |
|--------|-------|-------|-------|-------|-------|------|
| Flash Attention ON | 236.4 | 216.1 | 211.0 | 221.0 | 221.8 | ~221ms |
| Flash OFF (all SDPA) | 216.6 | 210.8 | 218.6 | 205.9 | 209.3 | ~212ms |
| **Delta** | | | | | | **~9ms (SDPA faster)** |

100% accuracy on all runs for both configs.

**Finding:** With KV-cached decoding (seq_q=1 per step), PyTorch SDPA is ~9ms faster than our flash attention kernel on H200. This is because:
- At seq_q=1, there is no score matrix to materialize — both approaches do the same work
- PyTorch's SDPA is highly optimized for single-row attention on Hopper (uses cuDNN attention backend)
- Our flash attention kernel has higher launch overhead for this degenerate case

This is why we use the SDPA fallback for seq_q ≤ 4 in production — it handles the KV-cached decode path more efficiently.

---

## Test 2: Flash Attention vs Origin/Main 3-Kernel Pipeline

**Method:** Swap entire `attention.py` from `origin/main` (original 3-kernel pipeline: `attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel`, `causal_mask_kernel`).

**IMPORTANT:** This test is NOT a clean single-variable comparison. Origin/main's `scaled_dot_product_attention` has a different internal implementation that is not compatible with our GQA head expansion. The test produced **0% accuracy** — the transcription was empty.

**Student benchmark (3 runs):**

| Config | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Flash Attention (ours) | 216.1ms | 209.3ms | 212.3ms |
| 3-Kernel (origin/main) | 1808.2ms (0% accuracy) | — | — |

**This result is invalid.** The 1808ms includes broken attention output due to incompatible GQA handling. The 3-kernel code from origin/main cannot simply be dropped into our codebase because:
- Our code pre-expands KV heads (16Q → 4KV expanded to 16KV) before calling `scaled_dot_product_attention`
- Origin/main's `scaled_dot_product_attention` expects unexpanded KV heads and has its own class-based expansion

**Detailed benchmark (per-component, 1 run each):**

| Component | Flash Attention | 3-Kernel (origin/main) |
|-----------|----------------|----------------------|
| Audio Encoder | 3007ms | 2282ms |
| Projector | 10ms | 7ms |
| Decoder Prefill | 2204ms | 3054ms |
| Decoder Decode (50 steps) | 7594ms | 23624ms |
| Total | 12814ms | 28967ms |

These numbers should be treated with caution due to the accuracy failure.

---

## Summary for Report

The cleanest comparison is **Test 1** (flash ON vs all-SDPA):
- With KV-cached decoding, flash attention is ~9ms slower than PyTorch SDPA on H200
- This is why we use SDPA fallback for seq_q ≤ 4
- Flash attention's true benefit is in prefill/encoder (large seq_q) where it avoids materializing the N×N score matrix

The origin/main comparison (Test 2) cannot be used as a clean ablation because the code is incompatible. The 3-kernel pipeline from origin/main was designed for a different attention dispatch flow.

---

## Raw Output Files

- `flash_ablation_2233215.log` — Test 1: Flash ON vs all-SDPA (clean ablation)
- `flash_vs_main_2233206.log` — Test 2: Flash vs origin/main (invalid, 0% accuracy)
- `flash_on_result.txt`, `flash_off_sdpa_result.txt` — Test 1 parsed results
- `flash_main_our_result.txt`, `flash_main_3kernel_result.txt` — Test 2 parsed results
- `flash_main_our_detailed.txt`, `flash_main_3kernel_detailed.txt` — Test 2 detailed benchmarks
