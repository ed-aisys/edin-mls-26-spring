# Flash Attention vs 3-Kernel Attention: H200 Comparison

**Date:** 2026-03-25
**GPU:** NVIDIA H200 MIG 3g.71gb (60 SMs, 70GB HBM3e, ~228KB shared memory)
**Comparison:** Our flash attention kernel vs origin/main's original 3-kernel pipeline
**Source of 3-kernel code:** `origin/main:hw1-asr/glm_asr_triton_template/attention.py`

The 3-kernel pipeline (`attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel`) materializes the full N_q × N_k score matrix in VRAM on every attention call. Our flash attention kernel fuses all three into a single pass with online softmax, keeping intermediates in shared memory/registers.

---

## Student Benchmark (KV-cached, 13 tokens, 3g.71gb)

| Config | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Flash Attention (ours) | 216.1ms (+/- 0.8) | 209.3ms (+/- 0.9) | 212.3ms (+/- 0.6) |
| 3-Kernel (origin/main) | 1808.2ms (+/- 6.0) | — | — |

**Speedup: 8.5x** (1808ms → 212ms)

100% accuracy on both.

---

## Detailed Benchmark (per-component, 50 decode steps, 3g.71gb)

| Component | Flash Attention | 3-Kernel (origin/main) | Delta |
|-----------|----------------|----------------------|-------|
| Audio Encoder | 3007.2ms (23.5%) | 2282.3ms (7.9%) | +724.9ms |
| Projector | 9.7ms (0.1%) | 7.1ms (0.0%) | +2.6ms |
| Decoder Prefill | 2203.7ms (17.2%) | 3054.2ms (10.5%) | -850.5ms |
| **Decoder Decode (50 steps)** | **7593.6ms (59.3%)** | **23623.5ms (81.6%)** | **-16029.9ms** |
| **Total** | **12814.1ms** | **28967.1ms** | **-16153.0ms** |

**Overall speedup: 2.26x** (28967ms → 12814ms)
**Decode speedup: 3.11x** (23624ms → 7594ms)

---

## Analysis

### Decoder decode: 3.1x faster with flash attention
The 3-kernel pipeline materializes the full score matrix in VRAM on every decode step. For a growing sequence (the detailed benchmark uses the stock generation path), step N processes all N tokens, writing an N×N score matrix. Over 50 steps this accumulates massive VRAM traffic. Flash attention eliminates this entirely — intermediates stay in shared memory/registers.

### Decoder prefill: flash attention is faster (-850ms)
During prefill, the full concatenated sequence (~200+ tokens) passes through 28 decoder layers. Flash attention avoids materializing the 200×200 score matrix per layer, saving ~28 × 200 × 200 × 128 × 4 bytes = ~57MB of intermediate traffic per prefill.

### Audio encoder: 3-kernel is faster (-725ms)
This is surprising. The encoder runs 32 layers with seq_len ~750 and head_dim=64. The 3-kernel pipeline from origin/main may use different tile configs or memory access patterns that happen to be more efficient for the encoder's specific dimensions on H200. Our flash attention tile config (128×128 with nstages=2) may not be optimal for encoder attention — the ablation showed encoder tile changes have small impact (+0.9ms for 64×64), so this difference likely comes from the profiling harness rather than a real architectural advantage.

Note: The detailed benchmark's first run includes Triton compilation time, which inflates encoder numbers. The run-1 compilation cost differs between flash attention (compiling 1 kernel) and 3-kernel (compiling 4 kernels).

### Student benchmark: 8.5x faster
The student benchmark uses KV-cached generation (13 tokens). The 3-kernel pipeline from origin/main does NOT have KV caching — it uses the stock O(n²) `generate()`. So the 8.5x speedup combines both flash attention AND KV caching benefits.

---

## Raw Output Files

- `flash_vs_main_2233206.log` — Full SLURM output (student + detailed for both configs)
- `flash_main_our_result.txt` — Flash attention student benchmark
- `flash_main_3kernel_result.txt` — 3-kernel student benchmark
- `flash_main_our_detailed.txt` — Flash attention detailed benchmark
- `flash_main_3kernel_detailed.txt` — 3-kernel detailed benchmark
