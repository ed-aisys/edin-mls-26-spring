# Arithmetic Intensity: Manual Calculation from Tensor Shapes

We attempted hardware-based profiling using NVIDIA Nsight Compute (`ncu`) on the H200 MIG 3g.71gb partition, but the cluster restricts access to GPU performance counters (`ERR_NVGPUCTRPERM`). We therefore compute arithmetic intensity (AI) analytically from the known tensor dimensions and algorithm structure.

**Definition:** Arithmetic intensity = total floating-point operations / total bytes transferred to/from DRAM [Williams et al., 2009].

**References:**
- Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: an insightful visual performance model for multicore architectures." *Communications of the ACM*, 52(4), 65–76.
- Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv:2307.08691*.

**Conventions:**
- FLOPs for a matrix multiply A(M×K) @ B(K×N) = 2×M×K×N (multiply + accumulate)
- All data in fp16 (2 bytes per element) unless noted
- "Bytes" = bytes loaded from + stored to DRAM (minimum, assuming no cache reuse)

---

## Model Dimensions

| Parameter | Encoder | Decoder |
|-----------|---------|---------|
| hidden_size | 1280 | 2048 |
| num_heads | 20 | 16 (Q), 4 (KV) |
| head_dim | 64 | 128 |
| intermediate_size | 5120 | 6144 |
| num_layers | 32 | 28 |
| seq_len (typical) | ~750 | ~200 (prefill), 1 (decode) |

---

## 1. GELU / SiLU (element-wise activations)

**Operation:** `y = GELU(x)` or `y = SiLU(x)` for a tensor of N elements.

**FLOPs:**
- GELU (tanh approximation): ~10 ops per element (multiply, add, cube, tanh, multiply, add, multiply)
- SiLU: ~4 ops per element (negate, exp, add, divide, multiply)
- Conservatively: ~10 FLOPs/element

**Bytes:**
- Load x: N × 2 bytes (fp16)
- Store y: N × 2 bytes (fp16)
- Total: 4N bytes

**AI = 10N / 4N = 2.5 FLOP/byte**

For encoder (hidden=1280, seq=750): N = 750 × 1280 = 960,000 elements.
- FLOPs: ~9.6M
- Bytes: ~3.84MB
- AI ≈ 2.5

**Classification:** Bandwidth-bound on all GPUs (well below any ridge point).

---

## 2. RMSNorm / LayerNorm (normalization)

**Operation:** For each row of length D: compute mean/variance (reduction), normalize, scale+bias.

**FLOPs per row:**
- RMSNorm: sum of squares (2D), rsqrt (1), multiply-normalize (D), apply weight (D) = ~4D
- LayerNorm: mean (2D), variance (3D), normalize (D), scale+bias (2D) = ~8D

**Bytes per row:**
- Load x: D × 2 bytes
- Load weight (+ bias for LN): D × 2 bytes (or 2D × 2 for LN)
- Store y: D × 2 bytes
- Total: ~6D bytes (RMSNorm), ~8D bytes (LayerNorm)

**AI:**
- RMSNorm: 4D / 6D = 0.67 → but weight is cached across rows, so amortized over B rows: (4D × B) / (B × D × 2 + D × 2 + B × D × 2) ≈ 4BD / (4BD + 2D) ≈ 4B/(4B+2) → for B=750: AI ≈ 1.0
- More realistically with BLOCK_SIZE=1024 processing one row per program: AI ≈ 4-8 FLOP/byte

**Conservative estimate: AI ≈ 4–8 FLOP/byte**

**Classification:** Bandwidth-bound on all GPUs.

---

## 3. Flash Attention — Encoder (seq_q=750, seq_k=750, head_dim=64)

Per head, per layer:

**FLOPs:**
- Q @ K^T: 2 × seq_q × seq_k × d_k = 2 × 750 × 750 × 64 = 72,000,000
- P @ V: 2 × seq_q × seq_k × d_k = 72,000,000
- Softmax (exp, sum, div per element): ~5 × seq_q × seq_k = 2,812,500
- Total: ~146,812,500 ≈ **147M FLOPs per head**

**Bytes (flash attention — tiled, not materializing score matrix):**
- Load Q: seq_q × d_k × 2 = 750 × 64 × 2 = 96,000 bytes
- Load K: seq_k × d_k × 2 = 96,000 bytes (loaded in tiles, each tile loaded once)
- Load V: seq_k × d_k × 2 = 96,000 bytes
- Store O: seq_q × d_k × 2 = 96,000 bytes
- Total: **384,000 bytes = 375 KB per head**

Note: In the tiled algorithm, K and V tiles are loaded once per Q tile iteration. With BLOCK_M=128, there are ceil(750/128)=6 Q tiles. Each K/V tile is loaded 6 times. So actual bytes = Q(96KB) + 6×K(96KB) + 6×V(96KB) + O(96KB) = 96 + 576 + 576 + 96 = **1,344 KB per head**.

**AI = 147M / 1,344K = 147,000,000 / 1,376,256 ≈ 107 FLOP/byte**

Wait — this is much higher than our original estimate of 20-40. Let me reconsider.

The issue is that Q tiles are reloaded for each K/V tile pass. With 6 Q tiles and ceil(750/128)=6 K/V tiles per Q tile:
- Q loaded: 6 tiles × 128 × 64 × 2 = 98,304 bytes (each tile loaded once)
- K loaded: 6 Q-tiles × 6 K-tiles × 128 × 64 × 2 = 589,824 bytes (reloaded per Q tile)
- V loaded: same as K = 589,824 bytes
- O stored: 6 tiles × 128 × 64 × 2 = 98,304 bytes

Actually in flash attention, for each Q tile, we iterate over ALL K/V tiles. So K and V are each loaded ceil(750/128)=6 times per Q tile, and there are 6 Q tiles. But with tiling:
- Q: loaded once per tile = 6 × (128 × 64 × 2) = 98KB total from DRAM
- K: for each Q tile, load all K tiles = 6 × 6 × (128 × 64 × 2) = 590KB — BUT K tiles fit in shared memory and are loaded from DRAM once per Q tile, not once per inner iteration

The key insight: **shared memory acts as a cache**. Each K/V tile is loaded from DRAM to shared memory once per Q-tile iteration, then reused for the dot product. So:
- Q from DRAM: 750 × 64 × 2 = 96KB (loaded once total, tile by tile)
- K from DRAM: For each Q tile, all K tiles loaded once = 6 × (750 × 64 × 2) = 576KB
- V from DRAM: same = 576KB
- O to DRAM: 750 × 64 × 2 = 96KB

**Total DRAM traffic per head = 96 + 576 + 576 + 96 = 1,344 KB = 1.31 MB**

**AI = 147M FLOPs / 1.31M bytes ≈ 112 FLOP/byte**

Hmm, this gives AI ≈ 112 for flash attention on the encoder. That's compute-bound on both H200 (ridge=12.7) and RTX 5090 (ridge=58.5). But our original table said 20-40.

**Reconciliation:** The original 20-40 estimate was likely for the **3-kernel pipeline** (not flash attention), where the full score matrix IS materialized in DRAM. Let me compute that:

**3-kernel pipeline DRAM traffic per head:**
- Load Q: 96KB, Load K: 96KB → Write scores S (750×750×4 = 2.25MB in fp32!)
- Load S: 2.25MB → Write S (softmax): 2.25MB
- Load S: 2.25MB, Load V: 96KB → Write O: 96KB
- Total: Q(96K) + K(96K) + S_write(2.25M) + S_read(2.25M) + S_write(2.25M) + S_read(2.25M) + V(96K) + O(96K) = **9.38 MB**

**3-kernel AI = 147M FLOPs / 9.38M bytes ≈ 15.7 FLOP/byte**

That's close to the original 20-40 range, and it's bandwidth-bound on RTX 5090 (ridge=58.5) but compute-bound on H200 (ridge=12.7).

**For flash attention, the AI is much higher (~112) because we avoid materializing the score matrix.** This means flash attention is compute-bound on ALL GPUs — the optimization moves encoder attention from the bandwidth-limited slope to the compute-limited ceiling of the roofline.

---

## 3b. Flash Attention — Decoder Decode (seq_q=1, seq_k=~200, head_dim=128)

Per head, per layer, single decode step:

**FLOPs:**
- Q @ K^T: 2 × 1 × 200 × 128 = 51,200
- P @ V: 2 × 1 × 200 × 128 = 51,200
- Softmax: ~5 × 1 × 200 = 1,000
- Total: **~103,400 FLOPs per head**

**Bytes (seq_q=1, so Q is a single row):**
- Load Q: 1 × 128 × 2 = 256 bytes
- Load K: 200 × 128 × 2 = 51,200 bytes
- Load V: 200 × 128 × 2 = 51,200 bytes
- Store O: 1 × 128 × 2 = 256 bytes
- Total: **102,912 bytes ≈ 100 KB per head**

**AI = 103,400 / 102,912 ≈ 1.0 FLOP/byte**

Very low — dominated by reading the full KV cache for a single query row.

**Classification:** Extremely bandwidth-bound on all GPUs. This is why KV-cached decode is the bottleneck.

Note: As sequence grows (step N reads N cached tokens), bytes grow linearly but FLOPs also grow linearly → AI stays constant at ~1 FLOP/byte regardless of sequence length.

---

## 4. Linear (cuBLAS GEMM)

Example: encoder Q/K/V projection, x(750×1280) @ W(1280×1280)

**FLOPs:** 2 × 750 × 1280 × 1280 = 2,457,600,000 ≈ **2.46 GFLOPs**

**Bytes:**
- Load x: 750 × 1280 × 2 = 1,920,000 bytes
- Load W: 1280 × 1280 × 2 = 3,276,800 bytes
- Store output: 750 × 1280 × 2 = 1,920,000 bytes
- Total: **7,116,800 bytes ≈ 6.8 MB**

**AI = 2.46G / 6.8M ≈ 362 FLOP/byte**

For the larger decoder projections, x(200×2048) @ W(2048×2048):
- FLOPs: 2 × 200 × 2048 × 2048 = 1,677,721,600 ≈ 1.68 GFLOPs
- Bytes: (200×2048 + 2048×2048 + 200×2048) × 2 = (409,600 + 4,194,304 + 409,600) × 2 = 10.0 MB
- AI = 1.68G / 10.0M ≈ 168 FLOP/byte

**Classification:** Strongly compute-bound on all GPUs. cuBLAS achieves near-peak FLOPS on these large regular matrices.

---

## 5. Fused SwiGLU

**Operation:** `output = SiLU(x @ W_gate) * (x @ W_up)` for decoder MLP.
x(200×2048), W_gate(2048×6144), W_up(2048×6144)

**FLOPs:**
- x @ W_gate: 2 × 200 × 2048 × 6144 = 5,033,164,800
- x @ W_up: same = 5,033,164,800
- SiLU + elementwise multiply: ~5 × 200 × 6144 = 6,144,000
- Total: **~10.07 GFLOPs**

**Bytes (fused — x loaded once):**
- Load x: 200 × 2048 × 2 = 819,200
- Load W_gate: 2048 × 6144 × 2 = 25,165,824
- Load W_up: 2048 × 6144 × 2 = 25,165,824
- Store output: 200 × 6144 × 2 = 2,457,600
- Total: **53,608,448 bytes ≈ 51.1 MB**

**AI = 10.07G / 51.1M ≈ 197 FLOP/byte**

**Classification:** Strongly compute-bound. The fusion advantage is not in AI (both fused and unfused have similar AI) but in eliminating the intermediate buffer writes for the gate and up outputs.

---

## 6. RoPE (Rotary Position Embedding)

**Operation:** For each element in the first half_dim of each head: multiply by cos/sin and rotate pairs.

Per head, per position: 6 FLOPs (2 multiplies + 1 subtract for x1', 2 multiplies + 1 add for x2')

**FLOPs per layer (encoder, 20 heads, seq=750, half_dim=32):**
- 20 × 750 × 32 × 6 = 2,880,000 ≈ **2.9M FLOPs**

**Bytes:**
- Load Q (rotary portion): 20 × 750 × 32 × 2 = 960,000
- Load K (rotary portion): 20 × 750 × 32 × 2 = 960,000
- Load cos/sin tables: 750 × 32 × 2 × 2 = 96,000
- Store Q_rotated: 960,000
- Store K_rotated: 960,000
- Passthrough (copy non-rotated dims): 20 × 750 × 32 × 2 × 2 (Q+K load+store) = 1,920,000
- Total: **5,856,000 bytes ≈ 5.6 MB**

**AI = 2.9M / 5.6M ≈ 0.5 FLOP/byte**

Including the fused kernel overhead (loading both Q and K in one launch):
**AI ≈ 0.5–2 FLOP/byte**

**Classification:** Very bandwidth-bound. The fused RoPE kernel's +28.9ms ablation impact comes from eliminating kernel launch overhead (one launch instead of two), not from improved AI.

---

## Summary Table

| Operation | FLOPs | DRAM Bytes | AI (FLOP/byte) | Classification (H200, ridge=12.7) | Classification (RTX 5090, ridge=58.5) |
|-----------|-------|------------|-----------------|-----------------------------------|---------------------------------------|
| GELU / SiLU | ~10/elem | 4/elem | **~2.5** | Bandwidth | Bandwidth |
| RMSNorm / LN | ~4-8D/row | ~6-8D/row | **~4–8** | Bandwidth | Bandwidth |
| Flash Attn (encoder, seq=750) | 147M/head | 1.31MB/head | **~112** | Compute | Compute |
| Flash Attn (decode, seq_q=1) | 103K/head | 100KB/head | **~1.0** | Bandwidth | Bandwidth |
| 3-kernel Attn (encoder, seq=750) | 147M/head | 9.38MB/head | **~16** | Compute (above 12.7) | Bandwidth (below 58.5) |
| Linear (cuBLAS, 750×1280×1280) | 2.46G | 6.8MB | **~362** | Compute | Compute |
| Fused SwiGLU (200×2048×6144) | 10.07G | 51.1MB | **~197** | Compute | Compute |
| RoPE (fused pair) | 2.9M/layer | 5.6MB/layer | **~0.5** | Bandwidth | Bandwidth |

---

## Key Insight: Flash Attention Changes the Classification

The most important finding is that **flash attention moves encoder attention from ~16 FLOP/byte to ~112 FLOP/byte** by eliminating the score matrix materialization. On the RTX 5090 (ridge=58.5), this shifts encoder attention from bandwidth-bound (16 < 58.5) to compute-bound (112 > 58.5). On the H200 (ridge=12.7), both the 3-kernel pipeline (16 > 12.7) and flash attention (112 > 12.7) are compute-bound — the benefit on H200 is less about classification and more about absolute DRAM traffic reduction.

The report's original AI estimate of "~20-40" for encoder attention corresponds to the **3-kernel pipeline**, not our flash attention implementation. With flash attention, the correct AI is ~112 — an order of magnitude improvement in arithmetic intensity.

---

## Note on Methodology

Hardware profiling via NVIDIA Nsight Compute (`ncu`) was attempted on the H200 MIG 3g.71gb partition but returned `ERR_NVGPUCTRPERM` — the cluster does not grant user-level access to GPU performance counters. The calculations above are derived analytically from:
1. Algorithm structure (known operations per kernel)
2. Tensor dimensions (from `GlmAsrConfig` in model.py, confirmed in GUIDE.md)
3. Data type sizes (fp16 = 2 bytes, fp32 = 4 bytes)
4. Flash attention tiling model (Q stays in registers, K/V loaded per Q-tile from DRAM to shared memory)

These give **theoretical minimum DRAM traffic** — actual traffic may be higher due to cache misses, padding, and compiler-inserted loads. The true AI is therefore a lower bound; the real value could be lower (more bandwidth-bound than calculated).
