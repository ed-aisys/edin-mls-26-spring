# Arithmetic Intensity: Calculations (encoder N = 1500)

**Definition:** Arithmetic intensity (AI) = total floating-point operations / total bytes transferred to/from DRAM [Williams et al., 2009].

## Conventions

- **GEMM** with *A* (*M*×*K*) and *B* (*K*×*N*): **FLOPs = 2MNK**.
- **fp16**: **2 bytes/element** for loads/stores unless a tensor is explicitly fp32 (e.g. baseline score matrix).
- Elementwise ops: **~1 FLOP** per elementary op where not spelled out; **tanh ~4 FLOPs** (GELU).
- Flash tiling model: same as the manual doc — full Q and O footprints once; **K and V each streamed once per Q-tile** (`ceil(N / BLOCK_M)` passes, **BLOCK_M = 128**)

---

## Ridge points

**H200 MIG 3g.71gb (60 SMs)**

- Peak FP32: (60/132) × 67 ≈ 30.5 TFLOP/s  
- Peak HBM: ≈ 2.41 TB/s (from profile metadata)  
- **Ridge** = 30.5×10^12 / 2.41×10^12 ≈ 12.7 FLOP/byte

**RTX 5090 (reference)**

- **Ridge** ≈ 105 / 1.79 ≈ 58.5 FLOP/byte

This document uses **encoder self-attention sequence length N = 1500**, which matches `audio_encoder` output shape for our benchmark clip (`test_audio.wav`: mel features **(1, 128, 3000)** after conv subsampling → **seq = 1500**). 

**Script:** Run `[compute_ai_encoder_n.py](compute_ai_encoder_n.py)` to recompute for any N:

```bash
python docs/compute_ai_encoder_n.py --n 1500 --three-kernel
```

---

## Model Dimensions


| Parameter         | Encoder | Decoder                 |
| ----------------- | ------- | ----------------------- |
| hidden_size       | 1280    | 2048                    |
| num_heads         | 20      | 16 (Q), 4 (KV)          |
| head_dim          | 64      | 128                     |
| intermediate_size | 5120    | 6144                    |
| num_layers        | 32      | 28                      |
| seq_len           | 1500    | ~200 prefill / 1 decode |


---

## 1. GELU / SiLU (element-wise)

**AI = 2.5** FLOP/byte (per-element ratio).

**FLOPs:**

- GELU (tanh approximation): ~10 ops per element (multiply, add, cube, tanh, multiply, add, multiply)
- SiLU: ~4 ops per element (negate, exp, add, divide, multiply)
- Conservatively: ~10 FLOPs/element

**Bytes:**

- Load x: N × 2 bytes (fp16)
- Store y: N × 2 bytes (fp16)
- Total: 4N byte

**Encoder example with N = 1500, hidden = 1280:**

- Elements: 1500 × 1280 = **1,920,000**  
- FLOPs: ~10 × 1.92M = **19.2M**  
- Bytes: 4 × 1.92M = **7.68M**  
- AI ≈ **2.5**

---

## 2. RMSNorm / LayerNorm

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

## 3a. Flash Attention — Encoder (seq_q = seq_k = **1500**, head_dim = 64)

Per head, per layer:

**FLOPs**

- Q @ K^T: 2 × 1500 × 1500 × 64 = **288,000,000**  
- P @ V: 2 × 1500 × 1500 × 64 = **288,000,000**  
- Softmax ~5 ops per score: 5 × 1500 × 1500 = **11,250,000**  
- **Total ≈ 587,250,000 FLOPs per head** (~4× the N = 750 case, since leading term is **N²**)

**DRAM bytes (tiled flash, BLOCK_M = 128)**

- num_Q_tiles = ceil(1500 / 128) = **12**  
- Q + O footprint: 1500 × 64 × 2 × 2 = **384,000** bytes  
- K streamed once per Q-tile: 12 × (1500 × 64 × 2) = **2,304,000** bytes  
- V streamed once per Q-tile: **2,304,000** bytes  
- **Total ≈ 4,992,000 bytes ≈ 4.99 MB per head**

**AI (flash) = 587,250,000 / 4,992,000 ≈ 118 FLOP/byte** 

**Classification:** Still **compute-bound** on H200 (ridge ~12.7) and on RTX 5090 (ridge ~58.5).

---

## 3b. 3-kernel pipeline (fp32 scores), N = 1500

**3-kernel pipeline DRAM traffic per head:**

- Total: Q+ K + S_write + S_read + S_write + S_read + V + O
- **DRAM ≈ 36,768,000 bytes per head** (~35 MB)  
- **AI ≈ 587.25M / 36.77M ≈ 16 FLOP/byte** (still ~**16**, like N = 750)

---

## 3c. Flash Attention — Decoder Decode (seq_q=1, seq_k=~200, head_dim=128)

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

## 4. Linear (cuBLAS) — encoder Q projection with N = 1500

**x (1500×1280) @ W (1280×1280)**

- FLOPs: 2 × 1500 × 1280 × 1280 = **4,915,200,000**  
- Bytes: (1500×1280 + 1280×1280 + 1500×1280) × 2 = **10,956,800**  
- **AI ≈ 449 FLOP/byte**

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

## 6. RoPE — encoder layer, seq = **1500**

**Operation:** For each element in the first half_dim of each head: multiply by cos/sin and rotate pairs.

Per head, per position: 6 FLOPs (2 multiplies + 1 subtract for x1', 2 multiplies + 1 add for x2')

**FLOPs  (encoder, 20 heads, seq=1500, half_dim=32):** 20 × 1500 × 32 × 6 = **5,760,000** per layer

**Bytes:**

- Load Q (rotary portion): 20 × 1500 × 32 × 2 = 1,920,000
- Load K (rotary portion): 20 × 1500 × 32 × 2 = 1,920,000
- Load cos/sin tables: 1500 × 32 × 2 × 2 = 192,000
- Store Q_rotated (rotary dims): 20 × 1500 × 32 × 2 = **1,920,000**
- Store K_rotated (rotary dims): 20 × 1500 × 32 × 2 = **1,920,000**
- Passthrough (non-rotated 32 dims per head; load+store for Q and K): 4 × (20 × 1500 × 32 × 2) = **3,840,000**
- Total: **11,712,000**  **≈ 11.7 MB**

**AI ≈ 5,760,000 / 11,712,000 ≈ 0.49 FLOP/byte** — **~0.5**, bandwidth-bound.

---

## Summary Table (N = 1500 encoder)


| Operation                 | AI (FLOP/byte) |
| ------------------------- | -------------- |
| GELU                      | ~2.5           |
| RMSNorm/LayerNorm         | ~4-8           |
| Flash attn encoder        | ~118           |
| Flash attn decoder        | ~1             |
| 3-kernel encoder (fp32 S) | ~16            |
| Linear enc Q proj         | ~449           |
| Fused SwiGLU              | ~197           |
| RoPE encoder layer        | ~0.5           |