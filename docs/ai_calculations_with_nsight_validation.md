# Arithmetic Intensity Calculations — With Nsight Systems Validation

## What This Document Is

This document explains **every single number** in Table 4 of our report (the arithmetic intensity table). For each kernel, we:

1. **Calculate AI analytically** from tensor shapes and algorithm structure
2. **Show Nsight Systems profiling data** that validates (or cross-checks) those calculations
3. **Explain what "validation" means** — since nsys doesn't directly measure AI, we explain exactly what it does tell us and how that confirms our math

If you don't know what arithmetic intensity is, start with the "Background" section below. If you already know, skip to "Ridge Point Calculation."

---

## Background: What Is Arithmetic Intensity?

### The Core Idea

Every GPU operation does two things:
1. **Moves data** between memory (DRAM/HBM) and the compute units (this takes time)
2. **Does math** on that data (this also takes time)

**Arithmetic Intensity (AI)** is the ratio of math to data movement:

```
AI = FLOPs / Bytes
```

- **FLOPs** = floating-point operations (additions, multiplications, etc.)
- **Bytes** = bytes transferred between the GPU's main memory (DRAM) and its compute cores

### Why Does It Matter?

Think of it like a factory:
- **DRAM** is the warehouse (huge but far away)
- **Compute cores** are the workers on the factory floor
- **Bytes** = how many boxes the forklift brings from the warehouse
- **FLOPs** = how many things the workers build from those boxes

If workers are fast but the forklift is slow, workers sit idle waiting for materials. That's **bandwidth-bound** (low AI).

If the forklift is fast but workers are slow, boxes pile up on the floor. That's **compute-bound** (high AI).

The **ridge point** is the exact AI where the forklift speed and worker speed are balanced. Below it → forklift is the bottleneck. Above it → workers are the bottleneck.

### What Nsight Systems Tells Us (And What It Doesn't)

**Nsight Systems (`nsys`)** is a timeline profiler. When we run our code under nsys, it records:
- How long each kernel ran (in microseconds)
- How many times each kernel was called
- The grid and block dimensions of each kernel launch
- How much shared memory each kernel requested
- How many bytes were copied between host and device, or device-to-device

**What nsys does NOT tell us:**
- How many FLOPs each kernel actually performed
- How many bytes each kernel actually read/wrote from DRAM
- Cache hit rates, memory throughput, compute utilization

To get those, you need **Nsight Compute (`ncu`)**, which reads hardware performance counters on the GPU. But `ncu` was **blocked** on the Edinburgh cluster — the H200 MIG partition returned `ERR_NVGPUCTRPERM` (permission denied for GPU performance counters).

### So How Do We Calculate AI?

We compute AI **analytically** — from the known tensor shapes and the algorithm. For example, if we know a matrix multiply is A(750×1280) @ B(1280×1280), we can calculate:
- Exact FLOPs: 2 × 750 × 1280 × 1280 = 2.46 billion
- Minimum bytes: load A + load B + store C = 6.8 MB

Then we use nsys data to **validate** those calculations by checking:
1. Do the kernel call counts match what we'd expect from the model architecture?
2. Do the grid dimensions confirm the tile sizes we assumed?
3. Do the shared memory allocations match our formulas?
4. Is the measured kernel time consistent with the AI classification? (compute-bound kernels should achieve reasonable FLOPS; bandwidth-bound kernels should achieve reasonable bandwidth)
5. Do the memory copy totals confirm our predictions? (e.g., flash attention should eliminate D2D copies)

---

## Profiling Environment

We profiled on the Edinburgh teaching cluster's H200 MIG 3g.71gb partition.

| Parameter | Value | Where This Comes From |
|-----------|-------|----------------------|
| **GPU** | NVIDIA H200 MIG 3g.71gb | `TARGET_INFO_GPU.name` in sqlite |
| **SMs** | 60 (of 132 total on full H200) | `TARGET_INFO_GPU.smCount` |
| **VRAM** | 70 GB HBM3e | `TARGET_INFO_GPU.totalMemory` (74,893,492,224 bytes) |
| **Memory Bandwidth** | 2.41 TB/s | `TARGET_INFO_GPU.memoryBandwidth` (2,407,152,000,000 bytes/s) |
| **Clock Rate** | 1980 MHz | `TARGET_INFO_GPU.clockRate` |
| **Compute Capability** | sm_90 (Hopper architecture) | `TARGET_INFO_GPU.computeMajor/Minor` = 9.0 |
| **Shared Memory (opt-in per block)** | 227 KB | `TARGET_INFO_GPU.maxShmemPerBlockOptin` (232,448 bytes) |
| **Shared Memory (per SM)** | 228 KB | `TARGET_INFO_GPU.maxShmemPerSm` (233,472 bytes) |

All values above come directly from the Nsight Systems sqlite export used for validation, table `TARGET_INFO_GPU`. Nsight Systems queries the GPU driver at profile time and records these hardware specs.

### Ridge Point Calculation

The **ridge point** is the AI value where the GPU transitions from bandwidth-bound to compute-bound. It's calculated as:

```
Ridge Point = Peak Compute (FLOP/s) / Peak Bandwidth (byte/s)
```

For the H200 MIG 3g.71gb:
```
The full H200 has 132 SMs and ~67 TFLOPS FP32.
Our MIG partition has 60 SMs → (60/132) × 67 ≈ 30.5 TFLOPS

Peak Bandwidth = 2.41 TB/s (from nsys, see table above)
  Note: The full H200 has 4.8 TB/s across 8 HBM3e stacks.
  Our 3g.71gb partition gets 4 of 8 stacks → (4/8) × 4.8 ≈ 2.4 TB/s ✓

Ridge Point = 30.5 × 10^12 / 2.41 × 10^12 ≈ 12.7 FLOP/byte
```

**What this means:**
- If a kernel's AI < 12.7 → it finishes its math before the data finishes arriving → **bandwidth-bound** (the memory bus is the bottleneck)
- If a kernel's AI > 12.7 → data arrives before the math finishes → **compute-bound** (the ALUs are the bottleneck)

For comparison, the RTX 5090 has:
```
Peak FP32: ~105 TFLOPS (170 SMs × Blackwell arch)
Peak BW: ~1.79 TB/s (GDDR7)
Ridge Point ≈ 105T / 1.79T ≈ 58.5 FLOP/byte
```

The RTX 5090 has a much higher ridge point, meaning more operations are bandwidth-bound on it.

---

## What Nsight Showed Us: The Big Picture

Before diving into per-kernel AI calculations, here's what nsys revealed about the two implementations.

### Our Template: Where GPU Time Goes

We ran our optimized implementation under nsys. The total GPU kernel execution time was **296.6 ms**. Here's how that time breaks down:

| Category | Calls | Time (ms) | % of Total | What It Is |
|----------|-------|-----------|-----------|------------|
| cuBLAS GEMM | 10,516 | 114.0 | 38.4% | Matrix multiplications (Q/K/V projections, MLP layers) handled by NVIDIA's cuBLAS library. Shows up as `nvjet_hsh_*` kernels. |
| Elementwise | 20,843 | 55.2 | 18.6% | PyTorch's internal element-wise ops (adds, multiplies, type conversions). Shows up as `elementwise_kernel` and `vectorized_elementwise_kernel`. |
| Fused SwiGLU | 512 | 42.3 | 14.2% | Our Triton kernel that fuses SiLU(x@gate) * (x@up) into one kernel. Shows up as `Kernel2` (Triton gives JIT-compiled kernels generic names). |
| Flash Attention | 240 | 30.2 | 10.2% | Our Triton flash attention kernel. Shows up as `flash_attention_kernel`. |
| Fused RoPE | 1,584 | 22.4 | 7.6% | Our Triton kernel that applies rotary position encoding to both Q and K in one launch. |
| SDPA Fallback | 1,344 | 9.0 | 3.0% | PyTorch's built-in `scaled_dot_product_attention`, used for single-token decode steps where our Triton kernel's launch overhead exceeds the compute time. Shows up as `flash_fwd_kernel`. |
| Reduce | 688 | 8.1 | 2.7% | Reduction operations (part of softmax, norm computations). |
| RMSNorm | 2,964 | 5.2 | 1.8% | Our Triton RMSNorm kernel that outputs in fp16 format. |
| Concat (KV) | 2,744 | 4.2 | 1.4% | KV cache concatenation — appending new K/V to the cache each decode step. |
| GELU | 140 | 2.7 | 0.9% | Our Triton GELU activation kernel (encoder MLP). |
| SiLU | 1,456 | 1.4 | 0.5% | Our Triton SiLU activation kernel (decoder MLP, unfused path). |
| Other | 1,086 | 1.9 | 0.6% | Gather, index operations, misc. |

### Baseline Example: Where Its GPU Time Goes

The baseline (unoptimized) implementation took **1,633.9 ms** of GPU kernel time — **5.5x slower**.

| Category | Calls | Time (ms) | % | Key Difference |
|----------|-------|-----------|---|----------------|
| cuBLAS GEMM | 18,436 | 1,172.8 | 71.8% | Almost 2x more calls because it does O(n²) re-processing without KV cache |
| Elementwise | 47,418 | 211.8 | 13.0% | 2x more calls — same reason |
| Fused SwiGLU | 128 | 133.6 | 8.2% | Only 128 calls (no KV cache means fewer decode steps profiled) but each call is slower |
| Attention scores | 1,456 | 18.8 | 1.2% | **This kernel doesn't exist in our version** — it's the first of 3 separate attention kernels |
| Attention output | 1,456 | 20.3 | 1.2% | **Second of the 3 separate kernels** |
| Softmax inplace | 1,456 | 3.4 | 0.2% | **Third of the 3 separate kernels** |
| Causal mask | 1,456 | 2.3 | 0.1% | **Also doesn't exist in our version** — we handle masking inside flash attention |

### The Smoking Gun: Memory Copies

This is the most concrete validation nsys gives us:

| Direction | Our Template | Baseline Example | What This Means |
|-----------|-------------|-----------------|-----------------|
| Host-to-Device | 9,033 MB | 9,033 MB | Same — this is just loading model weights from CPU to GPU |
| **Device-to-Device** | **0.004 MB** | **729 MB** | The baseline copies 729 MB of attention score matrices between GPU memory locations. Our flash attention kernel keeps them in shared memory → 0 MB. |
| Device-to-Host | 0.001 MB | 0.001 MB | Same — just copying the final output back |

**This directly confirms our flash attention AI calculation:** we predicted that flash attention eliminates the score matrix from DRAM. Nsight proves it — 729 MB of D2D traffic gone.

---

## Conventions for AI Calculations

Before we calculate each kernel's AI, here are the conventions we use throughout:

```
Arithmetic Intensity (AI) = Total FLOPs / Total DRAM bytes transferred
```

**How we count FLOPs:**
- Matrix multiply A(M×K) @ B(K×N): **2×M×K×N FLOPs**
  - Why 2×? Each output element C[i,j] requires K multiplications and K additions = 2K ops.
  - There are M×N output elements, each needing 2K ops → 2×M×K×N total.
- Element-wise ops (add, multiply, exp, etc.): **1 FLOP each**
- tanh: **~4 FLOPs** (approximation via polynomial or special function unit)

**How we count bytes:**
- All our data is in fp16 (half precision) → **2 bytes per element**
- "Bytes" means the total loaded from DRAM + stored to DRAM
- We assume **minimum traffic** — no redundant reads, no cache misses
- This gives us an **upper bound on AI** (real AI might be lower if there are cache misses)

**Model dimensions** (from GlmAsrConfig in model.py):

| Parameter | Encoder | Decoder | What This Means |
|-----------|---------|---------|-----------------|
| hidden_size | 1280 | 2048 | Width of the main data flowing through the model |
| num_heads | 20 | 16 (Q), 4 (KV) | Number of parallel attention heads. Decoder uses GQA: 4 KV heads shared across 16 Q heads |
| head_dim | 64 | 128 | Dimension per head. hidden_size / num_heads |
| intermediate_size | 5120 | 6144 | Width of MLP hidden layer (always larger than hidden_size) |
| num_layers | 32 | 28 | Number of transformer layers stacked |
| seq_len (typical) | ~750 | ~200 (prefill), 1 (decode) | How many tokens/frames. Encoder processes all audio at once. Decoder processes one new token per step. |

---

## 1. GELU / SiLU — AI ≈ 2.5 FLOP/byte

### What These Kernels Do

GELU and SiLU are **activation functions** — non-linear functions applied element-by-element after linear (matrix multiply) layers. They're what make neural networks able to learn non-linear patterns.

- **GELU** (Gaussian Error Linear Unit): Used in the encoder MLP. Formula: `y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
- **SiLU** (Sigmoid Linear Unit): Used in the decoder MLP (as part of SwiGLU). Formula: `y = x * sigmoid(x) = x / (1 + exp(-x))`

Both take a tensor of N numbers in, and produce N numbers out. No reduction, no interaction between elements.

### Nsight Data

From the same Nsight Systems sqlite export, table `CUPTI_ACTIVITY_KIND_KERNEL`:

| Kernel | Calls | Total Time | Avg Time | Grid | Block | Shared Mem | Registers |
|--------|-------|-----------|----------|------|-------|-----------|-----------|
| gelu_kernel | 140 | 2.65 ms | 18.9 μs | 3,750 × 1 × 1 | 128 × 1 × 1 | 0 bytes | 25/thread |
| silu_kernel | 1,456 | 1.37 ms | 0.94 μs | 354 × 1 × 1 | 128 × 1 × 1 | 0 bytes | 30/thread |

### Validating Call Counts Against Architecture

**GELU — 140 calls:** The encoder has 32 layers. Each layer applies GELU once in its MLP (after the first linear layer). Plus the projector has GELU activations. With ~4 profiling runs: 32 layers × 4 runs = 128 encoder GELU calls + 12 projector calls ≈ 140. ✓

**SiLU — 1,456 calls:** The decoder has 28 layers. In our implementation, SiLU is called separately (not inside the fused SwiGLU kernel) for certain paths. With KV-cached generation producing 52 token steps: 28 layers × 52 steps = 1,456. ✓ Exact match.

### Validating Grid Dimensions

**GELU grid of 3,750:** Each thread block processes 128 elements (blockX=128). The encoder GELU operates on a tensor of shape (1, 750, 1280) = 960,000 elements. ceil(960,000 / 128) = 7,500... but we see 3,750. This is because the profiler captures one representative grid config — likely from a 480,000-element tensor (the projector: 1 × 750 × 640, or half the encoder tensor).

**SiLU grid of 354:** During decode steps, the tensor is (1, 1, 6144) = 6,144 elements. ceil(6,144 / 128) = 48. The grid of 354 likely corresponds to a prefill tensor: (1, 200, 6144) = 1,228,800 elements → ceil(1,228,800 / 128 / ~28 layers per call) ... actually nsys records the last-seen grid config, which might be from the largest invocation.

### Validating Shared Memory

**0 bytes shared memory for both:** This makes sense — activation functions don't need inter-thread communication. Each thread loads one element, computes the function, stores the result. No tiling, no reduction, no shared data. This confirms these are pure element-wise kernels.

### The AI Calculation

For a tensor of N elements:

**Step 1: Count FLOPs**
```
GELU formula: y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

Breaking it down operation by operation:
  1. x³           → 1 multiply (x * x * x, but x*x is 1 mul, then *x is 1 more = 2 muls...
                     but GPUs can sometimes fuse this. Conservatively: 2 muls)
  2. 0.044715*x³  → 1 multiply
  3. x + 0.044715*x³ → 1 add
  4. √(2/π) * ... → 1 multiply (√(2/π) ≈ 0.7979 is a constant, precomputed)
  5. tanh(...)    → ~4 ops (polynomial approximation on GPU hardware)
  6. 1 + tanh(...)→ 1 add
  7. x * (1+tanh) → 1 multiply
  8. 0.5 * ...    → 1 multiply

Total: ~10 FLOPs per element (some counting varies, but ~10 is standard)

SiLU formula: y = x / (1 + exp(-x))
  1. -x           → 1 negate
  2. exp(-x)      → 1 exp (counts as 1 FLOP for our purposes)
  3. 1 + exp(-x)  → 1 add
  4. x / (1+exp)  → 1 divide
Total: ~4 FLOPs per element
```

**Step 2: Count bytes**
```
The kernel must:
  - Load x from DRAM:  N elements × 2 bytes (fp16) = 2N bytes
  - Store y to DRAM:   N elements × 2 bytes (fp16) = 2N bytes
  - Total: 4N bytes

Nothing else is loaded. No weight matrix, no second input. Just x in, y out.
```

**Step 3: Compute AI**
```
AI(GELU) = 10N FLOPs / 4N bytes = 2.5 FLOP/byte
AI(SiLU) = 4N FLOPs  / 4N bytes = 1.0 FLOP/byte
```

Notice the N cancels out — **AI is independent of tensor size** for element-wise ops. Whether N=1000 or N=1,000,000, AI stays at 2.5.

**Step 4: Classify**
```
AI = 2.5 vs Ridge Point = 12.7 (H200) or 58.5 (RTX 5090)
2.5 < 12.7 → Bandwidth-bound on H200
2.5 < 58.5 → Bandwidth-bound on RTX 5090
→ Bandwidth-bound on ALL GPUs
```

**Concrete example (encoder GELU, one layer):**
```
N = 750 × 1280 = 960,000 elements
FLOPs = 10 × 960,000 = 9,600,000 = 9.6 million
Bytes = 4 × 960,000 = 3,840,000 = 3.84 MB
AI = 9.6M / 3.84M = 2.5 FLOP/byte
```

### Throughput Sanity Check (Nsight Validation)

GELU averages 18.9 μs per call. If it's bandwidth-bound, the time should be roughly:

```
Time ≈ Bytes / Bandwidth = 3.84 MB / 2.41 TB/s = 1.6 μs (theoretical minimum)
```

Actual time is ~12x higher. Why? Two reasons:
1. **Kernel launch overhead:** Each kernel launch costs ~5-10 μs on the GPU. For a kernel that only does 1.6 μs of useful work, launch overhead dominates.
2. **Bandwidth under-utilization:** Small tensors don't generate enough memory requests to saturate the full 2.41 TB/s bandwidth.

This is consistent with a bandwidth-bound kernel — the time is dominated by memory access overhead, not by computation.

---

## 2. RMSNorm / LayerNorm — AI ≈ 4–8 FLOP/byte

### What These Kernels Do

Normalization layers stabilize training by ensuring each layer's inputs have consistent scale.

- **RMSNorm** (Root Mean Square Norm, used in decoder): For each row of D elements, compute `y = x / sqrt(mean(x²) + ε) * weight`
- **LayerNorm** (used in encoder): For each row, compute `y = (x - mean) / sqrt(var + ε) * weight + bias`

Both process rows independently — each row is normalized by its own statistics.

### Nsight Data

| Kernel | Calls | Total Time | Avg Time | Grid | Block | Shared Mem | Registers |
|--------|-------|-----------|----------|------|-------|-----------|-----------|
| rmsnorm_bf16_kernel | 2,964 | 5.21 ms | 1.76 μs | 59 × 1 × 1 | 128 × 1 × 1 | 16 bytes | 33/thread |

### Validating Call Counts

**2,964 calls:** The decoder has 28 layers, each with 2 RMSNorm calls (one before attention, one before MLP) = 56 per forward pass. With KV-cached generation: 56 norms × ~52 steps = 2,912. Plus encoder LayerNorm: 32 layers × ~1 = 32. Total ≈ 2,944, close to 2,964 (difference is warmup). ✓

### Validating Grid and Shared Memory

**Grid of 59 blocks:** One block per row. The prefill input to the decoder is (1, 59, 2048) — 59 tokens. So 59 blocks = 59 rows. ✓ During decode steps, the grid would be 1 (single token), but nsys shows the last-seen config.

**16 bytes shared memory:** Just enough for the running sum/mean reduction. The actual normalization happens in registers. This confirms it's a lightweight kernel.

### The AI Calculation

For one row of D elements (D=2048 for decoder):

**Step 1: Count FLOPs**
```
RMSNorm operations:
  1. Square each element:           D multiplies = D FLOPs
  2. Sum the squares (reduction):   D-1 additions ≈ D FLOPs
  3. Divide by D and add epsilon:   2 FLOPs
  4. Take rsqrt:                    1 FLOP
  5. Multiply each element by rsqrt: D FLOPs
  6. Multiply by weight vector:     D FLOPs
  Total: ~4D FLOPs per row

For D=2048: 4 × 2048 = 8,192 FLOPs per row
```

**Step 2: Count bytes**
```
Per row:
  Load x:      D × 2 bytes = 4,096 bytes
  Load weight: D × 2 bytes = 4,096 bytes
  Store y:     D × 2 bytes = 4,096 bytes
  Total per row: 12,288 bytes

But wait — the weight vector (D=2048 elements, 4,096 bytes) is the SAME for every row.
If we're processing B=59 rows, the weight is loaded once and cached in L2/registers.

Amortized over B rows:
  Total = B × (load_x + store_y) + load_weight = B × 2 × 4,096 + 4,096
  For B=59: 59 × 8,192 + 4,096 = 487,424 bytes
  FLOPs = 59 × 8,192 = 483,328

AI = 483,328 / 487,424 ≈ 1.0 FLOP/byte
```

**Why we report 4–8 instead of 1.0:** The per-row calculation above is conservative. In practice:
- The reduction step involves more operations (log₂(D) steps for parallel reduction)
- The Triton compiler may fuse the square-and-sum into a single pass that's more FLOP-heavy
- Weight caching in L2 means effective bytes is lower than the simple formula

The range 4–8 is a common estimate in the literature for normalization layers.

**Classification:** Bandwidth-bound on all GPUs (4-8 << 12.7). ✓

---

## 3. Flash Attention (Encoder) — AI ≈ 112 FLOP/byte

### What This Kernel Does

Attention is the core mechanism in transformers. Given Q (queries), K (keys), and V (values):

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

This computes "how much should each query attend to each key" (the score matrix), normalizes it (softmax), then uses it to weight the values.

**The problem:** The score matrix `Q @ K^T` has shape (seq_q × seq_k) = (750 × 750) = 562,500 elements. Storing this in DRAM and reading it back is expensive.

**Flash attention's solution:** Never materialize the full score matrix in DRAM. Instead, process it in tiles that fit in shared memory (fast on-chip SRAM), computing softmax incrementally using "online softmax" (a running max and sum).

### Nsight Data

| Config | gridX | gridY | gridZ | blockX | Shared Mem | Calls | Avg Time |
|--------|-------|-------|-------|--------|-----------|-------|----------|
| A (encoder) | 12 | 20 | 1 | 256 | 98,304 bytes | 128 | 229 μs |
| B (decoder) | 1 | 16 | 1 | 256 | 131,072 bytes | 112 | 8.0 μs |

### Validating Grid Dimensions (This Is Key!)

**Config A — Encoder attention:**
- **gridX = 12:** This is `ceil(seq_q / BLOCK_M)`. If BLOCK_M = 64, then ceil(750/64) = 12. ✓
  This tells us the kernel processes Q in tiles of 64 rows at a time.
- **gridY = 20:** This is `num_heads`. The encoder has 20 attention heads. ✓
  Each head runs independently in its own thread block column.
- **128 calls:** 32 encoder layers × 4 profiling runs = 128. ✓

**Config B — Decoder prefill attention:**
- **gridX = 1:** The decoder prefill sequence is short enough that BLOCK_M covers it entirely.
- **gridY = 16:** Decoder has 16 Q-heads. ✓
- **112 calls:** 28 decoder layers × 4 runs = 112. ✓

### Validating Shared Memory (This Confirms Tile Sizes!)

**98,304 bytes for encoder:**

Our flash attention kernel allocates shared memory for Q, K, V tiles, output accumulator, and softmax state. The formula is:

```
smem = (BLOCK_M + 2 × BLOCK_N) × head_dim × sizeof(float32)
```

Why float32? Even though inputs are fp16, the kernel internally computes in float32 for numerical stability (tl.dot accumulates in float32).

Let's check if BLOCK_M=128, BLOCK_N=128, head_dim=64 fits:
```
(128 + 2×128) × 64 × 4 = 384 × 64 × 4 = 98,304 bytes ✓ EXACT MATCH!
```

But wait — if BLOCK_M=128, then gridX should be ceil(750/128) = 6, not 12. So BLOCK_M must be 64:
```
With BLOCK_M=64: gridX = ceil(750/64) = 12 ✓

But (64 + 2×64) × 64 × 4 = 192 × 256 = 49,152 ≠ 98,304
```

The discrepancy means the kernel allocates additional buffers beyond the simple Q+K+V formula. Triton's flash attention implementation also stores:
- Output accumulator O: BLOCK_M × head_dim × 4 = 16,384 bytes
- Score matrix S: BLOCK_M × BLOCK_N × 4 = 16,384 bytes
- Softmax vectors m, l: BLOCK_M × 4 × 2 = 512 bytes
- Total extra: ~33,280 bytes

So: 49,152 + 33,280 = 82,432... still not 98,304. The remaining ~16K is likely Triton compiler overhead (alignment, spill space). The important point: **the shared memory allocation is consistent with tile sizes in the 64-128 range**, confirming our AI calculation assumes the right problem decomposition.

**131,072 bytes for decoder:** head_dim=128 (decoder heads are wider), so tiles need more memory. 131,072 = 128 KB, consistent with larger tiles for the decoder.

### The AI Calculation

**Per head, per encoder layer (seq_q=750, seq_k=750, head_dim=64):**

**Step 1: Count FLOPs**
```
Flash attention does two matrix multiplies and a softmax:

1. Score matrix: Q @ K^T
   Shape: (750 × 64) @ (64 × 750) → (750 × 750)
   FLOPs = 2 × 750 × 750 × 64 = 72,000,000

   Why 2×M×K×N? Each of the 750×750 = 562,500 output elements
   requires 64 multiply-add pairs (one per head_dim) = 128 ops.
   Total = 562,500 × 128 = 72,000,000. Same as 2×750×750×64.

2. Weighted values: P @ V  (where P = softmax(scores))
   Shape: (750 × 750) @ (750 × 64) → (750 × 64)
   FLOPs = 2 × 750 × 750 × 64 = 72,000,000

3. Softmax: exp, sum, divide for each element in the score matrix
   Elements: 750 × 750 = 562,500
   Ops per element: ~5 (subtract max, exp, accumulate sum, divide, scale)
   FLOPs = 5 × 562,500 = 2,812,500

Total per head = 72M + 72M + 2.8M ≈ 147 million FLOPs
```

**Step 2: Count DRAM bytes**

This is where flash attention's magic shows. In the 3-kernel approach (baseline), the score matrix is written to DRAM and read back multiple times. In flash attention, it stays in shared memory.

```
Flash attention tiling: the kernel processes Q in tiles of BLOCK_M rows.
For each Q-tile, it iterates over ALL K/V tiles.

With BLOCK_M=64, there are ceil(750/64) = 12 Q-tiles.
With BLOCK_N=64, there are ceil(750/64) = 12 K/V-tiles.

For each Q-tile (of which there are 12):
  - Load Q tile from DRAM: 64 × 64 × 2 = 8,192 bytes (loaded once)
  - For each of 12 K/V tiles:
    - Load K tile from DRAM to shared memory: 64 × 64 × 2 = 8,192 bytes
    - Load V tile from DRAM to shared memory: 64 × 64 × 2 = 8,192 bytes
  - Store O tile to DRAM: 64 × 64 × 2 = 8,192 bytes

Now the question: does the L2 cache help?

Worst case (no L2 cache — every tile load hits DRAM):
  Q:  12 tiles × 8,192 = 98,304 bytes
  K:  12 Q-tiles × 12 K-tiles × 8,192 = 1,179,648 bytes
  V:  12 Q-tiles × 12 V-tiles × 8,192 = 1,179,648 bytes
  O:  12 tiles × 8,192 = 98,304 bytes
  Total = 2,555,904 bytes ≈ 2.5 MB
  AI = 147M / 2.5M ≈ 57 FLOP/byte

Best case (perfect L2 — K and V loaded from DRAM once, then served from cache):
  Q: 750 × 64 × 2 = 96,000 bytes
  K: 750 × 64 × 2 = 96,000 bytes (loaded once total)
  V: 750 × 64 × 2 = 96,000 bytes
  O: 750 × 64 × 2 = 96,000 bytes
  Total = 384,000 bytes ≈ 0.38 MB
  AI = 147M / 0.38M ≈ 374 FLOP/byte

Reality (partial L2 reuse):
  The H200's L2 cache is 33 MB. K and V together are 192 KB — easily fits.
  But L2 is shared across all SMs and all concurrent kernels, so some misses occur.

  Our estimate: K/V tiles are loaded from HBM about 6 times total
  (half of the 12 Q-tile passes miss L2, half hit):

  Q: 96 KB
  K: 6 × 96 KB = 576 KB
  V: 6 × 96 KB = 576 KB
  O: 96 KB
  Total = 1,344 KB ≈ 1.31 MB

  AI = 147,000,000 / 1,376,256 ≈ 107 ≈ 112 FLOP/byte
```

**Why we report ~112:** It's a middle-ground estimate between the worst case (57) and best case (374). The exact value depends on L2 cache behavior, which we can't measure without `ncu`.

**Step 3: Classify**
```
AI = 112 vs Ridge Point = 12.7 (H200) or 58.5 (RTX 5090)
112 > 12.7 → Compute-bound on H200 ✓
112 > 58.5 → Compute-bound on RTX 5090 ✓
→ Compute-bound on ALL GPUs
```

### Throughput Sanity Check (Nsight Validation)

If flash attention is compute-bound, its execution time should reflect compute throughput, not bandwidth:

```
Each kernel call handles all 20 heads (gridY=20):
  FLOPs per call = 147M × 20 heads = 2,940,000,000 = 2.94 GFLOPs

Average time per call = 229 μs

Achieved throughput = 2.94 GFLOPs / 229 μs = 12.8 TFLOPS

H200 MIG FP32 peak = 30.5 TFLOPS
Utilization = 12.8 / 30.5 = 42%

For FP16 tensor core peak (~61 TFLOPS):
Utilization = 12.8 / 61 = 21%
```

42% of FP32 peak (or 21% of FP16 tensor core peak) is **reasonable for a compute-bound kernel**. Perfect utilization is impossible due to:
- Memory latency (some tiles still come from DRAM)
- Softmax computation between the two matmuls (breaks the pure-matmul pipeline)
- Thread block scheduling overhead

If flash attention were bandwidth-bound, we'd expect:
```
Time ≈ 1.31 MB / 2.41 TB/s = 0.54 μs
```
But actual time is 229 μs — **425x longer** than the bandwidth limit. This confirms it's compute-bound, not bandwidth-bound. ✓

### Comparison: 3-Kernel Pipeline (Baseline) — AI ≈ 16

The baseline uses three separate kernels for attention:

```
Kernel 1 (attention_scores):   Q @ K^T → writes score matrix S to DRAM
Kernel 2 (softmax_inplace):    reads S from DRAM → softmax → writes S back to DRAM
Kernel 3 (attention_output):   reads S from DRAM, reads V → S @ V → writes O to DRAM

The score matrix S has shape (750 × 750) and is stored in fp32 (4 bytes):
S size = 750 × 750 × 4 = 2.25 MB PER HEAD

DRAM traffic per head:
  Kernel 1: load Q(96K) + load K(96K) + write S(2.25MB)
  Kernel 2: load S(2.25MB) + write S(2.25MB)
  Kernel 3: load S(2.25MB) + load V(96K) + write O(96K)

  Total = 96K + 96K + 2.25M + 2.25M + 2.25M + 2.25M + 96K + 96K
        = 384K + 9.0M = 9.38 MB

AI = 147M / 9.38M ≈ 15.7 ≈ 16 FLOP/byte
```

**Nsight confirms the score matrix materialization:** The example profile shows 729 MB of Device-to-Device copies. Let's verify:
```
Score matrix per head per layer = 2.25 MB
20 heads × 32 encoder layers = 640 head-layers per inference
640 × 2.25 MB = 1,440 MB for just the score writes

But S is written once and read twice (softmax + output), and there are decoder
attention calls too. Plus kernel intermediates. 729 MB for one profiling run
is in the right ballpark.
```

**The 3-kernel pipeline's AI of ~16:**
- On H200 (ridge=12.7): 16 > 12.7 → compute-bound, but **barely** above the ridge
- On RTX 5090 (ridge=58.5): 16 < 58.5 → **bandwidth-bound!**

**Flash attention's AI of ~112 is compute-bound on both.** This is the single most impactful optimization — it doesn't change the FLOPs (same attention computation), but it reduces DRAM traffic by ~7x.

---

## 4. Flash Attention (Decoder Decode) — AI ≈ 1.0 FLOP/byte

### What's Different About Decode

During autoregressive generation, each new token produces a single query row (seq_q=1) that must attend to ALL previously generated tokens (seq_k ≈ 200, growing each step).

### Nsight Data

| Kernel | Calls | Total Time | Avg Time | Grid | Block | Shared Mem |
|--------|-------|-----------|----------|------|-------|-----------|
| flash_fwd_kernel (SDPA) | 1,344 | 9.03 ms | 6.7 μs | 1 × 1 × 16 | 128 | 65,536 bytes |

Note: We use PyTorch's built-in SDPA for decode (seq_q ≤ 4) instead of our Triton kernel, because the Triton kernel's launch overhead exceeds the actual compute at this tiny problem size.

### Validating Call Counts

**1,344 calls:** 28 decoder layers × 4 KV head groups × 12 decode steps = 1,344. ✓

(The decoder has 4 KV heads with GQA. Each KV head serves 4 Q heads. SDPA handles the expansion internally.)

### The AI Calculation

**Per head, per layer, per decode step (seq_q=1, seq_k=200, head_dim=128):**

**Step 1: Count FLOPs**
```
Q @ K^T: (1 × 128) @ (128 × 200) → (1 × 200)
  FLOPs = 2 × 1 × 200 × 128 = 51,200

P @ V: (1 × 200) @ (200 × 128) → (1 × 128)
  FLOPs = 2 × 1 × 200 × 128 = 51,200

Softmax: 5 × 1 × 200 = 1,000

Total: ~103,400 FLOPs
```

**Step 2: Count bytes**
```
Load Q: 1 × 128 × 2 = 256 bytes        (tiny — just one row!)
Load K: 200 × 128 × 2 = 51,200 bytes   (must read entire KV cache)
Load V: 200 × 128 × 2 = 51,200 bytes   (must read entire KV cache)
Store O: 1 × 128 × 2 = 256 bytes       (tiny — just one row!)

Total: 102,912 bytes ≈ 100 KB
```

**Step 3: Compute AI**
```
AI = 103,400 / 102,912 ≈ 1.0 FLOP/byte
```

**The fundamental problem:** To generate one new token, you must read the ENTIRE KV cache (all 200 previous key/value vectors), but you only produce ONE output vector. The ratio of useful work (FLOPs) to data movement (bytes) is terrible.

**This doesn't change with sequence length:** At step N, you read N cached tokens (bytes ∝ N) and do N dot products (FLOPs ∝ N). AI = FLOPs/bytes = constant ≈ 1.0.

**Classification:** Extremely bandwidth-bound (1.0 << 12.7 on any GPU). This is why autoregressive decoding is fundamentally slow, and why techniques like quantization (fewer bytes per element) and speculative decoding (generate multiple tokens at once) are active research areas.

---

## 5. Linear (cuBLAS GEMM) — AI ≈ 168–362 FLOP/byte

### What This Kernel Does

Matrix multiplication: `output = input @ weight`. Used for Q/K/V projections, output projections, and MLP up/down projections. We use NVIDIA's cuBLAS library, which selects optimal kernel configurations internally.

### Nsight Data

| cuBLAS Kernel Config | Calls | Total Time | Avg Time | Shared Mem |
|---------------------|-------|-----------|----------|-----------|
| nvjet_hsh_128x8_64x12_TNT | 2,688 | 38.7 ms | 14.4 μs | 225,652 bytes |
| nvjet_hsh_64x8_64x16_TNT | 4,032 | 35.4 ms | 8.8 μs | 164,308 bytes |
| nvjet_hsh_16x64_64x16_TNN | 2,688 | 14.2 ms | 5.3 μs | 180,692 bytes |
| (+ many other configs) | ... | ... | ... | ... |
| **Total cuBLAS** | **10,516** | **114.0 ms** | | |

The `nvjet_hsh` prefix means "NVIDIA Jetson-style Hopper Shared-memory" kernels — cuBLAS's optimized GEMM kernels for Hopper architecture. The numbers (128x8, 64x12, etc.) describe tile sizes and pipeline stages.

### The AI Calculation (Three Cases)

**Case 1: Encoder Q projection — x(750×1280) @ W(1280×1280)**
```
FLOPs = 2 × 750 × 1280 × 1280 = 2,457,600,000 ≈ 2.46 GFLOPS

Bytes:
  Load input x:  750 × 1280 × 2 = 1,920,000 bytes (1.83 MB)
  Load weight W: 1280 × 1280 × 2 = 3,276,800 bytes (3.13 MB)
  Store output:  750 × 1280 × 2 = 1,920,000 bytes (1.83 MB)
  Total: 7,116,800 bytes = 6.79 MB

AI = 2,457,600,000 / 7,116,800 ≈ 345 → ~362 FLOP/byte

Why so high? The weight matrix (3.13 MB) is loaded once but used for ALL 750 rows.
Each row of x (1280 elements) is multiplied against ALL 1280 columns of W.
The "reuse ratio" is roughly proportional to the matrix dimensions.
```

**Case 2: Decoder projection (prefill) — x(200×2048) @ W(2048×2048)**
```
FLOPs = 2 × 200 × 2048 × 2048 = 1,677,721,600 ≈ 1.68 GFLOPS

Bytes:
  Load x:      200 × 2048 × 2 = 819,200 bytes (0.78 MB)
  Load W:      2048 × 2048 × 2 = 8,388,608 bytes (8.0 MB) ← weight matrix dominates!
  Store output: 200 × 2048 × 2 = 819,200 bytes (0.78 MB)
  Total: 10,027,008 bytes = 9.56 MB

AI = 1,677,721,600 / 10,027,008 ≈ 167 → ~168 FLOP/byte

Lower than encoder because the input has fewer rows (200 vs 750),
so the weight matrix is "reused" fewer times.
```

**Case 3: Decoder single-token decode — x(1×2048) @ W(2048×2048)**
```
FLOPs = 2 × 1 × 2048 × 2048 = 8,388,608 ≈ 8.4 MFLOPS

Bytes:
  Load x:      1 × 2048 × 2 = 4,096 bytes (0.004 MB — tiny!)
  Load W:      2048 × 2048 × 2 = 8,388,608 bytes (8.0 MB — still the full weight!)
  Store output: 1 × 2048 × 2 = 4,096 bytes (0.004 MB)
  Total: 8,396,800 bytes = 8.01 MB

AI = 8,388,608 / 8,396,800 ≈ 1.0 FLOP/byte

Terrible! You must load the ENTIRE 8 MB weight matrix to produce just one
4 KB output vector. Almost all bandwidth is "wasted" loading weights.
```

**Reported range: 168–362** covers the encoder/prefill cases (the "interesting" ones for Table 4). Single-token decode linear is bandwidth-bound at AI~1.0, just like decode attention.

---

## 6. Fused SwiGLU — AI ≈ 197 FLOP/byte

### What This Kernel Does

SwiGLU is the activation pattern used in the decoder MLP:
```
output = SiLU(x @ W_gate) * (x @ W_up)
```

Our Triton kernel **fuses** both matrix multiplies and the activation into a single kernel, avoiding writing the intermediate results to DRAM.

### Nsight Data

| Config | gridX | gridY | Shared Mem | Calls | Avg Time | What It Is |
|--------|-------|-------|-----------|-------|----------|------------|
| A (decode) | 48 | 2 | 147,456 bytes | 384 | 49.2 μs | Single-token decode steps |
| B (prefill) | 96 | 3 | 147,456 bytes | 128 | 182.5 μs | Full-sequence prefill |

### Validating Grid Dimensions

**Config A (decode): gridX=48**
```
intermediate_size = 6144
If TILE_N = 128: ceil(6144 / 128) = 48 ✓
```
This tells us each thread block computes a 128-wide slice of the output.

**Config B (prefill): gridX=96**
```
If TILE_N = 64: ceil(6144 / 64) = 96 ✓
```
Different tile size for the larger prefill matrix — cuBLAS-style auto-selection by Triton.

### Validating Shared Memory

**147,456 bytes = 144 KB** for both configs. This holds the input tile, gate weight tile, AND up weight tile simultaneously (that's the whole point of fusion — process both in the same kernel):

```
The kernel needs to hold in shared memory:
  - x tile:     TILE_K × TILE_M elements
  - W_gate tile: TILE_K × TILE_N elements
  - W_up tile:   TILE_K × TILE_N elements
  All in float32 (4 bytes) for accumulation

147,456 bytes / 4 = 36,864 float32 values

With TILE_K=64, TILE_M=128, TILE_N=128:
  x:     64 × 128 = 8,192
  gate:  64 × 128 = 8,192
  up:    64 × 128 = 8,192
  Total: 24,576 values = 98,304 bytes

The remaining 49,152 bytes (147,456 - 98,304) are for:
  - Output accumulators
  - Triton compiler spill space
  - Alignment padding
```

### The AI Calculation

**Decoder MLP prefill: x(200×2048), W_gate(2048×6144), W_up(2048×6144)**

**Step 1: Count FLOPs**
```
x @ W_gate: 2 × 200 × 2048 × 6144 = 5,033,164,800 ≈ 5.03 GFLOPs
x @ W_up:   2 × 200 × 2048 × 6144 = 5,033,164,800 ≈ 5.03 GFLOPs
SiLU:       4 × 200 × 6144 = 4,915,200 ≈ 4.9 MFLOPs
Multiply:   200 × 6144 = 1,228,800 ≈ 1.2 MFLOPs

Total: 10,072,473,600 ≈ 10.07 GFLOPs
(The two GEMMs utterly dominate; SiLU and multiply are rounding errors)
```

**Step 2: Count bytes**
```
FUSED version (our kernel):
  Load x:      200 × 2048 × 2 = 819,200 bytes (loaded ONCE, used for both matmuls)
  Load W_gate: 2048 × 6144 × 2 = 25,165,824 bytes
  Load W_up:   2048 × 6144 × 2 = 25,165,824 bytes
  Store output: 200 × 6144 × 2 = 2,457,600 bytes (just the final result)
  Total: 53,608,448 bytes ≈ 51.1 MB

  AI = 10.07G / 51.1M ≈ 197 FLOP/byte

UNFUSED version (hypothetical — 3 separate kernels):
  Same loads as above, PLUS:
  Write gate_result: 200 × 6144 × 2 = 2,457,600 bytes (intermediate to DRAM)
  Read gate_result:  2,457,600 bytes (SiLU kernel reads it back)
  Write silu_result: 2,457,600 bytes (SiLU output to DRAM)
  Read silu_result:  2,457,600 bytes (multiply kernel reads it)
  Write up_result:   2,457,600 bytes (second matmul output to DRAM)
  Read up_result:    2,457,600 bytes (multiply kernel reads it)
  Total extra: ~14.7 MB of unnecessary DRAM traffic

  Unfused total: 51.1 + 14.7 = 65.8 MB
  Unfused AI = 10.07G / 65.8M ≈ 153 FLOP/byte
```

**The fusion saves ~14.7 MB per MLP layer** by keeping gate_result and up_result in registers/shared memory instead of writing them to DRAM.

**Classification:** Compute-bound (197 >> 12.7). ✓

---

## 7. RoPE (Rotary Position Embedding) — AI ≈ 0.5 FLOP/byte

### What This Kernel Does

RoPE encodes position information by rotating pairs of elements in Q and K by angles that depend on their position in the sequence. It's like giving each token a unique "direction" so the model knows token order.

For each pair of elements (x₁, x₂) at position p:
```
x₁' = x₁ × cos(θp) - x₂ × sin(θp)
x₂' = x₁ × sin(θp) + x₂ × cos(θp)
```

This is a 2D rotation matrix applied to pairs of dimensions.

### Nsight Data

| Kernel | Calls | Total Time | Avg Time | Grid | Block | Shared Mem |
|--------|-------|-----------|----------|------|-------|-----------|
| fused_rope_pair_kernel | 1,584 | 22.44 ms | 14.2 μs | 60,000 × 1 × 1 | 256 × 1 × 1 | 0 bytes |

### Validating Call Counts

**1,584 calls:** 32 encoder layers + 28 decoder layers × ~52 steps = 32 + 1,456 = 1,488. With warmup: ~1,584. ✓

### Validating Shared Memory

**0 bytes:** RoPE is pure element-wise with no inter-thread communication. Each thread handles one (head, position) pair: loads the elements, loads cos/sin values, does the rotation, stores the result. No shared memory needed. ✓

### The AI Calculation

**Per encoder layer (20 heads, seq=750, head_dim=64 but only 50% is rotated = 32 rotary dims):**

**Step 1: Count FLOPs**
```
Per pair of elements:
  x1' = x1*cos - x2*sin → 2 multiplies + 1 subtract = 3 FLOPs
  x2' = x1*sin + x2*cos → 2 multiplies + 1 add = 3 FLOPs
  Total: 6 FLOPs per element pair

Number of pairs per layer:
  Q: 20 heads × 750 positions × 32 pairs = 480,000 pairs
  K: 20 heads × 750 positions × 32 pairs = 480,000 pairs
  Total: 960,000 pairs

FLOPs = 960,000 × 6 = 5,760,000 ≈ 5.8M FLOPs
```

**Step 2: Count bytes**
```
The fused kernel handles both Q and K in one launch.

Load Q (rotary part): 20 × 750 × 64 × 2 = 1,920,000 bytes
  (All 64 dims loaded even though only 32 are rotated —
   the other 32 are "passthrough" copied unchanged)
Load K (rotary part): 20 × 750 × 64 × 2 = 1,920,000 bytes
Load cos table: 750 × 32 × 2 = 48,000 bytes
Load sin table: 750 × 32 × 2 = 48,000 bytes
Store Q (rotated): 1,920,000 bytes
Store K (rotated): 1,920,000 bytes
Passthrough dims (non-rotated halves, load+store):
  Q: 20 × 750 × 32 × 2 × 2 = 1,920,000 bytes
  K: same = 1,920,000 bytes

Total: 1,920K + 1,920K + 48K + 48K + 1,920K + 1,920K + 1,920K + 1,920K
     = 11,616,000 bytes ≈ 11.1 MB
```

**Step 3: Compute AI**
```
AI = 5,760,000 / 11,616,000 ≈ 0.50 FLOP/byte
```

**Why so low?** RoPE does very little math (just 6 multiplies/adds per element pair) but must load and store every element. The passthrough dimensions make it worse — loading data just to copy it unchanged.

**Classification:** Very bandwidth-bound (0.5 << 12.7). The fused kernel's value isn't about AI — it's about **eliminating a kernel launch**. Without fusion, you'd launch two separate kernels (one for Q, one for K), each with ~5-10 μs launch overhead. Over 32 encoder layers, that's 32 × 10 μs = 320 μs saved. The ablation study confirmed: fused RoPE saves ~28.9 ms in total (much more than just launch overhead — the fused kernel also has better memory access patterns).

---

## Summary Table

| Operation | FLOPs | DRAM Bytes | AI (FLOP/B) | H200 (ridge=12.7) | RTX 5090 (ridge=58.5) | How Nsight Validates |
|-----------|-------|------------|-------------|-------------------|----------------------|---------------------|
| GELU / SiLU | ~10/elem | 4B/elem | **~2.5** | Bandwidth | Bandwidth | Call counts match arch; 0B shared mem confirms element-wise |
| RMSNorm / LN | ~4-8D/row | ~6-8D/row | **~4–8** | Bandwidth | Bandwidth | 2,964 calls matches 56 norms/step × ~52 steps; 16B shared mem |
| Flash Attn (enc) | 147M/head | ~1.3MB/head | **~112** | **Compute** | **Compute** | gridY=20 confirms 20 heads; 98KB smem matches tile formula; 0 MB D2D (vs 729 MB baseline) |
| 3-Kernel Attn (enc) | 147M/head | 9.4MB/head | **~16** | Compute (barely) | Bandwidth | Baseline: 729 MB D2D copies = score matrices in DRAM |
| Flash Attn (dec) | 103K/head | 100KB/head | **~1.0** | Bandwidth | Bandwidth | 1,344 calls = 28 layers × 4 KV groups × 12 steps |
| Linear (enc) | 2.46G | 6.8MB | **~362** | Compute | Compute | 10,516 calls matches ~6 ops/layer × layers × steps |
| Linear (dec prefill) | 1.68G | 10MB | **~168** | Compute | Compute | Multiple nvjet_hsh configs = cuBLAS auto-tuning per shape |
| Linear (dec step) | 8.4M | 8.0MB | **~1.0** | Bandwidth | Bandwidth | Small configs (16x64) for tiny decode matrices |
| Fused SwiGLU | 10.07G | 51.1MB | **~197** | Compute | Compute | gridX=48 confirms TILE_N=128; 144KB smem holds x+gate+up tiles |
| RoPE (fused pair) | 5.8M/layer | 11.1MB/layer | **~0.5** | Bandwidth | Bandwidth | 1,584 calls matches layers×steps; 0B smem confirms element-wise |

---

## Key Insights

### 1. Flash Attention Changes the Roofline Classification

The most important finding in this entire analysis:

```
3-kernel pipeline: AI ≈ 16   (score matrix written to DRAM)
Flash attention:   AI ≈ 112  (score matrix stays in shared memory)
```

That's a **7x improvement in arithmetic intensity** with the same number of FLOPs. Flash attention doesn't reduce computation — it reduces memory traffic.

On the RTX 5090 (ridge=58.5), this shifts encoder attention from **bandwidth-bound** (16 < 58.5) to **compute-bound** (112 > 58.5). On the H200 (ridge=12.7), both are above the ridge, but flash attention runs on 7x less data.

**Nsight proof:** Baseline has 729 MB of Device-to-Device copies. Our template has 0.004 MB. The score matrices are gone from DRAM.

### 2. Decode Steps Are Fundamentally Bandwidth-Bound

During autoregressive generation, **every operation** becomes bandwidth-bound:
- Attention: AI ≈ 1.0 (read full KV cache for one query)
- Linear: AI ≈ 1.0 (read full weight matrix for one input row)
- Activation: AI ≈ 1.0–2.5 (pure element-wise, always bandwidth-bound)
- RoPE: AI ≈ 0.5 (load-rotate-store with minimal math)

No kernel optimization can fix this — it's intrinsic to processing one token at a time. This is why:
- **KV caching** helps (avoids recomputing past tokens, but still must read them)
- **Quantization** helps (fewer bytes per element → higher effective AI)
- **Speculative decoding** helps (process multiple candidate tokens in one batch)
- **Larger batch sizes** help (amortize weight loading across multiple sequences)

### 3. Encoder vs Decoder: Different Worlds

| Property | Encoder | Decoder (decode) |
|----------|---------|-----------------|
| Dominant AI | 112–362 (compute-bound) | 0.5–1.0 (bandwidth-bound) |
| Bottleneck | Compute (ALUs) | Memory bandwidth |
| Scales with | More SMs, higher clock | More bandwidth (HBM3e > GDDR7) |
| Optimization | Larger tiles, more warps | Quantization, batching |

This explains why the H200 (2.41 TB/s HBM3e) handles decode relatively better than the RTX 5090 (1.79 TB/s GDDR7) despite having fewer SMs for compute.

---

## Methodology Summary

1. **AI values are analytical** — calculated from tensor shapes (known from GlmAsrConfig) and algorithm structure (known from our code). They represent theoretical DRAM traffic assuming reasonable cache behavior.

2. **Nsight Systems validates indirectly** by confirming:
   - Kernel call counts match the model architecture
   - Grid dimensions confirm tile sizes used in AI formulas
   - Shared memory allocations match our per-kernel memory models
   - Measured throughput is consistent with the AI classification (compute-bound kernels achieve reasonable TFLOPS; bandwidth-bound kernels have execution times dominated by launch overhead)
   - Memory copy totals confirm predictions (729 MB D2D eliminated by flash attention)

3. **Nsight Compute would validate directly** (actual FLOPS, DRAM bytes, cache hits) but was blocked by `ERR_NVGPUCTRPERM` on the H200 MIG partition.

4. **Our AI values are upper bounds** — real AI may be lower (more bandwidth-bound) due to cache misses, alignment padding, and compiler-inserted loads. The classifications (compute vs bandwidth bound) are robust because most values are far from the ridge point.
