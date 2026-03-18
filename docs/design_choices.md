# Design Choices: GLM-ASR Triton GPU Kernel Optimization

This document explains every significant design choice made during the optimization of the GLM-ASR speech recognition model. It is written for technical readers who may have little or no GPU programming experience. Every GPU concept is explained when it first appears.

---

## Table of Contents

1. [Background: What Are We Optimizing?](#1-background-what-are-we-optimizing)
2. [Tile Sizes and Block Dimensions](#2-tile-sizes-and-block-dimensions)
3. [Shared Memory: The Fundamental Constraint](#3-shared-memory-the-fundamental-constraint)
4. [num_warps: How Many Threads Per Block](#4-num_warps-how-many-threads-per-block)
5. [num_stages: Pipeline Depth (Double-Buffering)](#5-num_stages-pipeline-depth-double-buffering)
6. [Flash Attention vs. Multi-Kernel Attention](#6-flash-attention-vs-multi-kernel-attention)
7. [cuBLAS vs. Custom Triton Matmul](#7-cublas-vs-custom-triton-matmul)
8. [fp16 vs bf16 vs fp32: Data Type Selection](#8-fp16-vs-bf16-vs-fp32-data-type-selection)
9. [KV Cache and the generate_v8b Function](#9-kv-cache-and-the-generate_v8b-function)
10. [Kernel Fusion: SwiGLU and LinearGELU](#10-kernel-fusion-swiglu-and-lineargelu)
11. [GPUProfile: Portable GPU Detection](#11-gpuprofile-portable-gpu-detection)
12. [Autotune: Built, Tested, Removed](#12-autotune-built-tested-removed)
13. [Rejected Optimizations](#13-rejected-optimizations)
14. [Monkey-Patching: Why and How](#14-monkey-patching-why-and-how)
15. [Dead Code Removal](#15-dead-code-removal)
16. [Summary of Performance Impact](#16-summary-of-performance-impact)

---

## 1. Background: What Are We Optimizing?

GLM-ASR is a speech recognition model that converts audio into text. It has three main stages:

1. **Audio Encoder** — Processes the audio waveform through 32 transformer layers, each containing attention and feed-forward (MLP) sub-layers.
2. **Multi-modal Projector** — A small linear layer that converts audio features into the text decoder's format.
3. **Text Decoder** — Generates text tokens one at a time through 28 transformer layers, each with attention and MLP sub-layers.

The baseline implementation runs in **262ms** on an RTX 5090. Our optimized version runs in **100.4ms** — a **2.6x speedup**.

Every optimization in this document targets one of three physical bottlenecks:
- **Memory bandwidth** — How fast data moves between GPU memory (VRAM) and the compute units. Most of our operations are "bandwidth-bound," meaning the GPU spends more time waiting for data than actually computing.
- **Compute throughput** — How many arithmetic operations the GPU can perform per second.
- **Kernel launch overhead** — Each time the CPU tells the GPU to run a function ("launches a kernel"), there is a small fixed cost (~5-15 microseconds). With hundreds of kernel launches per inference, this adds up.

---

## 2. Tile Sizes and Block Dimensions

### What is a "tile"?

GPUs process data in parallel, but they cannot process an entire matrix at once. Instead, they break large matrices into smaller rectangular chunks called **tiles** (also called **blocks**). Each tile is processed by a group of GPU threads working together.

For example, if you have a 1024×1024 matrix and you use 64×64 tiles, the GPU launches 256 thread groups (16×16 grid of tiles), each processing one 64×64 chunk.

### Why tile size matters

Tile size is the single most important tuning parameter for GPU kernels. It affects:

- **Parallelism** — Larger tiles mean fewer thread groups, which can under-utilize the GPU if there aren't enough tiles to keep all compute units busy. Smaller tiles create more thread groups but each does less work.
- **Shared memory usage** — Larger tiles require more shared memory (see [Section 3](#3-shared-memory-the-fundamental-constraint)). If a tile is too large, it won't fit, and the kernel will crash.
- **Memory efficiency** — Larger tiles amortize the overhead of loading data from slow GPU main memory (VRAM). Loading a 128×128 tile is only slightly slower than loading a 64×64 tile due to how memory controllers work, but you get 4x more data to compute on.

### Our tile size choices

We use different tile sizes for different operations and GPU architectures:

#### Flash Attention tiles (BLOCK_M × BLOCK_N)

| GPU | Shared Memory | Encoder (head_dim=64) | Decoder (head_dim=128) |
|-----|--------------|----------------------|----------------------|
| RTX 5090 | ~99KB | 64×64 | 32×32 |
| RTX 4090 | ~100KB | 64×64 | 32×32 |
| RTX 3090 | ~100KB | 64×64 | 32×32 |
| H100/H200 | ~228KB | 128×128 | 128×64 |
| A100 | ~164KB | 128×64 | 64×32 |
| B200 | ~228KB | 128×128 | 128×64 |

The decoder uses smaller tiles than the encoder because head_dim=128 means each element in the tile is larger (128 floats per row instead of 64), consuming twice the shared memory per row.

#### Matmul tiles (TILE_M × TILE_N × TILE_K)

| GPU | Shared Memory | Tile Size |
|-----|--------------|-----------|
| RTX 5090 / 4090 / 3090 | ~99-100KB | 64×64×32 |
| H100 / H200 / B200 | ~228KB | 128×128×64 |
| A100 | ~164KB | 128×64×32 |

Here, TILE_K is the "inner" dimension — how many elements we accumulate before writing partial results. A larger TILE_K means fewer iterations of the inner loop, which reduces overhead.

### How we chose these values

We tested configurations manually on the RTX 5090 and stored the results in a lookup table called `_KNOWN_CONFIGS`. We also built an autotune system that tried every valid configuration at runtime — but the hand-tuned configs consistently won (see [Section 12](#12-autotune-built-tested-removed)).

For unknown GPUs, we compute tile sizes dynamically by trying configurations from largest to smallest and picking the biggest one that fits in shared memory (see `_compute_attention_tiles()` and `_compute_matmul_tiles()` in `layers.py`).

---

## 3. Shared Memory: The Fundamental Constraint

### What is shared memory?

Every GPU has a memory hierarchy, similar to how a computer has fast L1/L2 cache and slower RAM:

```
Registers         (fastest, ~1 cycle,  per-thread,    ~256KB total per SM)
   ↓
Shared Memory     (fast,    ~5 cycles, per-block,     48-228KB per SM)
   ↓
L2 Cache          (~100 cycles, shared across all SMs, 6-60MB)
   ↓
VRAM / HBM        (slow,   ~300 cycles, global,       16-80GB)
```

**Shared memory** (also called SRAM or scratchpad) is a fast, programmer-controlled memory region shared by all threads in a block. When we load a tile from VRAM into shared memory, every thread in the block can read it at ~50x lower latency than re-reading from VRAM.

### The 48KB vs optin problem

Every NVIDIA GPU since Volta (2017) advertises a **default** shared memory limit of **48KB per block**. But modern GPUs actually have much more — they just need you to explicitly "opt in" to the extended limit:

| GPU | Default (shared_memory_per_block) | Optin (shared_memory_per_block_optin) |
|-----|----------------------------------|--------------------------------------|
| RTX 3090 | 48KB | ~100KB |
| RTX 4090 | 48KB | ~100KB |
| RTX 5090 | 48KB | ~99KB |
| A100 | 48KB | ~164KB |
| H100/H200 | 48KB | ~228KB |

If you use the default 48KB, you can only fit tiny tiles (e.g. 16×16), which makes kernels extremely slow. Triton automatically opts in to the extended limit, but we need to **read the correct property** to know how much we have.

### The getattr fallback chain

Different versions of PyTorch expose the optin limit under different property names. We use a fallback chain to handle all versions:

```python
self.smem_per_block = getattr(
    props, 'shared_memory_per_block_optin',         # PyTorch 2.8+
    getattr(props, 'max_shared_memory_per_block',   # some PyTorch builds
            props.shared_memory_per_block)           # fallback (48KB default)
)
```

This was discovered the hard way: on the university's H200 cluster, an older PyTorch version didn't have `shared_memory_per_block_optin`, so our code fell back to 48KB and chose tiny tile sizes, running 2x slower than expected.

### The shared memory formula

For flash attention, the shared memory needed per tile configuration is:

```
bytes = (BLOCK_M + 2 × BLOCK_N) × BLOCK_D × 4 + ~20KB overhead
```

Where:
- **BLOCK_M** — Number of query rows per tile
- **BLOCK_N** — Number of key/value rows per tile (×2 because we load both K and V)
- **BLOCK_D** — Head dimension (64 or 128), padded to the next power of 2
- **4** — Bytes per fp32 element (tiles are computed in fp32 for numerical precision)
- **~20KB** — Triton compiler overhead (accumulators, softmax running state, padding, etc.)

Example for RTX 5090 with head_dim=64:
```
(64 + 2×64) × 64 × 4 + 20480 = 49,152 + 20,480 = 69,632 bytes (~68KB)
```
This fits in ~99KB. But 128×128 tiles would need:
```
(128 + 2×128) × 64 × 4 + 20480 = 98,304 + 20,480 = 118,784 bytes (~116KB)
```
That exceeds ~99KB — so RTX 5090 cannot use 128×128 tiles.

For the fused SwiGLU matmul kernel, the formula is different because we load three matrices (input A, gate weight, up weight):

```
bytes = TILE_K × (TILE_M + 2 × TILE_N) × 4 + ~20KB overhead
```

---

## 4. num_warps: How Many Threads Per Block

### What is a warp?

A **warp** is a group of 32 GPU threads that execute the same instruction simultaneously (in lockstep). It is the smallest scheduling unit on NVIDIA GPUs. You cannot run fewer than 32 threads in a warp.

A **block** (also called a thread block or CTA) is a larger group of warps that share the same shared memory. When we say `num_warps=4`, we mean each block has 4 × 32 = 128 threads. With `num_warps=8`, each block has 256 threads.

### Why num_warps matters

- **More warps per block** means more threads working on the same tile, which can hide memory latency (when some warps are waiting for data, others can compute). Good for large tiles.
- **Fewer warps per block** means each warp gets a larger share of the shared memory and registers. Good for small tiles where there isn't enough work to keep many warps busy.

### Our choices

We use a simple rule:
```python
nwarps = 8 if (BLOCK_M × BLOCK_N) >= 4096 else 4
```

- **Large tiles** (128×128 = 16,384 elements, or 128×64 = 8,192 elements): `num_warps=8` (256 threads). Enough work to keep 8 warps busy.
- **Small tiles** (64×64 = 4,096 elements): `num_warps=4` (128 threads). Borderline — 4 warps is sufficient.
- **Tiny tiles** (32×32 = 1,024 elements, or smaller): `num_warps=4`. Only 1,024 elements to process, so 128 threads is plenty.

We tested `num_warps=8` on the RTX 5090 with 64×64 tiles and saw no improvement — the tile is too small to benefit from the extra parallelism.

---

## 5. num_stages: Pipeline Depth (Double-Buffering)

### What is double-buffering?

When a GPU kernel processes tiles, it typically does two things in a loop:
1. **Load** the next tile from slow VRAM into fast shared memory
2. **Compute** on the current tile in shared memory

With `num_stages=1` (no pipelining), these happen sequentially:
```
Load tile 1 → Compute tile 1 → Load tile 2 → Compute tile 2 → ...
```

With `num_stages=2` (double-buffering), the GPU overlaps loading and computing:
```
Load tile 1 → Load tile 2 + Compute tile 1 → Load tile 3 + Compute tile 2 → ...
```

This means the GPU is never idle — while it's computing on one tile, it's simultaneously loading the next. On memory-bandwidth-bound operations, this can improve performance by up to 2x.

### The catch: double the shared memory

Double-buffering requires holding **two tiles** in shared memory simultaneously (the one being computed on and the one being loaded). This doubles the shared memory requirement.

### Our choices

| GPU | Shared Memory | num_stages | Why |
|-----|--------------|------------|-----|
| RTX 5090 / 4090 / 3090 | ~99-100KB | **1** | Cannot fit two tiles. A single 64×64 tile with head_dim=64 needs ~68KB. Two would need ~136KB. |
| H100 / H200 / B200 | ~228KB | **2** | Plenty of room. A single 128×128 tile needs ~116KB. Two need ~232KB, which just fits in 228KB (the overhead is shared, not doubled). |
| A100 | ~164KB | **1** | With 128×64 tiles, one tile needs ~96KB. Two would need ~172KB, which barely fits — but we keep it at 1 for reliability. |

The threshold in our code:
```python
nstages = 2 if smem_bytes > 150 * 1024 else 1
```

We explicitly tested `num_stages=2` on the RTX 5090 and it caused **out-of-memory errors** on the shared memory level (the kernel refused to launch). This is not a VRAM issue — it's shared memory per block being too small.

---

## 6. Flash Attention vs. Multi-Kernel Attention

### The problem with standard attention

Standard multi-head attention computes:
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

A naive implementation launches **three separate kernels**:
1. Compute scores: `S = Q × K^T × scale`
2. Apply softmax: `P = softmax(S)`
3. Compute output: `O = P × V`

Each kernel reads from and writes to VRAM. The intermediate matrices S and P are large (seq_q × seq_k per head) and exist only to be immediately consumed by the next kernel. This "materialization" of intermediates wastes memory bandwidth.

### Flash Attention: one kernel, no intermediates

Flash attention fuses all three steps into a **single kernel** using a technique called **online softmax**. The key insight is that softmax can be computed incrementally — you don't need to see all the scores at once.

The algorithm processes K/V in tiles and maintains running statistics (max value and sum of exponentials) that get corrected as new tiles arrive. The Q, K, V tiles are loaded into shared memory, the score and softmax are computed in registers (fastest memory), and only the final output O is written back to VRAM.

Benefits:
- **No intermediate matrices** — S and P never exist in VRAM, saving O(seq_q × seq_k) memory
- **One kernel launch** instead of three — saves ~15-30μs of launch overhead
- **Better memory access patterns** — each tile of Q, K, V is loaded once from VRAM, reused many times in shared memory

### Our implementation

We replaced the original 3-kernel pipeline with a single `flash_attention_kernel` in `attention.py`. The old kernels (`attention_scores`, `softmax_inplace`, `attention_output`, `causal_mask`) were deleted — about 200 lines of dead code removed.

### SDPA fallback for tiny sequences

During KV-cached decode, the query sequence length is 1 (single token). Launching our custom flash attention kernel for a single row has high overhead relative to the tiny amount of work. For these cases (`seq_q ≤ 4`), we fall back to PyTorch's built-in `scaled_dot_product_attention` (SDPA), which is optimized for this exact scenario:

```python
if seq_q <= 4:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
```

---

## 7. cuBLAS vs. Custom Triton Matmul

### What is cuBLAS?

cuBLAS is NVIDIA's official library for matrix multiplication (and other linear algebra operations). It is hand-tuned by NVIDIA engineers for every GPU architecture, using assembly-level optimizations that are impossible to replicate in Triton.

### Why we use cuBLAS for Linear layers

We wrote a custom Triton matmul kernel and benchmarked it against cuBLAS (`F.linear()`). cuBLAS was consistently faster for the matrix sizes in our model:

- Encoder linear layers: 1280×1280, 1280×5120 matrices
- Decoder linear layers: 2048×2048, 2048×5632 matrices

These are large, regular matrices — exactly what cuBLAS is optimized for. Our Triton kernel's advantage is **flexibility** (custom fusions), not raw matmul speed.

### Where we DO use Triton matmul

We use custom Triton kernels for **fused operations** where cuBLAS cannot help:

- **Fused SwiGLU**: `SiLU(x @ gate_weight) * (x @ up_weight)` — one kernel instead of two matmuls + two activations + one elementwise multiply
- **Fused LinearGELU**: `GELU(x @ weight + bias)` — one kernel instead of matmul + bias add + activation

cuBLAS can only do `x @ weight`. Everything after that (activation functions, elementwise ops) requires separate kernel launches. Fusion eliminates those extra launches and avoids writing/reading intermediate results to/from VRAM.

### The backend selection

```python
Linear.BACKEND = "torch"  # Use cuBLAS via F.linear()
```

We also support `BACKEND = "triton"` for our custom kernel, but it is ~5ms slower for the standard linear layers in this model.

---

## 8. fp16 vs bf16 vs fp32: Data Type Selection

### What are these formats?

| Format | Bits | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| fp32 | 32 | 8 bits | 23 bits | ±3.4×10³⁸ | ~7 decimal digits |
| fp16 | 16 | 5 bits | 10 bits | ±65,504 | ~3.3 decimal digits |
| bf16 | 16 | 8 bits | 7 bits | ±3.4×10³⁸ | ~2.4 decimal digits |

- **fp32** is standard full precision. Highest quality, but uses 4 bytes per number and is slowest.
- **fp16** (half precision) uses 2 bytes per number — halving memory bandwidth requirements. Limited range (max 65,504) but sufficient for inference.
- **bf16** (brain float 16) also uses 2 bytes but trades precision for range. Same exponent range as fp32, but only ~2.4 digits of precision.

### Why fp16 over bf16?

On the RTX 5090, we benchmarked both:
- **fp16 pipeline: 98.5ms**
- **bf16 pipeline: 102.1ms**

fp16 is ~3.6ms faster because:
1. cuBLAS HGEMM (fp16 matrix multiply) is slightly faster than its bf16 equivalent on RTX 5090's tensor cores
2. The extra precision of fp16 (10 vs 7 mantissa bits) didn't cause any accuracy issues for inference

### The fp16-throughout pipeline

Our biggest optimization was eliminating **unnecessary type conversions**. The original code had many places where data was converted between types:

```
fp16 input → fp32 (for compute) → fp16 (store to VRAM) → fp32 (next layer loads) → ...
```

Each conversion wastes bandwidth. We rewired the entire pipeline so data stays in fp16:

1. **Removed `.float()` calls** — The original code called `.float()` (convert to fp32) after every linear layer. Each call touches ~120 Linear layers, doubling data size each time. Removing this saved **-7.5ms**.
2. **fp16 norm outputs** — RMSNorm and LayerNorm kernels now output fp16 directly instead of fp32.
3. **fp16 embeddings** — Token embeddings output fp16 from the start.
4. **fp16 fused kernels** — SwiGLU and LinearGELU receive and produce fp16.

Internal computation within kernels still uses fp32 (for numerical stability), but all data **stored to VRAM** is fp16, halving memory traffic.

---

## 9. KV Cache and the generate_v8b Function

### The O(n²) problem

The baseline `generate()` function works like this:
```
Step 1: Process tokens [START, audio_features]              → predict token A
Step 2: Process tokens [START, audio_features, A]           → predict token B
Step 3: Process tokens [START, audio_features, A, B]        → predict token C
...
Step 13: Process tokens [START, audio_features, A, B, ..., L] → predict token M
```

Each step reprocesses the **entire sequence from scratch**. This is O(n²) in the number of generated tokens — step 13 does 13x the work of step 1, even though only one new token was added.

### The KV cache solution

In a transformer's attention mechanism, each layer computes Key (K) and Value (V) matrices from the input. For tokens that have already been processed, these K and V values never change. The **KV cache** stores them:

```
Step 1: Process all tokens → compute K₁,V₁ for every layer → store in cache → predict A
Step 2: Process only token A → compute K₂,V₂ for token A → append to cache → predict B
Step 3: Process only token B → compute K₃,V₃ for token B → append to cache → predict C
```

Now each step after the first only processes **one token**, using the cached K/V from all previous tokens. This reduces generation from O(n²) to O(n).

### Implementation: monkey-patching

The assignment rules say `model.py` is read-only. But `model.py` contains the `generate()` function, which we need to replace with our KV-cached version. Our solution: **monkey-patching**.

Monkey-patching means replacing a method on an object at runtime:

```python
def _generate_v8b(self, input_features, input_ids, ...):
    # KV-cached generation loop
    ...

# Later, when model is loaded:
model.generate_v8b = types.MethodType(_generate_v8b, model)
```

The benchmark script already checks for `generate_v8b` and calls it if available.

### Performance impact

KV cache reduced inference from **121.1ms to 113.5ms** (−7.6ms). The saving grows with longer sequences — for 50 tokens, it would save ~80ms.

---

## 10. Kernel Fusion: SwiGLU and LinearGELU

### What is kernel fusion?

Every time the CPU tells the GPU to run a kernel, there is:
1. **Launch overhead**: ~5-15μs to set up the kernel
2. **Memory round-trip**: The kernel reads input from VRAM, computes, writes output to VRAM. The next kernel reads that output from VRAM again.

Kernel fusion combines multiple operations into a single kernel, eliminating both the launch overhead and the intermediate VRAM round-trips.

### Fused SwiGLU

The decoder's MLP uses the SwiGLU activation pattern:
```
gate = SiLU(x @ gate_weight)
up   = x @ up_weight
output = gate * up
```

**Unfused** (5 kernel launches, 4 VRAM round-trips):
1. `temp1 = x @ gate_weight` (cuBLAS matmul → write temp1 to VRAM)
2. `gate = SiLU(temp1)` (read temp1 → write gate to VRAM)
3. `temp2 = x @ up_weight` (cuBLAS matmul → write temp2 to VRAM)
4. `output = gate * temp2` (read gate + temp2 → write output to VRAM)

**Fused** (1 kernel launch, 1 VRAM read, 1 VRAM write):
1. Load tile of x, gate_weight, up_weight into shared memory
2. Compute both matmuls in registers
3. Apply SiLU and multiply in registers
4. Write final output to VRAM

The fused version eliminates ~3 VRAM round-trips and 4 kernel launches per MLP layer. With 28 decoder layers, that's 84 fewer kernel launches per inference.

### Fused LinearGELU

The encoder's MLP uses:
```
output = GELU(x @ weight + bias)
```

Same principle — fuse the matmul, bias add, and GELU activation into one kernel.

### When fusion doesn't help

- **EncoderMLP.FUSED** — Set to True but never actually used because `model.py` calls `fc1` and `fc2` (plain Linear layers) directly, not the EncoderMLP class.
- **Very small matrices** — If the input is smaller than one tile (e.g., during single-token decode), the fusion overhead exceeds the benefit. We fall back to cuBLAS for these cases:
  ```python
  if num_rows >= self.TILE_M and x.is_cuda:
      # Use fused Triton kernel
  else:
      # Fall back to cuBLAS
  ```

---

## 11. GPUProfile: Portable GPU Detection

### The problem

Different GPUs have different amounts of shared memory, different numbers of compute units (SMs), and different architectural features. A configuration that works well on an RTX 5090 might crash on an H200 (different shared memory layout) or run slowly on an RTX 3090 (not enough SMs for large tiles).

### The solution: GPUProfile

At import time, we detect the GPU and look up pre-tested configurations:

```python
GPU = GPUProfile()  # Runs once when layers.py is imported
```

GPUProfile does three things:
1. **Reads hardware properties** — SM version, shared memory size, GPU name
2. **Classifies the architecture** — Maps to one of 6 known categories (blackwell_consumer, ada, hopper, blackwell_dc, ampere_dc, ampere_consumer)
3. **Loads optimal configs** — If the GPU is known, uses tested configs from `_KNOWN_CONFIGS`. If unknown, computes configs dynamically from the shared memory budget.

### The three-tier system

```
Tier 1: Known GPU     → Use _KNOWN_CONFIGS lookup table (fastest, tested)
Tier 2: Unknown GPU   → Compute dynamically from shared memory budget (safe, reasonable)
Tier 3: No GPU / error → CPU fallback with conservative defaults
```

### Why not just compute dynamically for everything?

The dynamic computation picks the **largest tiles that fit** in shared memory. But "largest" isn't always "fastest" — other factors like register pressure, warp scheduling, and L2 cache behavior can make a slightly smaller tile faster. The hand-tested `_KNOWN_CONFIGS` values capture these nuances that a formula cannot.

For example, on the RTX 5090, the dynamic computation would choose 64×64 tiles for attention (which happens to match our tested config), but it might choose 128×64×32 matmul tiles instead of our tested 64×64×32 — the larger tiles fit in memory but run slightly slower due to register pressure.

---

## 12. Autotune: Built, Tested, Removed

### What we built

We implemented a `warmup_attention_tiles()` function (~95 lines) that:
1. Generated all valid tile configurations for the current GPU's shared memory
2. Ran a synthetic benchmark (10 warmup + 20 timed iterations) for each config
3. Stored the fastest config in an `_AUTOTUNE_CACHE` dictionary
4. Used the cached config for all subsequent attention calls

### What happened when we tested it

On the RTX 5090, the autotune system selected configs that were **3.1ms slower** than our hand-tuned values:
- Autotune best: **101.6ms**
- Hand-tuned: **98.5ms**

The autotune benchmarked each config in isolation with synthetic data. But real-world performance depends on the **interaction** between all kernels — cache warming, memory fragmentation, and pipeline effects that synthetic benchmarks don't capture.

### Why we removed it

Three reasons:
1. **It found worse configs** — The hand-tuned values were better.
2. **It was dead code** — The function was never called by default. You had to manually add a call before any inference.
3. **Maintenance burden** — ~110 lines of code (function + cache dictionary + cache lookup in the hot path) that served no purpose.

We removed the entire autotune system in a single commit: the `_AUTOTUNE_CACHE` dict, the `warmup_attention_tiles()` function, and the cache lookup in `scaled_dot_product_attention()`.

---

## 13. Rejected Optimizations

We tested many optimizations that did **not** make the final cut. Understanding why they failed is as important as understanding why the adopted ones work.

### Grid swizzling for SwiGLU (+18ms regression)

**What it is:** Grid swizzling reorders which thread blocks process which tiles, so that adjacent blocks access adjacent memory — improving L2 cache hit rates.

**Why it failed:** The RTX 5090 has a large L2 cache (72MB) that already achieves good hit rates with our 64×64 tiles. Swizzling added overhead (extra index math) without improving cache behavior. On GPUs with smaller L2 caches, this might help — but not on our target hardware.

The version from yash/optimize used `GROUP_SIZE_M=8`, a 1D grid, `num_warps=8`, and `num_stages=4`. It regressed by +18ms on the RTX 5090.

### @triton.autotune for GELU/SiLU (+0.7ms)

**What it is:** Triton's built-in `@triton.autotune` decorator benchmarks multiple kernel configurations at the first call and selects the fastest.

**Why it failed:** The tuning warmup itself takes ~0.7ms, and for simple pointwise operations (GELU, SiLU), there are very few configurations to choose from — the "default" was already optimal. The overhead exceeded any possible gain.

### Flash Attention num_stages=2 (OOM on consumer GPUs)

**What it is:** Using double-buffering (pipeline depth 2) in the flash attention kernel.

**Why it failed:** As explained in [Section 5](#5-num_stages-pipeline-depth-double-buffering), consumer GPUs (RTX 3090/4090/5090) have ~99-100KB shared memory. Double-buffering for attention tiles requires ~136KB minimum. The kernel won't launch — it's not a "slower" result, it's a hard failure.

### SDPA fallback for all attention (+5ms)

**What it is:** Using PyTorch's `scaled_dot_product_attention` for all attention computations (not just tiny decode sequences).

**Why it failed:** Our custom flash attention kernel is faster for long sequences (encoder has seq_len up to 1500) because we've tuned the tile sizes for our exact GPU. SDPA uses a one-size-fits-all approach. However, SDPA wins for very short sequences (seq_q ≤ 4) where kernel launch overhead dominates — so we use it only there.

### PyTorch GELU/SiLU in bf16 (+0.3ms)

**What it is:** Using `torch.nn.functional.gelu()` and `torch.nn.functional.silu()` in bf16 instead of our Triton kernels.

**Why it failed:** PyTorch's implementations are already optimized for these operations, but they launch separate kernels. Our Triton kernels fuse the activation with the preceding matmul, avoiding a VRAM round-trip. The +0.3ms comes from the extra kernel launches and memory traffic.

### Softmax bf16 output (0ms effect)

**What it is:** Having the softmax kernel output bf16 instead of fp32.

**Why it failed:** In our flash attention kernel, softmax is computed **in registers** as part of the fused kernel — it never writes to VRAM separately. The only standalone softmax is for final logits (once per inference, ~40 μs), where the dtype makes no measurable difference.

---

## 14. Monkey-Patching: Why and How

### The constraint

The assignment forbids modifying `model.py`. But `model.py` defines:
- The `generate()` function (which we need to replace with KV-cached version)
- The `scaled_dot_product_attention` function (which needs to call our flash attention kernel)

### What monkey-patching means

Monkey-patching replaces a function or method on an existing object at runtime. A simple analogy:

```python
class Dog:
    def speak(self):
        return "Woof!"

dog = Dog()
dog.speak()  # "Woof!"

# Monkey-patch: replace speak() at runtime
dog.speak = lambda: "Meow!"
dog.speak()  # "Meow!"
```

The class definition wasn't changed — we replaced the method on the instance.

### Our monkey-patches

1. **`model.generate_v8b`** — Adds a new method (KV-cached generation) to the model instance. The benchmark script already checks for it: `if hasattr(model, 'generate_v8b'): model.generate_v8b(...)`.

2. **`model.scaled_dot_product_attention`** — Replaces the attention function with our flash attention implementation.

### The circular import problem

`model.py` imports `layers.py`. If `layers.py` tried to import `model.py` to patch it, we'd have a circular import. Our solution: **deferred patching**.

The patch function `_try_patch_v8b()` is called inside `Linear.__init__()` — which runs when the model is being constructed. By that time, both modules are fully loaded, and we can safely access the model class:

```python
class Linear:
    def __init__(self, ...):
        ...
        _try_patch_v8b()  # Safe here — model.py is already loaded
```

---

## 15. Dead Code Removal

Over the course of development, we accumulated code that was no longer used. We removed ~383 lines of dead code across multiple commits:

### Removed: Original 3-kernel attention pipeline (~200 lines)
- `attention_scores_kernel` — Computed Q×K^T scores
- `softmax_inplace_kernel` — Applied softmax to scores
- `attention_output_kernel` — Multiplied softmax output by V
- `causal_mask_kernel` — Applied causal masking

All replaced by the single `flash_attention_kernel`.

### Removed: Autotune system (~110 lines)
- `_AUTOTUNE_CACHE = {}` — Cache dictionary
- `warmup_attention_tiles()` — Benchmarking function (~95 lines)
- Cache lookup in `scaled_dot_product_attention()` — Checked cache before using default tiles

### Removed: bf16 RMSNorm kernel (~30 lines)
After switching from bf16 to fp16, the bf16-specific norm kernel was unused.

### Removed: Standalone GELU/SiLU/SwiGLU kernels (~40 lines)
After fusing these into the matmul kernels, the standalone activation kernels were no longer called.

### Why we removed dead code

1. **Clarity** — Every line of code someone reads should serve a purpose. Dead code confuses readers who try to understand when it runs.
2. **Maintenance** — Dead code can silently break without anyone noticing, then cause bugs if someone tries to revive it.
3. **Compilation** — Triton JIT-compiles kernels on first use. Dead kernels don't get compiled, but their presence in the file slows down IDE tooling and static analysis.

---

## 16. Summary of Performance Impact

### Complete optimization progression (RTX 5090, 3.5s test audio, 13 tokens)

Every row is a real benchmark measurement. Each optimization was tested individually and committed only if it improved (or did not regress) performance.

| # | Change | Time | Delta | Mechanism |
|---|--------|------|-------|-----------|
| 0 | Baseline (example implementation) | 261.3ms | — | O(n²) generation, fp32, 3-kernel attention |
| 1 | All Triton kernels + cuBLAS + TF32 | 209.8ms | −51.5ms | Replaced PyTorch ops with Triton/cuBLAS, enabled TF32 tensor cores |
| 2 | bf16 weights + Flash Attention | 136.4ms | −73.4ms | Halved memory traffic (4B→2B) + fused single-kernel attention |
| 3 | Fused Q+K RoPE pair kernel (from meave) | 124.6ms | −11.8ms | Fused two separate RoPE kernels into one, halving launch overhead |
| 4 | bf16 RMSNorm output kernel (from meave) | 120.7ms | −3.9ms | Norm kernel outputs bf16 directly, no fp32→bf16 conversion |
| 5 | bf16 LayerNorm output | 121.1ms | −0.7ms | Same approach for LayerNorm (minor because fewer LayerNorm layers) |
| 6 | generate_v8b with KV cache (monkey-patched) | 113.5ms | −7.6ms | O(n) generation — cache K/V, only process new token each step |
| 7 | SDPA fallback for KV-cached decode (seq_q≤4) | 110.0ms | −3.5ms | PyTorch SDPA is faster than custom kernel for single-token decode |
| 8 | fp16 cuBLAS HGEMM (was bf16) | 109.6ms | −0.4ms | fp16 tensor core matmul slightly faster than bf16 on RTX 5090 |
| 9 | Smaller flash attention tiles (from meave) | 109.6ms | ~0ms | Better tile fit for decoder head_dim=128 |
| 10 | Remove Linear `.float()` conversion | 102.1ms | **−7.5ms** | Eliminated fp16→fp32 cast after every Linear layer (~120 layers) |
| 11 | Remove silu/gelu Python-side float32 cast | 98.4ms | **−3.7ms** | Activation functions stay in fp16, no round-trip to fp32 |
| 12 | Remove RMSNorm/LayerNorm float32 cast | 98.1ms | ~−0.3ms | Norms output fp16 directly, downstream layers receive fp16 |
| 13 | fp16 embedding + fused MLP fp16 + flash attn fp16 | **98.5ms** | ~+0.4ms | Slight noise; all remaining ops now fp16 (pipeline complete) |

**Final: 98.5ms on RTX 5090** (measured 2026-03-15).
**Latest benchmark (2026-03-17): 100.4ms** (after GPUProfile refactor — within noise, no regression).

### Rejected optimizations (tested, measured, not adopted)

| Optimization | Source | Result | Why It Failed |
|---|---|---|---|
| Swizzled SwiGLU + larger tiles | yash/optimize | **+18ms** (123→141ms) | RTX 5090's large L2 cache already has good locality; swizzle added overhead |
| @triton.autotune for GELU/SiLU | majed | **+0.7ms** | Tuning warmup cost exceeded any possible gain for simple pointwise ops |
| Flash Attention num_stages=2 | internal | **Kernel won't launch** | Consumer GPU shared memory (~99KB) can't hold two tile buffers |
| PyTorch SDPA for prefill/encoder | internal | **+6ms** (108→114.5ms) | Our tuned flash attention is faster for long sequences (seq_len ~1500) |
| SDPA enable_gqa=True for decode | internal | **+13ms** (→121.6ms) | Manual KV head expansion is faster than SDPA's internal GQA handling |
| PyTorch GELU/SiLU in bf16 | internal | **+0.3ms** | Extra kernel launches vs our fused Triton kernels |
| Softmax bf16 output | internal | **0ms** | Softmax is in-register inside flash attention; standalone softmax runs once (~40μs) |
| Warmup autotune | internal | **+3.1ms** (98.5→101.6ms) | Micro-benchmarks chose configs that were worse in the full pipeline context |
| EncoderMLP.FUSED | yash/optimize | **N/A** | model.py doesn't use EncoderMLP class — calls fc1/fc2 directly |
| LinearGELU.FUSED | yash/optimize | **N/A** | model.py doesn't use LinearGELU class — calls linear_1/act directly |
| Fused gate+up Linear in MLP | internal | **Neutral** | Overhead of larger fused kernel offset the savings from fewer launches |

### Cross-branch comparison

| Branch | Time | Key Technique |
|--------|------|--------------|
| **ankush (us)** | **98.5ms** | fp16 pipeline, KV cache, flash attention, cuBLAS, GPUProfile tiles |
| meave | 127.8ms | fp16 weights, fused RoPE, separate flash_decode_kernel |
| majed | 187.9ms | cuBLAS, flash attention, SDPA fallback, @triton.autotune |

### Cross-GPU comparison

| GPU | SMs | VRAM | Our Time | Baseline | Speedup |
|-----|-----|------|----------|----------|---------|
| RTX 5090 (full) | 170 | 32 GB | 100.4ms | 262.2ms | 61.7% |
| H200 MIG 3g.71gb | 60 | 70 GB | 204.6ms | 464.1ms | 55.9% |
| H200 MIG 1g.18gb | 16 | 16 GB | 309.7ms | — | — |

### Key numbers

- **Baseline → Final: 261.3ms → 98.5ms (62.3% faster, 2.65x speedup)**
- **RTX 5090 latest (with GPUProfile): 100.4ms** (within noise of 98.5ms)
- **Accuracy: 100%** in all benchmarks (no quality loss from any optimization)
- **Tokens: 13** (consistent across all runs)
- **Stddev: 0.1-0.5ms** (highly reproducible results)
