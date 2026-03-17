# Comprehensive Commit-by-Commit Changelog

## GLM-ASR Triton GPU Kernel Development

### University of Edinburgh, MLS 26 Spring

---

## What This Project Is

This project implements custom GPU kernels for a speech-to-text (ASR -- Automatic Speech Recognition) model called GLM-ASR. The model takes audio input and produces text output. Under the hood, it uses a **transformer** architecture with an audio encoder (32 layers) that processes the sound and a text decoder (28 layers) that generates the transcript word by word.

The goal of this assignment is to replace the model's default operations with hand-written GPU kernels that run faster, bringing down the total inference time from a baseline of ~261 milliseconds to as low as possible, while maintaining 100% transcription accuracy.

This changelog walks through every commit in chronological order, explaining what changed and why. Every GPU programming concept is explained in plain language when it first appears.

---

## Background: How Does a GPU Differ from a CPU?

> **GPU vs CPU -- A Primer for Beginners**
>
> Your computer's CPU (Central Processing Unit) has a small number of very powerful cores -- typically 8 to 24 on a modern desktop. Each core is extremely fast and can handle complex, branching logic well.
>
> A GPU (Graphics Processing Unit), on the other hand, has *thousands* of much simpler cores. An NVIDIA RTX 5090 (used in this project) has over 21,000 cores. Each individual core is slower and simpler than a CPU core, but the sheer number of them means the GPU can do the same operation on thousands of data elements *at the same time*.
>
> Think of it this way: a CPU is like one expert chef who can make any dish quickly. A GPU is like a kitchen with 10,000 line cooks who can each flip one pancake. If you need to flip 10,000 pancakes, the GPU kitchen finishes almost instantly.
>
> Machine learning models are almost entirely made of matrix multiplications and element-wise operations -- exactly the kind of "flip 10,000 pancakes" work that GPUs excel at. That is why all modern AI runs on GPUs.

---

## Phase 1: Initial Implementation (2026-03-09)

### Commit `12daf13` -- feat: implement all 10 Triton GPU kernels for ASR model

**Date:** 2026-03-09
**Files changed:** `layers.py` (+149/-109), `attention.py`, `rope.py`

This is the foundational commit. It implements all 10 GPU kernels from scratch in a single sitting. Before this commit, the model ran using standard PyTorch operations. After this commit, each critical operation has a custom kernel.

> **What is a "kernel"?**
>
> In GPU programming, a "kernel" is a function that runs on the GPU rather than the CPU. When you "launch" a kernel, the CPU sends instructions to the GPU saying "run this function across thousands of threads simultaneously." Each kernel launch has some overhead (the time it takes for the CPU to communicate with the GPU), so fewer kernel launches is generally better.

> **What are CUDA and Triton?**
>
> CUDA is NVIDIA's programming framework for writing GPU code. It gives you very fine-grained control but requires writing in C/C++ and managing many low-level details like thread indexing, memory allocation, and synchronization.
>
> Triton is a newer alternative from OpenAI. You write GPU kernels in Python, and Triton's compiler automatically generates optimized GPU machine code. Triton handles many of the tedious parts of GPU programming (like memory access patterns and thread scheduling) that you would need to manage manually in CUDA. It is a significant productivity boost -- you get 80-95% of CUDA's performance with perhaps 20% of the effort.

Here are the 10 kernels that were implemented:

#### 1. `rmsnorm_kernel` -- Root Mean Square Normalization

RMSNorm is a normalization technique used in the text decoder (all 28 layers). It takes each row of a matrix and divides every element by the "root mean square" of that row, then scales by a learned weight.

The formula is: `y = x / sqrt(mean(x^2) + eps) * weight`

In plain English: for each row, compute the average of all elements squared, take the square root of that average, then divide every element by that value. The `eps` (epsilon) is a tiny number like 0.00001 added to prevent division by zero. The `weight` is a learned parameter that lets the model scale each dimension differently.

Why normalize at all? Neural networks learn better when the values flowing through them stay in a reasonable range. Without normalization, values can grow or shrink exponentially as they pass through dozens of layers, making training unstable. RMSNorm keeps values well-behaved.

#### 2. `layernorm_kernel` -- Layer Normalization

Layer Normalization is similar to RMSNorm but slightly more involved. It is used in the audio encoder (all 32 layers). The formula is:

`y = (x - mean) / sqrt(variance + eps) * weight + bias`

The difference from RMSNorm is that LayerNorm also subtracts the mean of each row (centering the data around zero) and adds a learned bias term. This makes it slightly more expressive but also slightly more expensive to compute, since you need two passes over the data (one for the mean, one for the variance) instead of one.

#### 3. `gelu_kernel` -- Gaussian Error Linear Unit Activation

> **What is an activation function?**
>
> In a neural network, layers alternate between linear operations (matrix multiplications) and non-linear activation functions. Without non-linear activations, stacking 100 linear layers would be mathematically equivalent to a single linear layer -- you would gain nothing from depth. Activation functions introduce non-linearity, which is what allows deep networks to learn complex patterns.

GELU is a smooth activation function used in the audio encoder. It looks roughly like a "soft" version of ReLU (which simply zeroes out negative values). GELU instead applies a Gaussian-shaped curve that mostly passes positive values through unchanged and mostly zeroes out negative values, with a smooth transition around zero.

#### 4. `silu_kernel` -- Sigmoid Linear Unit (Swish) Activation

SiLU (also called Swish) is the activation function used in the text decoder's feed-forward network (specifically in the "SwiGLU" architecture). The formula is `silu(x) = x * sigmoid(x)`, where sigmoid squashes values to the range (0, 1). The effect is similar to GELU -- it is a smooth gate that lets positive values through and attenuates negative values.

#### 5. `linear_kernel_tf32` -- Tiled Matrix Multiplication

> **What are Tiles/Blocks?**
>
> GPUs process data in fixed-size chunks called "tiles" or "blocks." Instead of multiplying two entire matrices at once (which may not fit in fast memory), the GPU breaks the computation into smaller tiles -- for example, 64x64 element chunks -- and processes each tile. The key advantage of tiling is **data reuse**: when you load a tile of data into fast on-chip memory, you can reuse it many times before discarding it.
>
> Larger tiles mean more data reuse (better efficiency) but require more fast memory. There is always a tradeoff.

> **What is Shared Memory (SRAM)?**
>
> GPUs have two kinds of memory. "Global memory" (also called HBM or DRAM) is the main memory -- on an RTX 5090, this is 32 gigabytes. It has very high bandwidth (the amount of data you can read per second is enormous) but also high latency (each individual read takes a long time to start). Think of it as a massive warehouse -- it holds everything, but walking to the warehouse and back takes time.
>
> "Shared memory" (also called SRAM) is a tiny amount of very fast memory sitting right next to the processing cores -- typically 48 to 228 kilobytes per processing block. It is roughly 100 times faster to access than global memory. Think of it as a small workbench right next to you -- you can only fit a few items on it, but you can grab them instantly.
>
> The art of GPU programming is largely about keeping data in shared memory as much as possible and minimizing trips to global memory.

> **What are Tensor Cores?**
>
> Modern NVIDIA GPUs have specialized hardware units called "tensor cores" that are purpose-built for matrix multiplication. When Triton's `tl.dot()` function is called, it maps onto these tensor cores. They can perform matrix multiplications roughly 10 times faster than using the regular floating-point arithmetic units. Tensor cores are the reason GPUs are so extraordinarily fast at AI workloads -- the hardware is literally designed for the exact operation that neural networks need most.

This kernel implements matrix multiplication -- the single most important and computationally expensive operation in the entire model. Every linear layer, every projection, every feed-forward layer boils down to multiplying two matrices together. The model has dozens of these operations per inference.

The "tf32" in the name refers to TensorFloat-32, a number format specific to NVIDIA tensor cores that provides a good balance of speed and precision.

The kernel works by:
1. Dividing the output matrix into tiles (initially 64x64)
2. Each tile accumulates its result by iterating over the inner dimension in chunks (initially size 32)
3. For each chunk, it loads tiles of the input matrix and weight matrix into shared memory
4. It calls `tl.dot()` to multiply the tiles using tensor cores
5. It writes the finished output tile back to global memory

#### 6. `softmax_kernel` -- Numerically Stable Softmax

Softmax converts a vector of arbitrary real numbers into a probability distribution (all values between 0 and 1, summing to 1). It is used at the very end of the model to decide which word is most likely next.

The "numerically stable" part is important. The naive softmax formula `exp(x_i) / sum(exp(x_j))` can overflow (produce infinity) when values are large. The stable version first subtracts the maximum value from all elements: `exp(x_i - max) / sum(exp(x_j - max))`. This does not change the result mathematically but keeps all the numbers in a safe range.

#### 7. `attention_scores_kernel` -- Compute Q @ K^T * scale

This kernel computes the first step of the attention mechanism: taking the dot product of "query" vectors with "key" vectors to get similarity scores. If a query represents "what am I looking for?" and a key represents "what information do I contain?", then the dot product tells you how relevant each key is to each query.

The `* scale` part divides by the square root of the dimension size, which prevents the dot products from becoming very large (which would cause softmax to become too "peaked" -- putting all probability on one element).

#### 8. `softmax_inplace_kernel` -- Apply Softmax to Attention Scores

Takes the raw similarity scores from the previous kernel and applies softmax to turn them into attention weights (probabilities). "In-place" means it overwrites the input with the output, saving memory.

#### 9. `attention_output_kernel` -- Compute Weighted Sum of Values

Takes the attention weights and multiplies them with the "value" vectors to produce the final output. If the weights say "pay 70% attention to position 3 and 30% attention to position 7," this kernel produces a weighted combination that is 70% of position 3's value and 30% of position 7's value.

#### 10. `compute_freqs_kernel` -- Rotary Position Embeddings (RoPE)

Precomputes cosine and sine values used for Rotary Position Embeddings. RoPE is a technique that encodes the position of each element in the sequence (which word is first, second, third, etc.) by rotating the query and key vectors. This rotation means that the dot product between a query and key naturally depends on their relative position -- nearby elements interact more strongly.

**Important Note:** The three attention kernels (7, 8, and 9 above) would later be completely removed and replaced by a single fused Flash Attention kernel in commit `f0b4868`. They are included here because they represent the initial approach.

---

### Commit `5e8b191` -- docs + tile size tuning

**Date:** 2026-03-09
**Files changed:** Multiple documentation files, layers.py

This commit made two important performance changes and added project documentation.

**Activation kernel block size: 256 to 1024.** The activation kernels (GELU, SiLU) process elements in chunks. Increasing the chunk size from 256 to 1024 means each kernel instance processes 4 times as many elements. This reduces the total number of kernel instances (and thus launch overhead) by 4x, at the cost of each instance needing more registers.

**Matrix multiplication tiles: 64x64x32 to 128x128x64.** The larger tile sizes mean each block of the matrix multiplication reuses more data from shared memory before going back to global memory. A 128x128 output tile with a 64-wide inner dimension loads twice as much data per iteration but reuses it much more effectively.

**Enabled MLP.FUSED = True.** The SwiGLU feed-forward network has two linear projections whose results are combined with an activation. "Fusing" means computing both projections and the activation in fewer kernel launches, reducing the overhead of launching separate kernels for each step.

Documentation files created: `claude.md`, `TUTORIAL.md`, `REFERENCE.md`, `CODE_EXPLAINED.md`.

---

## Phase 2: Environment Fixes + cuBLAS (2026-03-10)

### Commit `82591ff` -- fix: enable GPU execution on RTX 5090 with broken cuBLAS

**Date:** 2026-03-10
**Files changed:** layers.py, conv.py

> **What is cuBLAS?**
>
> cuBLAS stands for CUDA Basic Linear Algebra Subroutines. It is NVIDIA's official, hand-optimized library for matrix multiplication (and other linear algebra operations) on their GPUs. NVIDIA engineers spend enormous effort tuning cuBLAS for each GPU generation. For standard matrix multiplications, cuBLAS is almost always the fastest option available -- it is extremely hard to beat with a hand-written kernel because NVIDIA has tuned it to exploit every hardware detail.

**The Problem:** When the project was first run on the RTX 5090, it crashed. The issue was a version conflict: the pip-installed `nvidia-cublas` Python package (version 13.1) was shadowing the system-installed cuBLAS library (version 13.0). The two versions were incompatible, causing the matrix multiplication to fail.

**The Workaround:** Since cuBLAS was broken, this commit switched the Linear layer backend to the hand-written Triton matrix multiplication kernel. It also added a tiled convolution kernel and reduced tile sizes to fit within the RTX 5090's approximately 101KB of shared memory per block.

**Result:** 188ms latency with 100% transcription accuracy. This was the first successful GPU benchmark run.

---

### Commit `714cdc9` -- fix: revert model.py and conv.py to originals

**Date:** 2026-03-10
**Files changed:** model.py, conv.py

The assignment has a strict rule: certain files (`model.py`, `weight_loader.py`, `conv.py`) must not be modified. These are the "protected" files that define the model architecture and weight loading. All optimizations must be done within the allowed files (`layers.py`, `attention.py`, `rope.py`).

This commit reverted any accidental changes to these protected files.

---

### Commit `bdc7690` -- perf: switch to cuBLAS backend (214ms, 18% faster than baseline)

**Date:** 2026-03-10
**Files changed:** layers.py

**The Fix:** The cuBLAS version conflict was resolved by running `pip uninstall nvidia-cublas`, which removed the conflicting pip package and allowed PyTorch to use the system cuBLAS library correctly.

**The Change:** Switched `Linear.BACKEND` from `"triton"` (our hand-written matrix multiplication kernel) to `"torch"` (which calls `F.linear()`, which in turn uses cuBLAS under the hood).

**Why cuBLAS is faster:** NVIDIA has teams of engineers who spend years optimizing cuBLAS for each GPU generation. They use proprietary knowledge of the hardware's memory hierarchy, instruction scheduling, and tensor core microarchitecture. A hand-written Triton kernel, while respectable, cannot match this level of tuning for standard matrix multiplications. Throughout this entire project, cuBLAS remains faster than the custom Triton matmul kernel.

**Result:** 214ms (down from the 261ms baseline). The Triton kernels for normalization, activations, and attention are still used -- only the matrix multiplication was switched to cuBLAS.

---

### Commit `a14e2d5` -- optimize runtime path (209.8ms)

**Date:** 2026-03-10
**Files changed:** layers.py

Three runtime optimizations:

1. **Enable TF32 flags.** TensorFloat-32 (TF32) is a computation mode on NVIDIA GPUs (Ampere and newer) where tensor cores process fp32 (32-bit) inputs but internally use reduced precision for the multiplication step (19-bit mantissa, 8-bit exponent, 1-bit sign = 19 bits of mantissa vs fp32's 23 bits). This gives approximately 2x speed improvement for fp32 matrix multiplications with negligible accuracy loss for neural network inference. Setting `torch.backends.cuda.matmul.allow_tf32 = True` enables this.

2. **Enable cudnn.benchmark.** cuDNN is NVIDIA's library for neural network primitives (convolutions, etc.). Setting `torch.backends.cudnn.benchmark = True` tells cuDNN to try multiple algorithm implementations for each operation on the first call, benchmark them, and then use the fastest one for all subsequent calls. There is a small one-time cost, but subsequent calls are faster.

3. **Switch GQA to explicit KV expansion before SDPA.**

> **What is GQA (Grouped Query Attention)?**
>
> In this model, the text decoder uses 16 "query heads" but only 4 "key/value heads." This is called Grouped Query Attention -- groups of 4 query heads share the same key and value head. The benefit is reduced memory usage and computation for the keys and values (4 heads instead of 16).
>
> However, when computing attention, you need the key/value tensors to be the same size as the query tensor. "Explicit KV expansion" means duplicating each key/value head 4 times before passing to the attention function, so dimensions match up. This is done with `torch.repeat_interleave()`.

**Result:** 209.8ms (a small improvement from 214ms).

---

### Commit `9453c39` -- KV-cache + bf16 weights + GQA (128.7ms, 51% faster than baseline)

**Date:** 2026-03-10
**Files changed:** layers.py

This commit delivered the single largest performance jump in the entire project -- a 38% reduction from 209.8ms to 128.7ms. Three major changes contributed:

> **What is a KV Cache?**
>
> In a transformer text decoder, generating text works one token (word piece) at a time. To decide the next token, the model needs to compute attention over *all* previously generated tokens plus the new one.
>
> Without a cache, generating the 50th token requires recomputing keys and values for all 50 tokens -- even though tokens 1 through 49 have not changed. This is O(n^2) total work across all generation steps: step 1 processes 1 token, step 2 processes 2 tokens, ..., step 50 processes 50 tokens. The sum 1 + 2 + ... + 50 = 1275 token computations.
>
> With a KV cache, the keys and values for tokens 1 through 49 are stored in memory. When generating the 50th token, you only compute the key and value for token 50, then concatenate with the cached values. This is O(n) total work: each step processes exactly 1 new token, for a total of 50 token computations.
>
> For long sequences, this can mean a 10x or greater speedup. It is one of the most important optimizations for transformer text generation.

> **What is bf16 (bfloat16)?**
>
> Numbers in computers are stored in binary. More bits means more precision but also more memory and bandwidth.
>
> - **fp32 (float32):** 32 bits (4 bytes). Full precision. The default for most computations.
> - **fp16 (float16):** 16 bits (2 bytes). Half the memory, half the bandwidth. Has limited range (can overflow on large values).
> - **bf16 (bfloat16):** 16 bits (2 bytes). Same range as fp32 (8-bit exponent) but less precision (7-bit mantissa vs fp32's 23-bit). Developed by Google Brain specifically for deep learning -- the full range means fewer overflow issues compared to fp16.
>
> Using half-precision (fp16 or bf16) for weights halves the amount of data the GPU needs to read from memory. Since GPU computations are often bottlenecked by memory bandwidth (waiting for data to arrive from global memory), this can nearly double speed.

**Changes in this commit:**

1. **Added `generate_v8b` with KV cache** -- a custom generation function that maintains a KV cache across decoding steps, turning O(n^2) total attention computation into O(n).

2. **Enabled bfloat16 weights** -- model weights are stored as bf16 instead of fp32, halving the memory bandwidth needed to read them. Since every Linear layer reads its weight matrix from global memory, and the model has dozens of Linear layers, this has a large cumulative effect.

3. **Switched SDPA to bf16** -- PyTorch's Scaled Dot Product Attention (SDPA) has a "cuDNN Flash Attention" backend that only activates when inputs are in half-precision (fp16 or bf16). By ensuring the inputs to SDPA are bf16, this significantly faster backend is triggered.

**Result:** 128.7ms -- a dramatic improvement from 209.8ms.

---

### Commit `f38ade2` -- fix duplicate GQA bug

**Date:** 2026-03-10
**Files changed:** layers.py

A subtle bug related to Grouped Query Attention. As described earlier, the 16 query heads share 4 key/value heads, and the KV heads need to be expanded (duplicated) to match the query head count. The bug was that this expansion was happening twice in the fallback code path -- the KV heads were being duplicated from 4 to 16, and then from 16 to 64. This produced incorrect attention results.

The fix was simple: remove the duplicate expansion call.

---

### Commit `e0bea91` -- restore model.py, remove monkey-patch (113.0ms)

**Date:** 2026-03-10
**Files changed:** model.py, layers.py

> **What is Monkey-Patching?**
>
> "Monkey-patching" is a technique where you modify a class or function at runtime -- while the program is running -- without changing its original source file. For example, you could write `model.generate = my_custom_generate` to replace the model's generation method with your own version.
>
> This is powerful but fragile. If the original code changes, the monkey-patch might break silently. It also makes code harder to understand because the behavior does not match what you see in the source file.

This commit cleaned up the codebase:

1. **Restored `model.py` and `conv.py`** to match `origin/main` exactly (the protected reference versions).
2. **Removed the monkey-patched `generate_v8b`** that was being attached to the model object at runtime from within `layers.py`.

All optimizations now work within the allowed files (`layers.py`, `attention.py`, `rope.py`) without modifying any protected files, even at runtime.

**Result:** 113.0ms.

---

### Commit `f0b4868` -- fused Flash Attention kernel (109.0ms, 58.3% faster than baseline)

**Date:** 2026-03-10
**Files changed:** attention.py

**This is the single most important algorithmic commit in the entire project.** It replaces the 3-kernel attention pipeline with a single fused Flash Attention kernel. Understanding this change requires understanding the problem with the old approach.

> **What is Kernel Fusion?**
>
> "Kernel fusion" means combining multiple GPU operations into a single kernel. Why does this matter?
>
> Every time a kernel writes its results to global memory (the slow 32GB main GPU memory), and the next kernel reads those results back, you pay the cost of a round trip to slow memory. If the intermediate data is large, this can be extremely expensive.
>
> Fusion eliminates these intermediate writes and reads. Instead of:
> - Kernel A computes result, writes to global memory
> - Kernel B reads from global memory, computes, writes to global memory
> - Kernel C reads from global memory, computes final result
>
> A fused kernel does:
> - Single kernel computes everything, keeping intermediate results in fast shared memory or registers
>
> The savings can be enormous when the intermediate data is large.

> **What is Flash Attention and Online Softmax?**
>
> Standard attention computes a matrix of scores: every query position gets a score against every key position. For a sequence of length N, this produces an N x N matrix. For N = 1000, that is 1,000,000 values that need to be written to global memory, then read back for softmax, then read back again for the weighted sum.
>
> Flash Attention is an algorithm (published by Tri Dao et al.) that avoids ever creating the full N x N matrix. Instead, it processes the queries in small tiles and iterates over the keys/values in blocks. The crucial insight is "online softmax" -- you can compute softmax *incrementally* by maintaining a running maximum and running sum:
>
> 1. Process the first block of keys. Compute scores, find the current max, compute exp(scores - max), accumulate a partial sum.
> 2. Process the second block of keys. Compute new scores. If the new max is larger, rescale all the previous partial results. Update the running sum.
> 3. Continue until all blocks are processed.
>
> At no point do you need the full N x N matrix in memory. Everything stays in fast shared memory and registers. The memory usage drops from O(N^2) to O(block_size), and you eliminate two round trips to global memory.

**The Old Approach (3 Kernels):**

1. **`attention_scores_kernel`**: Computes Q @ K^T * scale. This produces the full (seq_q x seq_k) scores matrix and **writes it to global memory**. For a 1500-token audio sequence, that is 1500 x 1500 = 2.25 million float values written to slow memory.

2. **`softmax_inplace_kernel`**: **Reads the entire scores matrix back from global memory**, applies softmax (subtract max, exponentiate, divide by sum), and **writes the result back to global memory**. That is another read + write of 2.25 million values.

3. **`attention_output_kernel`**: **Reads the softmax'd weights from global memory** again, along with the V (value) matrix, computes the weighted sum, and writes the final output.

Total global memory traffic for the scores alone: 3 writes + 3 reads of the N x N matrix. This is the bottleneck.

**The New Approach (1 Fused Kernel with Online Softmax):**

The `flash_attention_kernel` does everything in a single kernel launch:

1. For each tile of queries (e.g., 64 rows at a time):
   - Initialize running max = negative infinity, running sum = 0, accumulator = 0
   - For each block of keys/values:
     - Load K block into shared memory
     - Compute scores = Q_tile @ K_block^T * scale (using tensor cores, result stays in registers)
     - Apply causal mask if needed (set future positions to negative infinity)
     - Update running max and running sum for online softmax
     - Rescale previous accumulator if the max changed
     - Load V block into shared memory
     - Accumulate: output += softmax_weights @ V_block (using tensor cores)
   - Write final output tile to global memory

The full N x N scores matrix is **never materialized**. Each small block of scores is computed, used immediately, and discarded. Everything stays in shared memory and registers.

**Feature Flags:**
- `IS_CAUSAL`: A compile-time constant that enables causal masking (preventing attention to future positions) inside the kernel. When enabled, the kernel skips entire blocks of keys that are entirely in the future, saving compute.
- `HAS_MASK`: Allows passing an external attention mask for additional masking patterns.

**17 numerical parity tests** were added to verify that the fused kernel produces results matching PyTorch's reference implementation to within floating-point tolerance.

**Critical Note:** After this commit, the three old attention kernels (`attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel`) become **dead code**. They still exist in the source files but are never called by any code path. They will be formally removed in commit `220b990`.

**Result:** 109.0ms (down from 113.0ms with the kernel alone, and 58.3% faster than the 261ms baseline).

---

### Commit `4c0dd5a` -- expand attention parity tests

**Date:** 2026-03-10
**Files changed:** Test files

Added additional test cases for Flash Attention correctness verification, covering edge cases like very short sequences, very long sequences, different head dimensions, and mixed precision inputs.

---

### Commit `91c70b3` -- restore model.py to origin/main (Session 5)

**Date:** 2026-03-12
**Files changed:** model.py

After an upstream merge, ensured all protected files (`model.py`, etc.) match `origin/main` exactly. No monkey-patching is present. The model uses the stock `generate()` function, which does **not** have a KV cache.

**Result:** 137.9ms without KV cache. The performance regression from 109.0ms to 137.9ms is entirely due to losing the KV cache optimization -- the model now recomputes all keys and values from scratch at every decode step.

---

## Phase 3: Branch Optimizations (2026-03-13)

### Commit `e277e9f` -- fused RoPE pair kernel + bf16 RMSNorm (120.7ms, 53.8% faster than baseline)

**Date:** 2026-03-13
**Files changed:** rope.py, layers.py

This commit incorporates successful optimizations from teammate branches (meave's branch) and rejects unsuccessful ones.

**Adopted: Fused RoPE Pair Kernel (-14ms)**

Previously, applying Rotary Position Embeddings required two separate kernel launches -- one for the query tensor and one for the key tensor. Since both operations read the same cosine/sine frequency table and apply the same rotation logic, they can be combined into a single kernel launch.

The `fused_rope_pair_kernel` processes both Q and K in one launch. This eliminates one kernel launch overhead and allows the frequency data to be loaded from global memory once instead of twice.

Impact: -14ms (a surprisingly large gain, indicating that the kernel launch overhead was significant).

**Adopted: bf16 RMSNorm Output (-3ms)**

The `rmsnorm_bf16_kernel` stores its output as bf16 directly, rather than as fp32 which then gets converted to bf16 before the next Linear layer. This eliminates an unnecessary data type conversion step. Since RMSNorm is called in all 28 decoder layers, eliminating one conversion per layer adds up.

Impact: -3ms.

**Tested and Rejected:**

- **SwiGLU swizzling from yash/optimize branch:** Increased latency by 18ms (a regression). Swizzling rearranges memory layout to improve access patterns, but in this case the overhead of the rearrangement outweighed the benefit.
- **`@triton.autotune` from majed's branch:** Added 0.7ms overhead. Triton's built-in autotuning mechanism tries multiple kernel configurations at runtime to find the fastest. However, the overhead of the tuning process itself was more than any configuration improvement.

**Result:** 120.7ms.

---

### Commit `fe9f33b` -- KV-cached generate_v8b + bf16 LayerNorm (113.5ms, 56.6% faster than baseline)

**Date:** 2026-03-13
**Files changed:** layers.py

Two changes: re-adding the KV cache and optimizing LayerNorm output format.

**Re-added `generate_v8b` with KV Cache (-7.6ms)**

The KV cache had been removed in commit `91c70b3` when `model.py` was restored. This commit re-implements it within `layers.py` (an allowed file) using the model's existing KV cache infrastructure.

The implementation uses a clever technique to avoid modifying `model.py`: a function called `_try_patch_v8b()` is called from `Linear.__init__()`. Since `Linear.__init__` runs when the model is being constructed, this is a natural hook point. The function attaches `generate_v8b` to the model object using monkey-patching, but only from within the allowed `layers.py` file.

The benchmark script (`benchmark_student.py`) auto-detects the presence of this method via `hasattr(model, 'generate_v8b')` and uses it if available.

**bf16 LayerNorm Output (-0.7ms)**

Matching the approach taken for RMSNorm in the previous commit, the LayerNorm kernel now outputs bf16 directly instead of fp32-then-convert. This eliminates one dtype conversion per encoder layer (32 layers).

**Result:** 113.5ms.

---

### Commit `c00f9f9` -- Runtime GPU detection + docs

**Date:** 2026-03-13
**Files changed:** layers.py, documentation files

Added a `_detect_gpu_tier()` function that runs at import time to determine whether the code is running on a consumer GPU (like the RTX 5090) or a datacenter GPU (like the H200). Different GPUs have different amounts of shared memory, different numbers of streaming multiprocessors, and different performance characteristics, so the optimal kernel configurations differ.

Also documents the autotune failure analysis -- explaining why `@triton.autotune` was tested and rejected.

---

### Commit `0f0ce3b` -- switch generate_v8b to decode(use_cache=True)

**Date:** 2026-03-13
**Files changed:** layers.py

Per instructor guidance posted on Piazza (the course discussion forum):

Changed `generate_v8b` from calling `model.forward_with_kv_buffers()` (a lower-level internal method) to calling `model.decode(use_cache=True)` (a cleaner public API method).

Both approaches use the KV cache and produce the same performance, but `model.decode(use_cache=True)` is the officially recommended interface. This is a code quality improvement with no performance impact.

---

## Phase 4: Systematic Testing + Dead Code Cleanup (2026-03-15)

### Commit `0410b3b` -- SDPA fallback for KV-cached decode (110ms, 57.9% faster than baseline)

**Date:** 2026-03-15
**Files changed:** attention.py

A nuanced optimization that chooses different attention implementations based on the problem size.

**The Insight:** For single-token decode steps (when the KV cache is being used and only one new token is being processed), the query sequence length is just 1. The full Flash Attention kernel, while excellent for long sequences, has a fixed overhead from Triton kernel launch time (approximately 1 millisecond). When the actual computation is tiny (1 query token attending to perhaps 50 key tokens), this launch overhead dominates the total time.

**The Fix:** For small query sequence lengths (seq_q <= 4), fall back to PyTorch's built-in SDPA (Scaled Dot Product Attention). PyTorch's SDPA uses cuDNN internally, which has essentially zero additional launch overhead since it is already loaded and warmed up.

Impact: -3ms.

**Also Tested and Rejected:**
- `num_stages=2`: Caused an out-of-memory error (see concept explanation below).
- `num_warps=8`: No measurable change.
- softmax in bf16: No measurable change.

> **What is `num_stages`?**
>
> `num_stages` controls pipeline stages in Triton kernels. Modern GPUs can overlap computation with memory access -- while the arithmetic units are crunching numbers on one tile of data, the memory system can be prefetching the next tile. Each "stage" represents one step of this pipeline.
>
> More stages means the GPU can prefetch more data in advance, which helps hide memory latency. However, each stage requires its own buffer in shared memory to hold the prefetched data. With limited shared memory (about 101KB on the RTX 5090), increasing stages can cause the kernel to exceed the memory budget and fail to launch.
>
> `num_stages=2` was tried here but caused an out-of-memory error because the Flash Attention kernel already uses a lot of shared memory for its tiles, and adding another pipeline stage pushed it over the limit.

> **What is `num_warps`?**
>
> A "warp" is a group of 32 GPU threads that execute in lockstep -- they all run the same instruction at the same time, just on different data elements. A kernel block can contain multiple warps.
>
> More warps means more threads working in parallel within each block. This can help if there is enough work to keep all threads busy. However, more warps also means more register usage and potentially more contention for shared memory.
>
> `num_warps=8` was tested but showed no improvement -- the kernel was likely already compute-saturated at the default warp count.

**Result:** 110.0ms.

---

### Commit `220b990` -- GPU portability + dead code cleanup (110ms)

**Date:** 2026-03-15
**Files changed:** attention.py, layers.py

**This is the major dead code cleanup commit. Approximately 319 lines of code were removed across 2 files.**

#### What Was Removed from `attention.py` (~172 lines)

**1. `attention_scores_kernel` (~55 lines)**

This kernel computed `Q @ K^T * scale` for one query position at a time. It was launched with a grid of `(batch * num_heads, seq_q)` -- meaning one kernel instance per query position per attention head per batch element. Each instance computed one row of the scores matrix and wrote it to a pre-allocated output tensor in global memory.

**Why it was dead:** The `flash_attention_kernel` (added in commit `f0b4868`) computes these scores internally as part of its tiled loop. The scores are computed in shared memory, used immediately for the softmax and value weighting, and never written to global memory. There is no code path that calls `attention_scores_kernel` after commit `f0b4868`.

**2. `softmax_inplace_kernel` (~15 lines)**

This kernel read the full (seq_q x seq_k) scores matrix from global memory, applied the numerically stable softmax algorithm (find max, subtract max, exponentiate, divide by sum), and wrote the result back to the same memory location.

**Why it was dead:** The `flash_attention_kernel` uses online softmax -- it maintains a running maximum and running sum as it iterates over key blocks. It never needs the full scores matrix to exist in memory at once. The online approach is mathematically equivalent but vastly more memory-efficient.

**3. `attention_output_kernel` (~45 lines)**

This kernel read the softmax'd attention weights and the V (value) matrix from global memory, and for each query position, computed the weighted sum of values. It was essentially a matrix multiplication between the attention weight matrix and V.

**Why it was dead:** The `flash_attention_kernel` accumulates `P @ V` (where P is the softmax'd attention block) in registers as part of its tiled loop. The accumulation happens immediately after computing each block's attention weights, while the data is still in shared memory. There is no separate step.

**4. `causal_mask_kernel` (~30 lines)**

This kernel applied causal masking to the attention scores matrix. "Causal masking" means that position i can only attend to positions 0 through i (not future positions). The kernel set all entries where the key position is greater than the query position to negative infinity, so that softmax would assign them zero probability.

**Why it was dead:** The `flash_attention_kernel` has an `IS_CAUSAL` compile-time constant (a `tl.constexpr` flag). When enabled, the kernel applies the causal mask internally by setting future positions to negative infinity within the tiled computation. Moreover, it uses range clamping to *skip entire blocks of keys that are entirely in the future*, avoiding even the cost of computing scores that would be masked out. This is strictly better than a separate masking kernel.

#### What Was Removed from `layers.py` (~147 lines)

**5. `attention_scores_kernel` (~55 lines) -- DUPLICATE**

This was an exact copy of the `attention_scores_kernel` that also existed in `attention.py`. It was never called from `layers.py` -- it appears to have been copied during an earlier development phase and was simply never cleaned up.

**6. `attention_output_kernel` (~45 lines) -- DUPLICATE**

Same situation. An exact copy of the kernel from `attention.py`, never called from `layers.py`.

**7. `causal_mask_kernel` (~30 lines) -- DUPLICATE**

Same situation. An exact copy, never called.

#### Why These Were Safe to Remove

All attention in the model flows through the `scaled_dot_product_attention()` function. After commit `f0b4868`, that function exclusively uses either:
- `flash_attention_kernel` (for long sequences during encoding), or
- PyTorch SDPA (for short sequences during cached decode, added in commit `0410b3b`)

No code path calls any of the seven removed kernels. Removing them has zero impact on behavior or performance. It simply reduces the source file size and eliminates confusion about which kernels are actually active.

#### Other Changes in This Commit

- **GPU-adaptive Linear tile sizes:** Instead of hard-coding tile dimensions, the matrix multiplication kernel now selects tile sizes based on the detected GPU tier. Consumer GPUs (like RTX 5090, with ~101KB shared memory per block) use smaller tiles (64x64x32). Datacenter GPUs (with more shared memory) use larger tiles (128x128x64).
- **GPU-adaptive RoPE parameters:** The `num_stages` and `num_warps` for the RoPE kernel are also selected based on GPU tier.

**Result:** 110.0ms (unchanged -- dead code removal does not affect performance).

---

## Phase 5: fp16 Pipeline (2026-03-15)

### Commit `5c25921` -- fp16-throughout pipeline (98.5ms, 62.3% faster than baseline)

**Date:** 2026-03-15
**Files changed:** layers.py, attention.py

**This is the single most impactful optimization session**, bringing the time below 100ms for the first time. The core insight is subtle but powerful.

> **What are fp32, fp16, and bf16?**
>
> These are number formats with different precision:
>
> - **fp32 (float32):** 32 bits = 4 bytes per number. High precision (about 7 decimal digits). This is the default for most computations.
> - **fp16 (float16):** 16 bits = 2 bytes per number. Lower precision (about 3.3 decimal digits) and limited range (max value ~65504). Faster to process and uses half the memory bandwidth.
> - **bf16 (bfloat16):** 16 bits = 2 bytes per number. Same range as fp32 but lower precision (about 3.3 decimal digits). A compromise format designed for deep learning.
>
> The crucial insight for GPU performance: **memory bandwidth is often the bottleneck**, not arithmetic speed. Modern GPUs can compute far faster than they can read data from memory. Using fp16 instead of fp32 means the GPU needs to read only 2 bytes per number instead of 4 -- effectively doubling the useful memory bandwidth.

**The Key Realization:**

Triton kernels already convert data to fp32 *inside the kernel* after loading it. For example, a normalization kernel does:

```
x = tl.load(X_ptr + offsets).to(tl.float32)  # Load and convert to fp32
# ... do computation in fp32 ...
output = result.to(tl.float16)  # Convert back and store
tl.store(OUT_ptr + offsets, output)
```

This means that Python-side code like `x = x.float()` (converting to fp32 *before* calling the kernel) is completely redundant. The data gets converted to fp32 *twice* -- once in Python (which triggers a GPU dtype conversion kernel) and once inside the Triton kernel (which is free since it is part of the load operation). The Python-side conversion is pure waste.

**Changes Made (in order of impact):**

1. **`Linear._HALF_DTYPE = torch.float16`** -- Switched from bf16 to fp16 for the half-precision format. On the RTX 5090, fp16 HGEMM (Half-precision General Matrix Multiplication) is slightly faster than bf16 HGEMM. This is hardware-specific -- on some GPUs, bf16 is faster.

2. **Removed `.float()` from `Linear._forward_torch()` output (-7.5ms)** -- This was the single biggest win. The Linear layer was converting its output from fp16 back to fp32 before returning. This conversion alone cost 7.5ms across all the linear layers in the model. Since the next operation (normalization or activation) would convert to fp32 internally anyway, this conversion was completely redundant.

3. **Removed `x = x.float()` from `silu()` and `gelu()` wrapper functions (-3.7ms)** -- Same principle. The activation kernels convert to fp32 internally, so the Python-side conversion is wasted work.

4. **Removed `x = x.to(torch.float32)` from RMSNorm/LayerNorm calls (-0.5ms)** -- Again, the normalization kernels handle the conversion internally.

5. **Removed `.float()` from Flash Attention Q/K/V (-1ms)** -- The Flash Attention kernel also does internal fp32 conversion.

6. **fp16 embedding output** -- The embedding lookup (which maps token IDs to vectors) now outputs fp16 directly.

7. **fp16 fused SwiGLU allocations** -- Intermediate tensors in the fused SwiGLU computation are allocated as fp16.

8. **Norm kernel output: `tl.float16` instead of `tl.bfloat16`** -- Matching the pipeline-wide switch from bf16 to fp16.

9. **Smaller Flash Attention tiles from meave's branch** -- 64x64 for encoder attention, 32x32 for decoder attention. Smaller tiles reduce shared memory usage and can improve occupancy (the fraction of GPU resources that are actively being used).

10. **`BLOCK_M=16` for seq_q <= 16** -- When the number of query tokens is very small (as in cached decode), using a tiny tile size of 16 reduces wasted computation (no padding needed).

**Result:** 98.5ms -- a breakthrough below 100ms, and 62.3% faster than the baseline.

**Competition standings at this point:** ankush 98.5ms, meave 127.8ms, yash 128ms, majed 187.9ms.

---

## Phase 6: GPU Portability -- GPUProfile (2026-03-16)

### Commit `e496204` -- GPUProfile + dynamic tiles + docs

**Date:** 2026-03-16
**Files changed:** layers.py (+353/-62), attention.py, rope.py

The simple `_detect_gpu_tier()` function (which just classified GPUs as "consumer" or "datacenter") was replaced with a comprehensive `GPUProfile` class.

**Why This Was Needed:**

Different GPUs have vastly different amounts of shared memory, different numbers of streaming multiprocessors, and different memory bandwidths. The optimal tile sizes for matrix multiplication, attention, and RoPE kernels depend on these hardware details. Hardcoding tile sizes that work well on the RTX 5090 might crash or run slowly on an H200 or A100.

**The `GPUProfile` Class:**

- **`_KNOWN_CONFIGS` table:** Contains pre-tested, hand-optimized tile sizes for 6 known GPU architectures. Each entry specifies the ideal tile sizes for linear (matmul), attention, and RoPE kernels, along with the optimal `num_warps` and `num_stages` for each.

- **Dynamic tile computation for unknown GPUs:** If the code runs on a GPU not in the known table, it queries the GPU's shared memory budget and computes safe tile sizes dynamically. The formula works backwards from the shared memory limit: given S bytes of shared memory and a tile that uses `BLOCK_M * BLOCK_K * 2` bytes (for fp16), solve for the largest power-of-2 block dimensions that fit.

- **Runtime detection at import time:** When `layers.py` is first imported, it queries the GPU properties and instantiates a global `GPUProfile` object. All kernel launches read their configuration from this object.

**Result:** 98.8ms -- effectively no regression from the previous 98.5ms. The tiny 0.3ms difference is within measurement noise.

---

### Commit `8611863` -- Remove warmup autotune (dead code)

**Date:** 2026-03-16
**Files changed:** attention.py

**Approximately 108 lines of dead code removed.**

#### What Was Removed

**1. `_AUTOTUNE_CACHE` dictionary**

A module-level dictionary that was supposed to store the fastest tile configuration for each unique attention shape (sequence lengths, head dimensions, etc.).

**2. `warmup_attention_tiles()` function (~95 lines)**

This function was designed to perform empirical autotuning: given a specific attention shape, it would:

1. Generate all candidate `(BLOCK_M, BLOCK_N)` tile combinations that fit in shared memory
2. For each candidate, create random tensors of the appropriate shape
3. Launch the Flash Attention kernel with those tile sizes
4. Measure the execution time using CUDA events
5. Store the fastest configuration in `_AUTOTUNE_CACHE`

The idea is sound in principle -- different shapes might benefit from different tile sizes, and benchmarking on real hardware gives the most accurate results.

**3. Cache lookup code in `scaled_dot_product_attention()` (~5 lines)**

Before choosing tile sizes, this code checked `_AUTOTUNE_CACHE` for a pre-benchmarked configuration. If found, it used the cached config. If not, it fell back to `GPU.get_attention_tiles()`.

#### Why It Was Removed

When actually tested, the autotuned configurations were *worse* than the statically chosen ones: 101.6ms vs 98.5ms. This happened because:

1. **Micro-benchmarks on random data do not reflect real performance.** The autotuning benchmarked kernels on random tensors, but real model data has different value distributions, different sparsity patterns, and different memory access patterns due to caching effects from surrounding operations.

2. **The `GPUProfile._KNOWN_CONFIGS` table already handles all cases.** The known configs were tuned by actually running the full model end-to-end, which captures all the real-world effects that micro-benchmarks miss.

3. **The function was never called.** No code path in the final version invoked `warmup_attention_tiles()`. It was dead code that added complexity without benefit.

**Result:** No performance change (the code was not being executed).

---

## Phase 7: H200 Cluster Compatibility (2026-03-16)

### Commit `3791e21` -- handle numpy array input_features

**Date:** 2026-03-16
**Files changed:** layers.py

The teaching cluster uses NVIDIA H200 GPUs, which are datacenter-class GPUs with a different software environment than the local RTX 5090 development machine.

**The Problem:** The `_generate_v8b` function expected `input_features` to be a PyTorch CUDA tensor (already on the GPU). However, the H200 cluster's benchmark harness passes numpy arrays (CPU-based arrays from the NumPy library).

**The Fix:** Added conversion code at the beginning of `_generate_v8b` that checks if the input is a numpy array and, if so, converts it to a PyTorch tensor and moves it to the GPU.

---

### Commit `7d336d7` -- robust shared memory detection

**Date:** 2026-03-16
**Files changed:** layers.py

**The Problem:** The `GPUProfile` class queries the GPU's shared memory size to determine optimal tile sizes. Different versions of PyTorch expose this information under different property names:

- `shared_memory_per_block_optin` (newer PyTorch versions -- returns the maximum shared memory when opt-in dynamic shared memory is used)
- `max_shared_memory_per_block` (standard property)
- `shared_memory_per_block` (older fallback)

The H200 cluster's PyTorch version might not have the `optin` property.

**The Fix:** Used a `getattr` fallback chain that tries each property name in order, falling back to the next if the current one does not exist. This ensures the code works across all PyTorch versions.

---

### Commit `8f5e3d4` -- torch.as_tensor instead of from_numpy

**Date:** 2026-03-16
**Files changed:** layers.py

**The Problem:** `torch.from_numpy()` was failing on the H200 cluster with the error message `"expected np.ndarray (got ndarray)"`. This bizarre error was caused by a numpy version mismatch -- the cluster had a different numpy version than the development machine, and the internal type checking in `torch.from_numpy()` was failing because the `ndarray` class was not being recognized as `np.ndarray`.

**The Fix:** Replaced `torch.from_numpy()` with `torch.as_tensor()`. The `as_tensor()` function is more permissive -- it accepts any "array-like" input (numpy arrays, Python lists, other tensors, etc.) and converts it to a PyTorch tensor. It avoids the strict type check that was causing the failure.

---

### Commit `288ad9c` -- Defensive input conversion + teaching cluster benchmark

**Date:** 2026-03-16
**Files changed:** layers.py

**Added `_to_torch_tensor()` helper function.** This is a robust utility that handles any input type:
- If the input is already a CUDA tensor, return it as-is (no cost)
- If it is a CPU tensor, move it to CUDA
- If it is a numpy array or any other array-like, convert it to a tensor and move to CUDA
- Handles dtype conversion if needed

This single helper replaces the ad-hoc conversion code that was growing in `_generate_v8b`, making the code cleaner and more maintainable.

**Teaching cluster benchmark results:** Verified on H200 MIG (Multi-Instance GPU -- a feature that partitions one physical GPU into multiple smaller virtual GPUs): 204.6ms with 100% accuracy.

The 204.6ms time is much slower than the RTX 5090's 98.5ms, but this is expected: the H200 MIG partition (`3g.71gb`) is only a fraction of the full H200 GPU. The kernel configurations were optimized for the RTX 5090, and the H200 has different optimal settings.

---

### Commit `25b1fd9` -- Add comprehensive benchmark results

**Date:** 2026-03-16
**Files changed:** Documentation/benchmark files

Added raw benchmark output and analysis from the H200 MIG cluster, including 5 student benchmark runs, 5 detailed benchmark runs, and baseline comparison runs. This provides a complete record of the model's performance on the teaching cluster hardware.

---

## Summary of All Dead Code Removed

Throughout this project, approximately 383 lines of dead code were identified and removed across two cleanup commits. The following table provides a complete accounting:

| Commit | What Was Removed | Lines | Why It Was Dead |
|--------|-----------------|:-----:|----------------|
| `220b990` | `attention_scores_kernel` (attention.py) | ~55 | Replaced by `flash_attention_kernel` |
| `220b990` | `softmax_inplace_kernel` (attention.py) | ~15 | Replaced by online softmax in flash kernel |
| `220b990` | `attention_output_kernel` (attention.py) | ~45 | Replaced by tiled P@V accumulation in flash kernel |
| `220b990` | `causal_mask_kernel` (attention.py) | ~30 | Replaced by `IS_CAUSAL` flag in flash kernel |
| `220b990` | `attention_scores_kernel` (layers.py) | ~55 | Duplicate of attention.py version, never called |
| `220b990` | `attention_output_kernel` (layers.py) | ~45 | Duplicate of attention.py version, never called |
| `220b990` | `causal_mask_kernel` (layers.py) | ~30 | Duplicate of attention.py version, never called |
| `8611863` | `warmup_attention_tiles()` (attention.py) | ~95 | Found worse configs when tested, never called |
| `8611863` | `_AUTOTUNE_CACHE` dict + lookup (attention.py) | ~13 | Part of removed autotune system |
| **Total** | | **~383** | |

The dead code fell into two categories:

1. **Superseded by Flash Attention (275 lines):** Seven kernels that implemented the old 3-step attention pipeline. The fused Flash Attention kernel is strictly superior -- it uses a single kernel launch instead of three or four, keeps all intermediate data in shared memory instead of global memory, and reduces memory complexity from O(N^2) to O(block_size).

2. **Failed experiment (108 lines):** The warmup autotuning system that attempted to empirically find optimal tile sizes. It produced worse results than hand-tuned configurations because micro-benchmarks on random data do not reflect real-world performance.

---

## Performance Timeline

The following table tracks the model's inference latency across all significant commits:

| Date | Commit | Time | Speedup vs Baseline | Key Change |
|------|--------|-----:|:-------------------:|-----------|
| 2026-03-09 | `12daf13` | CPU only | -- | All 10 kernels implemented |
| 2026-03-10 | `82591ff` | 188ms | 28% faster | First GPU benchmark (Triton matmul) |
| 2026-03-10 | `bdc7690` | 214ms | 18% faster | cuBLAS backend (note: different baseline) |
| 2026-03-10 | `a14e2d5` | 209.8ms | 19.6% faster | TF32 + cudnn.benchmark |
| 2026-03-10 | `9453c39` | 128.7ms | 50.7% faster | KV cache + bf16 weights |
| 2026-03-10 | `f0b4868` | 109.0ms | 58.3% faster | Fused Flash Attention |
| 2026-03-12 | `91c70b3` | 137.9ms | 47.2% faster | Reverted to stock generate (no KV cache) |
| 2026-03-13 | `e277e9f` | 120.7ms | 53.8% faster | Fused RoPE + bf16 RMSNorm |
| 2026-03-13 | `fe9f33b` | 113.5ms | 56.6% faster | KV cache re-added + bf16 LayerNorm |
| 2026-03-15 | `0410b3b` | 110.0ms | 57.9% faster | SDPA fallback for decode |
| 2026-03-15 | `220b990` | 110.0ms | 57.9% faster | Dead code cleanup (no perf change) |
| 2026-03-15 | `5c25921` | 98.5ms | 62.3% faster | fp16-throughout pipeline |
| 2026-03-16 | `e496204` | 98.8ms | 62.1% faster | GPUProfile (no regression) |
| 2026-03-16 | `288ad9c` | 204.6ms | -- | H200 MIG (different, partitioned GPU) |

**Overall Result on RTX 5090:** From 261ms baseline to 98.5ms -- a 62.3% reduction in inference time, or equivalently the optimized model runs 2.65x faster than the baseline, while maintaining 100% transcription accuracy.

---

## Key Lessons and Themes

### Memory Bandwidth is King

The single most consistent theme across all optimizations is that **reducing the amount of data moved between the GPU's global memory and its compute units** is the most effective way to improve performance. This manifested in multiple ways:

- **KV cache** eliminated redundant recomputation (and thus redundant memory reads of the same weights)
- **bf16/fp16** halved the bytes per number, effectively doubling useful bandwidth
- **Flash Attention** eliminated the massive N x N intermediate matrix in global memory
- **Removing redundant dtype conversions** eliminated extra memory traffic from unnecessary fp16-to-fp32 conversions
- **Kernel fusion** (fused RoPE, fused SwiGLU) eliminated intermediate writes to global memory

### cuBLAS Wins for Standard Operations

The hand-written Triton matrix multiplication kernel was respectable, but NVIDIA's cuBLAS library was consistently faster for standard matrix multiplications. The lesson: do not rewrite what the hardware vendor has already spent years optimizing. Focus custom kernels on operations where you can do something cuBLAS cannot (like fusing attention, or combining normalization with dtype conversion).

### Profile Before Optimizing

Several attempted optimizations were rejected because they actually made things worse:
- SwiGLU swizzling: +18ms regression
- `@triton.autotune`: +0.7ms overhead
- Warmup autotuning: +3.1ms vs static configs

Without measuring, these might have been kept under the assumption that "more optimization = better." Real measurement proved otherwise.

### Portability Requires Defensive Programming

The H200 cluster exposed multiple assumptions that were valid on the local RTX 5090 but failed elsewhere:
- numpy arrays instead of torch tensors
- Different PyTorch versions with different APIs
- Different numpy versions with type-checking incompatibilities
- Different shared memory sizes requiring different tile configurations

The `GPUProfile` system and `_to_torch_tensor()` helper were direct responses to these portability issues.
