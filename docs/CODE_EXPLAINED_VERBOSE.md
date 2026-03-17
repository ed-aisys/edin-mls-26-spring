# GLM-ASR Codebase: Complete Code Explanation (Verbose Edition)

A detailed explanation of every component in the GLM-ASR Triton implementation,
written for readers with **no GPU programming background**.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [layers.py -- GPU Compute Kernels](#2-layerspy--gpu-compute-kernels)
3. [attention.py -- Attention Mechanism](#3-attentionpy--attention-mechanism)
4. [rope.py -- Positional Encodings](#4-ropepy--positional-encodings)
5. [model.py -- Full Model Pipeline](#5-modelpy--full-model-pipeline)
6. [conv.py -- Audio Feature Extraction](#6-convpy--audio-feature-extraction)
7. [weight_loader.py -- Model Weights](#7-weight_loaderpy--model-weights)
8. [benchmark_student.py -- Testing](#8-benchmark_studentpy--testing)
9. [How It All Fits Together](#9-how-it-all-fits-together)
10. [Optimization Sources](#10-optimization-sources)

---

## 1. Architecture Overview

### What This Project Is

GLM-ASR is a **multi-modal speech-to-text model** -- it takes audio (a recording
of someone speaking) and produces a text transcription of what they said.

> **Analogy:** Think of it like a human transcriptionist. They hear sounds (audio
> waveform), recognize patterns in the sound (the encoder), translate those patterns
> into language concepts (the projector), and then write out the words one at a time
> (the decoder).

The pipeline looks like this:

```
[Audio Waveform] -> [Mel Spectrogram] -> [Audio Encoder] -> [Projector] -> [Text Decoder] -> [Text]
```

Here is what each stage does in plain English:

1. **Audio Waveform**: The raw audio file -- just a list of numbers representing
   air pressure at 16,000 samples per second.
2. **Mel Spectrogram**: A "picture" of the audio. Instead of raw samples, it shows
   how much energy is at each frequency over time -- like a music visualizer. It has
   128 frequency bins (rows) and roughly 350 time frames (columns) for a ~3.5 second clip.
3. **Audio Encoder**: A stack of 32 transformer layers that reads the spectrogram and
   builds a rich understanding of the speech -- recognizing phonemes, words, and context.
4. **Projector**: A bridge that converts the encoder's 1280-dimensional representation
   down to the decoder's 2048-dimensional space.
5. **Text Decoder**: Another stack of transformer layers (28 this time) that generates
   text one token at a time, like autocomplete on a phone keyboard.

### Why Triton?

To understand why we use Triton, you need to understand how a GPU runs code.

**Background: How GPUs Execute Programs**

A GPU is a chip with thousands of tiny processors ("cores") that can all work
simultaneously. When you want the GPU to do something, you launch a **kernel** --
a small program that runs on many cores in parallel. For example, to add two vectors
of 1 million elements, you launch a kernel where each core adds one pair of elements.

PyTorch (the standard deep learning library) uses operations like `torch.matmul`
that internally call NVIDIA's cuBLAS and cuDNN libraries. These are extremely
optimized, but they have a fundamental limitation:

> **Analogy:** Imagine you need to wash dishes, dry them, and put them away. The
> PyTorch way is like having three separate workers: one washes ALL the dishes,
> stacks them on the counter (slow -- the counter is "far away"), then another
> worker picks them ALL up from the counter and dries them, stacks them again,
> and a third worker puts them ALL away. Each handoff through the counter (GPU
> main memory, called DRAM) is slow. The Triton way is like having one worker
> who washes, dries, and puts away each dish in sequence -- the dish never leaves
> their hands (stays in fast on-chip memory called SRAM/registers).

Specifically, Triton helps with three things:

1. **Kernel fusion**: Each separate PyTorch operation launches its own kernel.
   Between kernels, results must be written to the GPU's main memory (DRAM, which
   is slow -- think ~1 terabyte/second bandwidth, but that sounds fast until you
   realize you may need to move gigabytes of data). Triton lets you combine multiple
   operations into a single kernel, so intermediate results stay in the fast on-chip
   memory (SRAM, ~19 terabytes/second -- roughly 19x faster). For a concrete example:
   computing `GELU(x @ W + b)` as three PyTorch operations requires writing the
   matrix multiplication result to DRAM, reading it back for GELU -- that is 2 extra
   DRAM accesses of the full matrix. Fusing it into one Triton kernel avoids both.

2. **Custom operations**: Some operations, like Rotary Position Embeddings (RoPE),
   do not have hand-optimized NVIDIA library implementations. Triton lets you write
   custom GPU kernels in Python syntax instead of raw CUDA C++.

3. **Architecture-specific tuning**: Triton compiles your Python-like kernel code
   into PTX/SASS instructions specifically optimized for the GPU you are running on.
   Different GPUs have different amounts of memory, different numbers of cores, and
   different instruction sets -- Triton handles this automatically.

### File Modification Rules

Per GUIDE.md, these files are **read-only** (must match origin/main exactly):
- `model.py` -- The model architecture and generation loop. This defines WHAT
  computations happen. You cannot change the model structure.
- `weight_loader.py` -- Loads pre-trained weights from HuggingFace. The weights
  are fixed; you cannot change them.
- `conv.py` -- 1D convolution for audio subsampling. Already optimized.

You can only modify: `layers.py`, `attention.py`, `rope.py`, `__init__.py`.
These files define HOW the computations are executed (on the GPU), not WHAT
computations are performed.

---

## 2. layers.py -- GPU Compute Kernels

This is the core file containing all neural network building blocks as Triton
kernels and Python layer classes. Think of it as the "engine room" -- model.py
says "multiply these two matrices" and layers.py defines exactly how the GPU
should carry out that multiplication.

### 2.1 GPU Detection: GPUProfile + _KNOWN_CONFIGS

**What it does**: The very first thing layers.py does at import time (before any
model code runs) is figure out which GPU is installed and pick the best settings
for it.

**Why different GPUs need different settings**: GPUs differ in a critical resource
called **shared memory** (also known as SRAM or scratchpad memory).

> **Analogy:** Think of shared memory like a worker's desk. The worker (GPU core)
> needs to lay out materials (data tiles) on the desk to work with them. A bigger
> desk (more shared memory) means you can lay out bigger pieces at once, which
> is more efficient because you make fewer trips to the filing cabinet (main memory /
> DRAM). An NVIDIA RTX 5090 (consumer GPU) has about 99 KB of shared memory per
> processing block, while an H200 (datacenter GPU) has about 228 KB -- more than
> double. If you try to use desk-sized tiles on a card that only has a coffee-table
> amount of shared memory, the program will crash with an "out of shared memory"
> error.

Here is what the code does, step by step:

```python
# _KNOWN_CONFIGS -- a dictionary of tested tile configurations for 6 GPU types
# Each entry maps a GPU architecture name to its optimal settings:
#   attn_tiles: For each head_dim (64 for encoder, 128 for decoder),
#               what size tiles to use in the attention kernel
#   matmul_tiles: What size tiles to use for matrix multiplication
#   rope_nstages/rope_nwarps: Pipeline depth and parallelism for RoPE kernels
#
# Example entry for an RTX 5090:
#   "blackwell_consumer": {
#       attn_tiles: {
#           64: (BLOCK_M=64, BLOCK_N=64, nstages=1, nwarps=4),   # encoder
#           128: (BLOCK_M=32, BLOCK_N=32, nstages=1, nwarps=4),  # decoder
#       },
#       matmul_tiles: (TILE_M=32, TILE_N=32, TILE_K=64),
#       ...
#   }

class GPUProfile:
    """Detects GPU at import time, stores optimal tile sizes for all kernels."""

    def __init__(self):
        # Step 1: Read basic GPU properties
        # - sm_version: The "compute capability" (e.g., 90 for Hopper, 100 for Blackwell)
        #   This is like a CPU's generation -- it tells you what instructions are supported.
        # - shared_memory_per_block_optin: How much shared memory we can request (the "desk size")
        # - gpu_name: Human-readable name like "NVIDIA GeForce RTX 5090"

        # Step 2: Classify the GPU architecture
        # Based on sm_version and gpu_name, figure out which GPU family this is.
        # sm_version >= 100 + "RTX" in name -> "blackwell_consumer" (RTX 5090)
        # sm_version >= 100 + no "RTX"     -> "blackwell_datacenter" (B200)
        # sm_version >= 89  + "RTX" in name -> "ada" (RTX 4090)
        # sm_version >= 90                  -> "hopper" (H100, H200)
        # sm_version >= 80                  -> "ampere" (A100, RTX 3090)
        # else                              -> "turing" (RTX 2080, T4)

        # Step 3: Look up _KNOWN_CONFIGS for this GPU architecture
        # If found, directly assign the tested tile sizes -- these are known to work well.

        # Step 4: Unknown GPU -> compute tiles dynamically from shared memory budget
        # This is the fallback for GPUs we have not tested on. We calculate the largest
        # tile sizes that will fit in this GPU's shared memory.

    def get_attention_tiles(self, head_dim, seq_q=None):
        # Returns (BLOCK_M, BLOCK_N, nstages, nwarps) for the attention kernel.
        # BLOCK_M: how many query rows to process at once
        # BLOCK_N: how many key columns to process at once
        # nstages: pipeline depth (how many tiles to prefetch)
        # nwarps: how many groups of 32 threads to use
        #
        # Special case: if seq_q <= 16 (decoding one token with KV cache),
        # clamp BLOCK_M to 16. This avoids wasting work -- if you only have
        # 1 query row, using BLOCK_M=64 means 63 of those rows do nothing.

GPU = GPUProfile()  # Module-level singleton: computed once when layers.py is imported
```

**Why `shared_memory_per_block_optin`?** This is a subtle but critical detail.
Every NVIDIA GPU reports a default shared memory size of 48 KB through the standard
`shared_memory_per_block` property. But modern GPUs can actually use much more --
they just need to be asked (opted in). The `shared_memory_per_block_optin` property
reports the real maximum: 99 KB on RTX 5090, 228 KB on H200. If you read the wrong
property, you would think all GPUs have 48 KB, and you would always pick tiny tiles
suitable for old consumer GPUs -- even when running on a $30,000 datacenter GPU
that could handle tiles 4x larger.

**Robust fallback chain**: The code tries three property names in order:
`shared_memory_per_block_optin` -> `max_shared_memory_per_block` -> `shared_memory_per_block`.
This prevents crashes on older PyTorch versions that may not expose the optin property.
Without this fallback chain, an H200 running an older PyTorch version would silently
get consumer-sized tiles (64x64) instead of the optimal datacenter tiles (128x128),
leaving ~75% of its performance on the table.

**Dynamic tile computation for unknown GPUs** -- how it works:

The idea is to pick the largest tile size that fits in the GPU's shared memory.
Larger tiles mean more data reuse (each piece of data loaded from slow memory is
used for more computations), which means less time waiting for memory.

- `_compute_attention_tiles(head_dim, smem_bytes)`: Tries tile configurations from
  largest to smallest. For each candidate (e.g., 128x128), it computes the shared
  memory needed:
  - We need to store one Q tile (BLOCK_M x BLOCK_D), one K tile (BLOCK_N x BLOCK_D),
    and one V tile (BLOCK_N x BLOCK_D) in shared memory simultaneously.
  - Formula: `(BLOCK_M + 2*BLOCK_N) * BLOCK_D * 4 bytes + 20KB overhead`
  - The `* 4` is because each float32 number takes 4 bytes.
  - The 20KB overhead accounts for Triton's internal bookkeeping.
  - Example: For 128x128 tiles with head_dim=64:
    `(128 + 2*128) * 64 * 4 + 20480 = 384 * 64 * 4 + 20480 = 98304 + 20480 = 118,784 bytes`
    This fits in an H200's 228KB but NOT in an RTX 5090's 99KB.

- `_compute_matmul_tiles(smem_bytes)`: Similar logic for matrix multiplication,
  but uses the SwiGLU worst case. SwiGLU computes two matrix multiplications
  simultaneously (gate and up projections), so it needs room for both result tiles:
  - Formula: `TILE_K * (TILE_M + 2*TILE_N) * 4 bytes + 20KB overhead`


### 2.2 Helper Functions

```python
def next_power_of_two(x):
    """Rounds up to the nearest power of 2.
    Examples: 5 -> 8, 8 -> 8, 100 -> 128, 1 -> 1
    Implementation: 1 << (x-1).bit_length()
    """
    # Why powers of 2? GPU hardware is designed around powers of 2.
    # Memory access patterns, warp sizes (32 threads), and SIMD
    # (Single Instruction Multiple Data) lanes all work in powers of 2.
    # Using non-power-of-2 sizes would waste hardware lanes and cause
    # misaligned memory accesses, both of which hurt performance.
```

> **Analogy:** Think of a bus that holds 32 passengers. If you have 20 passengers,
> you still need one bus (wastes 12 seats). If you have 33 passengers, you need two
> buses (wastes 31 seats). GPU "buses" (warps) always carry 32 threads. Aligning
> your data to powers of 2 minimizes wasted seats.

```python
def pad_to_multiple(size, multiple):
    """Rounds up size to the nearest multiple of 'multiple'.
    Example: pad_to_multiple(100, 64) = 128
    Example: pad_to_multiple(64, 64) = 64
    Example: pad_to_multiple(65, 64) = 128
    """
    # Why pad? When we divide a matrix into tiles (e.g., 64x64 blocks),
    # the matrix dimensions must be exact multiples of the tile size.
    # Otherwise the last tile would be partial, requiring special handling
    # that slows everything down. Padding with zeros is cheaper than
    # handling edge cases in every kernel.
```

```python
def _to_torch_tensor(arr, dtype=torch.float32, device='cuda'):
    """Convert any array-like object to a PyTorch tensor on the GPU.

    This function is defensive -- it handles many input types because
    generate_v8b might receive data from different sources.
    """
    # Case 1: Already a PyTorch tensor -> just move to correct dtype/device if needed
    # Case 2: CuPy array (detected by hasattr(arr, 'get')) -> convert to numpy first
    #         CuPy is another GPU array library; we need to get data to CPU then back
    # Case 3: numpy array -> convert via torch.as_tensor() (NOT torch.from_numpy())
    # Case 4: Other array-like -> convert to numpy first via np.asarray()

    # Why torch.as_tensor() instead of torch.from_numpy()?
    # In some CUDA 12 environments, the pip-installed numpy version does not exactly
    # match the numpy bundled inside PyTorch's CUDA bindings. This mismatch causes
    # torch.from_numpy() to fail with:
    #   TypeError: expected np.ndarray (got ndarray)
    # Both ARE numpy arrays, but Python thinks they are different types because they
    # come from different numpy installations. torch.as_tensor() is more forgiving
    # and handles this correctly.
```

### 2.2 RMSNorm Kernel

**What it does and WHY**: RMSNorm normalizes each token's hidden state vector so
that its values have a consistent scale. Without normalization, values in the neural
network can grow very large or very small as they pass through many layers, causing
numerical instability (numbers too large for the computer to represent accurately,
or gradients that vanish to zero).

> **Analogy:** Imagine a classroom where students give answers in different units --
> one says "3 miles", another says "15,840 feet", another says "190,080 inches".
> Before you can compare or combine these answers, you need to normalize them to
> a common scale. RMSNorm does this for each token's hidden state vector.

**The math with a concrete example:**

Formula: `y = x / sqrt(mean(x^2) + eps) * weight`

Let us work through a small example. Suppose x = [3, 4, 5] (in reality, x has
2048 or 1280 dimensions, but the principle is the same):

1. Square each element: x^2 = [9, 16, 25]
2. Compute the mean of the squares: mean(x^2) = (9 + 16 + 25) / 3 = 50/3 = 16.67
3. Add eps (a tiny number like 1e-6 to prevent division by zero): 16.67 + 0.000001 = 16.670001
4. Take the square root: sqrt(16.670001) = 4.083
5. Divide each element by this: [3/4.083, 4/4.083, 5/4.083] = [0.735, 0.980, 1.224]
6. Multiply by the learned weight vector (element-wise): if weight = [0.5, 1.0, 0.8],
   then y = [0.735*0.5, 0.980*1.0, 1.224*0.8] = [0.367, 0.980, 0.980]

**How it differs from LayerNorm:** LayerNorm first subtracts the mean (centering
the data around zero), then divides by the standard deviation. RMSNorm skips the
mean subtraction step -- it only divides by the root-mean-square. Research showed
this simplification works almost as well while being ~10% faster because it
eliminates one pass over the data.

**How the GPU parallelism works:**

In the transformer, each token has a hidden state vector (e.g., 2048 numbers for
the decoder). If we have a batch with 80 tokens, we have 80 vectors to normalize,
and they are completely independent of each other.

- **Grid**: One thread block per row (i.e., one thread block per token). If there
  are 80 tokens, we launch 80 thread blocks.
- **Within each block**: All elements of that token's vector are loaded in parallel
  by the threads in the block. Triton provides `tl.load` which loads a range of
  elements at once.
- **Reduction**: `tl.sum(x * x)` computes the sum of squares using **parallel
  reduction**. This is a tree-structured computation where pairs of elements are
  added together, then pairs of those sums are added, and so on -- turning an
  O(N) sequential sum into an O(log N) parallel one. For 2048 elements, this takes
  about 11 steps instead of 2048.
- **`tl.rsqrt()`**: Computes 1/sqrt(x) in a single GPU hardware instruction. Modern
  GPUs have dedicated circuitry for this because it is used so frequently in graphics
  and deep learning. It is faster than computing sqrt(x) and then dividing.

**Where used:** Before every attention block and MLP block in the text decoder, and
at the final output of the text decoder. Specifically:
- `DecoderLayer.input_layernorm` (before attention)
- `DecoderLayer.post_attention_layernorm` (before MLP)
- `TextDecoder.norm` (final normalization before the language model head)


### 2.3 LayerNorm Kernel

**What it does and WHY**: LayerNorm is the audio encoder's normalization method.
It is more thorough than RMSNorm -- it both centers the data (subtracts the mean)
and scales it (divides by the standard deviation). The audio encoder uses it because
audio features benefit from full zero-centered normalization.

**The math with a concrete example:**

Formula: `y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`

Example with x = [2, 6, 4]:

1. Compute the mean: mean(x) = (2 + 6 + 4) / 3 = 4
2. Subtract the mean (centering): x_centered = [2-4, 6-4, 4-4] = [-2, 2, 0]
3. Compute variance: var = mean(x_centered^2) = (4 + 4 + 0) / 3 = 2.667
4. Add eps: 2.667 + 0.000001 = 2.667001
5. Take square root: sqrt(2.667001) = 1.633
6. Divide centered values: [-2/1.633, 2/1.633, 0/1.633] = [-1.225, 1.225, 0]
7. Multiply by weight and add bias: if weight = [1.0, 1.0, 1.0] and bias = [0, 0, 0],
   then y = [-1.225, 1.225, 0]

**Difference from RMSNorm:** LayerNorm has two extra operations:
1. It subtracts the mean (step 2 above) -- RMSNorm does not center the data.
2. It adds a bias term (step 7) -- RMSNorm only has weight, no bias.

This makes LayerNorm slightly slower but more expressive. The audio encoder uses
it because the original Whisper model (which this encoder is based on) was designed
with LayerNorm.

**GPU parallelism**: Identical strategy to RMSNorm -- one thread block per row.
The only difference is that it makes two passes: first to compute the mean, then
to compute variance after subtracting the mean.

**Where used:** In the audio encoder at two points per layer, plus the final norm:
- `AudioEncoderLayer.self_attn_layer_norm` (before attention)
- `AudioEncoderLayer.final_layer_norm` (before MLP)
- `AudioEncoder.layer_norm` (final normalization)


### 2.4 GELU Kernel

**What it does and WHY**: GELU (Gaussian Error Linear Unit) is an activation
function -- a non-linear transformation applied after linear layers. Without
non-linearities, stacking multiple linear layers would be equivalent to a single
linear layer (because a matrix times a matrix is still a matrix). Activation
functions break this linearity, allowing the network to learn complex patterns.

> **Analogy:** If a linear layer is like adjusting brightness and contrast on a
> photo (any combination of adjustments can be done in one step), an activation
> function is like applying a filter that makes bright pixels brighter and dark
> pixels darker in a non-proportional way. It introduces the "interesting" part --
> without it, no amount of layers would learn anything more than a single layer could.

**The math with a concrete example:**

Formula (tanh approximation):
```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

This looks complicated, but it essentially acts as a "soft gate":
- For large positive x (say x = 3): y is approximately 3 (passes through)
- For large negative x (say x = -3): y is approximately 0 (blocked)
- For x near 0: smooth transition (partially passes)

Concrete example with x = 2.0:
1. x^3 = 8.0
2. 0.044715 * 8.0 = 0.3577
3. x + 0.044715 * x^3 = 2.0 + 0.3577 = 2.3577
4. sqrt(2/pi) = 0.7979
5. 0.7979 * 2.3577 = 1.8812
6. tanh(1.8812) = 0.9545
7. 1 + 0.9545 = 1.9545
8. 0.5 * 2.0 * 1.9545 = 1.9545

So GELU(2.0) is approximately 1.95 -- very close to 2.0 (almost fully passed through).

Concrete example with x = -2.0:
1. x^3 = -8.0
2. 0.044715 * (-8.0) = -0.3577
3. x + 0.044715 * x^3 = -2.0 + (-0.3577) = -2.3577
4. 0.7979 * (-2.3577) = -1.8812
5. tanh(-1.8812) = -0.9545
6. 1 + (-0.9545) = 0.0455
7. 0.5 * (-2.0) * 0.0455 = -0.0455

So GELU(-2.0) is approximately -0.05 -- nearly zero (almost fully blocked).

**Implementation detail**: Uses `tl.extra.cuda.libdevice.tanh` for the tanh
computation. This calls into NVIDIA's libdevice math library -- a collection of
hardware-optimized mathematical functions that use the GPU's special function units
(SFUs), dedicated silicon designed specifically for transcendental functions like
tanh, sin, cos, exp, and log. These are much faster than computing tanh from scratch
using a Taylor series approximation.

**Where used:** Audio encoder MLP (after the first linear layer `fc1`), and the
Projector (after `linear_1`). These are standalone `gelu()` calls, not fused.


### 2.5 SiLU Kernel

**What it does and WHY**: SiLU (Sigmoid Linear Unit, also called Swish) is the
activation function used in the text decoder's MLP. It was chosen because the
text decoder uses a Llama-style architecture, and the Llama model family found
that SiLU produces better results than GELU or ReLU for language modeling tasks.

**The math with a concrete example:**

Formula: `y = x * sigmoid(x) = x / (1 + exp(-x))`

This is simpler than GELU but achieves a similar effect:
- For x = 5.0: sigmoid(5) = 1/(1+exp(-5)) = 1/1.0067 = 0.9933, so y = 5 * 0.9933 = 4.97
- For x = 0.0: sigmoid(0) = 1/(1+1) = 0.5, so y = 0 * 0.5 = 0.0
- For x = -5.0: sigmoid(-5) = 1/(1+exp(5)) = 1/149.41 = 0.0067, so y = -5 * 0.0067 = -0.034

Like GELU, it passes positive values through nearly unchanged and suppresses
negative values, but with a slightly different curve shape.

**Why SiLU over ReLU?** ReLU (Rectified Linear Unit) simply sets all negative
values to exactly 0. This hard cutoff can cause "dead neurons" -- neurons whose
inputs are always negative become permanently zeroed out and can never recover.
SiLU has a smooth curve and allows small negative values through, which provides
better gradient flow during training and avoids the dead neuron problem.

**Where used:** Inside the text decoder's SwiGLU MLP. Specifically, the `gate_proj`
output passes through SiLU before being multiplied with the `up_proj` output.


### 2.6 Linear (Matmul) Kernel

**What it does**: This is the workhorse of the entire model. Matrix multiplication
is the single most common and computationally expensive operation in neural networks.
Every projection (Q, K, V, output, every MLP layer) is a matrix multiplication.

**The math**: `C = A @ B` where A has shape (M, K), B has shape (K, N), and C has
shape (M, N).

In concrete terms: if you have 80 tokens, each with 2048 features (so A is 80x2048),
and you want to project them to 256 features (so B is 2048x256), the result C is
80x256. Computing this requires 80 * 256 * 2048 = 41,943,040 multiply-add operations.

> **Analogy:** Imagine computing C[i][j] (one element of the output). You take row i
> of A and column j of B, multiply them element-by-element, and sum the products.
> That is 2048 multiplications and additions for just ONE output element. There are
> 80 * 256 = 20,480 output elements. A GPU can compute many of these simultaneously,
> but it still needs smart data management to be fast.

**GPU Strategy -- Tiled Matrix Multiplication (step by step):**

The key insight is that matrix multiplication has a favorable ratio of computation
to data: it does O(N^3) arithmetic but only reads O(N^2) data. This means each
piece of data can be REUSED many times -- if we are smart about caching it.

Let us walk through a concrete small example. Suppose:
- A is 8x8 (8 rows, 8 columns)
- B is 8x8
- We use tiles of size 4x4 (BLOCK_M=4, BLOCK_N=4, BLOCK_K=4)

The GPU divides the output C into 4x4 tiles. There are (8/4) * (8/4) = 4 output
tiles, so we launch 4 thread blocks, one per output tile.

Thread block handling output tile C[0:4, 0:4]:
```
Step 1 (k = 0..3):
  Load A_tile = A[0:4, 0:4] from DRAM into SRAM    (4x4 = 16 numbers loaded)
  Load B_tile = B[0:4, 0:4] from DRAM into SRAM    (4x4 = 16 numbers loaded)
  acc += A_tile @ B_tile                             (done entirely in SRAM -- fast!)
  // This is 4*4*4 = 64 multiply-adds, but we only loaded 32 numbers.
  // Each number was used 4 times! That is the "reuse" that makes tiling efficient.

Step 2 (k = 4..7):
  Load A_tile = A[0:4, 4:8] from DRAM into SRAM    (16 numbers loaded)
  Load B_tile = B[4:8, 0:4] from DRAM into SRAM    (16 numbers loaded)
  acc += A_tile @ B_tile                             (64 more multiply-adds)

Done! Store acc to C[0:4, 0:4] in DRAM              (16 numbers stored)
```

Total DRAM accesses: 16+16+16+16+16 = 80 numbers.
Total computations: 64+64 = 128 multiply-adds.
Without tiling (naive approach): We would load each element of A and B once per
output element it contributes to, which is much more DRAM traffic.

**`tl.dot(a, b)`**: This Triton instruction compiles to **tensor core** instructions
(HMMA/WMMA) on NVIDIA GPUs that support them (Volta and newer). Tensor cores are
specialized hardware units that can multiply two 4x4 matrices in a SINGLE clock
cycle -- something that would take 64 clock cycles with regular multiply-add units.
This gives roughly a 10x speedup for matrix multiplication.

> **Analogy:** Regular GPU cores multiply numbers one pair at a time, like a
> calculator. Tensor cores multiply entire small matrices at once, like having a
> pre-built lookup table for every possible 4x4 matrix product. NVIDIA has teams of
> engineers who hand-tune tensor core behavior for each GPU generation, which is why
> cuBLAS (NVIDIA's matrix multiplication library that uses tensor cores) is so fast.

**Why we use cuBLAS (not our Triton kernel) for most matrix multiplications:**

The `Linear` class defaults to `BACKEND = "torch"`, which uses `F.linear()` which
calls cuBLAS internally. cuBLAS is faster than our Triton matmul kernel because
NVIDIA has literal teams of dozens of engineers who spend years hand-tuning the
assembly code for each GPU architecture, optimizing memory access patterns,
instruction scheduling, register allocation, and tensor core utilization.
Our Triton kernel is a solid fallback but cannot match that level of optimization.

**Where used:** Every Linear layer in the model -- Q, K, V, and output projections
in every attention layer, every MLP layer (gate_proj, up_proj, down_proj in decoder;
fc1, fc2 in encoder), the projector layers, and the language model head. Roughly
~168 linear operations per forward pass through the decoder.


### 2.7 Softmax Kernel

**What it does**: Converts raw scores (called "logits") into a probability
distribution -- a set of non-negative numbers that sum to 1.0.

**The math with a concrete example:**

Formula: `y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

Example with x = [2.0, 1.0, 0.1]:
1. Find the maximum: max(x) = 2.0
2. Subtract max (for numerical stability): [2.0-2.0, 1.0-2.0, 0.1-2.0] = [0, -1, -1.9]
3. Exponentiate: [exp(0), exp(-1), exp(-1.9)] = [1.0, 0.368, 0.150]
4. Sum: 1.0 + 0.368 + 0.150 = 1.518
5. Divide by sum: [1.0/1.518, 0.368/1.518, 0.150/1.518] = [0.659, 0.242, 0.099]

The result [0.659, 0.242, 0.099] sums to 1.0 and preserves the relative ordering.
The largest input (2.0) gets the highest probability (65.9%).

**Why subtract the max?** Without subtracting, exp(x) can overflow. For example,
exp(1000) is infinity in floating-point arithmetic. But exp(1000 - 1000) = exp(0) = 1,
which is perfectly representable. Subtracting the max does not change the final result
(it cancels out in the division) but prevents numerical overflow.

**Where used:** Only for the final token prediction -- converting the 59,264-dimensional
logit vector into probabilities over the vocabulary. Inside attention, softmax is
computed within the Flash Attention kernel (using online softmax, see section 3.2),
not by this standalone kernel.


### 2.8 Fused Kernels

**What "fusion" means in plain English:**

When the GPU runs two separate operations (like a matrix multiply followed by GELU),
here is what happens without fusion:

```
Operation 1 (matmul kernel):
  1. Read input matrix A from DRAM (slow, ~1 TB/s)
  2. Read weight matrix B from DRAM (slow)
  3. Compute A @ B in SRAM (fast, ~19 TB/s)
  4. Write result C to DRAM (slow)     <-- UNNECESSARY WRITE
  5. Kernel ends, GPU synchronizes

Operation 2 (GELU kernel):
  6. Read C from DRAM (slow)            <-- UNNECESSARY READ
  7. Compute GELU(C) in registers (fast)
  8. Write result to DRAM (slow)
```

Steps 4-6 are wasted work. The result of the matmul is written to slow main memory
only to be immediately read back. With fusion:

```
Fused kernel (matmul + GELU):
  1. Read input matrix A from DRAM (slow)
  2. Read weight matrix B from DRAM (slow)
  3. Compute A @ B in SRAM (fast)
  4. Compute GELU(result) in registers (fast, result never leaves the chip!)
  5. Write final result to DRAM (slow)
```

We eliminated one DRAM write and one DRAM read. For a matrix of size (80, 4096)
in float16, that is 80 * 4096 * 2 bytes = 655,360 bytes saved in EACH direction,
or about 1.3 MB of DRAM traffic eliminated per fused kernel call.

> **Analogy:** It is the difference between cooking a full meal and plating it
> immediately versus cooking the meal, putting it in the fridge, then taking it
> back out to plate it. The fridge trip (DRAM round-trip) is completely unnecessary.

**linear_gelu_kernel:** Computes `GELU(x @ W + b)` in a single kernel launch.
This fuses the matrix multiplication with the GELU activation, eliminating one
DRAM round-trip for the intermediate result.

**Note:** This kernel exists in layers.py but is NOT currently used by model.py.
Model.py calls `fc1(x)` and then `gelu(result)` as two separate operations. The
kernel is "dead code" -- it works correctly but nothing calls it.

**swiglu_fused_kernel:** Computes `SiLU(x @ W_gate) * (x @ W_up)` in one kernel.
This is more impressive because it fuses THREE operations:
1. Matrix multiply: x @ W_gate
2. Activation: SiLU(result_gate)
3. Matrix multiply: x @ W_up
4. Element-wise multiply: SiLU(result_gate) * result_up

Without fusion, the input x would be read from DRAM twice (once for each matmul).
With fusion, x is loaded once and used for both multiplications.

**Active when `MLP.FUSED = True`** -- this IS used by the decoder MLP and provides
real speedup.


### 2.9 Layer Classes

These Python classes wrap the low-level Triton kernels into clean interfaces
that model.py can call.

**`RMSNorm` class:** Wraps the RMSNorm kernel with safety checks.
```python
class RMSNorm:
    # At construction:
    # - Checks if hidden_size fits in BLOCK_SIZE (must be power of 2)
    #   The Triton kernel uses a fixed block size; hidden_size=2048 fits
    #   in BLOCK_SIZE=2048 (already a power of 2).
    # - Stores the learned weight parameter

    # At forward time:
    # - If not on CUDA (e.g., CPU for testing), falls back to PyTorch
    # - Otherwise, launches the Triton kernel
```

**`Linear` class:** The most important class -- every weight matrix in the model
goes through this. It supports switching between two backends:

```python
class Linear:
    BACKEND = "torch"   # Which backend to use for matrix multiplication
    BF16 = True         # Whether to cache half-precision copies of weights
    _HALF_DTYPE = torch.float16  # Actual dtype used for half-precision

    # BACKEND = "torch": Uses F.linear(...) which calls cuBLAS/cuBLASLt
    #   cuBLAS is NVIDIA's hand-optimized matrix multiplication library.
    #   It uses fp16 HGEMM (Half-precision General Matrix Multiply) when
    #   BF16=True and _HALF_DTYPE=float16.
    #   This is currently the fastest option.

    # BACKEND = "triton": Uses our custom linear_kernel_tf32
    #   Our Triton kernel is a fallback -- useful for debugging or for GPUs
    #   where cuBLAS behaves unexpectedly.

    # fp16-throughout pipeline:
    #   The output of Linear stays in fp16 (the .float() conversion to fp32
    #   was removed). This means all downstream operations receive fp16 data,
    #   which halves memory bandwidth requirements throughout the pipeline.
    #   The Triton kernels that need fp32 precision do the conversion internally
    #   via .to(tl.float32), compute in fp32, then output in fp16.
```

**Why fp16 instead of fp32?** Each fp16 number takes 2 bytes instead of 4 bytes
for fp32. This means:
- Half the memory bandwidth is needed to move data around
- Twice as many numbers fit in the GPU's caches and registers
- Tensor cores are faster with fp16 than fp32
The tradeoff is reduced precision (fp16 has ~3 decimal digits of precision vs ~7
for fp32), but neural networks are robust to this for inference.

**`MLP` class:** Implements the SwiGLU gating pattern for the text decoder:
```python
# SwiGLU computation:
# output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
#
# Step by step:
# 1. gate = gate_proj(x)    -- Linear projection to intermediate size
# 2. gate = SiLU(gate)      -- Apply activation (decides what to "gate" / allow through)
# 3. up = up_proj(x)        -- Another linear projection (the actual content)
# 4. combined = gate * up   -- Element-wise: the gate controls which content passes
# 5. output = down_proj(combined)  -- Project back to hidden size
#
# When FUSED=True: steps 1-4 happen in a single swiglu_fused_kernel call.
# This IS used by model.py -- the decoder's self.mlp = MLP(...).
```

> **Analogy:** SwiGLU is like a security checkpoint. The `gate_proj` + SiLU decides
> who gets through (produces numbers between 0 and ~1 for each feature). The `up_proj`
> carries the actual information. Multiplying them together means only the "approved"
> information gets through. The `down_proj` then compresses the result back down.

**`EncoderMLP` class:** A simpler MLP for the audio encoder:
```python
# output = fc2(GELU(fc1(x)))
# No gating -- just linear -> activation -> linear
```
**NOT used by origin/main model.py** -- the encoder uses plain `self.fc1 = Linear(...)`
and calls `gelu()` inline. This class exists for compatibility but is dead code.

**`LinearGELU` class:** A `GELU(Linear(x))` wrapper.
**NOT used by origin/main model.py** -- also dead code.


---

## 3. attention.py -- Attention Mechanism

### 3.1 Scaled Dot-Product Attention

**What attention is and WHY it matters:**

Attention is the core mechanism that makes transformers powerful. It allows each
token to "look at" every other token and decide which ones are relevant.

> **Analogy:** Imagine reading the sentence "The cat sat on the mat because it was
> tired." When processing the word "it", the model needs to figure out that "it"
> refers to "the cat", not "the mat". Attention computes a relevance score between
> "it" and every other word, finding that "cat" has the highest relevance.

The formula:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

In plain English:
1. **Q (Query)**: "What am I looking for?" Each token produces a query vector.
2. **K (Key)**: "What do I contain?" Each token produces a key vector.
3. **V (Value)**: "What information do I carry?" Each token produces a value vector.
4. **Q @ K^T**: Compute the dot product of each query with every key. This produces
   a "relevance score" for every pair of tokens. If the query of "it" and the key
   of "cat" are similar (high dot product), the score is high.
5. **/ sqrt(d_k)**: Scale down by the square root of the key dimension. Without this,
   the dot products would be too large, making softmax produce nearly one-hot
   distributions (all attention on one token, ignoring everything else).
   For d_k=128: sqrt(128) = 11.31, so scores are divided by 11.31.
6. **softmax(...)**: Convert scores to probabilities (non-negative, sum to 1).
7. **... @ V**: Weighted sum of value vectors. If "it" attends 70% to "cat" and
   30% to "mat", the output is 0.7 * V_cat + 0.3 * V_mat.

**Concrete example with tiny matrices:**

Suppose we have 3 tokens, each with dimension 2:
```
Q = [[1, 0],     K = [[1, 0],     V = [[10, 20],
     [0, 1],          [0, 1],          [30, 40],
     [1, 1]]          [1, 1]]          [50, 60]]
```

Step 1: Q @ K^T (3x3 score matrix):
```
Scores = [[1*1+0*0, 1*0+0*1, 1*1+0*1],   = [[1, 0, 1],
           [0*1+1*0, 0*0+1*1, 0*1+1*1],      [0, 1, 1],
           [1*1+1*0, 1*0+1*1, 1*1+1*1]]      [1, 1, 2]]
```

Step 2: Divide by sqrt(2) = 1.414:
```
Scaled = [[0.707, 0,     0.707],
          [0,     0.707, 0.707],
          [0.707, 0.707, 1.414]]
```

Step 3: Softmax each row:
```
Weights = [[0.380, 0.241, 0.380],
           [0.241, 0.380, 0.380],
           [0.245, 0.245, 0.511]]
```

Step 4: Weights @ V:
```
Output[0] = 0.380*[10,20] + 0.241*[30,40] + 0.380*[50,60] = [30.0, 38.0]
...
```

Each output row is a weighted combination of ALL value vectors, with weights
determined by how relevant each key was to that query.


### 3.2 Primary Path: Fused Flash Attention Kernel (Triton)

**The problem with naive attention:**

The naive approach (compute Q @ K^T as a full matrix, apply softmax, multiply by V)
has a critical issue: the score matrix Q @ K^T has size (seq_len x seq_len).

For our model with ~175 audio tokens processed by the encoder:
- Score matrix: 175 x 175 = 30,625 elements per head
- With 20 heads and fp16: 30,625 * 20 * 2 bytes = 1.2 MB
- Not terrible, but it must be written to DRAM and read back for softmax.

For longer sequences (like 2048 tokens in the decoder), the score matrix would be
2048 x 2048 = 4,194,304 elements per head -- this is **O(N^2)** memory, and it
gets worse quickly.

**Flash Attention solves this** by never materializing the full score matrix. Instead,
it processes attention in tiles, computing softmax incrementally ("online softmax").
This reduces memory from O(N^2) to O(N) -- the score matrix tiles exist only in
fast SRAM, never in DRAM.

> **Analogy:** Imagine you need to find the average height of everyone in a city of
> 1 million people. The naive approach is to write down all 1 million heights on a
> giant piece of paper, then compute the average. Flash Attention is like walking
> through the city with a calculator, updating a running average as you meet each
> person. You never need the giant piece of paper -- just the running total and
> the count.

**How the online softmax algorithm works, step by step:**

The challenge with computing softmax incrementally is that softmax requires knowing
the maximum value BEFORE you can compute any of the exponentials (for numerical
stability). But if you are processing data in blocks, you do not know the global
maximum until you have seen all blocks.

The online softmax trick: maintain a running maximum and a correction factor.
When you encounter a new, larger maximum, you retroactively correct all previous
work.

Let us trace through a concrete example. Suppose we have a query processing 3
blocks of key-value pairs, and the attention scores (after Q @ K^T) for the
blocks are: S_block1 = [2, 3], S_block2 = [5, 1], S_block3 = [4, 4].

**Block 1: S = [2, 3]**
```
m_i = -inf (initial maximum)
m_new = max(-inf, max(2, 3)) = 3
alpha = exp(-inf - 3) = 0  (anything times 0 is 0, which correctly ignores initial state)
p = exp([2, 3] - 3) = [exp(-1), exp(0)] = [0.368, 1.0]
l_i = 0 * 0 + sum([0.368, 1.0]) = 1.368   (running sum of exponentials)
acc = 0 * 0 + [0.368, 1.0] @ V_block1      (weighted sum of V vectors)
m_i = 3
```

**Block 2: S = [5, 1]**
```
m_new = max(3, max(5, 1)) = 5
alpha = exp(3 - 5) = exp(-2) = 0.135   (correction factor for old state)
p = exp([5, 1] - 5) = [exp(0), exp(-4)] = [1.0, 0.018]
l_i = 0.135 * 1.368 + sum([1.0, 0.018]) = 0.185 + 1.018 = 1.203
acc = 0.135 * (old acc) + [1.0, 0.018] @ V_block2
    // The old accumulator is RESCALED by 0.135 because we discovered a new max.
    // This is mathematically equivalent to having used max=5 from the start.
m_i = 5
```

**Block 3: S = [4, 4]**
```
m_new = max(5, max(4, 4)) = 5   (no change!)
alpha = exp(5 - 5) = exp(0) = 1.0   (no rescaling needed -- max did not change)
p = exp([4, 4] - 5) = [exp(-1), exp(-1)] = [0.368, 0.368]
l_i = 1.0 * 1.203 + sum([0.368, 0.368]) = 1.203 + 0.736 = 1.939
acc = 1.0 * (old acc) + [0.368, 0.368] @ V_block3
m_i = 5
```

**Final normalization:**
```
output = acc / l_i   (divide by the total sum of exponentials)
```

This gives the EXACT same result as computing the full score matrix, applying
softmax, and multiplying by V -- but without ever storing the full score matrix.

**The actual kernel code (conceptual, with line-by-line comments):**

```python
# Each thread block handles BLOCK_M query rows
m_i = -inf          # Running max, one value per query row (BLOCK_M values)
l_i = 0             # Running sum of exp, one value per query row
acc = 0             # Output accumulator, shape [BLOCK_M, BLOCK_D]

for each K/V block of BLOCK_N keys:
    # Step 1: Compute attention scores for this block
    S = Q_tile @ K_block^T          # tl.dot -- uses tensor cores
    # S has shape [BLOCK_M, BLOCK_N]: every query's score against every key in this block

    # Step 2: Update running maximum
    m_new = max(m_i, row_max(S))    # New max per query row

    # Step 3: Compute correction factor for old state
    alpha = exp(m_i - m_new)        # If max increased, alpha < 1 (shrinks old values)
                                    # If max unchanged, alpha = 1 (no correction needed)

    # Step 4: Compute new attention weights
    p = exp(S - m_new)              # Exponentiate scores with new max subtracted

    # Step 5: Update running sum of exponentials
    l_i = alpha * l_i + row_sum(p)  # Rescale old sum + add new sum

    # Step 6: Update output accumulator
    acc = alpha * acc + p @ V_block # Rescale old output + add new weighted values
    # p @ V_block uses tl.dot (tensor cores) -- this is the weighted sum of V vectors

    # Step 7: Update stored maximum
    m_i = m_new

# Final: divide by sum of exponentials to get proper probability-weighted average
output = acc / l_i
```

**Why this is better than using three separate kernels:**

| Aspect | 3 Separate Kernels | Flash Attention |
|--------|-------------------|-----------------|
| Score matrix storage | Full (seq x seq) in DRAM | Never stored (stays in SRAM) |
| DRAM reads/writes | 3 kernel launches, 4+ DRAM round-trips | 1 kernel launch, 1 DRAM write |
| Memory usage | O(N^2) per head | O(BLOCK_M * BLOCK_D) per head |
| Synchronization | 3 kernel launches need sync between them | No sync needed (single kernel) |
| Tensor core usage | Only for Q@K^T and P@V separately | Both Q@K^T and P@V in same kernel |

For a concrete memory comparison: with 175 tokens, 20 heads, fp16:
- 3-kernel approach: scores matrix = 175 * 175 * 20 * 2 = 1.2 MB in DRAM
- Flash Attention: no scores matrix in DRAM at all -- just BLOCK_M * BLOCK_N * 4 bytes
  in SRAM per block, e.g., 64 * 64 * 4 = 16 KB

**Tile sizes** are chosen by GPUProfile to match each GPU's shared memory:

- **Consumer GPUs** (RTX 4090/5090, ~100KB shared memory):
  - Encoder (head_dim=64): `BLOCK_M=64, BLOCK_N=64`
    - Shared memory needed: (64 + 2*64) * 64 * 4 + 20KB = ~69 KB (fits in 100KB)
  - Decoder (head_dim=128): `BLOCK_M=32, BLOCK_N=32`
    - Head_dim is 2x larger, so tiles must be smaller to fit
  - When decoding with KV cache (seq_q <= 16): `BLOCK_M=16`
    - Only 1 query token, so large BLOCK_M wastes threads

- **Datacenter GPUs** (H200/B200, ~228KB shared memory):
  - Encoder (head_dim=64): `BLOCK_M=128, BLOCK_N=128`
    - 4x more data processed per block than consumer GPUs
  - Decoder (head_dim=128): `BLOCK_M=128, BLOCK_N=64`
    - Much larger tiles possible thanks to 2x+ more shared memory

- **Unknown GPUs**: tiles computed dynamically by `_compute_attention_tiles()`

**Features:**
- `IS_CAUSAL` (constexpr): When set, the decoder cannot look at future tokens.
  Token 5 can only attend to tokens 1-5, not 6 onwards. This is implemented by
  skipping K/V blocks that are entirely in the "future" relative to the current
  Q block -- no wasted computation.
- `HAS_MASK` (constexpr): Supports an additive attention mask for the audio encoder
  (to mask padded positions). When False, the mask logic is completely compiled out
  (zero overhead).


### 3.3 SDPA Fallback for KV-Cached Decode

**What this is**: During text generation with KV caching (see section 5.5), each
decode step processes only ONE new token. Launching the full Triton Flash Attention
kernel for just one query row has significant overhead: the kernel needs to be
compiled (first time), the GPU launch infrastructure has fixed costs, and the
kernel has a minimum granularity.

For these tiny single-token decode steps (seq_q <= 4), we fall back to PyTorch's
built-in `scaled_dot_product_attention` (SDPA), which uses cuDNN internally and
avoids the Triton compilation/launch overhead.

```python
# In attention.py:
if q.is_cuda and seq_q <= 4:
    # Skip the Triton kernel entirely for tiny queries
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=scale
    )
```

**Impact**: Saves ~3ms per full inference (from 113.5ms to 110.0ms, and further
to 98.5ms with the fp16 pipeline). Since there are ~13 decode steps, that is about
0.23ms saved per step -- small individually, but it adds up.


### 3.4 Legacy Attention Kernels (REMOVED)

The original assignment provided three separate kernels that together implemented
attention the slow way:
1. ~~`attention_scores_kernel`~~: Computed Q @ K^T * scale (score matrix)
2. ~~`softmax_inplace_kernel`~~: Applied softmax to the score matrix
3. ~~`attention_output_kernel`~~: Computed attn_weights @ V

Plus ~~`causal_mask_kernel`~~ for generating the causal mask.

These were removed (~175 lines of code deleted) because they were completely
superseded by `flash_attention_kernel`, which does all of the above in a single
fused kernel.


### 3.5 Grouped Query Attention (GQA)

**What it is**: The text decoder uses 16 query heads but only 4 key/value heads.
This means every 4 query heads share the same key/value head. This reduces memory
and computation by 4x for the K and V projections without significantly hurting
quality.

> **Analogy:** Imagine 16 students (query heads) taking a test. Instead of each
> student getting their own copy of the reference book (16 K/V heads), groups of 4
> students share one copy (4 K/V heads). The students still ask different questions
> (different Q), but they all look up answers in the same book.

GQA is handled by `_expand_kv_heads()` before the Flash Attention kernel call.
This function duplicates each K/V head 4 times so that the 4 KV heads become 16,
matching the 16 query heads. This duplication is pure memory copying (no computation)
and allows the Flash Attention kernel to treat all heads uniformly.


### 3.6 Numerical Parity Tests

The `__main__` block of attention.py contains a 17-case deterministic test suite.
Each test:
1. Creates random Q, K, V tensors with specific shapes
2. Runs them through both our Flash Attention kernel and PyTorch's reference SDPA
3. Compares the results, asserting they match within floating-point tolerance

Test cases cover:
- Basic and causal attention at both head_dim=64 (encoder) and head_dim=128 (decoder)
- Additive masks with different broadcast shapes
- GQA configurations (16 query heads / 4 KV heads and 4/2)
- Encoder-like sequence lengths (175 tokens) and decoder-like (93 tokens)
- Single-token decode (the KV-cached generation case)
- Non-power-of-two shapes (17x61) to stress edge cases

---

## 4. rope.py -- Positional Encodings

### 4.1 What is RoPE?

**The problem**: Transformers treat their input as a SET, not a SEQUENCE. The
attention mechanism computes pairwise scores between all tokens regardless of their
positions. Without position information, "the cat sat on the mat" and "the mat sat
on the cat" would produce identical outputs -- clearly wrong.

> **Analogy:** Imagine reading a book where all the pages are shuffled. The words
> are all there, but without page numbers (position information), you cannot
> reconstruct the story. RoPE is like stamping page numbers onto the text in a
> clever way that the model can use to understand ordering.

**How RoPE works:** Rotary Position Embeddings encode position by ROTATING vectors
in 2D subspaces. The key insight is that after rotation, the dot product between
two vectors depends on their RELATIVE position difference, not their absolute positions.

For each pair of consecutive dimensions (x1, x2) in the Q and K vectors, RoPE
applies a 2D rotation by an angle that depends on the position:

```
x1_rot = x1 * cos(p * freq) - x2 * sin(p * freq)
x2_rot = x2 * cos(p * freq) + x1 * sin(p * freq)
```

Where:
- p is the token's position (0, 1, 2, ...)
- freq depends on which pair of dimensions we are rotating (higher dimensions
  get lower frequencies, allowing the model to encode both fine-grained and
  coarse-grained position information)

**Concrete example:** Suppose x1 = 3, x2 = 4, position p = 5, and freq = 0.1.
Then the rotation angle is theta = 5 * 0.1 = 0.5 radians (about 28.6 degrees).

```
cos(0.5) = 0.8776,  sin(0.5) = 0.4794
x1_rot = 3 * 0.8776 - 4 * 0.4794 = 2.6328 - 1.9176 = 0.7152
x2_rot = 4 * 0.8776 + 3 * 0.4794 = 3.5104 + 1.4382 = 4.9486
```

The vector [3, 4] has been rotated to [0.7152, 4.9486]. If another token at
position 7 has the same vector, its rotation angle would be 7 * 0.1 = 0.7, giving
a different rotation. When the model computes Q @ K^T, the dot product between these
rotated vectors will encode information about their position difference (7 - 5 = 2).


### 4.2 Partial RoPE

The audio encoder uses **50% partial RoPE** -- only the first half of each head's
dimensions are rotated. The second half passes through unchanged. This is a design
choice from the original Whisper architecture, where the authors found that rotating
all dimensions was unnecessary for audio processing and could even hurt quality.

For example, if head_dim=64, only dimensions 0-31 get rotated, while dimensions
32-63 stay unchanged.


### 4.3 Kernel Implementation

The `compute_freqs_kernel` precomputes the cos and sin values for all positions at
once. This is done once before the attention layers, and the results are reused by
every layer.

There is an interesting detail: the cos/sin arrays are duplicated -- the first half
equals the second half. This is because `apply_rotary_pos_emb` splits the input
vector into two halves and applies the same cos/sin to each half. By duplicating,
the kernel avoids special-case logic for the two halves.

The fused Q+K RoPE pair kernel (adopted from another team's "meave" branch) applies
RoPE to both Q and K in a single kernel launch, saving ~14ms by eliminating one
kernel launch and one DRAM round-trip.

---

## 5. model.py -- Full Model Pipeline (READ-ONLY, origin/main)

This file defines the model architecture -- WHAT computations happen and in what
order. You cannot modify it, but understanding it is essential for implementing
correct kernels.

### 5.1 AudioEncoder

The audio encoder converts the mel spectrogram into a rich representation of
the speech content. It uses 32 transformer layers, which is deep enough to
capture everything from individual phonemes to word-level meaning.

```python
class AudioEncoder:
    # Step 1: Two convolutional layers extract local features
    conv1: Conv1d(128 -> 1280, kernel=3, stride=1)   # 128 mel bins -> 1280 features, keep length
    conv2: Conv1d(1280 -> 1280, kernel=3, stride=2)  # 1280 -> 1280, halve length (stride 2)

    # Step 2: Position encoding
    rotary_emb: RotaryEmbedding(dim=64, partial=0.5)  # 50% partial RoPE

    # Step 3: 32 transformer layers that progressively understand the audio
    layers: [AudioEncoderLayer x 32]

    # Step 4: Final normalization
    layer_norm: LayerNorm(1280)
```

Each AudioEncoderLayer contains:
- Self-attention (tokens attend to each other -- "is this phoneme part of a word?")
- MLP (further processing -- "what does this sound mean?")

**AudioEncoderLayer MLP** uses plain `Linear` layers (NOT the `EncoderMLP` class):
```python
# In model.py, the encoder MLP is:
self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
    # hidden_size=1280, intermediate_size=5120
    # Projects each token from 1280 to 5120 dimensions (expansion)
self.fc2 = Linear(intermediate_size, hidden_size, bias=True)
    # Projects back from 5120 to 1280 dimensions (compression)

# The forward pass calls them separately:
hidden_states = self.fc1(hidden_states)   # Linear: (batch, seq, 1280) -> (batch, seq, 5120)
hidden_states = gelu(hidden_states)       # Activation: keeps shape, applies non-linearity
hidden_states = self.fc2(hidden_states)   # Linear: (batch, seq, 5120) -> (batch, seq, 1280)
```

### 5.2 MultiModalProjector

**What it does**: Bridges between the audio encoder's representation (1280-dimensional)
and the text decoder's expected input (2048-dimensional). Also pools (compresses)
4 audio frames into 1, reducing the sequence length by 4x.

```python
class MultiModalProjector:
    # Uses plain Linear layers:
    self.linear_1 = Linear(pooled_dim, config.projector_hidden_size, bias=True)
        # pooled_dim = 4 * 1280 = 5120 (4 frames concatenated)
        # projector_hidden_size = 4096
    self.act = gelu
    self.linear_2 = Linear(config.projector_hidden_size, config.text_hidden_size, bias=True)
        # text_hidden_size = 2048
```

**Pooling step**: Takes every 4 consecutive audio frames and concatenates them:
- Input: (1, T/2, 1280) -- e.g., (1, 175, 1280) from the encoder
- Pool 4 frames: (1, T/8, 5120) -- e.g., (1, ~44, 5120)
  (175/4 = 43.75, rounds to ~44)

**Projection step**: Two linear layers with GELU activation:
- Linear(5120 -> 4096): (1, ~44, 5120) -> (1, ~44, 4096)
- GELU: shape unchanged, applies non-linearity
- Linear(4096 -> 2048): (1, ~44, 4096) -> (1, ~44, 2048)

Now the audio is in the same 2048-dimensional space as text tokens.


### 5.3 TextDecoder

The text decoder generates text one token at a time, using both the audio context
and the previously generated tokens.

```python
class TextDecoder:
    embed_tokens: Embedding(59264, 2048)
        # Vocabulary of 59,264 possible tokens
        # Each token is represented as a 2048-dimensional vector
    rope: RotaryEmbedding(dim=128, base=500000)
        # Full RoPE (not partial like encoder)
        # base=500000 for longer-range position encoding
    layers: [DecoderLayer x 28]
        # 28 transformer layers
    norm: RMSNorm(2048)
        # Final normalization before language model head
```

**KV cache infrastructure**: The model.py code contains methods for KV-cached
decoding (`forward_with_kv_buffer`, `allocate_kv_buffers`, etc.), but the stock
`generate()` method does NOT use them. The `generate_v8b` function (monkey-patched
from layers.py) does use KV caching through the `decode(use_cache=True)` path.


### 5.4 DecoderLayer

Each of the 28 decoder layers performs these operations:

```python
# Step 1: Pre-attention normalization
# RMSNorm ensures the hidden states have a consistent scale before attention
normalized = input_layernorm(hidden_states)

# Step 2: Q/K/V projections (3 separate Linear layers)
Q = q_proj(normalized)  # (batch, seq, 2048) -> (batch, seq, 2048)  [16 heads * 128 dim]
K = k_proj(normalized)  # (batch, seq, 2048) -> (batch, seq, 512)   [4 heads * 128 dim]
V = v_proj(normalized)  # (batch, seq, 2048) -> (batch, seq, 512)   [4 heads * 128 dim]

# Step 3: Reshape to multi-head format
Q = Q.view(batch, seq, 16, 128).transpose(1, 2)  # (batch, 16, seq, 128)
K = K.view(batch, seq, 4, 128).transpose(1, 2)   # (batch, 4, seq, 128)
V = V.view(batch, seq, 4, 128).transpose(1, 2)   # (batch, 4, seq, 128)

# Step 4: Apply RoPE to Q and K (full, not partial)
Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

# Step 5: Attention (Flash Attention kernel with GQA)
# The 4 K/V heads are expanded to 16 (matching Q's 16 heads) via _expand_kv_heads
# Causal masking prevents looking at future tokens
attn_output = scaled_dot_product_attention(Q, K, V, is_causal=True)

# Step 6: Output projection + residual connection
attn_output = o_proj(attn_output)     # (batch, seq, 2048)
hidden_states = hidden_states + attn_output  # Residual: original + attention output

# Step 7: Pre-MLP normalization
normalized = post_attention_layernorm(hidden_states)

# Step 8: SwiGLU MLP
gate = gate_proj(normalized)   # (batch, seq, 2048) -> (batch, seq, 5632)
gate = silu(gate)              # Apply activation (gating signal)
up = up_proj(normalized)       # (batch, seq, 2048) -> (batch, seq, 5632)
combined = gate * up           # Element-wise multiply (gate controls information flow)
mlp_output = down_proj(combined)  # (batch, seq, 5632) -> (batch, seq, 2048)
hidden_states = hidden_states + mlp_output  # Residual connection
```

> **Why residual connections?** Without them, the gradient signal used for training
> gets weaker as it passes through each layer (the "vanishing gradient" problem).
> With 28 layers, the signal would practically disappear. Residual connections
> create a "highway" that lets the gradient flow directly from the output back to
> early layers, like express lanes on a highway that skip traffic.


### 5.5 Generation Pipeline

**`generate()` -- The stock generation method (O(n^2)):**

This is the original, unoptimized generation loop. It works correctly but is slow.

```python
def generate(self, input_features, input_ids=None, ...):
    # Step 1: Encode audio (done once, not repeated)
    audio_embeds = self.encode_audio(input_features, ...)
    # audio_embeds shape: (1, ~44, 2048)

    # Step 2: Build initial inputs_embeds
    # Combines the chat template tokens, audio embeddings, and any initial text tokens
    inputs_embeds = torch.cat([before_audio, audio_embeds, after_audio], dim=1)
    # inputs_embeds shape: (1, ~80, 2048)

    # Step 3: Autoregressive decode -- generates one token at a time
    for _ in range(max_new_tokens):
        # PROBLEM: Each step processes the ENTIRE sequence through all 28 layers!
        logits = self.decode(inputs_embeds=inputs_embeds)
        # Step 1: seq_len=80, decoder processes 80 tokens
        # Step 2: seq_len=81, decoder processes 81 tokens
        # Step 3: seq_len=82, decoder processes 82 tokens
        # ...
        # Step 13: seq_len=93, decoder processes 93 tokens
        #
        # Total work: 80 + 81 + 82 + ... + 93 = sum of N terms = O(N^2)!

        next_token = sample(logits[:, -1, :])  # Only use the LAST position's prediction
        if next_token == eos_token: break
        new_embeds = self.text_decoder.embed_tokens(next_token)
        inputs_embeds = torch.cat([inputs_embeds, new_embeds], dim=1)  # Sequence grows!
```

**Concrete O(n^2) accounting:** With 80 initial tokens and 13 generated tokens:
- Step 1: Process 80 tokens through 28 layers = 80 * 28 = 2,240 layer-applications
- Step 2: Process 81 tokens = 81 * 28 = 2,268
- ...
- Step 13: Process 92 tokens = 92 * 28 = 2,576
- **Total: (80 + 81 + ... + 92) * 28 = 1,118 * 28 = 31,304 layer-applications**

Notice that in step 13, the decoder reprocesses the original 80 tokens AGAIN even
though their computations have not changed since step 1. This is enormously wasteful.

**`generate_v8b` -- The KV-cached generation method (O(n)):**

KV caching remembers the key and value vectors computed in previous steps, so each
new step only needs to process the ONE new token.

```python
# Prefill step: Process the full initial sequence (done once)
logits, past_kv = decode(inputs_embeds=initial_embeds, use_cache=True)
# past_kv stores K and V for all 28 layers, all heads, all 80 positions

# Decode loop: Process only the new token each step
for _ in range(max_new_tokens):
    new_embeds = embed_tokens(next_token)  # Shape: (1, 1, 2048) -- just ONE token
    logits, past_kv = decode(
        inputs_embeds=new_embeds,           # Only the new token!
        past_key_values=past_kv,            # Reuse stored K/V from all previous steps
        use_cache=True
    )
    # In the attention layers, Q is (1, 1, ...) but K/V are (1, 80+n, ...)
    # The new Q attends to ALL previous K/V (stored in cache) + the new K/V
    next_token = sample(logits[:, -1, :])
    if next_token == eos_token: break
```

**Concrete O(n) accounting:** With 80 initial tokens and 13 generated tokens:
- Prefill: Process 80 tokens = 80 * 28 = 2,240 layer-applications (done once)
- Step 1: Process 1 token (attending to 81 K/V) = 1 * 28 = 28
- Step 2: Process 1 token = 28
- ...
- Step 13: Process 1 token = 28
- **Total: 2,240 + 13 * 28 = 2,240 + 364 = 2,604 layer-applications**

That is **31,304 vs 2,604** -- a **12x reduction** in total work for just 13 tokens.
For longer generations (e.g., 100 tokens), the savings would be even more dramatic:
- O(n^2): (80 + 81 + ... + 179) * 28 = 12,950 * 28 = 362,600
- O(n): 2,240 + 100 * 28 = 5,040
- **72x reduction!**

> **Analogy:** Imagine writing an essay and asking someone to proofread it. The O(n^2)
> approach is like having them re-read the ENTIRE essay from scratch after you add
> each new sentence. The O(n) approach (KV cache) is like having them remember
> everything they have already read and only look at the new sentence.


---

## 6. conv.py -- Audio Feature Extraction (READ-ONLY)

**What it does**: The first stage of audio processing. Takes the mel spectrogram
(a 2D representation of audio with 128 frequency bins and ~350 time steps) and
applies two convolutional layers to extract local features.

```python
class Conv1dSubsampler:
    conv1: Conv1d(128, 1280, kernel=3, stride=1)
        # Input: 128 frequency bins
        # Output: 1280 feature channels
        # kernel=3: looks at 3 consecutive time steps at a time
        # stride=1: moves one step at a time (output length = input length)

    conv2: Conv1d(1280, 1280, kernel=3, stride=2)
        # stride=2: skips every other step (output length = input length / 2)
        # This "downsamples" by 2x, reducing the sequence length
```

> **Analogy:** Convolution is like sliding a magnifying glass over the spectrogram.
> At each position, the magnifying glass (kernel) looks at 3 adjacent time frames
> and summarizes what it sees into 1280 features. The second convolution does the
> same but skips every other position (stride=2), cutting the length in half.

**Implementation detail**: Uses `im2col_1d()` to reshape the convolution into a
matrix multiplication. The idea is that convolution can be expressed as:
1. Rearrange the input so that each "window" of 3 consecutive frames becomes one row
2. Multiply this rearranged matrix by the kernel weights
This converts the convolution into a matrix multiply, which can then use the
highly optimized cuBLAS or Triton matmul kernels.

---

## 7. weight_loader.py -- Model Weights (READ-ONLY)

**What it does**: Downloads the pre-trained model weights from HuggingFace (a model
hosting platform) and maps them to our model's layer names.

The model was trained by the GLM team on massive amounts of speech data. The weights
encode everything the model "knows" -- the patterns it learned for converting speech
to text. Training takes thousands of GPU-hours; we just load the results.

**Why a separate loader?** The HuggingFace model format uses different names for
layers than our implementation. The weight loader handles this mapping, e.g.,
`model.encoder.layers.0.self_attn.q_proj.weight` in HuggingFace becomes
`audio_encoder.layers[0].self_attn.q_proj.weight` in our code.

---

## 8. benchmark_student.py -- Testing

### Generate Function Selection

The benchmark script automatically detects which generation method is available:

```python
# benchmark_student.py checks for optimized generate methods:
if hasattr(model, 'generate_v8b'):
    generate_fn = model.generate_v8b    # Best: KV-cached, O(n) decode
elif hasattr(model, 'generate_v8'):
    generate_fn = model.generate_v8     # Not available in our codebase
elif hasattr(model, 'generate_v6'):
    generate_fn = model.generate_v6     # Not available in our codebase
else:
    generate_fn = model.generate        # Stock O(n^2) from model.py
```

The `generate_v8b` function is "monkey-patched" onto the model at import time by
layers.py's `_try_patch_v8b()` function. Monkey-patching means attaching a new
method to an existing object at runtime, without modifying the original class
definition.

> **Analogy:** Monkey-patching is like taping a new button onto your TV remote that
> activates a feature the remote was not originally designed for. The original buttons
> still work, but now you have an extra one.

### Accuracy Check

The benchmark does not just measure speed -- it also verifies that the transcription
is correct:

```python
def check_transcription(transcription, expected):
    # Step 1: Normalize both strings -- uppercase, remove punctuation
    #   "Hello, world!" -> "HELLO WORLD"
    # Step 2: Split into sets of words
    #   "HELLO WORLD" -> {"HELLO", "WORLD"}
    # Step 3: Compare word sets (not exact string match)
    #   This is forgiving of minor differences in punctuation or word order
    # Step 4: Pass if > 80% of expected words are present
```

### Current Benchmark Results

**RTX 5090 (2026-03-15):**
- With fp16 pipeline + generate_v8b + SDPA fallback: **98.5ms**, 7.58 ms/token
- With bf16 pipeline + generate_v8b + SDPA fallback: 110.0ms (+/- 0.2ms), 8.46 ms/token
- Without generate_v8b (stock O(n^2)): 120.7ms (+/- 0.2ms), 9.29 ms/token
- 13 tokens generated, 100.0% transcription accuracy
- Competition standings: ankush 98.5ms, meave 127.8ms, yash 128ms, majed 187.9ms

**H200 MIG 3g.71gb Teaching Cluster (2026-03-16):**
- With fp16 pipeline + generate_v8b + SDPA fallback: **204.6ms** (+/- 1.7ms), 15.74 ms/token
- 60 SMs (Streaming Multiprocessors -- the GPU's processing units) vs RTX 5090's
  170 SMs, so proportionally slower
- GPUProfile correctly detected Hopper architecture (sm_90) and used datacenter
  tile configurations

**NOTE:** `benchmark_detailed.py` fails with the fp16 pipeline because it expects
float32 projector output. The student benchmark (the authoritative one) works correctly.

---

## 9. How It All Fits Together

### Data Flow (Single Inference) -- Traced with Concrete Tensor Shapes

This section traces through the COMPLETE pipeline for a ~3.5-second audio clip
being transcribed to "Concord returned to its place amidst the tents." We show
the exact tensor shapes at every step, so you can see how the data transforms.

```
1. Audio WAV (16kHz, ~3.5 seconds for test audio)
   -> Raw tensor: (1, 56000) -- 3.5 * 16000 = 56,000 samples
   -> Each sample is a float32 number representing air pressure at that instant
   |
2. Mel Spectrogram extraction (by HuggingFace processor, runs on CPU)
   -> The processor applies a Short-Time Fourier Transform (STFT) to convert
      the time-domain signal into a frequency-domain representation.
   -> 128 mel frequency bins capture energy at different pitches
   -> ~350 time frames (each covering ~10ms of audio with overlap)
   -> Tensor: (1, 128, ~350) -- shape is (batch, frequency_bins, time_frames)
   -> Memory: 1 * 128 * 350 * 4 bytes = ~175 KB (float32)
   |
3. Conv Feature Extraction (conv.py)
   -> Conv1 (kernel=3, stride=1) + GELU activation:
      Input: (1, 128, ~350)
      Each output position looks at 3 consecutive input frames and combines
      128 frequency features into 1280 features.
      Output: (1, 1280, ~350) -- length preserved (stride=1)
      Memory: 1 * 1280 * 350 * 2 bytes = ~896 KB (fp16)

   -> Conv2 (kernel=3, stride=2) + GELU activation:
      Input: (1, 1280, ~350)
      Stride=2 means the kernel jumps 2 positions each step, halving the length.
      Output: (1, 1280, ~175) -- length halved
      Memory: 1 * 1280 * 175 * 2 bytes = ~448 KB (fp16)

   -> Transpose to (1, ~175, 1280) for the transformer (batch, sequence, features)
   |
4. Audio Encoder (32 transformer layers)
   Input: (1, 175, 1280)
   For each of the 32 layers:

     a. LayerNorm(hidden_states)
        -> Normalizes each of the 175 tokens' 1280-dimensional vectors
        -> Launches 175 thread blocks (one per token)
        -> Shape stays (1, 175, 1280)
        -> Kernel: layernorm_kernel

     b. Q = Linear(normalized)
        -> (1, 175, 1280) @ weight(1280, 1280) = (1, 175, 1280)
        -> Then reshape to (1, 20, 175, 64) -- 20 heads, head_dim=64
        -> Kernel: F.linear (cuBLAS fp16 HGEMM)

     c. K = Linear(normalized)
        -> Same as Q: (1, 175, 1280) -> (1, 20, 175, 64)

     d. V = Linear(normalized)
        -> Same: (1, 175, 1280) -> (1, 20, 175, 64)

     e. Apply partial RoPE to Q, K (50% of dimensions: dims 0-31 rotated, 32-63 unchanged)
        -> Kernel: compute_freqs_kernel (once) + torch element-wise ops

     f. Attention = Flash Attention kernel
        -> Q: (1, 20, 175, 64), K: (1, 20, 175, 64), V: (1, 20, 175, 64)
        -> BLOCK_M=64, BLOCK_N=64 on RTX 5090 (encoder, head_dim=64)
        -> Grid: 20 heads * ceil(175/64) = 20 * 3 = 60 thread blocks
        -> Each block processes 64 query rows against all 175 keys
        -> Output: (1, 20, 175, 64)

     g. Reshape back: (1, 20, 175, 64) -> (1, 175, 1280)

     h. Output projection: (1, 175, 1280) @ weight(1280, 1280) = (1, 175, 1280)

     i. Residual: hidden = original_input + attention_output
        -> Element-wise addition, shape (1, 175, 1280)

     j. LayerNorm(hidden)

     k. MLP: fc1 -> gelu -> fc2
        -> fc1: (1, 175, 1280) @ weight(1280, 5120) = (1, 175, 5120)
           Memory for intermediate: 175 * 5120 * 2 = ~1.75 MB (fp16)
        -> gelu: (1, 175, 5120) -- standalone gelu_kernel
        -> fc2: (1, 175, 5120) @ weight(5120, 1280) = (1, 175, 1280)

     l. Residual: hidden = pre_mlp_hidden + mlp_output

   Final output after 32 layers: (1, 175, 1280)
   Total layernorm_kernel calls: 32 * 2 = 64
   Total gelu_kernel calls: 32 (one per layer's MLP)
   Total cuBLAS calls: 32 * 4 (Q, K, V, O) + 32 * 2 (fc1, fc2) = 192
   Total flash_attention_kernel calls: 32
   |
5. Multi-Modal Projector
   Input: (1, 175, 1280)

   -> Pool 4 frames by concatenation:
      Take frames [0,1,2,3] and concatenate: 4 * 1280 = 5120 features
      Take frames [4,5,6,7] and concatenate: 4 * 1280 = 5120 features
      ...
      175 / 4 = ~44 pooled frames (with some edge handling)
      Output: (1, ~44, 5120)

   -> Linear(5120 -> 4096):
      (1, 44, 5120) @ weight(5120, 4096) = (1, 44, 4096)
      Kernel: cuBLAS fp16 HGEMM, output stays in fp16

   -> gelu: (1, 44, 4096) -- gelu_kernel, fp16 in/out

   -> Linear(4096 -> 2048):
      (1, 44, 4096) @ weight(4096, 2048) = (1, 44, 2048)
      Kernel: cuBLAS fp16 HGEMM

   Output: (1, ~44, 2048) -- now in the text decoder's dimension space
   |
6. Embed input tokens (chat template + audio placeholders)
   -> The chat template provides framing like "<|system|>You are a helpful assistant...<|audio|>"
   -> The <|audio|> placeholder positions are replaced with the projected audio embeddings
   -> Initial text token embeddings: (1, ~36, 2048) [template tokens]
   -> Audio embeddings: (1, ~44, 2048)
   -> Combined: (1, ~80, 2048) via concatenation at the audio placeholder positions
   |
7. Text Decoder (28 transformer layers)
   Input: (1, 80, 2048)

   For each of the 28 layers:
     a. RMSNorm(hidden_states)
        -> 80 thread blocks, each normalizing a 2048-dim vector
        -> Kernel: rmsnorm_kernel

     b. Q projection (16 heads):
        -> (1, 80, 2048) @ weight(2048, 2048) = (1, 80, 2048)
        -> Reshape: (1, 16, 80, 128) -- 16 heads, head_dim=128

     c. K projection (4 heads -- GQA):
        -> (1, 80, 2048) @ weight(2048, 512) = (1, 80, 512)
        -> Reshape: (1, 4, 80, 128) -- only 4 KV heads!
        -> Memory for K is 4x smaller than Q

     d. V projection (4 heads):
        -> Same as K: (1, 4, 80, 128)

     e. Apply full RoPE to Q and K
        -> All 128 dimensions of each head are rotated (not partial like encoder)

     f. Flash Attention with GQA:
        -> K is expanded from 4 heads to 16 heads (each KV head is duplicated 4 times)
        -> Q: (1, 16, 80, 128), K: (1, 16, 80, 128), V: (1, 16, 80, 128)
        -> BLOCK_M=32, BLOCK_N=32 on RTX 5090 (decoder, head_dim=128)
        -> IS_CAUSAL=True: token 5 can only attend to tokens 1-5
        -> Grid: 16 heads * ceil(80/32) = 16 * 3 = 48 thread blocks
        -> Output: (1, 16, 80, 128)

     g. Reshape + output projection:
        -> (1, 16, 80, 128) -> (1, 80, 2048) -> Linear -> (1, 80, 2048)
        -> Residual add

     h. RMSNorm(hidden)

     i. SwiGLU MLP (with fused kernel when MLP.FUSED=True):
        -> gate_proj: (1, 80, 2048) @ weight(2048, 5632) = (1, 80, 5632)
        -> up_proj: (1, 80, 2048) @ weight(2048, 5632) = (1, 80, 5632)
        -> With fusion, both projections happen in a single kernel call,
           input x is loaded from DRAM only once (saves ~80 * 2048 * 2 = 320 KB DRAM read)
        -> SiLU + multiply: (1, 80, 5632)
        -> down_proj: (1, 80, 5632) @ weight(5632, 2048) = (1, 80, 2048)
        -> Residual add

   After 28 layers: (1, 80, 2048)
   |
8. Final RMSNorm + LM Head (Language Model Head)
   -> RMSNorm: (1, 80, 2048) -- final normalization
   -> LM Head Linear: (1, 80, 2048) @ weight(2048, 59264) = (1, 80, 59264)
      This projects each token position to the FULL vocabulary size.
      59,264 logits per position -- one for each possible next token.
   -> We only care about the LAST position: (1, 59264)
      Why? Because we are predicting what comes AFTER the entire sequence.
   |
9. Autoregressive Decode
   With generate_v8b (KV-cached):

   a. Prefill: Process the initial 80 tokens through all 28 layers
      -> logits: (1, 80, 59264), but we only use logits[:, -1, :]
      -> past_kv: stores K and V for all 28 layers, all positions
         Size per layer: 2 (K and V) * 4 heads * 80 positions * 128 dim * 2 bytes = 160 KB
         Total: 28 layers * 160 KB = ~4.4 MB

   b. Decode loop (each step):
      -> Embed new token: (1, 1, 2048) -- just ONE token's embedding
      -> Process through all 28 layers, but each layer only computes Q for 1 token:
         Q: (1, 16, 1, 128) -- tiny!
         K: past_kv K for this layer + new K = (1, 4, 81, 128)
         V: past_kv V for this layer + new V = (1, 4, 81, 128)
      -> Attention uses SDPA fallback (seq_q=1 <= 4) instead of Triton Flash kernel
         This is faster because SDPA has less launch overhead for tiny queries.
      -> LM Head: (1, 1, 2048) @ weight(2048, 59264) = (1, 1, 59264)
         Only one position to project, not 80!
      -> Sample next token from the 59264 logits
      -> Generates ~13 tokens total: "Concord returned to its place amidst the tents."
   |
10. Decode token IDs -> text
    -> Token IDs like [34521, 8723, 4091, ...] are looked up in the tokenizer's
       vocabulary to produce the final string:
       "Concord returned to its place amidst the tents."
```

### Kernel Call Count (Approximate, per full inference with 13 generated tokens)

**Stock generate() -- O(n^2), no KV cache:**

Each decode step reprocesses the full growing sequence through all layers.

| Kernel | Encode | First Decode | Per Step (growing) | Total (~) |
|--------|:------:|:------------:|:------------------:|:---------:|
| layernorm_kernel | 64 | 0 | 0 | 64 |
| rmsnorm_kernel | 0 | 56 | 56 | ~784 |
| gelu_kernel | 33 | 0 | 0 | 33 |
| silu_kernel | 0 | 28 | 28 | ~392 |
| linear (cuBLAS fp16) | ~160 | ~168 | ~168 | ~2512 |
| flash_attention_kernel | 32 | 28 | 28 | ~424 |
| compute_freqs | 1 | 1 | 1 | ~15 |
| softmax (standalone) | 0 | 1 | 1 | ~14 |

**How to read this table:** The "Encode" column is the audio encoder (run once).
"First Decode" is the first pass through the text decoder. "Per Step" is each
subsequent decode step. "Total" is the sum across the full inference.

Note: gelu_kernel count = 32 from encoder (one fc1 -> gelu per layer) + 1 from
projector (linear_1 -> act). These are standalone `gelu()` calls, NOT fused
(model.py does not use the EncoderMLP or LinearGELU classes).

---

## 10. Optimization Sources

This section catalogs every optimization that was tested, adopted, or rejected,
with explanations of the physics behind WHY each one helps (or does not).

### Currently Active

| Optimization | Source | Description |
|-------------|--------|-------------|
| cuBLAS backend | **majed**, **yash/optimize** | `F.linear` for all Linear layers |
| fp16 weights | **yash/optimize**, **majed**, **meave** | Cache fp16 copies, halve memory traffic, fp16 HGEMM |
| Flash Attention | **majed**, **meave** | Triton kernel with online softmax |
| Fused SwiGLU | **yash/optimize** | Single kernel for gate+up in decoder MLP |
| TF32 flags | Common | `allow_tf32`, `set_float32_matmul_precision("high")` |

**Why cuBLAS backend helps:**
cuBLAS (NVIDIA's CUDA Basic Linear Algebra Subroutines) is the result of decades
of optimization by NVIDIA's performance engineering team. They hand-tune assembly
code for each GPU architecture, exploiting every hardware quirk: optimal memory
access patterns, instruction interleaving to hide latency, register pressure
management, and perfect tensor core utilization. Our Triton kernel is a solid
implementation, but it cannot match the hand-tuned assembly that NVIDIA's engineers
spend years perfecting for each GPU generation.

**Why fp16 weights help:**
Every weight is stored in 16 bits instead of 32 bits. This means:
1. **Half the memory bandwidth**: The GPU can load weights 2x faster because each
   weight is half the size. Since matrix multiplication is often "memory-bound"
   (limited by how fast data can be fetched, not by computation speed), this can
   nearly double performance for some operations.
2. **Half the memory usage**: Weights take up half the GPU memory, leaving more room
   for activations and KV cache.
3. **Faster tensor core operations**: NVIDIA's tensor cores are optimized for fp16
   matrix multiplication (HGEMM -- Half-precision General Matrix Multiply). They
   can process fp16 operations faster than fp32.

**Why Flash Attention helps:**
See section 3.2 for the full explanation. In brief: it eliminates the O(N^2) memory
requirement for the attention score matrix by computing softmax incrementally, keeping
all intermediate data in fast SRAM instead of slow DRAM.

**Why Fused SwiGLU helps:**
The decoder MLP computes SiLU(x @ W_gate) * (x @ W_up). Without fusion, the input
x is read from DRAM twice (once for each matrix multiply). With fusion, x is read
once and used for both. For a (80, 2048) input in fp16, this saves 80 * 2048 * 2 =
327,680 bytes of DRAM bandwidth per MLP per layer, times 28 layers = ~9.2 MB saved
per decode step.

**Why TF32 flags help:**
TF32 (TensorFloat-32) is a special floating-point format that uses the range of
fp32 (8 exponent bits) with reduced precision (10 mantissa bits instead of 23).
Modern NVIDIA GPUs can do TF32 operations at nearly the speed of fp16, while
maintaining much of fp32's numerical range. Setting `allow_tf32=True` enables this
for any operations that happen in fp32.


### Adopted (tested, confirmed improvement)

| Optimization | Source | Actual Impact |
|-------------|--------|---------------|
| Fused Q+K RoPE pair kernel | **meave** | **-14ms** (138 -> 124ms) |
| bf16 RMSNorm output kernel | **meave** (adapted for bf16) | **-3ms** (124 -> 121ms) |

**Why fused Q+K RoPE helps (-14ms):**
RoPE must be applied to both Q and K vectors. Without fusion, this requires two
separate kernel launches and two separate passes over the cos/sin tables. Fusing
them into one kernel means: (1) only one kernel launch overhead instead of two,
and (2) the cos/sin values are loaded from memory once and applied to both Q and K.
The 14ms improvement is significant because RoPE is applied in every one of the
32+28 = 60 attention layers.

**Why bf16 RMSNorm output helps (-3ms):**
The original RMSNorm kernel computed in fp32 and stored the result in fp32. Converting
the output to bf16 (or fp16) inside the kernel itself means the downstream operations
receive half-precision data directly, avoiding a separate dtype conversion step and
halving the DRAM write bandwidth for the normalization output.


### Adopted (2026-03-13 to 2026-03-15)

| Optimization | Source | Actual Impact |
|-------------|--------|---------------|
| bf16 LayerNorm output | internal | **-0.7ms** (encoder norm stores bf16 directly) |
| generate_v8b (KV cache) | internal | **-7.6ms** (monkey-patched from layers.py) |
| SDPA fallback for seq_q<=4 | internal | **-3ms** (PyTorch SDPA for KV-cached decode steps) |
| GPUProfile + _KNOWN_CONFIGS + dynamic tiles | internal | portability (7 arch classifications, dynamic fallback) |
| Dead code cleanup | internal | -320 lines (removed legacy attention kernels) |

**Why generate_v8b (KV cache) helps (-7.6ms):**
See section 5.5 for the full explanation. Eliminates redundant recomputation of
attention keys and values for previously processed tokens. Reduces total
layer-applications from ~31,304 to ~2,604 (12x reduction for 13 generated tokens).

**Why SDPA fallback for seq_q<=4 helps (-3ms):**
During KV-cached decoding, each step has only 1 query token (seq_q=1). The Triton
Flash Attention kernel has fixed overhead per launch: compilation (first time), grid
setup, and minimum work granularity. For a single query row, this overhead dominates
the actual computation. PyTorch's built-in SDPA uses cuDNN internally, which has
a more efficient code path for tiny queries (essentially a batched vector-matrix
multiply). The savings of ~0.23ms per step add up over 13 decode steps.

**Why GPUProfile helps (portability):**
Without GPUProfile, a single set of tile sizes would either (a) be too large for
consumer GPUs (crash) or (b) be too small for datacenter GPUs (waste performance).
GPUProfile detects the GPU at import time and selects optimal tiles, making the
same code run well on an RTX 4090, RTX 5090, H100, H200, B200, or any future GPU.

**Why dead code cleanup helps (-320 lines):**
Removing unused code has no runtime performance impact, but it reduces cognitive
overhead for developers, eliminates potential confusion about which code paths are
active, and slightly speeds up Triton's compilation (fewer kernel definitions to
parse).


### Adopted (2026-03-15, fp16-throughout pipeline)

| Optimization | Source | Actual Impact |
|-------------|--------|---------------|
| fp16 cuBLAS HGEMM (was bf16) | internal | ~-0.4ms (fp16 HGEMM slightly faster on RTX 5090) |
| Smaller flash attention tiles | **meave** | improved prefill (64x64 encoder, 32x32 decoder) |
| Remove Linear `.float()` conversion | internal | **-7.5ms** (biggest single win) |
| Remove silu/gelu float32 cast | internal | **-3.7ms** |
| Remove RMSNorm/LayerNorm float32 cast | internal | ~-0.5ms |
| fp16 embedding output | internal | keeps decoder pipeline in fp16 from start |
| fp16 fused SwiGLU/EncoderMLP | internal | halves intermediate memory bandwidth |
| Remove flash attention float32 conversion | internal | ~-1ms |
| Norm kernel output fp16 (was bf16) | internal | matches fp16 pipeline |
| BLOCK_M=16 for seq_q<=16 | **meave** | optimized for KV-cached decode |
| topk instead of argsort in sampling | internal | neutral (cleaner code) |

**Why removing `.float()` conversion is the biggest win (-7.5ms):**

The original code had `output = (input_half @ weight_half).float()` in every Linear
layer. The `.float()` call converts the result from fp16 to fp32. This seems harmless,
but it has three compounding costs:

1. **The conversion itself**: For a (80, 2048) output, converting from fp16 to fp32
   means reading 80 * 2048 * 2 = 320 KB and writing 80 * 2048 * 4 = 640 KB. With ~168
   linear layers per decode step, that is ~168 * (320 + 640) KB = ~157 MB of extra
   memory traffic per decode step.

2. **Cascading effect**: Once converted to fp32, ALL downstream operations (RMSNorm,
   attention, MLP) must also operate in fp32. This doubles the memory bandwidth for
   EVERY subsequent operation until the data re-enters a Linear layer.

3. **Larger tensors in flight**: fp32 tensors are 2x larger, consuming more GPU cache
   space and evicting other useful data from the cache.

Removing `.float()` lets the fp16 result flow directly to downstream operations,
which handle precision internally (Triton kernels convert to fp32 for computation
using `.to(tl.float32)` and output fp16).

**Why removing silu/gelu float32 cast helps (-3.7ms):**
Same principle as above. The original activation functions converted inputs to fp32
before computing. But the Triton kernels already promote to fp32 internally for the
actual arithmetic (to maintain numerical precision in intermediate computations like
exp() and tanh()). The external fp32 cast just added an unnecessary conversion step
and forced the kernel to read fp32 data (2x larger) from DRAM.

**Why BLOCK_M=16 for seq_q<=16 helps:**
During KV-cached decode, seq_q=1. If BLOCK_M=64, the kernel allocates resources for
64 query rows but only uses 1. The other 63 rows are wasted computation and wasted
shared memory. Setting BLOCK_M=16 reduces this waste from 63/64 = 98.4% to
15/16 = 93.8% -- still wasteful, but less so, and the smaller shared memory footprint
allows more thread blocks to run simultaneously on the GPU.


### Rejected (tested, did not help on RTX 5090)

| Optimization | Source | Result |
|-------------|--------|--------|
| SwiGLU grid swizzling | **yash/optimize** | +18ms regression with GROUP_SIZE_M=8, 1D grid |
| @triton.autotune GELU/SiLU | **majed** | +0.7ms tuning overhead |
| @triton.autotune Flash Attention | internal | Massive regression |
| @triton.autotune SwiGLU | internal | Regression |
| Softmax bf16 output | internal | 0ms (not in hot path) |
| Flash Attention num_stages=2 | **yash/optimize** | OOM on consumer GPUs |
| Flash Attention num_warps=8 | **yash/optimize** | 0ms change on RTX 5090 |
| PyTorch GELU/SiLU bf16 | internal | +0.3ms |
| PyTorch SDPA for prefill/encoder | internal | +6ms |
| SDPA enable_gqa=True for decode | internal | +13ms |
| Fused gate+up Linear in MLP | internal | Neutral |

**Why SwiGLU grid swizzling hurt (+18ms):**
Grid swizzling reorders how thread blocks are assigned to GPU cores, aiming to improve
L2 cache hit rates by having nearby thread blocks process nearby data. However, the
RTX 5090's L2 cache is already large enough (96 MB) that the default block ordering
achieves good cache utilization. The swizzling overhead (extra index computation per
block) outweighed any cache benefit.

**Why @triton.autotune hurt (+0.7ms to massive regression):**
Triton's autotune feature tries multiple kernel configurations at runtime and picks
the fastest. This sounds great, but:
1. **Tuning overhead**: The first call must try ALL configurations, adding latency.
2. **Key sensitivity**: Autotune caches the best config for each set of "key" parameters.
   With KV caching, seq_k changes every decode step (81, 82, 83, ...), causing
   autotune to re-tune for EVERY step -- catastrophic for performance.
3. **Small matrices**: For decoder decode steps where matrices are small (seq_q=1),
   the autotuning overhead dominates the actual computation time.

**Why Flash Attention num_stages=2 caused OOM:**
`num_stages` controls software pipelining -- how many tiles are prefetched while the
current tile is being computed. More stages = more overlap of memory loads with
computation = potentially faster. But each additional stage requires storing one
more tile in shared memory. On consumer GPUs with ~100KB shared memory, 2 stages
of attention tiles exceed the available space, causing an out-of-memory error.
Datacenter GPUs with ~228KB can handle it, which is why they use num_stages=2 in
_KNOWN_CONFIGS.

**Why PyTorch SDPA for prefill/encoder hurt (+6ms):**
For long sequences (175 tokens in encoder, 80 in decoder prefill), our Triton Flash
Attention kernel is faster than PyTorch's SDPA. This is because our kernel is
specifically tuned for our exact workload (tile sizes chosen by GPUProfile), while
PyTorch's SDPA uses a general-purpose configuration. The SDPA fallback only helps
for tiny queries (seq_q<=4) where Triton's launch overhead dominates.

**Why SDPA enable_gqa=True hurt (+13ms):**
PyTorch's SDPA has a built-in GQA mode (`enable_gqa=True`) that should handle the
4-to-16 head expansion internally. However, its implementation is slower than our
approach of manually expanding KV heads with `_expand_kv_heads()` and then calling
standard SDPA. The manual expansion is a simple memory copy, while SDPA's internal
GQA adds branching and index computation overhead inside the kernel.


### Not Applicable

| Optimization | Source | Why Not |
|-------------|--------|---------|
| EncoderMLP.FUSED | yash/optimize | model.py (origin/main) does not use EncoderMLP class |
| LinearGELU.FUSED | yash/optimize | model.py (origin/main) does not use LinearGELU class |
| flash_decode_kernel | meave | generate_v8b uses same flash_attention_kernel for decode |

**Why EncoderMLP.FUSED is not applicable:** The `EncoderMLP` class exists in layers.py
and supports fused fc1+GELU computation. However, model.py (which is read-only) does
NOT use `EncoderMLP`. Instead, it creates plain `Linear` layers and calls `gelu()`
inline. Since we cannot modify model.py, the fused `EncoderMLP` class is dead code --
it works correctly but nothing in the model calls it.

**Why LinearGELU.FUSED is not applicable:** Same reason as EncoderMLP. The
`LinearGELU` class exists but model.py creates plain `Linear` layers and calls
`self.act()` (which is `gelu`) inline.

**Why flash_decode_kernel is not applicable:** Some implementations use a separate,
simpler Flash Attention kernel optimized for the decode case (seq_q=1). Our
implementation instead falls back to PyTorch SDPA for decode (seq_q<=4), which
achieves the same goal -- avoiding Triton Flash Attention overhead for tiny queries.
