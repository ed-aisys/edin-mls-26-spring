# Reference Guide: GLM-ASR Triton Kernel Project (Verbose Edition)

Quick reference for kernel signatures, model architecture, and performance tuning.

**This verbose edition** adds detailed explanations of every GPU programming concept,
kernel parameter, configuration choice, and benchmark result. It is written for someone
with no prior GPU programming experience.

---

## Preliminary: Key GPU Concepts

Before diving in, here are the fundamental GPU concepts you will encounter throughout
this document. Each is explained in more detail when it first appears, but this
overview gives you a mental map.

**GPU memory hierarchy:**
- **Global memory (HBM)** — High Bandwidth Memory. This is the GPU's main memory (e.g.,
  24 GB on an RTX 5090). It is large but relatively slow to access. Think of it as the
  GPU's hard drive. Every tensor you create in PyTorch lives here by default.
- **Shared memory (SRAM)** — A small, fast scratchpad memory on each Streaming
  Multiprocessor (SM). Typically 48-228 KB depending on the GPU. Think of it as the
  GPU's L1 cache that you control manually. Data must be explicitly loaded from global
  memory into shared memory before it can be used at full speed.

**Execution model:**
- **Kernel** — A function that runs on the GPU. When you call a Triton or CUDA kernel,
  the GPU launches thousands of threads that all execute the same code on different data.
- **Thread** — The smallest unit of execution on a GPU. A single thread processes one or
  a few elements.
- **Warp** — A group of exactly 32 threads that execute in lockstep (all 32 run the same
  instruction at the same time). This is a hardware constraint of NVIDIA GPUs.
- **Block (thread block)** — A group of threads (made up of one or more warps) that
  share the same shared memory. In Triton, each "program" corresponds to one block.
- **Grid** — The collection of all blocks launched by a kernel. The grid dimensions tell
  the GPU how many blocks to run.

**Data types:**
- **fp32 (float32)** — Standard 32-bit floating point. Full precision, but uses 4 bytes
  per number.
- **fp16 (float16)** — Half-precision, 16-bit floating point. Uses 2 bytes per number,
  so memory bandwidth is halved compared to fp32. Slightly less precise.
- **bf16 (bfloat16)** — "Brain float" 16-bit. Same dynamic range as fp32 but less
  mantissa precision. Also 2 bytes per number. Preferred for training stability, but fp16
  can be faster for inference on some GPUs.
- **TF32** — A special format used internally by tensor cores. It has 19 bits (the range
  of fp32 with reduced mantissa). Enabled via runtime flags; you do not store tensors
  in TF32 explicitly.

**Hardware features:**
- **Tensor cores** — Specialized matrix multiply units on NVIDIA GPUs (Volta and newer).
  They perform small matrix multiplications (e.g., 16x16) in a single clock cycle,
  achieving roughly 10x the throughput of standard CUDA cores for matrix math.
- **Streaming Multiprocessor (SM)** — The basic compute unit of a GPU. An RTX 5090 has
  ~170 SMs; an H200 MIG partition might have 60. Each SM has its own shared memory
  and runs one or more thread blocks concurrently.

**Libraries:**
- **cuBLAS** — NVIDIA's vendor-optimized library for dense linear algebra (matrix
  multiply, etc.). Years of hand-tuning make it extremely fast.
- **HGEMM** — Half-precision General Matrix Multiply. This is cuBLAS's matrix multiply
  using fp16 or bf16 inputs, leveraging tensor cores for maximum throughput.
- **Triton** — A Python-based language for writing GPU kernels. Easier than raw CUDA
  because it handles thread scheduling and memory access patterns for you, while still
  giving you control over tiling and shared memory usage.
- **SDPA** — Scaled Dot-Product Attention. PyTorch's built-in `torch.nn.functional.scaled_dot_product_attention`, which automatically selects an efficient backend (flash attention, memory-efficient attention, or math fallback).

---

## Model Configuration (GLM-ASR-Nano-2512)

This table describes the architecture of the GLM-ASR speech recognition model. The model
has three main components: an audio encoder that processes speech, a projector that
bridges audio and text representations, and a text decoder that generates transcriptions.

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Audio Encoder** | Hidden size | 1280 |
| | Num heads | 20 |
| | Head dim | 64 |
| | Num layers | 32 |
| | Intermediate | 5120 |
| | Norm | LayerNorm |
| | Activation | GELU |
| | RoPE | 50% partial (rotary_dim=32) |
| | MLP | Plain `fc1 -> gelu -> fc2` (not EncoderMLP class) |
| **Projector** | Pool factor | 4 |
| | Hidden | 5120 -> 4096 -> 2048 |
| | Uses | Plain `Linear -> gelu -> Linear` (not LinearGELU class) |
| **Text Decoder** | Hidden size | 2048 |
| | Q heads | 16 |
| | KV heads | 4 (GQA, 4:1 ratio) |
| | Head dim | 128 |
| | Num layers | 28 |
| | Intermediate | 6144 |
| | Norm | RMSNorm |
| | Activation | SiLU/SwiGLU |
| | RoPE | 100% (rotary_dim=128, base=500000) |
| **LM Head** | Vocab size | 59264 |

> **What does this mean?**
>
> - **Hidden size** is the width of each tensor flowing through the network. Larger =
>   more capacity but more computation. The encoder uses 1280; the decoder uses 2048.
> - **Num heads** is how many independent "attention heads" run in parallel. Each head
>   looks at the input from a different perspective. The encoder has 20 heads, each
>   seeing a 64-dimensional slice (20 x 64 = 1280).
> - **Head dim** is the size of each attention head's "view." Smaller head dims (64) are
>   cheaper to compute; larger ones (128) capture more information per head.
> - **GQA (Grouped Query Attention)** means the decoder uses fewer key/value heads (4)
>   than query heads (16). Each KV head is shared across 4 query heads. This saves
>   memory and computation without much accuracy loss.
> - **RoPE (Rotary Position Embeddings)** encodes position information by rotating the
>   query and key vectors. "50% partial" means only half the dimensions are rotated
>   (the encoder), while "100%" means all dimensions are rotated (the decoder).
> - **SwiGLU** is an activation function that uses a gating mechanism: it multiplies two
>   linear projections together (one through a SiLU activation). This requires two
>   weight matrices ("gate" and "up") instead of one, hence the larger intermediate size.
> - **Intermediate** is the hidden size of the MLP (feed-forward) layer between attention
>   layers. It is typically 4x the hidden size.
> - **Vocab size** (59264) is the number of possible output tokens. The LM Head projects
>   from hidden size (2048) to vocab size to produce a probability for each token.

---

## Files: What You Can and Cannot Modify

| File | Modifiable? | What's In It |
|------|:-----------:|--------------|
| `layers.py` | **Yes** | All 6 layer kernels + config knobs + fused kernels + layer classes |
| `attention.py` | **Yes** | Fused Flash Attention kernel + SDPA fallback (legacy kernels removed) |
| `rope.py` | **Yes** | 1 RoPE kernel |
| `__init__.py` | **Yes** | Backend/fusion configuration |
| `model.py` | **No** | Model architecture, stock `generate()` (O(n^2), KV cache infra exists but unused) |
| `weight_loader.py` | **No** | HuggingFace weight loading |
| `conv.py` | **No** | Conv1D for audio subsampling |

> **What does this mean?**
>
> - The "Do Not Modify" files define the model architecture and data loading. Changing
>   them would break compatibility with the grading infrastructure.
> - Your optimization work happens in `layers.py` (where the core compute kernels live),
>   `attention.py` (the attention mechanism), `rope.py` (positional encoding), and
>   `__init__.py` (configuration switches).
> - `model.py` uses a stock `generate()` function that is O(n^2) — meaning it recomputes
>   the entire sequence at every decoding step. This is the dominant bottleneck.

---

## Kernel Signatures

A **kernel signature** lists all the parameters a GPU function accepts. Understanding
each parameter is essential for debugging and optimization.

### layers.py

```python
# RMSNorm -- Grid: (num_rows,)
rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)

# LayerNorm -- Grid: (num_rows,)
layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)

# GELU -- Grid: (cdiv(n_elements, BLOCK_SIZE),)
gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)

# SiLU -- Grid: (cdiv(n_elements, BLOCK_SIZE),)
silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)

# Linear -- Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
linear_kernel_tf32(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K)

# Softmax -- Grid: (num_rows,)
softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE)
```

#### RMSNorm Kernel — Parameter-by-parameter

RMSNorm (Root Mean Square Normalization) normalizes each row of a tensor by dividing by
the root mean square of that row's values, then scaling by learned weights. It is simpler
than LayerNorm because it does not subtract the mean (no centering).

```
rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `x_ptr` | pointer | Address of the input tensor in GPU global memory (HBM). Each row has `hidden_size` elements. |
| `w_ptr` | pointer | Address of the learned weight (scale) vector, shape `(hidden_size,)`. Each element scales the corresponding feature after normalization. |
| `y_ptr` | pointer | Address of the output tensor in GPU global memory. Same shape as `x_ptr`. |
| `stride_x` | int | Number of elements to skip to move from one row to the next in `x`. For a contiguous 2D tensor of shape `(rows, hidden_size)`, this equals `hidden_size`. Needed because tensors can be non-contiguous in memory. |
| `stride_y` | int | Same as `stride_x`, but for the output tensor `y`. |
| `hidden_size` | int | Number of elements per row (e.g., 1280 for the encoder, 2048 for the decoder). This tells the kernel how many elements to process per row. |
| `eps` | float | A tiny constant (e.g., 1e-5) added inside the square root to prevent division by zero. |
| `BLOCK_SIZE` | constexpr int | A compile-time constant specifying how many elements each thread block processes at once. Must be >= `hidden_size`. This is a **tile size** — the kernel loads `BLOCK_SIZE` elements into fast shared memory at a time. |

**Grid: `(num_rows,)`** — One thread block per row. If the input is `(batch*seq, hidden_size)`, there is one block for each of the `batch*seq` rows, all running in parallel.

#### LayerNorm Kernel — Parameter-by-parameter

LayerNorm (Layer Normalization) normalizes each row by subtracting the mean and dividing
by the standard deviation, then applying a learned scale and bias. Used in the audio
encoder.

```
layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `x_ptr` | pointer | Input tensor address in global memory. |
| `w_ptr` | pointer | Learned scale (gamma) vector, shape `(hidden_size,)`. |
| `b_ptr` | pointer | Learned bias (beta) vector, shape `(hidden_size,)`. This is the extra parameter LayerNorm has over RMSNorm — it shifts the output after scaling. |
| `y_ptr` | pointer | Output tensor address. |
| `stride_x` | int | Row stride for input. |
| `stride_y` | int | Row stride for output. |
| `hidden_size` | int | Number of features per row. |
| `eps` | float | Small constant to prevent division by zero. |
| `BLOCK_SIZE` | constexpr int | Tile size — number of elements processed per block. |

**Grid: `(num_rows,)`** — Same as RMSNorm: one block per row.

#### GELU Kernel — Parameter-by-parameter

GELU (Gaussian Error Linear Unit) is an activation function: `gelu(x) = x * Phi(x)`
where Phi is the cumulative distribution function of the standard normal distribution.
It is a smooth approximation of ReLU. Used in the audio encoder's MLP.

```
gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `x_ptr` | pointer | Input tensor address. The tensor is treated as a flat 1D array. |
| `y_ptr` | pointer | Output tensor address (same shape as input). |
| `n_elements` | int | Total number of elements in the tensor. The kernel needs this to know when to stop (so threads past the end do not read garbage memory). |
| `BLOCK_SIZE` | constexpr int | How many elements each block processes. |

**Grid: `(cdiv(n_elements, BLOCK_SIZE),)`** — "cdiv" means ceiling division. We launch
enough blocks so that `num_blocks * BLOCK_SIZE >= n_elements`. Each block handles one
chunk of `BLOCK_SIZE` elements. The last block may have some threads that are masked off
(doing no work) if `n_elements` is not a perfect multiple of `BLOCK_SIZE`.

#### SiLU Kernel — Parameter-by-parameter

SiLU (Sigmoid Linear Unit), also called "swish", computes `silu(x) = x * sigmoid(x)`.
Used in the text decoder's SwiGLU MLP.

```
silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)
```

Parameters are identical to GELU — the only difference is the mathematical function
applied to each element.

#### Linear (Matrix Multiply) Kernel — Parameter-by-parameter

This kernel computes `C = A @ B` (matrix multiplication). It is the most compute-intensive
operation in the model. Each linear projection (Q, K, V, output, MLP layers) uses this.

```
linear_kernel_tf32(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `a_ptr` | pointer | Input matrix A, shape `(M, K)`. For a linear layer, this is the activations. |
| `b_ptr` | pointer | Weight matrix B, shape `(K, N)`. These are the learned weights. |
| `c_ptr` | pointer | Output matrix C, shape `(M, N)`. The result of A @ B. |
| `M` | int | Number of rows in A (and C). Typically `batch * seq_len`. |
| `N` | int | Number of columns in B (and C). This is the output feature dimension. |
| `K` | int | Inner dimension — columns of A and rows of B. This is the input feature dimension. |
| `stride_am` | int | Stride to move one row down in A. |
| `stride_ak` | int | Stride to move one column right in A. |
| `stride_bk` | int | Stride to move one row down in B. |
| `stride_bn` | int | Stride to move one column right in B. |
| `stride_cm` | int | Stride to move one row down in C. |
| `stride_cn` | int | Stride to move one column right in C. |
| `BLOCK_M` | constexpr int | Tile height for matrix A — how many rows of A each block processes. |
| `BLOCK_N` | constexpr int | Tile width for matrix B — how many columns of B each block processes. |
| `BLOCK_K` | constexpr int | Tile depth — the inner dimension is processed in chunks of BLOCK_K. |

**Grid: `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`** — This is a 2D grid. Each block
computes a `BLOCK_M x BLOCK_N` tile of the output matrix C. The block iterates over
the K dimension in chunks of `BLOCK_K`, loading tiles of A and B into shared memory,
multiplying them (using tensor cores via `tl.dot`), and accumulating the result. This
**tiled approach** is essential: without it, each element of C would require reading the
entire row of A and column of B from slow global memory.

**"TF32" in the name** refers to the TensorFloat-32 format used internally by tensor
cores. The inputs are still stored as fp16 or fp32, but tensor cores perform the
multiply-accumulate in TF32 precision, giving a good balance of speed and accuracy.

> **What does this mean?**
>
> Matrix multiplication is the backbone of neural networks. Every linear layer, every
> attention projection, and the LM head all boil down to matmuls. The tiled algorithm
> breaks large matrices into small blocks that fit in shared memory (SRAM), reusing each
> loaded value many times. This turns a memory-bandwidth-bound operation into a
> compute-bound one, which is what you want on a GPU with thousands of compute units.

#### Softmax Kernel — Parameter-by-parameter

Softmax converts a row of raw scores into a probability distribution (all values between
0 and 1 that sum to 1). It is numerically stabilized by subtracting the row maximum
before exponentiating.

```
softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `x_ptr` | pointer | Input tensor (logits) in global memory. |
| `y_ptr` | pointer | Output tensor (probabilities) in global memory. |
| `stride_x` | int | Row stride for input. |
| `stride_y` | int | Row stride for output. |
| `n_cols` | int | Number of columns per row (i.e., the number of classes or sequence positions to softmax over). |
| `BLOCK_SIZE` | constexpr int | Tile size for processing columns. |

**Grid: `(num_rows,)`** — One block per row. Each block finds the row's maximum, subtracts
it, exponentiates, sums, and divides — all in shared memory for speed.

### attention.py

```python
# Flash Attention (PRIMARY) -- Grid: (cdiv(seq_q, BLOCK_M), batch_heads)
flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, scale,
                       seq_q, seq_k, head_dim,
                       stride_qb..qd, stride_kb..kd, stride_vb..vd,
                       stride_ob..od, stride_mb..mk,
                       IS_CAUSAL, HAS_MASK, BLOCK_M, BLOCK_N, BLOCK_D)

# SDPA fallback for KV-cached decode (seq_q <= 4):
# torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)

# Legacy kernels (attention_scores, softmax_inplace, attention_output,
# causal_mask) were REMOVED -- superseded by flash_attention_kernel.
```

#### Flash Attention Kernel — Parameter-by-parameter

Flash Attention is an algorithm that computes attention without materializing the full
`seq_q x seq_k` attention matrix in global memory. Instead, it processes the attention
in tiles, keeping intermediate results in shared memory. This reduces memory usage from
O(n^2) to O(n) and is faster because it avoids slow global memory reads/writes for the
attention matrix.

```
flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, scale,
                       seq_q, seq_k, head_dim,
                       stride_qb..qd, stride_kb..kd, stride_vb..vd,
                       stride_ob..od, stride_mb..mk,
                       IS_CAUSAL, HAS_MASK, BLOCK_M, BLOCK_N, BLOCK_D)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `q_ptr` | pointer | Query tensor address. Shape: `(batch*heads, seq_q, head_dim)`. |
| `k_ptr` | pointer | Key tensor address. Shape: `(batch*heads, seq_k, head_dim)`. |
| `v_ptr` | pointer | Value tensor address. Shape: `(batch*heads, seq_k, head_dim)`. |
| `o_ptr` | pointer | Output tensor address. Shape: `(batch*heads, seq_q, head_dim)`. |
| `mask_ptr` | pointer | Optional attention mask address. Used when `HAS_MASK=True`. |
| `scale` | float | Scaling factor, typically `1/sqrt(head_dim)`. Prevents dot products from growing too large with high dimensionality. |
| `seq_q` | int | Length of the query sequence. |
| `seq_k` | int | Length of the key/value sequence (may differ from seq_q with KV cache). |
| `head_dim` | int | Dimension of each attention head (64 for encoder, 128 for decoder). |
| `stride_qb..qd` | int | Strides for Q tensor — batch, sequence, and head_dim dimensions. These tell the kernel how to navigate the multi-dimensional tensor in flat memory. |
| `stride_kb..kd` | int | Strides for K tensor. |
| `stride_vb..vd` | int | Strides for V tensor. |
| `stride_ob..od` | int | Strides for output tensor. |
| `stride_mb..mk` | int | Strides for mask tensor. |
| `IS_CAUSAL` | constexpr bool | If True, applies causal masking — each query position can only attend to positions at or before it. This is essential for autoregressive text generation (the decoder cannot see future tokens). |
| `HAS_MASK` | constexpr bool | If True, a custom attention mask is applied. |
| `BLOCK_M` | constexpr int | Tile size for query sequence dimension. Each block processes BLOCK_M query positions. |
| `BLOCK_N` | constexpr int | Tile size for key sequence dimension. The inner loop iterates over key positions in chunks of BLOCK_N. |
| `BLOCK_D` | constexpr int | Tile size for the head dimension. Typically equals head_dim. |

**Grid: `(cdiv(seq_q, BLOCK_M), batch_heads)`** — A 2D grid. The first dimension tiles
over query positions; the second dimension handles each (batch, head) pair independently.
Each block loads a tile of Q (BLOCK_M x head_dim) and iterates over all K/V tiles
(BLOCK_N x head_dim), computing attention scores, applying softmax (online/incremental),
and accumulating the weighted sum of V.

**"Online softmax"** means the kernel computes softmax incrementally as it processes each
K/V tile, without needing a separate pass over the entire attention matrix. This is the
key algorithmic trick that makes Flash Attention memory-efficient.

**SDPA fallback** — For very short query sequences (seq_q <= 4, which happens during
KV-cached decode when generating one token at a time), PyTorch's built-in
`scaled_dot_product_attention` is used instead. This avoids the fixed overhead of
launching a Triton kernel for such a tiny computation.

> **What does this mean?**
>
> Standard attention computes `softmax(Q @ K^T / sqrt(d)) @ V`, which creates an
> intermediate `seq_q x seq_k` matrix. For long sequences, this matrix is huge and
> must be stored in slow global memory. Flash Attention avoids this by never
> materializing the full matrix — it processes tiles in fast shared memory and
> accumulates results on the fly. The result is mathematically identical but uses
> far less memory and is significantly faster.

### rope.py

```python
# RoPE Frequencies -- Grid: (seq_len,)
compute_freqs_kernel(positions_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                     seq_len, half_dim,
                     stride_pos, stride_inv,
                     stride_cos0, stride_cos1, stride_sin0, stride_sin1,
                     BLOCK)
```

#### RoPE Frequency Kernel — Parameter-by-parameter

RoPE (Rotary Position Embeddings) encodes sequence position information by rotating
query and key vectors. Instead of adding position information, it multiplies pairs of
dimensions by rotation matrices (using cos and sin). This approach preserves the relative
distance between positions in the attention dot product.

| Parameter | Type | Meaning |
|-----------|------|---------|
| `positions_ptr` | pointer | Integer positions for each sequence element (e.g., [0, 1, 2, ...]). |
| `inv_freq_ptr` | pointer | Precomputed inverse frequency vector, shape `(half_dim,)`. Each element is `1 / (base^(2i/d))` where `base` is the RoPE base frequency and `i` is the dimension index. |
| `cos_ptr` | pointer | Output: cosine of (position * inv_freq), shape `(seq_len, half_dim)`. |
| `sin_ptr` | pointer | Output: sine of (position * inv_freq), shape `(seq_len, half_dim)`. |
| `seq_len` | int | Number of sequence positions. |
| `half_dim` | int | Half the rotary dimension (since rotation operates on pairs of dimensions). For the encoder: 16 (rotary_dim=32, so half=16). For the decoder: 64 (rotary_dim=128, so half=64). |
| `stride_pos` | int | Stride of the positions tensor. |
| `stride_inv` | int | Stride of the inverse frequency tensor. |
| `stride_cos0, stride_cos1` | int | Row and column strides for the cosine output. |
| `stride_sin0, stride_sin1` | int | Row and column strides for the sine output. |
| `BLOCK` | constexpr int | Tile size for the half_dim dimension — how many frequency values each block processes at once. |

**Grid: `(seq_len,)`** — One block per sequence position. Each block computes cos and sin
for all frequency dimensions at that position.

> **What does this mean?**
>
> RoPE is an elegant way to tell the model "where" each token is in the sequence.
> Instead of adding a position vector (like in the original Transformer), RoPE
> rotates the query and key vectors so that their dot product naturally depends on
> the relative distance between positions. The cos/sin outputs from this kernel are
> later applied to Q and K vectors before attention.

---

## Tensor Shapes Through the Pipeline

This section traces the exact tensor shapes as data flows through the model. Understanding
shapes helps you debug kernels and reason about performance.

```
Input: audio_array (float32, 16kHz, ~3.5s for test audio)
```

The raw input is a 1D audio waveform sampled at 16,000 samples per second.

```
Mel Spectrogram:     (1, 128, T)           # T depends on audio length
```
**Dimensions: (batch, mel_channels, time_frames)**
The audio is converted to a spectrogram — a 2D representation of frequency content over
time. 128 "mel channels" are frequency bins spaced according to human perception. T is
the number of time frames (depends on audio duration).

```
Conv1 output:        (1, 1280, T)          # Feature expansion + GELU
Conv2 output:        (1, 1280, T/2)        # Stride 2 + GELU
```
**Dimensions: (batch, hidden_size, time_frames)**
Two 1D convolutions process the spectrogram. Conv1 expands from 128 to 1280 channels.
Conv2 keeps 1280 channels but halves the time dimension (stride 2), reducing the sequence
length by 2x.

```
Permute:             (1, T/2, 1280)        # (batch, seq, hidden)
```
**Dimensions: (batch, sequence_length, hidden_size)**
The tensor is rearranged from "channels first" (used by convolutions) to "sequence first"
(used by transformers). Now each row is a 1280-dimensional feature vector for one time step.

```
Encoder (32 layers):
  Q/K/V proj:        (1, T/2, 1280)        # Linear (cuBLAS fp16 HGEMM)
```
**Dimensions: (batch, seq_len, hidden_size)**
Each attention layer projects the input into Query, Key, and Value tensors using matrix
multiplication. This is done using cuBLAS (NVIDIA's optimized library) with fp16 HGEMM
(half-precision general matrix multiply) for maximum throughput on tensor cores.

```
  Reshape:           (1, 20, T/2, 64)      # 20 heads, head_dim=64
```
**Dimensions: (batch, num_heads, seq_len, head_dim)**
The hidden dimension (1280) is reshaped into 20 attention heads, each with 64 dimensions.
This lets each head attend to different aspects of the input independently.

```
  RoPE:              Partial (first 32 dims rotated)
```
Only the first 32 out of 64 head dimensions are rotated by RoPE. The remaining 32 are
left unchanged. This "partial" RoPE is a design choice of this model.

```
  Attention:         (1, 20, T/2, 64)      # Flash Attention kernel
```
**Dimensions: (batch, num_heads, seq_len, head_dim)**
The Flash Attention kernel computes `softmax(Q @ K^T / sqrt(64)) @ V` without
materializing the full attention matrix. Output shape matches Q.

```
  MLP:               fc1(x) -> gelu(x) -> fc2(x)  # Plain Linear + gelu, NOT fused
```
The MLP (Multi-Layer Perceptron) expands from 1280 to 5120 (fc1), applies GELU activation,
then projects back down to 1280 (fc2). "NOT fused" means each operation is a separate
kernel launch. The encoder does NOT use the fused EncoderMLP class.

```
Encoder output:      (1, T/2, 1280)
Pool 4 frames:       (1, T/8, 5120)        # Concatenate 4 consecutive frames
```
**Dimensions: (batch, seq_len/4, hidden_size*4)**
The projector pools every 4 consecutive frames by concatenating them, reducing the
sequence length by 4x while increasing the feature dimension by 4x
(1280 * 4 = 5120). This compresses the audio representation before passing it to the
text decoder.

```
Projector:           (1, T/8, 2048)        # Linear->gelu->Linear (plain, NOT fused)
```
**Dimensions: (batch, seq_len, decoder_hidden_size)**
Two linear layers with GELU activation project from 5120 down to 2048 (the decoder's
hidden size). This bridges the audio and text representations.

```
Decoder input:       (1, N_tokens, 2048)   # Audio + text token embeddings
```
**Dimensions: (batch, total_tokens, hidden_size)**
The projected audio features are concatenated with text token embeddings. N_tokens is
the total number of tokens (audio frames + text tokens).

```
Decoder (28 layers):
  Q proj:            (1, N, 2048)          # 16 Q heads x 128 dim
  K/V proj:          (1, N, 512)           # 4 KV heads x 128 dim (GQA)
```
**Notice the asymmetry:** Q has 16 heads (16 * 128 = 2048) but K/V only have 4 heads
(4 * 128 = 512). This is GQA (Grouped Query Attention). Before attention, the 4 KV heads
are expanded to 16 by repeating each KV head 4 times (via `_expand_kv_heads`).

```
  Reshape Q:         (1, 16, N, 128)
  Reshape KV:        (1, 4, N, 128)
  Attention:         (1, 16, N, 128)       # Flash Attention (GQA via _expand_kv_heads)
```
**Dimensions: (batch, num_heads, seq_len, head_dim)**
Flash Attention runs with 16 heads. KV heads are expanded from 4 to 16 before the kernel.

```
  MLP (SwiGLU):      Fused when MLP.FUSED=True
```
The decoder MLP uses SwiGLU: `silu(gate(x)) * up(x)` followed by `down(...)`. When
`MLP.FUSED=True`, the silu and elementwise multiplication are combined into a single
kernel launch, avoiding an extra round-trip to global memory.

```
LM Head:             (1, N, 59264)         # Vocab logits
```
**Dimensions: (batch, seq_len, vocab_size)**
The final linear layer projects each position to a 59264-dimensional vector. These are
the "logits" — raw scores for each vocabulary token. The highest score at the last
position determines the next token.

```
Stock generate() -- O(n^2) decode:
  Each step: embed new token, concatenate to inputs_embeds, reprocess ALL through decoder
  No KV cache -- full sequence recomputed each step
```

> **What does this mean?**
>
> **O(n^2) decode** is the key bottleneck. In the stock `generate()` function, every time
> the model generates one new token, it re-runs the ENTIRE sequence through all 28 decoder
> layers. If generating 13 tokens, step 1 processes N tokens, step 2 processes N+1, step 3
> processes N+2, and so on. The total work is proportional to N + (N+1) + (N+2) + ... =
> O(n^2).
>
> **KV cache** is the standard solution: after the first pass, you save the Key and Value
> tensors for all past positions. On subsequent steps, you only compute Q/K/V for the
> NEW token and reuse the cached K/V. This makes each step O(1) instead of O(n),
> reducing total decode from O(n^2) to O(n). The project's `generate_v8b` implements this.

---

## Configuration Knobs (in __init__.py and layers.py)

Configuration knobs are settings that change how the kernels operate without modifying
kernel code. Choosing the right settings is critical for performance.

### Backend Selection
```python
layers.Linear.BACKEND = "torch"    # cuBLAS/cuBLASLt (current, fastest)
layers.Linear.BACKEND = "triton"   # strict linear-kernel path

layers.MLP.FUSED = True            # Fused SwiGLU (decoder MLP) -- EFFECTIVE
layers.EncoderMLP.FUSED = True     # Set but NOT USED (model.py uses plain fc1/fc2)
# LinearGELU.FUSED = False         # Set but NOT USED (model.py uses plain linear_1/act)
```

> **What does this mean?**
>
> - **`Linear.BACKEND = "torch"`** routes all matrix multiplications through PyTorch's
>   default path, which calls cuBLAS under the hood. cuBLAS is NVIDIA's hand-optimized
>   library with years of tuning for every GPU architecture. It is almost always faster
>   than a hand-written Triton kernel for standard matrix multiply, because NVIDIA
>   engineers have access to undocumented hardware features and spend enormous effort
>   optimizing tile sizes, data layouts, and instruction scheduling.
>
> - **`Linear.BACKEND = "triton"`** forces matrix multiplications through the Triton
>   kernel (`linear_kernel_tf32`). This is useful for learning and debugging but is
>   typically slower than cuBLAS.
>
> - **`MLP.FUSED = True`** enables fused SwiGLU, which combines the `silu` activation
>   and the elementwise gate multiplication into a single kernel. Without fusion, the
>   intermediate result (after silu) must be written back to global memory (HBM) and
>   then read again for the multiplication — a wasteful round-trip. Fusion keeps the
>   intermediate value in registers, saving memory bandwidth.
>
> - **`EncoderMLP.FUSED` and `LinearGELU.FUSED`** are set but have NO EFFECT because
>   `model.py` does not use these classes. It uses plain `Linear` + `gelu()` calls
>   instead. This is a common trap — the fused classes exist in `layers.py` but the
>   model bypasses them.

### fp16 Weights (flag name retained as BF16 for compatibility)
```python
Linear.BF16 = True                     # Class default in layers.py, enables half-precision
Linear._HALF_DTYPE = torch.float16     # Actual dtype: fp16 (faster HGEMM on RTX 5090)
```
Output stays fp16 (no `.float()` conversion), keeping the entire pipeline in fp16.

> **What does this mean?**
>
> - Setting `Linear.BF16 = True` tells the Linear layer to convert its weight matrices
>   to half-precision (16-bit) before computing. This **halves memory bandwidth** — the
>   GPU needs to move half as many bytes from global memory to the compute units.
>   Since matrix multiply on modern GPUs is often memory-bandwidth-bound (the tensor
>   cores can compute faster than data arrives), halving the data transfer roughly
>   doubles the effective throughput.
>
> - **Why fp16 instead of bf16?** Both are 16-bit formats. fp16 has more mantissa
>   precision (10 bits vs 7 bits) but less dynamic range than bf16. On the RTX 5090,
>   fp16 HGEMM is slightly faster (~0.4ms) than bf16 HGEMM because the hardware's
>   fp16 tensor core path is better optimized. The flag is named "BF16" for historical
>   reasons but actually selects fp16 via `_HALF_DTYPE = torch.float16`.
>
> - **No `.float()` conversion** is crucial. Previously, some layers would convert back
>   to fp32 after computation "for numerical stability." Removing these conversions
>   saves ~7.5ms because the GPU does not need to perform dtype conversion (which
>   requires reading and writing the entire tensor) between every operation.

### GPU Detection: GPUProfile (layers.py)
```python
# GPUProfile detects GPU architecture at import time
GPU = GPUProfile()  # Replaces old _detect_gpu_tier()

# Reads: sm_version, gpu_name, shared memory via getattr fallback chain:
#   shared_memory_per_block_optin -> max_shared_memory_per_block -> shared_memory_per_block
# Classifies: blackwell_consumer, ada, hopper, blackwell_dc, ampere_dc, ampere_consumer, older

# _KNOWN_CONFIGS table stores tested tile sizes for 6 GPU architectures
# For unknown GPUs: _compute_attention_tiles() and _compute_matmul_tiles()
# compute tiles dynamically from shared memory budget
```

> **What does this mean?**
>
> **GPUProfile** is a runtime detection system that identifies which GPU the code is
> running on and selects optimal kernel configurations automatically. This is important
> because different GPUs have vastly different capabilities:
>
> - **Shared memory size** varies from ~48KB (older GPUs) to ~228KB (datacenter GPUs
>   like H200). Larger shared memory allows bigger tiles, which means more data reuse
>   and higher performance. But if you configure tiles too large for the available
>   shared memory, the kernel will crash with an out-of-memory error.
>
> - **SM version** (e.g., sm_89 for Ada, sm_90 for Hopper, sm_100 for Blackwell)
>   identifies the GPU architecture. Each generation has different tensor core
>   capabilities, different shared memory sizes, and different optimal configurations.
>
> - **The getattr fallback chain** handles the fact that different CUDA/Triton versions
>   expose shared memory information under different attribute names. The code tries
>   three possible names to be robust across environments.
>
> - **`_KNOWN_CONFIGS`** is a lookup table of tile sizes that have been hand-tested on
>   specific GPUs. For known GPUs, these pre-tested values are used directly, avoiding
>   the need for auto-tuning. For unknown GPUs, the code computes tile sizes
>   dynamically based on the available shared memory budget.

### Flash Attention Configuration (GPUProfile-Aware)
```python
# Tile selection via GPU.get_attention_tiles(head_dim, seq_q):
# Consumer GPUs (RTX 4090/5090, ~100KB optin shared mem):
#   head_dim=64:  BLOCK_M=64,  BLOCK_N=64,  num_stages=1, num_warps=4
#   head_dim=128: BLOCK_M=32,  BLOCK_N=32,  num_stages=1, num_warps=4
#   seq_q <= 16:  BLOCK_M clamped to 16

# Datacenter GPUs (H200/B200, ~228KB optin shared mem):
#   head_dim=64:  BLOCK_M=128, BLOCK_N=128, num_stages=2, num_warps=8
#   head_dim=128: BLOCK_M=128, BLOCK_N=64,  num_stages=2, num_warps=8

# SDPA fallback for KV-cached decode (seq_q <= 4):
# torch.nn.functional.scaled_dot_product_attention -- avoids Triton launch overhead

```

> **What does this mean?**
>
> This section shows how tile sizes and kernel parameters are tuned for different GPUs.
> Every parameter has a specific reason:
>
> - **BLOCK_M and BLOCK_N** (tile sizes) — These determine how much Q and K/V data each
>   thread block loads into shared memory. Larger tiles mean more data reuse (each loaded
>   value is used in more computations), but require more shared memory. Consumer GPUs
>   have ~100KB of shared memory, so tiles must be smaller (32x32 or 64x64). Datacenter
>   GPUs have ~228KB, allowing larger tiles (128x128 or 128x64) for better efficiency.
>
> - **head_dim=64 allows bigger tiles than head_dim=128** because each tile element has
>   64 values instead of 128, so the same amount of shared memory can hold more tile
>   rows/columns.
>
> - **num_warps** — The number of warps (groups of 32 threads) per thread block.
>   Consumer GPUs use 4 warps (128 threads per block); datacenter GPUs use 8 warps
>   (256 threads per block). More warps can hide memory latency better (while some
>   threads wait for data, others compute), but too many warps can cause register
>   pressure (not enough registers for each thread).
>
> - **num_stages** — The number of pipeline stages for software prefetching. With
>   `num_stages=1`, the kernel loads a tile and then computes on it sequentially. With
>   `num_stages=2`, the kernel loads the NEXT tile while computing on the CURRENT tile
>   (double-buffering). This hides memory latency but requires 2x the shared memory for
>   tiles. Consumer GPUs cannot afford this (OOM), so they use `num_stages=1`. Datacenter
>   GPUs have enough shared memory for `num_stages=2`.
>
> - **`seq_q <= 16: BLOCK_M clamped to 16`** — When the query sequence is very short
>   (e.g., during KV-cached decode), using a large BLOCK_M would waste threads (most
>   would have no work). Clamping BLOCK_M to 16 avoids launching mostly-idle blocks.
>
> - **SDPA fallback for seq_q <= 4** — For extremely short queries (1-4 tokens, which
>   is the common case during token-by-token generation), the overhead of launching a
>   Triton kernel (compilation, dispatch, synchronization) exceeds the actual compute
>   time. PyTorch's built-in SDPA is faster in this regime because it is pre-compiled
>   and has lower launch overhead. This saves ~3ms.
>
> - **Kernel launch overhead** refers to the fixed cost of starting a GPU kernel: the
>   CPU must set up the grid dimensions, copy parameters, and signal the GPU to begin.
>   This takes microseconds, but when the kernel itself only runs for microseconds
>   (as with seq_q=1 attention), the overhead dominates.

---

## Benchmark Commands

```bash
cd hw1-asr
export HF_HOME=/workspace/.hf_cache

# Student benchmark
python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5

# Baseline comparison
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3

# Detailed per-operator profiling
python benchmark_detailed.py glm_asr_triton_template
```

> **What does this mean?**
>
> - **`HF_HOME`** sets the directory where HuggingFace caches downloaded model weights.
>   Without this, the weights default to `~/.cache/huggingface/`.
>
> - **`--warmup 2`** runs the model 2 times before timing. This is necessary because:
>   (1) The first run triggers Triton kernel compilation (JIT), which takes seconds.
>   (2) CUDA needs to "warm up" — the first kernel launch initializes GPU contexts.
>   (3) Caches (CPU/GPU L2) need to be populated for realistic measurements.
>   Without warmup, the first run would be dramatically slower and skew the average.
>
> - **`--runs 5`** averages the result over 5 timed runs for more stable measurements.
>   GPU timing can vary by a few percent between runs due to clock frequency changes,
>   thermal throttling, and background processes.
>
> - **`glm_asr_triton_template`** is your implementation (with optimized kernels).
>   **`glm_asr_triton_example`** is the unoptimized baseline for comparison.

---

## Benchmark Results

### RTX 5090 (CUDA 13.0, 2026-03-15)
| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template (fp16 pipeline + KV cache + SDPA)** | **98.5ms** | 7.58ms/tok | 100% |
| Our template (bf16 pipeline + KV cache + SDPA) | 110.0ms | 8.46ms/tok | 100% |
| Our template (no KV cache) | 120.7ms | 9.29ms/tok | 100% |
| Example baseline | 261.3ms | 20.10ms/tok | 100% |
| **Speedup** | **62.3%** | | |

> **What does this mean?**
>
> - **Time** is the total end-to-end latency to transcribe the test audio (~3.5 seconds
>   of speech), including audio encoding, projecting, and generating all output tokens.
>   Lower is better.
>
> - **Speed (ms/tok)** is the average time per generated token. The test output is
>   "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS" — 13 tokens. So 98.5ms / 13 =
>   ~7.58ms per token. This metric matters for interactive applications where users
>   are waiting for each word to appear.
>
> - **Accuracy (100%)** means every word in the transcription matches the expected output
>   exactly. Any kernel optimization that changes the output (even slightly) would show
>   up as reduced accuracy.
>
> - **fp16 vs bf16 pipeline:** fp16 is 11.5ms faster because the RTX 5090's tensor
>   cores have a faster path for fp16 HGEMM. Both maintain 100% accuracy.
>
> - **KV cache vs no KV cache:** Adding KV cache saves ~22ms (120.7 -> 98.5) because
>   the decoder no longer recomputes past tokens. The savings grow with longer outputs.
>
> - **62.3% speedup** over the baseline (from 261.3ms to 98.5ms) comes from the
>   combination of: fp16 pipeline (~11.5ms), KV cache (~22ms), SDPA fallback (~3ms),
>   fused SwiGLU, fused RoPE (~14ms), Flash Attention, and removing unnecessary
>   dtype conversions.

### H200 MIG 3g.71gb (Teaching Cluster, 60 SMs, 2026-03-16)
| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template (fp16 pipeline + KV cache + SDPA)** | **204.6ms** | 15.74ms/tok | 100% |

> **What does this mean?**
>
> - The H200 MIG (Multi-Instance GPU) partition has only 60 SMs compared to the RTX
>   5090's ~170 SMs. This is about 1/3 the compute capacity, which is why it takes
>   roughly 2x longer (204.6ms vs 98.5ms — not 3x because memory bandwidth is still
>   high and not all operations are compute-bound).
>
> - **MIG** partitions a single physical GPU into multiple isolated instances. The
>   "3g.71gb" partition provides 3/7 of the GPU's SMs and 71GB of memory. This is
>   common in shared teaching/research clusters.

### Detailed Benchmark (50 generated tokens)
| Component | Time | % Total |
|-----------|------|---------|
| Audio Encoder | 202.09ms | 8.7% |
| Projector | 4.14ms | 0.2% |
| Decoder Prefill | 191.59ms | 8.3% |
| **Decoder Decode (50 steps)** | **1919.94ms** | **82.8%** |
| **Total** | **2317.76ms** | 100% |

**Key bottleneck:** Decoder decode dominates because stock `generate()` is O(n^2).

> **What does this mean?**
>
> - **Audio Encoder (8.7%)** processes the mel spectrogram through 32 transformer layers.
>   This is a fixed cost that does not grow with output length.
>
> - **Projector (0.2%)** is trivially cheap — just two linear layers.
>
> - **Decoder Prefill (8.3%)** is the decoder's first pass over the full input (audio
>   tokens + prompt). This processes all input tokens in one forward pass. It runs once,
>   regardless of how many tokens are generated.
>
> - **Decoder Decode (82.8%)** is the autoregressive generation loop. Each of the 50
>   steps runs the full decoder on an increasingly long sequence. With KV cache, each
>   step processes only 1 new token (fast). WITHOUT KV cache (stock generate()), each
>   step reprocesses the ENTIRE growing sequence, making it O(n^2). This is why decode
>   dominates: 50 steps x ~38ms/step = ~1920ms.
>
> - This breakdown shows that optimizing the encoder or projector has minimal impact.
>   The real gains come from optimizing the decode loop — KV cache, SDPA fallback, and
>   faster per-token decoding.

---

## Optimization Roadmap

This table tracks every optimization that was tested, whether it was adopted or rejected,
and the measured impact.

| Optimization | Source | Impact | Status |
|-------------|--------|--------|--------|
| Fused Q+K RoPE kernel | meave | **-14ms** | **ADOPTED** |
| bf16 RMSNorm output | meave (adapted) | **-3ms** | **ADOPTED** |
| bf16 LayerNorm output | internal | **-0.7ms** | **ADOPTED** |
| generate_v8b (KV cache) | internal | **-7.6ms** | **ADOPTED** |
| SDPA fallback for seq_q<=4 | internal | **-3ms** | **ADOPTED** |
| GPUProfile + _KNOWN_CONFIGS + dynamic tiles | internal | portability | **ADOPTED** |
| Dead code cleanup | internal | -320 lines | **ADOPTED** |
| fp16 pipeline (remove float32 casts) | internal | **-11.5ms** | **ADOPTED** |
| fp16 cuBLAS HGEMM (was bf16) | internal | ~-0.4ms | **ADOPTED** |
| Smaller flash attention tiles | meave | improved prefill | **ADOPTED** |
| Swizzled SwiGLU | yash/optimize | +18ms regression | Rejected |
| @triton.autotune (lightweight) | majed | +0.7ms overhead | Rejected |
| @triton.autotune (heavy kernels) | internal | massive regression | Rejected |
| Softmax bf16 output | internal | 0ms | Rejected |
| Flash Attention num_stages=2 | yash/optimize | OOM on consumer GPUs | Rejected |
| PyTorch SDPA for prefill/encoder | internal | +6ms regression | Rejected |
| SDPA enable_gqa=True for decode | internal | +13ms regression | Rejected |
| Fused gate+up Linear in MLP | internal | Neutral | Rejected |

> **What does this mean? — Explaining each optimization and its outcome:**
>
> **Adopted optimizations:**
>
> - **Fused Q+K RoPE kernel (-14ms):** Instead of running separate kernels for
>   computing RoPE on Q and then on K, a single fused kernel processes both at once.
>   This halves the kernel launch overhead and improves memory access patterns (Q and K
>   are adjacent in memory). 14ms is a huge win — the single biggest optimization.
>
> - **bf16/fp16 norm output (-3ms, -0.7ms):** The RMSNorm and LayerNorm kernels were
>   modified to output in half-precision (fp16) directly, instead of outputting fp32 and
>   converting later. This eliminates a full read-write cycle of the entire tensor.
>
> - **generate_v8b / KV cache (-7.6ms):** Implements key-value caching in the decoder.
>   After the first pass, cached K/V tensors are reused so only the new token needs
>   full Q/K/V computation. Reduces decode from O(n^2) to O(n).
>
> - **SDPA fallback for seq_q<=4 (-3ms):** During KV-cached decode, each step only has
>   seq_q=1 (one new token). Launching a Triton kernel for such a tiny computation is
>   wasteful. PyTorch's built-in SDPA is pre-compiled and has lower overhead.
>
> - **GPUProfile (portability):** Not a speed improvement, but ensures the code runs
>   correctly across different GPU types by auto-detecting hardware capabilities.
>
> - **fp16 pipeline (-11.5ms):** Removed unnecessary `.float()` (fp32) conversions
>   between layers. Each conversion requires reading and writing the entire tensor.
>   Keeping everything in fp16 halves memory traffic throughout the pipeline.
>
> - **fp16 cuBLAS HGEMM (~-0.4ms):** Switched from bf16 to fp16 for matrix multiply,
>   which is slightly faster on the RTX 5090's tensor core path.
>
> - **Smaller flash attention tiles (improved prefill):** Using 64x64 tiles for the
>   encoder (head_dim=64) and 32x32 for the decoder (head_dim=128) instead of larger
>   tiles. Smaller tiles fit in shared memory more comfortably and can improve occupancy
>   (the number of blocks running simultaneously per SM).
>
> **Rejected optimizations:**
>
> - **Swizzled SwiGLU (+18ms regression):** "Swizzling" rearranges memory access patterns
>   to avoid bank conflicts in shared memory. In this case, the rearrangement overhead
>   outweighed the benefits, causing a net slowdown.
>
> - **@triton.autotune (+0.7ms to massive regression):** Triton's autotune decorator
>   tries multiple kernel configurations at runtime and picks the best one. However,
>   the auto-tuning overhead (extra kernel launches, timing, selection) is significant,
>   especially for kernels that run many times. The hand-tuned configurations in
>   `_KNOWN_CONFIGS` are already near-optimal, making autotune counterproductive.
>
> - **Softmax bf16 output (0ms):** Converting softmax output to bf16 had no effect
>   because the softmax values are immediately consumed by the next operation and the
>   bandwidth savings were negligible.
>
> - **Flash Attention num_stages=2 (OOM on consumer GPUs):** Double-buffering tiles
>   requires 2x shared memory. Consumer GPUs (RTX 4090/5090) with ~100KB shared memory
>   cannot fit two stages of attention tiles, causing an out-of-memory crash.
>
> - **PyTorch SDPA for prefill/encoder (+6ms regression):** Using PyTorch's built-in
>   SDPA for the encoder (where seq_q is large) was slower than the custom Flash
>   Attention kernel because SDPA has more general-purpose overhead.
>
> - **SDPA enable_gqa=True (+13ms regression):** PyTorch's native GQA support was
>   slower than the manual `_expand_kv_heads` approach used in the Flash Attention kernel.
>
> - **Fused gate+up Linear (neutral):** Fusing the two weight matrices of SwiGLU into
>   one larger matrix multiply showed no improvement because cuBLAS already handles
>   both matmuls efficiently, and the fused version has worse memory access patterns.

---

## Optimization Checklist

- [x] All 10 kernels implemented and passing tests
- [x] Correct transcription output (100% word accuracy)
- [x] Fused SwiGLU active for decoder MLP (`MLP.FUSED = True`)
- [x] Linear backend optimized (cuBLAS selected as fastest)
- [x] TF32 runtime flags enabled
- [x] bfloat16 weights -- halves memory traffic
- [x] Fused Flash Attention -- Triton kernel with online softmax
- [x] 17 deterministic numerical parity tests for Flash Attention
- [x] model.py, conv.py, weight_loader.py all match origin/main (zero diff)
- [x] Upstream merge with ed-aisys (19 commits, grading criteria, benchmark updates)
- [x] Fused Q+K RoPE pair kernel (from meave) -- **-14ms**
- [x] bf16 RMSNorm output kernel (from meave) -- **-3ms**
- [x] bf16 LayerNorm output -- **-0.7ms**
- [x] generate_v8b with KV cache (monkey-patched, decode(use_cache=True)) -- **-7.6ms**
- [x] SDPA fallback for KV-cached decode (seq_q<=4) -- **-3ms**
- [x] GPUProfile with _KNOWN_CONFIGS + dynamic tile computation for cross-GPU portability
- [x] Dead code cleanup -- removed ~320 lines of legacy attention kernels
- [x] SwiGLU swizzle tested, rejected (+18ms regression on RTX 5090)
- [x] @triton.autotune tested, rejected (lightweight: +0.7ms overhead; heavy kernels: massive regression)
- [x] Softmax bf16, num_stages=2, num_warps=8 -- tested, no improvement on consumer GPUs
- [x] fp16 cuBLAS HGEMM (`Linear._HALF_DTYPE = torch.float16`) -- slightly faster than bf16
- [x] Smaller flash attention tiles (from meave) -- 64x64 encoder, 32x32 decoder
- [x] Remove Linear `.float()` conversion -- fp16 output cascades through pipeline (**-7.5ms**)
- [x] Remove silu/gelu Python-side float32 cast -- kernels handle internally (**-3.7ms**)
- [x] Remove RMSNorm/LayerNorm Python-side float32 cast -- kernels handle internally (~-0.5ms)
- [x] fp16 embedding output -- keeps decoder pipeline in fp16 from start
- [x] fp16 fused SwiGLU/EncoderMLP -- halves intermediate memory bandwidth
- [x] Remove flash attention Python-side float32 conversion (~-1ms)
- [x] Norm kernel output dtype: fp16 (was bf16)
- [x] PyTorch SDPA for prefill/encoder -- tested, +6ms regression. Rejected
- [x] SDPA enable_gqa=True -- tested, +13ms regression. Rejected
- [x] Fused gate+up Linear in MLP -- tested, neutral. Rejected

> **What does this mean?**
>
> This checklist is the complete record of every optimization that was tried. The checked
> items include both adopted and rejected optimizations — all were tested. The key insight
> from this checklist is that not every "obvious" optimization helps. GPU programming is
> full of surprises:
>
> - Auto-tuning sounds great in theory but has real overhead.
> - Fusing operations sometimes hurts if the fused kernel has worse occupancy.
> - The biggest wins often come from eliminating unnecessary data movement (fp16 pipeline,
>   removing dtype conversions) rather than making individual kernels faster.
> - The single biggest optimization was fusing the RoPE kernel (-14ms), followed by the
>   fp16 pipeline (-11.5ms) and KV cache (-7.6ms).

---

## File Dependency Graph

```
model.py (DO NOT MODIFY -- stock generate(), no KV cache)
  |-- layers.py (RMSNorm, LayerNorm, Linear, MLP, EncoderMLP*, LinearGELU*, Embedding, softmax, gelu, silu)
  |-- attention.py (MultiHeadAttention, scaled_dot_product_attention)
  |-- rope.py (RotaryEmbedding, apply_rotary_pos_emb)
  |-- conv.py (Conv1d, Conv1dSubsampler) (DO NOT MODIFY)
  |-- weight_loader.py (load_model_from_hf) (DO NOT MODIFY)

* EncoderMLP and LinearGELU classes exist in layers.py but model.py does NOT use them.
  model.py uses plain Linear + gelu() for encoder MLP and projector.

benchmark_student.py
  |-- model.py (via dynamic import)
  |-- weight_loader.py (downloads from HuggingFace)
```

> **What does this mean?**
>
> This shows which file imports which. The key takeaways:
>
> - `model.py` is the central file that wires everything together. It imports from all
>   other module files. Since you cannot modify it, your optimizations must be
>   "drop-in replacements" — the functions and classes you modify must keep the same
>   API (same function names, same parameter names, same return types).
>
> - The asterisk (*) on `EncoderMLP` and `LinearGELU` is a critical trap. These classes
>   exist and can be configured, but the model does not actually use them. If you spend
>   time optimizing these classes, the effort will have zero impact on benchmarks.
>
> - `benchmark_student.py` dynamically imports the model (using Python's `importlib`),
>   so changing the module name on the command line switches between your implementation
>   and the baseline.

---

## HuggingFace Model

- **Model ID:** `zai-org/GLM-ASR-Nano-2512`
- **Size:** ~4.3GB (safetensors format)
- **Cache:** `$HF_HOME` or `~/.cache/huggingface/`

> **What does this mean?**
>
> The model weights are downloaded from HuggingFace's model hub. "Safetensors" is a
> safe, fast file format for storing tensors (as opposed to Python pickle, which can
> execute arbitrary code). The 4.3GB size is the total weight data — this is what
> gets loaded into GPU memory at startup.

## Test Audio

- **File:** `hw1-asr/test_audio.wav`
- **Expected output:** `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`
- **Duration:** ~3.5 seconds

> **What does this mean?**
>
> This is the reference test case. Your implementation must produce exactly this
> transcription (all 8 words, in order) to get 100% accuracy. Any kernel bug — even a
> tiny numerical error — could change the output tokens and fail the accuracy check.

## Troubleshooting

### cuBLAS Version Mismatch
If you see `CUBLAS_STATUS_INVALID_VALUE`, pip-installed `nvidia-cublas` may conflict:
```bash
pip uninstall nvidia-cublas
```

> **What does this mean?**
>
> PyTorch ships with its own cuBLAS library. If you also have a separate `nvidia-cublas`
> package installed via pip, the two versions can conflict (different function signatures
> or memory layouts), causing cryptic errors during matrix multiply. The fix is to remove
> the standalone package so PyTorch uses its bundled version.

### numpy Version Mismatch (cu12)
If you see `TypeError: expected np.ndarray (got ndarray)`, use `torch.as_tensor()` instead
of `torch.from_numpy()`. The `_to_torch_tensor()` helper in layers.py handles this automatically.

> **What does this mean?**
>
> Some CUDA 12 environments have numpy version incompatibilities where the `ndarray` type
> is not recognized correctly. `torch.as_tensor()` is more permissive and handles edge
> cases that `torch.from_numpy()` does not. The `_to_torch_tensor()` wrapper in layers.py
> abstracts this away.

### Teaching Cluster OOM
If SLURM kills your job during weight loading, request more RAM:
```bash
srun -p Teaching -w saxa --gres gpu:3g.71gb:1 --mem=32G --pty bash
```

> **What does this mean?**
>
> SLURM is the job scheduler on the teaching cluster. Loading 4.3GB of model weights
> into GPU memory also requires significant CPU (host) memory for the initial file
> read and tensor construction. The default memory allocation may not be enough.
> `--mem=32G` requests 32GB of host RAM, which provides enough headroom for weight
> loading plus PyTorch overhead plus Triton's JIT compilation cache.
>
> The flags mean:
> - `-p Teaching` — use the "Teaching" partition (queue)
> - `-w saxa` — target the specific node named "saxa"
> - `--gres gpu:3g.71gb:1` — request 1 MIG GPU partition (3g.71gb = 3/7 of the GPU)
> - `--mem=32G` — request 32GB of host RAM
> - `--pty bash` — open an interactive bash shell
