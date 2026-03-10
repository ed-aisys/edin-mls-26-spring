# GLM-ASR Codebase: Complete Code Explanation

A detailed explanation of every component in the GLM-ASR Triton implementation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [layers.py — GPU Compute Kernels](#2-layerspy--gpu-compute-kernels)
3. [attention.py — Attention Mechanism](#3-attentionpy--attention-mechanism)
4. [rope.py — Positional Encodings](#4-ropepy--positional-encodings)
5. [model.py — Full Model Pipeline](#5-modelpy--full-model-pipeline)
6. [conv.py — Audio Feature Extraction](#6-convpy--audio-feature-extraction)
7. [weight_loader.py — Model Weights](#7-weight_loaderpy--model-weights)
8. [benchmark_student.py — Testing](#8-benchmark_studentpy--testing)
9. [How It All Fits Together](#9-how-it-all-fits-together)

---

## 1. Architecture Overview

GLM-ASR is a **multi-modal speech-to-text model** that converts audio waveforms
into text transcriptions. It follows the encoder-decoder transformer architecture:

```
[Audio Waveform] -> [Mel Spectrogram] -> [Audio Encoder] -> [Projector] -> [Text Decoder] -> [Text]
```

### Why Triton?

PyTorch operations (like `torch.matmul`) call into cuBLAS/cuDNN libraries.
These are general-purpose and highly optimized, but:

1. **Kernel fusion:** Each PyTorch op launches a separate CUDA kernel. Between
   kernels, data must be written to and read from GPU memory (DRAM). Triton
   lets you fuse multiple operations into one kernel, keeping data in fast
   on-chip memory (SRAM/registers).

2. **Custom operations:** Some operations (like RoPE) don't have optimized
   library implementations. Triton lets you write custom GPU code in Python.

3. **Architecture-specific tuning:** Triton generates PTX/SASS code optimized
   for your specific GPU architecture.

### File Modification Rules

Per GUIDE.md, these files are **read-only** (shared infrastructure):
- `model.py` — model architecture and generation loop (includes `generate_v8b` with KV cache)
- `weight_loader.py` — loads pre-trained weights from HuggingFace
- `conv.py` — 1D convolution for audio subsampling

You can only modify: `layers.py`, `attention.py`, `rope.py`, `__init__.py`.

---

## 2. layers.py — GPU Compute Kernels

This is the core file containing all neural network building blocks as Triton
kernels and Python layer classes.

### 2.1 Helper Functions

```python
def next_power_of_two(x):
    """1 << (x-1).bit_length() -- e.g., 5->8, 8->8, 100->128"""
```
Triton requires block sizes to be powers of 2. This rounds up.

```python
def pad_to_multiple(size, multiple):
    """Rounds up size to nearest multiple. E.g., pad_to_multiple(100, 64) = 128"""
```
Matrix dimensions must be multiples of tile sizes for efficient tiling.

### 2.2 RMSNorm Kernel

**Purpose:** Normalizes hidden states in the text decoder (before attention and MLP).

**Math:** `y = x / sqrt(mean(x^2) + eps) * weight`

Unlike LayerNorm, RMSNorm doesn't subtract the mean — it only divides by the
root mean square. This is ~10% faster because it skips the mean computation.

**GPU Strategy:**
- One thread block processes one row (one token's hidden state)
- All elements in the row are loaded in parallel
- `tl.sum(x * x)` computes sum of squares using parallel reduction
- `tl.rsqrt()` computes 1/sqrt() in a single hardware instruction

**Where used:** `DecoderLayer.input_layernorm`, `DecoderLayer.post_attention_layernorm`, `TextDecoder.norm`

### 2.3 LayerNorm Kernel

**Purpose:** Normalizes hidden states in the audio encoder.

**Math:** `y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`

**Difference from RMSNorm:** Two-pass normalization:
1. Compute mean, subtract it (centering)
2. Compute variance of centered data, normalize
3. Apply both weight AND bias (RMSNorm only has weight)

**Where used:** `AudioEncoderLayer.self_attn_layer_norm`, `AudioEncoderLayer.final_layer_norm`, `AudioEncoder.layer_norm`

### 2.4 GELU Kernel

**Purpose:** Non-linear activation function for audio encoder MLP and projector.

**Math (tanh approximation):**
```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Uses `tl.extra.cuda.libdevice.tanh` for the tanh computation — this calls into
NVIDIA's libdevice math library for hardware-optimized transcendental functions.

**Where used:** Audio encoder MLP (via `EncoderMLP`), Projector (via `LinearGELU`)

### 2.5 SiLU Kernel

**Purpose:** Non-linear activation function for text decoder MLP (SwiGLU gating).

**Math:** `y = x * sigmoid(x) = x / (1 + exp(-x))`

SiLU (also called Swish) is used in Llama-style models. It's smooth and
has better gradient flow than ReLU.

**Where used:** Text decoder SwiGLU MLP (`gate_proj -> SiLU`, combined with `up_proj`)

### 2.6 Linear (Matmul) Kernel

**Purpose:** Matrix multiplication for all projection layers (Q, K, V, output, MLP).

**Math:** `C = A @ B` where A is (M, K), B is (K, N), C is (M, N)

**GPU Strategy — Tiled Matrix Multiplication:**

The key insight is that matrix multiplication has O(N^3) compute but only O(N^2)
data. By loading small tiles into fast on-chip memory and reusing them, we
minimize slow DRAM accesses.

```
For each output tile (BLOCK_M x BLOCK_N):
    acc = zeros(BLOCK_M, BLOCK_N)
    for k in range(0, K, BLOCK_K):
        Load A_tile (BLOCK_M x BLOCK_K) from DRAM to SRAM
        Load B_tile (BLOCK_K x BLOCK_N) from DRAM to SRAM
        acc += A_tile @ B_tile  # Done in SRAM using tensor cores
    Store acc to C in DRAM
```

**`tl.dot(a, b)`** compiles to tensor core instructions (HMMA/WMMA) on
supported GPUs, giving ~10x speedup over regular FP32 multiply-add.

**Padding:** The `Linear` class pads input dimensions to multiples of TILE_M/N/K
so the kernel doesn't need bounds checking in the hot loop.

**Where used:** Every Linear layer in the model (Q, K, V, O projections, MLP layers, LM head)

### 2.7 Softmax Kernel

**Purpose:** Converts raw logits to probability distributions.

**Math:** `y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

**Numerical Stability:** Without subtracting max, `exp(1000)` = infinity.
By subtracting the maximum value first, the largest exponent is `exp(0) = 1`.

**Where used:** Final token prediction (standalone softmax)

### 2.8 Fused Kernels (Pre-implemented in the template)

**linear_gelu_kernel:** Computes `GELU(x @ W + b)` in a single kernel launch.
Instead of: matmul -> write to DRAM -> read from DRAM -> GELU,
it does: matmul -> GELU (all in registers). Eliminates one DRAM round-trip.

**swiglu_fused_kernel:** Computes `SiLU(x @ W_gate) * (x @ W_up)` in one kernel.
Fuses THREE operations: two matmuls and the gating. Input `x` is loaded once.

### 2.9 Layer Classes

**`RMSNorm` class:** Wraps the kernel with device management and fallback.
- Checks if hidden_size fits in BLOCK_SIZE (must be power of 2)
- Falls back to PyTorch if not on CUDA
- Manages weight tensor device placement

**`Linear` class:** Switchable between torch (cuBLAS) and Triton backends.
- `BACKEND = "torch"`: Uses `torch.nn.functional.linear(...)`, which routes the
  matmul path through PyTorch's cuBLAS/cuBLASLt dispatch
- `BACKEND = "triton"`: Uses `linear_kernel_tf32`
- `BF16 = True` (class-level default): Caches bfloat16 copies of weights
  (`_weight_bf16`, `_bias_bf16`). All matmuls via `F.linear` run in bf16,
  halving memory traffic for memory-bound decode matmuls. Results are cast
  back to float32. This is set as a class default because `__init__.py` is
  not executed during benchmark imports.
- `__init__.py` also enables `torch.set_float32_matmul_precision("high")`,
  TF32 matmul, TF32 cuDNN, and `cudnn.benchmark`
- The committed configuration keeps the cuBLAS path because it wins end-to-end
  on this RTX 5090 stack
- Caches transposed/padded weights (`_weight_t_padded`) for Triton path

**`MLP` class:** Implements SwiGLU gating for the text decoder:
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
When `FUSED=True`, uses `swiglu_fused_kernel` for the gate+up computation.

**`EncoderMLP` class:** Simpler MLP without gating for the audio encoder:
```
output = fc2(GELU(fc1(x)))
```
When `FUSED=True`, uses `linear_gelu_kernel` for fc1+GELU.
`model.py` uses `EncoderMLP` for encoder layers (`self.mlp = EncoderMLP(...)`),
and aliases `self.fc1 = self.mlp.fc1` / `self.fc2 = self.mlp.fc2` for weight loading.

**`LinearGELU` class:** A `GELU(Linear(x))` wrapper used by the projector.
`model.py` creates `self.linear_1_gelu = LinearGELU(5120, 4096)` for the
first projector layer. `FUSED` is set to `False` in layers.py because the
large dimensions (5120x4096) with tile sizes 128x128x64 require 131KB shared
memory, exceeding the RTX 5090's hardware limit of 101KB. The unfused path
(cuBLAS matmul + separate GELU kernel) is used instead.

---

## 3. attention.py — Attention Mechanism

### 3.1 Scaled Dot-Product Attention

The core of every transformer. Computes:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 3.2 Primary Path: Fused Flash Attention Kernel (Triton)

The committed runtime path uses a fused Triton Flash Attention kernel with
**online softmax** as the primary attention implementation. This replaces both
SDPA and the old 3-kernel approach.

```python
# attention.py — scaled_dot_product_attention()
if q.is_cuda:
    # Expand GQA heads before kernel call
    if use_gqa:
        k = _expand_kv_heads(k, num_heads)
        v = _expand_kv_heads(v, num_heads)
    # Single fused kernel launch
    flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, output, mask_flat, scale, ...,
        IS_CAUSAL=is_causal, HAS_MASK=has_mask,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_stages=1,
    )
```

**How it works (online softmax algorithm):**

The kernel processes Q in tiles of `BLOCK_M` rows. For each Q tile, it iterates
over K/V in blocks of `BLOCK_N`, maintaining a running softmax:

```
m_i = -inf          # running max (per query row)
l_i = 0             # running sum of exp (per query row)
acc = 0             # output accumulator [BLOCK_M, BLOCK_D]

for each K/V block:
    S = Q_tile @ K_block^T          # tl.dot — tensor cores
    # Apply causal mask (if IS_CAUSAL)
    # Apply attention mask bias (if HAS_MASK)
    m_new = max(m_i, max(S))        # updated running max
    alpha = exp(m_i - m_new)        # rescale factor for old state
    p = exp(S - m_new)              # new attention weights
    l_i = alpha * l_i + sum(p)      # updated running sum
    acc = alpha * acc + p @ V_block # rescale old + accumulate new (tl.dot)
    m_i = m_new

output = acc / l_i                  # final normalization
```

**Why this is better than the 3-kernel approach:**
1. **No DRAM scores matrix** — the full `(batch*heads, seq_q, seq_k)` scores
   matrix is never materialized. Everything stays in SRAM/registers.
2. **Single kernel launch** — no synchronization overhead between kernels.
3. **O(BLOCK) SRAM** — memory-efficient for long sequences.
4. **Tensor cores** — `tl.dot` for both Q@K^T and P@V.

**Tile sizes** (chosen to fit 101KB shared memory):
- Encoder (head_dim=64): `BLOCK_M=128, BLOCK_N=64` — larger tiles for more parallelism
- Decoder (head_dim=128): `BLOCK_M=64, BLOCK_N=32` — smaller tiles due to larger SRAM per element
- `num_stages=1` — prevents Triton from double-buffering K/V loads

**Features:**
- `IS_CAUSAL` (constexpr): enables causal masking for decoder (`offs_m >= cur_offs_n`)
- `HAS_MASK` (constexpr): enables additive attention mask bias (zero overhead when False)
- Supports arbitrary sequence lengths (no `MAX_ATTENTION_DIM` limit)

### 3.3 Legacy Attention Kernels

The file also contains three legacy kernels from the original assignment:

1. **`attention_scores_kernel`**: Computes `Q @ K^T * scale` for one query position
   using broadcast-multiply and reduction.
2. **`softmax_inplace_kernel`**: In-place numerically stable softmax.
3. **`attention_output_kernel`**: Computes `attn_weights @ V` weighted sum.

These are superseded by `flash_attention_kernel` but remain in the codebase.

### 3.4 Grouped Query Attention (GQA)

The text decoder uses GQA: 16 query heads but only 4 KV heads.
Each KV head is shared by 4 query heads. This reduces KV-state size by 4x.

GQA is handled by explicitly expanding KV heads using `_expand_kv_heads()` before
the Flash Attention kernel call. The expansion is a zero-copy broadcast+reshape.

### 3.5 Causal Masking

In the Flash Attention kernel, causal masking is applied per-block:
```python
if IS_CAUSAL:
    s = tl.where(offs_m[:, None] >= cur_offs_n[None, :], s, -float("inf"))
```
The kernel also short-circuits the K/V iteration range for causal mode:
```python
kv_len = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M)  # skip future blocks entirely
```

### 3.6 Numerical Parity Tests

The file includes 8 parity tests (`__main__` block) that validate the Flash
Attention kernel against a pure PyTorch reference:
1. Basic attention (head_dim=64)
2. Causal attention (head_dim=64)
3. Masked attention (head_dim=64)
4. GQA (4Q/2KV heads, head_dim=64)
5. Basic attention (head_dim=128)
6. Causal attention (head_dim=128)
7. Causal + mask combined (head_dim=128)
8. Decode step (seq_q=1, seq_k=64, head_dim=128)

All tests pass with max diff < 0.01 (fp32 accumulation tolerance).

---

## 4. rope.py — Positional Encodings

### 4.1 What is RoPE?

Rotary Position Embeddings encode position information by rotating the
query and key vectors in 2D subspaces. For dimensions `(x1, x2)` at position `p`:

```
x1_rot = x1 * cos(p*freq) - x2 * sin(p*freq)
x2_rot = x2 * cos(p*freq) + x1 * sin(p*freq)
```

### 4.2 Frequency Computation

Each pair of dimensions has a different frequency:
```
freq_i = 1 / (base^(2i/d))    where base=10000 (encoder) or 500000 (decoder)
```
Low-frequency pairs capture long-range position differences,
high-frequency pairs capture fine-grained positions.

### 4.3 Partial RoPE

The audio encoder uses 50% partial RoPE — only the first half of each
head's dimensions are rotated. The rest pass through unchanged.

### 4.4 Kernel Implementation

The `compute_freqs_kernel` precomputes cos/sin for all positions:
```
For each position p:
    freqs = p * inv_freq[:]           # Element-wise multiply
    cos_cache[p, :half] = cos(freqs)  # First half
    cos_cache[p, half:] = cos(freqs)  # Second half (duplicated)
    sin_cache[p, :half] = sin(freqs)
    sin_cache[p, half:] = sin(freqs)
```

The duplication is because `apply_rotary_pos_emb` splits the input into
first half and second half, each multiplied by the same cos/sin.

---

## 5. model.py — Full Model Pipeline (READ-ONLY)

This file cannot be modified. Understanding it helps you implement correct kernels.

### 5.1 AudioEncoder

```python
class AudioEncoder:
    conv1: Conv1d(128 -> 1280, k=3, s=1)   # Feature expansion
    conv2: Conv1d(1280 -> 1280, k=3, s=2)  # 2x downsample
    rotary_emb: RotaryEmbedding(dim=64, partial=0.5)
    layers: [AudioEncoderLayer x 32]        # Transformer layers
    layer_norm: LayerNorm(1280)             # Final normalization
```

**Forward pass:**
1. Conv1 + GELU: `(1, 128, T)` -> `(1, 1280, T)`
2. Conv2 + GELU: `(1, 1280, T)` -> `(1, 1280, T/2)` (stride 2)
3. Permute: `(1, T/2, 1280)` (seq_len first for transformer)
4. 32x transformer layers with partial RoPE
5. Final LayerNorm

**AudioEncoderLayer MLP:** Uses `EncoderMLP` class from layers.py:
```python
self.mlp = EncoderMLP(hidden_size, intermediate_size, activation="gelu", bias=True)
self.fc1 = self.mlp.fc1  # Alias for weight loading
self.fc2 = self.mlp.fc2
```
When `EncoderMLP.FUSED = True`, the fused `linear_gelu_kernel` is used for fc1+GELU.

### 5.2 MultiModalProjector

Bridges audio and text spaces:
```python
class MultiModalProjector:
    pool_factor = 4                       # Concatenate 4 frames -> 1
    linear_1_gelu: LinearGELU(5120, 4096) # Fused Linear+GELU (fusion disabled due to shared mem)
    linear_2: Linear(4096, 2048)          # Match text hidden size
```

Pool 4 audio frames by concatenation: `(1, T/2, 1280)` -> `(1, T/8, 5120)`
Then `LinearGELU(5120->4096)` -> `Linear(4096->2048)`.

### 5.3 TextDecoder

```python
class TextDecoder:
    embed_tokens: Embedding(59264, 2048)
    rope: RotaryEmbedding(dim=128, base=500000)
    layers: [DecoderLayer x 28]
    norm: RMSNorm(2048)
```

**KV cache infrastructure** (used by `generate_v8b`):
- `allocate_kv_buffers(batch_size, max_seq_len)` — creates pre-allocated
  storage for all 28 layers' key and value states
- `forward_with_kv_buffers(embeds, kv_buffers, cache_pos)` — runs the forward
  pass writing/reading from the KV cache, returns updated `cache_pos`

### 5.4 DecoderLayer

Each decoder layer:
1. RMSNorm -> Q/K/V projections -> Reshape to heads
2. Apply full RoPE to Q and K
3. Pass 4-head K/V tensors into `scaled_dot_product_attention`
4. Inside `attention.py`, GQA handled via `_expand_kv_heads` + Flash Attention kernel
5. Causal attention (fused Flash Attention with online softmax)
6. Output projection + residual
7. RMSNorm -> SwiGLU MLP + residual

### 5.5 Generation Pipeline

**`generate()` delegates to `generate_v8b()` — both are natively in model.py.**

`generate_v8b()` uses KV-cached O(n) decode:

```python
def generate_v8b(self, input_features, input_ids=None, ...):
    # 1. Prepare inputs (encode audio, splice into text embeddings)
    inputs_embeds, seed_tokens = self._prepare_generation_inputs(...)

    # 2. Allocate KV buffers for all 28 decoder layers
    kv_buffers = self.text_decoder.allocate_kv_buffers(batch_size, max_len)

    # 3. Prefill: process full sequence once, cache all KV states
    hidden, cache_pos = self.text_decoder.forward_with_kv_buffers(
        inputs_embeds, kv_buffers, 0
    )
    logits = self.lm_head(hidden[:, -1:, :])

    # 4. Decode loop: only pass the new token each step
    for _ in range(max_new_tokens):
        next_token = self._sample_next_token(logits[:, -1, :] / temperature, ...)
        if all_finished: break
        next_embeds = self.text_decoder.embed_tokens(next_token)
        hidden, cache_pos = self.text_decoder.forward_with_kv_buffers(
            next_embeds, kv_buffers, cache_pos  # Just 1 token!
        )
        logits = self.lm_head(hidden[:, -1:, :])
```

`benchmark_student.py` checks `hasattr(model, 'generate_v8b')` and uses it
when available.

---

## 6. conv.py — Audio Feature Extraction (READ-ONLY)

Conv1D layers that downsample the mel spectrogram:

```python
class Conv1d:
    """1D convolution using im2col + matrix multiply."""

class Conv1dSubsampler:
    conv1: Conv1d(128, 1280, kernel=3, stride=1)   # Keep length
    conv2: Conv1d(1280, 1280, kernel=3, stride=2)  # Halve length
```

Uses `im2col_1d()` to reshape convolution into matrix multiply, then either
a Triton kernel (for small padded shapes on CUDA) or `torch.einsum` (fallback).

---

## 7. weight_loader.py — Model Weights (READ-ONLY)

Downloads pre-trained weights from HuggingFace and maps them to our model:

```python
def load_model_from_hf(model_id="zai-org/GLM-ASR-Nano-2512"):
    # 1. Load config from HuggingFace, create GlmAsrConfig
    # 2. Create GlmAsrModel with config
    # 3. Download and load HF model (GlmAsrForConditionalGeneration)
    # 4. Map HF state_dict keys to our model attributes
    # 5. Load processor (tokenizer + feature extractor)
    return model, processor
```

---

## 8. benchmark_student.py — Testing

### Input Preparation
```python
processor.apply_transcription_request(audio_array)
# Returns: input_features (mel spectrogram), input_ids (with audio placeholders)
```

### Generate Function Selection
```python
# benchmark_student.py checks for optimized generate methods:
if hasattr(model, 'generate_v8b'):
    generate_fn = model.generate_v8b    # KV-cached (natively in model.py)
elif hasattr(model, 'generate_v8'):
    generate_fn = model.generate_v8
elif hasattr(model, 'generate_v6'):
    generate_fn = model.generate_v6
else:
    generate_fn = model.generate        # Fallback
```

### Benchmark Loop
```python
# Warmup (compile Triton kernels, warm caches)
for _ in range(warmup_runs):
    model.generate(input_features, input_ids=input_ids, ...)

# Timed runs
for _ in range(num_runs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(...)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000  # ms
```

### Accuracy Check
```python
def check_transcription(transcription, expected):
    # Normalize: uppercase, remove punctuation
    # Compare word sets (not exact string match)
    # Pass if > 80% word overlap
```

### Current Committed Benchmark

On the RTX 5090 test box, the current committed runtime path measured:
- `109.0ms` average end-to-end (with KV-cached `generate_v8b`)
- `8.39 ms/token`
- `100.0%` transcription accuracy

---

## 9. How It All Fits Together

### Data Flow (Single Inference)

```
1. Audio WAV (16kHz, ~3.5 seconds for test audio)
   |
2. Mel Spectrogram extraction by the HF processor (128 frequency bins)
   -> Tensor: (1, 128, ~350)
   |
3. Conv Feature Extraction (conv.py)
   -> Conv1+GELU: (1, 1280, ~350)
   -> Conv2+GELU: (1, 1280, ~175)   [stride 2 halves length]
   |
4. Audio Encoder (32 transformer layers)
   For each layer:
     a. LayerNorm(hidden_states)           [layernorm_kernel]
     b. Q = Linear(normalized)             [F.linear bf16 -> cuBLAS]
     c. K = Linear(normalized)
     d. V = Linear(normalized)
     e. Reshape to (batch, heads, seq, dim)
     f. Apply partial RoPE to Q, K         [compute_freqs_kernel + torch ops]
     g. Attention = softmax(QK^T/sqrt(d))V [Flash Attention kernel]
     h. output = Linear(attention)
     i. hidden = residual + output
     j. LayerNorm(hidden)
     k. EncoderMLP: fc1 + GELU + fc2      [fused linear_gelu_kernel when FUSED=True]
     l. hidden = residual + MLP_output
   -> Tensor: (1, ~175, 1280)
   |
5. Multi-Modal Projector
   -> Pool 4 frames: (1, ~44, 5120)
   -> LinearGELU: (1, ~44, 4096)          [cuBLAS + gelu_kernel, fusion disabled]
   -> Linear: (1, ~44, 2048)
   |
6. Embed input tokens (chat template + audio placeholders)
   -> Replace audio placeholders with projected audio embeddings
   -> Combined: (1, ~80, 2048)
   |
7. Text Decoder PREFILL (28 transformer layers, one pass)
   For each layer:
     a. RMSNorm(hidden_states)             [rmsnorm_kernel]
     b. Q (16 heads) = Linear(normalized)  [F.linear bf16]
     c. K (4 heads) = Linear(normalized)   [GQA: 4 KV heads shared by 16 Q heads]
     d. V (4 heads) = Linear(normalized)
     e. Apply full RoPE to Q, K
     f. Causal Attention via Flash Attention kernel (GQA via _expand_kv_heads)
     g. KV states cached in kv_buffers     [forward_with_kv_buffers]
     h. RMSNorm(hidden)
     i. SwiGLU MLP:
        gate = SiLU(gate_proj(x))          [silu_kernel or swiglu_fused_kernel]
        up = up_proj(x)
        down = down_proj(gate * up)
     j. hidden = residual + MLP_output
   |
8. Final RMSNorm + LM Head
   -> Logits: (1, ~80, 59264)
   -> Take last position: (1, 59264)
   |
9. Autoregressive Decode (repeat for each new token)
   With KV cache (generate_v8b — native in model.py):
     a. Embed new token: (1, 1, 2048)
     b. forward_with_kv_buffers: only 1 token through 28 layers
        (reads cached KV, appends new KV, computes attention over all cached keys)
     c. RMSNorm + LM Head -> next token logits
   -> Generates ~13 tokens for test audio
   |
10. Decode token IDs -> text
    -> "Concord returned to its place amidst the tents."
```

### Kernel Call Count (Approximate, per full inference with 13 generated tokens)

**With KV cache (`generate_v8b`):** Prefill processes the full sequence once.
Each decode step processes only 1 token through the decoder.

| Kernel | Encode | Prefill | Per Decode | x13 Steps | Total |
|--------|:------:|:-------:|:----------:|:---------:|:-----:|
| layernorm_kernel | 64 | 0 | 0 | 0 | 64 |
| rmsnorm_kernel | 0 | 56 | 56 | 728 | 784 |
| gelu_kernel | 33 | 0 | 0 | 0 | 33 |
| silu_kernel | 0 | 28 | 28 | 364 | 392 |
| linear (cuBLAS bf16) | ~160 | ~168 | ~168 | ~2184 | ~2512 |
| flash_attention_kernel | 32 | 28 | 28 | 364 | 424 |
| compute_freqs | 1 | 1 | 0 | 0 | 2 |
| softmax (standalone) | 0 | 1 | 1 | 13 | 14 |

Note: gelu_kernel count includes 32 from encoder `EncoderMLP` (when fused, this
becomes part of `linear_gelu_kernel` instead) + 1 from projector `LinearGELU`
(unfused, so standalone gelu_kernel).

**Key difference from no-KV-cache path:** With KV cache, each decode step's
attention operates on a (1, heads, 1, head_dim) query against the growing
cached KV, instead of reprocessing the entire sequence. The compute per
decode step is O(1) in sequence length (for everything except attention,
which is O(n) for reading cached keys — but this is just reading, not
recomputing through 28 layers).
