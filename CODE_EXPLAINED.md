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
- `model.py` — model architecture and generation loop
- `weight_loader.py` — loads pre-trained weights from HuggingFace
- `conv.py` — 1D convolution for audio subsampling

You can only modify: `layers.py`, `attention.py`, `rope.py`, `__init__.py`.

---

## 2. layers.py — GPU Compute Kernels

This is the core file containing all neural network building blocks.

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

**Purpose:** Non-linear activation function for audio encoder MLP.

**Math (tanh approximation):**
```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Uses `tl.extra.cuda.libdevice.tanh` for the tanh computation — this calls into
NVIDIA's libdevice math library for hardware-optimized transcendental functions.

**Where used:** Audio encoder MLP (`fc1 -> GELU -> fc2`), Projector MLP

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

**linear_gelu_kernel:** Computes `GELU(x @ W)` in a single kernel launch.
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
- `BACKEND = "torch"`: Uses `x @ weight.t()` (dispatches to cuBLAS GEMM)
- `BACKEND = "triton"`: Uses `linear_kernel_tf32`
- On RTX 5090, cuBLAS is slightly faster (214ms vs 226ms)
- Caches transposed/padded weights (`_weight_t_padded`)

**`MLP` class:** Implements SwiGLU gating for the text decoder:
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
When `FUSED=True`, uses `swiglu_fused_kernel` for the gate+up computation.

**`EncoderMLP` class:** Simpler MLP without gating for audio encoder:
```
output = fc2(GELU(fc1(x)))
```
When `FUSED=True`, uses `linear_gelu_kernel` for fc1+GELU.
Note: The original `model.py` does NOT use `EncoderMLP` — it uses separate
`fc1`/`fc2` + `gelu()` calls. So this class exists in layers.py but is not
instantiated by the unmodified model.

**`LinearGELU` class:** A helper for `GELU(Linear(x))` used by the projector
*only if model.py creates one*. The original model.py uses separate
`linear_1` + `gelu()` calls instead.

---

## 3. attention.py — Attention Mechanism

### 3.1 Scaled Dot-Product Attention

The core of every transformer. Computes:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

This is implemented as three separate kernels:
1. `attention_scores_kernel`: Computes `Q @ K^T * scale`
2. `softmax_inplace_kernel`: Applies softmax to scores
3. `attention_output_kernel`: Computes `softmax_scores @ V`

### 3.2 Attention Scores Kernel

For each query position `q` and each batch-head `bh`:
```
Load Q[bh, q, :] as a 1D vector (head_dim elements)
Load K[bh, :, :] as a 2D matrix (seq_k x head_dim)
scores = sum(K * Q[broadcast], dim=-1) * scale
```

This is a matrix-vector product: one row of Q dotted with all rows of K.

### 3.3 Grouped Query Attention (GQA)

The text decoder uses GQA: 16 query heads but only 4 KV heads.
Each KV head is shared by 4 query heads. This reduces KV cache memory by 4x.

In the original model.py, GQA is handled by expanding KV heads:
```python
# In model.py DecoderLayer:
k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
```

### 3.4 Causal Masking

For autoregressive generation, position `i` can only attend to positions `<= i`:
```python
scores = tl.where(offs_k > current_pos, -1e9, scores)
```
Setting future positions to `-1e9` makes them zero after softmax.

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

**AudioEncoderLayer MLP:** Uses separate `fc1` + `gelu()` + `fc2` calls
(NOT the `EncoderMLP` class, even though it exists in layers.py).

### 5.2 MultiModalProjector

Bridges audio and text spaces:
```python
class MultiModalProjector:
    pool_factor = 4  # Concatenate 4 frames -> 1
    linear_1: Linear(5120, 4096)    # 1280*4 = 5120
    act = gelu                       # Standalone gelu() function
    linear_2: Linear(4096, 2048)    # Match text hidden size
```

Pool 4 audio frames by concatenation: `(1, T/2, 1280)` -> `(1, T/8, 5120)`
Then `linear_1` -> `gelu` -> `linear_2`.

### 5.3 TextDecoder

```python
class TextDecoder:
    embed_tokens: Embedding(59264, 2048)
    rope: RotaryEmbedding(dim=128, base=500000)
    layers: [DecoderLayer x 28]
    norm: RMSNorm(2048)
```

### 5.4 DecoderLayer

Each decoder layer:
1. RMSNorm -> Q/K/V projections -> Reshape to heads
2. Apply full RoPE to Q and K
3. Expand KV heads (4 -> 16 for GQA)
4. Causal attention (scaled_dot_product_attention)
5. Output projection + residual
6. RMSNorm -> SwiGLU MLP + residual

The layer supports KV cache (via `past_key_value` parameter), but the `generate()`
method in `GlmAsrModel` concatenates the full sequence each step instead of
using cached KV, so cache is not leveraged for performance.

### 5.5 Generation Pipeline

```python
def generate(input_features, input_ids, ...):
    # 1. Encode audio
    audio_embeds = self.encode_audio(input_features)

    # 2. Splice audio embeddings into token sequence
    inputs_embeds = [text_before_audio | audio_embeds | text_after_audio]

    # 3. Autoregressive loop (NO KV cache)
    for _ in range(max_new_tokens):
        logits = self.decode(inputs_embeds=inputs_embeds)  # Full sequence!
        next_token = sample_or_argmax(logits[:, -1, :])
        if next_token == EOS: break
        new_embeds = embed(next_token)
        inputs_embeds = cat([inputs_embeds, new_embeds], dim=1)  # Grows each step

    return generated_token_ids
```

**Performance implication:** Each decode step processes the ENTIRE sequence through
28 decoder layers. With 13 tokens generated, the last step processes ~93 tokens
through 28 layers. This is O(n^2) in generation length — the main bottleneck.

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
     b. Q = Linear(normalized)             [linear_kernel_tf32 or cuBLAS]
     c. K = Linear(normalized)
     d. V = Linear(normalized)
     e. Reshape to (batch, heads, seq, dim)
     f. Apply partial RoPE to Q, K         [compute_freqs_kernel + torch ops]
     g. Attention = softmax(QK^T/sqrt(d))V [attention kernels]
     h. output = Linear(attention)
     i. hidden = residual + output
     j. LayerNorm(hidden)
     k. MLP: fc1 + GELU + fc2             [gelu_kernel + linear ops]
     l. hidden = residual + MLP_output
   -> Tensor: (1, ~175, 1280)
   |
5. Multi-Modal Projector
   -> Pool 4 frames: (1, ~44, 5120)
   -> Linear + GELU: (1, ~44, 4096)
   -> Linear: (1, ~44, 2048)
   |
6. Embed input tokens (chat template + audio placeholders)
   -> Replace audio placeholders with projected audio embeddings
   -> Combined: (1, ~80, 2048)
   |
7. Text Decoder (28 transformer layers) x N decode steps
   For each layer:
     a. RMSNorm(hidden_states)             [rmsnorm_kernel]
     b. Q (16 heads) = Linear(normalized)
     c. K (4 heads) = Linear(normalized)   [GQA: 4 KV heads shared by 16 Q heads]
     d. V (4 heads) = Linear(normalized)
     e. Expand KV: 4 -> 16 heads
     f. Apply full RoPE to Q, K
     g. Causal Attention (no looking ahead)
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
9. Autoregressive Generation (repeat steps 7-8)
   -> Generates ~13 tokens for test audio
   -> Each step re-runs the FULL sequence (no KV cache in original model.py)
   |
10. Decode token IDs -> text
    -> "Concord returned to its place amidst the tents."
```

### Kernel Call Count (Approximate, per full inference with 13 generated tokens)

The original model.py's `generate()` re-runs the entire sequence each step,
so the kernel counts grow with generation length.

| Kernel | Per Encode | Per Decode Step | x13 Steps | Total |
|--------|:----------:|:---------------:|:---------:|:-----:|
| layernorm_kernel | 64 | 0 | 0 | 64 |
| rmsnorm_kernel | 0 | 56 | 728 | 728 |
| gelu_kernel | 32 | 0 | 0 | 32 |
| silu_kernel | 0 | 28 | 364 | 364 |
| linear (cuBLAS/triton) | ~160 | ~168 | ~2184 | ~2344 |
| attention kernels | 96 | 84 | 1092 | 1188 |
| compute_freqs | 1 | 1 | 13 | 14 |
| softmax (standalone) | 0 | 1 | 13 | 13 |

**Total kernel launches per inference: ~4700+**

This is why the decode loop dominates runtime — 13 full decoder passes through
28 layers, each with multiple kernel launches. KV cache would reduce this to
just the new token's computation per step.
