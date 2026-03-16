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
10. [Optimization Sources](#10-optimization-sources)

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

Per GUIDE.md, these files are **read-only** (must match origin/main):
- `model.py` — model architecture and generation loop (stock `generate()`, no KV cache)
- `weight_loader.py` — loads pre-trained weights from HuggingFace
- `conv.py` — 1D convolution for audio subsampling

You can only modify: `layers.py`, `attention.py`, `rope.py`, `__init__.py`.

---

## 2. layers.py — GPU Compute Kernels

This is the core file containing all neural network building blocks as Triton
kernels and Python layer classes.

### 2.1 GPU Detection: GPUProfile + _KNOWN_CONFIGS

The first thing layers.py does (after imports) is detect the GPU and set tile sizes.

```python
# _KNOWN_CONFIGS — tested tile configurations for 6 GPU architectures
# Each entry: {attn_tiles: {head_dim: (BLOCK_M, BLOCK_N, nstages, nwarps)},
#              matmul_tiles: (TILE_M, TILE_N, TILE_K),
#              rope_nstages, rope_nwarps}

class GPUProfile:
    """Detects GPU at import time, stores optimal tile sizes for all kernels."""
    def __init__(self):
        # 1. Read sm_version, shared_memory_per_block_optin, gpu_name
        # 2. Classify architecture (blackwell_consumer, ada, hopper, etc.)
        # 3. Look up _KNOWN_CONFIGS for tested GPU → direct tile assignment
        # 4. Unknown GPU → compute tiles dynamically from shared memory budget

    def get_attention_tiles(self, head_dim, seq_q=None):
        # Returns (BLOCK_M, BLOCK_N, nstages, nwarps)
        # Clamps BLOCK_M to 16 for seq_q <= 16 (KV-cached decode)

GPU = GPUProfile()  # Module-level singleton, computed once at import
```

**Why `shared_memory_per_block_optin`?** The default `shared_memory_per_block` returns
48KB on all GPUs. The optin value is what Triton can actually request — 99KB on RTX 5090,
228KB on H200. Using the wrong property led to the old code always detecting "consumer"
even on datacenter GPUs (it hit the except fallback).

**Robust fallback:** The property is read via a `getattr` chain:
`shared_memory_per_block_optin` → `max_shared_memory_per_block` → `shared_memory_per_block`.
This prevents silent fallback to CPU profile on older PyTorch versions that lack the optin
property. Without this, H200s running older PyTorch would get consumer-sized tiles (64x64)
instead of datacenter tiles (128x128).

**Dynamic tile computation** for unknown GPUs:
- `_compute_attention_tiles(head_dim, smem_bytes)`: Tries ranked balanced configs
  (e.g., 128x128, 128x64, 64x64) largest first. Formula:
  `(BLOCK_M + 2*BLOCK_N) * BLOCK_D * 4 + 20KB overhead`
- `_compute_matmul_tiles(smem_bytes)`: Uses SwiGLU worst case (gate + up tiles):
  `TILE_K * (TILE_M + 2*TILE_N) * 4 + 20KB overhead`

### 2.2 Helper Functions

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

**Where used:** Audio encoder MLP (standalone `gelu()` call after `fc1`), Projector (`self.act = gelu`)

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

**Where used:** Every Linear layer in the model (Q, K, V, O projections, MLP layers, LM head)

### 2.7 Softmax Kernel

**Purpose:** Converts raw logits to probability distributions.

**Math:** `y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

**Where used:** Final token prediction (standalone softmax)

### 2.8 Fused Kernels

**linear_gelu_kernel:** Computes `GELU(x @ W + b)` in a single kernel launch.
Instead of: matmul -> write to DRAM -> read from DRAM -> GELU,
it does: matmul -> GELU (all in registers). Eliminates one DRAM round-trip.
**Note:** This kernel exists in layers.py but is NOT currently used by model.py
(model.py calls `fc1` then `gelu` separately).

**swiglu_fused_kernel:** Computes `SiLU(x @ W_gate) * (x @ W_up)` in one kernel.
Fuses THREE operations: two matmuls and the gating. Input `x` is loaded once.
**Active when `MLP.FUSED = True`** — used by the decoder MLP.

### 2.9 Layer Classes

**`RMSNorm` class:** Wraps the kernel with device management and fallback.
- Checks if hidden_size fits in BLOCK_SIZE (must be power of 2)
- Falls back to PyTorch if not on CUDA

**`Linear` class:** Switchable between torch (cuBLAS) and Triton backends.
- `BACKEND = "torch"`: Uses `F.linear(...)` → cuBLAS/cuBLASLt (current, fastest)
- `BACKEND = "triton"`: Uses `linear_kernel_tf32`
- `BF16 = True` (class default): Enables half-precision weight caching
- `_HALF_DTYPE = torch.float16`: Actual dtype used for cuBLAS HGEMM (faster than bf16 on RTX 5090)
- **fp16-throughout pipeline:** Output stays fp16 (`.float()` removed), cascading fp16 through
  the entire model. Triton kernels handle float32 precision internally via `.to(tl.float32)`.

**`MLP` class:** Implements SwiGLU gating for the text decoder:
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
When `FUSED=True`, uses `swiglu_fused_kernel` for the gate+up computation.
**This IS used by model.py** — the decoder's `self.mlp = MLP(...)`.

**`EncoderMLP` class:** Simpler MLP without gating for the audio encoder:
```
output = fc2(GELU(fc1(x)))
```
When `FUSED=True`, uses `linear_gelu_kernel` for fc1+GELU.
**NOT used by origin/main model.py** — the encoder uses plain `self.fc1 = Linear(...)`
and calls `gelu()` inline. The class exists for compatibility but is dead code.

**`LinearGELU` class:** A `GELU(Linear(x))` wrapper.
**NOT used by origin/main model.py** — the projector uses plain `self.linear_1 = Linear(...)`
and calls `self.act()` inline. Dead code.

---

## 3. attention.py — Attention Mechanism

### 3.1 Scaled Dot-Product Attention

The core of every transformer. Computes:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 3.2 Primary Path: Fused Flash Attention Kernel (Triton)

The committed runtime path uses a fused Triton Flash Attention kernel with
**online softmax** as the primary attention implementation.

```python
# attention.py — scaled_dot_product_attention()
if q.is_cuda:
    if use_gqa:
        k = _expand_kv_heads(k, num_heads)
        v = _expand_kv_heads(v, num_heads)
    flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, output, mask_flat, scale, ...,
        IS_CAUSAL=is_causal, HAS_MASK=has_mask,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=head_dim_padded,
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
    m_new = max(m_i, max(S))        # updated running max
    alpha = exp(m_i - m_new)        # rescale factor for old state
    p = exp(S - m_new)              # new attention weights
    l_i = alpha * l_i + sum(p)      # updated running sum
    acc = alpha * acc + p @ V_block # rescale old + accumulate new (tl.dot)
    m_i = m_new

output = acc / l_i                  # final normalization
```

**Why this is better than the 3-kernel approach:**
1. **No DRAM scores matrix** — everything stays in SRAM/registers.
2. **Single kernel launch** — no synchronization overhead.
3. **O(BLOCK) SRAM** — memory-efficient for long sequences.
4. **Tensor cores** — `tl.dot` for both Q@K^T and P@V.

**Tile sizes** (GPUProfile-aware via `GPU.get_attention_tiles(head_dim, seq_q)`):
- **Consumer GPUs** (RTX 4090/5090, ~100KB optin shared mem):
  - Encoder (head_dim=64): `BLOCK_M=64, BLOCK_N=64`, `num_stages=1, num_warps=4`
  - Decoder (head_dim=128): `BLOCK_M=32, BLOCK_N=32`, `num_stages=1, num_warps=4`
  - `seq_q <= 16`: `BLOCK_M=16` (optimized for KV-cached decode)
- **Datacenter GPUs** (H200/B200, ~228KB optin shared mem):
  - Encoder (head_dim=64): `BLOCK_M=128, BLOCK_N=128`, `num_stages=2, num_warps=8`
  - Decoder (head_dim=128): `BLOCK_M=128, BLOCK_N=64`, `num_stages=2, num_warps=8`
- **Unknown GPUs**: tiles computed dynamically by `_compute_attention_tiles()`

**Features:**
- `IS_CAUSAL` (constexpr): causal masking for decoder
- `HAS_MASK` (constexpr): additive attention mask bias (zero overhead when False)
- Supports arbitrary sequence lengths

### 3.3 SDPA Fallback for KV-Cached Decode

For single-token decode steps (seq_q ≤ 4) during KV-cached generation, the Triton
Flash Attention kernel's launch overhead dominates. Instead, we fall back to
`torch.nn.functional.scaled_dot_product_attention` (PyTorch's SDPA), which uses
cuDNN/cuBLAS internally and avoids the Triton compilation/launch path.

```python
if q.is_cuda and seq_q <= 4:
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=scale
    )
```

**Impact:** -3ms on decode steps (113.5ms → 110.0ms, further to 98.5ms with fp16 pipeline).

### 3.4 Legacy Attention Kernels (REMOVED)

Three legacy kernels from the original assignment were **removed** (~175 lines):
1. ~~`attention_scores_kernel`~~: Q @ K^T * scale
2. ~~`softmax_inplace_kernel`~~: In-place softmax
3. ~~`attention_output_kernel`~~: attn_weights @ V
4. ~~`causal_mask_kernel`~~: Causal mask generation

These were never invoked at runtime — fully superseded by `flash_attention_kernel`.

### 3.5 Grouped Query Attention (GQA)

The text decoder uses GQA: 16 query heads but only 4 KV heads.
GQA is handled by `_expand_kv_heads()` before the Flash Attention kernel call.

### 3.6 Numerical Parity Tests

17-case deterministic parity suite in `__main__` block, covering:
- basic and causal attention at head_dim=64 and head_dim=128
- additive masks with both `(batch,1,seq_q,seq_k)` and `(batch,heads,seq_q,seq_k)` layouts
- GQA cases (16Q/4KV and 4Q/2KV)
- encoder-like ragged lengths (175), decoder-like prefill (93)
- single-token decode, decode with causal+mask
- non-power-of-two shapes (17x61)

---

## 4. rope.py — Positional Encodings

### 4.1 What is RoPE?

Rotary Position Embeddings encode position information by rotating the
query and key vectors in 2D subspaces. For dimensions `(x1, x2)` at position `p`:
```
x1_rot = x1 * cos(p*freq) - x2 * sin(p*freq)
x2_rot = x2 * cos(p*freq) + x1 * sin(p*freq)
```

### 4.2 Partial RoPE

The audio encoder uses 50% partial RoPE — only the first half of each
head's dimensions are rotated. The rest pass through unchanged.

### 4.3 Kernel Implementation

The `compute_freqs_kernel` precomputes cos/sin for all positions.
The duplication (first half = second half) is because `apply_rotary_pos_emb`
splits the input into halves, each multiplied by the same cos/sin.

---

## 5. model.py — Full Model Pipeline (READ-ONLY, origin/main)

This file cannot be modified. Understanding it helps you implement correct kernels.

### 5.1 AudioEncoder

```python
class AudioEncoder:
    conv1: Conv1d(128 -> 1280, k=3, s=1)
    conv2: Conv1d(1280 -> 1280, k=3, s=2)
    rotary_emb: RotaryEmbedding(dim=64, partial=0.5)
    layers: [AudioEncoderLayer x 32]
    layer_norm: LayerNorm(1280)
```

**AudioEncoderLayer MLP:** Uses plain `Linear` layers, NOT `EncoderMLP`:
```python
# model.py lines 86-87, 127-129:
self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
self.fc2 = Linear(intermediate_size, hidden_size, bias=True)
# ...
hidden_states = self.fc1(hidden_states)
hidden_states = gelu(hidden_states)   # standalone gelu() call
hidden_states = self.fc2(hidden_states)
```

### 5.2 MultiModalProjector

Uses plain `Linear` layers, NOT `LinearGELU`:
```python
# model.py lines 578-580, 619-621:
self.linear_1 = Linear(pooled_dim, config.projector_hidden_size, bias=True)
self.act = gelu
self.linear_2 = Linear(config.projector_hidden_size, config.text_hidden_size, bias=True)
# ...
hidden_states = self.linear_1(pooled)
hidden_states = self.act(hidden_states)   # standalone gelu() call
hidden_states = self.linear_2(hidden_states)
```

Pool 4 audio frames by concatenation: `(1, T/2, 1280)` -> `(1, T/8, 5120)`
Then `Linear(5120->4096) + gelu + Linear(4096->2048)`.

### 5.3 TextDecoder

```python
class TextDecoder:
    embed_tokens: Embedding(59264, 2048)
    rope: RotaryEmbedding(dim=128, base=500000)
    layers: [DecoderLayer x 28]
    norm: RMSNorm(2048)
```

**KV cache infrastructure exists** in origin/main but is NOT used by `generate()`:
- `TextDecoderLayer.forward_with_kv_buffer()` (line 318)
- `TextDecoder.forward_with_kv_buffers()` (line 492)
- `TextDecoder.allocate_kv_buffers()` (line 534)

The stock `generate()` ignores these methods — each forward pass processes the
full input sequence. `generate_v8b` (monkey-patched from layers.py) uses
`decode(use_cache=True)` which internally leverages the KV cache via
`TextDecoder.__call__`'s `past_key_values` parameter.

### 5.4 DecoderLayer

Each decoder layer:
1. RMSNorm -> Q/K/V projections -> Reshape to heads
2. Apply full RoPE to Q and K
3. Pass 4-head K/V tensors into `scaled_dot_product_attention`
4. GQA handled via `_expand_kv_heads` + Flash Attention kernel
5. Causal attention
6. Output projection + residual
7. RMSNorm -> SwiGLU MLP + residual

### 5.5 Generation Pipeline

**`generate()`** is the stock generation method in model.py — O(n²) decode.
`generate_v8b` is monkey-patched from layers.py (see Section 10) and uses
`decode(use_cache=True)` for O(n) KV-cached generation.
Stock generate() reprocesses the full growing sequence each step:

```python
def generate(self, input_features, input_ids=None, ...):
    # 1. Encode audio
    audio_embeds = self.encode_audio(input_features, ...)

    # 2. Build initial inputs_embeds (audio + text tokens)
    inputs_embeds = torch.cat([before_audio, audio_embeds, after_audio], dim=1)

    # 3. Autoregressive decode — O(n²)
    for _ in range(max_new_tokens):
        logits = self.decode(inputs_embeds=inputs_embeds)  # Full sequence!
        next_token = sample(logits[:, -1, :])
        if eos: break
        new_embeds = self.text_decoder.embed_tokens(next_token)
        inputs_embeds = torch.cat([inputs_embeds, new_embeds], dim=1)  # Grows!
```

Each decode step processes the ENTIRE `inputs_embeds` through all 28 decoder layers.
This is why decoder decode dominates at 82.8% in detailed benchmarks.

---

## 6. conv.py — Audio Feature Extraction (READ-ONLY)

Conv1D layers that downsample the mel spectrogram:
```python
class Conv1dSubsampler:
    conv1: Conv1d(128, 1280, kernel=3, stride=1)   # Keep length
    conv2: Conv1d(1280, 1280, kernel=3, stride=2)  # Halve length
```

Uses `im2col_1d()` to reshape convolution into matrix multiply, then either
a Triton kernel or `torch.einsum` (fallback).

---

## 7. weight_loader.py — Model Weights (READ-ONLY)

Downloads pre-trained weights from HuggingFace and maps them to our model.

---

## 8. benchmark_student.py — Testing

### Generate Function Selection
```python
# benchmark_student.py checks for optimized generate methods:
if hasattr(model, 'generate_v8b'):
    generate_fn = model.generate_v8b    # Monkey-patched from layers.py (KV-cached, O(n))
elif hasattr(model, 'generate_v8'):
    generate_fn = model.generate_v8     # Not available
elif hasattr(model, 'generate_v6'):
    generate_fn = model.generate_v6     # Not available
else:
    generate_fn = model.generate        # Stock O(n²) from model.py
```

When `generate_v8b` is monkey-patched onto GlmAsrModel (via `_try_patch_v8b()` in
layers.py), the benchmark automatically detects and uses it.

### Accuracy Check
```python
def check_transcription(transcription, expected):
    # Normalize: uppercase, remove punctuation
    # Compare word sets (not exact string match)
    # Pass if > 80% word overlap
```

### Current Benchmark (RTX 5090, 2026-03-15)
- With fp16 pipeline + generate_v8b + SDPA fallback: `98.5ms`, `7.58 ms/token`
- With bf16 pipeline + generate_v8b + SDPA fallback: `110.0ms (+/- 0.2ms)`, `8.46 ms/token`
- Without generate_v8b: `120.7ms (+/- 0.2ms)`, `9.29 ms/token`
- 13 tokens generated, `100.0%` transcription accuracy
- **Competition:** ankush 98.5ms, meave 127.8ms, yash 128ms, majed 187.9ms

**NOTE:** `benchmark_detailed.py` fails with fp16 pipeline (expects float32 projector output).
Student benchmark (authoritative) works perfectly.

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
     b. Q = Linear(normalized)             [F.linear fp16 -> cuBLAS HGEMM]
     c. K = Linear(normalized)
     d. V = Linear(normalized)
     e. Reshape to (batch, heads, seq, dim)
     f. Apply partial RoPE to Q, K         [compute_freqs_kernel + torch ops]
     g. Attention = softmax(QK^T/sqrt(d))V [Flash Attention kernel]
     h. output = Linear(attention)
     i. hidden = residual + output
     j. LayerNorm(hidden)
     k. fc1(x) -> gelu(x) -> fc2(x)       [cuBLAS + standalone gelu_kernel]
     l. hidden = residual + MLP_output
   -> Tensor: (1, ~175, 1280)
   |
5. Multi-Modal Projector
   -> Pool 4 frames: (1, ~44, 5120)
   -> Linear(5120->4096): (1, ~44, 4096)   [cuBLAS fp16 HGEMM, output stays fp16]
   -> gelu: (1, ~44, 4096)                 [gelu_kernel, fp16 in/out]
   -> Linear(4096->2048): (1, ~44, 2048)   [cuBLAS fp16 HGEMM]
   |
6. Embed input tokens (chat template + audio placeholders)
   -> Replace audio placeholders with projected audio embeddings
   -> Combined: (1, ~80, 2048)
   |
7. Text Decoder (28 transformer layers, full sequence each step)
   For each layer:
     a. RMSNorm(hidden_states)             [rmsnorm_kernel]
     b. Q (16 heads) = Linear(normalized)  [F.linear fp16]
     c. K (4 heads) = Linear(normalized)   [GQA: 4 KV heads shared by 16 Q heads]
     d. V (4 heads) = Linear(normalized)
     e. Apply full RoPE to Q, K
     f. Causal Attention via Flash Attention kernel (GQA via _expand_kv_heads)
     g. RMSNorm(hidden)
     h. SwiGLU MLP:
        gate = SiLU(gate_proj(x))          [swiglu_fused_kernel when MLP.FUSED=True]
        up = up_proj(x)
        down = down_proj(gate * up)
     i. hidden = residual + MLP_output
   |
8. Final RMSNorm + LM Head
   -> Logits: (1, ~80, 59264)
   -> Take last position: (1, 59264)
   |
9. Autoregressive Decode
   Stock generate() — O(n²): reprocesses full sequence each step
   generate_v8b (KV-cached) — O(n): 1 token per step via decode(use_cache=True)
     a. Prefill: logits, past_kv = decode(inputs_embeds=..., use_cache=True)
     b. Decode loop:
        - Embed new token: (1, 1, 2048)
        - logits, past_kv = decode(inputs_embeds=new_embeds,
                                    past_key_values=past_kv, use_cache=True)
        - Attention uses SDPA fallback (seq_q=1) instead of Triton Flash kernel
     -> Generates ~13 tokens for test audio
   |
10. Decode token IDs -> text
    -> "Concord returned to its place amidst the tents."
```

### Kernel Call Count (Approximate, per full inference with 13 generated tokens)

**Stock generate() — O(n²), no KV cache:**
Each decode step reprocesses the full growing sequence.

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

Note: gelu_kernel count = 32 from encoder fc1→gelu + 1 from projector linear_1→act.
These are standalone `gelu()` calls, NOT fused (model.py doesn't use EncoderMLP/LinearGELU).

---

## 10. Optimization Sources

Optimizations adopted or planned from analysis of other branches:

### Currently Active
| Optimization | Source | Description |
|-------------|--------|-------------|
| cuBLAS backend | **majed**, **yash/optimize** | `F.linear` for all Linear layers |
| fp16 weights (was bf16) | **yash/optimize**, **majed**, **meave** | Cache fp16 copies, halve memory traffic, fp16 HGEMM |
| Flash Attention | **majed**, **meave** | Triton kernel with online softmax |
| Fused SwiGLU | **yash/optimize** | Single kernel for gate+up in decoder MLP |
| TF32 flags | Common | `allow_tf32`, `set_float32_matmul_precision("high")` |

### Adopted (tested, confirmed improvement)
| Optimization | Source | Actual Impact |
|-------------|--------|---------------|
| Fused Q+K RoPE pair kernel | **meave** | **-14ms** (138→124ms) |
| bf16 RMSNorm output kernel | **meave** (adapted for bf16) | **-3ms** (124→121ms) |

### Adopted (2026-03-13 to 2026-03-15)
| Optimization | Source | Actual Impact |
|-------------|--------|---------------|
| bf16 LayerNorm output | internal | **-0.7ms** (encoder norm stores bf16 directly) |
| generate_v8b (KV cache) | internal | **-7.6ms** (monkey-patched from layers.py, uses decode(use_cache=True)) |
| SDPA fallback for seq_q≤4 | internal | **-3ms** (PyTorch SDPA for KV-cached decode steps) |
| GPUProfile + _KNOWN_CONFIGS + dynamic tiles | internal | portability (7 arch classifications, dynamic fallback) |
| Dead code cleanup | internal | -320 lines (removed legacy attention kernels) |

### Adopted (2026-03-15, fp16-throughout pipeline)
| Optimization | Source | Actual Impact |
|-------------|--------|---------------|
| fp16 cuBLAS HGEMM (was bf16) | internal | ~-0.4ms (fp16 HGEMM slightly faster on RTX 5090) |
| Smaller flash attention tiles | **meave** | improved prefill (64x64 encoder, 32x32 decoder) |
| Remove Linear `.float()` conversion | internal | **-7.5ms** (biggest single win — fp16 cascades through pipeline) |
| Remove silu/gelu float32 cast | internal | **-3.7ms** (kernels do `.to(tl.float32)` internally) |
| Remove RMSNorm/LayerNorm float32 cast | internal | ~-0.5ms (same reasoning) |
| fp16 embedding output | internal | keeps decoder pipeline in fp16 from start |
| fp16 fused SwiGLU/EncoderMLP | internal | halves intermediate memory bandwidth |
| Remove flash attention float32 conversion | internal | ~-1ms (pass fp16 to kernel directly) |
| Norm kernel output fp16 (was bf16) | internal | matches fp16 pipeline |
| BLOCK_M=16 for seq_q<=16 | **meave** | optimized for KV-cached decode |
| topk instead of argsort in sampling | internal | neutral (cleaner code) |

### Rejected (tested, did not help on RTX 5090)
| Optimization | Source | Result |
|-------------|--------|--------|
| SwiGLU grid swizzling | **yash/optimize** | +18ms regression with GROUP_SIZE_M=8, 1D grid |
| @triton.autotune GELU/SiLU | **majed** | +0.7ms tuning overhead |
| @triton.autotune Flash Attention | internal | Massive regression — seq_k changes every decode step with KV cache |
| @triton.autotune SwiGLU | internal | Regression — wrapper overhead dominates small decode matmuls |
| Softmax bf16 output | internal | 0ms — softmax only used for final logits (not in hot path) |
| Flash Attention num_stages=2 | **yash/optimize** | OOM on consumer GPUs (~100KB shared mem) |
| Flash Attention num_warps=8 | **yash/optimize** | 0ms change on RTX 5090 |
| PyTorch GELU/SiLU bf16 | internal | +0.3ms — Triton kernels faster |
| PyTorch SDPA for prefill/encoder | internal | +6ms — Triton flash kernel faster (114.5 vs 108ms) |
| SDPA enable_gqa=True for decode | internal | +13ms — manual KV expansion + standard SDPA faster |
| Fused gate+up Linear in MLP | internal | Neutral — kernel launch savings offset by reshape overhead |

### Not Applicable
| Optimization | Source | Why Not |
|-------------|--------|---------|
| EncoderMLP.FUSED | yash/optimize | model.py (origin/main) doesn't use EncoderMLP class |
| LinearGELU.FUSED | yash/optimize | model.py (origin/main) doesn't use LinearGELU class |
| flash_decode_kernel | meave | generate_v8b uses same flash_attention_kernel for decode |
