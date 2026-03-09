# Reference Guide: GLM-ASR Triton Kernel Project

Quick reference for kernel signatures, model architecture, and performance tuning.

---

## Model Configuration

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
| **Projector** | Pool factor | 4 |
| | Hidden | 5120 -> 4096 -> 3584 |
| **Text Decoder** | Hidden size | 3584 |
| | Q heads | 28 |
| | KV heads | 4 (GQA, 7:1 ratio) |
| | Head dim | 128 |
| | Num layers | 28 |
| | Intermediate | 18944 |
| | Norm | RMSNorm |
| | Activation | SiLU/SwiGLU |
| | RoPE | 100% (rotary_dim=128) |
| **LM Head** | Vocab size | 151552 |

---

## Kernel Signatures

### layers.py

```python
# RMSNorm - Grid: (batch_size,)
rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)

# LayerNorm - Grid: (batch_size,)
layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)

# GELU - Grid: (cdiv(n_elements, BLOCK_SIZE),)
gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)

# SiLU - Grid: (cdiv(n_elements, BLOCK_SIZE),)
silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)

# Linear - Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
linear_kernel_tf32(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K)

# Softmax - Grid: (batch_size,)
softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE)
```

### attention.py

```python
# Attention Scores - Grid: (batch_heads, seq_q)
attention_scores_kernel(q_ptr, k_ptr, scores_ptr, scale, seq_k, head_dim,
                        stride_q0..q2, stride_k0..k2, stride_s0..s2,
                        BLOCK_K, BLOCK_D)

# Softmax In-place - Grid: (batch_heads * seq_q,)
softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE)

# Attention Output - Grid: (batch_heads, seq_q)
attention_output_kernel(attn_ptr, v_ptr, output_ptr, seq_k, head_dim,
                        stride_w0..w2, stride_v0..v2, stride_o0..o2,
                        BLOCK_K, BLOCK_D)
```

### rope.py

```python
# RoPE Frequencies - Grid: (seq_len,)
compute_freqs_kernel(positions_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                     seq_len, half_dim,
                     stride_pos, stride_inv,
                     stride_cos0, stride_cos1, stride_sin0, stride_sin1,
                     BLOCK)
```

---

## Tensor Shapes Through the Pipeline

```
Input: audio_array (float32, 16kHz)

Mel Spectrogram:     (1, 128, T)           # T ~ 3000 for 30s audio
Conv1 output:        (1, 1280, T)          # Same length, gelu
Conv2 output:        (1, 1280, T/2)        # Stride 2, gelu
Permute:             (1, T/2, 1280)        # (batch, seq, hidden)

Encoder (32 layers):
  Q/K/V proj:        (1, T/2, 1280)        # Linear
  Reshape:           (1, 20, T/2, 64)      # (batch, heads, seq, head_dim)
  RoPE:              (1, 20, T/2, 64)      # Partial (first 32 dims)
  Attention:         (1, 20, T/2, 64)      # Scaled dot-product
  MLP:               (1, T/2, 1280) -> 5120 -> 1280  # GELU

Encoder output:      (1, T/2, 1280)
Pool 4 frames:       (1, T/8, 5120)        # Concatenate
Projector:           (1, T/8, 3584)        # 5120 -> 4096 -> 3584

Decoder input:       (1, N_tokens, 3584)   # Audio + text embeddings
Decoder (28 layers):
  Q proj:            (1, N, 28*128=3584)   # 28 Q heads
  K/V proj:          (1, N, 4*128=512)     # 4 KV heads (GQA)
  Reshape Q:         (1, 28, N, 128)
  Reshape K/V:       (1, 4, N, 128)
  Expand KV:         (1, 28, N, 128)       # Repeat 7x for GQA
  Attention:         (1, 28, N, 128)
  MLP (SwiGLU):      (1, N, 3584) -> 18944 -> 3584

LM Head:             (1, N, 151552)        # Vocab logits
Argmax:              (1, N)                # Token IDs
```

---

## Configuration Knobs

### Backend Selection
```python
from layers import Linear, MLP, EncoderMLP

# Linear backend: "torch" (cuBLAS), "triton", or "auto"
Linear.BACKEND = "torch"    # Fastest for large matmuls
Linear.BACKEND = "triton"   # Uses your custom kernel
Linear.BACKEND = "auto"     # Triton if M >= TILE_M, else torch

# Fusion: True enables fused kernels (faster)
MLP.FUSED = True            # Fused SwiGLU
EncoderMLP.FUSED = True     # Fused Linear+GELU

# Tile sizes for Triton matmul
Linear.TILE_M = 64          # Output tile rows
Linear.TILE_N = 64          # Output tile columns
Linear.TILE_K = 32          # Reduction tile
```

### Generation Options
```python
model.generate(
    input_features,                    # (1, 128, T) mel
    input_ids=input_ids,               # Token IDs with audio placeholders
    input_features_mask=mask,          # Audio frame mask
    max_new_tokens=100,                # Max tokens to generate
    temperature=1.0,                   # Sampling temperature
    top_k=1,                           # Top-k (1 = greedy)
)
```

---

## Benchmark Commands

```bash
cd hw1-asr

# Quick correctness test
./benchmark.sh glm_asr_triton_template

# Reference baseline timing
./benchmark.sh glm_asr_triton_example

# Detailed per-operator profiling
./benchmark_detailed.sh glm_asr_triton_template

# Custom options
python benchmark_student.py glm_asr_triton_template --warmup 3 --runs 5

# With specific audio
python benchmark_student.py glm_asr_triton_template --audio test_audio.wav
```

---

## Optimization Checklist

- [ ] All 10 kernels implemented and passing tests
- [ ] Correct transcription output (>80% word accuracy)
- [ ] Tile/block sizes tuned (tried at least 2-3 configs)
- [ ] At least 1 kernel fusion active (SwiGLU or Linear+GELU)
- [ ] Linear backend selection optimized (torch vs triton per layer size)
- [ ] KV cache enabled for decoder generation
- [ ] Activation block size tuned (try 256, 512, 1024)
- [ ] Total inference time < baseline (example implementation)
- [ ] Target: < 200ms total inference

---

## File Dependency Graph

```
model.py
  |-- layers.py (RMSNorm, LayerNorm, Linear, MLP, EncoderMLP, Embedding, softmax, gelu, silu)
  |-- attention.py (MultiHeadAttention, scaled_dot_product_attention)
  |-- rope.py (RotaryEmbedding, apply_rotary_pos_emb)
  |-- conv.py (Conv1d, Conv1dSubsampler)
  |-- weight_loader.py (load_model_from_hf)

benchmark_student.py
  |-- model.py (via dynamic import)
  |-- weight_loader.py (downloads from HuggingFace)
```

---

## HuggingFace Model

- **Model ID:** `zai-org/GLM-ASR-Nano-2512`
- **Size:** ~2GB (safetensors format)
- **Auto-downloaded** by `weight_loader.py` on first run
- **Cache:** `~/.cache/huggingface/`

---

## Test Audio

- **File:** `hw1-asr/test_audio.wav`
- **Expected output:** `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`
- **Sample rate:** 16kHz
- **Duration:** ~5 seconds
