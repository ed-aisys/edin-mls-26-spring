# Reference Guide: GLM-ASR Triton Kernel Project

Quick reference for kernel signatures, model architecture, and performance tuning.

---

## Model Configuration (GLM-ASR-Nano-2512)

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
| | Hidden | 5120 -> 4096 -> 2048 |
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

---

## Files: What You Can and Cannot Modify

| File | Modifiable? | What's In It |
|------|:-----------:|--------------|
| `layers.py` | **Yes** | All 6 layer kernels + config knobs + fused kernels |
| `attention.py` | **Yes** | 3 attention kernels |
| `rope.py` | **Yes** | 1 RoPE kernel |
| `__init__.py` | **Yes** | Backend/fusion configuration |
| `model.py` | **No** | Model architecture, generation loop |
| `weight_loader.py` | **No** | HuggingFace weight loading |
| `conv.py` | **No** | Conv1D for audio subsampling |

---

## Kernel Signatures

### layers.py

```python
# RMSNorm — Grid: (num_rows,)
rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)

# LayerNorm — Grid: (num_rows,)
layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE)

# GELU — Grid: (cdiv(n_elements, BLOCK_SIZE),)
gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)

# SiLU — Grid: (cdiv(n_elements, BLOCK_SIZE),)
silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE)

# Linear — Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
linear_kernel_tf32(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K)

# Softmax — Grid: (num_rows,)
softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE)
```

### attention.py

```python
# Attention Scores — Grid: (batch_heads, seq_q)
attention_scores_kernel(q_ptr, k_ptr, scores_ptr, scale, seq_k, head_dim,
                        stride_q0..q2, stride_k0..k2, stride_s0..s2,
                        BLOCK_K, BLOCK_D)

# Softmax In-place — Grid: (batch_heads * seq_q,)
softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE)

# Attention Output — Grid: (batch_heads, seq_q)
attention_output_kernel(attn_ptr, v_ptr, output_ptr, seq_k, head_dim,
                        stride_w0..w2, stride_v0..v2, stride_o0..o2,
                        BLOCK_K, BLOCK_D)
```

### rope.py

```python
# RoPE Frequencies — Grid: (seq_len,)
compute_freqs_kernel(positions_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                     seq_len, half_dim,
                     stride_pos, stride_inv,
                     stride_cos0, stride_cos1, stride_sin0, stride_sin1,
                     BLOCK)
```

---

## Tensor Shapes Through the Pipeline

```
Input: audio_array (float32, 16kHz, ~3.5s for test audio)

Mel Spectrogram:     (1, 128, T)           # T depends on audio length
Conv1 output:        (1, 1280, T)          # Feature expansion + GELU
Conv2 output:        (1, 1280, T/2)        # Stride 2 + GELU
Permute:             (1, T/2, 1280)        # (batch, seq, hidden)

Encoder (32 layers):
  Q/K/V proj:        (1, T/2, 1280)        # Linear
  Reshape:           (1, 20, T/2, 64)      # 20 heads, head_dim=64
  RoPE:              Partial (first 32 dims rotated)
  Attention:         (1, 20, T/2, 64)      # Scaled dot-product
  MLP:               (1, T/2, 1280) -> 5120 -> 1280  # GELU

Encoder output:      (1, T/2, 1280)
Pool 4 frames:       (1, T/8, 5120)        # Concatenate 4 consecutive frames
Projector:           (1, T/8, 2048)        # 5120 -> 4096 (GELU) -> 2048

Decoder input:       (1, N_tokens, 2048)   # Audio + text token embeddings
Decoder (28 layers):
  Q proj:            (1, N, 2048)          # 16 Q heads x 128 dim
  K/V proj:          (1, N, 512)           # 4 KV heads x 128 dim (GQA)
  Expand KV:         (1, 16, N, 128)       # Repeat 4x for GQA
  Attention:         (1, 16, N, 128)       # Causal masked
  MLP (SwiGLU):      (1, N, 2048) -> 6144 -> 2048

LM Head:             (1, N, 59264)         # Vocab logits
Argmax:              next token ID
```

---

## Configuration Knobs (in __init__.py and layers.py)

### Backend Selection
```python
# In __init__.py:
layers.Linear.BACKEND = "torch"    # cuBLAS — fastest on RTX 5090
layers.Linear.BACKEND = "triton"   # Your custom Triton kernel

# Fusion flags:
layers.MLP.FUSED = True            # Fused SwiGLU (decoder MLP)
layers.EncoderMLP.FUSED = True     # Fused Linear+GELU (has no effect with original model.py)
```

### Tile Sizes (in layers.py)
```python
# Linear layer tiles (used for Triton backend and fused kernels)
Linear.TILE_M = 128    # Output tile rows
Linear.TILE_N = 128    # Output tile columns
Linear.TILE_K = 64     # Reduction tile

# MLP fused kernel tiles
MLP.TILE_M, MLP.TILE_N, MLP.TILE_K = 64, 64, 32
EncoderMLP.TILE_M, EncoderMLP.TILE_N, EncoderMLP.TILE_K = 64, 64, 32
```

---

## Benchmark Commands

```bash
cd hw1-asr

# If model cache is on overlay with limited space:
export HF_HOME=/workspace/.hf_cache

# Quick correctness test
python benchmark_student.py glm_asr_triton_template --warmup 1 --runs 1

# Full benchmark
python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5

# Baseline comparison
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3

# Detailed per-operator profiling
python benchmark_detailed.py glm_asr_triton_template
```

---

## Benchmark Results (RTX 5090, CUDA 13.0)

| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template** | **214ms** | 16.5ms/tok | 100% |
| Example baseline | 261ms | 20.1ms/tok | 100% |
| **Speedup** | **18%** | | |

**Why not <200ms?** The original `model.py` `generate()` has no KV cache —
it re-runs the full sequence through 28 decoder layers per token. Adding KV cache
would require modifying `model.py`, which is not allowed.

---

## Optimization Checklist

- [x] All 10 kernels implemented and passing tests
- [x] Correct transcription output (100% word accuracy)
- [x] Tile/block sizes tuned (tested multiple configs)
- [x] Fused SwiGLU active for decoder MLP
- [x] Linear backend optimized (cuBLAS selected as fastest)
- [x] Activation block size 1024 (up from 256)
- [x] Total inference time < baseline (214ms vs 261ms)
- [ ] Target: <200ms (blocked by read-only model.py — no KV cache possible)

---

## File Dependency Graph

```
model.py (DO NOT MODIFY)
  |-- layers.py (RMSNorm, LayerNorm, Linear, MLP, Embedding, softmax, gelu, silu)
  |-- attention.py (MultiHeadAttention, scaled_dot_product_attention)
  |-- rope.py (RotaryEmbedding, apply_rotary_pos_emb)
  |-- conv.py (Conv1d, Conv1dSubsampler) (DO NOT MODIFY)
  |-- weight_loader.py (load_model_from_hf) (DO NOT MODIFY)

benchmark_student.py
  |-- model.py (via dynamic import)
  |-- weight_loader.py (downloads from HuggingFace)
```

---

## HuggingFace Model

- **Model ID:** `zai-org/GLM-ASR-Nano-2512`
- **Size:** ~4.3GB (safetensors format)
- **Auto-downloaded** by `weight_loader.py` on first run
- **Cache:** `$HF_HOME` or `~/.cache/huggingface/`

## Test Audio

- **File:** `hw1-asr/test_audio.wav`
- **Expected output:** `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`
- **Sample rate:** 16kHz
- **Duration:** ~3.5 seconds

## Troubleshooting: cuBLAS

If you see `CUBLAS_STATUS_INVALID_VALUE` errors, you likely have a pip-installed
`nvidia-cublas` package that conflicts with the system CUDA libraries:
```bash
pip list | grep nvidia-cublas
# If version doesn't match your CUDA toolkit:
pip uninstall nvidia-cublas
# PyTorch will then use the system cuBLAS library
```
