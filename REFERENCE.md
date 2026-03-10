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
| | Uses | LinearGELU + Linear |
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
| `layers.py` | **Yes** | All 6 layer kernels + config knobs + fused kernels + layer classes |
| `attention.py` | **Yes** | Fused Flash Attention kernel + 3 legacy attention kernels |
| `rope.py` | **Yes** | 1 RoPE kernel |
| `__init__.py` | **Yes** | Backend/fusion configuration |
| `model.py` | **No** | Model architecture, KV-cached generation (`generate_v8b`) |
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
# Flash Attention (PRIMARY) — Grid: (cdiv(seq_q, BLOCK_M), batch_heads)
# Fused online softmax: single kernel, no DRAM scores matrix
flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, scale,
                       seq_q, seq_k, head_dim,
                       stride_qb..qd, stride_kb..kd, stride_vb..vd,
                       stride_ob..od, stride_mb..mk,
                       IS_CAUSAL, HAS_MASK, BLOCK_M, BLOCK_N, BLOCK_D)

# Legacy: Attention Scores — Grid: (batch_heads, seq_q)
attention_scores_kernel(q_ptr, k_ptr, scores_ptr, scale, seq_k, head_dim,
                        stride_q0..q2, stride_k0..k2, stride_s0..s2,
                        BLOCK_K, BLOCK_D)

# Legacy: Softmax In-place — Grid: (batch_heads * seq_q,)
softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE)

# Legacy: Attention Output — Grid: (batch_heads, seq_q)
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
  MLP (EncoderMLP):  (1, T/2, 1280) -> 5120 -> 1280  # GELU (fused when FUSED=True)

Encoder output:      (1, T/2, 1280)
Pool 4 frames:       (1, T/8, 5120)        # Concatenate 4 consecutive frames
Projector:           (1, T/8, 2048)        # LinearGELU(5120->4096) + Linear(4096->2048)

Decoder input:       (1, N_tokens, 2048)   # Audio + text token embeddings
Decoder (28 layers):
  Q proj:            (1, N, 2048)          # 16 Q heads x 128 dim
  K/V proj:          (1, N, 512)           # 4 KV heads x 128 dim (GQA)
  Reshape Q:         (1, 16, N, 128)
  Reshape KV:        (1, 4, N, 128)
  Attention:         (1, 16, N, 128)       # Flash Attention kernel (GQA via _expand_kv_heads)
  MLP (SwiGLU):      (1, N, 2048) -> 6144 -> 2048

LM Head:             (1, N, 59264)         # Vocab logits
Argmax:              next token ID

KV-Cache Decode (generate_v8b in model.py):
  Prefill:           Full sequence processed once, KV states cached
  Each decode step:  (1, 1, 2048) input -> only new token processed
  KV buffers:        Pre-allocated (batch, num_layers, heads, max_seq, head_dim)
```

---

## Configuration Knobs (in __init__.py and layers.py)

### Backend Selection
```python
# In __init__.py:
layers.Linear.BACKEND = "torch"    # current config; F.linear -> cuBLAS/cuBLASLt
layers.Linear.BACKEND = "triton"   # strict linear-kernel path

# Fusion flags:
layers.MLP.FUSED = True            # Fused SwiGLU (decoder MLP)
layers.EncoderMLP.FUSED = True     # Fused Linear+GELU (encoder MLP — used by model.py)
# LinearGELU.FUSED = False         # Disabled in layers.py (shared memory exceeds hardware limit)
```

### bfloat16 Weights
```python
# In layers.py (class-level default — can't rely on __init__.py during benchmarks):
Linear.BF16 = True     # Caches bf16 weight copies, halves memory traffic
Linear.BF16 = False    # Standard float32 path
```

### Runtime Flags
```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

These are the current committed defaults in `__init__.py`.

### Flash Attention Configuration
```python
# In attention.py:
# Primary CUDA path: fused Triton Flash Attention kernel with online softmax
# GQA handled via _expand_kv_heads before kernel call
# Tile sizes chosen per head_dim to stay within 101KB shared memory
if head_dim <= 64:   BLOCK_M, BLOCK_N = 128, 64  # Encoder
else:                BLOCK_M, BLOCK_N = 64, 32    # Decoder
flash_attention_kernel[grid](
    q_flat, k_flat, v_flat, output, mask_flat, scale, ...,
    IS_CAUSAL=is_causal, HAS_MASK=has_mask,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=head_dim_padded,
    num_stages=1,
)
```

### Attention Self-Test
```bash
cd hw1-asr/glm_asr_triton_template
python attention.py
```

The current self-test is a deterministic 17-case parity suite. It prints the
active device, warns if it is only exercising the CPU fallback path, and checks
encoder-like ragged lengths, decoder-like prefill lengths, both mask layouts,
GQA, single-token decode, decode with causal+mask, and non-power-of-two shapes.

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

## KV-Cache Generation (in model.py — read-only)

`model.py` natively includes `generate_v8b()` which uses pre-allocated KV buffers:

```python
# model.py — GlmAsrModel.generate_v8b()
inputs_embeds, seed_tokens = self._prepare_generation_inputs(...)
kv_buffers = self.text_decoder.allocate_kv_buffers(batch_size, max_len)
hidden, cache_pos = self.text_decoder.forward_with_kv_buffers(inputs_embeds, kv_buffers, 0)
logits = self.lm_head(hidden[:, -1:, :])
for _ in range(max_new_tokens):
    next_token = self._sample_next_token(logits[:, -1, :] / temperature, ...)
    next_embeds = self.text_decoder.embed_tokens(next_token)
    hidden, cache_pos = self.text_decoder.forward_with_kv_buffers(next_embeds, kv_buffers, cache_pos)
    logits = self.lm_head(hidden[:, -1:, :])
```

`generate()` delegates to `generate_v8b()` by default. `benchmark_student.py`
also checks `hasattr(model, 'generate_v8b')` and uses it when available.

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
| **Our template** | **110.0ms** | 8.46ms/tok | 100% |
| Example baseline | 261.3ms | 20.10ms/tok | 100% |
| **Speedup** | **57.9%** | | |

---

## Optimization Checklist

- [x] All 10 kernels implemented and passing tests
- [x] Correct transcription output (100% word accuracy)
- [x] Tile/block sizes tuned (tested multiple configs)
- [x] Fused SwiGLU active for decoder MLP
- [x] Fused EncoderMLP active for encoder layers
- [x] LinearGELU fusion disabled (shared memory limit)
- [x] Linear backend optimized (cuBLAS selected as fastest)
- [x] TF32 runtime flags enabled in `__init__.py`
- [x] GQA path optimized — `_expand_kv_heads` before Flash Attention kernel
- [x] Activation block size 1024 (up from 256)
- [x] bfloat16 weights — halves memory traffic for decode matmuls
- [x] Fused Flash Attention — Triton kernel with online softmax, replaces SDPA and 3-kernel approach
- [x] Flash Attention supports causal, attention_mask, and arbitrary seq lengths
- [x] 17 deterministic numerical parity tests for Flash Attention, including ragged encoder/decode shapes and both mask layouts
- [x] KV-cache generation — O(n) decode via native `generate_v8b` in model.py
- [x] Total inference time < 200ms target (110.0ms achieved)
- [x] model.py, conv.py, weight_loader.py all match origin/ankush (zero diff)

---

## File Dependency Graph

```
model.py (DO NOT MODIFY — includes generate_v8b with KV cache)
  |-- layers.py (RMSNorm, LayerNorm, Linear, MLP, EncoderMLP, LinearGELU, Embedding, softmax, gelu, silu)
  |-- attention.py (MultiHeadAttention, scaled_dot_product_attention)
  |-- rope.py (RotaryEmbedding, apply_rotary_pos_emb)
  |-- conv.py (Conv1d, Conv1dSubsampler) (DO NOT MODIFY)
  |-- weight_loader.py (load_model_from_hf) (DO NOT MODIFY)

benchmark_student.py
  |-- model.py (via dynamic import)
  |-- weight_loader.py (downloads from HuggingFace)
  |-- checks for generate_v8b/v8/v6 on model instance
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
