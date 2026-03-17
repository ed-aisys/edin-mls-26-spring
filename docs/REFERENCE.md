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
| | MLP | Plain `fc1 → gelu → fc2` (not EncoderMLP class) |
| **Projector** | Pool factor | 4 |
| | Hidden | 5120 -> 4096 -> 2048 |
| | Uses | Plain `Linear → gelu → Linear` (not LinearGELU class) |
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
| `attention.py` | **Yes** | Fused Flash Attention kernel + SDPA fallback (legacy kernels removed) |
| `rope.py` | **Yes** | 1 RoPE kernel |
| `__init__.py` | **Yes** | Backend/fusion configuration |
| `model.py` | **No** | Model architecture, stock `generate()` (O(n²), KV cache infra exists but unused) |
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
flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, scale,
                       seq_q, seq_k, head_dim,
                       stride_qb..qd, stride_kb..kd, stride_vb..vd,
                       stride_ob..od, stride_mb..mk,
                       IS_CAUSAL, HAS_MASK, BLOCK_M, BLOCK_N, BLOCK_D)

# SDPA fallback for KV-cached decode (seq_q <= 4):
# torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)

# Legacy kernels (attention_scores, softmax_inplace, attention_output,
# causal_mask) were REMOVED — superseded by flash_attention_kernel.
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
  Q/K/V proj:        (1, T/2, 1280)        # Linear (cuBLAS fp16 HGEMM)
  Reshape:           (1, 20, T/2, 64)      # 20 heads, head_dim=64
  RoPE:              Partial (first 32 dims rotated)
  Attention:         (1, 20, T/2, 64)      # Flash Attention kernel
  MLP:               fc1(x) → gelu(x) → fc2(x)  # Plain Linear + gelu, NOT fused

Encoder output:      (1, T/2, 1280)
Pool 4 frames:       (1, T/8, 5120)        # Concatenate 4 consecutive frames
Projector:           (1, T/8, 2048)        # Linear→gelu→Linear (plain, NOT fused)

Decoder input:       (1, N_tokens, 2048)   # Audio + text token embeddings
Decoder (28 layers):
  Q proj:            (1, N, 2048)          # 16 Q heads x 128 dim
  K/V proj:          (1, N, 512)           # 4 KV heads x 128 dim (GQA)
  Reshape Q:         (1, 16, N, 128)
  Reshape KV:        (1, 4, N, 128)
  Attention:         (1, 16, N, 128)       # Flash Attention (GQA via _expand_kv_heads)
  MLP (SwiGLU):      Fused when MLP.FUSED=True

LM Head:             (1, N, 59264)         # Vocab logits

Stock generate() — O(n²) decode:
  Each step: embed new token, concatenate to inputs_embeds, reprocess ALL through decoder
  No KV cache — full sequence recomputed each step
```

---

## Configuration Knobs (in __init__.py and layers.py)

### Backend Selection
```python
layers.Linear.BACKEND = "torch"    # cuBLAS/cuBLASLt (current, fastest)
layers.Linear.BACKEND = "triton"   # strict linear-kernel path

layers.MLP.FUSED = True            # Fused SwiGLU (decoder MLP) — EFFECTIVE
layers.EncoderMLP.FUSED = True     # Set but NOT USED (model.py uses plain fc1/fc2)
# LinearGELU.FUSED = False         # Set but NOT USED (model.py uses plain linear_1/act)
```

### fp16 Weights (flag name retained as BF16 for compatibility)
```python
Linear.BF16 = True                     # Class default in layers.py, enables half-precision
Linear._HALF_DTYPE = torch.float16     # Actual dtype: fp16 (faster HGEMM on RTX 5090)
```
Output stays fp16 (no `.float()` conversion), keeping the entire pipeline in fp16.

### GPU Detection: GPUProfile (layers.py)
```python
# GPUProfile detects GPU architecture at import time
GPU = GPUProfile()  # Replaces old _detect_gpu_tier()

# Reads: sm_version, gpu_name, shared memory via getattr fallback chain:
#   shared_memory_per_block_optin → max_shared_memory_per_block → shared_memory_per_block
# Classifies: blackwell_consumer, ada, hopper, blackwell_dc, ampere_dc, ampere_consumer, older

# _KNOWN_CONFIGS table stores tested tile sizes for 6 GPU architectures
# For unknown GPUs: _compute_attention_tiles() and _compute_matmul_tiles()
# compute tiles dynamically from shared memory budget
```

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
# torch.nn.functional.scaled_dot_product_attention — avoids Triton launch overhead

```

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

### H200 MIG 3g.71gb (Teaching Cluster, 60 SMs, 2026-03-16)
| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template (fp16 pipeline + KV cache + SDPA)** | **204.6ms** | 15.74ms/tok | 100% |

### Detailed Benchmark (50 generated tokens)
| Component | Time | % Total |
|-----------|------|---------|
| Audio Encoder | 202.09ms | 8.7% |
| Projector | 4.14ms | 0.2% |
| Decoder Prefill | 191.59ms | 8.3% |
| **Decoder Decode (50 steps)** | **1919.94ms** | **82.8%** |
| **Total** | **2317.76ms** | 100% |

**Key bottleneck:** Decoder decode dominates because stock `generate()` is O(n²).

---

## Optimization Roadmap

| Optimization | Source | Impact | Status |
|-------------|--------|--------|--------|
| Fused Q+K RoPE kernel | meave | **-14ms** | **ADOPTED** |
| bf16 RMSNorm output | meave (adapted) | **-3ms** | **ADOPTED** |
| bf16 LayerNorm output | internal | **-0.7ms** | **ADOPTED** |
| generate_v8b (KV cache) | internal | **-7.6ms** | **ADOPTED** |
| SDPA fallback for seq_q≤4 | internal | **-3ms** | **ADOPTED** |
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

---

## Optimization Checklist

- [x] All 10 kernels implemented and passing tests
- [x] Correct transcription output (100% word accuracy)
- [x] Fused SwiGLU active for decoder MLP (`MLP.FUSED = True`)
- [x] Linear backend optimized (cuBLAS selected as fastest)
- [x] TF32 runtime flags enabled
- [x] bfloat16 weights — halves memory traffic
- [x] Fused Flash Attention — Triton kernel with online softmax
- [x] 17 deterministic numerical parity tests for Flash Attention
- [x] model.py, conv.py, weight_loader.py all match origin/main (zero diff)
- [x] Upstream merge with ed-aisys (19 commits, grading criteria, benchmark updates)
- [x] Fused Q+K RoPE pair kernel (from meave) — **-14ms**
- [x] bf16 RMSNorm output kernel (from meave) — **-3ms**
- [x] bf16 LayerNorm output — **-0.7ms**
- [x] generate_v8b with KV cache (monkey-patched, decode(use_cache=True)) — **-7.6ms**
- [x] SDPA fallback for KV-cached decode (seq_q≤4) — **-3ms**
- [x] GPUProfile with _KNOWN_CONFIGS + dynamic tile computation for cross-GPU portability
- [x] Dead code cleanup — removed ~320 lines of legacy attention kernels
- [x] SwiGLU swizzle tested, rejected (+18ms regression on RTX 5090)
- [x] @triton.autotune tested, rejected (lightweight: +0.7ms overhead; heavy kernels: massive regression)
- [x] Softmax bf16, num_stages=2, num_warps=8 — tested, no improvement on consumer GPUs
- [x] fp16 cuBLAS HGEMM (`Linear._HALF_DTYPE = torch.float16`) — slightly faster than bf16
- [x] Smaller flash attention tiles (from meave) — 64x64 encoder, 32x32 decoder
- [x] Remove Linear `.float()` conversion — fp16 output cascades through pipeline (**-7.5ms**)
- [x] Remove silu/gelu Python-side float32 cast — kernels handle internally (**-3.7ms**)
- [x] Remove RMSNorm/LayerNorm Python-side float32 cast — kernels handle internally (~-0.5ms)
- [x] fp16 embedding output — keeps decoder pipeline in fp16 from start
- [x] fp16 fused SwiGLU/EncoderMLP — halves intermediate memory bandwidth
- [x] Remove flash attention Python-side float32 conversion (~-1ms)
- [x] Norm kernel output dtype: fp16 (was bf16)
- [x] PyTorch SDPA for prefill/encoder — tested, +6ms regression. Rejected
- [x] SDPA enable_gqa=True — tested, +13ms regression. Rejected
- [x] Fused gate+up Linear in MLP — tested, neutral. Rejected

---

## File Dependency Graph

```
model.py (DO NOT MODIFY — stock generate(), no KV cache)
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

---

## HuggingFace Model

- **Model ID:** `zai-org/GLM-ASR-Nano-2512`
- **Size:** ~4.3GB (safetensors format)
- **Cache:** `$HF_HOME` or `~/.cache/huggingface/`

## Test Audio

- **File:** `hw1-asr/test_audio.wav`
- **Expected output:** `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`
- **Duration:** ~3.5 seconds

## Troubleshooting

### cuBLAS Version Mismatch
If you see `CUBLAS_STATUS_INVALID_VALUE`, pip-installed `nvidia-cublas` may conflict:
```bash
pip uninstall nvidia-cublas
```

### numpy Version Mismatch (cu12)
If you see `TypeError: expected np.ndarray (got ndarray)`, use `torch.as_tensor()` instead
of `torch.from_numpy()`. The `_to_torch_tensor()` helper in layers.py handles this automatically.

### Teaching Cluster OOM
If SLURM kills your job during weight loading, request more RAM:
```bash
srun -p Teaching -w saxa --gres gpu:3g.71gb:1 --mem=32G --pty bash
```
