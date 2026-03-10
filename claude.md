# Claude Development Log

## Project: GLM-ASR Triton GPU Kernel Implementation
**Date:** 2026-03-09 to 2026-03-10
**Branch:** `dev/complete-and-optimize` -> `ankush`
**GPU:** NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32GB VRAM)
**CUDA Toolkit:** 13.0 | **Driver:** 580.126.20
**PyTorch:** 2.10.0+cu130 | **Triton:** 3.6.0

---

## Summary

Completed all 10 Triton kernel implementations for the GLM-ASR speech-to-text model.
The project is a University of Edinburgh MLS course assignment implementing GPU kernels
for a multi-modal transformer (audio encoder + text decoder).

**Final benchmark result: 209.8ms average, 100% transcription accuracy, 19.7% faster than baseline.**

---

## Important Constraints (from GUIDE.md)

**Do NOT modify these files:**
- `model.py` — model architecture and generation loop
- `weight_loader.py` — loads pre-trained weights from HuggingFace
- `conv.py` — 1D convolution for audio subsampling

**Files we CAN modify:**
- `layers.py` — kernel implementations + layer classes + config knobs
- `attention.py` — attention kernels
- `rope.py` — RoPE kernel
- `__init__.py` — backend/fusion configuration

---

## Step-by-Step Development Log

### Step 1: Environment Assessment (Session 1, 2026-03-09)
- Initial container had CUDA 13.1 toolkit but driver 580.126.09
- CUDA runtime error 804 (forward compatibility not supported on consumer GPUs)
- Installed PyTorch cu128 as workaround, then later restored cu130
- Code validated on CPU fallback (13.8s, 100% accuracy)

### Step 2: Codebase Analysis
- `hw1-asr/glm_asr_triton_template/` — Student template (10 TODO kernels)
- `hw1-asr/glm_asr_triton_example/` — Reference implementation (complete)
- Model: GLM-ASR-Nano-2512 (32-layer audio encoder + 28-layer text decoder)

### Step 3: Kernel Implementations (all in allowed files)

#### 3.1 `layers.py` — 6 kernels

**rmsnorm_kernel**: `y = x / sqrt(mean(x^2) + eps) * weight`
- Load row, compute sum of squares, normalize, apply weight
- Grid: (num_rows,), one block per row

**layernorm_kernel**: `y = (x - mean) / sqrt(var + eps) * weight + bias`
- Two-pass: compute mean, center, compute variance, normalize, affine transform
- Grid: (num_rows,), one block per row

**gelu_kernel**: `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
- Element-wise activation with tanh approximation via `tl.extra.cuda.libdevice.tanh`
- Grid: (ceil(n_elements / BLOCK_SIZE),), BLOCK_SIZE=1024

**silu_kernel**: `y = x * sigmoid(x) = x / (1 + exp(-x))`
- Element-wise activation
- Grid: (ceil(n_elements / BLOCK_SIZE),), BLOCK_SIZE=1024

**linear_kernel_tf32**: `C = A @ B` (tiled matmul)
- 2D tiling: BLOCK_M x BLOCK_N output tiles, accumulate over K in BLOCK_K chunks
- `tl.dot(a, b)` for tensor core acceleration
- Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))

**softmax_kernel**: `y = exp(x - max(x)) / sum(exp(x - max(x)))`
- Numerically stable: subtract max before exp to prevent overflow
- Grid: (num_rows,), one block per row

#### 3.2 `attention.py` — 3 kernels

**attention_scores_kernel**: `scores = sum(K * Q[broadcast], dim=-1) * scale`
- Loads Q vector and K matrix, broadcast-multiply and reduce
- Grid: (batch_heads, seq_q)

**softmax_inplace_kernel**: In-place numerically stable softmax
- Same as softmax_kernel but writes back to input buffer
- Grid: (batch_heads * seq_q,)

**attention_output_kernel**: `output = sum(V * weights[:, None], dim=0)`
- Weighted sum of value vectors by attention weights
- Grid: (batch_heads, seq_q)

#### 3.3 `rope.py` — 1 kernel

**compute_freqs_kernel**: `cos/sin(position * inv_freq)` precomputation
- Load position scalar and inverse frequency vector
- Store concatenated cos/sin (duplicated into both halves)
- Grid: (seq_len,)

### Step 4: Performance Optimizations (in allowed files only)

#### 4.1 Linear Backend Selection
```python
# __init__.py and layers.py
Linear.BACKEND = "torch"  # cuBLAS — fastest for Blackwell RTX 5090
```
The committed code keeps the cuBLAS path enabled. `Linear._forward_torch()`
now uses `torch.nn.functional.linear(...)`, which lets PyTorch pick the best
cuBLAS/cuBLASLt implementation for the current shape and bias configuration.

#### 4.2 Runtime Flags
```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```
These settings expose TF32 tensor-core paths for float32 math and let cuDNN
cache the best kernel choice for the observed shapes.

#### 4.3 Linear Tile Sizes
```python
Linear.TILE_M = 128  # Larger tiles for better GPU occupancy
Linear.TILE_N = 128
Linear.TILE_K = 64
```
These are used when BACKEND="triton" and for the fused kernels.

#### 4.4 Kernel Fusion (pre-implemented, enabled via config)
```python
MLP.FUSED = True         # Fused SwiGLU: SiLU(x @ gate) * (x @ up) in one kernel
EncoderMLP.FUSED = True  # Fused Linear+GELU in one kernel
```
Note: `EncoderMLP.FUSED` is set but the original `model.py` doesn't use `EncoderMLP`
(it uses separate `fc1`/`fc2` + `gelu()` calls). This setting is harmless but has no
effect with the unmodified model.py.

#### 4.5 SDPA GQA Fast Path
```python
if use_gqa:
    k = _expand_kv_heads(k, num_heads)
    v = _expand_kv_heads(v, num_heads)
    use_gqa = False
```
`attention.py` now expands KV heads explicitly before
`torch.nn.functional.scaled_dot_product_attention(...)` instead of relying on
`enable_gqa=True`. On this CUDA/PyTorch stack, that reduced decode latency.

#### 4.6 Activation Block Sizes
- GELU/SiLU block size: 1024 (up from default 256) for better GPU occupancy

### Step 5: Environment Fixes (Session 2, 2026-03-10)

#### 5.1 New Pod with CUDA 13.0
- Driver 580.126.20 + CUDA toolkit 13.0 = matching versions
- `torch.cuda.is_available()` returns True
- RTX 5090 detected and working

#### 5.2 cuBLAS Version Mismatch Fix
- pip-installed `nvidia-cublas 13.1.0.3` conflicted with system cuBLAS 13.0
- cuBLAS loaded 13.1 from pip but cuBLASLt loaded 13.0 from system
- **Fix:** `pip uninstall nvidia-cublas` — torch falls back to system cuBLAS 13.0
- `torch.matmul` works correctly after this fix

#### 5.3 Restricted Files Reverted
- Discovered GUIDE.md rule: model.py, weight_loader.py, conv.py must NOT be modified
- Previous session had modified model.py (KV cache, generate_v8b, EncoderMLP, shared RoPE)
  and conv.py (tiled kernel for broken cuBLAS workaround)
- Reverted both to their original versions from the base commit (4da607d)

---

## Benchmark Results

### Our Implementation (`glm_asr_triton_template`)
| Metric | Value |
|--------|-------|
| **Average time** | **209.8ms** (+/- 1.1ms) |
| **Tokens** | 13 |
| **Speed** | 16.14 ms/token |
| **Accuracy** | 100.0% |
| **Transcription** | "Concord returned to its place amidst the tents." |

### Example Baseline (`glm_asr_triton_example`)
| Metric | Value |
|--------|-------|
| **Average time** | 261.3ms (+/- 0.5ms) |
| **Tokens** | 13 |
| **Speed** | 20.10 ms/token |
| **Accuracy** | 100.0% |

### Comparison
- **19.7% faster** than the example baseline (209.8ms vs 261.3ms)
- Best clean single run observed after the final optimization pass: **205.4ms**
- Detailed profiler after the SDPA GQA change showed decode-step average
  reduced to **7.26ms** from the earlier **9.91ms** path
- Sub-200ms would require KV cache in model.py (which we cannot modify)
- The original model.py `generate()` re-runs the full sequence through all 28 decoder
  layers for each new token — no KV cache, O(n^2) decode cost

---

## Architecture Overview (GLM-ASR-Nano-2512)

```
Audio (WAV 16kHz)
  -> Mel Spectrogram (128 bins)
  -> Conv1D Subsampler (4x downsample)
  -> Audio Encoder (32 layers, hidden=1280, 20 heads, LayerNorm + GELU, 50% RoPE)
  -> Projector (pool 4 frames, 5120 -> 4096 -> 2048, GELU)
  -> Text Decoder (28 layers, hidden=2048, 16 Q-heads / 4 KV-heads, RMSNorm + SiLU/SwiGLU, 100% RoPE)
  -> LM Head (2048 -> 59264 vocab)
  -> Text Output
```

---

## Key Files

| File | Purpose | Modifiable? |
|------|---------|:-----------:|
| `glm_asr_triton_template/layers.py` | Layer kernels (6 implemented) + config | Yes |
| `glm_asr_triton_template/attention.py` | Attention kernels (3 implemented) | Yes |
| `glm_asr_triton_template/rope.py` | RoPE kernel (1 implemented) | Yes |
| `glm_asr_triton_template/__init__.py` | Backend/fusion configuration | Yes |
| `glm_asr_triton_template/model.py` | Model architecture + generation loop | **No** |
| `glm_asr_triton_template/conv.py` | Conv1D layers | **No** |
| `glm_asr_triton_template/weight_loader.py` | HuggingFace weight loading | **No** |
| `benchmark_student.py` | End-to-end benchmark | N/A |

---

## Running the Benchmark

```bash
cd hw1-asr

# IMPORTANT: Set HF_HOME if overlay disk space is limited (<5GB free)
export HF_HOME=/workspace/.hf_cache

# Test your implementation
python benchmark_student.py glm_asr_triton_template --warmup 1 --runs 3

# Compare against baseline
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3

# Detailed per-operator profiling
python benchmark_detailed.py glm_asr_triton_template
```

---

## Disk Space Notes

The overlay filesystem only has ~10GB total. The model is 4.3GB.
**Solution:** Set `HF_HOME=/workspace/.hf_cache` to use the workspace mount (2+ PB).

If cuBLAS fails with `CUBLAS_STATUS_INVALID_VALUE`, check for pip-installed NVIDIA
packages that conflict with the system CUDA libraries:
```bash
pip list | grep nvidia-cublas  # Should match system CUDA version
pip uninstall nvidia-cublas    # Remove if version mismatches
```

---

## Commits

1. `12daf13` — feat: implement all 10 Triton GPU kernels for ASR model
2. `5e8b191` — docs: add full documentation and optimize kernel tile sizes
3. `01fc806` — docs: update claude.md with benchmark results and correct model config
4. `714cdc9` — fix: revert model.py and conv.py to originals (do-not-modify files)
5. `bdc7690` — perf: switch to cuBLAS backend and tune tile sizes
6. `a14e2d5` — Codex commit: optimize Triton template runtime path (209.8ms avg, TF32 runtime flags, explicit GQA expansion before SDPA)
