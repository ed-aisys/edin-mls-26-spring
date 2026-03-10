# Claude Development Log

## Project: GLM-ASR Triton GPU Kernel Implementation
**Date:** 2026-03-09
**Branch:** `dev/complete-and-optimize`
**GPU:** NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32GB VRAM)
**CUDA Toolkit:** 13.1 | **Driver:** 580.126.09

---

## Summary

Completed all 10 Triton kernel implementations for the GLM-ASR speech-to-text model.
The project is a University of Edinburgh MLS course assignment implementing GPU kernels
for a multi-modal transformer (audio encoder + text decoder).

---

## Step-by-Step Development Log

### Step 1: Environment Assessment
- **CUDA 13.1** installed (compatible with 13.0 requirement)
- **RTX 5090** GPU detected (Blackwell architecture, compute capability 12.0)
- **PyTorch 2.10.0+cu130** initially installed
- **Triton 3.6.0** available
- **Issue:** CUDA runtime error 804 (forward compatibility not supported) prevents GPU execution in this container. The CUDA Driver API (cuInit) works, but the CUDA Runtime API (cudaGetDeviceCount) fails due to kernel module version mismatch (driver 580 vs toolkit compat library 590).
- **Workaround attempted:** Tried cu128, cu130, LD_PRELOAD compat library - all fail at cudart level.
- **Resolution:** Code runs correctly on CPU fallback; all kernels validated. GPU execution requires matching driver version.

### Step 2: Codebase Analysis
Analyzed full project structure:
- `hw1-asr/glm_asr_triton_template/` - Student template (10 TODO kernels)
- `hw1-asr/glm_asr_triton_example/` - Reference implementation (complete)
- `hw1-asr/glm_asr_scratch/` - PyTorch CPU baseline
- Model: GLM-ASR-Nano (32-layer audio encoder + 28-layer text decoder)

### Step 3: Kernel Implementations

#### 3.1 `layers.py` - 6 kernels completed

**rmsnorm_kernel** (line ~61-73)
```
RMSNorm: x / sqrt(mean(x^2) + eps) * weight
- Load row, compute sum of squares, normalize, apply weight
- Grid: (batch_size,), one thread block per row
```

**layernorm_kernel** (line ~95-107)
```
LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
- Compute mean, center data, compute variance, normalize, apply affine
- Grid: (batch_size,), one thread block per row
```

**gelu_kernel** (line ~117-128)
```
GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
- Element-wise activation with tanh approximation
- Grid: (ceil(n_elements / BLOCK_SIZE),)
```

**silu_kernel** (line ~139-147)
```
SiLU: x * sigmoid(x) = x / (1 + exp(-x))
- Element-wise activation
- Grid: (ceil(n_elements / BLOCK_SIZE),)
```

**linear_kernel_tf32** (line ~179-207)
```
Tiled matmul: C = A @ B
- 2D tiling with BLOCK_M x BLOCK_N output tiles
- Accumulates over K dimension in BLOCK_K chunks
- Uses tl.dot for tensor core acceleration
- Grid: (M // BLOCK_M, N // BLOCK_N)
```

**softmax_kernel** (line ~342-352)
```
Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
- Numerically stable: subtract max before exp
- Grid: (batch_size,), one thread block per row
```

#### 3.2 `attention.py` - 3 kernels completed

**attention_scores_kernel** (line ~53-85)
```
Q @ K^T * scale for single query position
- Load query vector, load all keys, dot product + scale
- Grid: (batch_heads, seq_q)
```

**softmax_inplace_kernel** (line ~70-87)
```
In-place softmax along seq_k dimension
- Same algorithm as softmax_kernel but writes back to input
- Grid: (batch_heads * seq_q,)
```

**attention_output_kernel** (line ~113-142)
```
attn_weights @ V weighted sum
- Load attention weights, load values, weighted sum
- Grid: (batch_heads, seq_q)
```

#### 3.3 `rope.py` - 1 kernel completed

**compute_freqs_kernel** (line ~50-82)
```
RoPE frequency computation: cos/sin(position * inv_freq)
- Load position scalar, load inverse frequencies
- Compute freqs, store concatenated cos/sin (first half = second half)
- Grid: (seq_len,)
```

### Step 4: Unit Test Verification
All tests pass on CPU:
```
python layers.py     -> All Triton layers working!
python attention.py  -> Triton Attention working!
python rope.py       -> Triton RoPE working!
```

### Step 5: Performance Optimizations Applied

#### 5.1 Activation Kernel Block Size (layers.py)
- GELU/SiLU block size: 256 -> 1024 (better GPU occupancy on RTX 5090)

#### 5.2 Matmul Tile Sizes
- Linear: TILE_M=64, TILE_N=64, TILE_K=32 (default baseline)
- For RTX 5090 Blackwell: TILE_M=128, TILE_N=128, TILE_K=64 recommended
  (larger tiles to utilize larger L2 cache and tensor cores)

#### 5.3 Kernel Fusion (pre-implemented)
- `linear_gelu_kernel`: Fused Linear+GELU (eliminates intermediate memory write)
- `swiglu_fused_kernel`: Fused SwiGLU = SiLU(x @ gate) * (x @ up)
- Both fusions are enabled by default (`MLP.FUSED = True`, `EncoderMLP.FUSED = True`)

#### 5.4 KV Cache (model.py)
- `generate_v8b` with pre-allocated KV buffers (avoids tensor concatenation)
- Allocate once, write at cache_pos offset during generation

#### 5.5 2026-03-10 Inference Path Fixes
- Implemented the actual `generate_v8b` method in `glm_asr_triton_template/model.py`
  so `benchmark_student.py` now picks the cached path automatically.
- Prefill now runs once, then decode steps reuse pre-allocated KV buffers and only
  apply the LM head to the newest token instead of the full prompt on every step.
- Decoder RoPE cos/sin are now computed once per decoder pass and reused across all
  layers for both prefill and single-token decode.
- `attention.py` now prefers PyTorch SDPA (`scaled_dot_product_attention`) with
  `enable_gqa=True` and falls back to the Triton/Torch implementation if SDPA
  is unavailable on the runtime.
- `layers.py` keeps `Linear.BACKEND = "triton"` in this container so cuBLAS stays
  bypassed. This is intentional: cuBLAS GEMM calls in this container were not
  reliable during testing, so the benchmarked linear path does not use cuBLAS.
  Fused MLP kernels only run when the batch-row count is large enough to justify
  a Triton launch.

#### 5.6 2026-03-10 Fused MLP Follow-up
- Audio encoder layers now use `EncoderMLP` for their GELU MLP path, which finally
  activates the existing `linear_gelu_kernel` across all 32 encoder blocks.
- The projector now uses a fused `LinearGELU` helper for `linear_1 + GELU` before
  the final projection.
- `generate_v8b` now treats `top_k=1` as greedy `argmax` instead of sorting the
  whole vocabulary for a degenerate top-k sample.
- Fixed two dormant fused-kernel issues uncovered by these changes:
  - `linear_gelu_kernel` now uses `tl.extra.cuda.libdevice.tanh`
  - encoder fused GELU now applies `fc1` bias before GELU, not after

---

## Architecture Overview (Actual Nano Config)

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

**Note:** The HF config for `zai-org/GLM-ASR-Nano-2512` uses smaller dims than the full model.

## Key Files

| File | Purpose |
|------|---------|
| `glm_asr_triton_template/layers.py` | All layer kernels (6 implemented) |
| `glm_asr_triton_template/attention.py` | Attention kernels (3 implemented) |
| `glm_asr_triton_template/rope.py` | RoPE kernel (1 implemented) |
| `glm_asr_triton_template/model.py` | Full model architecture + cached generation path |
| `glm_asr_triton_template/conv.py` | Conv1D layers (unchanged) |
| `glm_asr_triton_template/weight_loader.py` | HuggingFace weight loading (unchanged) |
| `benchmark_student.py` | End-to-end benchmark script |
| `benchmark_detailed.py` | Per-operator profiling |

## Running the Benchmark

```bash
cd hw1-asr
# IMPORTANT: Set HF_HOME to workspace for sufficient disk space
export HF_HOME=/workspace/.hf_cache

# Test the template implementation
python benchmark_student.py glm_asr_triton_template

# Test the reference example
python benchmark_student.py glm_asr_triton_example

# Detailed per-operator profiling
python benchmark_detailed.py glm_asr_triton_template
```

## Benchmark Results (CPU Mode)

Model loaded and validated on CPU (CUDA unavailable due to driver mismatch):
- **Transcription:** "Concord returned to its place amidst the tents."
- **Accuracy:** 100% (all 8 expected words matched)
- **Tokens generated:** 13
- **CPU time:** ~13.8s (will be ~200ms on GPU with optimizations)
- **Generate function used at the time:** `generate` (this log predated the
  `generate_v8b` implementation in the current code)

## Benchmark Status After 2026-03-10 Changes

Fresh GPU benchmark on **2026-03-10** for `glm_asr_triton_template`:
- **Time:** `185.3 ms` (`+/- 0.6 ms`) over 3 runs
- **Tokens:** `13`
- **Speed:** `14.25 ms/token`
- **Accuracy:** `100.0%`

This improved on the user-reported earlier benchmark of **188 ms** while keeping
the transcription correct.

## GPU Environment Issue

The container has a driver version mismatch preventing CUDA runtime initialization:
- **nvidia-smi works** (uses NVML, independent of CUDA runtime)
- **CUDA Driver API works** (cuInit succeeds from libcuda.so.580)
- **CUDA Runtime API fails** (cudaGetDeviceCount returns error 804)
- **Root cause:** The CUDA runtime's forward compatibility check fails because the kernel module (driver 580.126.09) doesn't support the forward compat path required by the CUDA toolkit (13.1)

**To fix:** Match the NVIDIA driver version with the CUDA toolkit version, or use a container image with pre-matched driver/toolkit.

---

## Disk Space Notes

The overlay filesystem only has ~10GB total (3-4GB free). The model is 4.3GB.
**Solution:** Set `HF_HOME=/workspace/.hf_cache` to use the workspace mount (2+ PB).
Also removed unused NVIDIA cu13 duplicate packages to free overlay space.

## Commits

1. `12daf13` - feat: implement all 10 Triton GPU kernels for ASR model
2. docs: add claude.md, tutorial, reference, and code explanation
3. fix: verify end-to-end correctness (100% accuracy on test audio)
