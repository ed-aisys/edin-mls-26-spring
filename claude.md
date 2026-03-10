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

**Final benchmark result: 110.0ms average, 100% transcription accuracy, 57.9% faster than baseline.**

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

#### 3.2 `attention.py` — 4 kernels (3 legacy + 1 fused)

**flash_attention_kernel** (PRIMARY): Fused Flash Attention with online softmax
- Single kernel launch replaces the 3-kernel approach and SDPA
- Online softmax: running max `m_i`, running sum `l_i`, accumulator rescaling
- `tl.dot` for Q@K^T and P@V (tensor cores)
- Supports causal (`IS_CAUSAL`), attention mask (`HAS_MASK`), and arbitrary seq lengths
- Tile sizes: BLOCK_M=128/BLOCK_N=64 for head_dim≤64 (encoder), BLOCK_M=64/BLOCK_N=32 for head_dim=128 (decoder)
- `num_stages=1` to stay within 101KB shared memory limit
- Grid: (cdiv(seq_q, BLOCK_M), batch_heads)

**attention_scores_kernel** (legacy): `scores = sum(K * Q[broadcast], dim=-1) * scale`
- Grid: (batch_heads, seq_q)

**softmax_inplace_kernel** (legacy): In-place numerically stable softmax
- Grid: (batch_heads * seq_q,)

**attention_output_kernel** (legacy): `output = sum(V * weights[:, None], dim=0)`
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
uses `torch.nn.functional.linear(...)`, which lets PyTorch pick the best
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

#### 4.4 Kernel Fusion
```python
MLP.FUSED = True            # Fused SwiGLU: SiLU(x @ gate) * (x @ up) in one kernel
EncoderMLP.FUSED = True     # Fused Linear+GELU for encoder MLP (used by model.py)
LinearGELU.FUSED = False    # Disabled — shared memory exceeds hardware limit (131KB > 101KB)
```
`model.py` uses `EncoderMLP` for encoder layers and `LinearGELU` for the projector.
`EncoderMLP.FUSED` is active and provides speedup. `LinearGELU.FUSED` is disabled
because the projector's large dimensions (5120x4096) exceed the GPU's shared memory
limit with tile sizes 128x128x64. The unfused cuBLAS + separate GELU path is used instead.

#### 4.5 Fused Flash Attention (Triton)
```python
# attention.py — primary path: fused Flash Attention kernel
flash_attention_kernel[grid](
    q_flat, k_flat, v_flat, output, mask_flat,
    scale, seq_q, seq_k, head_dim, ...,
    IS_CAUSAL=is_causal, HAS_MASK=has_mask,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    num_stages=1,
)
```
Replaces both SDPA and the old 3-kernel approach. Uses online softmax to avoid
materializing the full attention scores matrix in DRAM. Single kernel launch
with `tl.dot` for tensor core utilization. Supports causal masking, attention
masks, and GQA (via explicit `_expand_kv_heads` before the kernel call).

#### 4.6 Activation Block Sizes
- GELU/SiLU block size: 1024 (up from default 256) for better GPU occupancy

#### 4.7 bfloat16 Weights
```python
# layers.py
Linear.BF16 = True  # Class attribute default
```
Caches bfloat16 copies of weights (`_weight_bf16`, `_bias_bf16`) on first use.
All matmuls via `F.linear` run in bf16, halving memory traffic for memory-bound
decode matmuls. Results are cast back to float32 for downstream ops.

#### 4.8 Triton Flash Attention Tile Sizes
```python
# attention.py — chosen to stay within 101KB shared memory limit
if head_dim <= 64:   # Encoder (20 heads, head_dim=64)
    BLOCK_M, BLOCK_N = 128, 64
else:                # Decoder (16 heads, head_dim=128)
    BLOCK_M, BLOCK_N = 64, 32
```
Larger tiles for encoder (more parallelism) and smaller tiles for decoder
(head_dim=128 needs more SRAM per tile). `num_stages=1` prevents Triton
from double-buffering K/V blocks, which would exceed shared memory.

#### 4.9 KV-Cached Generation (`generate_v8b`)
`model.py` natively includes `generate_v8b()`, which uses the KV cache
infrastructure (`allocate_kv_buffers` + `forward_with_kv_buffers`) for O(n)
decode instead of the O(n^2) `generate()` path. The `generate()` method
delegates to `generate_v8b()` by default. `benchmark_student.py` also checks
for `generate_v8b` and uses it when available.

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

#### 5.3 Restricted Files Verified
- GUIDE.md rule: model.py, weight_loader.py, conv.py must NOT be modified
- Verified all three files match `origin/ankush` exactly (zero diff)

### Step 6: Optimizations (Sessions 3-4, 2026-03-10)

#### 6.1 bfloat16 Weight Path
- Added `Linear.BF16 = True` as class-level default (can't rely on `__init__.py`)
- Caches bf16 copies of weights on first use
- Halves memory traffic for decode matmuls

#### 6.2 bfloat16 SDPA
- Cast Q/K/V to bfloat16 before `F.scaled_dot_product_attention`
- Enables Flash Attention / cuDNN backends (require fp16/bf16)

#### 6.3 Native GQA
- Removed explicit KV head expansion before SDPA
- `enable_gqa=True` lets PyTorch handle GQA natively with bf16
- Eliminates memory copies from head expansion

#### 6.4 Fixed Duplicate GQA Bug
- Removed duplicate `if use_gqa: k = _expand_kv_heads(...)` block in attention.py fallback path

#### 6.5 LinearGELU Fusion Disabled
- `LinearGELU.FUSED = False` — fused kernel's tile sizes (128x128x64) require 131KB shared memory,
  exceeding hardware limit of 101KB on RTX 5090
- Unfused cuBLAS + separate GELU path is used instead

#### 6.6 Removed Redundant Monkey-Patch
- `_generate_v8b` and `_try_patch_model()` removed from layers.py
- `generate_v8b` is natively defined in model.py (origin/ankush already had it)
- No runtime class modification needed

#### 6.7 Removed SDPA Fast-Path
- SDPA (`F.scaled_dot_product_attention`) try-block removed from attention.py
- Ensured Triton attention kernels are the active path (GUIDE.md compliance)
- Cost: 105.2ms → 151.0ms with old 3-kernel approach

#### 6.8 Fused Flash Attention Kernel
- Implemented `flash_attention_kernel` in Triton with online softmax
- Single kernel launch replaces the 3-kernel approach (scores → softmax → output)
- No full attention matrix materialization in DRAM
- Supports `IS_CAUSAL` (decoder) and `HAS_MASK` (attention_mask bias) via constexprs
- Tile sizes: BLOCK_M=128/BLOCK_N=64 for head_dim≤64, BLOCK_M=64/BLOCK_N=32 for head_dim=128
- `num_stages=1` to fit within RTX 5090's 101KB shared memory limit
- GQA handled via `_expand_kv_heads` before kernel call
- Result: clean 3-run benchmark at `110.0ms` average, faster than the earlier SDPA path (`113.0ms`) and GUIDE.md compliant

#### 6.9 Expanded Attention Validation
- `attention.py` self-test expanded from a small smoke/parity set to a deterministic 17-case parity suite
- Added fixed RNG seeds and explicit device reporting (`cuda` vs CPU fallback)
- Coverage now includes encoder-like ragged lengths (`175`), decoder-like prefill lengths (`93`), both mask layouts (`batch,1,...` and `batch,heads,...`), GQA, single-token decode, decode with causal+mask, and non-power-of-two shapes (`17x61`)
- GPU parity validation now passes across all 17 cases with max diff below `1e-2`

---

## Benchmark Results

### Our Implementation (`glm_asr_triton_template`)
| Metric | Value |
|--------|-------|
| **Average time** | **110.0ms** (+/- 0.3ms) |
| **Tokens** | 13 |
| **Speed** | 8.46 ms/token |
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
- **57.9% faster** than the example baseline (110.0ms vs 261.3ms)

### Optimization Progression
| Change | Time | Delta |
|--------|------|-------|
| Baseline (example) | 261.3ms | -- |
| All kernels + cuBLAS + TF32 | 209.8ms | -51.5ms |
| bf16 weights + bf16 SDPA + native GQA | 113.0ms | -96.8ms |
| Fused Flash Attention kernel (Triton), clean revalidation | 110.0ms | -3.0ms |

---

## Architecture Overview (GLM-ASR-Nano-2512)

```
Audio (WAV 16kHz)
  -> Mel Spectrogram (128 bins)
  -> Conv1D Subsampler (4x downsample)
  -> Audio Encoder (32 layers, hidden=1280, 20 heads, LayerNorm + GELU, 50% RoPE)
  -> Projector (pool 4 frames, 5120 -> 4096 -> 2048, LinearGELU + Linear)
  -> Text Decoder (28 layers, hidden=2048, 16 Q-heads / 4 KV-heads, RMSNorm + SiLU/SwiGLU, 100% RoPE)
  -> LM Head (2048 -> 59264 vocab)
  -> Text Output
```

---

## Key Files

| File | Purpose | Modifiable? |
|------|---------|:-----------:|
| `glm_asr_triton_template/layers.py` | Layer kernels (6) + config + fused kernels | Yes |
| `glm_asr_triton_template/attention.py` | Flash Attention kernel + 3 legacy kernels | Yes |
| `glm_asr_triton_template/rope.py` | RoPE kernel (1) | Yes |
| `glm_asr_triton_template/__init__.py` | Backend/fusion configuration | Yes |
| `glm_asr_triton_template/model.py` | Model architecture + KV-cached generate | **No** |
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

## GUIDE.md Compliance

| Rule | Status | Notes |
|------|--------|-------|
| 1. Triton inside kernels only | **Pass** | All `@triton.jit` kernels use only `tl.*`; cuBLAS in Python wrappers |
| 2. May use examples as reference | **Pass** | -- |
| 3. May refactor and fuse kernels | **Pass** | Fused SwiGLU + EncoderMLP + Flash Attention |
| 4. Don't modify model/weight_loader/conv | **Pass** | All three match `origin/ankush` exactly (zero diff) |

---

## Commits

1. `12daf13` — feat: implement all 10 Triton GPU kernels for ASR model
2. `5e8b191` — docs: add full documentation and optimize kernel tile sizes
3. `01fc806` — docs: update claude.md with benchmark results and correct model config
4. `714cdc9` — fix: revert model.py and conv.py to originals (do-not-modify files)
5. `bdc7690` — perf: switch to cuBLAS backend and tune tile sizes
6. `a14e2d5` — Codex commit: optimize Triton template runtime path
7. `9453c39` — Claude commit: KV-cache generate + bf16 weights + native GQA (128.7ms, 51% faster)
8. `f38ade2` — Claude commit: update docs + fix duplicate GQA bug
9. `e0bea91` — Claude commit: restore model.py/conv.py to origin, remove monkey-patch, update docs (113.0ms)
