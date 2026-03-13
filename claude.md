# Claude Development Log

## Project: GLM-ASR Triton GPU Kernel Implementation
**Date:** 2026-03-09 to 2026-03-12
**Branch:** `ankush`
**GPU:** NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32GB VRAM)
**CUDA Toolkit:** 13.0 | **Driver:** 580.126.20
**PyTorch:** 2.10.0+cu130 | **Triton:** 3.6.0

---

## Summary

Completed all 10 Triton kernel implementations + 1 fused Flash Attention kernel for the
GLM-ASR speech-to-text model. The project is a University of Edinburgh MLS course assignment
implementing GPU kernels for a multi-modal transformer (audio encoder + text decoder).

**Current benchmark: 120.7ms average, 100% transcription accuracy.**
**Baseline: 261.3ms → 53.8% faster.**

---

## Important Constraints (from GUIDE.md)

**Do NOT modify these files (must match origin/main exactly):**
- `model.py` — model architecture and generation loop (stock `generate()`, no KV cache)
- `weight_loader.py` — loads pre-trained weights from HuggingFace
- `conv.py` — 1D convolution for audio subsampling

**Files we CAN modify:**
- `layers.py` — kernel implementations + layer classes + config knobs
- `attention.py` — attention kernels
- `rope.py` — RoPE kernel
- `__init__.py` — backend/fusion configuration

**Key model.py facts (origin/main):**
- Encoder MLP: plain `self.fc1(x) → gelu(x) → self.fc2(x)` (does NOT use `EncoderMLP` class)
- Projector: plain `self.linear_1(x) → self.act(x) → self.linear_2(x)` (does NOT use `LinearGELU` class)
- Generation: stock `generate()` — O(n²), reprocesses full sequence each decode step, no KV cache
- No `generate_v8b`, no `generate_v8`, no `generate_v6`

**Grading (from GUIDE.md, upstream merge 2026-03-12):**
- Correctness: 60 pts (accuracy > 80%)
- Performance: 30 pts
- Code quality: 10 pts

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
**softmax_inplace_kernel** (legacy): In-place numerically stable softmax
**attention_output_kernel** (legacy): `output = sum(V * weights[:, None], dim=0)`

#### 3.3 `rope.py` — 1 kernel

**compute_freqs_kernel**: `cos/sin(position * inv_freq)` precomputation
- Grid: (seq_len,)

### Step 4: Performance Optimizations (in allowed files only)

#### 4.1 Linear Backend Selection
```python
Linear.BACKEND = "torch"  # cuBLAS — fastest for Blackwell RTX 5090
```

#### 4.2 Runtime Flags
```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

#### 4.3 Kernel Fusion
```python
MLP.FUSED = True            # Fused SwiGLU: SiLU(x @ gate) * (x @ up) in one kernel (decoder)
EncoderMLP.FUSED = True     # Set in __init__.py but NOT USED — model.py uses plain fc1/fc2
LinearGELU.FUSED = False    # Set in layers.py but NOT USED — model.py uses plain linear_1/act
```
**Note:** Only `MLP.FUSED` is effective. `model.py` (origin/main) does NOT use `EncoderMLP` or
`LinearGELU` classes — the encoder and projector use plain `Linear` + standalone activation calls.

#### 4.4 Fused Flash Attention (Triton)
Replaces both SDPA and the old 3-kernel approach. Uses online softmax to avoid
materializing the full attention scores matrix in DRAM.

#### 4.5 bfloat16 Weights
```python
Linear.BF16 = True  # Class attribute default in layers.py
```
Caches bfloat16 copies of weights. All matmuls via `F.linear` run in bf16,
halving memory traffic. Results cast back to float32.

#### 4.6 Flash Attention Tile Sizes
```python
if head_dim <= 64:   BLOCK_M, BLOCK_N = 128, 64   # Encoder
else:                BLOCK_M, BLOCK_N = 64, 32     # Decoder
```
`num_stages=1` prevents double-buffering that would exceed shared memory.

### Step 5: Environment Fixes (Session 2, 2026-03-10)

#### 5.1 New Pod with CUDA 13.0
- Driver 580.126.20 + CUDA toolkit 13.0 = matching versions

#### 5.2 cuBLAS Version Mismatch Fix
- pip-installed `nvidia-cublas 13.1.0.3` conflicted with system cuBLAS 13.0
- **Fix:** `pip uninstall nvidia-cublas`

#### 5.3 Restricted Files Verified
- All three read-only files match `origin/main` exactly (zero diff)

### Step 6: Upstream Merge (Session 5, 2026-03-12)

#### 6.1 Merged ed-aisys/edin-mls-26-spring upstream
- 19 commits merged from upstream (grading criteria, benchmark updates, GUIDE.md)
- Resolved merge conflict in layers.py (kept our implementations)
- benchmark_detailed.py updated with `--attention-only`, `--linear-only` profiling

### Step 7: Detailed Profiling (Session 5, 2026-03-12)

#### 7.1 Detailed Benchmark Results (50 generated tokens)
| Component | Time | % Total |
|-----------|------|---------|
| Audio Encoder | 202.09ms | 8.7% |
| Projector | 4.14ms | 0.2% |
| Decoder Prefill | 191.59ms | 8.3% |
| **Decoder Decode (50 steps)** | **1919.94ms** | **82.8%** |
| **Total** | **2317.76ms** | 100% |

**Key insight:** Decoder decode steps dominate at 82.8%. This is because the stock
`generate()` in origin/main is O(n²) — it reprocesses the full growing sequence each step.

#### 7.2 Student Benchmark
| Metric | Value |
|--------|-------|
| **Average time** | **120.7ms** (+/- 0.2ms) |
| **Tokens** | 13 |
| **Speed** | 9.29 ms/token |
| **Accuracy** | 100.0% |

### Step 8: Branch Optimizations (Session 5, 2026-03-12)

#### 8.1 Fused Q+K RoPE Pair Kernel (from meave branch)
- Added `fused_rope_pair_kernel` to rope.py — single Triton kernel launch for both Q and K
- Grid: `((total_qh + total_kh) * seq_len,)` — programs 0..total_qh*seq_len handle Q, rest handle K
- Supports partial RoPE (audio encoder 50%) via passthrough copy for remaining dims
- Impact: **-14ms** (138→124ms)

#### 8.2 bf16 RMSNorm Output Kernel (from meave, adapted for bf16)
- Added `rmsnorm_bf16_kernel` — computes RMSNorm in fp32, stores output as bf16
- Used when `Linear.BF16 = True` — avoids fp32→bf16 conversion in next Linear layer
- Impact: **-3ms** (124→121ms)

#### 8.3 Rejected Optimizations (tested, not adopted)
- **SwiGLU grid swizzling** (yash/optimize): GROUP_SIZE_M=8, 1D grid, num_warps=8, num_stages=4. Regressed +18ms on RTX 5090 with 64x64 tiles.
- **@triton.autotune for GELU/SiLU** (majed): Added +0.7ms overhead from tuning warmup. Grid must use `lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` with autotune.

---

## Optimization Roadmap

### Planned optimizations from other branches (prioritized by expected impact):

| Priority | Optimization | Source Branch | Actual Impact | Status |
|----------|-------------|---------------|---------------|--------|
| HIGH | Fused Q+K RoPE kernel | meave | **-14ms** (138→124ms) | **ADOPTED** |
| HIGH | bf16 RMSNorm output | meave (adapted) | **-3ms** (124→121ms) | **ADOPTED** |
| MEDIUM | Swizzled SwiGLU + larger tiles | yash/optimize | **+18ms regression** (123→141ms) | Rejected |
| LOW | @triton.autotune for GELU/SiLU | majed | **+0.7ms overhead** (tuning warmup) | Rejected |
| N/A | EncoderMLP.FUSED | yash/optimize | NOT APPLICABLE — model.py doesn't use EncoderMLP | Skipped |
| N/A | LinearGELU.FUSED | yash/optimize | NOT APPLICABLE — model.py doesn't use LinearGELU | Skipped |

### Branch analysis summary:
- **majed**: cuBLAS backend, Flash Attention, PyTorch SDPA fallback for decode, @triton.autotune
- **yash/optimize**: Aggressive bf16, swizzled SwiGLU (GROUP_SIZE_M=8), num_warps=16/num_stages=7, LinearGELU.FUSED with BLOCK_K=32
- **meave**: fp16 weights, fused RMSNorm→fp16 output kernel, fused Q+K RoPE pair kernel, separate flash_decode_kernel

---

## Benchmark Results

### Current (2026-03-12, after upstream merge + branch optimizations)
| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template** | **120.7ms** | 9.29ms/tok | 100% |
| Example baseline | 261.3ms | 20.10ms/tok | 100% |
| **Speedup** | **53.8%** | | |

### Optimization Progression
| Change | Time | Delta |
|--------|------|-------|
| Baseline (example) | 261.3ms | -- |
| All kernels + cuBLAS + TF32 | 209.8ms | -51.5ms |
| bf16 weights + Flash Attention | 136.4ms | -73.4ms |
| Fused Q+K RoPE pair kernel (from meave) | 124.6ms | -11.8ms |
| bf16 RMSNorm output kernel (from meave) | 120.7ms | -3.9ms |

Note: Previous 110.0ms result was with origin/ankush model.py (had KV-cached `generate_v8b`).
With origin/main's stock `generate()` (O(n²) decode), performance is 120.7ms for 13 tokens.

---

## Architecture Overview (GLM-ASR-Nano-2512)

```
Audio (WAV 16kHz)
  -> Mel Spectrogram (128 bins)
  -> Conv1D Subsampler (4x downsample)
  -> Audio Encoder (32 layers, hidden=1280, 20 heads, LayerNorm + GELU, 50% RoPE)
  -> Projector (pool 4 frames, 5120 -> 4096 -> 2048, Linear+GELU + Linear)
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
| `glm_asr_triton_template/model.py` | Model architecture + stock generate (no KV cache) | **No** |
| `glm_asr_triton_template/conv.py` | Conv1D layers | **No** |
| `glm_asr_triton_template/weight_loader.py` | HuggingFace weight loading | **No** |
| `benchmark_student.py` | End-to-end benchmark | N/A |
| `benchmark_detailed.py` | Per-operator profiling | N/A |

---

## Running the Benchmark

```bash
cd hw1-asr

# IMPORTANT: Set HF_HOME if overlay disk space is limited (<5GB free)
export HF_HOME=/workspace/.hf_cache

# Test your implementation
python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5

# Compare against baseline
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3

# Detailed per-operator profiling
python benchmark_detailed.py glm_asr_triton_template
```

---

## GUIDE.md Compliance

| Rule | Status | Notes |
|------|--------|-------|
| 1. Triton inside kernels only | **Pass** | All `@triton.jit` kernels use only `tl.*`; cuBLAS in Python wrappers |
| 2. May use examples as reference | **Pass** | -- |
| 3. May refactor and fuse kernels | **Pass** | Fused SwiGLU + Flash Attention |
| 4. Don't modify model/weight_loader/conv | **Pass** | All three match `origin/main` exactly (zero diff) |
