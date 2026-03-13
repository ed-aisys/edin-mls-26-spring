# Tutorial: Implementing GPU Kernels for GLM-ASR with Triton

A step-by-step guide to completing the GPU kernel implementations for the
GLM-ASR speech-to-text model using OpenAI Triton.

---

## Prerequisites

- Python 3.11+
- NVIDIA GPU (Blackwell recommended, Hopper/Ampere also work)
- CUDA Toolkit 13.x
- PyTorch 2.10+
- Triton 3.6+

## 1. Environment Setup

### Option A: Using the setup script (recommended for cluster)
```bash
cd edin-mls-26-spring
source utils/setup-triton.sh
```

### Option B: Manual pip install
```bash
pip install torch triton numpy transformers datasets huggingface_hub safetensors accelerate soundfile
```

### Verify GPU access
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Verify the baseline works FIRST
```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_example
```
You should see `Accuracy: 100.0%` and `Status: PASS`. If this fails, fix your
environment before writing any code.

---

## 2. Understanding the Project Structure

```
hw1-asr/
  glm_asr_triton_template/    <- YOUR WORK (fill in TODOs)
    layers.py                  <- 6 kernels to implement + layer classes
    attention.py               <- 3 kernels to implement
    rope.py                    <- 1 kernel to implement
    __init__.py                <- Configuration (backend, fusion flags)
    model.py                   <- DO NOT MODIFY (stock generate, no KV cache)
    conv.py                    <- DO NOT MODIFY
    weight_loader.py           <- DO NOT MODIFY

  glm_asr_triton_example/      <- REFERENCE (study this)
    layers.py                  <- Working implementations
    attention.py               <- Working implementations
    rope.py                    <- Working implementations
```

**Important:** Per GUIDE.md, you must NOT modify `model.py`, `weight_loader.py`, or `conv.py`.

**Key model.py facts (origin/main):**
- Encoder MLP uses plain `self.fc1(x) → gelu(x) → self.fc2(x)` — NOT the `EncoderMLP` class
- Projector uses plain `self.linear_1(x) → self.act(x) → self.linear_2(x)` — NOT `LinearGELU`
- Only has stock `generate()` — O(n²) decode, no KV cache
- `EncoderMLP` and `LinearGELU` classes exist in layers.py but are NOT used by model.py

---

## 3. Triton Kernel Basics

Every Triton kernel follows this pattern:

```python
@triton.jit
def my_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 1. Get block ID
    pid = tl.program_id(0)

    # 2. Compute element offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create bounds mask
    mask = offs < N

    # 4. Load data
    x = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # 5. Compute
    y = x * 2.0  # your operation here

    # 6. Store result
    tl.store(output_ptr + offs, y, mask=mask)
```

Key concepts:
- `tl.program_id(axis)` — Which block am I? (like CUDA blockIdx)
- `tl.arange(0, N)` — Vector of [0, 1, ..., N-1] (like threadIdx)
- `tl.load/tl.store` — Memory access with mask for bounds checking
- `tl.constexpr` — Compile-time constant (determines block shape)

---

## 4. Implementing Each Kernel (Recommended Order)

### Phase 1: Element-wise Operations

#### 4.1 SiLU Kernel (simplest — start here)

**What it does:** Sigmoid Linear Unit / Swish activation
```
y = x * sigmoid(x) = x / (1 + exp(-x))
```

```python
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid
    tl.store(y_ptr + offs, y, mask=mask)
```

**Grid:** `(ceil(n_elements / BLOCK_SIZE),)` — element-wise, one block per chunk.

#### 4.2 GELU Kernel

**What it does:** Gaussian Error Linear Unit (tanh approximation)
```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

```python
sqrt_2_over_pi = 0.7978845608028654
x3 = x * x * x
inner = sqrt_2_over_pi * (x + 0.044715 * x3)
y = x * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
```

**Grid:** `(ceil(n_elements / BLOCK_SIZE),)` — same pattern as SiLU.

### Phase 2: Reductions

#### 4.3 RMSNorm Kernel

**What it does:** Root Mean Square Normalization (text decoder)
```
y = x / sqrt(mean(x^2) + eps) * weight
```

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y,
                   hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Which row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    var = tl.sum(x * x, axis=0) / hidden_size
    x_norm = x * tl.rsqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x_norm * w
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)
```

**Grid:** `(num_rows,)` — one block per row.

#### 4.4 LayerNorm Kernel

**What it does:** Layer Normalization (audio encoder)
```
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

**Key difference from RMSNorm:** Subtract mean first, then compute variance.
Also applies bias in addition to weight.

#### 4.5 Softmax Kernel

**What it does:** Numerically stable softmax
```
y = exp(x - max(x)) / sum(exp(x - max(x)))
```

**Why subtract max?** Without this, `exp(1000)` overflows to infinity.

### Phase 3: Tiled Matrix Multiplication

#### 4.6 Linear (Matmul) Kernel

**What it does:** `C = A @ B` using 2D tiled algorithm

This is the most complex kernel. It divides the output matrix into tiles and
accumulates each tile's result by iterating over the K dimension.

**`tl.dot(a, b)`** compiles to tensor core instructions (HMMA/WMMA) on
supported GPUs, giving ~10x speedup over regular FP32 multiply-add.

**Grid:** `(ceil(M/BLOCK_M), ceil(N/BLOCK_N))` — 2D grid of output tiles.

### Phase 4: Attention Kernels

#### 4.7-4.9 Legacy Attention Kernels

The original assignment has three separate attention kernels:
- **Attention Scores**: `Q @ K^T * scale` per query position
- **Softmax In-Place**: writes softmax back to input buffer
- **Attention Output**: `attn_weights @ V` weighted sum

These still exist in the codebase but are **superseded by the fused Flash Attention
kernel** (see Section 4.10).

#### 4.10 Fused Flash Attention Kernel (Advanced)

The `flash_attention_kernel` replaces the 3-kernel approach with a single kernel
using the **online softmax** algorithm:

```python
# Inner loop: iterate over K/V blocks
for start_n in range(0, kv_len, BLOCK_N):
    k = tl.load(...)                          # K block [BLOCK_N, BLOCK_D]
    s = tl.dot(q, tl.trans(k))                # Q @ K^T [BLOCK_M, BLOCK_N]
    # Apply causal/attention masks if needed
    m_ij = tl.max(s, axis=1)                  # Block max
    m_new = tl.maximum(m_i, m_ij)             # Running max update
    alpha = tl.exp(m_i - m_new)               # Rescale factor
    p = tl.exp(s - m_new[:, None])            # Attention weights
    l_i = alpha * l_i + tl.sum(p, axis=1)     # Running sum
    acc = alpha[:, None] * acc                 # Rescale accumulator
    v = tl.load(...)                           # V block
    acc += tl.dot(p.to(v.dtype), v)           # Accumulate P @ V
    m_i = m_new
acc = acc / l_i[:, None]                      # Final normalization
```

### Phase 5: Positional Encoding

#### 4.11 RoPE Frequency Kernel

Precomputes `cos/sin(position * inv_freq)` for all positions and frequencies.
The output is duplicated into both halves because `apply_rotary_pos_emb` splits
the input and applies the same frequencies to each half.

---

## 5. Testing Your Implementation

### Unit tests (test individual kernels):
```bash
cd hw1-asr/glm_asr_triton_template
python layers.py        # Tests RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax, MLP
python attention.py     # 17-case numerical parity suite for Flash Attention
python rope.py          # Tests RoPE frequency computation
```

### End-to-end benchmark:
```bash
cd hw1-asr
python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5

# Compare against baseline
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3
```

Expected output: `Concord returned to its place amidst the tents.`

---

## 6. Optimization Strategies

### 6.1 Backend Selection
```python
# In __init__.py:
layers.Linear.BACKEND = "torch"   # cuBLAS — fastest on RTX 5090
layers.Linear.BACKEND = "triton"  # strict Triton kernel path
```

### 6.2 Runtime Flags
```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### 6.3 Kernel Fusion
```python
layers.MLP.FUSED = True           # Fused SwiGLU in decoder MLP — EFFECTIVE
layers.EncoderMLP.FUSED = True    # NOT USED by origin/main model.py
# LinearGELU.FUSED = False        # NOT USED by origin/main model.py
```

**Important:** Only `MLP.FUSED` actually affects performance. The origin/main `model.py`
does NOT use `EncoderMLP` or `LinearGELU` — it uses plain `Linear` + `gelu()` for the
encoder MLP and projector.

### 6.4 bfloat16 Weights
```python
Linear.BF16 = True  # Class default in layers.py
```
Caches bf16 copies of weights. Must be set as class default (not just `__init__.py`)
because `__init__.py` is not always executed during benchmark imports.

### 6.5 Fused Flash Attention
Single Triton kernel with online softmax. Tile sizes chosen per head_dim:
- `head_dim=64` (encoder): `BLOCK_M=128, BLOCK_N=64`
- `head_dim=128` (decoder): `BLOCK_M=64, BLOCK_N=32`

`num_stages=1` to prevent shared memory overflow on RTX 5090 (101KB limit).

### 6.6 Planned Optimizations (from branch analysis)

| Optimization | Source | Impact | Status |
|-------------|--------|--------|--------|
| Fused Q+K RoPE kernel | **meave** | **-14ms** | ADOPTED |
| bf16 RMSNorm output | **meave** (adapted) | **-3ms** | ADOPTED |
| Swizzled SwiGLU | **yash/optimize** | +18ms regression | Rejected |
| @triton.autotune | **majed** | +0.7ms overhead | Rejected |

---

## 7. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA error: invalid configuration argument` | BLOCK_SIZE too large | Reduce to power of 2, max ~1024 |
| `triton.CompilationError` | Mismatched tensor shapes | Check mask dimensions match data |
| `CUBLAS_STATUS_INVALID_VALUE` | cuBLAS version mismatch | `pip uninstall nvidia-cublas` |
| `OutOfResources: shared memory` | Fused kernel tiles too large | Reduce tile sizes or disable fusion |
| Values all zero | Mask not applied correctly | Verify `offs < size` mask |
| NaN/Inf in output | Missing numerical stability | Subtract max before exp in softmax |
| `__init__.py` settings not taking effect | Benchmark imports modules directly | Set defaults as class attributes in layers.py |

---

## 8. Performance Results (RTX 5090, 2026-03-12)

| Implementation | Time | Speed | vs Baseline |
|----------------|------|-------|-------------|
| Our optimized template | **120.7ms** | 9.29ms/tok | **53.8% faster** |
| Example baseline | 261.3ms | 20.10ms/tok | -- |
| CPU fallback (no GPU) | ~14,000ms | ~1,000ms/tok | -- |

Key optimizations ranked by impact:
1. **cuBLAS-backed `F.linear`** + TF32 flags — cuBLAS outperforms Triton linear kernel
2. **bfloat16 weights** — halves memory traffic for decode matmuls
3. **Fused Flash Attention** — Triton kernel with online softmax
4. **Fused SwiGLU** — reduces kernel launch overhead and DRAM round-trips for decoder MLP

Detailed profiling shows decoder decode steps dominate (82.8% of total time with
50 tokens) because stock `generate()` is O(n²) — no KV cache.

---

## 9. Quick Reference: Triton API

```python
# Thread/block identification
tl.program_id(axis)           # Block index (0, 1, or 2)
tl.arange(start, end)         # Vector of indices

# Memory operations
tl.load(ptr, mask, other)     # Load with bounds check
tl.store(ptr, val, mask)      # Store with bounds check

# Reductions
tl.sum(x, axis)               # Sum reduction
tl.max(x, axis)               # Max reduction

# Math
tl.dot(a, b)                  # Matrix multiply (uses tensor cores)
tl.exp(x)                     # Exponential
tl.rsqrt(x)                   # 1/sqrt(x)
tl.cos(x), tl.sin(x)          # Trig
tl.extra.cuda.libdevice.tanh(x)  # tanh via libdevice

# Type conversion
x.to(tl.float32)              # Cast to float32

# Control
tl.where(cond, a, b)          # Conditional select
```
