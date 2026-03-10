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

---

## 2. Understanding the Project Structure

```
hw1-asr/
  glm_asr_triton_template/    <- YOUR WORK (fill in TODOs)
    layers.py                  <- 6 kernels to implement
    attention.py               <- 3 kernels to implement
    rope.py                    <- 1 kernel to implement
    model.py                   <- Complete (don't modify)
    conv.py                    <- Complete (don't modify)
    weight_loader.py           <- Complete (don't modify)

  glm_asr_triton_example/      <- REFERENCE (study this)
    layers.py                  <- Working implementations
    attention.py               <- Working implementations
    rope.py                    <- Working implementations
```

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
- `tl.program_id(axis)` - Which block am I? (like CUDA blockIdx)
- `tl.arange(0, N)` - Vector of [0, 1, ..., N-1] (like threadIdx)
- `tl.load/tl.store` - Memory access with mask for bounds checking
- `tl.constexpr` - Compile-time constant (determines block shape)

---

## 4. Implementing Each Kernel

### 4.1 RMSNorm Kernel

**What it does:** Root Mean Square Normalization
```
y = x / sqrt(mean(x^2) + eps) * weight
```

**Implementation steps:**
```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y,
                   hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Which row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load one row of input
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Compute variance = mean(x^2)
    var = tl.sum(x * x, axis=0) / hidden_size

    # Normalize
    x_norm = x * tl.rsqrt(var + eps)

    # Apply weight
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x_norm * w
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)
```

**Grid:** `(batch_size,)` - one block per input row.

### 4.2 LayerNorm Kernel

**What it does:** Layer Normalization
```
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

**Key difference from RMSNorm:** Subtract mean first, then compute variance of centered data.

```python
# After loading x:
mean = tl.sum(x, axis=0) / hidden_size
x_centered = x - mean
var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
x_norm = x_centered * tl.rsqrt(var + eps)
# Then apply weight AND bias:
y = x_norm * w + b
```

### 4.3 GELU Kernel

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

**Grid:** `(ceil(n_elements / BLOCK_SIZE),)` - element-wise operation.

### 4.4 SiLU Kernel

**What it does:** Sigmoid Linear Unit / Swish
```
y = x * sigmoid(x) = x / (1 + exp(-x))
```

```python
sigmoid = 1.0 / (1.0 + tl.exp(-x))
y = x * sigmoid
```

### 4.5 Linear (Matmul) Kernel

**What it does:** Tiled matrix multiplication C = A @ B

This is the most complex kernel. It uses 2D tiling:

```python
@triton.jit
def linear_kernel_tf32(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # Row tile index
    pid_n = tl.program_id(1)  # Column tile index

    # Offset ranges for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am +
                     (k + offs_k[None, :]) * stride_ak,
                     mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                     other=0.0)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk +
                     offs_n[None, :] * stride_bn,
                     mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                     other=0.0)
        acc += tl.dot(a, b)  # Uses tensor cores!

    # Store result
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

**Grid:** `(M // BLOCK_M, N // BLOCK_N)` - 2D grid of output tiles.

### 4.6 Softmax Kernel

**What it does:** Numerically stable softmax
```
y = exp(x - max(x)) / sum(exp(x - max(x)))
```

```python
x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
x = x - tl.max(x, axis=0)       # Subtract max for stability
exp_x = tl.exp(x)                # Exponentiate
denom = tl.sum(exp_x, axis=0)    # Sum
y = exp_x / denom                # Normalize
```

**Why subtract max?** Without this, `exp(large_number)` overflows to infinity.

### 4.7 Attention Score Kernel

**What it does:** Compute Q @ K^T * scale for one query position
```
scores[bh, q, :] = sum(K[bh, :, :] * Q[bh, q, :], dim=-1) * scale
```

Uses the pattern: load one query vector (1D), load all keys (2D),
broadcast-multiply and reduce.

### 4.8 Softmax In-Place Kernel

Same as softmax_kernel but writes back to the input buffer (saves memory allocation).

### 4.9 Attention Output Kernel

**What it does:** Compute weighted sum: output = attn_weights @ V
```
output[bh, q, :] = sum(V[bh, :, :] * weights[bh, q, :, None], dim=0)
```

### 4.10 RoPE Frequency Kernel

**What it does:** Compute cos/sin for rotary position embeddings
```
freqs = position * inv_freq
cos_cache[pos, :half] = cos(freqs)
cos_cache[pos, half:] = cos(freqs)  # Duplicated
sin_cache[pos, :half] = sin(freqs)
sin_cache[pos, half:] = sin(freqs)  # Duplicated
```

---

## 5. Testing Your Implementation

### Unit tests (test individual kernels):
```bash
cd hw1-asr/glm_asr_triton_template
python layers.py        # Tests RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax, MLP
python attention.py     # Tests attention score, softmax, output
python rope.py          # Tests RoPE frequency computation
```

### End-to-end benchmark:
```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_template     # Your implementation
./benchmark.sh glm_asr_triton_example       # Reference baseline
```

Expected output: `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`

The benchmark harness prefers `generate_v8b()` automatically when the model
exposes it. In the current Triton template that means:
- one prefill pass for the full prompt
- pre-allocated KV buffers for all later decode steps
- LM-head evaluation only on the newest token
- direct greedy `argmax` when `top_k=1`

---

## 6. Optimization Strategies

### 6.1 Tune Tile/Block Sizes
Try different configurations:
```python
# In layers.py, class Linear:
TILE_M = 128  # Try 32, 64, 128
TILE_N = 128  # Try 32, 64, 128
TILE_K = 64   # Try 16, 32, 64
```

### 6.2 Use Triton Autotuning
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel_tf32(...):
```

### 6.3 Kernel Fusion
Fuse consecutive operations into one kernel to eliminate intermediate memory reads/writes:
- Linear + GELU -> `linear_gelu_kernel` (already provided)
- Linear + SiLU * Linear -> `swiglu_fused_kernel` (already provided)
- Audio encoder MLP -> `EncoderMLP` now uses the fused `linear_gelu_kernel`
- Projector first stage -> `LinearGELU` now fuses `linear_1 + GELU`

### 6.4 Keep Linear on Triton in This Container
The benchmarked environment bypasses cuBLAS, so the optimized setting here is:
```python
Linear.BACKEND = "triton"
```

Reason: cuBLAS GEMM calls in this container were not reliable during testing,
while the Triton linear kernels were stable and produced the benchmarked result.
So the current benchmark configuration intentionally avoids cuBLAS for `Linear`.

### 6.5 KV Cache for Decoder
Use pre-allocated KV buffers to avoid tensor concatenation during generation:
```python
output = model.generate_v8b(
    input_features,
    input_ids=input_ids,
    input_features_mask=input_features_mask,
)
```

### 6.6 Reuse RoPE and Avoid Tiny Fused Launches
- Precompute decoder RoPE `(cos, sin)` once per prefill/decode step and reuse it
  across all decoder layers.
- Keep fused Triton MLP kernels for larger row counts, but fall back to the
  unfused path for `M=1` decode work where kernel launch and padding overhead
  can outweigh the fusion benefit.

### 6.7 Prefer SDPA for Attention Hot Paths
`attention.py` now tries `torch.nn.functional.scaled_dot_product_attention`
first. On supported CUDA runtimes this can dispatch to fused SDPA /
FlashAttention-style kernels and handle GQA without explicitly expanding KV
heads. The original Triton path remains as a fallback.

---

## 7. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA error: invalid configuration argument` | BLOCK_SIZE too large | Reduce block size to power of 2 |
| `triton.CompilationError` | Mismatched tensor shapes | Check mask dimensions match data |
| `RuntimeError: Triton Error [CUDA]: invalid argument` | Grid size 0 or negative | Add `max(1, ...)` to grid dims |
| Values all zero | Mask not applied correctly | Verify `offs < size` mask |
| NaN/Inf in output | Missing numerical stability | Subtract max before exp in softmax |
| Wrong results in matmul | Stride computation error | Print strides, verify A(M,K) @ B(K,N) |

---

## 8. Performance Targets

| Implementation | Expected Time |
|----------------|---------------|
| PyTorch CPU (`glm_asr_scratch`) | ~30s |
| Triton baseline (`glm_asr_triton_example`) | ~500ms |
| Optimized Triton (your goal) | <200ms |

Key optimizations for <200ms:
1. Keep `Linear.BACKEND = "triton"` so cuBLAS stays bypassed
2. Keep fused kernels enabled, but skip them for tiny decode rows
3. Use `generate_v8b` so decode reuses pre-allocated KV buffers
4. Reuse decoder RoPE setup across all layers in a step
5. Let attention use SDPA/FlashAttention-style kernels when the runtime supports it
6. Treat `top_k=1` as greedy decoding instead of sorting the full vocabulary

Current measured benchmark on the provided `test_audio.wav` after these changes:
- `185.3 ms (+/- 0.6 ms)` over 3 runs
- `14.25 ms/token`
- `100.0%` accuracy

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
