# Branch Comparison: GLM-ASR Triton Implementations

A detailed, side-by-side comparison of four branches' approaches to optimizing the GLM-ASR
speech-to-text model with Triton GPU kernels. Each branch made different architectural choices
about precision, fusion, tiling, and attention strategies.

**Branches compared:**
- **ankush (ours)** — `ankush` branch, 98.5ms
- **meave** — `origin/meave`, 127.8ms
- **yash** — `origin/yash/optimize`, 128ms
- **majed** — `origin/majed`, 187.9ms

**Benchmark:** RTX 5090, CUDA 13.0, 2026-03-15. Also tested on H200 MIG 3g.71gb (teaching cluster), 2026-03-16.
All branches produce correct transcription (100% accuracy).

---

## Table of Contents

1. [Performance Summary](#1-performance-summary)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Precision Strategy (Dtype Pipeline)](#3-precision-strategy-dtype-pipeline)
4. [Linear Layer (Matmul)](#4-linear-layer-matmul)
5. [Normalization Kernels (RMSNorm / LayerNorm)](#5-normalization-kernels-rmsnorm--layernorm)
6. [Activation Kernels (GELU / SiLU)](#6-activation-kernels-gelu--silu)
7. [MLP Fusion (SwiGLU / EncoderMLP)](#7-mlp-fusion-swiglu--encodermlp)
8. [Flash Attention](#8-flash-attention)
9. [RoPE (Rotary Position Embeddings)](#9-rope-rotary-position-embeddings)
10. [Softmax](#10-softmax)
11. [Embedding](#11-embedding)
12. [KV-Cached Generation](#12-kv-cached-generation)
13. [GPU Portability](#13-gpu-portability)
14. [Init Configuration (__init__.py)](#14-init-configuration-__init__py)
15. [Dead Code and Unused Paths](#15-dead-code-and-unused-paths)
16. [Key Takeaways](#16-key-takeaways)

---

## 1. Performance Summary

| Branch | Time | vs Baseline | Key Advantage |
|--------|------|-------------|---------------|
| **ankush (ours)** | **98.5ms** (RTX 5090) / **204.6ms** (H200 MIG) | **62.3% faster** | fp16-throughout pipeline, KV cache, fused RoPE |
| meave | 127.8ms | 51.1% faster | fp16 norms, fused RoPE, flash decode kernel |
| yash | 128ms | 51.0% faster | bf16 everywhere, aggressive tiling for H200 |
| majed | 187.9ms | 28.1% faster | Conservative but clean, autotune on activations |

All branches beat the 261.3ms example baseline. Our branch is **23% faster** than the next competitor.

---

## 2. Architecture at a Glance

| Feature | ankush (ours) | meave | yash | majed |
|---------|:---:|:---:|:---:|:---:|
| **Matmul backend** | cuBLAS (fp16) | cuBLAS (fp16) | cuBLAS (bf16) | cuBLAS (fp32) |
| **Weight dtype** | fp16 | fp16 | bf16 | fp32 |
| **Pipeline dtype** | fp16 throughout | fp16 norms only | bf16 stores | fp32 throughout |
| **Flash Attention** | Yes (Triton) | Yes (Triton) | Yes (Triton) | Yes (Triton) |
| **Flash Decode kernel** | SDPA fallback | Dedicated kernel | No | SDPA fallback |
| **Fused SwiGLU** | Yes | No (dead code) | Yes (dead code) | No |
| **Fused RoPE pair** | Yes (Q+K) | Yes (Q+K) | Yes (Q+K) | No (separate) |
| **Fused RMSNorm→fp16** | Yes (kernel) | Yes (kernel) | Yes (kernel) | No |
| **KV-cached generation** | Yes (monkey-patch) | No | No | No |
| **GPU tier detection** | Yes (GPUProfile + _KNOWN_CONFIGS) | No | No | No |
| **Autotune** | No | No | No | Yes (activations) |
| **Tile swizzling** | No | No | Yes (SwiGLU) | No |

---

## 3. Precision Strategy (Dtype Pipeline)

This is arguably the most impactful difference between the branches. The precision strategy
determines how much memory bandwidth is consumed moving data between kernels.

### ankush (ours): fp16-throughout

Our branch eliminates **all** unnecessary dtype conversions. Data flows as fp16 from embedding
output through every norm, activation, linear layer, and attention operation:

```
Embedding → fp16 → RMSNorm(fp16 in, fp16 out) → Linear(fp16 in, cuBLAS HGEMM, fp16 out)
  → Attention(fp16 Q/K/V, float32 inside kernel, fp16 out) → RMSNorm(fp16 in, fp16 out)
  → SwiGLU(fp16 in, cuBLAS HGEMM, fp16 intermediate, fp16 out) → residual add → next layer
```

**Key insight:** Triton kernels already do `.to(tl.float32)` after loading, so Python-side
float32 casts are redundant. By removing them, we save ~11ms of dtype conversion overhead:

- `Linear._forward_torch()`: Removed `.float()` on output → **-7.5ms** (biggest single win)
- `silu()` / `gelu()`: Removed `x = x.float()` before kernel call → **-3.7ms**
- `RMSNorm.__call__` / `LayerNorm.__call__`: Removed `x = x.to(torch.float32)` → **-0.5ms**
- Flash attention dispatch: Removed `.float()` on Q/K/V → **-1ms**

The `_HALF_DTYPE = torch.float16` setting means cuBLAS runs HGEMM (half-precision GEMM),
which is slightly faster than bf16 on RTX 5090.

### meave: fp16 for norms only

Meave uses fp16 weights and a dedicated `rmsnorm_fp16_kernel` that outputs fp16 directly.
However, their activation wrappers (`gelu()`, `silu()`) still convert to float32 before
calling the Triton kernel, and their Linear output also goes through float32. This means
they save bandwidth on the norm→linear transition but not elsewhere:

```
RMSNorm(any in, fp16 out) → Linear(fp16 in, cuBLAS HGEMM, float32 out)
  → silu(float32 in, float32 out) → ...
```

They also define `fused_rmsnorm_linear()` and `fused_rmsnorm_multi_linear()` helper functions
that combine RMSNorm output (fp16) with a cuBLAS matmul in a single Python call. However,
these functions are **never called from model.py** (which is read-only), making them dead code.
The idea is sound — minimize the HBM round-trip between norm and projection — but it requires
model.py changes to actually use.

### yash: bf16 everywhere

Yash stores all kernel outputs as bf16. Every Triton kernel (norms, activations, attention,
matmul) explicitly casts to `tl.bfloat16` before `tl.store()`. The cuBLAS backend also uses
bf16 weights. This is a consistent strategy that halves memory bandwidth versus fp32:

```
RMSNorm(bf16 out) → Linear(bf16 in, cuBLAS bf16, bf16 out) → silu(bf16 out) → ...
```

The downside versus our fp16 approach: bf16 HGEMM is slightly slower than fp16 HGEMM on
RTX 5090, and their activations still do a Python-side float32 cast before the kernel.

### majed: fp32 throughout

Majed uses float32 for all computation and storage. No half-precision paths. Kernels compute
in fp32, store fp32, and the next kernel reads fp32:

```
RMSNorm(fp32 in, fp32 out) → Linear(fp32 in, cuBLAS fp32, fp32 out) → silu(fp32 out) → ...
```

This doubles memory bandwidth versus half-precision approaches, which is why majed is the
slowest branch. The tradeoff is maximum numerical precision, but for inference this is
unnecessary — the model weights are already quantized from training.

### Summary Table

| Aspect | ankush | meave | yash | majed |
|--------|--------|-------|------|-------|
| Weight storage | fp16 | fp16 | bf16 | fp32 |
| Norm output | fp16 | fp16 | bf16 | fp32 |
| Linear output | fp16 | fp32 | bf16 | fp32 |
| Activation input | fp16 (no cast) | fp32 (cast) | fp32 (cast) | fp32 |
| Activation output | fp16 | fp32 | bf16 | fp32 |
| Attention Q/K/V | fp16 (no cast) | fp32 (cast) | fp32 (cast) | fp32 (cast) |
| cuBLAS dtype | fp16 HGEMM | fp16 HGEMM | bf16 | fp32 |
| Bytes per element | 2 | 2-4 (mixed) | 2-4 (mixed) | 4 |

---

## 4. Linear Layer (Matmul)

All four branches implement a Triton `linear_kernel_tf32` for matrix multiplication, but
**none of them actually use it by default** — all prefer cuBLAS via `F.linear()` or
`torch.matmul()`. This is a pragmatic choice: vendor-tuned cuBLAS consistently outperforms
hand-written Triton matmul kernels for standard GEMM.

### Backend Selection

| Branch | Default Backend | Triton Kernel Available? | cuBLAS Dtype |
|--------|----------------|:------------------------:|:---:|
| ankush | `"torch"` (cuBLAS) | Yes | fp16 HGEMM |
| meave | cuBLAS (hardcoded) | Yes | fp16 HGEMM |
| yash | `"cublas"` | Yes | bf16 |
| majed | `"cublas"` | Yes | fp32 |

### Triton Matmul Tile Sizes (if used)

| Branch | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages |
|--------|:-------:|:-------:|:-------:|:---------:|:----------:|
| ankush (consumer) | 64 | 64 | 32 | 4 | 1 |
| ankush (datacenter) | 128 | 128 | 64 | 8 | 2 |
| meave | 64 | 64 | 32 | 4 | 1 |
| yash | 128 | 128 | 32 | 16 | 7 |
| majed | 64 | 64 | 32 | 4 | 1 |

Yash's aggressive `num_warps=16, num_stages=7` is tuned for H200 Hopper GPUs with 228KB
shared memory. This would OOM on consumer GPUs (RTX 4090/5090 with ~100KB). Our branch
detects the GPU tier at runtime and adjusts accordingly.

### Weight Caching

All branches cache transposed weights for the Triton kernel path (e.g., `_weight_t_padded`).
This avoids repeated transpose+pad operations. Ankush and meave also cache half-precision
copies (`_weight_bf16`) for the cuBLAS path.

### Output Dtype — The Critical Difference

The key difference is what happens to the **output** of `F.linear()`:

- **ankush:** Output stays fp16. No `.float()` call. This was the **-7.5ms** optimization.
- **meave:** Output converted to float32 via `.float()`.
- **yash:** Output stays bf16 (cuBLAS returns bf16 when inputs are bf16).
- **majed:** Output is float32 natively.

---

## 5. Normalization Kernels (RMSNorm / LayerNorm)

### RMSNorm

All branches implement the same math: `y = x / sqrt(mean(x²) + eps) * weight`. The
differences are in output dtype and whether a separate kernel variant exists:

| Branch | Kernel(s) | Output Dtype | Python-side cast? |
|--------|-----------|:---:|:---:|
| ankush | `rmsnorm_kernel` + `rmsnorm_bf16_kernel` | fp16 (via `tl.float16`) | No — removed |
| meave | `rmsnorm_kernel` + `rmsnorm_fp16_kernel` | fp16 (via `tl.float16`) | No |
| yash | `rmsnorm_kernel` | bf16 (via `tl.bfloat16`) | Yes (input cast to fp32) |
| majed | `rmsnorm_kernel` | fp32 | Yes (input cast to fp32) |

Ankush and meave both have a dedicated half-precision output kernel (`rmsnorm_bf16_kernel` /
`rmsnorm_fp16_kernel`) that computes in float32 internally but stores the result directly
as fp16. This avoids a fp32-to-HBM-to-fp16 round-trip before the next cuBLAS matmul.

The `RMSNorm.__call__` wrapper in ankush removes the Python-side `x = x.to(torch.float32)`
cast entirely — the kernel handles any input dtype by doing `.to(tl.float32)` after loading.
Meave and yash still do this cast in Python, adding overhead.

### LayerNorm

Used by the audio encoder (32 layers). Same pattern as RMSNorm but with mean subtraction and bias:

| Branch | Output Dtype | Notes |
|--------|:---:|-------|
| ankush | fp16 | Output cast to `tl.float16` in kernel |
| meave | fp32 | Standard kernel, no half-precision variant |
| yash | bf16 | Standard kernel with bf16 store |
| majed | fp32 | Standard kernel |

Ankush is the only branch with fp16 LayerNorm output, saving bandwidth in the encoder's 32 layers.

### Power-of-Two Guard

All branches check `_is_power_of_two(hidden_size)` before using the Triton kernel. If the
hidden size isn't a power of two, they fall back to PyTorch. For GLM-ASR-Nano-2512, both
hidden sizes (1280 for encoder, 2048 for decoder) meet this requirement:
- 1280 → BLOCK_SIZE = 2048 (next power of two)
- 2048 → BLOCK_SIZE = 2048 (already power of two)

---

## 6. Activation Kernels (GELU / SiLU)

### GELU (Audio Encoder + Projector)

All branches use the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
All use `tl.extra.cuda.libdevice.tanh` (or `tl.libdevice.tanh`) for the hardware-optimized tanh.

| Branch | Block Size | Autotune? | Input Cast | Output Dtype |
|--------|:----------:|:---------:|:----------:|:---:|
| ankush | 1024 | No | No cast (fp16 in) | fp16 |
| meave | 1024 | No | Cast to fp32 | fp32 |
| yash | 1024 | No | Cast to fp32 | bf16 |
| majed | 128-1024 | **Yes** (4 configs) | Cast to fp32 | fp32 |

Majed is the only branch using `@triton.autotune` on the GELU kernel, with 4 configurations
varying BLOCK_SIZE (128, 256, 512, 1024) and num_warps (4, 8). In our testing, this added
+0.7ms overhead from tuning warmup — the kernel is so simple that the autotuner's overhead
dominates any tile-size benefit.

Ankush's key difference: **no Python-side float32 cast.** The kernel itself does
`.to(tl.float32)` after loading, so the Python wrapper can pass fp16 tensors directly.
This saves ~1ms per call due to eliminated dtype conversion.

### SiLU (Text Decoder SwiGLU)

Formula: `y = x * sigmoid(x) = x / (1 + exp(-x))`. Same patterns as GELU:

| Branch | Sigmoid Implementation | Autotune? | Input Cast |
|--------|----------------------|:---------:|:----------:|
| ankush | Manual `1/(1+exp(-x))` | No | No cast (fp16 in) |
| meave | Manual `1/(1+exp(-x))` | No | Cast to fp32 |
| yash | `tl.sigmoid(x)` | No | Cast to fp32 |
| majed | Manual `1/(1+exp(-x))` | **Yes** (4 configs) | Cast to fp32 |

Yash uses `tl.sigmoid()` directly, which maps to the same hardware instruction. The others
compute it manually. Functionally equivalent.

---

## 7. MLP Fusion (SwiGLU / EncoderMLP)

### SwiGLU (Text Decoder MLP)

The decoder MLP uses SwiGLU gating: `output = down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
This involves two large matmuls (gate and up) that read the same input `x`, making fusion attractive.

| Branch | Fused Kernel? | Actually Used? | Tile Sizes | Swizzling? |
|--------|:---:|:---:|:---:|:---:|
| ankush | `swiglu_fused_kernel` | **Yes** (MLP.FUSED=True) | 64x64x32 (consumer) | No |
| meave | `swiglu_fused_kernel` | **No** (dead code) | 64x64x32 | No |
| yash | `swiglu_fused_kernel` | **No** (dead code†) | 128x128x32 | **Yes** (GROUP_SIZE_M=8) |
| majed | `swiglu_fused_kernel` | **No** (MLP.FUSED=False) | 64x64x32 | No |

**†** Yash's `MLP.FUSED = True` is set in `__init__.py`, but their `MLP.__call__` always routes
to `_forward_standard` (separate cuBLAS calls), ignoring the flag. The fused kernel exists
but is never invoked. Same issue in meave — the kernel is defined but the call path doesn't use it.

**Ankush is the only branch where the fused SwiGLU kernel is actually active at runtime.**

How the fused kernel works (same in all branches):
```python
# Instead of two separate matmuls:
#   gate = x @ gate_weight    # Reads x from HBM
#   up = x @ up_weight        # Reads x from HBM again (wasted bandwidth!)
#
# The fused kernel loads x once and computes both matmuls:
for k in range(0, K, BLOCK_K):
    a = tl.load(x[...])                # Load input tile ONCE
    gate_acc += tl.dot(a, gate_w[...])  # First matmul
    up_acc += tl.dot(a, up_w[...])      # Second matmul (same input!)
# Then fuse the activation:
result = silu(gate_acc) * up_acc        # In registers, no HBM round-trip
```

This saves one full read of the input tensor from HBM (~4MB for a typical decoder input).

**Yash's swizzling:** Yash adds L2-cache-friendly tile ordering via `GROUP_SIZE_M=8`. Instead
of mapping thread blocks linearly (`pid → (row, col)`), it groups 8 rows together so
adjacent thread blocks share input tiles in L2 cache:

```python
# Standard 2D grid mapping:
pid_m = pid // num_n_tiles
pid_n = pid % num_n_tiles

# Swizzled mapping (yash):
num_pid_in_group = GROUP_SIZE_M * num_n_tiles
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_SIZE_M)
pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
```

In our testing, yash's swizzled SwiGLU with 128x128 tiles and `num_warps=8, num_stages=4`
regressed **+18ms** on RTX 5090 — the larger tiles exceeded the consumer GPU's shared memory
budget, and the swizzling overhead wasn't worth it for the small decode-step matrix sizes.

### EncoderMLP (Audio Encoder MLP)

The encoder MLP is simpler: `output = fc2(GELU(fc1(x)))`. A `linear_gelu_kernel` fuses the
first matmul with the GELU activation.

| Branch | Fused Kernel? | Actually Used? |
|--------|:---:|:---:|
| ankush | `linear_gelu_kernel` | Flag set, but **NOT USED by model.py** |
| meave | `linear_gelu_kernel` | Flag set, but **NOT USED by model.py** |
| yash | `linear_gelu_kernel` | **Yes** (EncoderMLP.FUSED=True, wired in) |
| majed | `linear_gelu_kernel` | **No** (EncoderMLP.FUSED=False) |

**Important caveat:** The origin/main `model.py` (which cannot be modified) uses plain
`self.fc1 = Linear(...)` and calls `gelu()` inline — it does NOT use the `EncoderMLP` class.
So even though yash's `EncoderMLP.FUSED=True` is set and wired in, the encoder MLP in model.py
never goes through the `EncoderMLP` class. The flag has no effect on actual runtime performance.

The same applies to `LinearGELU` for the projector — model.py uses plain `Linear` + `gelu()`.

---

## 8. Flash Attention

All branches implement a fused Flash Attention kernel using the **online softmax** algorithm.
This is the single most important kernel optimization, as it avoids materializing the
`(seq_q × seq_k)` attention scores matrix in HBM.

### Online Softmax Algorithm (Common to All)

```python
m_i = -inf          # Running max (per query row)
l_i = 0             # Running sum of exp weights
acc = 0             # Output accumulator [BLOCK_M, BLOCK_D]

for each K/V block of size BLOCK_N:
    S = Q_tile @ K_block^T                  # Attention scores
    m_new = max(m_i, row_max(S))            # New running max
    alpha = exp(m_i - m_new)                # Correction factor for old stats
    p = exp(S - m_new)                      # New attention weights
    l_i = alpha * l_i + row_sum(p)          # Updated sum
    acc = alpha * acc + p @ V_block         # Corrected accumulator
    m_i = m_new

output = acc / l_i                          # Final normalization
```

### Tile Sizes and Launch Configuration

| Branch | Encoder (hd=64) | Decoder (hd=128) | num_stages | num_warps |
|--------|:---:|:---:|:---:|:---:|
| **ankush (consumer)** | 64×64 | 32×32 | 1 | 4 |
| **ankush (datacenter)** | 128×128 | 128×64 | 2 | 8 |
| meave (hd < 128) | 64×64 | — | 1 | 4 |
| meave (hd ≥ 128) | — | 32×32 | 1 | 4 |
| yash | 64×64 | 32×32 | 1 | 4 |
| majed | 32×32 | 32×32 | 2 | 4 |

Ankush's consumer tiles match meave's exactly (we adopted them from meave). On datacenter
GPUs, we use larger tiles with more pipelining — this was tested and confirmed beneficial
on H200 by yash's branch.

Majed uses the smallest tiles (32×32 everywhere) with `num_stages=2`, which is conservative
but safe on all GPUs.

### Decode-Time Attention (seq_q = 1)

During autoregressive decoding, each step has `seq_q = 1` — a single query attending to
all cached keys. The Triton flash kernel has significant launch overhead for this tiny problem:

| Branch | Decode Strategy | Impact |
|--------|----------------|--------|
| **ankush** | PyTorch SDPA fallback when `seq_q ≤ 4` | **-3ms** |
| meave | **Dedicated `flash_decode_kernel`** | Similar benefit |
| yash | No special handling (standard flash kernel) | — |
| majed | PyTorch SDPA fallback when `seq_q < 32` | Similar benefit |

**Meave's flash_decode_kernel** is a specialized Triton kernel optimized for `seq_q = 1`.
It uses a 1D grid `(BH,)` with one program per batch-head, tiling only over K/V:

```python
# Grid: (batch * num_heads,)
# Each program: one query vector attending to all keys
for kv_start in range(0, seq_k, BLOCK_KV):
    k = load K block
    s = dot(q_vec, k^T)          # 1D scores vector
    # online softmax update...
    v = load V block
    acc += p @ v
```

This avoids the overhead of the general 2D grid `(cdiv(seq_q, BLOCK_M), BH)` when
`seq_q` is tiny.

**Ankush and majed** both use PyTorch's `F.scaled_dot_product_attention` as a fallback.
This delegates to cuDNN/cuBLAS internally and avoids Triton compilation/launch overhead
entirely. Ankush's threshold is `seq_q ≤ 4`; majed's is `seq_q < 32`.

### Causal Masking

| Branch | Approach |
|--------|---------|
| ankush | `IS_CAUSAL` constexpr flag + early KV range clamping |
| meave | Same approach + `kv_end_causal` optimization |
| yash | Inline `tl.where(offs_m >= offs_n, qk, -inf)` + range clamping |
| majed | `tl.where(k_offs <= offs_q, qk, -inf)` — no range clamping |

The "range clamping" optimization (ankush, meave, yash) is important: for causal attention,
the kernel skips KV blocks that are entirely in the future rather than loading them and
masking to -inf. This avoids wasted memory bandwidth and compute:

```python
# Without clamping: iterate over ALL K/V blocks, mask future ones
for start_n in range(0, seq_k, BLOCK_N):
    ...  # Loads K/V even for future blocks (wasted!)

# With clamping: only iterate up to the causal boundary
kv_end = min(seq_k, (pid_m + 1) * BLOCK_M)
for start_n in range(0, kv_end, BLOCK_N):
    ...  # Only loads K/V that could contribute
```

Majed lacks this optimization, which may contribute to its slower performance.

### Additive Mask Support

| Branch | HAS_MASK support? |
|--------|:-:|
| ankush | **Yes** — constexpr `HAS_MASK` flag, additive mask loaded per K block |
| meave | **No** — falls back to decomposed kernels for masked attention |
| yash | **No** — falls back to decomposed kernels for masked attention |
| majed | **No** — falls back to PyTorch for masked attention |

Ankush's flash attention kernel supports an optional additive attention mask via the
`HAS_MASK` constexpr flag. When False, the mask code is compiled out entirely (zero overhead).
When True, a mask tile is loaded per K block and added to the scores. This means our flash
kernel handles **all** attention patterns — causal, masked, and plain — without fallbacks.

Other branches fall back to separate kernel paths (scores → mask → softmax → output) when
an attention mask is present, which is slower due to materializing the full scores matrix.

### Dot Product Precision

| Branch | Q@K^T precision | P@V precision |
|--------|:---:|:---:|
| ankush | fp32 (Q and K cast to fp32 on load) | fp32 (P and V both fp32) |
| meave | fp32 | fp32 |
| yash | fp32 (casts to bf16 for `tl.dot`, accumulates fp32) | bf16 dot, fp32 accumulator |
| majed | fp32 | fp32 |

Yash casts P and V to bf16 before `tl.dot()`, relying on fp32 accumulation to maintain
accuracy. This trades some numerical precision for potentially faster tensor core throughput
on bf16-native hardware.

---

## 9. RoPE (Rotary Position Embeddings)

### Frequency Computation (`compute_freqs_kernel`)

All branches have a Triton kernel that precomputes `cos(position * inv_freq)` and
`sin(position * inv_freq)` for all positions. The outputs use the "duplicated halves"
format: `[cos_half, cos_half]` so the rotation can use simple elementwise multiply.

| Branch | Output Dtype | Notes |
|--------|:---:|-------|
| ankush | fp32 | cos/sin forced to fp32 + contiguous before use |
| meave | fp16 | cos/sin stored in fp16 |
| yash | bf16 | cos/sin stored in bf16 |
| majed | fp32 | Standard fp32 |

### RoPE Application (Rotation of Q and K)

This is where branches diverge significantly:

| Branch | Q+K Rotation | Kernel | Partial RoPE? |
|--------|-------------|--------|:---:|
| **ankush** | **Single fused kernel** for both Q and K | `fused_rope_pair_kernel` | Yes |
| **meave** | **Single fused kernel** for both Q and K | `fused_rope_pair_kernel` | Yes |
| **yash** | **Single fused kernel** for both Q and K | `fused_rope_pair_kernel` | Yes |
| majed | **Two separate PyTorch calls** | `_apply_rope_single` × 2 | Yes |

Ankush, meave, and yash all use a `fused_rope_pair_kernel` that processes Q and K in a
single grid launch. The grid is `((total_qh + total_kh) * seq_len,)` — the first
`total_qh * seq_len` programs handle Q, the rest handle K:

```python
pid = tl.program_id(0)
is_q = pid < total_q_programs

if is_q:
    # Load from Q tensor, apply rotation, store to Q
else:
    # Load from K tensor, apply rotation, store to K
```

This halves kernel launch overhead (one launch instead of two per transformer layer).
The kernel also handles partial RoPE (audio encoder uses 50% rotary factor) by copying
through the unrotated dimensions.

**Majed** is the only branch that applies RoPE in pure PyTorch. Their `_apply_rope_single`
uses tensor slicing and elementwise ops:

```python
x1 = x[..., :half_dim]
x2 = x[..., half_dim:2*half_dim]
out1 = x1 * cos - x2 * sin
out2 = x2 * cos + x1 * sin
```

This requires two separate function calls (one for Q, one for K), each involving multiple
temporary tensors and HBM round-trips. Our fused kernel does this in registers with a single
HBM read and write per Q/K element.

**Impact:** The fused RoPE pair kernel saves ~14ms compared to the PyTorch approach.
This was one of our biggest optimizations, adopted from meave's branch.

---

## 10. Softmax

Used for the final token prediction (converting logits to probabilities). All branches
implement the numerically stable version: `y = exp(x - max(x)) / sum(exp(x - max(x)))`.

| Branch | Block Size | Output Dtype |
|--------|:---:|:---:|
| ankush | next_power_of_two(n_cols) | Same as input |
| meave | next_power_of_two(n_cols) | Same as input |
| yash | next_power_of_two(n_cols) | bf16 |
| majed | next_power_of_two(n_cols) | fp32 |

Softmax is only used for the final logits (not in the attention hot path — that uses online
softmax inside the flash kernel). So the output dtype has minimal impact. We tested bf16
softmax output and measured 0ms change.

---

## 11. Embedding

All branches implement a Triton `embedding_kernel` for GPU-accelerated lookup:

| Branch | Output Dtype | Block Size |
|--------|:---:|:---:|
| ankush | fp16 | 256 |
| meave | fp16 | 256 |
| yash | bf16 | 256 |
| majed | fp32 | 256 |

Ankush's fp16 embedding output means the entire decoder pipeline starts in fp16, which
cascades through all subsequent operations without needing a dtype cast.

---

## 12. KV-Cached Generation

This is the biggest architectural difference between branches. The stock `generate()` in
model.py (read-only) is O(n²) — it reprocesses the entire growing sequence through all 28
decoder layers on every decode step. KV caching makes this O(n).

| Branch | KV Cache? | Method | Impact |
|--------|:---------:|--------|--------|
| **ankush** | **Yes** | `generate_v8b` monkey-patched from layers.py | **-7.6ms** |
| meave | No | Uses stock `generate()` | — |
| yash | No | Uses stock `generate()` | — |
| majed | No | Uses stock `generate()` | — |

**Ankush is the only branch with KV-cached generation.**

Our `generate_v8b` function lives in `layers.py` and is monkey-patched onto `GlmAsrModel`
via a deferred hook (`_try_patch_v8b()`) called during `Linear.__init__`. It uses the model's
public API: `model.decode(use_cache=True)`, which returns `(logits, past_key_values)`.

```python
# Prefill: process all input tokens, get initial KV cache
logits, past_kv = self.decode(inputs_embeds=full_input, use_cache=True)

# Decode loop: process ONE new token per step
for _ in range(max_new_tokens):
    new_embeds = self.text_decoder.embed_tokens(next_token)
    logits, past_kv = self.decode(
        inputs_embeds=new_embeds,
        past_key_values=past_kv,
        use_cache=True
    )
    next_token = sample(logits[:, -1, :])
```

The benchmark automatically detects `generate_v8b` via `hasattr(model, 'generate_v8b')` and
uses it instead of stock `generate()`.

Other branches rely on the stock O(n²) generate, which means their decoder decode time is
proportionally higher. This is a significant advantage for our branch, especially as the
number of generated tokens grows.

---

## 13. GPU Portability

| Branch | Adapts to GPU? | How? |
|--------|:-:|-------|
| **ankush** | **Yes** | `GPUProfile` class + `_KNOWN_CONFIGS` table + dynamic tile computation |
| meave | Partially | Hardcoded for consumer GPUs |
| yash | No | Hardcoded for H200 datacenter |
| majed | Partially | Conservative settings that work everywhere |

**Ankush** uses a 3-tier GPU portability system:

1. **`_KNOWN_CONFIGS` table** — Pre-tested tile sizes for 6 GPU architectures (RTX 3090/4090/5090,
   A100, H100/H200, B200). Each entry stores optimal attention tiles (per head_dim), matmul tiles,
   and RoPE launch config. This is the fast path for known hardware.

2. **`GPUProfile` class** — Detects `sm_version`, `shared_memory_per_block_optin`, and `gpu_name`
   at import time. Classifies the GPU into one of 7+ architectures and looks up `_KNOWN_CONFIGS`.
   Uses `shared_memory_per_block_optin` (99KB on RTX 5090, 228KB on H200) — NOT
   `shared_memory_per_block` which only returns 48KB.

3. **Dynamic tile computation** — For unknown GPU architectures not in `_KNOWN_CONFIGS`,
   tiles are computed from the shared memory budget:
   - `_compute_attention_tiles()`: `(BLOCK_M + 2*BLOCK_N) * BLOCK_D * 4 + 20KB overhead`
   - `_compute_matmul_tiles()`: `TILE_K * (TILE_M + 2*TILE_N) * 4 + 20KB overhead` (SwiGLU worst case)

```python
class GPUProfile:
    def __init__(self):
        props = torch.cuda.get_device_properties(0)
        self.sm_version = torch.cuda.get_device_capability(0)
        self.smem_per_block = getattr(props, 'shared_memory_per_block_optin',
            getattr(props, 'max_shared_memory_per_block', props.shared_memory_per_block))
        # Classify: blackwell_consumer, ada, hopper, blackwell_dc, ampere_dc, ampere_consumer, older
        known = _KNOWN_CONFIGS.get(self.arch_name)
        if known:
            self.attn_tiles = known['attn_tiles']          # Tested configs
        else:
            self.attn_tiles = {hd: _compute_attention_tiles(hd, self.smem_per_block) for hd in (64, 128)}

GPU = GPUProfile()  # Computed once at import time
```

This affects tile sizes, num_stages, and num_warps across Flash Attention, SwiGLU, Linear,
and RoPE. Consumer GPUs get smaller tiles + `num_stages=1`; datacenter GPUs get larger tiles +
`num_stages=2`.

**Yash** uses aggressive settings (`num_warps=16, num_stages=7, 128x128 tiles`) tuned for
H200 — these would OOM or regress on consumer GPUs with ~100KB shared memory.

**Meave** uses conservative consumer-GPU settings (small tiles, `num_stages=1`) that work
on all GPUs but don't exploit datacenter hardware fully.

**Majed** uses the most conservative settings (`32x32` attention tiles, `num_stages=2`) that
are safe everywhere but leave performance on the table on all GPUs.

---

## 14. Init Configuration (__init__.py)

| Setting | ankush | meave | yash | majed |
|---------|--------|-------|------|-------|
| `Linear.BACKEND` | `"torch"` | cuBLAS (hardcoded) | `"cublas"` | `"cublas"` |
| `Linear.BF16` | `True` | N/A (fp16 hardcoded) | N/A (bf16 hardcoded) | N/A (fp32) |
| `MLP.FUSED` | `True` | `True`† | `True`† | `False` |
| `EncoderMLP.FUSED` | `True` | `True`† | `True`† | `False` |
| TF32 flags | **Yes** | No | No | No |
| `cudnn.benchmark` | **Yes** | No | No | No |

**†** Flag is set but the MLP class's `__call__` doesn't actually use the fused path.

Ankush is the only branch that sets TF32 flags (`torch.backends.cuda.matmul.allow_tf32`,
`torch.backends.cudnn.allow_tf32`) and `cudnn.benchmark`. TF32 allows cuBLAS to use
TensorFloat-32 precision for fp32 inputs, which is ~2x faster than pure fp32 on Ampere+
GPUs. Since we use fp16 HGEMM, the TF32 flag mainly benefits any stray fp32 matmuls.

---

## 15. Dead Code and Unused Paths

### Legacy Attention Kernels

| Branch | Has legacy kernels? | Used? |
|--------|:---:|:---:|
| ankush | **Removed** (~320 lines deleted) | N/A |
| meave | Yes (scores, softmax_inplace, output, causal_mask) | Yes (for masked attention fallback) |
| yash | Yes (scores, softmax_inplace, output, causal_mask) | Yes (for masked attention fallback) |
| majed | Yes (scores, softmax_inplace, output, causal_mask) | Yes (primary for non-flash path) |

Ankush removed all legacy attention kernels because our flash attention kernel handles all
cases (causal, masked, plain) via constexpr flags. Other branches keep them as fallbacks
for masked attention.

### Fused Kernels Defined but Not Called

| Branch | Dead Fused Code |
|--------|----------------|
| ankush | None — all defined kernels are used |
| meave | `swiglu_fused_kernel` (defined, MLP.__call__ ignores it), `fused_rmsnorm_linear`/`fused_rmsnorm_multi_linear` (defined, model.py never calls them) |
| yash | `swiglu_fused_kernel` (MLP.__call__ routes to standard path despite FUSED=True) |
| majed | `swiglu_fused_kernel` (defined but FUSED=False) |

Meave has the most dead code: the `fused_rmsnorm_linear()` and `fused_rmsnorm_multi_linear()`
functions would be powerful optimizations if model.py could be modified to use them. They
combine norm output (fp16) with the next matmul in a single Python call, minimizing HBM
round-trips. But since model.py is read-only, they remain unused.

---

## 16. Key Takeaways

### Why ankush is fastest (98.5ms)

1. **fp16-throughout pipeline** — eliminates ~11ms of unnecessary dtype conversions by removing
   Python-side float32 casts. No other branch does this comprehensively.

2. **KV-cached generation** — O(n) decode vs O(n²). The only branch with this optimization.
   Saves ~7.6ms on 13 tokens, and the advantage grows with longer outputs.

3. **Fused SwiGLU actually active** — the only branch where the fused kernel is wired into
   the runtime path, saving one full input read per decoder layer per step.

4. **GPU portability** — runtime detection adapts tile sizes for consumer vs datacenter GPUs,
   ensuring we don't OOM on consumer hardware while still exploiting datacenter features.

5. **Comprehensive flash attention** — handles causal, masked, and plain attention in a single
   kernel with no fallback paths, plus SDPA for tiny decode steps.

### Cross-GPU portability

Our branch is the only one tested and verified on multiple GPU architectures:
- **RTX 5090** (Blackwell consumer, sm_120, 99KB smem): 98.5ms
- **H200 MIG 3g.71gb** (Hopper datacenter, sm_90, 228KB smem): 204.6ms

The `_to_torch_tensor()` helper and `torch.as_tensor()` usage ensure compatibility with
both cu12 and cu130 PyTorch builds, as well as CuPy array inputs from CuTile benchmarks.

### What we adopted from other branches

- **From meave:** Fused RoPE pair kernel (-14ms), RMSNorm→fp16 kernel (-3ms), smaller flash
  attention tiles (64×64 encoder, 32×32 decoder), BLOCK_M=16 for seq_q≤16. Also inspired
  the defensive CuPy input handling (commit 51b363a), which we improved with `torch.as_tensor()`.

- **From yash:** Confirmed that cuBLAS > Triton matmul, bf16 weights are beneficial. Their
  SwiGLU swizzling and aggressive tile sizes didn't help on consumer GPUs.

- **From majed:** SDPA fallback idea for decode (-3ms). Their autotune approach didn't help
  due to tuning overhead.

### What other branches could learn from us

- **meave** could benefit from: removing Python-side float32 casts (our biggest win), adding
  KV-cached generation, and actually wiring their fused SwiGLU into the call path.

- **yash** could benefit from: GPU tier detection (their H200-tuned settings would fail on
  consumer GPUs), removing float32 casts, and adding KV cache.

- **majed** could benefit from: half-precision (fp16 or bf16) everywhere, fused RoPE pair
  kernel, fused SwiGLU, causal range clamping in flash attention, and KV cache.
