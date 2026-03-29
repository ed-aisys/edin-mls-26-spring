# Design Choices: GLM-ASR Triton GPU Kernel Optimization

Implementation-focused reference for the GLM-ASR speech recognition model optimization. Covers what was chosen, why (with benchmark data), and how it maps to code.

> Benchmark evidence on this submission branch is canonicalized under
> `benchmarks/`. Treat benchmark numbers in this file as design-history context;
> use `benchmarks/benchmark_provenance.md` to find the source-of-record for any
> canonical benchmark number.

**Result: 261.3ms baseline to 98.5ms (2.65x speedup) on RTX 5090, 100% accuracy.**

---

## Table of Contents

1. [Tile Sizes and `_KNOWN_CONFIGS`](#1-tile-sizes-and-_known_configs)
2. [Shared Memory Budget and Dynamic Tile Selection](#2-shared-memory-budget-and-dynamic-tile-selection)
3. [Flash Attention Kernel](#3-flash-attention-kernel)
4. [cuBLAS vs. Triton Matmul](#4-cublas-vs-triton-matmul)
5. [fp16 Pipeline](#5-fp16-pipeline)
6. [KV Cache via Monkey-Patching](#6-kv-cache-via-monkey-patching)
7. [Fused Kernels: SwiGLU and LinearGELU](#7-fused-kernels-swiglu-and-lineargelu)
8. [GPUProfile: Architecture Detection](#8-gpuprofile-architecture-detection)
9. [Rejected Optimizations](#9-rejected-optimizations)
10. [Performance Summary](#10-performance-summary)
11. [H200 Ablation Testing](#11-h200-ablation-testing)

---

## 1. Tile Sizes and `_KNOWN_CONFIGS`

Tile sizes are stored in `_KNOWN_CONFIGS` (`layers.py:23-86`), a dict mapping architecture name to tested tile configurations:

```python
_KNOWN_CONFIGS = {
    "blackwell_consumer": {  # RTX 5090 (sm_120, ~99KB smem)
        "attn_tiles": {
            64:  (64, 64, 1, 4),   # (BLOCK_M, BLOCK_N, nstages, nwarps)
            128: (32, 32, 1, 4),
        },
        "matmul_tiles": (64, 64, 32),  # (TILE_M, TILE_N, TILE_K)
        "rope_nstages": 1,
        "rope_nwarps": 4,
    },
    "hopper": {  # H100/H200 (sm_90, ~228KB smem)
        "attn_tiles": {
            64:  (128, 128, 2, 8),
            128: (128, 64, 2, 8),
        },
        "matmul_tiles": (128, 128, 64),
        "rope_nstages": 2,
        "rope_nwarps": 8,
    },
    # ... ada, blackwell_dc, ampere_dc, ampere_consumer
}
```

**Attention tiles** are keyed by `head_dim` (64 for encoder, 128 for decoder). The decoder uses smaller tiles because head_dim=128 doubles per-row shared memory consumption. `GPUProfile.get_attention_tiles()` (`layers.py:188`) also clamps `BLOCK_M` to 16 when `seq_q <= 16` for KV-cached decode steps.

**Matmul tiles** are used by `Linear`, `MLP`, and `EncoderMLP` classes. Set once at class definition time:
```python
class MLP:
    TILE_M, TILE_N, TILE_K = GPU.matmul_tile_m, GPU.matmul_tile_n, GPU.matmul_tile_k
```

### Development history: tile sizes tried

The tile configuration evolved through 7 commits by 3 contributors:

| Commit | Author | Matmul Tiles | Activation BLOCK_SIZE | Benchmark | Key Change |
|--------|--------|-------------|----------------------|-----------|------------|
| `12daf13` | Person1 | 64×64×32 | dynamic `next_power_of_two(hidden_size)` | ~261ms (baseline) | Initial implementation of all 10 kernels |
| `893eb35` | Person2 | 64×64×32 | `@triton.autotune` over {128, 256, 512, 1024} | **+0.7ms** overhead | Added autotune to RMSNorm, LayerNorm, GELU, SiLU |
| `7f93bfd` | Person4 | **128×64×32** | — | SwiGLU **196→83ms** | Tuned tiles for register pressure; fused MLP ops (cuTile branch) |
| `5d5bc8a` | Person4 | **128×128×32** | kept autotune | **+18ms** regression | Added grid swizzling (`GROUP_SIZE_M=8`), bf16 weights |
| `bdc7690` | Person1 | **128×128×64** | kept autotune | **214ms** (18% faster) | Switched to cuBLAS backend. TILE_K 32→64 |
| `5e8b191` | Person1 | 128×128×64 | hardcoded **1024** | bundled with docs | Removed autotune, fixed BLOCK_SIZE=1024 for all element-wise kernels |
| `e496204` | Person1 | per-GPU via `GPUProfile` | 1024 | part of 110ms result | Introduced `_KNOWN_CONFIGS` + dynamic tile computation |
| `8611863` | Person1 | per-GPU | 1024 | autotune was **101.6 vs 98.5ms** (+3.1ms) | Removed warmup autotune (~110 lines) after it found worse configs |

**Person2's `@triton.autotune` (`893eb35`)** added 4-config search on element-wise kernels:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['hidden_size'],
)
```
This was applied to `rmsnorm_kernel`, `layernorm_kernel`, `gelu_kernel`, and `silu_kernel`. The tuning warmup added +0.7ms overhead that exceeded any possible gain for these simple pointwise operations. Removed in `5e8b191`.

**Person4's swizzling (`5d5bc8a`)** added `GROUP_SIZE_M` to the fused SwiGLU kernel for L2 cache-friendly tile ordering:
```python
# Grid swizzling logic added to swiglu_fused_kernel
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
```
With `GROUP_SIZE_M=8`, `num_warps=8`, `num_stages=4`, and a 1D grid. This regressed by +18ms on RTX 5090 — the 72MB L2 cache already provides good locality without explicit reordering. Reverted.

**Warmup autotune (built then removed):** We also built a `warmup_attention_tiles()` function (~95 lines) that benchmarked all valid attention tile configs at runtime using synthetic data. It selected `BLOCK_M=16` as optimal in micro-benchmarks, but the full-pipeline benchmark showed 101.6ms vs 98.5ms for hand-tuned 64×64 — a 3.1ms regression. Synthetic micro-benchmarks don't capture inter-kernel cache interactions. The entire autotune system was removed in `8611863`.

---

## 2. Shared Memory Budget and Dynamic Tile Selection

### The optin property problem

`torch.cuda.get_device_properties().shared_memory_per_block` returns 48KB on all GPUs (the default limit). The actual usable amount requires reading the optin property. `GPUProfile.__init__()` (`layers.py:120-123`) uses a `getattr` fallback chain:

```python
self.smem_per_block = getattr(
    props, 'shared_memory_per_block_optin',         # PyTorch 2.8+
    getattr(props, 'max_shared_memory_per_block',   # some builds
            props.shared_memory_per_block)           # fallback (48KB)
)
```

Without this, H200s running older PyTorch silently fell back to 48KB and selected consumer-sized tiles (64x64 instead of 128x128), running 2x slower.

### Dynamic tile computation

For unknown GPUs, `_compute_attention_tiles()` (`layers.py:206-235`) iterates ranked configs largest-first and picks the biggest that fits:

```python
# Shared memory formula for flash attention:
needed = (BLOCK_M + 2 * BLOCK_N) * BLOCK_D * 4 + 20 * 1024  # overhead

# Example: RTX 5090, head_dim=64
# (64 + 2*64) * 64 * 4 + 20480 = 69,632 bytes (~68KB) -- fits in 99KB
# (128 + 2*128) * 64 * 4 + 20480 = 118,784 bytes (~116KB) -- exceeds 99KB
```

`_compute_matmul_tiles()` (`layers.py:238-258`) uses the SwiGLU worst case (loading A + gate_w + up_w):
```python
needed = TILE_K * (TILE_M + 2 * TILE_N) * 4 + 20 * 1024
```

`nstages` and `nwarps` are derived from tile area:
```python
nstages = 2 if smem_bytes > 150 * 1024 else 1
nwarps = 8 if bm * bn >= 4096 else 4
```

---

## 3. Flash Attention Kernel

### Implementation

`flash_attention_kernel` (`attention.py:34-157`) replaces the original 3-kernel pipeline (~200 lines of `attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel`, `causal_mask_kernel`). Signature:

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, scale,
    seq_q, seq_k, head_dim: tl.constexpr,
    # ... stride parameters ...
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
```

Grid: `(cdiv(seq_q, BLOCK_M), batch * num_heads)`. Processes Q in tiles of BLOCK_M rows, iterating K/V in BLOCK_N chunks with online softmax (running `m_i`, `l_i`, `acc` state vectors).

Key online softmax loop (`attention.py:94-145`):
```python
for start_n in range(0, kv_len, BLOCK_N):
    s = tl.dot(q, tl.trans(k))          # Q @ K^T in SRAM
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp(m_i - m_new)          # rescale factor
    p = tl.exp(s - m_new[:, None])       # attention weights
    l_i = alpha * l_i + tl.sum(p, axis=1)
    acc = alpha[:, None] * acc + tl.dot(p, v)  # P @ V in SRAM
```

`IS_CAUSAL` and `HAS_MASK` are `tl.constexpr` -- Triton compiles separate kernels with dead branches eliminated. For causal, K/V iteration is bounded: `kv_len = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M)`.

### SDPA fallback

`scaled_dot_product_attention()` (`attention.py:265-269`) falls back to PyTorch SDPA for tiny queries:
```python
if q.is_cuda and seq_q <= 4:
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=scale
    )
```

During KV-cached decode, seq_q=1. Triton kernel launch overhead dominates for a single row. Impact: -3.5ms (113.5ms to 110.0ms).

### GQA handling

The decoder uses 16 query heads / 4 KV heads. `_expand_kv_heads()` (`attention.py:224-234`) uses broadcast expansion (zero-copy `expand` + `reshape`) before the kernel call, so the flash attention kernel always operates on equal head counts.

---

## 4. cuBLAS vs. Triton Matmul

`Linear.BACKEND = "torch"` (`layers.py:807`) routes all linear layers through `F.linear()` (cuBLAS). The custom `linear_kernel_tf32` (`layers.py:430-481`) is available via `BACKEND = "triton"` but is ~5ms slower for the model's matrix sizes (1280x1280, 1280x5120, 2048x2048, 2048x5632).

The `__call__` dispatch (`layers.py:844-852`):
```python
def __call__(self, x):
    if Linear.BACKEND in ("torch", "cublas"):
        return self._forward_torch(x)
    if Linear.BACKEND == "triton":
        return self._forward_triton(x)
```

Triton matmul is used only inside fused kernels (`swiglu_fused_kernel`, `linear_gelu_kernel`) where cuBLAS cannot fuse the activation.

---

## 5. fp16 Pipeline

### Configuration

Set in `__init__.py`:
```python
layers.Linear.BF16 = True        # Enable half-precision path
layers.Linear._HALF_DTYPE = torch.float16  # fp16 over bf16
```

Despite the attribute name `BF16`, the actual dtype is controlled by `_HALF_DTYPE`. fp16 is 3.6ms faster than bf16 on RTX 5090 (98.5ms vs 102.1ms) due to cuBLAS HGEMM performance differences.

### How it flows through the model

`Linear._forward_torch()` (`layers.py:877-884`) caches fp16 weight copies and calls cuBLAS with fp16 inputs:
```python
if Linear.BF16:
    hdtype = Linear._HALF_DTYPE  # torch.float16
    if self._weight_bf16 is None:
        self._weight_bf16 = self.weight.to(hdtype)
        self._bias_bf16 = bias.to(hdtype) if bias is not None else None
    output = F.linear(x_2d.to(hdtype), self._weight_bf16, self._bias_bf16)
```

Output stays fp16 (no `.float()` conversion), so the next layer receives fp16 directly. This was the biggest single win: removing `.float()` across ~120 Linear layers saved 7.5ms.

Norm kernels output fp16 when `Linear.BF16` is True. `rmsnorm_bf16_kernel` (`layers.py:339-364`) stores the result as `tl.float16`:
```python
y = (x_norm * w).to(tl.float16)
tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)
```

`layernorm_kernel` (`layers.py:398-399`) does the same:
```python
y = (x_norm * w + b).to(tl.float16)
```

Internal computation stays fp32 throughout (`.to(tl.float32)` on load).

### Embedding fp16 output

`Embedding.__call__()` (`layers.py:975`) allocates the output in fp16:
```python
out_dtype = torch.float16 if Linear.BF16 else torch.float32
output = torch.empty((batch_size, self.embedding_dim), dtype=out_dtype, device=...)
```

---

## 6. KV Cache via Monkey-Patching

### The constraint

`model.py` is read-only. Its `generate()` is O(n^2) (reprocesses full sequence each step). The KV cache infrastructure (`TextDecoder.forward_with_kv_buffers()`, `allocate_kv_buffers()`) exists in `model.py` but `generate()` doesn't use it.

### Deferred patching mechanism

`_try_patch_v8b()` (`layers.py:1482-1493`) is called inside `Linear.__init__()` (`layers.py:810`), which runs during model construction (after `model.py` is fully loaded, avoiding circular imports):

```python
_v8b_patched = False

def _try_patch_v8b():
    global _v8b_patched
    if _v8b_patched:
        return
    import sys
    for mod_name in ('model', 'glm_asr_triton_template.model'):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, 'GlmAsrModel') and not hasattr(mod.GlmAsrModel, 'generate_v8b'):
            mod.GlmAsrModel.generate_v8b = _generate_v8b
            _v8b_patched = True
            return
```

It patches as a **class method** on `GlmAsrModel` (not an instance method), so `self` is the model instance. The `_v8b_patched` guard makes repeated calls from `Linear.__init__()` no-ops.

### The generate_v8b function

`_generate_v8b()` (`layers.py:1381-1477`) uses `self.decode(use_cache=True)`:

```python
# Prefill: all tokens, build KV cache
logits, past_kv = self.decode(inputs_embeds=inputs_embeds, use_cache=True)

# Decode loop: one token per step
for _ in range(max_new_tokens):
    # ... sampling ...
    new_embeds = self.text_decoder.embed_tokens(next_token)
    logits, past_kv = self.decode(
        inputs_embeds=new_embeds, past_key_values=past_kv, use_cache=True
    )
```

The benchmark script detects `generate_v8b` via `hasattr(model, 'generate_v8b')` and calls it automatically. Impact: -7.6ms (121.1ms to 113.5ms). Savings grow with sequence length.

Input conversion uses `_to_torch_tensor()` (`layers.py:288-300`) which handles numpy arrays, CuPy arrays (`hasattr(arr, 'get')`), and uses `torch.as_tensor()` over `torch.from_numpy()` to avoid numpy version mismatch errors in cu12 environments.

---

## 7. Fused Kernels: SwiGLU and LinearGELU

### Fused SwiGLU

`swiglu_fused_kernel` (`layers.py:546-605`) computes `SiLU(x @ gate_weight) * (x @ up_weight)` in a single kernel. It maintains two accumulators and loads `x` once:

```python
gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)
    gate_w = tl.load(gate_ptr + ...)
    up_w = tl.load(up_ptr + ...)
    gate_acc += tl.dot(a, gate_w)
    up_acc += tl.dot(a, up_w)

sigmoid = 1.0 / (1.0 + tl.exp(-gate_acc))
out = gate_acc * sigmoid * up_acc
```

Activated via `MLP.FUSED = True` (`layers.py:1124`). `MLP.__call__()` (`layers.py:1161-1165`) guards on row count:
```python
if self.use_gating and MLP.FUSED and x.is_cuda and num_rows >= self.TILE_M:
    return self._forward_fused(x)
return self._forward_standard(x)
```

For small inputs (KV-cached decode, `num_rows < TILE_M`), falls back to unfused cuBLAS. The fused path pre-caches transposed weights in `_prepare_fused_weights()` (`layers.py:1152-1159`) using `Linear._HALF_DTYPE`.

### Fused LinearGELU

`linear_gelu_kernel` (`layers.py:485-542`) computes `GELU(x @ W + b)`. Applies GELU inline after the matmul accumulation loop:
```python
acc = acc + bias[None, :]
inner = sqrt_2_over_pi * (acc + 0.044715 * acc ** 3)
acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
```

**Dead code note:** `EncoderMLP.FUSED` and `LinearGELU.FUSED` are set but never triggered because `model.py` (read-only) calls `fc1` / `gelu` / `fc2` as separate operations rather than using the `EncoderMLP` or `LinearGELU` classes.

---

## 8. GPUProfile: Architecture Detection

`GPUProfile.__init__()` (`layers.py:109-170`) classifies GPUs using `sm_version` and `smem_per_block`:

```python
if sm >= (10, 0) and self.smem_per_block > 120 * 1024:
    self.arch_name = "blackwell_dc"       # B200
elif sm >= (9, 0) and self.smem_per_block > 120 * 1024:
    self.arch_name = "hopper"             # H100/H200
elif sm >= (8, 9):
    self.arch_name = "ada"                # RTX 4090
elif sm >= (8, 0) and self.smem_per_block > 120 * 1024:
    self.arch_name = "ampere_dc"          # A100
elif sm >= (8, 0):
    self.arch_name = "ampere_consumer"    # RTX 3090
# RTX 5090 reports sm_120 = (12, 0) with ~101KB
if sm >= (12, 0) and self.smem_per_block <= 120 * 1024:
    self.arch_name = "blackwell_consumer"
```

The 120KB threshold discriminates consumer (~99-100KB) from datacenter (~164-228KB) GPUs. The RTX 5090 special case at the end overrides the `blackwell_dc` match because its smem (101KB) is consumer-class despite its sm_120 version.

The singleton is created at module scope (`layers.py:262`):
```python
GPU = GPUProfile()
```

Three-tier fallback: known GPU -> dynamic computation -> CPU defaults (layers.py:172-182).

---

## 9. Rejected Optimizations

All measured on RTX 5090 (3.5s test audio, 13 tokens):

| Optimization | Result | Root Cause |
|---|---|---|
| SwiGLU grid swizzling (GROUP_SIZE_M=8, 1D grid) | **+18ms** | RTX 5090's 72MB L2 already has good locality with 64x64 tiles |
| `@triton.autotune` for GELU/SiLU | **+0.7ms** | Tuning warmup cost exceeds gain for pointwise ops |
| Flash attention `num_stages=2` | **Kernel won't launch** | ~99KB smem can't hold two tile buffers (needs ~136KB) |
| `num_stages=2` threshold: `smem_bytes > 150 * 1024` | -- | Only datacenter GPUs (228KB) benefit from double-buffering |
| Flash attention `num_warps=8` with 64x64 tiles | **0ms** | Tile too small (4096 elements) to benefit from 256 threads |
| PyTorch SDPA for encoder/prefill | **+6ms** | Custom kernel with tuned tiles is faster for seq_len ~1500 |
| SDPA `enable_gqa=True` for decode | **+13ms** | Manual `_expand_kv_heads` + standard SDPA is faster |
| Runtime autotune (~95 lines) | **+3.1ms** | Micro-benchmarks chose configs worse in full-pipeline context |
| Softmax bf16 output | **0ms** | Softmax is in-register inside flash attention; standalone runs once (~40us for final logits) |
| Fused gate+up Linear in MLP | **Neutral** | Reshape overhead offset kernel launch savings |

---

## 10. Performance Summary

### Optimization progression (RTX 5090)

| # | Change | Time | Delta |
|---|--------|------|-------|
| 0 | Baseline | 261.3ms | -- |
| 1 | Triton kernels + cuBLAS + TF32 | 209.8ms | -51.5ms |
| 2 | bf16 weights + flash attention | 136.4ms | -73.4ms |
| 3 | Fused Q+K RoPE pair kernel | 124.6ms | -11.8ms |
| 4 | bf16 RMSNorm output kernel | 120.7ms | -3.9ms |
| 5 | bf16 LayerNorm output | 121.1ms | -0.7ms |
| 6 | `generate_v8b` with KV cache | 113.5ms | -7.6ms |
| 7 | SDPA fallback for seq_q<=4 | 110.0ms | -3.5ms |
| 8 | fp16 cuBLAS HGEMM | 109.6ms | -0.4ms |
| 9 | Remove `Linear._forward_torch` `.float()` | 102.1ms | **-7.5ms** |
| 10 | Remove silu/gelu fp32 cast | 98.4ms | **-3.7ms** |
| 11 | fp16 norm outputs + embeddings + fused kernels | **98.5ms** | ~0ms |

### Fused RoPE pair kernel

`fused_rope_pair_kernel` (`rope.py:189-265`) processes both Q and K in a single grid launch. Grid: `((total_qh + total_kh) * seq_len,)`. Programs `[0, total_qh * seq_len)` handle Q, the rest handle K. Called in `apply_rotary_pos_emb()` (`rope.py:351-365`):

```python
total_programs = (total_qh + total_kh) * seq_len
fused_rope_pair_kernel[(total_programs,)](
    q_flat, k_flat, cos_half, sin_half, qo_flat, ko_flat,
    half_dim, head_dim, seq_len, total_qh, total_kh,
    ...,
    BLOCK_HD=BLOCK_HD,
    num_stages=GPU.rope_nstages,
    num_warps=GPU.rope_nwarps,
)
```

Handles partial RoPE for encoder (50% rotary factor) via passthrough copy of remaining dimensions (`rope.py:260-265`).

### Cross-GPU results

| GPU | Our Time | Baseline | Speedup |
|-----|----------|----------|---------|
| RTX 5090 (170 SMs) | 100.4ms | 262.2ms | 61.7% |
| H200 MIG 3g.71gb (60 SMs) | 204.6ms (historical) | 464.1ms | 55.9% |

The canonical H200 benchmark number on this branch is `204.8ms`, backed by the
five-invocation raw log in `benchmarks/benchmarks.md`. The `204.6ms` value above
is an older historical note and should not be used as the branch's source of record.

### Cross-branch comparison

| Branch | Time | Notes |
|--------|------|-------|
| Person1 | 98.5ms | fp16 pipeline, KV cache, flash attention, cuBLAS, GPUProfile |
| Person3 | 127.8ms | fp16 weights, fused RoPE, separate flash_decode_kernel |
| Person2 | 187.9ms | cuBLAS, flash attention, @triton.autotune |

---

## 11. H200 Ablation Testing

Systematic ablation study on **H200 MIG 3g.71gb** (60 SMs, 70GB HBM3e, 227KB shared memory per block). Each test disables or modifies a single optimization from the fully-optimized baseline, measuring end-to-end inference time (3.5s test audio, 13 tokens, 10 iterations). Baseline optimized: **205.2ms**.

### Full results

| # | Test | Time (ms) | Std (ms) | Delta (ms) |
|---|------|-----------|----------|------------|
| 1 | baseline_optimized | 205.2 | 0.8 | +0.0 |
| 2 | precision_fp32 | 206.7 | 1.4 | +1.5 |
| 3 | fusion_off_mlp | 204.5 | 2.7 | -0.7 |
| 4 | fusion_off_rope | 234.1 | 1.6 | +28.9 |
| 5 | backend_triton_matmul | 206.2 | 0.5 | +1.0 |
| 6 | sdpa_fallback_off | 208.6 | 1.7 | +3.4 |
| 7 | sdpa_threshold_1 | 203.8 | 0.7 | -1.4 |
| 8 | sdpa_threshold_8 | 204.8 | 2.3 | -0.4 |
| 9 | sdpa_threshold_16 | 205.7 | 2.6 | +0.5 |
| 10 | attn_enc_64x64_s1_w4 | 206.1 | 3.3 | +0.9 |
| 11 | attn_enc_128x64_s2_w8 | 207.0 | 0.7 | +1.8 |
| 12 | attn_enc_64x32_s1_w4 | 209.5 | 0.6 | +4.3 |
| 13 | attn_dec_64x64_s2_w8 | 205.8 | 0.3 | +0.6 |
| 14 | attn_dec_32x32_s1_w4 | 212.8 | 1.6 | +7.6 |
| 15 | attn_dec_64x32_s2_w8 | 205.7 | 1.6 | +0.5 |
| 16 | enc_nstages_1 | 217.4 | 4.8 | +12.2 |
| 17 | enc_nstages_3 | 207.6 | 2.4 | +2.4 |
| 18 | enc_nwarps_4 | 217.2 | 0.5 | +12.0 |
| 19 | enc_nwarps_16 | 209.5 | 0.7 | +4.3 |
| 20 | matmul_64x64x32 | 207.2 | 1.8 | +2.0 |
| 21 | matmul_128x64x32 | 204.8 | 0.6 | -0.4 |
| 22 | matmul_128x128x32 | 208.1 | 1.6 | +2.9 |

### Key findings (ranked by impact)

**1. Fused RoPE is the single largest contributor (+28.9ms).** Disabling `fused_rope_pair_kernel` (`rope.py:189-265`) and falling back to separate Q and K RoPE applications costs 14% of total inference time. On the H200, with only 60 SMs available on the MIG partition, kernel launch overhead is proportionally more expensive than on the RTX 5090 (170 SMs). Fusing the two rotary embedding passes into a single grid launch eliminates one full kernel dispatch and halves global memory round-trips for the position embedding tensors.

**2. Attention pipeline tuning (`num_stages`, `num_warps`) is the second most impactful class of parameters.** Dropping `num_stages` from 2 to 1 costs +12.2ms; dropping `num_warps` from 8 to 4 costs +12.0ms. The H200's 227KB shared memory comfortably supports double-buffered (`num_stages=2`) tiles at 128x128, hiding global memory latency behind compute. With `num_stages=1`, the kernel stalls on DRAM fetches. Similarly, 8 warps (256 threads) fully utilize the 128x128 tile (16,384 elements), while 4 warps leave half the tile bandwidth idle.

**3. Decoder attention tile size matters more than encoder tile size.** `attn_dec_32x32_s1_w4` costs +7.6ms vs +4.3ms for the equivalent encoder downgrade (`attn_enc_64x32_s1_w4`). Decoder attention runs once per generated token across 12 layers with `head_dim=128`, so each tile configuration change is amplified by the autoregressive loop. The default `(128, 64, 2, 8)` for decoder head_dim=128 (`_KNOWN_CONFIGS["hopper"]`) already accounts for the larger per-row shared memory footprint.

**4. SDPA fallback threshold is well-calibrated.** The default threshold of `seq_q <= 4` (`attention.py:265`) saves +3.4ms over no fallback. Lowering to `seq_q <= 1` saves an additional -1.4ms (203.8ms total), suggesting that on the H200 MIG partition, PyTorch SDPA is marginally faster than the custom kernel even for seq_q=2-4. The optimal threshold may differ on full H200 (132 SMs).

**5. Matmul TILE_K=64 justifies the 32-to-64 upgrade.** The default `(128, 128, 64)` vs `(128, 128, 32)` saves 2.9ms. TILE_K=64 means fewer loop iterations in the matmul accumulation loop, and the H200's 227KB smem can hold the larger K-dimension tiles without pressure.

### H200 vs RTX 5090 comparison

The two GPUs expose fundamentally different bottlenecks:

| Parameter | H200 MIG 3g.71gb | RTX 5090 |
|-----------|-------------------|----------|
| SMs | 60 | 170 |
| Shared memory | 227KB | ~99KB |
| Memory | 70GB HBM3e | 32GB GDDR7 |
| `num_stages` | 2 (critical: +12.2ms) | 1 (won't launch with 2) |
| `num_warps` | 8 (critical: +12.0ms) | 4 (8 warps = 0ms change) |
| Fused RoPE | +28.9ms without | -11.8ms gain (Section 10) |
| fp16 vs fp32 | +1.5ms (negligible) | -3.6ms (fp16 vs bf16 alone) |
| Decoder tile sensitivity | +7.6ms (32x32 vs 128x64) | Not tested at same configs |

**What matters on H200:** Double-buffering and warp count. The 227KB shared memory budget enables `num_stages=2` which the RTX 5090 physically cannot use. Disabling double-buffering on H200 is equivalent to running the RTX 5090's memory access pattern on H200 hardware — it costs 12ms because the kernel stalls waiting for HBM3e fetches that could be overlapped with compute.

**What matters on RTX 5090:** Precision and kernel fusion. The fp16-vs-bf16 distinction (3.6ms) and removing `.float()` casts (7.5ms) dominate because the RTX 5090's cuBLAS HGEMM throughput is more sensitive to dtype. The H200's HBM3e bandwidth means data format changes have less relative impact.

### Surprising results

**MLP fusion is neutral on H200 (-0.7ms, within noise).** `fusion_off_mlp` (disabling `swiglu_fused_kernel` and falling back to separate cuBLAS calls for gate/up projections) shows no measurable regression. The fused kernel (`layers.py:546-605`) saves one `x` reload and one kernel launch, but cuBLAS on the H200 is already extremely efficient at the model's matrix sizes (1280x5120). The HBM3e bandwidth (4.8 TB/s peak) means reloading `x` is nearly free for the batch sizes encountered. On the RTX 5090, MLP fusion was similarly neutral (Section 9: "Fused gate+up Linear in MLP — Neutral"), confirming this is a model-size effect rather than an architecture effect.

**fp16 barely matters on H200 (+1.5ms).** `precision_fp32` forces all computation to fp32, yet the regression is only 1.5ms (0.7% of total time). On the RTX 5090, fp16 was a major win (Section 5: removing `.float()` saved 7.5ms alone). The H200 MIG partition is compute-bound rather than bandwidth-bound at these matrix sizes — the 60 SMs cannot saturate HBM3e bandwidth regardless of data width. The fp16 path still helps by reducing shared memory tile footprint (enabling larger tiles), but the direct bandwidth savings are negligible.

**Triton matmul nearly matches cuBLAS (+1.0ms).** `backend_triton_matmul` switches all `Linear` layers from `F.linear()` (cuBLAS HGEMM) to `linear_kernel_tf32` (`layers.py:430-481`). The gap is only 1.0ms on H200 vs ~5ms on RTX 5090 (Section 4). The H200's `sm_90` ISA gives Triton better codegen targets (wgmma instructions), narrowing the gap with cuBLAS's hand-tuned PTX. For a project with tighter deadlines, the Triton matmul backend would have been acceptable on H200.

### What the data tells us about the H200 architecture

**Latency-bound, not throughput-bound (on MIG).** The 60-SM MIG partition shifts the bottleneck from memory bandwidth to kernel launch overhead and pipeline stalls. Evidence: fused RoPE (+28.9ms) and `num_stages` (+12.2ms) are the top two regressions, both of which address latency hiding rather than raw throughput. On a full H200 (132 SMs), the balance would shift toward bandwidth sensitivity.

**Double-buffering is the H200's defining advantage.** The 227KB shared memory enables `num_stages=2` for attention tiles up to 128x128, which is physically impossible on consumer GPUs (~99KB). This single capability accounts for 12.2ms of savings — nearly 6% of inference time. The `_KNOWN_CONFIGS["hopper"]` entry (`layers.py:39-47`) correctly prioritizes larger tiles with `nstages=2` over the smaller single-buffered configurations used on consumer architectures.

**Warp occupancy scales with tile area.** 8 warps benefit 128x128 tiles (+12.0ms regression at 4 warps) but 16 warps show diminishing returns (+4.3ms regression). The sweet spot is `tile_area / 32 = warps`: 128x128=16384 elements, 16384/32=512 threads=16 warps would be ideal by this metric, but register pressure at 16 warps likely causes spills. The 8-warp default (256 threads) balances occupancy against register file usage.

**HBM3e bandwidth masks precision effects.** The 4.8 TB/s peak bandwidth of HBM3e (even partitioned to ~1.6 TB/s on MIG 3g) means that fp16-vs-fp32 memory traffic differences are absorbed by the memory controller before becoming a bottleneck. This contrasts sharply with GDDR7 on the RTX 5090, where every byte of memory traffic competes with compute for attention.
