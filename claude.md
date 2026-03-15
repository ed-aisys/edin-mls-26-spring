# Claude Development Log

## Project: GLM-ASR Triton GPU Kernel Implementation
**Date:** 2026-03-09 to 2026-03-15
**Branch:** `ankush`
**GPU:** NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32GB VRAM)
**CUDA Toolkit:** 13.0 | **Driver:** 580.126.20
**PyTorch:** 2.10.0+cu130 | **Triton:** 3.6.0

---

## Summary

Completed all 10 Triton kernel implementations + 1 fused Flash Attention kernel for the
GLM-ASR speech-to-text model. The project is a University of Edinburgh MLS course assignment
implementing GPU kernels for a multi-modal transformer (audio encoder + text decoder).

**Current benchmark: 110.0ms average, 100% transcription accuracy.**
**Baseline: 261.3ms → 57.9% faster.**

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

#### 3.2 `attention.py` — 1 fused kernel + SDPA fallback

**flash_attention_kernel** (PRIMARY): Fused Flash Attention with online softmax
- Single kernel launch replaces the old 3-kernel approach and SDPA
- Online softmax: running max `m_i`, running sum `l_i`, accumulator rescaling
- `tl.dot` for Q@K^T and P@V (tensor cores)
- Supports causal (`IS_CAUSAL`), attention mask (`HAS_MASK`), and arbitrary seq lengths
- Tile sizes: GPU-tier aware (consumer: 128/64 encoder, 64/32 decoder; datacenter: larger)
- `num_stages=1` on consumer GPUs, `num_stages=2` on datacenter
- Grid: (cdiv(seq_q, BLOCK_M), batch_heads)

**SDPA fallback** for KV-cached decode: `torch.nn.functional.scaled_dot_product_attention`
- Used when `seq_q <= 4` (single-token decode steps) — avoids Triton kernel launch overhead
- Impact: **-3ms** on decode steps

**Legacy kernels REMOVED** (Session 7): `attention_scores_kernel`, `softmax_inplace_kernel`,
`attention_output_kernel`, `causal_mask_kernel` — ~175 lines of dead code removed.
These were superseded by `flash_attention_kernel` and never invoked.

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

#### 4.6 Flash Attention Tile Sizes (GPU-Tier Aware)
```python
# Consumer GPUs (RTX 4090/5090, ~100KB shared mem):
if head_dim <= 64:   BLOCK_M, BLOCK_N = 128, 64   # Encoder
else:                BLOCK_M, BLOCK_N = 64, 32     # Decoder
# num_stages=1, num_warps=4

# Datacenter GPUs (H200/B200, ~228KB shared mem):
# Larger tiles + num_stages=2, num_warps=8
```

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
| **Average time** | **110.0ms** (+/- 0.2ms) |
| **Tokens** | 13 |
| **Speed** | 8.46 ms/token |
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

### Step 9: KV Cache + bf16 LayerNorm (Session 6, 2026-03-13)

#### 9.1 bf16 LayerNorm Output
- Modified `layernorm_kernel` to store output as bf16 (matching RMSNorm bf16 approach)
- Updated `LayerNorm.__call__` to allocate bf16 output when `Linear.BF16 = True`
- Impact: **-0.7ms** (121.8→121.1ms) — small because encoder only runs once

#### 9.2 generate_v8b with KV Cache (monkey-patched)
- Wrote `_generate_v8b()` in layers.py — uses `model.decode(use_cache=True)` per instructor guidance
- `decode()` with `use_cache=True` returns `(logits, past_key_values)` — concatenation-based KV cache
- Prefill: `logits, past_kv = self.decode(inputs_embeds=..., use_cache=True)`
- Decode loop: `logits, past_kv = self.decode(inputs_embeds=new_embeds, past_key_values=past_kv, use_cache=True)`
- Deferred monkey-patch via `_try_patch_v8b()` called in `Linear.__init__` — avoids circular imports
- Benchmark detects it via `hasattr(model, 'generate_v8b')` (already built into benchmark_student.py)
- Impact: **-7.6ms** (121.1→113.5ms) — KV cache eliminates redundant decoder computation

#### 9.3 yash/optimize Analysis
- yash/optimize model.py is **identical** to origin/main (no KV cache usage)
- Their speed advantage comes from: aggressive bf16 in all kernels, `num_stages=2`,
  `num_warps=8` in flash attention (tuned for H200 with 228KB shared memory)

### Step 10: SDPA Fallback + GPU Portability + Dead Code Cleanup (Session 7, 2026-03-15)

#### 10.1 Systematic Optimization Testing (6 optimizations tested)
Tested all remaining optimization ideas from other branches:

| Optimization | Result | Impact |
|-------------|--------|--------|
| **SDPA fallback for seq_q≤4** | **ADOPTED** | **-3ms** (113.5→110.0ms) |
| Softmax bf16 output | No change | 0ms (softmax only used for final logits) |
| Flash Attention num_stages=2 | Rejected | OOM on consumer GPUs (~100KB shared mem) |
| Flash Attention num_warps=8 | No change | 0ms (already using 4 warps on consumer) |
| PyTorch GELU/SiLU bf16 | Rejected | +0.3ms (Triton kernels faster) |
| SDPA fallback for all attention | Rejected | +5ms (Flash Attention better for long seqs) |

#### 10.2 SDPA Fallback for KV-Cached Decode
- Added `torch.nn.functional.scaled_dot_product_attention` fallback in attention.py
- Triggered when `seq_q <= 4` (single-token KV-cached decode steps)
- Avoids Triton kernel launch overhead for tiny problem sizes
- Impact: **-3ms** (113.5→110.0ms)

#### 10.3 GPU Portability (All Modules)
- `_detect_gpu_tier()` in layers.py checks `props.max_shared_memory_size_per_block > 120 * 1024`
- Applied to: Linear tile sizes, Flash Attention tiles + num_stages + num_warps, RoPE num_stages + num_warps
- Consumer GPUs (RTX 4090/5090): smaller tiles, num_stages=1, num_warps=4
- Datacenter GPUs (H200/B200): larger tiles, num_stages=2, num_warps=8

#### 10.4 Dead Code Cleanup
- Removed ~175 lines from attention.py: `attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel`, `causal_mask_kernel`
- Removed ~145 lines from layers.py: same legacy kernels that were duplicated
- These were never invoked — fully superseded by `flash_attention_kernel`

#### 10.5 generate_v8b Updated to Use decode(use_cache=True)
- Per Piazza instructor guidance: "use an existing separate decode function and handle the KV cache management yourself"
- Changed from `forward_with_kv_buffers()` to `self.decode(inputs_embeds=..., past_key_values=past_kv, use_cache=True)`
- Same performance, cleaner approach using model.py's public API

---

## Optimization Roadmap

### Planned optimizations from other branches (prioritized by expected impact):

| Priority | Optimization | Source Branch | Actual Impact | Status |
|----------|-------------|---------------|---------------|--------|
| HIGH | Fused Q+K RoPE kernel | meave | **-14ms** (138→124ms) | **ADOPTED** |
| HIGH | bf16 RMSNorm output | meave (adapted) | **-3ms** (124→121ms) | **ADOPTED** |
| HIGH | SDPA fallback for seq_q≤4 | majed (idea) | **-3ms** (113.5→110.0ms) | **ADOPTED** |
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

### Current (2026-03-15, with KV-cached generate_v8b + SDPA fallback)
| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template** | **110.0ms** | 8.46ms/tok | 100% |
| Example baseline | 261.3ms | 20.10ms/tok | 100% |
| **Speedup** | **57.9%** | | |

### Optimization Progression
| Change | Time | Delta |
|--------|------|-------|
| Baseline (example) | 261.3ms | -- |
| All kernels + cuBLAS + TF32 | 209.8ms | -51.5ms |
| bf16 weights + Flash Attention | 136.4ms | -73.4ms |
| Fused Q+K RoPE pair kernel (from meave) | 124.6ms | -11.8ms |
| bf16 RMSNorm output kernel (from meave) | 120.7ms | -3.9ms |
| bf16 LayerNorm output | 121.1ms | -0.7ms |
| generate_v8b with KV cache (monkey-patched) | 113.5ms | -7.6ms |
| SDPA fallback for KV-cached decode (seq_q≤4) | **110.0ms** | **-3.5ms** |

Note: generate_v8b uses `model.decode(use_cache=True)` per instructor guidance.
The function lives in layers.py and is monkey-patched onto GlmAsrModel via a deferred
hook in Linear.__init__. SDPA fallback uses `torch.nn.functional.scaled_dot_product_attention`
for single-token decode steps, avoiding Triton kernel launch overhead.

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

---

## What is Monkey-Patching? (And What We Were Doing)

### The Concept

**Monkey-patching** is a technique where you modify or extend code at runtime — you
replace or add methods/attributes on existing classes or objects *after* they've been
imported, without changing the original source file.

```python
# Example: monkey-patching a method onto an existing class
class Dog:
    def speak(self):
        return "Woof"

# Monkey-patch: replace or add a method at runtime
def new_speak(self):
    return "WOOF WOOF!"

Dog.speak = new_speak  # Now ALL Dog instances use new_speak
```

In Python, this works because classes are mutable objects. You can reassign their
methods, add new attributes, or swap out entire functions at runtime.

### What We Were Monkey-Patching

**The problem:** `model.py` is READ-ONLY (GUIDE.md rule 4), but its stock `generate()`
method is O(n²) — it reprocesses the entire growing sequence through all 28 decoder
layers on every decode step. With 13 tokens, that means steps of length 80, 81, 82...92,
each going through 28 layers of attention + MLP. This is the #1 performance bottleneck
(82.8% of total time in detailed benchmarks).

**The solution (on origin/ankush):** We wrote `generate_v8b()` in `layers.py` — an
optimized generation function with **KV caching**. Instead of reprocessing the full
sequence, it:
1. **Prefill once:** Process the full input through all layers, cache all K/V states
2. **Decode O(1) per step:** Each new token only passes through the 28 layers once,
   reading from cached K/V and appending the new K/V

Then we monkey-patched it onto the model:

```python
# In layers.py (current approach, per instructor guidance):
def _generate_v8b(self, input_features, input_ids=None, ...):
    # ... KV-cached generation code using decode(use_cache=True) ...
    # Prefill: process all input tokens
    logits, past_kv = self.decode(inputs_embeds=inputs_embeds, use_cache=True)
    for _ in range(max_new_tokens):
        # Only 1 token through decoder each step!
        new_embeds = self.text_decoder.embed_tokens(next_token)
        logits, past_kv = self.decode(
            inputs_embeds=new_embeds, past_key_values=past_kv, use_cache=True
        )

def _try_patch_v8b():
    """Deferred monkey-patch: called from Linear.__init__ to avoid circular imports."""
    import sys
    mod = sys.modules.get('model')
    if mod and hasattr(mod, 'GlmAsrModel'):
        mod.GlmAsrModel.generate_v8b = _generate_v8b
```

When `benchmark_student.py` checked `hasattr(model, 'generate_v8b')`, it would find
our monkey-patched method and use it instead of the stock `generate()`.

**Result:** 110.0ms with KV cache + SDPA fallback vs 120.7ms without — the KV cache
eliminated redundant computation in decode steps, and SDPA fallback avoided Triton
kernel launch overhead for single-token decode.

### Evolution of the Approach

1. **First attempt:** Added `generate_v8b` directly to model.py — violated GUIDE.md Rule 4
2. **Second attempt:** Used `forward_with_kv_buffers()` via monkey-patch from layers.py
3. **Final approach (per instructor Piazza guidance):** Use `model.decode(use_cache=True)` — cleaner,
   uses model.py's public API. `decode()` internally handles `past_key_values` concatenation.

**Result:** 110.0ms with KV cache + SDPA fallback vs 120.7ms without.

### Compliance Question

The monkey-patch does NOT modify model.py on disk (zero diff with origin/main).
It adds a NEW method (`generate_v8b`) to the class at runtime. The benchmark
already checks for this method (`hasattr(model, 'generate_v8b')`).

Whether this is allowed under GUIDE.md Rule 4 ("Do NOT modify model.py") is
debatable. Instructor guidance on Piazza: "use an existing separate decode function
and handle the KV cache management yourself." Two branches exist:
- `ankush` — with monkey-patch + SDPA fallback (110.0ms)
- `ankush-no-monkeypatch` — without monkey-patch (120.7ms)

---

## Next Steps to Explore (for 2026-03-13)

### 1. Cross-GPU Portable Optimizations (Research Completed 2026-03-13)

**Architecture-portable optimizations (work on all GPUs):**
- Flash Attention with online softmax — algorithmic improvement, always wins
- Kernel fusion (SwiGLU, RoPE pair) — reduces kernel launch overhead & DRAM round-trips
- bf16 weights — halves memory bandwidth on any GPU with bf16 support (Ampere+)
- cuBLAS backend for Linear — cuBLAS auto-tunes per GPU architecture
- TF32 flags — available on Ampere+ (sm_80+)

**GPU-specific parameters that need tuning:**

| Parameter | RTX 5090 (sm_120) | H200 (sm_90) | RTX 4090 (sm_89) | B200 (sm_120) |
|-----------|-------------------|--------------|-------------------|---------------|
| Shared memory | 101KB/SM | 228KB/SM | 100KB/SM | 228KB/SM |
| Flash attn num_stages | 1 (101KB limit) | 2-3 (228KB) | 1 (100KB limit) | 2-3 (228KB) |
| Flash attn BLOCK_M/N (hd=64) | 128/64 | 128/128 | 128/64 | 128/128 |
| Flash attn BLOCK_N (hd=128) | 32 | 64 | 32 | 64 |
| num_warps | 4 | 8 | 4 | 8 |
| SwiGLU tiles | 64x64 | 128x128 | 64x64 | 128x128 |

Key insight: Hopper/Blackwell **data-center** GPUs (H100/H200/B200) have ~2x shared
memory vs consumer GPUs (4090/5090), allowing larger tiles and more pipeline stages.
yash/optimize uses `num_stages=2, num_warps=8` — likely optimized for H200.

**Cluster-specific (multi-GPU):**
- Tensor parallelism: split attention heads across GPUs (16 Q heads → 4 per GPU)
- Pipeline parallelism: split decoder layers (28 layers → 7 per GPU)
- Not applicable for this assignment (single-GPU benchmark)

### 2. Why yash/optimize Runs Faster in Detailed Benchmark (Analysis Completed 2026-03-13)

**Key finding:** yash/optimize model.py is **identical** to origin/main — same stock
O(n²) `generate()`, no KV cache usage. But origin/main model.py does include KV cache
infrastructure (`forward_with_kv_buffer`, `allocate_kv_buffers`) that `generate()`
simply doesn't call.

**Differences that could explain their faster detailed benchmark:**

1. **More aggressive bf16 everywhere** — All kernels store output as bf16, including:
   - RMSNorm → bf16 (we do this too)
   - LayerNorm → bf16 (we store fp32)
   - Linear Triton kernel → bf16 (we use cuBLAS which handles this)
   - Softmax → bf16 (we store fp32)
   - This reduces memory bandwidth across more operations

2. **Flash Attention tuning** — `num_stages=2, num_warps=8` vs our `num_stages=1,
   num_warps=4`. More pipeline parallelism and warps can help on some GPUs.
   *Warning:* `num_stages=2` may exceed shared memory on RTX 5090 (101KB).

3. **EncoderMLP.FUSED = True** — Their encoder MLP uses a fused linear+gelu kernel.
   We have this code but it was disabled. Worth re-testing.

4. **No fused RoPE pair kernel** — They don't have our -14ms optimization.
   So their advantage must come from the other factors.

5. **No attention mask support in Flash** — Their `can_use_flash` requires
   `attention_mask is None`, falling back to legacy 3-kernel path for masked attention.
   Simpler flash kernel may compile faster.

**Actionable items (tested in Session 7):**
- [x] LayerNorm bf16 output — **DONE** (Session 6, -0.7ms)
- [x] Softmax bf16 output — **TESTED, no impact** (softmax only for final logits)
- [x] Test num_stages=2 on RTX 5090 — **TESTED, OOM** (exceeds ~100KB shared memory)
- [x] Test num_warps=8 — **TESTED, no change** on RTX 5090
- N/A: EncoderMLP.FUSED — model.py doesn't use EncoderMLP class

### 3. Autotune Attempt and Failure (2026-03-13)

**@triton.autotune for Flash Attention:** Tried 7 configs with `key=['seq_q', 'seq_k', 'head_dim']`.
- Problem: `seq_k` changes every decode step with KV cache (grows by 1 each token),
  causing re-tuning every single step. Even with `key=['head_dim']`, the Autotuner
  wrapper overhead was ~30ms per call.
- Result: Massive regression (113ms → 7800ms+ — though GPU was failing at this point).

**@triton.autotune for SwiGLU:** Tried 6 configs with varying tile sizes.
- Problem: Autotuner overhead dominated small decode-step matmuls. Padding logic
  also needed to account for max possible tile size across all autotune configs.
- Result: Regression even after fixing padding.

**All autotune code was fully reverted.** Lesson: autotune is great for static shapes
but harmful when tensor dimensions change every call (KV-cached decode).

### 4. Runtime GPU Detection (Implemented 2026-03-13)

Alternative to autotune: detect GPU class once at import time and set tile sizes accordingly.

```python
def _detect_gpu_tier():
    props = torch.cuda.get_device_properties(0)
    if props.max_shared_memory_size_per_block > 120 * 1024:
        return 'datacenter'   # H200/B200: 228KB shared mem
    return 'consumer'         # RTX 4090/5090: ~100KB shared mem

_GPU_TIER = _detect_gpu_tier()
```

Applied to: Flash Attention tile sizes + num_stages + num_warps, SwiGLU tile sizes,
Linear tile sizes, RoPE num_stages + num_warps.
Consumer GPUs get smaller tiles + num_stages=1 + num_warps=4;
datacenter GPUs get larger tiles + num_stages=2 + num_warps=8.

### 5. Further Kernel Optimizations (Lower Priority)
- [x] ~~SDPA fallback for single-token decode~~ — **DONE** (Session 7, -3ms)
- [x] ~~Softmax bf16 output~~ — **TESTED, no impact** (softmax only for final logits)
- Fuse encoder fc1→gelu into single kernel (model.py doesn't use EncoderMLP, so limited benefit)
- Profile individual kernels to find remaining hotspots

### 4. Correction: origin/main model.py Has KV Cache Infrastructure

Previous notes incorrectly stated origin/main had no KV cache support. In fact:
- `TextDecoderLayer.forward_with_kv_buffer()` — exists at line 318
- `TextDecoder.forward_with_kv_buffers()` — exists at line 492
- `TextDecoder.allocate_kv_buffers()` — exists at line 534
- **But `generate()` at line 723 does NOT use them** — it's still O(n²) concat

This means a `generate_v8b` function could be written in `layers.py` and monkey-patched
onto the model to use the existing KV cache infrastructure without modifying model.py.
However, the benchmark calls `model.generate()` directly, so it would need to either:
- Monkey-patch `generate` itself (risky — could be detected as modifying model behavior)
- Add `generate_v8b` and modify the benchmark (not allowed)
