# HW1-ASR: Detailed Student Guide

> This guide walks you through implementing GPU kernels that power a real automatic speech recognition (ASR) model.
> For a quick command reference, see the [README](README.md).

---

## 1. What You Will Build

You will implement the GPU kernels behind **GLM-ASR**, a speech-to-text model that converts audio into text. Your kernels -- written in either **Triton** or **cuTile** -- will handle the core operations (normalization, activations, matrix multiplication, attention, positional encoding) that the model uses for inference. By the end, your implementation will run the full ASR pipeline end-to-end on a GPU.

```
Audio (WAV)
  |
  v
Mel Spectrogram (128 bins)
  |
  v
Conv Subsampler (4x downsample)
  |
  v
Audio Encoder (32 layers)          <-- LayerNorm, GELU, Linear, Attention, RoPE
  |
  v
Projector (pool 4 frames, MLP)     <-- GELU, Linear
  |
  v
Text Decoder (28 layers)           <-- RMSNorm, SiLU, Linear, Attention, RoPE
  |
  v
Text Output
```

**Expected outcome** when all kernels are correct:

```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

---

## 2. Background: The GLM-ASR Pipeline

### Model Architecture

The model has three major components. The numbers below come from the actual `GlmAsrConfig` in `model.py`.

| Component | Details |
|-----------|---------|
| **Audio Encoder** | 32 layers, hidden=1280, 20 heads (head_dim=64), intermediate=5120, **LayerNorm** + **GELU**, partial RoPE (50% of head_dim) |
| **Projector** | Pools 4 consecutive audio frames (1280x4 = 5120 -> 4096 -> 2048), **GELU** activation |
| **Text Decoder** | 28 layers, hidden=2048, 16 Q heads / 4 KV heads (GQA, head_dim=128), intermediate=6144, **RMSNorm** + **SiLU/SwiGLU**, full RoPE (base=500000) |

### Which Kernel Goes Where

| Kernel | Audio Encoder | Projector | Text Decoder |
|--------|:---:|:---:|:---:|
| `layernorm` | x | | |
| `rmsnorm` | | | x |
| `gelu` | x | x | |
| `silu` | | | x |
| `linear_kernel_tf32` | x | x | x |
| `attention_scores` | x | | x |
| `softmax_inplace` | x | | x |
| `attention_output` | x | | x |
| `compute_freqs` (RoPE) | x | | x |
| `softmax` (standalone) | | | x |

### Do NOT Modify

These files are shared infrastructure. Modifying them will break the pipeline:

- `model.py` -- model architecture and generation loop
- `weight_loader.py` -- loads pre-trained weights from HuggingFace
- `conv.py` -- 1D convolution for audio subsampling

---

## 3. Choosing Your Track

| | **Triton** | **cuTile** |
|---|---|---|
| **Tensor library** | PyTorch (`torch`) | CuPy (`cp`) |
| **Kernel language** | Triton (`triton.language` / `tl`) | NVIDIA cuTile (`cuda.tile` / `ct`) |
| **Hardware** | Any NVIDIA GPU (Ampere+) | Blackwell native; compatibility layer for others |
| **Template folder** | `glm_asr_triton_template/` | `glm_asr_cutile_template/` |
| **Example folder** | `glm_asr_triton_example/` | `glm_asr_cutile_example/` |

**Pick ONE track.** We recommend **Triton** for broader hardware compatibility and a gentler learning curve. Both tracks are graded equally and receive equal coverage in this guide.

---

## 4. Getting Started

### 4.1 Environment Setup

From the **repository root** (one level above `hw1-asr/`):

```bash
# Triton track
source utils/setup-triton.sh

# cuTile track
source utils/setup-cutile-fix.sh
```

Both scripts create a conda environment with Python 3.11 and install all required dependencies.

### 4.2 Verify the Baseline Works

Before writing any code, confirm the example (reference) implementation runs:

```bash
# Triton
./benchmark.sh glm_asr_triton_example

# cuTile
./benchmark.sh glm_asr_cutile_example
```

You should see:

```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

**If this fails, the problem is your environment, not your code.** Fix the environment first before proceeding.

### 4.3 Understanding the Codebase

Recommended reading order:

1. **Template files** (your work) -- read the TODOs to understand what needs implementing:
   - `glm_asr_triton_template/layers.py`, `attention.py`, `rope.py`
   - (or the `cutile_template` equivalents)

2. **Example files** (reference) -- see complete, working implementations:
   - `glm_asr_triton_example/layers.py`, `attention.py`, `rope.py`
   - (or the `cutile_example` equivalents)

3. **Compare side-by-side** -- diff a template against its example to see exactly what code you need to add.

4. **Optional**: `model.py` (how layers are assembled), `benchmark_student.py` (how correctness is checked).

---

## 5. Your Assignment

### 5.1 Files to Edit

**Triton track** -- all files in `glm_asr_triton_template/`:

| Kernel | File | Formula |
|--------|------|---------|
| `silu_kernel` | `layers.py` | `x * sigmoid(x)` |
| `gelu_kernel` | `layers.py` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))` |
| `softmax_kernel` | `layers.py` | `exp(x - max) / sum(exp(x - max))` |
| `rmsnorm_kernel` | `layers.py` | `x / sqrt(mean(x^2) + eps) * weight` |
| `layernorm_kernel` | `layers.py` | `(x - mean) / sqrt(var + eps) * weight + bias` |
| `linear_kernel_tf32` | `layers.py` | `A @ B` (tiled matmul) |
| `attention_scores_kernel` | `attention.py` | `Q @ K^T * scale` |
| `softmax_inplace_kernel` | `attention.py` | Softmax (in-place) |
| `attention_output_kernel` | `attention.py` | `attn_weights @ V` |
| `compute_freqs_kernel` | `rope.py` | `cos/sin(pos * inv_freq)` |

**cuTile track** -- all files in `glm_asr_cutile_template/`:

| Kernel | File | Formula |
|--------|------|---------|
| `silu_kernel` | `layers.py` | `x * sigmoid(x)` |
| `gelu_kernel` | `layers.py` | `0.5 * x * (1 + tanh(...))` |
| `softmax_kernel` | `layers.py` | `exp(x - max) / sum(exp(x - max))` |
| `rmsnorm_kernel` | `layers.py` | `x / sqrt(mean(x^2) + eps) * weight` |
| `layernorm_kernel` | `layers.py` | `(x - mean) / sqrt(var + eps) * w + b` |
| `linear_kernel_tf32` | `layers.py` | `x @ weight_t` (TF32 matmul) |
| `attention_scores_kernel` | `attention.py` | `Q @ K^T * scale` |
| `softmax_inplace_kernel` | `attention.py` | Softmax (in-place) |
| `attention_output_kernel` | `attention.py` | `attn_weights @ V` |
| `compute_freqs_kernel` | `rope.py` | `cos/sin(pos * inv_freq)` |

### 5.2 Recommended Implementation Order

Work through these phases in order, testing after each one:

| Phase | Kernels | What You Learn |
|-------|---------|----------------|
| **1. Element-wise ops** | `silu_kernel`, `gelu_kernel` | Load/compute/store pattern, masking |
| **2. Reductions** | `softmax_kernel`, `softmax_inplace_kernel`, `rmsnorm_kernel`, `layernorm_kernel` | `tl.sum`/`ct.sum`, `tl.max`/`ct.max`, row-wise processing |
| **3. 2D tiled matmul** | `linear_kernel_tf32` | Tiled computation, `tl.dot`/`ct.mma`, accumulation loop |
| **4. Attention** | `attention_scores_kernel`, `attention_output_kernel` | Combining dot-product and reduction patterns |
| **5. Positional encoding** | `compute_freqs_kernel` | Scalar-vector ops, `tl.cos`/`ct.cos`, concatenated stores |

Test after each phase:

```bash
# Quick unit test -- Triton (from inside the template folder)
cd glm_asr_triton_template && python layers.py     # Phase 1-3
cd glm_asr_triton_template && python attention.py   # Phase 4
cd glm_asr_triton_template && python rope.py        # Phase 5

# Quick unit test -- cuTile (from inside the template folder)
cd glm_asr_cutile_template && python layers.py     # Phase 1-3
cd glm_asr_cutile_template && python attention.py   # Phase 4
cd glm_asr_cutile_template && python rope.py        # Phase 5

# Full end-to-end test (from hw1-asr/)
./benchmark.sh glm_asr_triton_template   # Triton track
./benchmark.sh glm_asr_cutile_template   # cuTile track
```

---

## 6. How to Implement a Kernel

This is the core section. It covers **both Triton and cuTile** with parallel examples.

### 6.1 Kernel Anatomy

Every GPU kernel follows the same pattern: **identify your tile -> load -> compute -> store**.

**Triton:**

```python
@triton.jit
def my_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                # Which block am I?
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)    # My element indices
    mask = offs < n_elements                               # Bounds check
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)       # Load
    y = x * 2.0                                            # Compute
    tl.store(y_ptr + offs, y, mask=mask)                   # Store
```

**cuTile:**

```python
@ct.kernel
def my_kernel(x, output, tile_size: ct.Constant[int]):
    pid = ct.bid(0)                                        # Which block am I?
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))  # Load a tile
    result = x_tile * 2.0                                   # Compute
    ct.store(output, index=(pid,), tile=result)             # Store
```

Key differences:

| Concept | Triton | cuTile |
|---------|--------|--------|
| Block ID | `tl.program_id(0)` | `ct.bid(0)` |
| Index range | `tl.arange(0, BLOCK_SIZE)` | Implicit (via `shape` in `ct.load`) |
| Load | `tl.load(ptr + offsets, mask=..., other=0.0)` | `ct.load(array, index=(...), shape=(...))` |
| Store | `tl.store(ptr + offsets, data, mask=...)` | `ct.store(array, index=(...), tile=data)` |
| Bounds masking | Explicit `mask` parameter | Handled by cuTile runtime |
| Compile-time constant | `tl.constexpr` | `ct.Constant[type]` |

### 6.2 Worked Example: `silu_kernel`

Let's walk through implementing `silu_kernel` step by step. This is the simplest kernel -- a good first target.

**Step 1: Understand the math**

SiLU (also called Swish):

```
SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
```

It's element-wise: each output element depends on exactly one input element.

**Step 2: Read the template**

Open the template and find `silu_kernel`. You'll see:
- The function signature (inputs, outputs, sizes, block size)
- `pid = tl.program_id(0)` / `pid = ct.bid(0)` already provided
- TODO comments listing the steps

**Step 3: Figure out the data layout**

SiLU operates on a flattened 1D array. The caller (the `silu()` function below the kernel) reshapes the input to 1D, divides it into blocks of `BLOCK_SIZE=256`, and launches one kernel instance per block. Your kernel handles one block.

**Step 4: Think through the implementation**

For a 1D element-wise kernel you need to:
1. Compute which elements this block owns (using `pid` and `BLOCK_SIZE`)
2. Load those elements from input
3. Apply the math
4. Store the results to output

**Hints (Triton):**
- Use `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` to compute your offsets
- Use `mask = offs < n_elements` for bounds checking
- `tl.exp(-x)` computes the exponential
- Sigmoid is `1.0 / (1.0 + tl.exp(-x))`

**Hints (cuTile):**
- Use `ct.load(x, index=(pid,), shape=(tile_size,))` to load your tile
- `ct.exp(-x_tile)` computes the exponential
- Sigmoid is `1.0 / (1.0 + ct.exp(-x_tile))`

**Step 5: Verify**

After implementing, test with:

```bash
cd glm_asr_triton_template && python layers.py   # Should print SiLU output shape
cd glm_asr_cutile_template && python layers.py   # If you choose cutile track, run this
```

Compare your output shape and values with what the example produces. If both print the same shape without errors, your kernel is likely correct.

> **The example implementation (`glm_asr_triton_example/layers.py` or `glm_asr_cutile_example/layers.py`) contains the complete solution.** Use it as a reference if you get stuck, but try to implement on your own first.

### 6.3 Functions Cheat Sheet

| Operation | Triton (`tl.*`) | cuTile (`ct.*`) |
|-----------|----------------|-----------------|
| Block/program ID | `tl.program_id(axis)` | `ct.bid(axis)` |
| Index range | `tl.arange(start, end)` | N/A (implicit in `ct.load`) |
| Load | `tl.load(ptr + offs, mask=..., other=0.0)` | `ct.load(arr, index=(...), shape=(...))` |
| Store | `tl.store(ptr + offs, val, mask=...)` | `ct.store(arr, index=(...), tile=val)` |
| Zeros | `tl.zeros((M, N), dtype=tl.float32)` | `ct.zeros((M, N), dtype=ct.float32)` |
| Sum | `tl.sum(x, axis=0)` | `ct.sum(x)` |
| Max | `tl.max(x, axis=0)` | `ct.max(x)` |
| Exp | `tl.exp(x)` | `ct.exp(x)` |
| Sqrt | `tl.sqrt(x)` or `tl.rsqrt(x)` | `ct.sqrt(x)` |
| Cos / Sin | `tl.cos(x)` / `tl.sin(x)` | `ct.cos(x)` / `ct.sin(x)` |
| Tanh | `tl.libdevice.tanh(x)` | `ct.tanh(x)` |
| Dot product (2D) | `tl.dot(a, b)` | `ct.mma(a, b, acc)` or `ct.matmul(a, b)` |
| Cast | `x.to(tl.float32)` | `ct.astype(x, ct.tfloat32)` |
| Reshape | N/A (use indexing) | `ct.reshape(tile, (new_shape))` |
| Transpose | N/A (use indexing) | `ct.transpose(tile)` |
| Concatenate | N/A (store twice) | `ct.cat((a, b), dim)` |
| Ceiling div | `triton.cdiv(a, b)` | `ct.cdiv(a, b)` |

---

## 7. Testing Your Implementation

### 7.1 Unit Tests

Each template file has a `__main__` block with basic shape tests:

```bash
# Triton track (from inside glm_asr_triton_template/)
cd glm_asr_triton_template
python layers.py       # Tests RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax, MLP
python attention.py    # Tests basic, causal, masked, and GQA attention
python rope.py         # Tests RoPE cos/sin computation and rotation

# cuTile track (from inside glm_asr_cutile_template/)
cd glm_asr_cutile_template
python layers.py       # Tests RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax, MLP
python attention.py    # Tests basic, causal, masked, and GQA attention
python rope.py         # Tests RoPE cos/sin computation and rotation
```

What success looks like (Triton):

```
Testing Triton Layers...
=== RMSNorm ===
Input: torch.Size([2, 16, 256]) -> Output: torch.Size([2, 16, 256])
...
All Triton layers working!
```

What success looks like (cuTile):

```
Testing Pure CuTile Layers...
=== RMSNorm ===
Input: (2, 16, 256) -> Output: (2, 16, 256)
...
All Pure CuTile layers working!
```

If a kernel is not yet implemented (`pass`), the test may print zero-filled outputs or crash. Implement kernels in the recommended order (Section 5.2) and test after each one.

### 7.2 Full Benchmark

The end-to-end test runs the complete ASR pipeline:

```bash
# Triton track
./benchmark.sh glm_asr_triton_template
# cuTile track
./benchmark.sh glm_asr_cutile_template
```

Annotated output:

```
Loading model weights...             # Downloads ~4GB on first run
Running warmup...                    # JIT compilation (slow first time)
Running benchmark (3 runs)...
  Run 1: 2.34s
  Run 2: 1.89s
  Run 3: 1.87s
Average inference time: 2.03s
Transcription: CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS
Reference:     CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS
Accuracy: 100.0%
Status: PASS
```

### 7.3 Performance Profiling

Get per-operator timing breakdown:

```bash
# If you select Triton track
./benchmark_detailed.sh glm_asr_triton_template
# Or cuTile track
./benchmark_detailed.sh glm_asr_cutile_template
```

Compare your performance against the example baseline:

```bash
# If you select Triton track
./benchmark_detailed.sh glm_asr_triton_example
# Or cuTile track
./benchmark_detailed.sh glm_asr_cutile_example
```

Profile specific operators:

```bash
./benchmark_detailed.sh --attention-only
./benchmark_detailed.sh --linear-only
```

Generate an Nsight Systems profile:

```bash
# If you select Triton track
./benchmark_detailed.sh glm_asr_triton_template --nsys
# Or cuTile track
./benchmark_detailed.sh glm_asr_cutile_template --nsys
```

---

## 8. Grading Criteria

| Criteria | Points | What It Means |
|----------|:------:|---------------|
| **Correctness** | **60** | Transcription accuracy > 80% (word-level match against reference) |
| **Performance** | **30** | Total inference time faster than the example baseline |
| **Code quality** | **10** | Clean, readable kernel code; appropriate use of Triton/cuTile idioms |

**Correctness**: Run `./benchmark.sh glm_asr_triton_template` (or `./benchmark.sh glm_asr_cutile_template` for cuTile). If `Accuracy: 100.0%` and `Status: PASS`, you get full correctness marks. Any accuracy above 80% passes.

**Performance**: Compare your timing against the example baseline (`./benchmark.sh glm_asr_triton_example` or `./benchmark.sh glm_asr_cutile_example`). Faster = more marks. Consider:
- Using fused kernels (e.g., `swiglu_fused_kernel` or `linear_gelu_kernel` are already provided)
- Tuning `BLOCK_SIZE` / tile dimensions
- Enabling fused mode: set `MLP.FUSED = True` in `layers.py`

**Code quality**: Write clear kernel code. Comment non-obvious logic. Use meaningful variable names.

---

## 9. Troubleshooting Guide

### 9.1 Common Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `triton.compiler.errors.CompilationError` (Triton) | Syntax error in kernel or wrong Triton API usage | Check `tl.*` function names and signatures. Verify constexpr parameters. |
| `ct.CompilationError` or cuTile build error (cuTile) | Syntax error in kernel or wrong cuTile API usage | Check `ct.*` function names. Verify `ct.Constant` types match. |
| `RuntimeError: CUDA error: an illegal memory access` | Out-of-bounds memory access in kernel | Check mask logic (Triton: `offs < n_elements`). Verify stride calculations. For cuTile, check `index` and `shape` in `ct.load`/`ct.store`. |
| `NaN` or `inf` in output | Division by zero or missing epsilon | Add `eps` to denominators: `sqrt(var + eps)`. Check softmax stability (subtract max first). |
| Empty transcription `""` | All kernels produce zeros (still have `pass`) | Implement all required kernels. Check that `pass` is removed. |
| `RuntimeError: shape mismatch` | Kernel writes wrong number of elements | Verify `BLOCK_SIZE` matches hidden dimension. Check 2D strides. |
| `AssertionError` in model | Attention output has wrong shape | Check `attention_output_kernel` stores to correct offsets. Verify head_dim. |
| `Accuracy: 0.0%` | Kernels run but produce wrong values | Compare kernel output against example. Check math formulas. |
| `ModuleNotFoundError: No module named 'triton'` (Triton) | Environment not activated | Run `source utils/setup-triton.sh` from repo root. |
| `ModuleNotFoundError: No module named 'cuda.tile'` (cuTile) | Environment not activated | Run `source utils/setup-cutile.sh` from repo root. |
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size. Close other GPU processes. Check for memory leaks in kernel. |
| `triton.runtime.errors.OutOfResources` (Triton) | BLOCK_SIZE too large for GPU shared memory | Reduce BLOCK_SIZE. Use smaller tile dimensions. |

### 9.2 Debugging Strategies

**Strategy 1: Compare with the example implementation**

The example folders contain working code. Diff your template against the example:

```bash
# Triton track
diff glm_asr_triton_template/layers.py glm_asr_triton_example/layers.py

# cuTile track
diff glm_asr_cutile_template/layers.py glm_asr_cutile_example/layers.py
```

Focus on the kernel functions -- the class code is identical.

**Strategy 2: Print tensor shapes**

Add prints in the class methods (not inside `@triton.jit` or `@ct.kernel` functions -- those can't print):

```python
# Triton -- in the RMSNorm.__call__ method, BEFORE the kernel launch:
print(f"x_flat.shape = {x_flat.shape}, hidden_size = {self.hidden_size}")
print(f"x_flat[:5] = {x_flat[0, :5]}")

# cuTile -- same idea, but arrays are CuPy:
print(f"x_flat.shape = {x_flat.shape}, hidden_size = {self.hidden_size}")
print(f"x_flat[:5] = {x_flat[0, :5]}")
```

**Strategy 3: Test with known inputs**

Create simple test cases where you can verify the answer by hand:

```python
# Triton -- all-ones input -> RMSNorm should return the weight vector
import torch
x = torch.ones(1, 256, device="cuda")
norm = RMSNorm(256)
result = norm(x)
# RMS of all-ones = sqrt(1.0) = 1.0, so result should equal norm.weight
print(f"Expected: {norm.weight[:5]}")
print(f"Got:      {result[0, :5]}")
```

```python
# cuTile -- same test with CuPy
import cupy as cp
x = cp.ones((1, 256), dtype=cp.float32)
norm = RMSNorm(256)
result = norm(x)
print(f"Expected: {norm.weight[:5]}")
print(f"Got:      {result[0, :5]}")
```

**Strategy 4: Check for NaN propagation**

If you get NaN in the final output, track where it first appears:

```python
# Triton -- in model.py, temporarily add after each layer:
hidden_states = layer(hidden_states, ...)
if torch.isnan(hidden_states).any():
    print(f"NaN detected after layer {i}!")
    break

# cuTile -- same idea with CuPy:
hidden_states = layer(hidden_states, ...)
if cp.isnan(hidden_states).any():
    print(f"NaN detected after layer {i}!")
    break
```

### 9.3 Performance Debugging

Use `benchmark_detailed.sh` to find your bottleneck:

```bash
# Triton track
./benchmark_detailed.sh glm_asr_triton_template
# cuTile track
./benchmark_detailed.sh glm_asr_cutile_template
```

This shows timing for each operator (attention, linear, normalization, etc.). Focus optimization on the slowest operators first -- typically `linear` (matrix multiplication) dominates.

**Enable fused kernels** for better performance. In your `layers.py`:
- Set `MLP.FUSED = True` to use fused SwiGLU (already implemented)
- Set `EncoderMLP.FUSED = True` to use fused Linear+GELU (already implemented)

**Tune BLOCK_SIZE**: Experiment with different tile dimensions. Common good values:
- Element-wise: 256 or 1024
- Matmul: TILE_M=64, TILE_N=64, TILE_K=32 (default, usually optimal)

---

## 10. Rules

1. **Must use Triton or cuTile only** -- do not use PyTorch/CuPy operators inside kernels (e.g., no `torch.matmul`, `cp.matmul` as substitutes for your kernel code).

2. **May use examples as reference** -- the example implementations are explicitly provided for you to study and learn from.

3. **May refactor and fuse kernels** -- you are not limited to filling in the existing TODOs. You can combine multiple operations into a single kernel for better performance. The fused kernels (`swiglu_fused_kernel`, `linear_gelu_kernel`) in the template show how this is done.

4. **Do NOT modify**: `model.py`, `weight_loader.py`, `conv.py`.

---

## 11. References

**Triton:**
- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorial: Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton Tutorial: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton Tutorial: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

**cuTile:**
- [CuPy Documentation](https://docs.cupy.dev/)

**Papers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE (RoFormer)](https://arxiv.org/abs/2104.09864)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
