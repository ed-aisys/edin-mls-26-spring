# Step 1 — Adjust tile / block sizes (Triton)

The assignment asks for **three** optimizations. **This document is only for the first one:**

| Step | Optimization | Covered here? |
|------|----------------|---------------|
| **1** | **Adjust tile/block sizes** — tune `BLOCK_M` / `BLOCK_N` / `BLOCK_K`, `num_warps`, `num_stages` (Triton) or analogous params (cuTile). Show **≥ 2–3** configs and the best on **your** GPU. | **Yes — read below** |
| 2 | Kernel fusion | Not here — do after tile tuning |
| 3 | FlashAttention-style attention | Not here — do after tile tuning |

Stay focused: **tile tuning only** in this phase. Do not mix in fusion or attention changes until step 1 is done and `./benchmark.sh` still passes.

---

## What you are proving (grading / report)

For **step 1** you should be able to show:

1. You **tried at least 2–3 different** tiling setups (different `BLOCK_*` and/or `num_warps` / `num_stages`).
2. You **measured** them on your GPU (timings from [`benchmark_tile_sizes.py`](benchmark_tile_sizes.py) or equivalent).
3. You **chose one** setup per kernel path (or per layer class) that is fastest on your hardware and **wired it into** [`glm_asr_triton_template/layers.py`](glm_asr_triton_template/layers.py).

The script [`benchmark_tile_sizes.py`](benchmark_tile_sizes.py) prints many rows automatically; you only need to **highlight 2–3 meaningful comparisons** in your write-up (e.g. default vs two alternatives, then the winner).

---

## Step-by-step workflow

### Step 1.1 — Environment and GPU

- Use a machine with **CUDA** (`nvidia-smi` works).
- From the **repository root** (parent of `hw1-asr/`):

  ```bash
  source utils/setup-triton.sh
  ```

### Step 1.2 — Run the tile microbenchmark

From **`hw1-asr/`** (required for imports):

```bash
cd hw1-asr
python benchmark_tile_sizes.py
```

Optional flags:

```bash
python benchmark_tile_sizes.py --help
python benchmark_tile_sizes.py --quick --skip-gelu --skip-linear-gelu   # shorter run
```

The script benchmarks kernel **patterns** that match `layers.py` (matmul, fused Linear+GELU, SwiGLU, element-wise GELU). **Pick the winner by lowest time (ms)**; TFLOPS uses padded sizes so comparisons across `BLOCK_*` are fair.

### Step 1.3 — Record your comparisons

For your report or notes, write down **at least 2–3 configurations** you actually compared, for example:

| Config | BLOCK_M×N×K | warps | stages | Time (ms) on your GPU |
|--------|-------------|-------|--------|------------------------|
| A (template default) | … | … | … | … |
| B | … | … | … | … |
| C (best) | … | … | … | … |

Copy numbers from the script output.

### Step 1.4 — Apply the best values in `layers.py`

Only **change tiling / launch constants** — no fusion or attention edits in this step.

| Where you saw the win | Edit in `glm_asr_triton_template/layers.py` |
|------------------------|---------------------------------------------|
| Matmul section | `class Linear`: `TILE_M`, `TILE_N`, `TILE_K`, `NUM_WARPS`, `NUM_STAGES` |
| Fused Linear+GELU | `class EncoderMLP`: same fields |
| SwiGLU fused | `class MLP`: same fields |
| Element-wise GELU | `GELU_BLOCK_SIZE`, `GELU_NUM_WARPS` (module-level, near imports) |

Different classes may use **different** winning tiles; that is fine.

### Step 1.5 — Verify correctness

```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_template
```

You must still see **PASS** and correct transcription. If not, revert or fix before moving to optimization 2.

### Step 1.6 — (Optional) End-to-end timing

```bash
./benchmark_detailed.sh glm_asr_triton_template
```

Useful to see how tile changes affect the full model; microbenchmarks alone are not the only story.

---

## What each benchmark section measures

| Section | Kernel pattern | Typical use |
|--------|----------------|-------------|
| MATMUL | `linear_kernel_tf32` | `Linear` |
| FUSED LINEAR + GELU | `linear_gelu_kernel` | `EncoderMLP` fused path |
| SWIGLU FUSED | `swiglu_fused_kernel` | Decoder `MLP` |
| ELEMENT-WISE GELU | `gelu_kernel` | `gelu()` |

**Out of scope for this file:** attention tile sizes in `attention.py` — that belongs to **optimization 3**, not step 1.

---

## cuTile track (same assignment step, different code)

If you use **cuTile**, tune the **tile shapes / scheduling parameters** in your cuTile kernels the same way: try **≥ 2–3** configs on your GPU, pick the best, document it. This repo’s `benchmark_tile_sizes.py` is **Triton**-oriented; use your own timings or `benchmark_detailed.py` for cuTile.

---

## Troubleshooting

- **`CUDA is required`**: Run on a GPU node.
- **`ModuleNotFoundError: glm_asr_triton_template`**: Run from `hw1-asr/`.
- **Some rows `FAILED`**: Some tile/warp combinations are invalid on some GPUs; skip them and keep valid rows.
- **Microbench vs full model**: Tile wins in isolation can look smaller in `benchmark_detailed`; both are still valid for step 1.
