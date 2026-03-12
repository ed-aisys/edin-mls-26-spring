# GLM-ASR Student Assignment

This assignment helps you understand GPU kernel optimization by implementing a speech recognition model using Triton and NVIDIA cuTile.

## Overview

GLM-ASR is a speech-to-text model that converts audio into text. This HW1 includes Triton and cuTile tracks (example + template) and focuses on performance optimization. **You only need to choose Triton/cuTile to complete**, which we recommand Triton for its compatability in a lot of hardwares.

What you will learn in this HW:

- **GPU kernel optimization** fundamentals
- **Writing Triton kernels** for neural network workloads
- **Writing NVIDIA cuTile kernels** as an alternative track
- **Performance optimization** techniques for GPU-based inference

## Task

### What to Do

Open the template for your track and complete the TODO sections in:

**Triton**
- `glm_asr_triton_template/attention.py`
- `glm_asr_triton_template/layers.py`
- `glm_asr_triton_template/rope.py`

**cuTile**
- `glm_asr_cutile_template/attention.py`
- `glm_asr_cutile_template/layers.py`
- `glm_asr_cutile_template/rope.py`

> [!NOTE]
> You are not limited to filling the existing TODO kernels. You may refactor and fuse kernels (for example, implement logic that currently spans multiple kernels within a single Triton/cuTile kernel).
> However, you must implement kernels using Triton/cuTile only (do not use prebuilt operator libraries such as PyTorch).

## QuickStart

### Triton Version QuickStart

From the **repository root** (one level above `hw1-asr/`):

```bash
# Installation: set up Triton environment from the repository root (one level above hw1-asr/)
source utils/setup-triton.sh

# Verify the environment works by running the reference baseline
./benchmark.sh glm_asr_triton_example

# After you fill your code in the template, run the end-to-end test
./benchmark.sh glm_asr_triton_template
```

### cuTile Version QuickStart

```bash
# Installation: set up cuTile environment from the repository root (one level above hw1-asr/)
source utils/setup-cutile.sh

# Verify the environment works by running the reference baseline
./benchmark.sh glm_asr_cutile_example

# After you fill your code in the template, run the end-to-end test
./benchmark.sh glm_asr_cutile_template
```

## Description

> **New here?** Read the **[Detailed Student Guide (GUIDE.md)](GUIDE.md)** for a step-by-step walkthrough, kernel patterns, and troubleshooting tips.

### Directory Structure

```
student_version/
├── glm_asr_triton_example/     # Reference: Triton baseline (Torch + Triton)
├── glm_asr_triton_template/    # YOUR WORK OPTION 1: Complete the TODOs here (Triton)
├── glm_asr_cutile_example/     # Reference: Example baseline (Initial CuPy + cuTile)
├── glm_asr_cutile_template/    # YOUR WORK OPTION 2: Complete the TODOs here (cuTile)
├── glm_asr_scratch/            # Reference: PyTorch baseline
├── demo.py                    # Streamlit interactive demo
├── benchmark.sh               # Shell wrapper for benchmark_student.py
├── benchmark_student.py       # Python benchmark script
├── benchmark_detailed.sh      # Shell wrapper for benchmark_detailed.py
├── benchmark_detailed.py      # Detailed operator profiling
├── test_audio.wav             # Test audio file
└── test_audio.txt             # Expected transcription
```

### Reference Implementations

| Version                  | Description                                                                   |
| ------------------------ | ----------------------------------------------------------------------------- |
| `glm_asr_scratch`        | PyTorch Reference: explicitly shows model structure (for understanding only)  |
| `glm_asr_triton_example` | Triton Baseline: use this as your reference if you chose the **Triton** track |
| `glm_asr_cutile_example` | cuTile Baseline: use this as your reference if you chose the **cuTile** track |

> [!IMPORTANT]
> Match your reference to your track:
> - **Triton track** → study `glm_asr_triton_example/` as your baseline
> - **cuTile track** → study `glm_asr_cutile_example/` as your baseline

### Student Templates

| Version                   | Description                    |
| ------------------------- | ------------------------------ |
| `glm_asr_triton_template` | Triton template (TODO kernels) |
| `glm_asr_cutile_template` | cuTile template (TODO kernels) |

> [!IMPORTANT]
> **Minimum optimization requirements (choose your track: Triton or cuTile).**  
> Your submission should include **at least these 3 optimizations** (we will check them during grading/report review):
>
> 1. **Adjust tile/block sizes**  
>    - Tune key tiling hyperparameters (e.g., `BLOCK_M/BLOCK_N/BLOCK_K`, `num_warps`, `num_stages` in Triton; the corresponding tile shapes / scheduling params in cuTile).  
>    - Show you tried **at least 2–3 configurations** and picked the best for your GPU.
>
> 2. **Kernel fusion (at least 1 fused kernel)**  
>    - Fuse two or more ops that are currently separate.  
>    - The goal is to reduce intermediate reads/writes and kernel launch overhead.
>
> 3. **FlashAttention-style attention**  
>    - Implement a **FlashAttention** (or FlashAttention-like) kernel for the self-attention path (streaming softmax with good memory efficiency, blockwise QK^T, numerically stable softmax, then multiply by V).  
>    - You may refactor `attention.py` as needed, but must keep reuslts correctness.

### Key Files Explained

- **layers.py**: Basic neural network layers (Linear, LayerNorm, MLP)
- **attention.py**: Self-attention mechanism
- **rope.py**: Rotary Position Embedding (RoPE) for position encoding
- **model.py**: Full model architecture (AudioEncoder, TextDecoder)
- **weight_loader.py**: Loads pre-trained weights (no changes needed)

## Quick Start

Choose a track below (Triton first).

### Environment Setup

From the repo root, source the setup script for your chosen track:

```bash
# Triton track
source utils/setup-triton.sh
# Optional: demo deps (if not already installed)
# pip install transformers huggingface_hub streamlit soundfile scipy

# cuTile track
source utils/setup-cutile-fix.sh
```

`setup-cutile-fix.sh` installs common ML tooling used by the demo:
`transformers`, `huggingface_hub`, `streamlit`, `soundfile`, `scipy`.

### Triton Track

1. Test reference implementation:

```bash
./benchmark.sh glm_asr_triton_example
```

2. Test your implementation:

```bash
./benchmark.sh glm_asr_triton_template
```

3. Check performance:

```bash
./benchmark_detailed.sh glm_asr_triton_template
```

4. Try interactive demo:

```bash
streamlit run demo.py
```

### cuTile Track

1. Test reference implementation:

```bash
./benchmark.sh glm_asr_cutile_example
```

2. Test your implementation:

```bash
./benchmark.sh glm_asr_cutile_template
```

3. Check performance:

```bash
./benchmark_detailed.sh glm_asr_cutile_template
```

4. Try interactive demo:

```bash
streamlit run demo.py
```

### Expected Output

```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

## Benchmark Tools

There are two ways to run benchmarks: **Shell scripts** (convenience wrappers) and **Python scripts** (direct execution).

### Shell Scripts (Recommended for beginners)

Shell scripts provide user-friendly wrappers with folder validation and help messages.

```bash
# Show available folders
./benchmark.sh

# Basic correctness test (Triton)
./benchmark.sh glm_asr_triton_template

# Basic correctness test (cuTile)
./benchmark.sh glm_asr_cutile_template

# Test baselines
./benchmark.sh glm_asr_triton_example
./benchmark.sh glm_asr_cutile_example

# Detailed performance analysis
./benchmark_detailed.sh glm_asr_triton_template
./benchmark_detailed.sh glm_asr_cutile_template

# Detailed performance analysis (baselines)
./benchmark_detailed.sh glm_asr_triton_example
./benchmark_detailed.sh glm_asr_cutile_example

# Optional profiling knobs
./benchmark_detailed.sh glm_asr_triton_template --runs 5
./benchmark_detailed.sh glm_asr_triton_template --seq-len 512
./benchmark_detailed.sh glm_asr_triton_template --audio /path/to/test_audio.wav

# Generate Nsight Systems profile
./benchmark_detailed.sh glm_asr_triton_template --nsys
```

### Python Scripts (More control)

Python scripts offer more options and can be used directly without shell.

```bash
# Basic benchmark with options
python benchmark_student.py glm_asr_triton_template
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3
python benchmark_student.py glm_asr_cutile_template
python benchmark_student.py glm_asr_cutile_example --warmup 1 --runs 3
# Detailed profiling
python benchmark_detailed.py glm_asr_triton_template
python benchmark_detailed.py glm_asr_triton_example
python benchmark_detailed.py glm_asr_cutile_template
python benchmark_detailed.py glm_asr_cutile_example
```

### Streamlit Demo

Interactive web UI for testing transcription:

```bash
streamlit run demo.py
```

Select from: `Triton Example (Baseline)`, `Triton Template`, `CuTile Example (Baseline)`, `CuTile Template`, `Scratch (PyTorch)`

### Check the WebUI of your slurm job on your PC

First, check the port from the output of `streamlit run demo.py`.

Then, you are using slurm, run `show_tunnel.sh` on your **login node/head node**. The script will scan your running jobs to get the node name (the first running job).

```bash
bash show_tunnel.sh <port>
```

In the output of `show_tunnel.sh`, you will get the instruction of running a specific command on your local PC and open a website.

## Tips

1. **Study the references**:
   - `glm_asr_triton_example/` - Triton baseline, easier to map to template
   - `glm_asr_cutile_example/` - Simple baseline, easier to understand

2. **Test incrementally**: After implementing each layer, run the benchmark to check correctness.

3. **Use CuPy + Triton** (CuTile) / **Use Torch + Triton** (Triton): The implementation uses CuPy for CuTile kernels and Torch + Triton for Triton kernels. Key functions:
   - `cp.matmul()` - Matrix multiplication
   - `cp.einsum()` - Einstein summation
   - `cp.exp()`, `cp.sqrt()` - Element-wise operations

4. **Check shapes**: Print tensor shapes when debugging:

   ```python
   print(f"x.shape = {x.shape}")
   ```

5. **Understand the data flow**:

   ```
   Audio (wav) → AudioEncoder → Projector → TextDecoder → Text
   ```

## Common Errors

| Error | Solution |
|-------|----------|
| Shape mismatch | Check input/output dimensions |
| NaN values | Check for division by zero, use epsilon |
| Empty transcription | Verify attention mask and position IDs |
| Out of memory | Reduce batch size or sequence length |

## Reference

- [Triton Documentation](https://triton-lang.org/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)

## Questions?

If you encounter issues:
1. Check the example implementation first
2. Verify your tensor shapes match expected dimensions
3. Ask during office hours

Good luck!
