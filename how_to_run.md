# How to Run: GLM-ASR Triton Kernel Project

## Prerequisites

### Hardware
- NVIDIA GPU with Tensor Core support (tested on RTX 5090, Blackwell sm_120)
- 32GB+ VRAM recommended (model is ~4.3GB, plus activations and KV cache)

### Software
- Linux (tested on Ubuntu with kernel 6.8.0)
- NVIDIA Driver: 580.x or compatible
- CUDA Toolkit: 13.0

---

## Python Dependencies

Install Python 3.12+ and the following packages:

```bash
# PyTorch with CUDA 13.0 support
pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# Triton (GPU compiler for custom kernels)
pip install triton==3.6.0

# HuggingFace ecosystem (model loading)
pip install transformers==5.3.0
pip install huggingface_hub==1.6.0
pip install safetensors==0.7.0

# Numerical and audio
pip install numpy==2.3.5
pip install soundfile==0.13.1
```

### All at once (pip)
```bash
pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install triton==3.6.0 transformers==5.3.0 huggingface_hub==1.6.0 safetensors==0.7.0 numpy==2.3.5 soundfile==0.13.1
```

---

## cuBLAS Fix (if needed)

If you see `CUBLAS_STATUS_INVALID_VALUE` errors, a pip-installed `nvidia-cublas`
may conflict with the system CUDA libraries:

```bash
pip list | grep nvidia-cublas
# If version doesn't match your CUDA toolkit:
pip uninstall nvidia-cublas
# PyTorch will then fall back to the system cuBLAS
```

---

## Disk Space

The HuggingFace model (~4.3GB) downloads on first run. If your root filesystem
has limited space, redirect the cache:

```bash
export HF_HOME=/path/to/large/disk/.hf_cache
```

---

## Running the Benchmark

```bash
cd hw1-asr

# Set model cache location (if needed)
export HF_HOME=/workspace/.hf_cache

# Quick correctness test (1 warmup, 1 timed run)
python benchmark_student.py glm_asr_triton_template --warmup 1 --runs 1

# Full benchmark (2 warmup, 5 timed runs)
python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5

# Compare against the example baseline
python benchmark_student.py glm_asr_triton_example --warmup 2 --runs 5

# Detailed per-operator profiling
python benchmark_detailed.py glm_asr_triton_template
```

### Expected Output
```
Time: ~110ms (+/- 0.3ms)
Tokens: 13
Speed: ~8.5ms/token
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

---

## Running Attention Parity Tests

Validates the Triton Flash Attention kernel against a PyTorch reference:

```bash
cd hw1-asr
python -m glm_asr_triton_template.attention
```

This runs 17 deterministic test cases covering basic, causal, masked, GQA,
encoder/decoder shapes, and single-token decode.

---

## Project Structure

```
hw1-asr/
  glm_asr_triton_template/   # Student implementation (our code)
    layers.py                 # 6 Triton kernels + layer classes
    attention.py              # Flash Attention kernel + 3 legacy kernels
    rope.py                   # RoPE kernel
    __init__.py               # Backend/fusion configuration
    model.py                  # Model architecture (DO NOT MODIFY)
    conv.py                   # Conv1D layers (DO NOT MODIFY)
    weight_loader.py          # Weight loading (DO NOT MODIFY)
  glm_asr_triton_example/     # Reference baseline implementation
  benchmark_student.py        # End-to-end benchmark script
  benchmark_detailed.py       # Per-operator profiling
  test_audio.wav              # Test audio file (3.5s)
```
