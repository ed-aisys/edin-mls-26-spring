# Running the GLM-ASR Triton Implementation

## Prerequisites

- NVIDIA GPU with Triton support (tested on RTX 5090, Blackwell architecture)
- Python 3.12 with PyTorch 2.10+ and Triton 3.6+
- ~5GB disk space for model weights

## Quick Start

```bash
cd hw1-asr

# Set HF_HOME to a directory with sufficient disk space (~5GB for model weights).
# The default ~/.cache may be on a small overlay filesystem.
export HF_HOME=/workspace/.hf_cache

# Run the benchmark
python benchmark_student.py glm_asr_triton_template
```

Expected output:
```
Time: ~188ms
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

## Unit Tests

Test individual kernel modules before running the full benchmark:

```bash
cd glm_asr_triton_template

python layers.py       # RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax
python attention.py    # Attention scores, softmax, output
python rope.py         # RoPE frequency computation
```

## Detailed Profiling

Per-operator timing breakdown:

```bash
cd hw1-asr
python benchmark_detailed.py glm_asr_triton_template
```

## Known Issues

### cuBLAS broken on RTX 5090 (driver 580 / CUDA 13.x)

All `torch.matmul`, `torch.einsum`, and `torch.mm` operations fail with
`CUBLAS_STATUS_INVALID_VALUE`. This affects any PyTorch code that uses cuBLAS
internally.

**Our workaround:** All matrix multiplications use Triton kernels instead of
cuBLAS. This is configured via:

- `Linear.BACKEND = "triton"` in `layers.py` (default was `"torch"`)
- A tiled Triton conv kernel (`conv1d_matmul_tiled_kernel`) in `conv.py` replaces
  the `torch.einsum` fallback for convolution layers whose dimensions exceed the
  single-tile limit

If cuBLAS is fixed in a future driver update, you can switch `Linear.BACKEND`
back to `"torch"` for potentially faster large matmuls.

### Shared memory limit (101KB on RTX 5090)

Triton tile sizes must fit within 101,376 bytes of shared memory. The default
pipeline staging (2 stages) doubles memory usage, so tile configurations of
128x128x64 exceed the limit.

Current tile sizes: `TILE_M=64, TILE_N=64, TILE_K=32` for Linear, MLP, and
EncoderMLP. These fit comfortably. Larger tiles may work with `num_stages=1`.

### Mel spectrogram padding

The HuggingFace processor pads mel spectrograms to 3000 frames (30 seconds),
but the test audio is only 3.5 seconds (~350 real frames). Processing the full
padded input through 32 encoder layers wastes ~8x compute and exceeds the
single-tile attention kernel limit (MAX_ATTENTION_DIM=256).

**Our fix:** `model.py:encode_audio()` trims the mel spectrogram to the actual
audio length using `input_features_mask` before passing it to the encoder.

## Files Modified (vs. reference example)

| File | Change | Why |
|------|--------|-----|
| `layers.py` | `Linear.BACKEND = "triton"` | cuBLAS broken |
| `layers.py` | Tile sizes 128/128/64 → 64/64/32 | Shared memory limit |
| `layers.py` | GELU/SiLU block size 256 → 1024 | Better GPU occupancy |
| `conv.py` | Added `conv1d_matmul_tiled_kernel` | Replace cuBLAS einsum fallback |
| `model.py` | Trim mel features in `encode_audio()` | Avoid processing padding |

## Benchmark Results

| Metric | Value |
|--------|-------|
| Inference time | 188.2ms (+/- 0.1ms) |
| Tokens generated | 13 |
| Speed | 14.48 ms/token |
| Accuracy | 100% (8/8 words) |
| GPU | NVIDIA RTX 5090 (32GB) |
