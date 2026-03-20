# Ablation Testing Guide

## Quick Start

```bash
# On the Edinburgh teaching cluster (hastings login node)
cd ~/edin-mls-26-spring/hw1-asr

# Option 1: Interactive (you wait for it)
srun -p Teaching -w saxa --gres gpu:3g.71gb:1 --mem=32G --time=02:00:00 bash -c "
    export PATH=/home/\$USER/.conda/envs/mls/bin:\$PATH
    export HF_HOME=/home/\$USER/.cache/huggingface
    python3 ablation_test.py
"

# Option 2: Batch job with email notification
# Edit ablation_job.sh to set your email, then:
sbatch ablation_job.sh
```

## What It Does

Runs 22 configurations, each modifying a single variable from the optimized baseline:

1. **Precision**: fp32 vs fp16
2. **Fusion**: MLP fusion on/off, RoPE fusion on/off
3. **Backend**: cuBLAS vs Triton matmul
4. **SDPA fallback**: off, threshold=1, 4 (default), 8, 16
5. **Encoder attention tiles**: 128x128 (default), 128x64, 64x64, 64x32
6. **Decoder attention tiles**: 128x64 (default), 64x64, 64x32, 32x32
7. **num_stages**: 1, 2 (default), 3
8. **num_warps**: 4, 8 (default), 16
9. **Matmul tiles**: 128x128x64 (default), 128x128x32, 128x64x32, 64x64x32

For each test:
- Original files are backed up
- Exact code change is recorded as a unified diff
- Triton cache is cleared (forces recompilation)
- `benchmark_student.py` runs (2 warmup + 5 timed iterations)
- Individual run times, mean, stddev, accuracy are captured
- Original files are restored

## Output Files

- `ablation_results.json` — Structured results with diffs
- `ablation_results.md` — Human-readable table + detailed per-test breakdown
- `ablation_output.log` — Raw console output

## Requirements

- NVIDIA H200 MIG 3g.71gb (or any GPU — configs auto-adapt)
- `mls` conda environment with PyTorch 2.10+, Triton 3.6+
- Model weights cached in `$HF_HOME`

## Customizing

To add new tests, add entries to the `TESTS` list in `ablation_test.py`:

```python
{
    "name": "my_test",
    "desc": "What this tests",
    "changes": {
        "glm_asr_triton_template/__init__.py": [
            ("old string", "new string"),
        ],
    },
}
```

Each `(old, new)` tuple does a single `str.replace()` on the file. The script records the diff automatically.

## Previous Results

See `ablation_results.md` for H200 MIG 3g.71gb results from 2026-03-19.
Key findings:
- Fused RoPE: most impactful (+28.9ms when disabled)
- Double-buffering (nstages=2): +12.2ms when disabled
- num_warps=8 optimal: 4 adds +12.0ms, 16 adds +4.3ms
- fp16 vs fp32: only +1.5ms on H200 (vs -11.5ms on RTX 5090)
