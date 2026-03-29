# H200 Ablation Benchmark Results

This file is the canonical summary for the H200 ablation results cited in
`report/report_no_abstract.tex`.

## Evidence Chain

- Script: `hw1-asr/ablation_test.py`
- Optional batch runner: `hw1-asr/ablation_job.sh`
- Generated outputs:
  - `hw1-asr/ablation_results.json`
  - `hw1-asr/ablation_results.md`
  - `hw1-asr/ablation_output.log`

The generated result files are the authoritative evidence on this branch.

## Important Note About Earlier SLURM Logs

The files below are retained as historical failed or incomplete attempts and are
not the final evidence source:

- `hw1-asr/ablation_slurm_2228518.log`
- `hw1-asr/ablation_slurm_2228602.log`

Job `2228602` failed early due to a `total_mem` / `total_memory` property bug in
`ablation_test.py`, so it must not be used as the ablation source of record.

## Final Run Summary

| Parameter | Value |
|-----------|-------|
| Date | `2026-03-19 03:00:48` |
| GPU | H200 MIG 3g.71gb |
| Config Count | 22 |
| Baseline | `205.2 ms (+/- 0.8 ms)` |
| Accuracy | `100%` on all reported top-impact configurations in the report table |

## Top-Impact Results Used In The Report

| Change | Time (ms) | Delta (ms) |
|--------|----------:|-----------:|
| Disable fused RoPE | 234.1 | +28.9 |
| `num_stages=1` | 217.4 | +12.2 |
| `num_warps=4` | 217.2 | +12.0 |
| Decoder attention `32x32` | 212.8 | +7.6 |
| `num_warps=16` | 209.5 | +4.3 |
| SDPA fallback off | 208.6 | +3.4 |
| fp32 pipeline | 206.7 | +1.5 |
| Triton matmul | 206.2 | +1.0 |
| MLP fusion off | 204.5 | -0.7 |

## Report Use

Use this file for:

- Table `tab:ablation`
- the ablation discussion in Sections 4 and 6

If you need the full 22-test table or the exact per-test diffs, go to the
generated files in `hw1-asr/`.
