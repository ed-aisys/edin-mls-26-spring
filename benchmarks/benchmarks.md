# H200 End-to-End Benchmark Results

This file is the canonical source for the H200 MIG 3g.71gb end-to-end numbers
used on the submission branch.

## Evidence Chain

| Item | Value |
|------|-------|
| Script | `hw1-asr/benchmark_student.py` |
| Raw artifact | `../logs/h200_e2e_2225992/benchmark_raw_output.txt` |
| Date | `Mon 16 Mar 21:20:06 GMT 2026` |
| Node | `saxa.inf.ed.ac.uk` |
| Job ID | `2225992` |
| Compared configs | `glm_asr_triton_template` vs `glm_asr_triton_example` |

## Canonical Use

Use this file for:

- the H200 headline result (`204.8 ms`, baseline `464.1 ms`)
- the H200 end-to-end timing table
- the end-to-end comparison against the baseline
- the H200 row of the cross-GPU comparison

Do not use this file for H200 component timings. Those live in
`benchmarks_detailed.md`.

## Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA H200 MIG 3g.71gb |
| Architecture | Hopper (`sm_90`) |
| VRAM | 69.8 GB |
| SMs | 60 |
| Driver | 580.126.09 |
| CUDA | 13.0 |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cu130 |
| Triton | 3.6.0 |
| NumPy | 2.4.3 |
| Transformers | 5.3.0 |
| Generate function | `_generate_v8b` for our implementation, `generate` for baseline |

## Our Implementation: Five Independent Benchmark Invocations

Each invocation runs `2` warmup iterations followed by `5` timed iterations.

| Invocation | Time (ms) | Stddev (ms) | Speed (ms/tok) | Tokens | Accuracy | Status |
|------------|----------:|------------:|---------------:|-------:|---------:|--------|
| 1 | 203.4 | 1.6 | 15.65 | 13 | 100% | PASS |
| 2 | 204.0 | 1.6 | 15.69 | 13 | 100% | PASS |
| 3 | 203.4 | 2.1 | 15.64 | 13 | 100% | PASS |
| 4 | 209.4 | 1.7 | 16.11 | 13 | 100% | PASS |
| 5 | 203.9 | 1.4 | 15.68 | 13 | 100% | PASS |
| **Mean** | **204.8** | **1.7** | **15.75** | **13** | **100%** | **PASS** |

## Baseline End-to-End Benchmark

The baseline transcript preserved in the raw log is a single benchmark invocation
with `2` warmup iterations followed by `5` timed iterations.

| Implementation | Time (ms) | Stddev (ms) | Speed (ms/tok) | Tokens | Accuracy | Status |
|----------------|----------:|------------:|---------------:|-------:|---------:|--------|
| `glm_asr_triton_example` | 464.1 | 0.4 | 35.70 | 13 | 100% | PASS |

## End-to-End Comparison

| Metric | Our Template | Baseline | Relative |
|--------|-------------:|---------:|---------:|
| Time | 204.8ms | 464.1ms | 2.27x faster |
| Speed | 15.75 ms/tok | 35.70 ms/tok | 2.27x faster |
| Tokens | 13 | 13 | same |
| Accuracy | 100% | 100% | same |

## Notes

- The raw H200 artifact also contains an older detailed benchmark section with
  warmup-contaminated component timings. Those numbers are historical only and
  should not be reused in the report.
- Older historical docs sometimes quote `204.6 ms` on H200. The canonical report
  number on this branch is `204.8 ms`, taken from the five-invocation mean above.
