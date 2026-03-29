# RTX 5090 Confirmation Benchmark Results

This file is the canonical source for the later RTX 5090 confirmation rerun used
by the submission branch headline RTX 5090 number and the cross-GPU comparison.

## Evidence Chain

| Item | Value |
|------|-------|
| Script | `hw1-asr/benchmark_student.py` |
| Date | `Mon 17 Mar 11:46 UTC 2026` |
| Host | `5448230d8a1e (RunPod)` |
| GPU | NVIDIA GeForce RTX 5090 |
| Compared configs | `glm_asr_triton_template` vs `glm_asr_triton_example` |
| Evidence type | Embedded console transcript in this file |

No standalone raw console log for this confirmation rerun is currently archived
in the repo. The console transcript preserved here is the source of record.

## Canonical Use

Use this file for:

- the RTX 5090 headline number (`100.4 ms`)
- the RTX 5090 row of the cross-GPU comparison

Do not use this file for the historical progression chain. That chain uses the
older development chain preserved in `benchmarks_history.md`.

## Environment

| Parameter | Value |
|-----------|-------|
| Architecture | Blackwell (`sm_120`) |
| VRAM | 31.4 GB |
| SMs | 170 |
| Driver | 580.126.20 |
| CUDA | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu130 |
| Triton | 3.6.0 |
| NumPy | 2.3.5 |
| Transformers | 5.3.0 |
| Generate function | `_generate_v8b` for our implementation, `generate` for baseline |

## Our Implementation: Five Independent Benchmark Invocations

Each invocation runs `2` warmup iterations followed by `5` timed iterations.

| Invocation | Time (ms) | Stddev (ms) | Speed (ms/tok) | Tokens | Accuracy | Status |
|------------|----------:|------------:|---------------:|-------:|---------:|--------|
| 1 | 100.4 | 0.3 | 7.72 | 13 | 100% | PASS |
| 2 | 100.5 | 0.1 | 7.73 | 13 | 100% | PASS |
| 3 | 100.5 | 0.3 | 7.73 | 13 | 100% | PASS |
| 4 | 100.4 | 0.5 | 7.72 | 13 | 100% | PASS |
| 5 | 100.4 | 0.1 | 7.72 | 13 | 100% | PASS |
| **Mean** | **100.4** | **0.3** | **7.72** | **13** | **100%** | **PASS** |

## Baseline Confirmation Benchmark

The baseline transcript preserved here is a single benchmark invocation with `2`
warmup iterations followed by `5` timed iterations.

| Implementation | Time (ms) | Stddev (ms) | Speed (ms/tok) | Tokens | Accuracy | Status |
|----------------|----------:|------------:|---------------:|-------:|---------:|--------|
| `glm_asr_triton_example` | 262.2 | 0.1 | 20.17 | 13 | 100% | PASS |

## End-to-End Comparison

| Metric | Our Template | Baseline | Relative |
|--------|-------------:|---------:|---------:|
| Time | 100.4ms | 262.2ms | 2.61x faster |
| Speed | 7.72 ms/tok | 20.17 ms/tok | 2.61x faster |
| Tokens | 13 | 13 | same |
| Accuracy | 100% | 100% | same |

## Transcript Extract

### Our Template

```text
Time: 100.4ms (+/- 0.3ms)
Tokens: 13
Speed: 7.72ms/token
Accuracy: 100.0%
Status: PASS
```

### Baseline

```text
Time: 262.2ms (+/- 0.1ms)
Tokens: 13
Speed: 20.17ms/token
Accuracy: 100.0%
Status: PASS
```
