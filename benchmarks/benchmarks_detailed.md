# H200 Detailed Component Benchmark Results

This file is the canonical source for the warmup-corrected H200 component
numbers used on the submission branch.

## Evidence Chain

| Item | Value |
|------|-------|
| Script | `hw1-asr/benchmark_detailed.py` |
| Batch runner | `hw1-asr/benchmark_detailed_job.sh` |
| Job ID | `2236079` |
| Date | `Fri 27 Mar 04:54 GMT 2026` |
| Compared configs | `glm_asr_triton_template` vs `glm_asr_triton_example` |
| Args | `--runs 5 --warmup-benchmarks 1 --benchmark-repeats 3` |
| Raw bundle | `../logs/h200_detailed_2236079/` |

## Canonical Use

Use this file for:

- the H200 component breakdown
- the per-operator comparison
- the root-cause discussion that cites the corrected H200 component timings

## Why This Replaced The Older Detailed Benchmark

The older H200 detailed numbers were inflated by first-use Triton compilation and
warmup effects. Job `2236079` fixes that by discarding one full benchmark pass
and averaging three measured benchmark passes, while still using `5` timed runs
per component inside each pass.

## Raw Artifacts

- [job_metadata.txt](../logs/h200_detailed_2236079/job_metadata.txt)
- [benchmark_detailed_slurm_2236079.log](../logs/h200_detailed_2236079/benchmark_detailed_slurm_2236079.log)
- [glm_asr_triton_template_detailed.log](../logs/h200_detailed_2236079/glm_asr_triton_template_detailed.log)
- [glm_asr_triton_example_detailed.log](../logs/h200_detailed_2236079/glm_asr_triton_example_detailed.log)
- [comparison_summary.json](../logs/h200_detailed_2236079/comparison_summary.json)
- [comparison_summary.md](../logs/h200_detailed_2236079/comparison_summary.md)

## Final Parsed Summary

| Component | Template (ms) | Baseline (ms) | Speedup |
|-----------|--------------:|--------------:|--------:|
| Audio Encoder | 36.53 | 167.02 | 4.57x |
| Multi-modal Projector | 0.14 | 1.11 | 7.93x |
| Decoder (Prefill) | 12.97 | 20.51 | 1.58x |
| Decoder (50 decode steps) | 580.45 | 817.09 | 1.41x |
| **TOTAL (estimated for 50 tokens)** | **630.09** | **1005.73** | **1.60x** |

## Template-Only Share Used For The H200 Component Breakdown

| Component | Time (ms) | % of Total |
|-----------|----------:|-----------:|
| Audio Encoder | 36.53 | 5.8% |
| Multi-modal Projector | 0.14 | 0.0% |
| Decoder (Prefill) | 12.97 | 2.1% |
| Decoder (50 decode steps) | 580.45 | 92.1% |
| **TOTAL** | **630.09** | **100%** |

Average template decode step: `580.45 / 50 = 11.61 ms`

Average baseline decode step: `817.09 / 50 = 16.34 ms`

## Stable Measured Passes After Warmup Discard

Template (`glm_asr_triton_template`):

| Pass | Audio Encoder | Projector | Prefill | Decode Step Avg |
|------|--------------:|----------:|--------:|----------------:|
| Measured 1/3 | 36.61 | 0.14 | 13.01 | 11.64 |
| Measured 2/3 | 36.47 | 0.14 | 12.98 | 11.61 |
| Measured 3/3 | 36.51 | 0.14 | 12.93 | 11.58 |

Baseline (`glm_asr_triton_example`):

| Pass | Audio Encoder | Projector | Prefill | Decode Step Avg |
|------|--------------:|----------:|--------:|----------------:|
| Measured 1/3 | 167.09 | 1.12 | 20.49 | 16.32 |
| Measured 2/3 | 167.00 | 1.10 | 20.15 | 16.56 |
| Measured 3/3 | 166.97 | 1.10 | 20.88 | 16.14 |

## Notes

- This is an H200 MIG component benchmark, not an RTX 5090 benchmark.
- `benchmark_detailed.py` measures isolated forward passes and estimates decode
  cost using 50 single-step decode iterations.
- The larger end-to-end H200 speedup still depends on KV-cached generation in
  the full student benchmark path.
