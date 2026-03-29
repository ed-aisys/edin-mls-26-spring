# Detailed Benchmark Results: H200 Warmup-Corrected Run

This file records the final usable H200 detailed benchmark that replaced the
older warmup-contaminated component numbers.

---

## Environment

| Parameter | Value |
|-----------|-------|
| **Date** | Fri 27 Mar 04:54 GMT 2026 |
| **Hostname** | saxa.inf.ed.ac.uk |
| **GPU** | NVIDIA H200 MIG 3g.71gb |
| **Branch** | ankush |
| **Commit** | `8202f955868cb0c76b8b71f45adc794a1f3c10ee` |
| **Job ID** | `2236079` |
| **Script** | `hw1-asr/benchmark_detailed.py` |
| **Args** | `--runs 5 --warmup-benchmarks 1 --benchmark-repeats 3` |
| **Compared Configs** | `glm_asr_triton_template` vs `glm_asr_triton_example` |

---

## Why this file exists

The older H200 detailed benchmark in [benchmarks.md](./benchmarks.md) included
first-use Triton compilation and warmup in the timed component measurements.
That inflated the encoder and prefill stages and made the component breakdown
less representative of steady-state behavior.

This file records the corrected run that:

- discards one full benchmark pass for warmup
- averages three measured full benchmark passes
- preserves both raw logs and a parsed summary in the repo

---

## Local Artifacts

Pulled from the cluster into `logs/detailed_2236079/`:

- [job_metadata.txt](../logs/detailed_2236079/job_metadata.txt)
- [benchmark_detailed_slurm_2236079.log](../logs/detailed_2236079/benchmark_detailed_slurm_2236079.log)
- [glm_asr_triton_template_detailed.log](../logs/detailed_2236079/glm_asr_triton_template_detailed.log)
- [glm_asr_triton_example_detailed.log](../logs/detailed_2236079/glm_asr_triton_example_detailed.log)
- [comparison_summary.json](../logs/detailed_2236079/comparison_summary.json)
- [comparison_summary.md](../logs/detailed_2236079/comparison_summary.md)

---

## Final Parsed Summary

| Component | Template (ms) | Baseline (ms) | Speedup |
|-----------|--------------:|--------------:|--------:|
| Audio Encoder | 36.53 | 167.02 | 4.57x |
| Multi-modal Projector | 0.14 | 1.11 | 7.93x |
| Decoder (Prefill) | 12.97 | 20.51 | 1.58x |
| Decoder (50 decode steps) | 580.45 | 817.09 | 1.41x |
| **TOTAL (estimated for 50 tokens)** | **630.09** | **1005.73** | **1.60x** |

Template-only component share:

| Component | Time (ms) | % of Total |
|-----------|----------:|-----------:|
| Audio Encoder | 36.53 | 5.8% |
| Multi-modal Projector | 0.14 | 0.0% |
| Decoder (Prefill) | 12.97 | 2.1% |
| Decoder (50 decode steps) | 580.45 | 92.1% |
| **TOTAL** | **630.09** | **100%** |

Average template decode step: `580.45 / 50 = 11.61ms`.
Average baseline decode step: `817.09 / 50 = 16.34ms`.

---

## Stable Measured Passes

Template (`glm_asr_triton_template`), measured passes after warmup discard:

| Pass | Audio Encoder | Projector | Prefill | Decode Step Avg |
|------|--------------:|----------:|--------:|----------------:|
| Measured 1/3 | 36.61 | 0.14 | 13.01 | 11.64 |
| Measured 2/3 | 36.47 | 0.14 | 12.98 | 11.61 |
| Measured 3/3 | 36.51 | 0.14 | 12.93 | 11.58 |

Baseline (`glm_asr_triton_example`), measured passes after warmup discard:

| Pass | Audio Encoder | Projector | Prefill | Decode Step Avg |
|------|--------------:|----------:|--------:|----------------:|
| Measured 1/3 | 167.09 | 1.12 | 20.49 | 16.32 |
| Measured 2/3 | 167.00 | 1.10 | 20.15 | 16.56 |
| Measured 3/3 | 166.97 | 1.10 | 20.88 | 16.14 |

These are the numbers that should be used for:

- the corrected H200 component breakdown in the report
- the H200 per-operator comparison table

---

## Notes

- This is an H200 MIG detailed benchmark, not an RTX 5090 benchmark.
- `benchmark_detailed.py` measures isolated forward passes and estimates decode
  cost using 50 single-step decode iterations.
- The larger end-to-end H200 speedup still depends on KV-cached generation in
  the full student benchmark path.
