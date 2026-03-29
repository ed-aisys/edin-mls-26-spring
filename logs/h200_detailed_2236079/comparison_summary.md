# H200 Detailed Component Benchmark

- Job ID: `2236079`
- GPU: H200 MIG 3g.71gb on `saxa`
- Script: `hw1-asr/benchmark_detailed.py --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3`
- Configs compared: `glm_asr_triton_template` vs `glm_asr_triton_example`

## Parsed Summary

| Component | Template (ms) | Baseline (ms) | Speedup |
|-----------|---------------:|--------------:|--------:|
| Audio Encoder | 36.53 | 167.02 | 4.57x |
| Multi-modal Projector | 0.14 | 1.11 | 7.93x |
| Decoder (Prefill) | 12.97 | 20.51 | 1.58x |
| Decoder (50 decode steps) | 580.45 | 817.09 | 1.41x |
| TOTAL (estimated for 50 tokens) | 630.09 | 1005.73 | 1.60x |

## Raw Artifacts

- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/job_metadata.txt`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/glm_asr_triton_template_detailed.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/glm_asr_triton_example_detailed.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/comparison_summary.json`

