# Flash Attention vs Materialized-Score Attention

- Job ID: `2238638`
- GPU: H200 MIG 3g.71gb on `saxa`
- Target benchmark doc: `benchmarks/benchmarks_attention.md`
- Config A: `GLM_ASR_ATTENTION_MODE=auto`
  Current deployed path: flash kernel for seq_q > 4, SDPA fallback for seq_q <= 4.
- Config B: `GLM_ASR_ATTENTION_MODE=three_kernel`
  Historical materialized-score path: same current codebase, but attention materializes the score matrix instead of using flash.

## End-to-End Student Benchmark

| Mode | Mean (ms) | Std (ms) | Accuracy | Relative |
|------|----------:|---------:|---------:|---------:|
| Current deployed path | 210.6 | 0.3 | 100.0% | 1.00x |
| Historical materialized-score path | 280.8 | 0.7 | 100.0% | 1.33x slower vs current |

## Detailed Component Benchmark

| Component | Current path (ms) | Three-kernel path (ms) | Relative |
|-----------|------------------:|-----------------------:|---------:|
| Audio Encoder | 36.51 | 63.25 | 1.73x |
| Multi-modal Projector | 0.14 | 0.14 | 1.00x |
| Decoder (Prefill) | 12.96 | 17.55 | 1.35x |
| Decoder (50 decode steps) | 581.82 | 742.55 | 1.28x |
| TOTAL (estimated for 50 tokens) | 631.42 | 823.49 | 1.30x |

## Raw Artifacts

- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238638/job_metadata.txt`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238638/auto_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238638/auto_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238638/three_kernel_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238638/three_kernel_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238638/comparison_summary.json`
