# Flash Attention vs Materialized-Score Attention

- Job ID: `2238637`
- GPU: H200 MIG 3g.71gb on `saxa`
- Target benchmark doc: `benchmarks/benchmarks_attention.md`
- Config A: `GLM_ASR_ATTENTION_MODE=auto`
  Current deployed path: flash kernel for seq_q > 4, SDPA fallback for seq_q <= 4.
- Config B: `GLM_ASR_ATTENTION_MODE=three_kernel`
  Historical materialized-score path: same current codebase, but attention materializes the score matrix instead of using flash.

## End-to-End Student Benchmark

| Mode | Mean (ms) | Std (ms) | Accuracy | Relative |
|------|----------:|---------:|---------:|---------:|
| Current deployed path | 214.1 | 0.6 | 100.0% | 1.00x |
| Historical materialized-score path | 291.9 | 0.7 | 100.0% | 1.36x slower vs current |

## Detailed Component Benchmark

| Component | Current path (ms) | Three-kernel path (ms) | Relative |
|-----------|------------------:|-----------------------:|---------:|
| Audio Encoder | 36.47 | 63.26 | 1.73x |
| Multi-modal Projector | 0.14 | 0.14 | 1.00x |
| Decoder (Prefill) | 13.15 | 17.59 | 1.34x |
| Decoder (50 decode steps) | 578.67 | 745.70 | 1.29x |
| TOTAL (estimated for 50 tokens) | 628.43 | 826.69 | 1.32x |

## Raw Artifacts

- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238637/job_metadata.txt`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238637/auto_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238637/auto_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238637/three_kernel_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238637/three_kernel_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238637/comparison_summary.json`
