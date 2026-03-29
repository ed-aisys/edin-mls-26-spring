# Flash Attention vs Materialized-Score Attention

- Job ID: `2238022`
- GPU: H200 MIG 3g.71gb on `saxa`
- Target report section: `Section 5.3 (FlashAttention-Style Attention)`
- Config A: `GLM_ASR_ATTENTION_MODE=auto`
  Current deployed path: flash kernel for seq_q > 4, SDPA fallback for seq_q <= 4.
- Config B: `GLM_ASR_ATTENTION_MODE=three_kernel`
  Historical materialized-score path: same current codebase, but attention materializes the score matrix instead of using flash.

## End-to-End Student Benchmark

| Mode | Mean (ms) | Std (ms) | Accuracy | Relative |
|------|----------:|---------:|---------:|---------:|
| Current deployed path | 210.9 | 2.1 | 100.0% | 1.00x |
| Historical materialized-score path | 212.0 | 2.0 | 100.0% | 1.01x slower vs current |

## Detailed Component Benchmark

| Component | Current path (ms) | Three-kernel path (ms) | Relative |
|-----------|------------------:|-----------------------:|---------:|
| Audio Encoder | 36.19 | 36.18 | 1.00x |
| Multi-modal Projector | 0.14 | 0.15 | 1.07x |
| Decoder (Prefill) | 12.82 | 13.36 | 1.04x |
| Decoder (50 decode steps) | 575.89 | 591.41 | 1.03x |
| TOTAL (estimated for 50 tokens) | 625.04 | 641.09 | 1.03x |

## Raw Artifacts

- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238022/job_metadata.txt`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238022/auto_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238022/auto_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238022/three_kernel_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238022/three_kernel_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2238022/comparison_summary.json`

