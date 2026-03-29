# Flash Attention vs Materialized-Score Attention

- Job ID: `2237998`
- GPU: H200 MIG 3g.71gb on `saxa`
- Target report section: `Section 5.3 (FlashAttention-Style Attention)`
- Config A: `GLM_ASR_ATTENTION_MODE=auto`
  Current deployed path: flash kernel for seq_q > 4, SDPA fallback for seq_q <= 4.
- Config B: `GLM_ASR_ATTENTION_MODE=three_kernel`
  Historical materialized-score path: same current codebase, but attention materializes the score matrix instead of using flash.

## End-to-End Student Benchmark

| Mode | Mean (ms) | Std (ms) | Accuracy | Relative |
|------|----------:|---------:|---------:|---------:|
| Current deployed path | 219.8 | 2.1 | 100.0% | 1.00x |
| Historical materialized-score path | 207.0 | 0.9 | 100.0% | 0.94x slower vs current |

## Detailed Component Benchmark

| Component | Current path (ms) | Three-kernel path (ms) | Relative |
|-----------|------------------:|-----------------------:|---------:|
| Audio Encoder | 36.33 | 36.33 | 1.00x |
| Multi-modal Projector | 0.15 | 0.15 | 1.00x |
| Decoder (Prefill) | 12.86 | 13.37 | 1.04x |
| Decoder (50 decode steps) | 575.64 | 601.65 | 1.05x |
| TOTAL (estimated for 50 tokens) | 624.98 | 651.51 | 1.04x |

## Raw Artifacts

- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2237998/job_metadata.txt`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2237998/auto_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2237998/auto_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2237998/three_kernel_student.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2237998/three_kernel_detailed.log`
- `edin-mls-26-spring/hw1-asr/attention_mode_runs/flash_vs_three_kernel_2237998/comparison_summary.json`

