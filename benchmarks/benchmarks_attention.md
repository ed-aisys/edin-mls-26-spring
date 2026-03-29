# Section 5.3 Attention Benchmark Results

This file is the canonical source for Section 5.3
(`FlashAttention-Style Attention`) in `report/report_no_abstract.tex`.

## Evidence Chain

| Item | Value |
|------|-------|
| Canonical script | `hw1-asr/flash_vs_three_kernel_job.sh` |
| Job ID | `2237998` |
| Date | `Sun 29 Mar 02:33 BST 2026` |
| Branch / commit | `ankush` / `8202f955868cb0c76b8b71f45adc794a1f3c10ee` |
| Compared modes | `GLM_ASR_ATTENTION_MODE=auto` vs `GLM_ASR_ATTENTION_MODE=three_kernel` |
| Raw bundle | `../logs/h200_attention_2237998/` |

## What The Canonical Script Measures

`flash_vs_three_kernel_job.sh` benchmarks the current template codebase in two
modes:

- `auto`
  current deployed attention path
  (`flash` for larger `seq_q`, `SDPA` fallback for tiny decode steps)
- `three_kernel`
  reintroduced historical materialized-score attention path inside the same
  current codebase

That design keeps GQA expansion, KV cache logic, and the rest of the pipeline
fixed while changing only the attention backend.

## Raw Artifacts

- [job_metadata.txt](../logs/h200_attention_2237998/job_metadata.txt)
- [auto_student.log](../logs/h200_attention_2237998/auto_student.log)
- [three_kernel_student.log](../logs/h200_attention_2237998/three_kernel_student.log)
- [auto_detailed.log](../logs/h200_attention_2237998/auto_detailed.log)
- [three_kernel_detailed.log](../logs/h200_attention_2237998/three_kernel_detailed.log)
- [comparison_summary.json](../logs/h200_attention_2237998/comparison_summary.json)
- [comparison_summary.md](../logs/h200_attention_2237998/comparison_summary.md)

## Final Same-Codebase Results

### End-to-End Student Benchmark

This job ran one end-to-end benchmark invocation per mode, with `2` warmup runs
and `5` timed runs inside each invocation.

| Mode | Mean (ms) | Std (ms) | Accuracy |
|------|----------:|---------:|---------:|
| Current deployed path (`auto`) | 219.8 | 2.1 | 100.0% |
| Reintroduced 3-kernel path (`three_kernel`) | 207.0 | 0.9 | 100.0% |

### Warmup-Corrected Detailed Benchmark

This job also ran the detailed benchmark with `--runs 5 --warmup-benchmarks 1
--benchmark-repeats 3`.

| Component | Current path (ms) | Three-kernel path (ms) | Relative |
|-----------|------------------:|-----------------------:|---------:|
| Audio Encoder | 36.33 | 36.33 | 1.00x |
| Multi-modal Projector | 0.15 | 0.15 | 1.00x |
| Decoder (Prefill) | 12.86 | 13.37 | 1.04x |
| Decoder (50 decode steps) | 575.64 | 601.65 | 1.05x |
| **TOTAL (estimated for 50 tokens)** | **624.98** | **651.51** | **1.04x** |

## Report Guidance

The two measurements do not tell the same story:

- the end-to-end student benchmark measured `219.8 ms` for the current deployed
  path and `207.0 ms` for the reintroduced 3-kernel path
- the warmup-corrected detailed benchmark measured lower isolated component time
  for the current deployed path (`624.98 ms` vs `651.51 ms`)

Because of that, Section 5.3 should not present a single universal
"flash-vs-3-kernel speedup" number. The evidence-backed statement is that the
same-codebase comparison is now reproducible, and it produced mixed end-to-end
versus operator-level results on H200.

## Historical Evidence Preserved During Cleanup

Before the canonical same-codebase benchmark existed, the repo had two older
attention experiments:

- `hw1-asr/flash_ablation_test.sh`
  clean historical ablation that forced all attention through PyTorch SDPA
  instead of the current flash path
- older branch/file-swap tests
  (`flash_vs_main_test.sh`, `flash_quick_test.sh`, `flash_detailed_test.sh`)
  that swapped in another branch's `attention.py`

Only the first of those was a clean single-variable experiment, and it compared
the current flash path against all-SDPA, not against the original 3-kernel path.
The branch/file-swap tests were not valid report evidence because they mixed in
incompatible attention-dispatch changes.
