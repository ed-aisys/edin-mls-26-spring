# Section 5.3 Attention Benchmark Results

This file is the canonical source for Section 5.3
(`FlashAttention-Style Attention`) in `report/report_no_abstract.tex`.

## Evidence Chain

| Item | Value |
|------|-------|
| Canonical script | `hw1-asr/flash_vs_three_kernel_job.sh` |
| Job ID | `2238022` |
| Date | `Sun 29 Mar 03:18 BST 2026` |
| Compared modes | `GLM_ASR_ATTENTION_MODE=auto` vs `GLM_ASR_ATTENTION_MODE=three_kernel` |
| Raw bundle | `../logs/h200_attention_2238022/` |

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

Important clarification: `auto` is not "flash everywhere." It is the runtime
mode the actual model uses in deployment:

- if `seq_q <= 4`, it routes to PyTorch SDPA to reduce launch overhead in the
  tiny KV-cached decode regime
- otherwise it routes to the fused Triton flash-attention kernel

So the Section 5.3 benchmark compares the real deployed mixed dispatch against
the reintroduced 3-kernel/materialized-score path, not "pure flash-only"
against 3-kernel.

## Raw Artifacts

- [job_metadata.txt](../logs/h200_attention_2238022/job_metadata.txt)
- [auto_student.log](../logs/h200_attention_2238022/auto_student.log)
- [three_kernel_student.log](../logs/h200_attention_2238022/three_kernel_student.log)
- [auto_detailed.log](../logs/h200_attention_2238022/auto_detailed.log)
- [three_kernel_detailed.log](../logs/h200_attention_2238022/three_kernel_detailed.log)
- [comparison_summary.json](../logs/h200_attention_2238022/comparison_summary.json)
- [comparison_summary.md](../logs/h200_attention_2238022/comparison_summary.md)

## Final Same-Codebase Results

### End-to-End Student Benchmark

This job used the updated end-to-end methodology:

- each benchmark pass used `2` warmup runs and `5` timed runs
- `1` full benchmark pass was discarded
- the final number averages `3` measured benchmark passes

| Mode | Mean (ms) | Std (ms) | Accuracy |
|------|----------:|---------:|---------:|
| Current deployed path (`auto`) | 210.9 | 2.1 | 100.0% |
| Reintroduced 3-kernel path (`three_kernel`) | 212.0 | 2.0 | 100.0% |

### Warmup-Corrected Detailed Benchmark

This job also ran the detailed benchmark with `--runs 5 --warmup-benchmarks 1
--benchmark-repeats 3`.

| Component | Current path (ms) | Three-kernel path (ms) | Relative |
|-----------|------------------:|-----------------------:|---------:|
| Audio Encoder | 36.19 | 36.18 | 1.00x |
| Multi-modal Projector | 0.14 | 0.15 | 1.07x |
| Decoder (Prefill) | 12.82 | 13.36 | 1.04x |
| Decoder (50 decode steps) | 575.89 | 591.41 | 1.03x |
| **TOTAL (estimated for 50 tokens)** | **625.04** | **641.09** | **1.03x** |

## Report Guidance

The rerun now gives a consistent same-codebase result:

- the end-to-end student benchmark is slightly faster for the current deployed
  path (`210.9 ms` vs `212.0 ms`)
- the warmup-corrected detailed benchmark is also lower for the current path
  (`625.04 ms` vs `641.09 ms`)

Section 5.3 should still present this as a modest improvement rather than a
dramatic standalone headline result. The evidence-backed statement is that, on
H200 MIG 3g.71gb, the current deployed attention strategy is slightly but
consistently faster than the reintroduced 3-kernel/materialized-score path in
the same codebase.

## Superseded Earlier Same-Codebase Run

The previous same-codebase job `2237998` remains archived in
`../logs/h200_attention_2237998/`, but it used a weaker end-to-end methodology
and produced a mixed result. Job `2238022` supersedes it as the canonical
Section 5.3 evidence.

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
