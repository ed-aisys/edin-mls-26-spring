# Attention Backend Benchmark Results

This file is the canonical source for the same-codebase attention-backend
comparison on the submission branch.

## Evidence Chain

| Item | Value |
|------|-------|
| Canonical script | `hw1-asr/flash_vs_three_kernel_job.sh` |
| Clean rerun jobs | `2238637`, `2238638` |
| Date | `Mon 30 Mar 08:56-09:08 BST 2026` |
| Commit | `7cf521a3456797332a34c647360d9efba5b9d3be` |
| Compared modes | `GLM_ASR_ATTENTION_MODE=auto` vs `GLM_ASR_ATTENTION_MODE=three_kernel` |
| Raw bundles | `../logs/h200_attention_2238637/`, `../logs/h200_attention_2238638/` |

## Clean Reproducibility Verification

Two identical reruns were executed from a clean anonymous submission-branch
checkout at commit `7cf521a3456797332a34c647360d9efba5b9d3be`:

- clean rerun `2238637`
- clean rerun `2238638`

These jobs were submitted from fresh cluster worktrees of the pushed submission
branch, not from a dirty development checkout. They agree closely with each
other and supersede the earlier dirty-checkout result that had previously been
treated as canonical.

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

So this benchmark compares the real deployed mixed dispatch against
the reintroduced 3-kernel/materialized-score path, not "pure flash-only"
against 3-kernel.

## Raw Artifacts

- [2238637 job_metadata.txt](../logs/h200_attention_2238637/job_metadata.txt)
- [2238637 auto_student.log](../logs/h200_attention_2238637/auto_student.log)
- [2238637 three_kernel_student.log](../logs/h200_attention_2238637/three_kernel_student.log)
- [2238637 auto_detailed.log](../logs/h200_attention_2238637/auto_detailed.log)
- [2238637 three_kernel_detailed.log](../logs/h200_attention_2238637/three_kernel_detailed.log)
- [2238637 comparison_summary.json](../logs/h200_attention_2238637/comparison_summary.json)
- [2238637 comparison_summary.md](../logs/h200_attention_2238637/comparison_summary.md)
- [2238638 job_metadata.txt](../logs/h200_attention_2238638/job_metadata.txt)
- [2238638 auto_student.log](../logs/h200_attention_2238638/auto_student.log)
- [2238638 three_kernel_student.log](../logs/h200_attention_2238638/three_kernel_student.log)
- [2238638 auto_detailed.log](../logs/h200_attention_2238638/auto_detailed.log)
- [2238638 three_kernel_detailed.log](../logs/h200_attention_2238638/three_kernel_detailed.log)
- [2238638 comparison_summary.json](../logs/h200_attention_2238638/comparison_summary.json)
- [2238638 comparison_summary.md](../logs/h200_attention_2238638/comparison_summary.md)

## Final Same-Codebase Results

### End-to-End Student Benchmark

This job used the updated end-to-end methodology:

- each benchmark pass used `2` warmup runs and `5` timed runs
- `1` full benchmark pass was discarded
- the final number averages `3` measured benchmark passes

| Job | Auto (ms) | Three-kernel (ms) | Relative | Accuracy |
|-----|----------:|------------------:|---------:|---------:|
| `2238637` | 214.1 | 291.9 | 1.36x slower | 100.0% |
| `2238638` | 210.6 | 280.8 | 1.33x slower | 100.0% |

### Warmup-Corrected Detailed Benchmark

This job also ran the detailed benchmark with `--runs 5 --warmup-benchmarks 1
--benchmark-repeats 3`.

| Job | Auto total (ms) | Three-kernel total (ms) | Prefill (ms) | Decode-50 (ms) | Relative |
|-----|----------------:|------------------------:|-------------:|---------------:|---------:|
| `2238637` | 628.43 | 826.69 | 13.15 vs 17.59 | 578.67 vs 745.70 | 1.32x slower |
| `2238638` | 631.42 | 823.49 | 12.96 vs 17.55 | 581.82 vs 742.55 | 1.30x slower |

## Report Guidance

The two clean reruns now give a consistent same-codebase result:

- the end-to-end student benchmark is consistently faster for the current
  deployed path (`210.6-214.1 ms` vs `280.8-291.9 ms`)
- the warmup-corrected detailed benchmark is also much lower for the current
  path (`628.43-631.42 ms` vs `823.49-826.69 ms`)

The evidence-backed statement is now stronger: on H200 MIG 3g.71gb, the current
deployed mixed attention strategy is substantially faster than the reintroduced
3-kernel/materialized-score path in the same codebase, while preserving 100%
accuracy.

## Superseded Earlier Same-Codebase Runs

An earlier same-codebase job `2237998` used a weaker end-to-end methodology and
produced a mixed result. The later `2238022` result is also no longer treated
as canonical because it did not come from a clean committed checkout. The clean
reruns `2238637` and `2238638` supersede both of those earlier runs.

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
