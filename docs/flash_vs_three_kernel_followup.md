# Flash vs Three-Kernel Follow-up

This note tracks the clean Section 5.3 comparison between the current deployed
attention path and the historical materialized-score path.

## Why this benchmark exists

Section 5.3 currently explains the qualitative benefit of FlashAttention-style
attention, but it does not have a clean same-codebase latency comparison against
the original materialized-score path. Earlier branch/file-swap tests were not
good report evidence because they mixed in incompatible attention dispatch
changes.

The goal of this follow-up is to measure the attention change cleanly by
benchmarking the current template codebase with two modes:

- `GLM_ASR_ATTENTION_MODE=auto`
  Current deployed path: flash kernel for `seq_q > 4`, SDPA fallback for tiny
  KV-cached decode steps.
- `GLM_ASR_ATTENTION_MODE=three_kernel`
  Historical materialized-score path reintroduced inside the current
  `glm_asr_triton_template/attention.py`, so GQA expansion, KV-cache logic, and
  the rest of the pipeline stay unchanged.

## Source changes used for this benchmark

- `hw1-asr/glm_asr_triton_template/attention.py`
  Added `GLM_ASR_ATTENTION_MODE` with `auto`, `three_kernel`, and `sdpa_all`
  modes.
- `hw1-asr/benchmark_student.py`
  Prints the active attention mode in benchmark logs.
- `hw1-asr/benchmark_detailed.py`
  Prints the active attention mode in benchmark logs.
- `hw1-asr/flash_vs_three_kernel_job.sh`
  SLURM job that runs the comparison and writes a markdown summary artifact.

## Intended report insertion point

- `report/report_no_abstract.tex`
- Section 5.3: `FlashAttention-Style Attention`

The expected use is to replace the current “direct comparison is not feasible”
wording with a clean same-codebase comparison once the benchmark completes.

## Expected artifacts

The SLURM job writes:

- raw end-to-end logs for `auto` and `three_kernel`
- raw detailed benchmark logs for `auto` and `three_kernel`
- `comparison_summary.json`
- `comparison_summary.md`
- a report-facing copy in `docs/flash_vs_three_kernel_job_<jobid>.md`

This file should be updated with the final job ID and parsed results after the
benchmark finishes.
