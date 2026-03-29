# Benchmark Provenance

This file maps the canonical benchmark-backed claims on the submission branch
to their exact evidence paths.

## Evidence Type Legend

- `raw-log-backed`: backed by a raw console or batch log kept in `logs/`
- `generated-artifact-backed`: backed by generated result files committed in the repo
- `embedded-transcript-backed`: backed by a benchmark doc that preserves the console transcript directly
- `history-backed`: preserved from pre-cleanup historical docs because no standalone raw log is archived
- `pending`: benchmark script exists, but the final run has not completed yet

## Provenance Matrix

| Claim | Numbers | Script / Runner | Canonical Doc | Raw Artifact | Evidence Type |
|-------------|---------|-----------------|---------------|--------------|---------------|
| Headline H200 result | `204.8 ms`, baseline `464.1 ms`, `2.27x` | `hw1-asr/benchmark_student.py` | `benchmarks.md` | `logs/h200_e2e_2225992/benchmark_raw_output.txt` | raw-log-backed |
| H200 end-to-end five-run table | `203.4`, `204.0`, `203.4`, `209.4`, `203.9`, mean `204.8` | `hw1-asr/benchmark_student.py` | `benchmarks.md` | `logs/h200_e2e_2225992/benchmark_raw_output.txt` | raw-log-backed |
| End-to-end comparison | `204.8` vs `464.1` | `hw1-asr/benchmark_student.py` | `benchmarks.md` | `logs/h200_e2e_2225992/benchmark_raw_output.txt` | raw-log-backed |
| H200 component breakdown | `36.53`, `0.14`, `12.97`, `580.45`, `630.09` | `hw1-asr/benchmark_detailed.py` via `hw1-asr/benchmark_detailed_job.sh` | `benchmarks_detailed.md` | `logs/h200_detailed_2236079/` | raw-log-backed |
| Per-operator comparison | `36.53/167.02`, `0.14/1.11`, `12.97/20.51`, `580.45/817.09`, `630.09/1005.73` | `hw1-asr/benchmark_detailed.py` via `hw1-asr/benchmark_detailed_job.sh` | `benchmarks_detailed.md` | `logs/h200_detailed_2236079/` | raw-log-backed |
| H200 ablation summary | baseline `205.2 ms` and top 9 deltas | `hw1-asr/ablation_test.py` via `hw1-asr/ablation_job.sh` or direct run | `benchmarks_ablation.md` | `hw1-asr/ablation_results.{json,md}`, `hw1-asr/ablation_output.log` | generated-artifact-backed |
| H200 cross-GPU comparison row | `204.8` vs `464.1` | `hw1-asr/benchmark_student.py` | `benchmarks.md` | `logs/h200_e2e_2225992/benchmark_raw_output.txt` | raw-log-backed |
| RTX 5090 cross-GPU comparison row | `100.4` vs `262.2`, `2.61x` | `hw1-asr/benchmark_student.py` | `benchmarks_5090.md` | transcript embedded in `benchmarks_5090.md` | embedded-transcript-backed |
| Development progression chain | `261.3 -> 98.5` chain | historical development runs, preserved from pre-cleanup docs | `benchmarks_history.md` | extracted from pre-cleanup history at branch base `f8c2f36` | history-backed |
| Rejected optimization summary | `+18 ms`, `+0.7 ms`, `Crash`, `+6 ms`, `+13 ms`, `+3.1 ms` | historical development runs, preserved from pre-cleanup docs | `benchmarks_history.md` | extracted from pre-cleanup history at branch base `f8c2f36` | history-backed |
| Same-codebase attention comparison | end-to-end `210.9` vs `212.0`; detailed `625.04` vs `641.09` | `hw1-asr/flash_vs_three_kernel_job.sh` | `benchmarks_attention.md` | `logs/h200_attention_2238022/` | raw-log-backed |

## Important Historical Notes

- The canonical H200 end-to-end result used in the report is `204.8 ms`, taken
  from the five-invocation mean in `logs/h200_e2e_2225992/benchmark_raw_output.txt`.
  Older `204.6 ms` references in historical docs are not the canonical report number.
- The canonical H200 detailed/component result used in the report is the
  warmup-corrected job `2236079`. Older H200 detailed numbers in the repo are
  historical and should not be reused for the report.
- The appendix progression table intentionally uses the original RTX 5090
  development session (`261.3 -> 98.5`) rather than the later RTX 5090
  confirmation rerun (`262.2 -> 100.4`).
