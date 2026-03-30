# Benchmark README

This folder is the canonical entry point for benchmark evidence used on the
submission branch.

For project-level setup and reproduction instructions, start with
`../PROJECT_README.md`. This file stays focused on benchmark provenance.

## Canonical Benchmark Docs

| File | Purpose | Evidence Type | Supports |
|------|---------|---------------|----------|
| `benchmarks.md` | Final H200 MIG 3g.71gb end-to-end benchmark | Raw log-backed (`logs/h200_e2e_2225992/benchmark_raw_output.txt`) | Headline H200 result, end-to-end comparison, H200 cross-GPU row |
| `benchmarks_detailed.md` | Final warmup-corrected H200 component benchmark | Raw log-backed (`logs/h200_detailed_2236079/`) | H200 component breakdown, per-operator comparison |
| `benchmarks_ablation.md` | H200 ablation benchmark summary | Generated-artifact-backed (`hw1-asr/ablation_results.{json,md}`, `hw1-asr/ablation_output.log`) | H200 ablation summary and discussion |
| `benchmarks_5090.md` | RTX 5090 confirmation rerun | Embedded console transcript in doc | RTX 5090 cross-GPU row, headline RTX 5090 number |
| `benchmarks_history.md` | Original RTX 5090 development benchmark chain | History-backed, extracted from pre-cleanup docs | Historical progression chain, rejected-optimization summary |
| `benchmarks_attention.md` | Attention-backend comparison evidence | Raw log-backed (`logs/h200_attention_2238637/`, `logs/h200_attention_2238638/`) | Same-codebase `auto` vs `three_kernel` comparison |
| `benchmark_provenance.md` | Benchmark provenance matrix | Consolidated index | All canonical benchmark-backed claims |

## Raw Artifact Locations

| Path | What It Contains |
|------|------------------|
| `../logs/h200_e2e_2225992/benchmark_raw_output.txt` | Raw H200 end-to-end transcript for `glm_asr_triton_template` plus the H200 baseline benchmark |
| `../logs/h200_detailed_2236079/` | Warmup-corrected H200 detailed benchmark bundle (`job_metadata`, SLURM log, template log, baseline log, parsed summaries) |
| `../logs/h200_attention_2238637/` | First clean same-codebase H200 attention-backend comparison bundle (`auto` vs `three_kernel`) |
| `../logs/h200_attention_2238638/` | Second clean same-codebase H200 attention-backend comparison bundle (`auto` vs `three_kernel`) |
| `../hw1-asr/ablation_results.json` | Structured H200 ablation results |
| `../hw1-asr/ablation_results.md` | Human-readable H200 ablation results |
| `../hw1-asr/ablation_output.log` | Raw console output for the successful H200 ablation run |

## Benchmark Scripts

| Script | Status | What It Measures | Outputs | Canonical Doc |
|--------|--------|------------------|---------|---------------|
| `../hw1-asr/benchmark_student.py` | Authoritative | End-to-end inference on a chosen implementation folder | Console transcript only unless wrapped | `benchmarks.md`, `benchmarks_5090.md`, `benchmarks_attention.md` |
| `../hw1-asr/benchmark.sh` | Wrapper | Shell wrapper for `benchmark_student.py` | Console transcript only unless redirected | Same as `benchmark_student.py` |
| `../hw1-asr/benchmark_detailed.py` | Authoritative | Isolated component profiling and 50-step decode estimate | Console transcript only unless wrapped | `benchmarks_detailed.md` |
| `../hw1-asr/benchmark_detailed.sh` | Wrapper | Shell wrapper for `benchmark_detailed.py` | Console transcript only unless redirected | `benchmarks_detailed.md` |
| `../hw1-asr/benchmark_detailed_job.sh` | Authoritative batch wrapper | H200 detailed/component benchmark with discarded warmup benchmark pass | `benchmark_runs/detailed_<jobid>/...` and parsed summaries | `benchmarks_detailed.md` |
| `../hw1-asr/ablation_test.py` | Authoritative | H200 ablation study across 22 configurations | `ablation_results.json`, `ablation_results.md`, `ablation_output.log` | `benchmarks_ablation.md` |
| `../hw1-asr/ablation_job.sh` | Authoritative batch wrapper | Batch execution of `ablation_test.py` on a Slurm GPU node | `ablation_slurm_<jobid>.{log,err}` plus generated ablation files | `benchmarks_ablation.md` |
| `../hw1-asr/flash_vs_three_kernel_job.sh` | Authoritative for the attention-backend comparison | Same-codebase comparison of `GLM_ASR_ATTENTION_MODE=auto` vs `three_kernel` | `attention_mode_runs/flash_vs_three_kernel_<jobid>/...` | `benchmarks_attention.md` |
| `../hw1-asr/flash_ablation_test.sh` | Historical exploratory | Flash/SDPA ablation on H200 by forcing all attention through SDPA | Local log files inside `hw1-asr/` | Historical note inside `benchmarks_attention.md` |
| `../hw1-asr/nsys_profile.sh` | Supplementary | Nsight Systems profiling | `.nsys-rep` / `.sqlite` outputs | Not a benchmark source of record |
| `../hw1-asr/ncu_profile.py` | Supplementary | Nsight Compute profiling | Nsight Compute output | Not a benchmark source of record |
| `../hw1-asr/ncu_job.sh` | Supplementary | Batch wrapper for Nsight Compute profiling | Batch logs / profiler outputs | Not a benchmark source of record |

## Reproduction Helper Files

| File | Purpose |
|------|---------|
| `../PROJECT_README.md` | Project-level reproduction guide for the submission branch |
| `../hw1-asr/setup_saxa_env.sh` | Small checkout-relative runtime environment helper used by the canonical benchmark wrappers |
| `../hw1-asr/run_report_benchmarks.sh` | Thin wrapper that runs or submits the exact report benchmark entry points |

## Removed or Superseded Sources

These files were intentionally removed or sidelined during cleanup because they
duplicated benchmark numbers without being the canonical evidence path:

- old report variants (`report.tex`, `report_with_abstract.tex`)
- verbose tutorial/reference/code-explained docs
- exploratory branch-swap flash-vs-main scripts
- duplicate benchmark-history notes once their numbers were extracted into
  `benchmarks_history.md`

## How To Look Up A Canonical Benchmark Claim

1. Open `benchmark_provenance.md`.
2. Find the benchmark claim you care about.
3. Follow the canonical doc link in that row.
4. If you need the raw artifact, use the raw path listed in the same row.

The rule on this branch is simple: every canonical benchmark number should
resolve to exactly one canonical benchmark doc and one traceable evidence path.
