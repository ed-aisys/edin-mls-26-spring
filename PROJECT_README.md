# Final Project Reproduction Guide

This file is the project-level entry point for reproducing the benchmark
evidence used on the submission branch.

## Source Of Truth

- Benchmark provenance manual: `benchmarks/benchmarks_README.md`
- Claim-to-benchmark map: `benchmarks/benchmark_provenance.md`

This submission branch carries the maintained benchmark evidence set. The
canonical benchmark docs record the exact branch/commit used for each
historical benchmark job. For any new reproduction run, record the current
checkout explicitly with:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

## Code Entry Points

These are the main files to read if you want to understand the submitted
implementation rather than just rerun the benchmarks.

| Path | Role |
|------|------|
| `hw1-asr/glm_asr_triton_template/layers.py` | Main optimized layer implementations, generation path, fused kernels, and KV-cache-aware decode path |
| `hw1-asr/glm_asr_triton_template/attention.py` | Current deployed attention dispatch, Triton flash attention, reintroduced `three_kernel` path, and `GLM_ASR_ATTENTION_MODE` switch |
| `hw1-asr/glm_asr_triton_template/rope.py` | Fused rotary position embedding implementation |
| `hw1-asr/glm_asr_triton_example/` | Baseline Triton reference implementation used for comparison |
| `hw1-asr/benchmark_student.py` | End-to-end benchmark script used for headline timings and cross-GPU confirmation runs |
| `hw1-asr/benchmark_detailed.py` | Detailed component benchmark used for Table 3 / Table 7 style timings |
| `hw1-asr/ablation_test.py` | H200 ablation study script |
| `hw1-asr/flash_vs_three_kernel_job.sh` | Canonical attention-backend comparison batch benchmark entry point |
| `benchmarks/` | Canonical benchmark docs and provenance map |
| `logs/` | Pulled raw H200 evidence bundles kept in-repo |

## Evidence Classes

The submission evidence intentionally mixes a few evidence types. The canonical status is in
`benchmarks/benchmark_provenance.md`.

| Evidence Type | Used For |
|---------------|----------|
| `raw-log-backed` | H200 end-to-end, H200 detailed/component timings, same-codebase attention comparison |
| `generated-artifact-backed` | H200 ablation study |
| `embedded-transcript-backed` | RTX 5090 confirmation rerun |
| `history-backed` | Appendix progression table and rejected-optimization table |

## One-Time Environment Install

For a fresh machine, install the Triton/PyTorch environment first from the repo
root:

```bash
source utils/setup-triton.sh
```

That script installs the Python stack. It is the installation entry point.

## Runtime Environment Helper

For cluster or workstation reproduction, source the small runtime helper from
the repo root:

```bash
source hw1-asr/setup_saxa_env.sh ./hw1-asr/.repro_env
```

That helper does not install packages. It codifies a checkout-relative runtime
environment for benchmark runs:

- optional conda env autodiscovery via `$HOME`
- `TMPDIR`, `TMP`, `TEMP`
- `TRITON_CACHE_DIR`
- `TORCH_EXTENSIONS_DIR`
- `HF_HOME` (defaults to `$HOME/.cache/huggingface`)

Optional overrides:

- `MLS_CONDA_ENV_NAME`
- `MLS_CONDA_ENV_BIN`
- `MLS_HF_HOME`
- `MLS_REPO_DIR`
- `MLS_HW1_DIR`

## Exact Benchmark Entry Points

The thin wrapper below is the easiest way to run the canonical benchmark
entry points:

```bash
./hw1-asr/run_report_benchmarks.sh <target>
```

Supported targets:

| Target | What It Does | Canonical Doc |
|--------|---------------|---------------|
| `h200-e2e` | Runs the exact H200 end-to-end command pattern used for the report and writes per-invocation logs under `hw1-asr/benchmark_runs/report_e2e_<stamp>/` | `benchmarks/benchmarks.md` |
| `h200-detailed` | Submits `hw1-asr/benchmark_detailed_job.sh` with `sbatch` | `benchmarks/benchmarks_detailed.md` |
| `h200-ablation` | Submits `hw1-asr/ablation_job.sh` with `sbatch` | `benchmarks/benchmarks_ablation.md` |
| `h200-attention` | Submits `hw1-asr/flash_vs_three_kernel_job.sh` with `sbatch` | `benchmarks/benchmarks_attention.md` |
| `print` | Prints the exact underlying commands without running them | n/a |

## Exact Commands Behind Each Canonical Benchmark

The wrapper above intentionally stays thin. These are the underlying commands it
uses.

### H200 End-To-End

From `hw1-asr/`, the report methodology is:

- run `glm_asr_triton_template` five independent times
- each invocation uses `--warmup 2 --runs 5`
- run `glm_asr_triton_example` once with the same flags

Exact command:

```bash
python3 benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5
python3 benchmark_student.py glm_asr_triton_example --warmup 2 --runs 5
```

The wrapper repeats the template command five times and stores the logs in a
dedicated run directory.

### H200 Detailed / Component Benchmark

```bash
sbatch hw1-asr/benchmark_detailed_job.sh
```

That job runs:

```bash
python3 benchmark_detailed.py glm_asr_triton_template --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3
python3 benchmark_detailed.py glm_asr_triton_example --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3
```

### H200 Ablation

```bash
sbatch hw1-asr/ablation_job.sh
```

That job runs:

```bash
python3 ablation_test.py
```

### Attention Backend Comparison

```bash
sbatch hw1-asr/flash_vs_three_kernel_job.sh
```

For batch jobs on a specific cluster, pass resource selection via `sbatch`
arguments or the wrapper environment variables:

- `MLS_PARTITION`
- `MLS_NODELIST`
- `MLS_GRES`
- `MLS_MEM`
- `MLS_TIME`
- `MLS_MAIL_TYPE`
- `MLS_MAIL_USER`

That job compares:

- `GLM_ASR_ATTENTION_MODE=auto`
- `GLM_ASR_ATTENTION_MODE=three_kernel`

with:

```bash
python3 benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3
python3 benchmark_detailed.py glm_asr_triton_template --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3
```

### RTX 5090 Confirmation Rerun

The RTX 5090 confirmation rerun is transcript-backed in
`benchmarks/benchmarks_5090.md`. The benchmark script is still:

```bash
python3 benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5
python3 benchmark_student.py glm_asr_triton_example --warmup 2 --runs 5
```

## Where Outputs Land

| Benchmark | Output Location |
|-----------|-----------------|
| H200 end-to-end wrapper run | `hw1-asr/benchmark_runs/report_e2e_<stamp>/` |
| H200 detailed batch job | `hw1-asr/benchmark_runs/detailed_<jobid>/` |
| H200 ablation | `hw1-asr/ablation_results.{json,md}` and `hw1-asr/ablation_output.log` |
| H200 attention comparison | `hw1-asr/attention_mode_runs/flash_vs_three_kernel_<jobid>/` |
| Pulled canonical raw bundles on this branch | `logs/` |

## Canonical Pulled Raw Bundles

- `logs/h200_e2e_2225992/`
- `logs/h200_detailed_2236079/`
- `logs/h200_attention_2238637/`
- `logs/h200_attention_2238638/`

## What To Read First

If you are grading or reproducing the project, read in this order:

1. `PROJECT_README.md`
2. `benchmarks/benchmarks_README.md`
3. `benchmarks/benchmark_provenance.md`
That path gives you:

- the canonical benchmark docs
- the exact script or job used for each number
- the raw or historical evidence class behind each number
