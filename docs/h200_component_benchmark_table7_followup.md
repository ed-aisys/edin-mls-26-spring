# H200 Detailed Component Benchmark Follow-up

**Status:** Final usable results come from SLURM job `2236079`, which completed successfully using a warmup-corrected benchmark flow. Earlier runs `2236033` and `2236054` are retained as intermediate artifacts but should not be used as the final report evidence.

## Why this benchmark exists

The report currently has a per-component comparison table in `report/report_no_abstract.tex` around the current Table 7 / Comparison section. During repo review, I could not find committed supporting artifacts for the baseline component timings in that table. That makes the table weaker than the surrounding sections, even though the rest of the report is mostly well supported.

This follow-up benchmark was created to fix that evidence gap by collecting a clean, reproducible component-level comparison between:

- `glm_asr_triton_template`
- `glm_asr_triton_example`

using `hw1-asr/benchmark_detailed.py` on the H200 MIG `3g.71gb` slice on `saxa`.

## What the SLURM job records

The batch script writes:

- raw template detailed benchmark output
- raw baseline detailed benchmark output
- job metadata: branch, commit, hostname, GPU, and PyTorch version
- a parsed summary table with:
  - Audio Encoder
  - Multi-modal Projector
  - Decoder (Prefill)
  - Decoder (50 decode steps)
  - Total estimated latency for 50 tokens

The script also copies the parsed markdown summary into the remote repo's `docs/` folder so the benchmark has a report-facing artifact, not just a terminal log.

## Benchmark run history

### Run 1: `2236033` failed due to `/tmp` exhaustion

- Status: failed after `00:05:05`
- Problem: Triton/GCC attempted to write temporary build artifacts to node-local `/tmp`
- Error: `No space left on device`
- Action taken: patched `benchmark_detailed_job.sh` to redirect temp files, Triton cache, and Torch extension builds into the job run directory

Artifacts retained:

- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_detailed_slurm_2236033.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236033/glm_asr_triton_template_detailed.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236033/glm_asr_triton_example_detailed.log`

### Run 2: `2236054` completed but was not methodologically clean enough

- Status: completed successfully
- Script version: old `benchmark_detailed.py --runs 5`
- Problem: first-use Triton compilation and warmup were still included in the timed component measurements
- Consequence: the template audio encoder and several other stages showed inflated times and very large standard deviations
- Conclusion: useful as a debugging run, not ideal as final report evidence

Artifacts retained:

- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_detailed_slurm_2236054.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236054/comparison_summary.md`
- `/home/s2884198/edin-mls-26-spring/docs/h200_detailed_component_benchmark_job_2236054.md`

### Run 3: `2236079` is the final usable benchmark

- Status: completed successfully in `00:04:41`
- Script version: `hw1-asr/benchmark_detailed.py --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3`
- Method: discard one full benchmark pass for warmup, then average three measured full-benchmark passes
- Why this matters: Triton compilation and first-use warmup are excluded from the reported component timings

This is the run that should be cited for the H200 component-level comparison.

## Final job details

- Cluster command: `sbatch benchmark_detailed_job.sh`
- Final job ID: `2236079`
- Job name: `detailed-bench`
- Partition: `Teaching`
- Node request: `saxa`
- GPU request: `gpu:3g.71gb:1`
- Time limit: `01:30:00`
- Mail target: `ankushburman.ab@gmail.com`

Final remote artifacts for job `2236079`:

- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_detailed_slurm_2236079.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_detailed_slurm_2236079.err`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/job_metadata.txt`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/glm_asr_triton_template_detailed.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/glm_asr_triton_example_detailed.log`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/comparison_summary.json`
- `/home/s2884198/edin-mls-26-spring/hw1-asr/benchmark_runs/detailed_2236079/comparison_summary.md`
- `/home/s2884198/edin-mls-26-spring/docs/h200_detailed_component_benchmark_job_2236079.md`

## Final parsed results from job `2236079`

| Component | Template (ms) | Baseline (ms) | Speedup |
|-----------|---------------:|--------------:|--------:|
| Audio Encoder | 36.53 | 167.02 | 4.57x |
| Multi-modal Projector | 0.14 | 1.11 | 7.93x |
| Decoder (Prefill) | 12.97 | 20.51 | 1.58x |
| Decoder (50 decode steps) | 580.45 | 817.09 | 1.41x |
| TOTAL (estimated for 50 tokens) | 630.09 | 1005.73 | 1.60x |

## Why the final run is more trustworthy

The completed warmup-corrected log shows stable measured passes after the discarded warmup run:

- template audio encoder: `36.61`, `36.47`, `36.51` ms across the three measured passes
- template decoder prefill: `13.01`, `12.98`, `12.93` ms
- template decode step: `11.64`, `11.61`, `11.58` ms
- baseline audio encoder: `167.09`, `167.00`, `166.97` ms
- baseline decoder prefill: `20.49`, `20.15`, `20.88` ms
- baseline decode step: `16.32`, `16.56`, `16.14` ms

This is much cleaner than run `2236054`, where first-use compilation inflated several component timings and produced extremely large standard deviations.

## Why H200 data is still useful

This does **not** validate an RTX 5090 table. It produces defensible H200 numbers instead. That means it should be used in one of two ways:

1. replace the unsupported current Table 7 with an H200-labelled component table, or
2. rewrite the comparison section to explicitly say that the component-level breakdown comes from H200 MIG measurements, while RTX 5090 claims remain limited to the existing end-to-end benchmark docs.

Using these numbers as if they came from the RTX 5090 would be a mistake.

## Intended report insertion point

This benchmark is meant to support the comparison section in:

- `report/report_no_abstract.tex`

Specifically:

- the component comparison table currently around the `Per-Operator Comparison` subsection
- the surrounding discussion that explains where the speedups come from
- the earlier H200 component profiling table in the profiling section, which now needs to match the warmup-corrected methodology

## Recommendation

Use job `2236079` as the cited component-level evidence in the local report and in Overleaf.

The main reporting caveat is not about correctness, but scope:

- these are H200 MIG component timings, not RTX 5090 component timings
- `benchmark_detailed.py` measures isolated operator behavior without KV history
- the larger 2.27x end-to-end speedup still depends on KV-cached generation in `generate_v8b`
