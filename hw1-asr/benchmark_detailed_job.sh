#!/bin/bash
# Canonical output doc: ../benchmarks/benchmarks_detailed.md
# Raw outputs: benchmark_runs/detailed_<jobid>/...
#SBATCH --job-name=detailed-bench
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=benchmark_detailed_slurm_%j.log
#SBATCH --error=benchmark_detailed_slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ankushburman.ab@gmail.com

set -euo pipefail

REPO_DIR="/home/s2884198/edin-mls-26-spring"
HW1_DIR="$REPO_DIR/hw1-asr"
RUN_DIR="$HW1_DIR/benchmark_runs/detailed_${SLURM_JOB_ID}"
DOC_COPY="$REPO_DIR/docs/h200_detailed_component_benchmark_job_${SLURM_JOB_ID}.md"

mkdir -p "$RUN_DIR"
cd "$HW1_DIR"

# Shared Saxa/H200 runtime environment used by the canonical report jobs.
source "$HW1_DIR/setup_saxa_env.sh" "$RUN_DIR"

COMPONENT_RUNS=5
WARMUP_BENCHMARKS=1
MEASURED_BENCHMARKS=3
BENCHMARK_ARGS=(
    --runs "$COMPONENT_RUNS"
    --warmup-benchmarks "$WARMUP_BENCHMARKS"
    --benchmark-repeats "$MEASURED_BENCHMARKS"
)

{
    echo "=== Detailed Benchmark Job Metadata ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Working directory: $(pwd)"
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit: $(git rev-parse HEAD)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
    echo "Benchmark args: ${BENCHMARK_ARGS[*]}"
    python3 -c 'import torch; print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")'
    echo
} | tee "$RUN_DIR/job_metadata.txt"

run_model() {
    local model_name="$1"
    local log_path="$RUN_DIR/${model_name}_detailed.log"

    echo "=== ${model_name} ===" | tee -a "$RUN_DIR/combined.log"
    python3 benchmark_detailed.py "$model_name" "${BENCHMARK_ARGS[@]}" 2>&1 | tee "$log_path"
    echo | tee -a "$RUN_DIR/combined.log"
}

run_model "glm_asr_triton_template"
run_model "glm_asr_triton_example"

python3 - "$RUN_DIR" "$COMPONENT_RUNS" "$WARMUP_BENCHMARKS" "$MEASURED_BENCHMARKS" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
component_runs = sys.argv[2]
warmup_benchmarks = sys.argv[3]
measured_benchmarks = sys.argv[4]
labels = {
    "glm_asr_triton_template": "Template",
    "glm_asr_triton_example": "Baseline",
}
patterns = {
    "Audio Encoder": re.compile(r"Audio Encoder\s+([0-9.]+)ms"),
    "Multi-modal Projector": re.compile(r"Multi-modal Projector\s+([0-9.]+)ms"),
    "Decoder (Prefill)": re.compile(r"Decoder \(Prefill\)\s+([0-9.]+)ms"),
    "Decoder (50 decode steps)": re.compile(r"Decoder \(50 decode steps\)\s+([0-9.]+)ms"),
    "TOTAL (estimated for 50 tokens)": re.compile(r"TOTAL \(estimated for 50 tokens\)\s+([0-9.]+)ms"),
}

parsed = {}
for model_name in labels:
    log_text = (run_dir / f"{model_name}_detailed.log").read_text()
    parsed[model_name] = {}
    for key, pattern in patterns.items():
        matches = pattern.findall(log_text)
        if not matches:
            raise SystemExit(f"Failed to parse {key} from {model_name}_detailed.log")
        parsed[model_name][key] = float(matches[-1])

summary_json = run_dir / "comparison_summary.json"
summary_json.write_text(json.dumps(parsed, indent=2))

template = parsed["glm_asr_triton_template"]
baseline = parsed["glm_asr_triton_example"]

lines = []
lines.append("# H200 Detailed Component Benchmark")
lines.append("")
lines.append(f"- Job ID: `{run_dir.name.split('_')[-1]}`")
lines.append("- GPU: H200 MIG 3g.71gb on `saxa`")
lines.append(
    "- Script: "
    f"`hw1-asr/benchmark_detailed.py --runs {component_runs} "
    f"--warmup-benchmarks {warmup_benchmarks} "
    f"--benchmark-repeats {measured_benchmarks}`"
)
lines.append("- Configs compared: `glm_asr_triton_template` vs `glm_asr_triton_example`")
lines.append("")
lines.append("## Parsed Summary")
lines.append("")
lines.append("| Component | Template (ms) | Baseline (ms) | Speedup |")
lines.append("|-----------|---------------:|--------------:|--------:|")

for key in [
    "Audio Encoder",
    "Multi-modal Projector",
    "Decoder (Prefill)",
    "Decoder (50 decode steps)",
    "TOTAL (estimated for 50 tokens)",
]:
    t = template[key]
    b = baseline[key]
    speedup = b / t if t else 0.0
    lines.append(f"| {key} | {t:.2f} | {b:.2f} | {speedup:.2f}x |")

lines.append("")
lines.append("## Raw Artifacts")
lines.append("")
lines.append(f"- `{run_dir / 'job_metadata.txt'}`")
lines.append(f"- `{run_dir / 'glm_asr_triton_template_detailed.log'}`")
lines.append(f"- `{run_dir / 'glm_asr_triton_example_detailed.log'}`")
lines.append(f"- `{summary_json}`")
lines.append("")

(run_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n")
PY

cp "$RUN_DIR/comparison_summary.md" "$DOC_COPY"

{
    echo "=== Detailed Benchmark Complete ==="
    echo "Date: $(date)"
    echo "Run directory: $RUN_DIR"
    echo "Summary copy: $DOC_COPY"
} | tee -a "$RUN_DIR/job_metadata.txt"
