#!/bin/bash
# Canonical Section 5.3 source: ../benchmarks/benchmarks_attention.md
# Raw outputs: attention_mode_runs/flash_vs_three_kernel_<jobid>/...
#SBATCH --job-name=flash-3kernel
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=flash_vs_three_kernel_%j.log
#SBATCH --error=flash_vs_three_kernel_%j.err

set -euo pipefail

HW1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HW1_DIR/.." && pwd)"
JOB_TOKEN="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$HW1_DIR/attention_mode_runs/flash_vs_three_kernel_${JOB_TOKEN}"
DOC_COPY="$REPO_DIR/docs/flash_vs_three_kernel_job_${JOB_TOKEN}.md"

mkdir -p "$RUN_DIR"
cd "$HW1_DIR"

# Shared Saxa/H200 runtime environment used by the canonical report jobs.
source "$HW1_DIR/setup_saxa_env.sh" "$RUN_DIR"

STUDENT_ARGS=(--warmup 2 --runs 5 --warmup-benchmarks 1 --benchmark-repeats 3)
DETAILED_ARGS=(--runs 5 --warmup-benchmarks 1 --benchmark-repeats 3)

{
    echo "=== Flash vs Three-Kernel Benchmark Metadata ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Working directory: $(pwd)"
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit: $(git rev-parse HEAD)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
    echo "Student benchmark args: ${STUDENT_ARGS[*]}"
    echo "Detailed benchmark args: ${DETAILED_ARGS[*]}"
    python3 -c 'import torch; print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")'
    echo
} | tee "$RUN_DIR/job_metadata.txt"

run_mode() {
    local mode="$1"
    local label="$2"
    local mode_cache="$RUN_DIR/triton_cache_${mode}"
    mkdir -p "$mode_cache"

    export GLM_ASR_ATTENTION_MODE="$mode"
    export TRITON_CACHE_DIR="$mode_cache"

    echo "================================================================"
    echo "MODE: ${label} (${mode})"
    echo "================================================================"
    echo "GLM_ASR_ATTENTION_MODE=$GLM_ASR_ATTENTION_MODE"
    echo

    python3 benchmark_student.py glm_asr_triton_template "${STUDENT_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/${mode}_student.log"

    echo

    python3 benchmark_detailed.py glm_asr_triton_template "${DETAILED_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/${mode}_detailed.log"

    echo
}

run_mode "auto" "Current deployed path"
run_mode "three_kernel" "Historical materialized-score path"

python3 - "$RUN_DIR" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
modes = {
    "auto": "Current deployed path",
    "three_kernel": "Historical materialized-score path",
}

student_time_re = re.compile(r"Time:\s*([0-9.]+)ms\s*\(\+/-\s*([0-9.]+)ms\)")
accuracy_re = re.compile(r"Accuracy:\s*([0-9.]+)%")
component_patterns = {
    "Audio Encoder": re.compile(r"Audio Encoder\s+([0-9.]+)ms"),
    "Multi-modal Projector": re.compile(r"Multi-modal Projector\s+([0-9.]+)ms"),
    "Decoder (Prefill)": re.compile(r"Decoder \(Prefill\)\s+([0-9.]+)ms"),
    "Decoder (50 decode steps)": re.compile(r"Decoder \(50 decode steps\)\s+([0-9.]+)ms"),
    "TOTAL (estimated for 50 tokens)": re.compile(r"TOTAL \(estimated for 50 tokens\)\s+([0-9.]+)ms"),
}

parsed = {"student": {}, "detailed": {}}

for mode in modes:
    student_text = (run_dir / f"{mode}_student.log").read_text()
    detailed_text = (run_dir / f"{mode}_detailed.log").read_text()

    student_match = student_time_re.findall(student_text)
    if not student_match:
        raise SystemExit(f"Failed to parse end-to-end timing for {mode}")
    time_ms, std_ms = student_match[-1]

    accuracy_match = accuracy_re.findall(student_text)
    accuracy_pct = float(accuracy_match[-1]) if accuracy_match else None

    parsed["student"][mode] = {
        "time_ms": float(time_ms),
        "std_ms": float(std_ms),
        "accuracy_pct": accuracy_pct,
    }

    parsed["detailed"][mode] = {}
    for key, pattern in component_patterns.items():
        matches = pattern.findall(detailed_text)
        if not matches:
            raise SystemExit(f"Failed to parse {key} for {mode}")
        parsed["detailed"][mode][key] = float(matches[-1])

summary_json = run_dir / "comparison_summary.json"
summary_json.write_text(json.dumps(parsed, indent=2))

auto_student = parsed["student"]["auto"]
three_student = parsed["student"]["three_kernel"]
auto_detailed = parsed["detailed"]["auto"]
three_detailed = parsed["detailed"]["three_kernel"]

lines = []
lines.append("# Flash Attention vs Materialized-Score Attention")
lines.append("")
lines.append(f"- Job ID: `{run_dir.name.split('_')[-1]}`")
lines.append("- Hardware: see `job_metadata.txt` for the runtime GPU and host.")
lines.append("- Target report section: `Section 5.3 (FlashAttention-Style Attention)`")
lines.append("- Config A: `GLM_ASR_ATTENTION_MODE=auto`")
lines.append("  Current deployed path: flash kernel for seq_q > 4, SDPA fallback for seq_q <= 4.")
lines.append("- Config B: `GLM_ASR_ATTENTION_MODE=three_kernel`")
lines.append("  Historical materialized-score path: same current codebase, but attention materializes the score matrix instead of using flash.")
lines.append("")
lines.append("## End-to-End Student Benchmark")
lines.append("")
lines.append("| Mode | Mean (ms) | Std (ms) | Accuracy | Relative |")
lines.append("|------|----------:|---------:|---------:|---------:|")
relative = three_student["time_ms"] / auto_student["time_ms"]
auto_acc = f'{auto_student["accuracy_pct"]:.1f}%' if auto_student["accuracy_pct"] is not None else "n/a"
three_acc = f'{three_student["accuracy_pct"]:.1f}%' if three_student["accuracy_pct"] is not None else "n/a"
lines.append(
    f"| Current deployed path | {auto_student['time_ms']:.1f} | {auto_student['std_ms']:.1f} | {auto_acc} | 1.00x |"
)
if relative >= 1.0:
    relative_text = f"{relative:.2f}x slower vs current"
else:
    relative_text = f"{(1.0 / relative):.2f}x faster vs current"
lines.append(
    f"| Historical materialized-score path | {three_student['time_ms']:.1f} | {three_student['std_ms']:.1f} | {three_acc} | {relative_text} |"
)
lines.append("")
lines.append("## Detailed Component Benchmark")
lines.append("")
lines.append("| Component | Current path (ms) | Three-kernel path (ms) | Relative |")
lines.append("|-----------|------------------:|-----------------------:|---------:|")
for key in [
    "Audio Encoder",
    "Multi-modal Projector",
    "Decoder (Prefill)",
    "Decoder (50 decode steps)",
    "TOTAL (estimated for 50 tokens)",
]:
    auto_value = auto_detailed[key]
    three_value = three_detailed[key]
    relative = three_value / auto_value if auto_value else 0.0
    lines.append(f"| {key} | {auto_value:.2f} | {three_value:.2f} | {relative:.2f}x |")
lines.append("")
lines.append("## Raw Artifacts")
lines.append("")
lines.append(f"- `{run_dir / 'job_metadata.txt'}`")
lines.append(f"- `{run_dir / 'auto_student.log'}`")
lines.append(f"- `{run_dir / 'auto_detailed.log'}`")
lines.append(f"- `{run_dir / 'three_kernel_student.log'}`")
lines.append(f"- `{run_dir / 'three_kernel_detailed.log'}`")
lines.append(f"- `{summary_json}`")
lines.append("")

(run_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n")
PY

cp "$RUN_DIR/comparison_summary.md" "$DOC_COPY"

{
    echo "=== Flash vs Three-Kernel Benchmark Complete ==="
    echo "Date: $(date)"
    echo "Run directory: $RUN_DIR"
    echo "Summary copy: $DOC_COPY"
} | tee -a "$RUN_DIR/job_metadata.txt"
