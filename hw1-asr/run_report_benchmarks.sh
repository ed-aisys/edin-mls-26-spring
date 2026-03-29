#!/bin/bash
# Thin wrapper around the canonical report benchmark entry points.
# Project-level reproduction guide: ../PROJECT_README.md
# Benchmark provenance guide: ../benchmarks/benchmarks_README.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./run_report_benchmarks.sh print
  ./run_report_benchmarks.sh h200-e2e [run_label]
  ./run_report_benchmarks.sh h200-detailed
  ./run_report_benchmarks.sh h200-ablation
  ./run_report_benchmarks.sh h200-attention

Targets:
  print           Show the exact underlying commands without running them
  h200-e2e        Run the canonical H200 end-to-end benchmark pattern locally
  h200-detailed   Submit the canonical H200 detailed benchmark job with sbatch
  h200-ablation   Submit the canonical H200 ablation job with sbatch
  h200-attention  Submit the canonical Section 5.3 attention job with sbatch

Optional environment overrides for sbatch targets:
  MLS_PARTITION   Slurm partition (for example: gpu, Teaching)
  MLS_NODELIST    Specific node list
  MLS_GRES        GPU request (for example: gpu:1 or gpu:3g.71gb:1)
  MLS_MEM         Memory request
  MLS_TIME        Wall-clock limit
  MLS_MAIL_TYPE   Slurm mail type
  MLS_MAIL_USER   Slurm mail recipient
EOF
}

print_commands() {
    cat <<'EOF'
H200 end-to-end report pattern:
  cd hw1-asr
  python3 benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5
  # repeat the template command five independent times
  python3 benchmark_student.py glm_asr_triton_example --warmup 2 --runs 5

H200 detailed/component:
  sbatch hw1-asr/benchmark_detailed_job.sh

H200 ablation:
  sbatch hw1-asr/ablation_job.sh

Section 5.3 attention comparison:
  sbatch hw1-asr/flash_vs_three_kernel_job.sh
EOF
}

ensure_sbatch() {
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch is required for this target." >&2
        exit 1
    fi
}

build_sbatch_args() {
    SBATCH_ARGS=()
    [[ -n "${MLS_PARTITION:-}" ]] && SBATCH_ARGS+=(--partition "$MLS_PARTITION")
    [[ -n "${MLS_NODELIST:-}" ]] && SBATCH_ARGS+=(--nodelist "$MLS_NODELIST")
    [[ -n "${MLS_GRES:-}" ]] && SBATCH_ARGS+=(--gres "$MLS_GRES")
    [[ -n "${MLS_MEM:-}" ]] && SBATCH_ARGS+=(--mem "$MLS_MEM")
    [[ -n "${MLS_TIME:-}" ]] && SBATCH_ARGS+=(--time "$MLS_TIME")
    [[ -n "${MLS_MAIL_TYPE:-}" ]] && SBATCH_ARGS+=(--mail-type "$MLS_MAIL_TYPE")
    [[ -n "${MLS_MAIL_USER:-}" ]] && SBATCH_ARGS+=(--mail-user "$MLS_MAIL_USER")
}

run_h200_e2e() {
    local label="${1:-$(date +%Y%m%d_%H%M%S)}"
    local run_dir="$SCRIPT_DIR/benchmark_runs/report_e2e_${label}"

    mkdir -p "$run_dir"
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/setup_saxa_env.sh" "$run_dir"

    cd "$SCRIPT_DIR"

    {
        echo "=== Report H200 End-to-End Benchmark ==="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "Working directory: $(pwd)"
        echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
        echo "Commit: $(git rev-parse HEAD)"
        echo "Command: python3 benchmark_student.py <folder> --warmup 2 --runs 5"
        echo "Template invocations: 5"
        echo "Baseline invocations: 1"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
        echo
    } | tee "$run_dir/job_metadata.txt"

    local i
    for i in 1 2 3 4 5; do
        echo "=== TEMPLATE INVOCATION ${i}/5 ===" | tee -a "$run_dir/job_metadata.txt"
        python3 benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5 \
            2>&1 | tee "$run_dir/template_invocation_${i}.log"
        echo | tee -a "$run_dir/job_metadata.txt"
    done

    echo "=== BASELINE INVOCATION 1/1 ===" | tee -a "$run_dir/job_metadata.txt"
    python3 benchmark_student.py glm_asr_triton_example --warmup 2 --runs 5 \
        2>&1 | tee "$run_dir/baseline_invocation_1.log"

    echo
    echo "Run complete. Logs written to: $run_dir"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    print)
        print_commands
        ;;
    h200-e2e)
        run_h200_e2e "${2:-}"
        ;;
    h200-detailed)
        ensure_sbatch
        build_sbatch_args
        cd "$SCRIPT_DIR"
        sbatch "${SBATCH_ARGS[@]}" benchmark_detailed_job.sh
        ;;
    h200-ablation)
        ensure_sbatch
        build_sbatch_args
        cd "$SCRIPT_DIR"
        sbatch "${SBATCH_ARGS[@]}" ablation_job.sh
        ;;
    h200-attention)
        ensure_sbatch
        build_sbatch_args
        cd "$SCRIPT_DIR"
        sbatch "${SBATCH_ARGS[@]}" flash_vs_three_kernel_job.sh
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown target: $1" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
