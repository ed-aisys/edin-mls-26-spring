#!/bin/bash
# Canonical output doc: ../benchmarks/benchmarks_ablation.md
# Generated outputs: ablation_results.json, ablation_results.md, ablation_output.log
#SBATCH --job-name=ablation
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ablation_slurm_%j.log
#SBATCH --error=ablation_slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ankushburman.ab@gmail.com

set -euo pipefail

HW1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HW1_DIR/.." && pwd)"
RUN_DIR="$HW1_DIR/ablation_runs/ablation_${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR"
cd "$HW1_DIR"

# Shared Saxa/H200 runtime environment used by the canonical report jobs.
source "$HW1_DIR/setup_saxa_env.sh" "$RUN_DIR"

echo "=== Ablation Test Starting ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python3 -c 'import torch; print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")'
echo ""

python3 ablation_test.py 2>&1 | tee ablation_output.log

echo ""
echo "=== Ablation Test Complete ==="
echo "Date: $(date)"
echo "Results saved to: ablation_results.json, ablation_results.md"
