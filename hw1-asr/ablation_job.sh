#!/bin/bash
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

set -e

cd /home/s2884198/edin-mls-26-spring/hw1-asr

# Activate conda env from known location
export PATH="/home/s2884198/.conda/envs/mls/bin:$PATH"
export HF_HOME=/home/s2884198/.cache/huggingface

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
