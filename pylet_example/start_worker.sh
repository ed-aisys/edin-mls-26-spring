#!/bin/bash
#SBATCH --job-name=pylet-worker
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pylet-worker-%j.out

# ──────────────────────────────────────────────
# PyLet Worker — runs on a SLURM compute node
# Connects back to the head node on the login node.
#
# Usage:
#   1. On login node, start the head:  pylet start
#   2. Submit this script:             sbatch start_worker.sh
# ──────────────────────────────────────────────

# Change this if your login node hostname differs
HEAD_NODE="${HEAD_NODE:-hastings}"
HEAD_PORT="${HEAD_PORT:-8000}"

echo "=== PyLet Worker ==="
echo "Compute node : $(hostname)"
echo "Head address : ${HEAD_NODE}:${HEAD_PORT}"
echo "GPUs visible : ${CUDA_VISIBLE_DEVICES:-all}"
echo "===================="

# Activate your conda environment
source ~/.bashrc
conda activate mlsys

# Start a single pylet worker that exposes 2 GPUs to the head
pylet start --head "${HEAD_NODE}:${HEAD_PORT}" --gpu-units 2 --memory-mb 8192 --cpu-cores 8
