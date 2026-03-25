#!/bin/bash
#SBATCH --job-name=flash-quick
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=flash_quick_%j.log
#SBATCH --error=flash_quick_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ankushburman.ab@gmail.com

set -e
cd /home/s2884198/edin-mls-26-spring/hw1-asr
export PATH="/home/s2884198/.conda/envs/mls/bin:$PATH"
export HF_HOME=/home/s2884198/.cache/huggingface

echo "=== Quick Flash vs 3-Kernel Test (1g.18gb) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cp glm_asr_triton_template/attention.py glm_asr_triton_template/attention.py.backup

echo ""
echo "=== Flash Attention ==="
rm -rf ~/.triton/cache
python3 benchmark_student.py glm_asr_triton_template 2>&1

echo ""
echo "=== 3-Kernel Attention (meave) ==="
git show origin/meave:hw1-asr/glm_asr_triton_template/attention.py > glm_asr_triton_template/attention.py
rm -rf ~/.triton/cache
python3 benchmark_student.py glm_asr_triton_template 2>&1

cp glm_asr_triton_template/attention.py.backup glm_asr_triton_template/attention.py
rm glm_asr_triton_template/attention.py.backup
echo ""
echo "=== Done ==="
