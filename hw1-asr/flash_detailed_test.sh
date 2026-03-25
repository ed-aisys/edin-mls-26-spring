#!/bin/bash
#SBATCH --job-name=flash-detail
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=flash_detailed_%j.log
#SBATCH --error=flash_detailed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ankushburman.ab@gmail.com

set -e

cd /home/s2884198/edin-mls-26-spring/hw1-asr

export PATH="/home/s2884198/.conda/envs/mls/bin:$PATH"
export HF_HOME=/home/s2884198/.cache/huggingface

echo "=== Flash Attention vs 3-Kernel — Detailed Benchmark ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""

# Save our current attention.py
cp glm_asr_triton_template/attention.py glm_asr_triton_template/attention.py.backup

# ── Test 1: Flash attention detailed benchmark ──
echo "================================================================"
echo "TEST 1: Flash Attention (our implementation) — Detailed"
echo "================================================================"
rm -rf ~/.triton/cache

echo ""
echo "--- Detailed benchmark (per-component profiling) ---"
for run in 1 2 3; do
    echo ""
    echo "--- Detailed Run $run ---"
    python3 benchmark_detailed.py glm_asr_triton_template 2>&1
done 2>&1 | tee flash_detailed_result.txt

echo ""

# ── Test 2: 3-kernel attention detailed benchmark ──
echo "================================================================"
echo "TEST 2: 3-Kernel Attention (meave branch) — Detailed"
echo "================================================================"

git show origin/meave:hw1-asr/glm_asr_triton_template/attention.py > glm_asr_triton_template/attention.py

rm -rf ~/.triton/cache

echo ""
echo "--- Detailed benchmark (per-component profiling) ---"
for run in 1 2 3; do
    echo ""
    echo "--- Detailed Run $run ---"
    python3 benchmark_detailed.py glm_asr_triton_template 2>&1
done 2>&1 | tee three_kernel_detailed_result.txt

echo ""

# ── Restore ──
cp glm_asr_triton_template/attention.py.backup glm_asr_triton_template/attention.py
rm glm_asr_triton_template/attention.py.backup

# ── Summary ──
echo ""
echo "================================================================"
echo "DETAILED RESULTS SUMMARY"
echo "================================================================"
echo ""
echo "--- Flash Attention (per-component) ---"
grep -E "Audio Encoder|Projector|Decoder \(Prefill\)|Decoder \(.*decode|TOTAL" flash_detailed_result.txt
echo ""
echo "--- 3-Kernel Attention (per-component) ---"
grep -E "Audio Encoder|Projector|Decoder \(Prefill\)|Decoder \(.*decode|TOTAL" three_kernel_detailed_result.txt
echo ""
echo "Full outputs: flash_detailed_result.txt, three_kernel_detailed_result.txt"
echo "=== Done ==="
