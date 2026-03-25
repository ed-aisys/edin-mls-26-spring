#!/bin/bash
#SBATCH --job-name=flash-main
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=flash_vs_main_%j.log
#SBATCH --error=flash_vs_main_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ankushburman.ab@gmail.com

set -e
cd /home/s2884198/edin-mls-26-spring/hw1-asr
export PATH="/home/s2884198/.conda/envs/mls/bin:$PATH"
export HF_HOME=/home/s2884198/.cache/huggingface

echo "=== Flash Attention vs Original 3-Kernel (origin/main) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""

cp glm_asr_triton_template/attention.py glm_asr_triton_template/attention.py.backup

# ── Test 1: Our flash attention — 3 runs ──
echo "================================================================"
echo "TEST 1: Flash Attention (our implementation)"
echo "================================================================"
rm -rf ~/.triton/cache
for run in 1 2 3; do
    echo ""
    echo "--- Flash Attention Run $run ---"
    python3 benchmark_student.py glm_asr_triton_template 2>&1
done 2>&1 | tee flash_main_our_result.txt

echo ""
echo "--- Student benchmark done, now detailed ---"
rm -rf ~/.triton/cache
python3 benchmark_detailed.py glm_asr_triton_template 2>&1 | tee flash_main_our_detailed.txt

echo ""

# ── Test 2: Original 3-kernel from origin/main — 3 runs ──
echo "================================================================"
echo "TEST 2: Original 3-Kernel (origin/main)"
echo "================================================================"

echo "=== Diff ==="
diff <(cat glm_asr_triton_template/attention.py.backup) <(git show origin/main:hw1-asr/glm_asr_triton_template/attention.py) | head -80
echo "(first 80 lines of diff)"
echo ""

git show origin/main:hw1-asr/glm_asr_triton_template/attention.py > glm_asr_triton_template/attention.py
rm -rf ~/.triton/cache
for run in 1 2 3; do
    echo ""
    echo "--- 3-Kernel (main) Run $run ---"
    python3 benchmark_student.py glm_asr_triton_template 2>&1
done 2>&1 | tee flash_main_3kernel_result.txt

echo ""
echo "--- Student benchmark done, now detailed ---"
rm -rf ~/.triton/cache
python3 benchmark_detailed.py glm_asr_triton_template 2>&1 | tee flash_main_3kernel_detailed.txt

# ── Restore ──
cp glm_asr_triton_template/attention.py.backup glm_asr_triton_template/attention.py
rm glm_asr_triton_template/attention.py.backup

# ── Summary ──
echo ""
echo "================================================================"
echo "RESULTS SUMMARY"
echo "================================================================"
echo ""
echo "--- Flash Attention (student benchmark) ---"
grep "Time:" flash_main_our_result.txt
echo ""
echo "--- 3-Kernel origin/main (student benchmark) ---"
grep "Time:" flash_main_3kernel_result.txt
echo ""
echo "--- Flash Attention (detailed per-component) ---"
grep -E "Audio Encoder|Projector|Prefill|decode steps|TOTAL" flash_main_our_detailed.txt
echo ""
echo "--- 3-Kernel origin/main (detailed per-component) ---"
grep -E "Audio Encoder|Projector|Prefill|decode steps|TOTAL" flash_main_3kernel_detailed.txt
echo ""
echo "=== Done ==="
