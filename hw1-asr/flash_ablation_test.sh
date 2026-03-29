#!/bin/bash
# Historical exploratory script.
# Canonical Section 5.3 source is ../benchmarks/benchmarks_attention.md.
#SBATCH --job-name=flash-ablat
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=flash_ablation_%j.log
#SBATCH --error=flash_ablation_%j.err

set -euo pipefail

HW1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_TOKEN="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$HW1_DIR/attention_mode_runs/flash_ablation_${JOB_TOKEN}"
mkdir -p "$RUN_DIR"

cd "$HW1_DIR"
source "$HW1_DIR/setup_saxa_env.sh" "$RUN_DIR"

echo "=== Flash Attention Ablation (same codebase, toggle flash on/off) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""

cp glm_asr_triton_template/attention.py "$RUN_DIR/attention.py.backup"

# ── Test 1: Flash attention ON (baseline) — 5 runs ──
echo "================================================================"
echo "TEST 1: Flash Attention ON (our implementation)"
echo "================================================================"
rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
for run in 1 2 3 4 5; do
    echo "--- Run $run ---"
    python3 benchmark_student.py glm_asr_triton_template 2>&1 | grep -E "Run [0-9]:|^Time:|^Accuracy:|^Transcription:|^Status:"
    echo ""
done 2>&1 | tee "$RUN_DIR/flash_on_result.txt"

echo ""

# ── Test 2: Flash attention OFF (force PyTorch SDPA for all) — 5 runs ──
# This replaces our flash kernel with PyTorch's built-in SDPA for ALL attention
# (not just seq_q<=4). This is a clean single-variable toggle.
echo "================================================================"
echo "TEST 2: Flash Attention OFF (PyTorch SDPA for all attention)"
echo "================================================================"

echo "Code change:"
echo "  Replacing: if q.is_cuda and seq_q <= 4:"
echo "  With:      if q.is_cuda:  # always use SDPA"

sed -i 's/if q.is_cuda and seq_q <= 4:/if q.is_cuda:  # ABLATION: always SDPA/' glm_asr_triton_template/attention.py

rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
for run in 1 2 3 4 5; do
    echo "--- Run $run ---"
    python3 benchmark_student.py glm_asr_triton_template 2>&1 | grep -E "Run [0-9]:|^Time:|^Accuracy:|^Transcription:|^Status:"
    echo ""
done 2>&1 | tee "$RUN_DIR/flash_off_sdpa_result.txt"

echo ""

# ── Restore ──
cp "$RUN_DIR/attention.py.backup" glm_asr_triton_template/attention.py
rm "$RUN_DIR/attention.py.backup"

# ── Summary ──
echo "================================================================"
echo "RESULTS SUMMARY"
echo "================================================================"
echo ""
echo "Flash Attention ON:"
grep "^Time:" "$RUN_DIR/flash_on_result.txt"
echo ""
echo "Flash Attention OFF (all SDPA):"
grep "^Time:" "$RUN_DIR/flash_off_sdpa_result.txt"
echo ""
echo "Accuracy check:"
echo "Flash ON:"
grep "^Accuracy:" "$RUN_DIR/flash_on_result.txt"
echo "Flash OFF:"
grep "^Accuracy:" "$RUN_DIR/flash_off_sdpa_result.txt"
echo ""
echo "Artifacts: $RUN_DIR"
echo "=== Done ==="
