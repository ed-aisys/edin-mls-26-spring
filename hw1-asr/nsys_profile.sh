#!/bin/bash
# Nsight Systems profiling for GLM-ASR.
# Usage: allocate a GPU node, then run this script from hw1-asr/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/.nsys_runtime_$(date +%Y%m%d_%H%M%S)"
source "$SCRIPT_DIR/setup_saxa_env.sh" "$RUN_DIR"
cd "$SCRIPT_DIR"

PROFILE_DIR="../nsys_profiles/h200_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DIR"

echo "=== Nsight Systems Profiling ==="
echo "Output dir: $PROFILE_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""

# 1. Full pipeline profile (our optimized implementation)
echo "[1/3] Profiling optimized implementation..."
nsys profile \
    --output="$PROFILE_DIR/optimized_full" \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    python3 benchmark_student.py glm_asr_triton_template 2>&1 | tee "$PROFILE_DIR/optimized_full.log"

# 2. Detailed benchmark profile (50 tokens, stock generate for per-component timing)
echo ""
echo "[2/3] Profiling detailed benchmark..."
nsys profile \
    --output="$PROFILE_DIR/detailed_50tok" \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    python3 benchmark_detailed.py glm_asr_triton_template 2>&1 | tee "$PROFILE_DIR/detailed_50tok.log"

# 3. Baseline profile for comparison
echo ""
echo "[3/3] Profiling baseline implementation..."
nsys profile \
    --output="$PROFILE_DIR/baseline_full" \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    python3 benchmark_student.py glm_asr_triton_example 2>&1 | tee "$PROFILE_DIR/baseline_full.log"

echo ""
echo "=== Profiling complete ==="
echo "Files in $PROFILE_DIR:"
ls -la "$PROFILE_DIR/"

# Extract key stats
echo ""
echo "=== Kernel Summary (optimized) ==="
nsys stats "$PROFILE_DIR/optimized_full.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null | head -30

echo ""
echo "=== CUDA API Summary (optimized) ==="
nsys stats "$PROFILE_DIR/optimized_full.nsys-rep" --report cuda_api_sum 2>/dev/null | head -20
