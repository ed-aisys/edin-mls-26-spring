#!/bin/bash
#SBATCH --job-name=ncu-profile
#SBATCH --mem=32G
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --time=01:00:00
#SBATCH --output=ncu_slurm_%j.log
#SBATCH --error=ncu_slurm_%j.err

set -euo pipefail

HW1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_TOKEN="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$HW1_DIR/ncu_runs/ncu_${JOB_TOKEN}"
mkdir -p "$RUN_DIR"
cd "$HW1_DIR"

source "$HW1_DIR/setup_saxa_env.sh" "$RUN_DIR"
CUDA_BIN="${MLS_CUDA_BIN:-/usr/local/cuda/bin}"
if [[ -d "$CUDA_BIN" ]]; then
    export PATH="$CUDA_BIN:$PATH"
fi

echo "=== Nsight Compute Profiling ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "ncu: $(which ncu)"
echo ""

# Profile with key metrics for arithmetic intensity:
# - FLOPs (fadd, fmul, ffma for fp32; hadd, hmul, hfma for fp16)
# - DRAM bytes read/written
# - L2 bytes read/written
# - SM active cycles and occupancy

echo "[1/2] Running ncu with roofline metrics..."
ncu \
    --metrics \
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum \
    --target-processes all \
    --kernel-name-base function \
    --csv \
    python3 ncu_profile.py 2>&1 | tee "$RUN_DIR/ncu_raw_output.csv"

echo ""
echo "[2/2] Parsing results..."

python3 - "$RUN_DIR/ncu_raw_output.csv" << 'PYEOF'
import csv
import io
import sys

with open(sys.argv[1], "r") as f:
    content = f.read()

lines = content.split("\n")
csv_start = None
for i, line in enumerate(lines):
    if '"Kernel Name"' in line or '"ID"' in line:
        csv_start = i
        break

if csv_start is None:
    print("Could not find CSV data in ncu output")
    print("Raw output (last 50 lines):")
    for l in lines[-50:]:
        print("  " + l)
else:
    csv_data = "\n".join(lines[csv_start:])
    reader = csv.DictReader(io.StringIO(csv_data))

    hdr = "{:<50} {:>12} {:>12} {:>12} {:>8} {:>6} {:>10}".format(
        "Kernel", "FP32 FLOP", "FP16 FLOP", "DRAM (B)", "AI", "Occ%", "Time(us)")
    print(hdr)
    print("-" * 112)

    for row in reader:
        name = row.get("Kernel Name", row.get("kernel_name", ""))[:50]

        fadd = int(row.get("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "0").replace(",","") or "0")
        fmul = int(row.get("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "0").replace(",","") or "0")
        ffma = int(row.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "0").replace(",","") or "0")
        fp32 = fadd + fmul + 2 * ffma

        hadd = int(row.get("sm__sass_thread_inst_executed_op_hadd_pred_on.sum", "0").replace(",","") or "0")
        hmul = int(row.get("sm__sass_thread_inst_executed_op_hmul_pred_on.sum", "0").replace(",","") or "0")
        hfma = int(row.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", "0").replace(",","") or "0")
        fp16 = hadd + hmul + 2 * hfma

        total_flops = fp32 + fp16

        dram_r = int(row.get("dram__bytes_read.sum", "0").replace(",","") or "0")
        dram_w = int(row.get("dram__bytes_write.sum", "0").replace(",","") or "0")
        dram_total = dram_r + dram_w

        ai = total_flops / dram_total if dram_total > 0 else 0
        occ = row.get("sm__warps_active.avg.pct_of_peak_sustained_active", "0")
        dur = row.get("gpu__time_duration.sum", "0")

        if total_flops > 0 or dram_total > 0:
            print("{:<50} {:>12,} {:>12,} {:>12,} {:>8.1f} {:>6} {:>10}".format(
                name, fp32, fp16, dram_total, ai, occ, dur))
PYEOF

echo ""
echo "=== Done ==="
echo "Raw output: $RUN_DIR/ncu_raw_output.csv"
