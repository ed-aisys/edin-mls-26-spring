#!/bin/bash

# SLURM Batch Script for GLM-ASR Benchmark Testing
# This script runs benchmark tests on multiple audio files and implementations
# and writes results to a markdown file

#SBATCH --job-name=glm-asr-benchmark
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=meave.sullivan@live.co.uk

set -e

cd /home/s2259298/edin-mls-26-spring/hw1-asr

# Activate conda env from known location
export PATH="/home/s2259298/.conda/envs/mls/bin:$PATH"
export HF_HOME=/home/s2259298/.cache/huggingface

# Function to check memory usage
check_memory() {
    free -h | grep Mem
}

# Function to force garbage collection
cleanup_memory() {
    echo "Cleaning up memory..."
    python3 -c "import gc; gc.collect()" 2>/dev/null || true
    sleep 1
}

echo "=== Benchmark Test Starting ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python3 -c 'import torch; print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")'
echo ""

# Set up output file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="benchmark_results_${TIMESTAMP}.md"

# Initialize markdown file with header
cat > "$OUTPUT_FILE" << 'EOF'
# GLM-ASR Benchmark Results

This document contains benchmark results for GLM-ASR implementations.

**Generated:** $(date)

---

EOF

echo "Benchmark script started at $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Arrays for audio files and implementations
AUDIO_FILES=(
    "student_test_audio.wav"
    "student_test_audio_1.wav"
    "student_test_audio_2.wav"
    "student_test_audio_3.wav"
    "student_test_audio_4.wav"
)

IMPLEMENTATIONS=(
    "glm_asr_triton_example"
    "glm_asr_triton_example_ank"
)

# Counter for tracking progress
TOTAL_TESTS=$((${#AUDIO_FILES[@]} * ${#IMPLEMENTATIONS[@]}))
CURRENT_TEST=0

# Loop through implementations and audio files
for IMPL in "${IMPLEMENTATIONS[@]}"; do
    echo "## Implementation: $IMPL" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    for AUDIO in "${AUDIO_FILES[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))

        echo "Running test $CURRENT_TEST/$TOTAL_TESTS: $IMPL with $AUDIO"
        echo "" >> "$OUTPUT_FILE"
        echo "### Test: $AUDIO" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo '```' >> "$OUTPUT_FILE"

        # Check if audio file exists
        if [ ! -f "$AUDIO" ]; then
            echo "ERROR: Audio file not found: $AUDIO" | tee -a "$OUTPUT_FILE"
            echo "File not found: $AUDIO" >> "$OUTPUT_FILE"
        else
            # Run the benchmark command and capture output
            ./benchmark.sh "$IMPL" --audio "$AUDIO" 2>&1 | tee -a "$OUTPUT_FILE"
        fi

        echo '```' >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

        # Cleanup memory to prevent OOM issues
        cleanup_memory
    done

    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

# Add completion information
{
    echo "## Summary"
    echo ""
    echo "**Completion Time:** $(date)"
    echo "**Total Tests Run:** $CURRENT_TEST"
    echo "**Output File:** $OUTPUT_FILE"
} >> "$OUTPUT_FILE"

echo ""
echo "=== Benchmark Test Complete ==="
echo "Date: $(date)"
echo "Results written to: $OUTPUT_FILE"
