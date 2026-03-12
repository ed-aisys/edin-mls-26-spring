#!/bin/bash
#
# Detailed Operator Profiling Script
# Measures execution time for each operator/layer in the model.
#
# Usage:
#   ./benchmark_detailed.sh <folder_name>           # Full profiling
#   ./benchmark_detailed.sh <folder_name> --nsys    # Nsight Systems profile
#
# Examples:
#   ./benchmark_detailed.sh glm_asr_cutile_example
#   ./benchmark_detailed.sh glm_asr_cutile_template --runs 5
#   ./benchmark_detailed.sh glm_asr_cutile_example --nsys
#   ./benchmark_detailed.sh glm_asr_triton_example
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    echo "GLM-ASR Detailed Operator Profiling"
    echo ""
    echo "Usage: $0 [folder_name] [options]"
    echo ""
    echo "Options:"
    echo "  --audio PATH      Path to test audio file"
    echo "  --runs N          Number of profiling runs (default: 3)"
    echo "  --nsys            Run Nsight Systems profiling"
    echo "  --seq-len N       Sequence length for micro-benchmarks (default: 256)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Available folders:"
    for dir in "$SCRIPT_DIR"/glm_asr_*/; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            echo "  - $dirname"
        fi
    done
    echo ""
    echo "Output includes:"
    echo "  - Audio encoder timing"
    echo "  - Multi-modal projector timing"
    echo "  - Decoder prefill timing"
    echo "  - Per-step decode timing"
    echo "  - Individual layer timing"
    echo "  - Attention method comparison (standard vs cuBLAS)"
    echo "  - Linear/GEMM method comparison"
}

# Check for help flag
for arg in "$@"; do
    if [ "$arg" == "-h" ] || [ "$arg" == "--help" ]; then
        show_help
        exit 0
    fi
done

# If no arguments, show help
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# If first argument is an option, pass through and let argparse handle defaults
if [[ "$1" == --* ]]; then
    cd "$SCRIPT_DIR"
    python benchmark_detailed.py "$@"
else
    FOLDER="$1"
    shift

    # Check if the explicitly provided folder exists
    if [ ! -d "$SCRIPT_DIR/$FOLDER" ]; then
        echo "Error: Folder '$FOLDER' not found in $SCRIPT_DIR"
        echo ""
        echo "Available folders:"
        for dir in "$SCRIPT_DIR"/glm_asr_*/; do
            if [ -d "$dir" ]; then
                echo "  - $(basename "$dir")"
            fi
        done
        exit 1
    fi

    cd "$SCRIPT_DIR"
    python benchmark_detailed.py "$FOLDER" "$@"
fi
