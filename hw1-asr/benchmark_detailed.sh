#!/bin/bash
#
# Shell wrapper for benchmark_detailed.py.
# Canonical benchmark doc mapping is recorded in ../benchmarks/benchmarks_README.md.
#
# Usage:
#   ./benchmark_detailed.sh <folder_name>
#   ./benchmark_detailed.sh <folder_name> --runs 5
#   ./benchmark_detailed.sh <folder_name> --warmup-benchmarks 1 --benchmark-repeats 3
#
# Examples:
#   ./benchmark_detailed.sh glm_asr_triton_template
#   ./benchmark_detailed.sh glm_asr_triton_example --runs 5
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
    echo "  --runs N          Number of timed runs per component (default: 3)"
    echo "  --warmup-benchmarks N"
    echo "                    Number of full benchmark passes to discard before measuring"
    echo "  --benchmark-repeats N"
    echo "                    Number of full benchmark passes to aggregate"
    echo "  --nsys            Run Nsight Systems profiling"
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
    echo "  - Individual decoder-layer timing"
    echo ""
    echo "For the report-ready H200 component benchmark, use:"
    echo "  sbatch benchmark_detailed_job.sh"
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
