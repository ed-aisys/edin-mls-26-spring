#!/bin/bash
#
# Shell wrapper for benchmark_student.py.
# Canonical benchmark doc mapping is recorded in ../benchmarks/benchmarks_README.md.
#
# Usage: ./benchmark.sh <folder_name> [options]
# Examples:
#   ./benchmark.sh glm_asr_triton_template
#   ./benchmark.sh glm_asr_triton_example
#   ./benchmark.sh glm_asr_triton_template --warmup 2 --runs 5
#   ./benchmark.sh glm_asr_triton_template --audio /path/to/test.wav
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_name>"
    echo ""
    echo "Available folders:"
    for dir in "$SCRIPT_DIR"/*/; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            if [ "$dirname" != "__pycache__" ]; then
                echo "  - $dirname"
            fi
        fi
    done
    exit 1
fi

FOLDER="$1"

# Check if folder exists
if [ ! -d "$SCRIPT_DIR/$FOLDER" ]; then
    echo "Error: Folder '$FOLDER' not found in $SCRIPT_DIR"
    exit 1
fi

# Run the Python benchmark
cd "$SCRIPT_DIR"
python benchmark_student.py "$FOLDER" "${@:2}"
