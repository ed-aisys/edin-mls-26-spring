#!/bin/bash
# Source this file on Saxa/H200 shells to reproduce the runtime environment used
# by the canonical H200 benchmarks. This does not install packages.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it:"
    echo "  source hw1-asr/setup_saxa_env.sh [run_dir]"
    exit 1
fi

MLS_REPO_DIR="${MLS_REPO_DIR:-/home/s2884198/edin-mls-26-spring}"
MLS_HW1_DIR="${MLS_HW1_DIR:-$MLS_REPO_DIR/hw1-asr}"
MLS_RUN_DIR_INPUT="${1:-${MLS_RUN_DIR:-$MLS_HW1_DIR/.runtime}}"
MLS_RUN_DIR="$MLS_RUN_DIR_INPUT"

export PATH="/home/s2884198/.conda/envs/mls/bin:$PATH"
export HF_HOME="${HF_HOME:-/home/s2884198/.cache/huggingface}"

mkdir -p "$MLS_RUN_DIR"
export TMPDIR="$MLS_RUN_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$MLS_RUN_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$MLS_RUN_DIR/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

echo "Configured Saxa benchmark environment:"
echo "  MLS_REPO_DIR=$MLS_REPO_DIR"
echo "  MLS_HW1_DIR=$MLS_HW1_DIR"
echo "  MLS_RUN_DIR=$MLS_RUN_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  TMPDIR=$TMPDIR"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
