#!/bin/bash
# Source this file to configure a checkout-relative runtime environment for the
# benchmark scripts. This does not install packages.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it:"
    echo "  source hw1-asr/setup_saxa_env.sh [run_dir]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_HW1_DIR="$SCRIPT_DIR"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MLS_REPO_DIR="${MLS_REPO_DIR:-$DEFAULT_REPO_DIR}"
MLS_HW1_DIR="${MLS_HW1_DIR:-$DEFAULT_HW1_DIR}"
MLS_RUN_DIR_INPUT="${1:-${MLS_RUN_DIR:-$MLS_HW1_DIR/.runtime}}"
MLS_RUN_DIR="$MLS_RUN_DIR_INPUT"

MLS_CONDA_ENV_NAME="${MLS_CONDA_ENV_NAME:-mls}"
MLS_CONDA_ENV_BIN="${MLS_CONDA_ENV_BIN:-}"

if [[ -n "$MLS_CONDA_ENV_BIN" ]]; then
    export PATH="$MLS_CONDA_ENV_BIN:$PATH"
elif [[ -z "${CONDA_PREFIX:-}" ]]; then
    for candidate in \
        "$HOME/.conda/envs/$MLS_CONDA_ENV_NAME/bin" \
        "$HOME/miniconda3/envs/$MLS_CONDA_ENV_NAME/bin" \
        "$HOME/anaconda3/envs/$MLS_CONDA_ENV_NAME/bin"
    do
        if [[ -d "$candidate" ]]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
fi

export HF_HOME="${HF_HOME:-${MLS_HF_HOME:-$HOME/.cache/huggingface}}"

mkdir -p "$MLS_RUN_DIR"
export TMPDIR="$MLS_RUN_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$MLS_RUN_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$MLS_RUN_DIR/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

echo "Configured benchmark environment:"
echo "  MLS_REPO_DIR=$MLS_REPO_DIR"
echo "  MLS_HW1_DIR=$MLS_HW1_DIR"
echo "  MLS_RUN_DIR=$MLS_RUN_DIR"
echo "  MLS_CONDA_ENV_NAME=$MLS_CONDA_ENV_NAME"
echo "  HF_HOME=$HF_HOME"
echo "  TMPDIR=$TMPDIR"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
