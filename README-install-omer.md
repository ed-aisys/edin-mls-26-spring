# Triton Environment Setup — UV Alternative (Omer)

This guide describes how to set up the Triton tutorial environment using **uv** instead of conda. Use this if you prefer uv's faster, lighter-weight workflow.

## Prerequisites

- A Unix-like system (Linux or macOS)
- `curl` (for installing uv)

## Quick Start

From the repository root:

```bash
bash utils/setup-triton-uv.sh
```

For non-interactive mode (e.g. CI):

```bash
bash utils/setup-triton-uv.sh -y
```

## What the Script Does

1. **Installs uv** (if not present) — uv is a fast Python package manager written in Rust
2. **Creates a virtual environment** at `.venv` in the project root (Python 3.11)
3. **Installs the Triton stack**: torch, numpy, triton, cupy, datasets

## After Setup

### Activate the environment

```bash
source .venv/bin/activate
```

### Or use `uv run` (no activation needed)

From the project root:

```bash
uv run python triton-tutorial/0-environment/check.py
```

## Comparison: Conda vs UV

| Aspect | `setup-triton.sh` (conda) | `setup-triton-uv.sh` (uv) |
|--------|---------------------------|---------------------------|
| Environment | Global conda env `mls` | Project-local `.venv` |
| Installer | Miniconda (~100MB+) | uv (~10MB) |
| Speed | Slower | 10–100× faster |
| Activation | `conda activate mls` | `source .venv/bin/activate` |

## Optional: pyproject.toml

If the project has a `pyproject.toml` with the Triton dependencies, the script uses `uv sync` for reproducible installs. Otherwise it runs `uv pip install` with the required packages.

## Troubleshooting

- **uv not found after install**: Add `~/.local/bin` to PATH:  
  `export PATH="${HOME}/.local/bin:${PATH}"`
- **Network timeout** (e.g. `Failed to download nvidia-cudnn-cu12`): The script uses a 180s timeout. For slow connections, run:  
  `UV_HTTP_TIMEOUT=300 bash utils/setup-triton-uv.sh`
- **CUDA / GPU**: For CUDA-enabled PyTorch, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/) and install the matching CUDA wheel for your driver.
