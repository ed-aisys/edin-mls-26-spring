#!/usr/bin/env bash
set -eo pipefail

# =========================
# Config
# =========================
PYTHON_VERSION="3.11"
# Large CUDA packages (e.g. nvidia-cudnn-cu12) need longer timeout; default is 30s
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
VENV_NAME=".venv"

# Parse command line arguments
AUTO_YES=false
while [[ $# -gt 0 ]]; do
	case $1 in
		-y|--yes)
			AUTO_YES=true
			shift
			;;
		-h|--help)
			echo "Usage: $0 [OPTIONS]"
			echo ""
			echo "Options:"
			echo "  -y, --yes    Non-interactive mode, answer yes to all prompts"
			echo "  -h, --help   Show this help message"
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

# =========================
# Paths
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
VENV_PATH="${PROJECT_ROOT}/${VENV_NAME}"

# =========================
# Helper functions
# =========================
ask_continue() {
	local prompt="${1:-Continue?}"
	if [ "${AUTO_YES}" = true ]; then
		echo ">>> ${prompt} [Y/n] y (auto)"
		return 0
	fi
	read -rp ">>> ${prompt} [Y/n] " answer
	case "${answer}" in
	[nN] | [nN][oO])
		echo ">>> Aborted by user."
		exit 1
		;;
	*) ;;
	esac
}

# =========================
# Check / Install uv
# =========================
if command -v uv >/dev/null 2>&1; then
	echo ">>> uv found: $(uv --version)"
elif [ -x "${HOME}/.local/bin/uv" ]; then
	echo ">>> uv found at ${HOME}/.local/bin/uv"
	export PATH="${HOME}/.local/bin:${PATH}"
else
	echo ">>> uv not found."
	ask_continue "Install uv?"

	curl -LsSf "${UV_INSTALL_URL}" | sh
	export PATH="${HOME}/.local/bin:${PATH}"

	if ! command -v uv >/dev/null 2>&1; then
		echo ">>> uv installed. Please ensure ~/.local/bin is on PATH."
		echo ">>> Run: export PATH=\"\${HOME}/.local/bin:\${PATH}\""
		echo ">>> Then re-run this script."
		exit 1
	fi
	echo ">>> uv installed successfully"
fi

# =========================
# Create virtual environment
# =========================
cd "${PROJECT_ROOT}"

if [ -d "${VENV_PATH}" ]; then
	echo ">>> Found existing virtual environment: ${VENV_PATH}"
	ask_continue "Reuse existing environment?"
else
	echo ">>> Creating virtual environment: ${VENV_PATH} (Python ${PYTHON_VERSION})"
	ask_continue "Create new virtual environment?"
	uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}"
fi

# =========================
# Install Triton stack
# =========================
echo ">>> Installing Triton stack (torch, numpy, triton, cupy, datasets)"
ask_continue "Install Python packages (torch, numpy, triton, cupy, datasets)?"

if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
	uv sync --directory "${PROJECT_ROOT}"
else
	uv pip install torch numpy triton cupy datasets
fi

# =========================
# Done
# =========================
echo
echo "============================================="
echo " Triton Python environment is ready (uv)."
echo "============================================="
echo
echo "To activate the environment, run:"
echo "  source ${VENV_PATH}/bin/activate"
echo
echo "Or use uv run from project root (no activation needed):"
echo "  cd ${PROJECT_ROOT} && uv run your_script.py"
echo
echo "Installed key packages:"
echo "  - torch"
echo "  - numpy"
echo "  - triton"
echo "  - cupy"
echo "  - datasets"
echo
echo "NOTE: For CUDA-enabled torch builds, follow the PyTorch install guide"
echo "and install the matching CUDA wheel for your driver/toolkit."
echo
