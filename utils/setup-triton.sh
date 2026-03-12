#!/usr/bin/env bash
set -eo pipefail

# =========================
# Config
# =========================
ENV_NAME="mls"  # Shared environment name
PYTHON_VERSION="3.11"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALL_DIR="${HOME}/miniconda3"
CONDA_ACTIVATE=""

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
# Check / Install conda
# =========================
if command -v conda >/dev/null 2>&1; then
	echo ">>> conda found: $(conda --version)"
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [ -n "${CONDA_BASE}" ]; then
		CONDA_ACTIVATE="${CONDA_BASE}/bin/activate"
	else
		CONDA_BIN="$(command -v conda)"
		if [[ "${CONDA_BIN}" == /* ]]; then
			CONDA_ACTIVATE="$(dirname "${CONDA_BIN}")/activate"
		fi
	fi
	eval "$(conda shell.bash hook)"
elif [ -x "${MINICONDA_INSTALL_DIR}/bin/conda" ]; then
	echo ">>> conda found at ${MINICONDA_INSTALL_DIR}/bin/conda"
	CONDA_ACTIVATE="${MINICONDA_INSTALL_DIR}/bin/activate"
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"
elif [ -x /opt/conda/bin/conda ]; then
	echo ">>> conda found at /opt/conda/bin/conda"
	CONDA_ACTIVATE="/opt/conda/bin/activate"
	eval "$(/opt/conda/bin/conda shell.bash hook)"
else
	echo ">>> conda not found."
	ask_continue "Install Miniconda to ${MINICONDA_INSTALL_DIR}?"

	MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
	curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
	bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_INSTALL_DIR}"
	rm -f "${MINICONDA_INSTALLER}"

	# Activate conda for current session
	CONDA_ACTIVATE="${MINICONDA_INSTALL_DIR}/bin/activate"
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"

	# Initialize conda for future shells (both bash and zsh)
	"${MINICONDA_INSTALL_DIR}/bin/conda" init bash
	"${MINICONDA_INSTALL_DIR}/bin/conda" init zsh
	echo ">>> Miniconda installed at ${MINICONDA_INSTALL_DIR}"
	echo ">>> Please restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
fi

# =========================
# Accept conda Terms of Service
# =========================
echo ">>> Accepting conda channel Terms of Service"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# =========================
# Create conda environment
# =========================
if conda env list | grep -q "^${ENV_NAME} "; then
	echo ">>> Found existing conda environment: ${ENV_NAME}"
	ask_continue "Reuse existing environment?"
else
	echo ">>> Will create conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
	ask_continue "Create new conda environment?"
	conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" --override-channels -c conda-forge
fi

# Activate the environment
# Note: In non-interactive shells, we need to use conda run or source activate
if [ "${AUTO_YES}" = true ]; then
	# For non-interactive mode, set up the environment path directly
	CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
	export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
	export CONDA_PREFIX="${CONDA_ENV_PATH}"
	echo ">>> Activated environment: ${ENV_NAME}"
else
	conda activate "${ENV_NAME}"
fi

if [ -z "${CONDA_ACTIVATE}" ]; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [ -n "${CONDA_BASE}" ]; then
		CONDA_ACTIVATE="${CONDA_BASE}/bin/activate"
	fi
fi

# =========================
# Install Triton stack
# =========================
echo ">>> Installing Triton stack (torch, numpy, triton, cupy, datasets)"
ask_continue "Install Python packages (torch, numpy, triton, cupy, datasets)?"

pip install --upgrade pip
pip install torch numpy triton cupy-cuda12x datasets

# =========================
# Done
# =========================
echo
echo "============================================="
echo " Triton Python environment is ready."
echo "============================================="
echo
echo "If conda isn't on PATH, run:"
echo "  source ${CONDA_ACTIVATE}"
echo "  conda activate ${ENV_NAME}"
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
