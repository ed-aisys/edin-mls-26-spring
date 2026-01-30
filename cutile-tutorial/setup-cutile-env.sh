#!/usr/bin/env bash
set -eo pipefail

# =========================
# Config
# =========================
ENV_NAME="cutile"
PYTHON_VERSION="3.11"
CUDA_TAG="cuda13x"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALL_DIR="${HOME}/miniconda3"

# =========================
# Helper functions
# =========================
ask_continue() {
	local prompt="${1:-Continue?}"
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
# Detect GPU Architecture
# =========================
detect_gpu_arch() {
	echo ">>> Detecting GPU architecture..."

	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "    WARNING: nvidia-smi not found, cannot detect GPU"
		IS_BLACKWELL=false
		return
	fi

	# Get GPU name
	GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
	echo "    GPU detected: ${GPU_NAME}"

	# Blackwell GPUs: B100, B200, GB200, RTX 50xx series
	# Compute Capability 10.0+ (sm_100+)
	if echo "${GPU_NAME}" | grep -qiE "(B100|B200|GB200|RTX 50|Blackwell)"; then
		IS_BLACKWELL=true
		echo "    Architecture: Blackwell (CC 10.x)"
	else
		IS_BLACKWELL=false
		# Try to get compute capability via deviceQuery or python
		echo "    Architecture: Non-Blackwell (will use Hopper hack)"
	fi
}

# =========================
# Sanity hints (non-fatal)
# =========================
echo ">>> Assumptions:"
echo "    - NVIDIA driver >= r580 (Blackwell) or >= r550 (Hopper)"
echo "    - CUDA Toolkit >= 13.1"
echo "    - Blackwell GPU (CC 10.x) or Hopper GPU (CC 9.x with hack)"
echo

detect_gpu_arch
echo
ask_continue "Proceed with environment setup?"

# =========================
# Check / Install conda
# =========================
if command -v conda >/dev/null 2>&1; then
	echo ">>> conda found: $(conda --version)"
	eval "$(conda shell.bash hook)"
elif [ -x "${MINICONDA_INSTALL_DIR}/bin/conda" ]; then
	echo ">>> conda found at ${MINICONDA_INSTALL_DIR}/bin/conda"
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"
elif [ -x /opt/conda/bin/conda ]; then
	echo ">>> conda found at /opt/conda/bin/conda"
	eval "$(/opt/conda/bin/conda shell.bash hook)"
else
	echo ">>> conda not found."
	ask_continue "Install Miniconda to ${MINICONDA_INSTALL_DIR}?"

	MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
	curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
	bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_INSTALL_DIR}"
	rm -f "${MINICONDA_INSTALLER}"

	# Activate conda for current session
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

conda activate "${ENV_NAME}"

# =========================
# Install CUDA Toolkit
# =========================
echo ">>> Installing CUDA Toolkit from nvidia channel"
ask_continue "Install CUDA Toolkit via conda?"
conda install -y nvidia::cuda

# =========================
# Core CUDA Python stack
# =========================
echo ">>> Installing CUDA Python stack (CUDA 13)"
ask_continue "Install Python packages (cupy, cuda-python, cuda-tile)?"

# CuPy for CUDA 13
pip install "cupy-${CUDA_TAG}"

# NVIDIA CUDA Python bindings (driver/runtime API)
pip install cuda-python

# cuTile Python
pip install cuda-tile

# =========================
# CUDA Environment Variables
# =========================
echo ">>> Configuring CUDA environment variables..."

CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
if [ -n "${CONDA_ENV_PATH}" ]; then
	mkdir -p "${CONDA_ENV_PATH}/etc/conda/activate.d"
	mkdir -p "${CONDA_ENV_PATH}/etc/conda/deactivate.d"

	# Get the directory containing this script (for hack-hopper path)
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

	# Create activation script
	cat >"${CONDA_ENV_PATH}/etc/conda/activate.d/cutile_env.sh" <<EOF
#!/bin/bash
# CUDA_PATH for CuPy to find CUDA headers
export CUDA_PATH=\${CONDA_PREFIX}/targets/x86_64-linux
EOF

	# Create deactivation script
	cat >"${CONDA_ENV_PATH}/etc/conda/deactivate.d/cutile_env.sh" <<'EOF'
#!/bin/bash
unset CUDA_PATH
EOF
	echo "    CUDA_PATH configured for CuPy."
fi

# =========================
# Hopper Hack (non-Blackwell only)
# =========================
if [ "${IS_BLACKWELL}" = false ]; then
	echo
	echo ">>> Non-Blackwell GPU detected. Applying Hopper compatibility hack..."
	echo "    This uses a CuPy-based compatibility layer instead of tileiras compiler."
	ask_continue "Apply Hopper hack?"

	# Add hack-hopper to PYTHONPATH in activation script
	if [ -n "${CONDA_ENV_PATH}" ] && [ -n "${SCRIPT_DIR}" ]; then
		# Append to existing activation script
		cat >>"${CONDA_ENV_PATH}/etc/conda/activate.d/cutile_env.sh" <<EOF

# Hopper hack: use CuPy-based compatibility layer for non-Blackwell GPUs
export CUTILE_HACK_HOPPER_DIR="${SCRIPT_DIR}/hack-hopper"
export PYTHONPATH="\${CUTILE_HACK_HOPPER_DIR}:\${PYTHONPATH}"
EOF

		# Append to existing deactivation script
		cat >>"${CONDA_ENV_PATH}/etc/conda/deactivate.d/cutile_env.sh" <<'EOF'

# Remove hack-hopper from PYTHONPATH
if [ -n "${CUTILE_HACK_HOPPER_DIR}" ]; then
    PYTHONPATH="${PYTHONPATH//${CUTILE_HACK_HOPPER_DIR}:/}"
    PYTHONPATH="${PYTHONPATH//:${CUTILE_HACK_HOPPER_DIR}/}"
    PYTHONPATH="${PYTHONPATH//${CUTILE_HACK_HOPPER_DIR}/}"
fi
unset CUTILE_HACK_HOPPER_DIR
EOF
		echo "    Hopper hack installed to conda environment activation scripts."
		echo "    hack-hopper path: ${SCRIPT_DIR}/hack-hopper"
	fi
fi

# =========================
# Optional but recommended
# =========================
echo ">>> Installing optional tooling"

# NVML access (driver introspection, useful for debugging)
pip install pynvml

# NumPy (used by almost all examples)
pip install numpy

# =========================
# Freeze snapshot
# =========================
echo ">>> Writing lock snapshot (requirements.lock)"
conda list --export >requirements.lock

# =========================
# Done
# =========================
echo
echo "============================================="
echo " cuTile Python environment is ready."
echo "============================================="
echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Installed key packages:"
echo "  - nvidia::cuda (via conda)"
echo "  - cupy-${CUDA_TAG}"
echo "  - cuda-python"
echo "  - cuda-tile"
echo
echo "GPU: ${GPU_NAME:-unknown}"
if [ "${IS_BLACKWELL}" = true ]; then
	echo "Architecture: Blackwell (native support)"
else
	echo "Architecture: Non-Blackwell (CuPy-based compatibility layer)"
	echo "  PYTHONPATH includes hack-hopper on activation"
fi
echo
echo "NOTE: You may need to restart your shell or run:"
echo "  source ~/.bashrc  # or ~/.zshrc"
echo "  conda activate ${ENV_NAME}"
echo
