#!/usr/bin/env bash
set -eo pipefail

# Activate conda mls environment
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate mls

echo ">>> Uninstalling cu12 torch and nvidia packages..."
pip uninstall -y torch nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
    nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 \
    cuda-bindings cuda-pathfinder 2>/dev/null || true

echo ">>> Installing torch with CUDA 13.0..."
pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130

echo ">>> Installing remaining dependencies..."
pip install triton==3.6.0 transformers safetensors huggingface_hub soundfile numpy

echo ">>> Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('(No GPU on head node - this is expected)')
import triton
print(f'Triton: {triton.__version__}')
"

echo ">>> Done! Now run the benchmark with:"
echo "    srun -p Teaching -w saxa --gres gpu:1 --mem=16G --pty bash"
echo "    conda activate mls"
echo "    cd hw1-asr"
echo "    python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5"
