#!/bin/bash

# --- Slurm Resource Requests ---
#SBATCH --job-name=nsys_glm_triton
#SBATCH --partition=Teaching         # Explicitly request the Teaching queue
#SBATCH --nodelist=saxa              # Force the job onto the saxa node
#SBATCH --gres=gpu:3g.71gb:1         # Request the massive 71GB MIG slice
#SBATCH --mem=32G                    # Request 32GB of RAM
#SBATCH --time=02:00:00              # Adjust time limit if needed
#SBATCH --output=slurm_nsys_%j.out   # Standard output log
#SBATCH --error=slurm_nsys_%j.err    # Standard error log

echo "Starting Nsight Systems profiling on node: $SLURM_JOB_NODELIST"
source /opt/conda/bin/activate mls
# --- Profiling Command ---
/usr/local/cuda/bin/nsys profile \
    -t cuda,osrt,nvtx \
    --stats=true \
    --force-overwrite true \
    -o profile_template2 \
    python benchmark_student.py glm_asr_triton_template

echo "Profiling complete. Check profile_template2.nsys-rep and slurm logs."