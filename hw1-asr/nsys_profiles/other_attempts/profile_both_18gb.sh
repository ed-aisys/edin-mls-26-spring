#!/bin/bash

# --- Slurm Resource Requests ---
#SBATCH --job-name=nsys_both_18gb    # Changed the job name so you can tell them apart in squeue
#SBATCH --partition=Teaching         # Explicitly request the Teaching queue
#SBATCH --nodelist=saxa              # Force the job onto the saxa node
#SBATCH --gres=gpu:1g.18gb:1         # Request the standard 18GB MIG slice
#SBATCH --mem=32G                    # Request 32GB of RAM
#SBATCH --time=02:00:00              # Time limit (2 hours is plenty for both runs)
#SBATCH --output=slurm_18gb_%j.out   # Standard output log (renamed for clarity)
#SBATCH --error=slurm_18gb_%j.err    # Standard error log (renamed for clarity)

echo "Starting Nsight Systems profiling on node: $SLURM_JOB_NODELIST"

# Activate your environment
source /opt/conda/bin/activate mls

# --- Profiling Command 1: The Example ---
echo "Running profiling for: glm_asr_triton_example..."
/usr/local/cuda/bin/nsys profile \
    -t cuda,osrt,nvtx \
    --stats=true \
    --force-overwrite true \
    -o profile_example_18gb \
    python benchmark_student.py glm_asr_triton_example

# --- Profiling Command 2: The Template ---
echo "Running profiling for: glm_asr_triton_template..."
/usr/local/cuda/bin/nsys profile \
    -t cuda,osrt,nvtx \
    --stats=true \
    --force-overwrite true \
    -o profile_template_18gb \
    python benchmark_student.py glm_asr_triton_template

echo "Both profiling runs complete! Check the new .nsys-rep files."