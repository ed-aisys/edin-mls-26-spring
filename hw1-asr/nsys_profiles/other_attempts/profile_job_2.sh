#!/bin/bash
#SBATCH --job-name=nsys_profile_2
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=32G
#SBATCH --output=nsys_profile_2_%j.out
#SBATCH --error=nsys_profile_2_%j.err

source /opt/conda/bin/activate mls

# Execute the Nsight Systems profiling command
/usr/local/cuda/bin/nsys profile -t cuda,osrt,nvtx --stats=true --force-overwrite true -o profile_example2 python benchmark_student.py glm_asr_triton_example