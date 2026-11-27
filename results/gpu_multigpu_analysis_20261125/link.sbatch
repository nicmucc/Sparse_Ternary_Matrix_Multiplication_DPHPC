#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=00:10:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --output=results/gpu_multigpu_analysis_20261125/link-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,uuid,pcie.link.gen.max,pcie.link.width.max --format=csv
nvcc -O2 -std=c++17 -arch=sm_120 \
  results/gpu_multigpu_analysis_20261125/measure_link.cu \
  -o results/gpu_multigpu_analysis_20261125/measure_link
results/gpu_multigpu_analysis_20261125/measure_link
