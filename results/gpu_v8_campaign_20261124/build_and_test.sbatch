#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=00:30:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=results/gpu_v8_campaign_20261124/build-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

nvcc --version
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
make build-gpu CUDA_ARCH=sm_120
make test-gpu CUDA_ARCH=sm_120 TEST_ARGS="spr.v8.cuda"
