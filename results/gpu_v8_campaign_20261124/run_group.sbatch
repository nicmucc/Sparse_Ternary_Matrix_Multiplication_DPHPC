#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=02:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --output=results/gpu_v8_campaign_20261124/group-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
group="$1"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
python3 bench/run_gpu_v8_campaign.py run --group "$group"
