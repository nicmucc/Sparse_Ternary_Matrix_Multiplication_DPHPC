#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=00:05:00
#SBATCH --gpus=5060ti:2
#SBATCH --cpus-per-task=3
#SBATCH --mem=2G
#SBATCH --output=results/gpu_multigpu_analysis_20261125/topology-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=index,name,uuid,memory.total,pcie.link.gen.max,pcie.link.width.max \
  --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p r || true
