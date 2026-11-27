#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=00:20:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=results/gpu_multigpu_analysis_20261125/n_sweep-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
out="results/gpu_multigpu_analysis_20261125/n_sweep-${SLURM_JOB_ID}.csv"
samples="results/gpu_multigpu_analysis_20261125/n_sweep_samples-${SLURM_JOB_ID}.csv"

for M in 512 4096; do
  if (( M == 512 )); then rounds=21; warmup=3; else rounds=9; warmup=2; fi
  for N in 4096 14336 28672; do
    build/release/gpu/bench_gpu --M "$M" --K 4096 --N "$N" --s 0.5 \
      --k 32 --seed 42 --kernels '=spr.v8.wide.cuda' --ref-rows 4 \
      --rounds "$rounds" --warmup "$warmup" \
      --tag "m${M}_n${N}" --out "$out" --samples "$samples"
  done
done
