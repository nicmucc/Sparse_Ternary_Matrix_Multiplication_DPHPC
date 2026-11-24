#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --time=00:15:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=results/v8_k_sweep_large/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

out_dir=results/v8_k_sweep_large
build/release/gpu/bench_gpu \
    --M 4096 --K 4096 --N 14336 --s 0.5 --k 4,5,6,7,8 --seed 42 \
    --kernels spr.v8.cuda --rounds 9 --warmup 2 --ref-rows 4 \
    --out "$out_dir/comparison.csv" \
    --samples "$out_dir/samples.csv"
