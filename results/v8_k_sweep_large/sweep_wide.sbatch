#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --time=00:15:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=results/v8_k_sweep_large/wide-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

out_dir=results/v8_k_sweep_large
build/release/gpu/bench_gpu \
    --M 4096 --K 4096 --N 14336 --s 0.5 --k 9,10,11,12,13,14,15,16 --seed 42 \
    --kernels spr.v8.cuda --rounds 7 --warmup 2 --ref-rows 4 \
    --out "$out_dir/wide_comparison.csv" \
    --samples "$out_dir/wide_samples.csv"
