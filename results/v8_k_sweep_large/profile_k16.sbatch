#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --time=00:15:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=results/v8_k_sweep_large/profile-k16-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

out_dir=results/v8_k_sweep_large
binary=results/v8_profile_M4096_K4096_N14336_k4_s05/bench_v8_ncu
profiler=/cluster/data/cuda/13.4.1/bin/ncu
report="$out_dir/v8_K4096_N14336_M4096_k16_z0.5"

"$binary" v8 4096 4096 14336 16 0.5 42 1 1 > "$out_dir/k16_smoke.log" 2>&1
"$profiler" --set full --cache-control none \
    --launch-skip 5 --launch-count 1 \
    --force-overwrite --export "$report" \
    "$binary" v8 4096 4096 14336 16 0.5 42 1 1 \
    > "$out_dir/k16_ncu.log" 2>&1
