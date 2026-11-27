#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=00:40:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --output=results/gpu_v8_wide_20261125/bench-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

out="results/gpu_v8_wide_20261125/summary-${SLURM_JOB_ID}.csv"
samples="results/gpu_v8_wide_20261125/samples-${SLURM_JOB_ID}.csv"
binary="build/release/gpu/bench_gpu"

run_interleaved() {
  local tag="$1" M="$2" K="$3" N="$4" sparsity="$5" rounds="$6" warmup="$7" ref_rows="$8"
  "$binary" --M "$M" --K "$K" --N "$N" --s "$sparsity" --seed 42 \
    --kernels '=spr.v8.cuda@16,=spr.v8.wide.cuda@24/32' \
    --rounds "$rounds" --warmup "$warmup" --ref-rows "$ref_rows" \
    --tag "$tag" --out "$out" --samples "$samples"
}

run_isolated() {
  local tag="$1" M="$2" K="$3" N="$4" sparsity="$5" name="$6" width="$7"
  "$binary" --M "$M" --K "$K" --N "$N" --s "$sparsity" --seed 42 \
    --kernels "=$name" --k "$width" --rounds 3 --warmup 1 --ref-rows 1 \
    --tag "$tag" --out "$out" --samples "$samples"
}

run_interleaved m512_s05 512 4096 4096 0.5 31 5 4
run_interleaved m4096_s05 4096 4096 14336 0.5 9 2 4
run_interleaved m16384_s05 16384 4096 14336 0.5 5 1 2
run_interleaved m4096_s09 4096 4096 14336 0.9 9 2 4

# At M=100000 three simultaneous X/Y bindings exceed 16 GiB VRAM.
for sparsity in 0.5 0.9; do
  for pair in 'spr.v8.cuda 16' 'spr.v8.wide.cuda 24' 'spr.v8.wide.cuda 32'; do
    read -r name width <<< "$pair"
    run_isolated "m100000_s${sparsity}" 100000 4096 14336 "$sparsity" "$name" "$width"
  done
done
