#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=results/gpu_v8_k48_cublas/job-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
out="results/gpu_v8_k48_cublas/$SLURM_JOB_ID"
mkdir -p "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvcc --version
git rev-parse HEAD
sha256sum gpu/{libs.cu,spr/spr_v8_wide.cu} bench/bench.cpp include/spr/harness.hpp
make BUILD_DIR="$out/build" GPU_KERNEL_SRCS=gpu/spr/spr_v8_wide.cu CUDA_ARCH=sm_120 "$out/build/gpu/bench_gpu"
bench="$out/build/gpu/bench_gpu"
run() {
  "$bench" --M "$1" --K "$2" --N "$3" --s "$4" --k 48 --seed 42 \
    --kernels "$5" --rounds "$6" --warmup "$7" --ref-rows "$8" \
    --out "$out/summary.csv" --samples "$out/samples.csv" --tag "m${1}_k${2}_n${3}"
}
both='=spr.v8.wide.cuda,=lib.cublas.cuda'
for s in 0.5 0.9 0.99; do
  run 1 4096 14336 "$s" "$both" 21 5 1
  run 256 1024 4096 "$s" "$both" 21 5 4
  run 512 4096 4096 "$s" "$both" 21 5 4
  run 4096 4096 14336 "$s" "$both" 9 2 4
  run 16384 4096 14336 "$s" "$both" 5 2 2
done
for s in 0.5 0.9 0.99; do
  for kernel in '=lib.cublas.cuda' '=spr.v8.wide.cuda'; do
    run 100000 4096 14336 "$s" "$kernel" 3 1 1
  done
done
