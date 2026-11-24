#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --partition=interactive
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=00:30:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=results/gpu_v8_correctness_20261124/verify-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

nvcc --version
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
make build-gpu CUDA_ARCH=sm_120
make test-gpu CUDA_ARCH=sm_120 TEST_ARGS="spr.v8.cuda"
make build/release/gpu/test_gpu_v8_correctness CUDA_ARCH=sm_120
build/release/gpu/test_gpu_v8_correctness
/usr/local/cuda-13.4/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/release/gpu/test_gpu_v8_correctness
/usr/local/cuda-13.4/bin/compute-sanitizer --tool initcheck --error-exitcode 99 \
  build/release/gpu/test_gpu_v8_correctness
