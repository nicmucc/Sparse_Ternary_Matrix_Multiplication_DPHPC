#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --time=00:30:00
#SBATCH --gpus=2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --output=spr-gpu-%j.out

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9
cd "$SLURM_SUBMIT_DIR"

# All versions run on the same allocated GPU. No production kernel files change.
python3 bench/run_gpu_compare.py --suite full --samples 30 --profile
