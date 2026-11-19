#!/bin/bash
#SBATCH --account=dphpc
#SBATCH --time=01:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --output=spr-gpu-%j.out

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9
cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch bench/gpu_compare.sbatch smoke|full [samples [baseline candidate]]
# Examples: smoke 5 v2 v4; full 30 v2 v4. Default pair remains v2/v3.
if (( $# == 3 || $# > 4 )); then
    echo 'Usage: sbatch bench/gpu_compare.sbatch smoke|full [samples [baseline candidate]]' >&2
    exit 2
fi
suite=${1:-smoke}
case "$suite" in
    smoke) samples=${2:-5} ;;
    full) samples=${2:-30} ;;
    *) echo 'Usage: sbatch bench/gpu_compare.sbatch smoke|full [samples [baseline candidate]]' >&2; exit 2 ;;
esac
baseline=${3:-v2}
candidate=${4:-v3}
PYTHON=${PYTHON:-python3}
output="$PWD/results/gpu_compare_${baseline}_${candidate}_${SLURM_JOB_ID}"

# All checks and timings run on the allocated GPU; kernel files are unchanged.
"$PYTHON" bench/run_gpu_compare.py --versions "$baseline" "$candidate" --suite "$suite" \
    --samples "$samples" --arch sm_120 --sanitize --output "$output"

# comparison.html always exists and uses no Python plotting packages.
if "$PYTHON" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
    "$PYTHON" bench/plot_gpu_compare.py "$output"
else
    echo 'PNG/PDF plots need numpy and matplotlib; generate them later with:'
    echo "python3 bench/plot_gpu_compare.py $output"
fi
echo "Completed: $output"
