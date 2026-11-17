#!/usr/bin/env bash
#SBATCH --account=dphpc
#SBATCH --partition=jobs
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=24:00:00
#SBATCH --job-name=spr-cpu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mode=${1:-${CAMPAIGN:-pilot}}
case "$mode" in pilot|full|smoke) ;; *) echo "Use pilot, full or smoke" >&2; exit 2;; esac
if [[ -z ${SLURM_JOB_ID:-} && $mode != smoke ]]; then echo "Submit this job from the repository root with sbatch" >&2; exit 2; fi
if [[ -f build/cluster-deps/env-openmp.sh && -n ${SLURM_JOB_ID:-} ]]; then source build/cluster-deps/env-openmp.sh; fi
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LC_ALL=C
build=${BUILD_DIR:-build/papers-$(hostname -s)}
mkdir -p "${RESULTS:-build/results}"
out=$(mktemp -d "${RESULTS:-build/results}/spr-${mode}-${SLURM_JOB_ID:-local}-XXXXXX")
echo "Results: $out"
make -s -j4 BUILD_DIR="$build" build
git rev-parse HEAD > "$out/commit.txt" 2>/dev/null || echo "source copy" > "$out/commit.txt"
sha256sum "$build/bench_cpu" "$build/peaks" bench/campaign{,_plots}.py > "$out/checksums.txt"
cp "$build/config.txt" "$out/build-config.txt"
ldd "$build/bench_cpu" > "$out/libraries.txt"
OMP_THREAD_LIMIT=4 "$build/test_correctness" spr.v8 > "$out/correctness.log"
tail -1 "$out/correctness.log"
"$PYTHON" -u bench/campaign.py "$mode" "$build" "$out"
"$PYTHON" bench/campaign_plots.py "$out"
echo "Completed: $out"
