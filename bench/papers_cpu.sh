#!/usr/bin/env bash
# Portable v8 campaign, reusing make compare / run.sh and spr-bench / spr_bench.sh.
# Default: short verified smoke test. Full: CAMPAIGN=full bash bench/papers_cpu.sh
# D-INFK: sbatch -A COURSE_TAG --gpus=2080ti:1 --cpus-per-task=4 --time=00:30:00 \
#           --export=ALL,TARGET=dinfk,CAMPAIGN=smoke bench/papers_cpu.sh
# Official node/job docs: https://isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs
# Run on the compute node: the default Makefile compiles for its native ISA.
set -euo pipefail
# sbatch executes a spool copy, so its dirname is not the repository.
SCRIPT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -f $SCRIPT_ROOT/bench/run.sh ]]; then
  cd "$SCRIPT_ROOT"
else
  cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root.}"
  [[ -f bench/run.sh ]] || { echo "Submit from the repository root." >&2; exit 2; }
fi
PYTHON=${PYTHON:-python3}
TARGET=${TARGET:-local}
CAMPAIGN=${CAMPAIGN:-smoke}
PHASE=${PHASE:-all} # compare | roofline | all
THREADS=${THREADS:-"1 2 4"}
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
if [[ $TARGET == dinfk ]]; then
  [[ -n ${SLURM_JOB_ID:-} ]] || { echo "Run inside a Slurm job, not on the login node." >&2; exit 2; }
  grep -Eq 'Xeon.*Gold 6240 ' /proc/cpuinfo && grep -qw avx512f /proc/cpuinfo || {
    echo "Expected Xeon Gold 6240 with AVX-512; select the documented 2080ti node class." >&2; exit 2;
  }
fi
ALLOWED=$("$PYTHON" -c 'import os; n=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count(); print(min(n,int(os.environ.get("SLURM_CPUS_PER_TASK",n))))')
for t in $THREADS; do
  [[ $t =~ ^[1-9][0-9]*$ && $t -le $ALLOWED ]] || { echo "Requested $t threads, allocation allows $ALLOWED CPU slots." >&2; exit 2; }
done
case $CAMPAIGN in
  smoke) MS=${MS:-"1 64"}; SHAPES=${SHAPES:-128x256}; S=${S:-"0.5 0.9"}; KS=${KS:-1,2,4};
         SEEDS=${SEEDS:-43}; REPEATS=${REPEATS:-1}; ROUNDS=${ROUNDS:-3}; TUNE_ROUNDS=${TUNE_ROUNDS:-3};
         PREP_ROUNDS=${PREP_ROUNDS:-1}; REQUIRE_LIBS=${REQUIRE_LIBS:-0} ;;
  full) MS=${MS:-"1 64 256 512"}; SHAPES=${SHAPES:-"1024x4096 2048x8192 4096x16384"};
        S=${S:-"0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95"}; KS=${KS:-1,2,3,4,5,6,7,8,10};
        SEEDS=${SEEDS:-"43 44 45"}; REPEATS=${REPEATS:-2}; ROUNDS=${ROUNDS:-21}; TUNE_ROUNDS=${TUNE_ROUNDS:-7};
        PREP_ROUNDS=${PREP_ROUNDS:-3}; REQUIRE_LIBS=${REQUIRE_LIBS:-1} ;;
  *) echo "CAMPAIGN must be smoke or full" >&2; exit 2 ;;
esac
[[ $PHASE =~ ^(compare|roofline|all)$ ]] || { echo "PHASE must be compare, roofline or all" >&2; exit 2; }
# Additional rectangular/cache-sensitive cases reuse the existing SPR roofline workflow.
if [[ -z ${ROOF_SHAPES+x} ]]; then
  ROOF_SHAPES=$SHAPES
  [[ $CAMPAIGN != full ]] || ROOF_SHAPES="$SHAPES 4096x1024 4096x4096 4096x11008 11008x4096"
fi
export KS BASE_KS=${BASE_KS:-$KS} TUNE_SEED=${TUNE_SEED:-42} WARMUP=${WARMUP:-3}
export ROUNDS TUNE_ROUNDS PREP_ROUNDS BUDGET=${BUDGET:-0}
export SPR=spr.v8. METHODS=all
BUILD_DIR=${BUILD_DIR:-build/papers-$(hostname -s)}
RESULTS=${RESULTS:-build/results/v8-campaign}
# Exclude THREADS (a list here) from Make's scalar thread setting during build.
make -s -j"${JOBS:-$((ALLOWED < 4 ? ALLOWED : 4))}" BUILD_DIR="$BUILD_DIR" THREADS=1 build
if [[ $TARGET == dinfk ]] && grep -q SPR_AVX512_EMULATED "$BUILD_DIR/config.txt"; then
  echo "The executable was built without native AVX-512; rebuild on this compute node." >&2; exit 2
fi
if [[ $REQUIRE_LIBS == 1 ]]; then
  grep -q SPR_HAVE_BLAS "$BUILD_DIR/config.txt" && grep -q SPR_HAVE_EIGEN "$BUILD_DIR/config.txt" || {
    echo "Full comparison requires BLAS and Eigen. Configure pkg-config/build flags, or explicitly set REQUIRE_LIBS=0 and report the omission." >&2; exit 2;
  }
fi
mkdir -p "$RESULTS"
META=$(mktemp "$RESULTS/context-XXXX.json")
KEY=$("$PYTHON" bench/tune.py context "$BUILD_DIR" "$META" \
  "$CAMPAIGN|$MS|$SHAPES|$S|$KS|$BASE_KS|$THREADS|$SEEDS|$REPEATS|$ROUNDS|$TUNE_ROUNDS|$TUNE_SEED|$PREP_ROUNDS|$WARMUP|$BUDGET|$ROOF_SHAPES")
ROOT=$RESULTS/$KEY
mkdir -p "$ROOT/comparison" "$ROOT/roofline" "$ROOT/done"
mv "$META" "$ROOT/context.json"
# Preserve allocation history on restart; job IDs do not invalidate reusable tuning.
cp "$ROOT/context.json" "$ROOT/invocation-$(date +%Y%m%d-%H%M%S)-$$.json"
printf 'CAMPAIGN=%s\nMS=%s\nSHAPES=%s\nS=%s\nKS=%s\nTHREADS=%s\nSEEDS=%s\nREPEATS=%s\n' \
  "$CAMPAIGN" "$MS" "$SHAPES" "$S" "$KS" "$THREADS" "$SEEDS" "$REPEATS" > "$ROOT/campaign.txt"
# Correctness gate cached only under the exact executable/environment/campaign identity.
if [[ ! -f $ROOT/done/correctness ]]; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    "$BUILD_DIR/test_correctness" > "$ROOT/correctness.log"
  touch "$ROOT/done/correctness"
fi
tail -n 1 "$ROOT/correctness.log"
export BUILD_DIR PYTHON SPR_SKIP_BUILD=1 SPR_SKIP_TEST=1
for t in $THREADS; do
  if [[ $PHASE != roofline ]]; then
    for m in $MS; do for shape in $SHAPES; do
      for seed in $SEEDS; do for ((rep=1; rep<=REPEATS; rep++)); do
        marker=$ROOT/done/compare-t${t}-m${m}-${shape}-s${seed}-r${rep}
        [[ ! -f $marker ]] || continue
        MODE=paper M=$m SHAPES=$shape S="$S" THREADS=$t SEED=$seed TAG=seed${seed}_rep${rep} \
          RESULTS="$ROOT/comparison" bash bench/run.sh
        touch "$marker"
      done; done
    done; done
  fi
  if [[ $PHASE != compare ]]; then
    first=1
    for m in $MS; do
      marker=$ROOT/done/roofline-t${t}-m${m}
      [[ ! -f $marker ]] || continue
      MODE=full M=$m SHAPES="$ROOF_SHAPES" S="$S" THREADS=$t SEED=$TUNE_SEED PEAKS=$first REF=spr.v8.b16 \
        RESULTS="$ROOT/roofline" bash bench/spr_bench.sh
      first=0
      touch "$marker"
    done
  fi
done
"$PYTHON" bench/tune.py scaling "$ROOT"
echo "completed $PHASE: $ROOT"
