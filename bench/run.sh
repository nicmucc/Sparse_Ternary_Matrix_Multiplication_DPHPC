#!/usr/bin/env bash
# Verified comparison through the existing Kernel API. Default SPR: final v8 family.
# MODE=quick|paper, M, SHAPES="KxN ...", S="0.5 0.9", KS, BASE_KS,
# THREADS, SEED, TUNE_SEED, ROUNDS, TUNE_ROUNDS, WARMUP, PREP_ROUNDS, BUDGET.
# Tuning picks one (variant,k) per method, including v8; final runs are independent.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build/release}
PYTHON=${PYTHON:-python3}
RESULTS=${RESULTS:-build/results}
MODE=${MODE:-quick}
THREADS=${THREADS:-1}
METHODS=${METHODS:-${BASELINES:-all}}
SPR=${SPR:-spr.v8.}
TUNE=${TUNE:-0}
case $MODE in
  quick) M=${M:-64}; SHAPES=${SHAPES:-"1024x4096 2048x8192"}; S=${S:-"0.3 0.5 0.7 0.9"}; K=1,2,3,4,5,6,7,8; ROUNDS=${ROUNDS:-11} ;;
  paper) M=${M:-256}; SHAPES=${SHAPES:-"1024x4096 2048x8192 4096x16384"}; S=${S:-"0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95"}; K=1,2,3,4,5,6,7,8,10; ROUNDS=${ROUNDS:-21} ;;
  *) echo "MODE must be quick or paper"; exit 2 ;;
esac
KS=${KS:-$K}; BASE_KS=${BASE_KS:-$KS}
SEED=${SEED:-42}; TUNE_SEED=${TUNE_SEED:-42}
TUNE_ROUNDS=${TUNE_ROUNDS:-7}; WARMUP=${WARMUP:-3}; PREP_ROUNDS=${PREP_ROUNDS:-3}
BUDGET=${BUDGET:-0} # 0 keeps the requested sample count, even for slow methods.
TAG=${TAG:-seed${SEED}}
[[ $METHODS =~ ^(all|scalar|opt)$ ]] || { echo "METHODS must be all, scalar or opt"; exit 2; }
# All registered general-ternary FP32 baselines; original scalar SSR is known broken.
# Tune families separately to limit resident memory, then interleave ALL winners together.
BASE_GROUPS=("rsr." "rsrcore." "=ssr.ssr_inference_patched,ssr.ssr_inference_float_vec"
             "tcsc.GEMM_CPU_FP32_colMajor_" "tcsc.GEMM_CPU_FP32_rowMajor_" "tcsc.GEMV_" "lib.")
[[ ${SPR_SKIP_BUILD:-0} == 1 ]] || make -s BUILD_DIR="$BUILD_DIR" build > /dev/null
mkdir -p "$RESULTS/tuning"
OUT=$(mktemp -d "$RESULTS/$(date +%Y%m%d-%H%M%S)_${MODE}_t${THREADS}_XXXX")
export OMP_NUM_THREADS=$THREADS OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
CONTEXT=$("$PYTHON" bench/tune.py context "$BUILD_DIR" "$OUT/context.json" \
  "threads=$THREADS|SPR=$SPR|KS=$KS|BASE_KS=$BASE_KS|tune_seed=$TUNE_SEED|rounds=$TUNE_ROUNDS|warmup=$WARMUP|budget=$BUDGET")
CACHE=$RESULTS/tuning/$CONTEXT.csv
cp "$BUILD_DIR/config.txt" "$OUT/build-config.txt"
printf 'M=%s\nSHAPES=%s\nS=%s\nSEED=%s\nTAG=%s\nROUNDS=%s\nPREP_ROUNDS=%s\n' \
  "$M" "$SHAPES" "$S" "$SEED" "$TAG" "$ROUNDS" "$PREP_ROUNDS" > "$OUT/measurement.txt"
git rev-parse HEAD > "$OUT/commit.txt"
git status --porcelain >> "$OUT/commit.txt"
if [[ ${SPR_SKIP_TEST:-0} != 1 ]]; then
  "$BUILD_DIR/test_correctness" > "$OUT/correctness.log"
  tail -n 1 "$OUT/correctness.log"
fi
COMMON=(--M "$M" --threads "$THREADS" --warmup "$WARMUP" --ref-rows -1)
for shape in $SHAPES; do
  KK=${shape%x*}; NN=${shape#*x}
  for s in $S; do
    best=$("$PYTHON" bench/tune.py get "$CACHE" "$M" "$KK" "$NN" "$s" all)
    if [[ -z $best || $TUNE == 1 ]]; then
      echo "== tuning M=$M KxN=$shape s=$s threads=$THREADS"
      sweep=$OUT/tuning_${M}_${KK}_${NN}_${s}.csv
      for group in "${BASE_GROUPS[@]}" "$SPR"; do
        grid=$BASE_KS; [[ $group != "$SPR" ]] || grid=$KS
        "$BUILD_DIR/bench_cpu" "${COMMON[@]}" --K "$KK" --N "$NN" --s "$s" --k "$grid" \
          --seed "$TUNE_SEED" --tag tuning --rounds "$TUNE_ROUNDS" --budget-ms "$BUDGET" \
          --out "$sweep" --kernels "$group" >> "$OUT/tuning.log"
      done
      "$PYTHON" bench/tune.py add "$sweep" "$CACHE" --include-spr
      "$PYTHON" bench/tune.py edges "$sweep" >> "$OUT/tuning-boundaries.txt"
    fi
    best=$("$PYTHON" bench/tune.py get "$CACHE" "$M" "$KK" "$NN" "$s" "$METHODS")
    [[ -n $best ]] || { echo "no verified tuning selections" >&2; exit 1; }
    echo "== measuring M=$M KxN=$shape s=$s threads=$THREADS seed=$SEED"
    "$BUILD_DIR/bench_cpu" "${COMMON[@]}" --K "$KK" --N "$NN" --s "$s" --k "$KS" \
      --seed "$SEED" --tag "$TAG" --rounds "$ROUNDS" --budget-ms "$BUDGET" --prep-rounds "$PREP_ROUNDS" \
      --out "$OUT/raw.csv" --samples "$OUT/samples.csv" --kernels "$best" >> "$OUT/log.txt"
  done
done
"$PYTHON" bench/tune.py check "$OUT/raw.csv"
cp "$CACHE" "$OUT/selected.csv"
"$PYTHON" bench/summarize.py "$OUT/raw.csv" > "$OUT/summary.md"
echo "saved: $OUT"
