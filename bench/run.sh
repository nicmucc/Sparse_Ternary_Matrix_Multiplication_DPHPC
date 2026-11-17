#!/usr/bin/env bash
# One command: build, check SPR is correct, benchmark SPR vs the published kernels of every
# method (and the dense libraries) on several shapes / sparsities, and write a summary.
#
#   make compare                      # = bench/run.sh, quick grid
#   MODE=paper bench/run.sh           # paper shapes (SSR / TCSC papers), much longer
#   METHODS=opt bench/run.sh          # only each method's optimized kernels; METHODS=scalar: only the scalar ones
#   THREADS=8 bench/run.sh            # threads for every kernel (OpenMP, BLAS)
#   SPR=spr.v8.b32 bench/run.sh       # only this SPR version (default: every spr.* kernel)
#   TUNE=1 bench/run.sh               # redo the k / configuration sweep of the other methods
#   KS=2,3,4 bench/run.sh             # sweep only these k for SPR (the other methods use their tuned k)
#
# Why it is fast:
#   - the other methods are TUNED ONCE per (machine, threads, shape, sparsity): every kernel of every
#     method is swept over k the first time, and the fastest (kernel, k) of each method is cached in
#     build/results/tuning/. Later runs time only that one config per method, interleaved with SPR
#     (so the comparison stays fair and fresh).
#   - only spr.* is checked for correctness here (make test checks every kernel), on 16 rows of Y.
#   - slow configs get fewer timed rounds (--budget-ms, never fewer than 3).
#
# Output: build/results/<timestamp>_<mode>/raw.csv and summary.md (also printed).
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build/release}
PYTHON=${PYTHON:-python3}
RESULTS=${RESULTS:-build/results}

MODE=${MODE:-quick}
THREADS=${THREADS:-1}
METHODS=${METHODS:-${BASELINES:-all}}
SPR=${SPR:-spr.}
TUNE=${TUNE:-0}

case $MODE in
  quick) M=64;  SHAPES="1024x4096 2048x8192";            S="0.3 0.5 0.7 0.9";                 K=2,3,4,5,6,8;        ROUNDS=7;  BUDGET=1500 ;;
  paper) M=256; SHAPES="1024x4096 2048x8192 4096x16384"; S="0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95"; K=1,2,3,4,5,6,7,8,10; ROUNDS=11; BUDGET=4000 ;;
  *) echo "MODE must be quick or paper"; exit 2 ;;
esac
SPR_K=${KS:-$K}   # k swept for SPR (the baseline tuning always uses the full K list)
[[ $METHODS =~ ^(all|scalar|opt)$ ]] || { echo "METHODS must be all, scalar or opt"; exit 2; }

# ---- every method at the scalar level and at its most optimized published level (cpu/orig/README.md).
# Used only for tuning: the summary / the cache keep the fastest (kernel, k) of each method.
# AVX-512 kernels report themselves unsupported on machines without AVX-512 and are skipped there.
T=tcsc.GEMM_CPU_FP32_colMajor_TCSC
TR=tcsc.GEMM_CPU_FP32_rowMajor_TCSC
ALL_METHODS="=rsr.rsr_inference,=rsr.rsr_pp_inference,=ssr.ssr_inference_patched"              # scalar
ALL_METHODS+=",=${T}_Naive_oneFor,=${T}_Naive_oneFor_4x4_Unroll,=lib.dense_naive"             # scalar
ALL_METHODS+=",rsrcore.,ssr.ssr_inference_float_vec,=lib.dense_blas,=lib.eigen_dense"          # RSR-core, SSR AVX2, BLAS, Eigen
for mu in 8 16 32; do for g in 4 8 16 32; do ALL_METHODS+=",=${T}_Merged_GroupMin_${mu}xG${g}_AVX2_OpenMP"; done; done
ALL_METHODS+=",=${T}_Merged_GroupMin_64xG4_AVX2_OpenMP,=${T}_Merged_GroupMid_8xG4_AVX2_OpenMP"
ALL_METHODS+=",${TR}_Merged_GroupMin_"                                                         # rowMajor GroupMin (AVX2)
for mu in 16 32;   do for g in 4 8 16 32; do ALL_METHODS+=",=${T}_Merged_GroupMin_${mu}xG${g}_AVX512_OpenMP"; done; done

# ---- build + correctness gate (a wrong SPR is never benchmarked)
[[ ${SPR_SKIP_BUILD:-0} == 1 ]] || make -s BUILD_DIR="$BUILD_DIR" build > /dev/null
"$BUILD_DIR/test_correctness" $(echo "$SPR" | tr "," " " | tr -d "=") | tail -n 1

OUT=$RESULTS/$(date +%Y%m%d-%H%M%S)_${MODE}
mkdir -p "$OUT" "$RESULTS/tuning"
git rev-parse --short HEAD > "$OUT/commit.txt" 2>/dev/null || true
git status --porcelain >> "$OUT/commit.txt" 2>/dev/null || true
CACHE=$RESULTS/tuning/$(hostname -s)_t${THREADS}.csv

export VECLIB_MAXIMUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
COMMON="--M $M --threads $THREADS --warmup 1 --tag $MODE"
for shape in $SHAPES; do
  KK=${shape%x*}; NN=${shape#*x}
  for s in $S; do
    best=$("$PYTHON" bench/tune.py get "$CACHE" $M $KK $NN $s all)
    if [[ -z $best || $TUNE == 1 ]]; then
      echo "== tuning the other methods once: M=$M K×N=$shape s=$s (cached in $CACHE)"
      "$BUILD_DIR/bench_cpu" $COMMON --K $KK --N $NN --s $s --k $K --rounds 3 --budget-ms $BUDGET --verify none \
          --out "$OUT/tuning_sweep.csv" --kernels "$ALL_METHODS" > /dev/null
      "$PYTHON" bench/tune.py add "$OUT/tuning_sweep.csv" "$CACHE"
    fi
    best=$("$PYTHON" bench/tune.py get "$CACHE" $M $KK $NN $s $METHODS)
    echo "== M=$M K×N=$shape s=$s threads=$THREADS"
    "$BUILD_DIR/bench_cpu" $COMMON --K $KK --N $NN --s $s --k $SPR_K --rounds $ROUNDS --budget-ms $BUDGET --verify "$SPR" \
        --out "$OUT/raw.csv" --kernels "$SPR,$best" >> "$OUT/log.txt"
  done
done

"$PYTHON" bench/summarize.py "$OUT/raw.csv" | tee "$OUT/summary.md"
echo; echo "saved: $OUT/summary.md"
