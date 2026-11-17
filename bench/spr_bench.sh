#!/usr/bin/env bash
# SPR kernels ONLY: runtime on several matrices + roofline model of this machine.
# (make compare / bench/run.sh compares against the other methods.)
#
#   make spr-bench                              # = bench/spr_bench.sh
#   bench/spr_bench.sh                          # every spr.* kernel, quick grid
#   SPR=spr.naive,spr.v1 bench/spr_bench.sh     # only these versions
#   MODE=full bench/spr_bench.sh                # more / larger matrices, M=256
#   KS=2,3,4 THREADS=6 bench/spr_bench.sh       # k sweep / threads
#   REF=spr.v1 bench/spr_bench.sh               # speed-ups measured against spr.v1 (default spr.naive)
#   PEAKS=1 bench/spr_bench.sh                  # re-measure the machine ceilings
#   SHAPES="2048x8192" S="0.5" bench/spr_bench.sh
#
# Output: build/results/<timestamp>_spr_<mode>/{raw.csv, report.md, roofline.png, runtime.png}
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build/release}
PYTHON=${PYTHON:-python3}
RESULTS=${RESULTS:-build/results}

MODE=${MODE:-quick}
THREADS=${THREADS:-1}
SPR=${SPR:-spr.}
REF=${REF:-spr.naive}
PEAKS=${PEAKS:-0}
case $MODE in
  # square, wide, tall and LLM-like (K x N) matrices
  quick) DM=64;  DSHAPES="1024x4096 4096x1024 2048x2048 2048x8192";                       DS="0.3 0.5 0.7 0.9";          DK=2,3,4,5,6,8;   DR=7;  DB=1500 ;;
  full)  DM=256; DSHAPES="1024x4096 4096x1024 4096x4096 2048x8192 4096x11008 11008x4096"; DS="0.3 0.5 0.6 0.7 0.8 0.9 0.95"; DK=1,2,3,4,5,6,7,8,10; DR=11; DB=4000 ;;
  *) echo "MODE must be quick or full"; exit 2 ;;
esac
M=${M:-$DM}; SHAPES=${SHAPES:-$DSHAPES}; S=${S:-$DS}; KS=${KS:-$DK}; ROUNDS=${ROUNDS:-$DR}; BUDGET=${BUDGET:-$DB}

# `make spr-bench` has already built the binaries (SPR_SKIP_BUILD=1)
[[ ${SPR_SKIP_BUILD:-0} == 1 ]] || make -s BUILD_DIR="$BUILD_DIR" build > /dev/null
"$BUILD_DIR/test_correctness" $(echo "$SPR" | tr "," " " | tr -d "=") | tail -n 1   # one filter per argument

mkdir -p "$RESULTS/machine"
PK=$RESULTS/machine/$(hostname -s)_t${THREADS}.json
if [[ ! -f $PK || $PEAKS == 1 ]]; then
  echo "== measuring machine ceilings ($THREADS threads) -> $PK"
  "$BUILD_DIR/peaks" --threads "$THREADS" > "$PK"
fi

OUT=$RESULTS/$(date +%Y%m%d-%H%M%S)_spr_${MODE}
mkdir -p "$OUT"
git rev-parse --short HEAD > "$OUT/commit.txt" 2>/dev/null || true
git status --porcelain >> "$OUT/commit.txt" 2>/dev/null || true
cp "$PK" "$OUT/peaks.json"

for shape in $SHAPES; do
  echo "== M=$M K×N=$shape s={$S} threads=$THREADS"
  "$BUILD_DIR/bench_cpu" --M "$M" --K "${shape%x*}" --N "${shape#*x}" --s "${S// /,}" --k "$KS" --threads "$THREADS" \
      --kernels "$SPR" --verify "$SPR" --rounds "$ROUNDS" --warmup 1 --budget-ms "$BUDGET" --tag "spr_$MODE" \
      --out "$OUT/raw.csv" >> "$OUT/log.txt"
done

"$PYTHON" bench/roofline.py "$OUT/raw.csv" "$OUT/peaks.json" --out "$OUT" --ref "$REF"
echo; echo "saved: $OUT/report.md  $OUT/roofline.png  $OUT/runtime.png"
