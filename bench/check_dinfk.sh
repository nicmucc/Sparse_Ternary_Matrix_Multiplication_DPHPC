#!/usr/bin/env bash
# Read-only preflight ON an allocated D-INFK compute node. Does not start a benchmark.
# srun -A COURSE_TAG --gpus=2080ti:1 --cpus-per-task=4 --time=00:05:00 bash bench/check_dinfk.sh
# https://isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
failed=0
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "ERROR: no Slurm allocation; do not benchmark on a login node."; failed=1; }
printf 'host=%s job=%s account=%s CPUs/task=%s memory/node=%s MiB\n' \
  "$(hostname)" "${SLURM_JOB_ID:-unset}" "${SLURM_JOB_ACCOUNT:-unset}" \
  "${SLURM_CPUS_PER_TASK:-unset}" "${SLURM_MEM_PER_NODE:-unset}"
lscpu
if ! grep -Eq 'Xeon.*Gold 6240 ' /proc/cpuinfo || ! grep -qw avx512f /proc/cpuinfo; then
  echo "ERROR: expected Xeon Gold 6240 with AVX-512. Request the 2080ti node class."; failed=1
fi
for tool in make g++ pkg-config "$PYTHON"; do
  command -v "$tool" || { echo "ERROR: missing $tool"; failed=1; }
done
g++ --version | head -n 1
macros=$(printf '' | g++ -march=native -dM -E -x c++ -)
grep -E '__AVX(2|512F)__' <<< "$macros"
grep -q '__AVX512F__' <<< "$macros" || { echo "ERROR: compiler does not enable native AVX-512."; failed=1; }
pkg-config --modversion openblas eigen3 || true
if ! pkg-config --exists openblas && [[ -z ${BLAS_FLAGS:-} || -z ${BLAS_LIBS:-} ]]; then
  echo "ERROR: OpenBLAS unavailable through pkg-config or explicit BLAS_FLAGS/BLAS_LIBS."; failed=1
fi
if ! pkg-config --exists eigen3 && [[ -z ${EIGEN_FLAGS:-} ]]; then
  echo "ERROR: Eigen unavailable through pkg-config or explicit EIGEN_FLAGS."; failed=1
fi
"$PYTHON" - <<'PY' || failed=1
import importlib, os, pathlib
allowed = sorted(os.sched_getaffinity(0))
print("Allowed CPU IDs:", allowed)
physical = set()
for cpu in allowed:
    topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        physical.add(((topology/"physical_package_id").read_text().strip(),
                      (topology/"core_id").read_text().strip()))
    except OSError:
        pass
print("Distinct physical (socket,core) IDs visible in the affinity mask:", sorted(physical))
print("Logical affinity is not proof of exclusive physical-core ownership.")
for name in ("numpy", "matplotlib"):
    module = importlib.import_module(name)
    print(name, module.__version__)
print("Cgroup membership:", pathlib.Path("/proc/self/cgroup").read_text())
for line in pathlib.Path("/proc/self/cgroup").read_text().splitlines():
    if line.startswith("0::"):
        base = pathlib.Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
            p = base/name
            if p.exists():
                print(name, p.read_text().strip())
PY
if [[ $failed == 0 ]]; then
  echo "Preflight passed. Use no more threads than allocated; other jobs may still share the node."
else
  echo "Preflight incomplete: resolve the reported items before a full performance run."
fi
exit "$failed"
