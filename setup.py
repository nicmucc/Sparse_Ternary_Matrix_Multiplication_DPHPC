"""Build the PyTorch extension that exposes the existing C++ baselines."""

from pathlib import Path
import shlex
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent


def pkg_config(option: str, *packages: str) -> list[str]:
    """Return one category of pkg-config flags as individual tokens."""
    try:
        output = subprocess.check_output(
            ["pkg-config", option, *packages],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        names = ", ".join(packages)
        raise RuntimeError(
            f"pkg-config could not find the required packages: {names}"
        ) from error
    return shlex.split(output)


include_dirs = [str(ROOT / "include")]
for flag in pkg_config("--cflags-only-I", "eigen3", "openblas"):
    if flag.startswith("-I"):
        include_dirs.append(flag[2:])

library_dirs = [
    flag[2:]
    for flag in pkg_config("--libs-only-L", "openblas")
    if flag.startswith("-L")
]
libraries = [
    flag[2:]
    for flag in pkg_config("--libs-only-l", "openblas")
    if flag.startswith("-l")
]

sources = [
    "python/csrc/bindings.cpp",
    "src/baselines/blas_baseline.cpp",
    "src/baselines/eigen_baseline.cpp",
    "src/baselines/naive.cpp",
    "src/baselines/rsr.cpp",
    "src/baselines/rsrpp.cpp",
    "src/baselines/ssr_baseline.cpp",
    "src/baselines/tcsc_baseline.cpp",
    "src/utils/utils.cpp",
]

setup(
    name="stmm-benchmark",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CppExtension(
            name="stmm_benchmark._C",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            # Current PyTorch headers require C++20. The standalone project
            # remains C++17; only extension translation units use this flag.
            extra_compile_args=["-O3", "-std=c++20"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
