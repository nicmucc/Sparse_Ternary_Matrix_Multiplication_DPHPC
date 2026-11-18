cmake_minimum_required(VERSION 3.18)
project(spr LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Same flags for EVERY kernel, ours and theirs (fairness).
# The TCSC paper used MSVC /Ox /fp:fast /openmp + AVX2/AVX-512; this is the clang/gcc equivalent.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(SPR_FLAGS -O3 -ffast-math -DNDEBUG -mcpu=native)
  set(SPR_SHIM ON)       # AVX2/AVX-512 intrinsics -> portable vectors (cpu/orig/shim)
else()
  set(SPR_FLAGS -O3 -ffast-math -DNDEBUG -march=native)
  set(SPR_SHIM OFF)      # real <immintrin.h>
endif()

# ---------------------------------------------------------------- OpenMP (macOS: brew install libomp)
if(APPLE AND NOT OpenMP_CXX_FOUND)
  set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
if(APPLE AND NOT OpenMP_C_FOUND)
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_C_LIB_NAMES "omp")
endif()
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------- library baselines
if(APPLE)
  find_library(SPR_BLAS_LIBS Accelerate)
  set(SPR_HAVE_BLAS ON)
else()
  find_package(BLAS)
  if(BLAS_FOUND)
    set(SPR_BLAS_LIBS ${BLAS_LIBRARIES})
    set(SPR_HAVE_BLAS ON)
  endif()
endif()
find_package(Eigen3 QUIET NO_MODULE)

# ---------------------------------------------------------------- the authors' code, UNMODIFIED
set(UP ${CMAKE_SOURCE_DIR}/third_party/SSR-ternaryLLM-repo)
set(TCSC_DIR ${UP}/ternaryLLM_CPU/src)
set(SSR_DIR  ${UP}/SSR/SSR)
set(RSR_DIR  ${UP}/SSR/SSR/original_rsr)
set(UPSTREAM_SOURCES
  ${TCSC_DIR}/GEMM_CPU_FP32.cpp
  ${SSR_DIR}/ssr.cpp
  ${SSR_DIR}/ssr_vec.cpp
  ${RSR_DIR}/rsr.cpp
  ${RSR_DIR}/utils.cpp
  ${RSR_DIR}/naive.cpp
)
add_library(upstream OBJECT ${UPSTREAM_SOURCES})
target_compile_options(upstream PRIVATE ${SPR_FLAGS} -w)
target_link_libraries(upstream PUBLIC OpenMP::OpenMP_CXX)
if(SPR_SHIM)
  target_include_directories(upstream PRIVATE ${CMAKE_SOURCE_DIR}/cpu/orig/shim)
endif()

# RSR-core (UIC-InDeXLab/RSR-core @ 99bcccd): the authors' optimized ternary RSR kernels, plain C.
# Own target WITHOUT the AVX shim: these files have native NEON and AVX2 paths.
set(RSRCORE_DIR ${CMAKE_SOURCE_DIR}/third_party/RSR-core-repo/kernels/bit_1_58/cpu)
add_library(upstream_rsrcore OBJECT
  ${RSRCORE_DIR}/rsr_ternary_prep_nonsquare.c
  ${RSRCORE_DIR}/rsr_ternary_v3_1.c
  ${RSRCORE_DIR}/rsr_ternary_v3_3.c
)
target_compile_options(upstream_rsrcore PRIVATE ${SPR_FLAGS} -w)
target_link_libraries(upstream_rsrcore PUBLIC OpenMP::OpenMP_C)

# ---------------------------------------------------------------- our harness + wrappers + kernels
add_library(kernels OBJECT
  cpu/core.cpp
  cpu/libs.cpp
  cpu/orig/tcsc.cpp
  cpu/orig/ssr.cpp
  cpu/orig/rsr.cpp
  cpu/orig/rsrcore.cpp
  # SPR kernels: add every new version here
  cpu/spr/spr_naive.cpp
  cpu/spr/spr_v0.cpp
  cpu/spr/spr_v1.cpp
  cpu/spr/spr_v2.cpp
  cpu/spr/spr_v3.cpp
  cpu/spr/spr_v4.cpp
  cpu/spr/spr_v5.cpp
  cpu/spr/spr_v6.cpp
  cpu/spr/spr_v7.cpp
  cpu/spr/spr_v8.cpp
)
target_include_directories(kernels PUBLIC include)
target_compile_options(kernels PUBLIC ${SPR_FLAGS})
target_link_libraries(kernels PUBLIC OpenMP::OpenMP_CXX ${SPR_BLAS_LIBS})
if(SPR_HAVE_BLAS)
  target_compile_definitions(kernels PUBLIC SPR_HAVE_BLAS)
endif()
if(Eigen3_FOUND)
  target_compile_definitions(kernels PUBLIC SPR_HAVE_EIGEN)
  target_link_libraries(kernels PUBLIC Eigen3::Eigen)
endif()
# each wrapper only sees the headers of the repo it wraps (they reuse names like utils.h)
set_source_files_properties(cpu/orig/tcsc.cpp PROPERTIES INCLUDE_DIRECTORIES "${TCSC_DIR};${CMAKE_SOURCE_DIR}/cpu/orig/shim" COMPILE_OPTIONS "-w")
set_source_files_properties(cpu/orig/ssr.cpp  PROPERTIES INCLUDE_DIRECTORIES "${SSR_DIR}" COMPILE_OPTIONS "-w")
set_source_files_properties(cpu/orig/rsr.cpp  PROPERTIES INCLUDE_DIRECTORIES "${RSR_DIR}" COMPILE_OPTIONS "-w")
if(NOT SPR_SHIM)
  set_source_files_properties(cpu/orig/tcsc.cpp PROPERTIES INCLUDE_DIRECTORIES "${TCSC_DIR}")
endif()

add_executable(test_correctness tests/test_correctness.cpp $<TARGET_OBJECTS:upstream> $<TARGET_OBJECTS:upstream_rsrcore>)
target_link_libraries(test_correctness PRIVATE kernels OpenMP::OpenMP_C)

add_executable(bench_cpu bench/bench_cpu.cpp $<TARGET_OBJECTS:upstream> $<TARGET_OBJECTS:upstream_rsrcore>)
target_link_libraries(bench_cpu PRIVATE kernels OpenMP::OpenMP_C)

# Machine ceilings for the roofline (bench/roofline.py). No -ffast-math and no auto-vectorization on purpose:
# the scalar loop must stay scalar; the SIMD loops use intrinsics.
add_executable(peaks bench/peaks.cpp)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_compile_options(peaks PRIVATE -O2 -mcpu=native -fno-vectorize -fno-slp-vectorize)
else()
  target_compile_options(peaks PRIVATE -O2 -march=native -fno-vectorize -fno-slp-vectorize)
endif()
target_link_libraries(peaks PRIVATE OpenMP::OpenMP_CXX)

# Knock-out profiler of spr.v1's inference loop (PROFILING.md). Same flags as the kernels.
add_executable(prof_v1 bench/prof_v1.cpp cpu/core.cpp)
target_include_directories(prof_v1 PRIVATE include ${CMAKE_SOURCE_DIR})
target_compile_options(prof_v1 PRIVATE ${SPR_FLAGS})
target_link_libraries(prof_v1 PRIVATE OpenMP::OpenMP_CXX)
