# Run `make help` for the list of targets.

# --- Settings (override on the command line) --------------------------------

# release | sanitizer
PROFILE   ?= release
BUILD_DIR ?= build/$(PROFILE)
PYTHON    ?= python3
NVCC      ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -lineinfo -std=c++17
THREADS   ?= 1

TEST_ARGS  ?=
BENCH_ARGS ?=

DATA_DIR      ?= data/benchmarks
GENERATE_ARGS ?=
OVERWRITE     ?= 0

# --- Flags -------------------------------------------------------------------
# One set of flags for EVERY kernel (ours and the papers'), as in the papers: host ISA
# (AVX2 / AVX-512 on x86) and fast-math (the TCSC paper used MSVC /Ox /fp:fast).

ARCH      := $(shell uname -m)
NATIVE    := $(if $(filter arm64 aarch64,$(ARCH)),-mcpu=native,-march=native)
CXX_ID    := $(shell $(CXX) --version 2>/dev/null | head -1)

ifeq ($(PROFILE),release)
  OPT_FLAGS := -O3 -ffast-math -DNDEBUG $(NATIVE)
else ifeq ($(PROFILE),sanitizer)
  OPT_FLAGS := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(NATIVE)
else
  $(error PROFILE must be release or sanitizer)
endif

# OpenMP everywhere (the papers' optimized kernels and SPR v5+ are multi-threaded).
# Apple clang has no -fopenmp driver flag: use Homebrew's libomp (brew install libomp).
ifneq ($(findstring Apple clang,$(CXX_ID)),)
  LIBOMP    ?= $(shell brew --prefix libomp 2>/dev/null)
  OMP_FLAGS ?= -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMP_LIBS  ?= -L$(LIBOMP)/lib -lomp -Wl,-rpath,$(LIBOMP)/lib
endif
OMP_FLAGS ?= -fopenmp
OMP_LIBS  ?= -fopenmp

# Dense baselines: Accelerate on macOS, OpenBLAS elsewhere (if found); Eigen if found.
ifeq ($(shell uname -s),Darwin)
  BLAS_FLAGS ?= -DSPR_HAVE_BLAS
  BLAS_LIBS  ?= -framework Accelerate
else ifneq ($(shell pkg-config --exists openblas 2>/dev/null && echo yes),)
  BLAS_FLAGS ?= -DSPR_HAVE_BLAS $(shell pkg-config --cflags openblas)
  BLAS_LIBS  ?= $(shell pkg-config --libs openblas)
endif
ifneq ($(shell pkg-config --exists eigen3 2>/dev/null && echo yes),)
  EIGEN_FLAGS ?= -DSPR_HAVE_EIGEN $(shell pkg-config --cflags eigen3)
endif

CXXFLAGS ?= $(OPT_FLAGS) -std=c++17 -Wall -Wextra
CFLAGS   ?= $(OPT_FLAGS) -std=c11
CPPFLAGS += -Iinclude

# Machine ceilings (peaks): host ISA, no auto-vectorization, no fast-math, so the scalar loop
# stays scalar and the SIMD loops use intrinsics.
ifneq ($(findstring clang,$(CXX_ID)),)
  NOVEC := -fno-vectorize -fno-slp-vectorize
else
  NOVEC := -fno-tree-vectorize -fno-tree-slp-vectorize
endif
PEAKS_FLAGS := -O2 -std=c++17 $(NATIVE) $(NOVEC)

# --- Third-party sources, compiled UNMODIFIED (see third_party/README.md) ----

TP       := third_party
TCSC_DIR := $(TP)/SSR-ternaryLLM-repo/ternaryLLM_CPU/src
SSR_DIR  := $(TP)/SSR-ternaryLLM-repo/SSR/SSR
RSR_DIR  := $(SSR_DIR)/original_rsr
RSRC_DIR := $(TP)/RSR-core-repo/kernels

UPSTREAM_CXX := $(TCSC_DIR)/GEMM_CPU_FP32.cpp $(SSR_DIR)/ssr.cpp $(SSR_DIR)/ssr_vec.cpp \
                $(RSR_DIR)/rsr.cpp $(RSR_DIR)/utils.cpp $(RSR_DIR)/naive.cpp
UPSTREAM_C   := $(RSRC_DIR)/bit_1_58/cpu/rsr_ternary_prep_nonsquare.c \
                $(RSRC_DIR)/bit_1_58/cpu/rsr_ternary_v3_1.c $(RSRC_DIR)/bit_1_58/cpu/rsr_ternary_v3_3.c \
                $(RSRC_DIR)/bit_1/cpu/rsr_prep_nonsquare.c $(RSRC_DIR)/bit_1/cpu/rsr_v4_2.c

# SSR / TCSC are written with AVX2 / AVX-512 intrinsics.
#  - x86: cpu/orig/x86_compat wraps the real <immintrin.h>: aligned index loads become unaligned
#    (the authors' MSVC build did that implicitly), and without AVX-512 it lets GEMM_CPU_FP32.cpp
#    compile; its AVX-512 kernels then report themselves unsupported (SPR_AVX512_EMULATED).
#  - Apple Silicon: cpu/orig/shim maps them to portable vectors (development only; timings meaningless).
ifneq ($(filter arm64 aarch64,$(ARCH)),)
  AVX_COMPAT := -Icpu/orig/shim
else ifeq ($(shell $(CXX) $(NATIVE) -dM -E -x c++ /dev/null 2>/dev/null | grep -c __AVX512F__),0)
  AVX_COMPAT := -Icpu/orig/x86_compat -DSPR_AVX512_EMULATED
else
  AVX_COMPAT := -Icpu/orig/x86_compat
endif

# --- Files -------------------------------------------------------------------

OBJ_DIR      := $(BUILD_DIR)/obj
KERNEL_SRCS  := cpu/core.cpp cpu/libs.cpp $(wildcard cpu/orig/*.cpp) $(wildcard cpu/spr/*.cpp)
SOURCES      := $(KERNEL_SRCS) $(UPSTREAM_CXX) $(UPSTREAM_C)
OBJECTS      := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(filter %.cpp,$(SOURCES))) \
                $(patsubst %.c,$(OBJ_DIR)/%.o,$(filter %.c,$(SOURCES)))
TEST_MAIN    := $(OBJ_DIR)/tests/test_correctness.o
# The benchmark adapter compiles Ahmed's original v8 source plus the fixed-tile factory.
BENCH_OBJECTS := $(filter-out $(OBJ_DIR)/cpu/spr/spr_v8.o,$(OBJECTS)) $(OBJ_DIR)/bench/v8_adapter.o
BENCH_MAIN   := $(OBJ_DIR)/bench/bench.o
ifeq ($(shell uname -s),Linux)
ifeq ($(findstring clang,$(CXX_ID)),)
  BENCH_TEAM_LINK := -Wl,--wrap=GOMP_parallel
  $(BENCH_MAIN): EXTRA := -DSPR_OBSERVE_GOMP
endif
endif
# Preserve NaN checks in the original verifier under the release fast-math flags.
$(BENCH_MAIN): EXTRA += -DSPR_BENCH_CPU -fno-finite-math-only

TEST_BIN     := $(BUILD_DIR)/test_correctness
BENCH_BIN    := $(BUILD_DIR)/bench_cpu
PEAKS_BIN    := $(BUILD_DIR)/peaks
BUILD_CONFIG := $(BUILD_DIR)/config.txt

GPU_BUILD_DIR := $(BUILD_DIR)/gpu
GPU_TEST_BIN := $(GPU_BUILD_DIR)/test_correctness
GPU_BENCH_BIN := $(GPU_BUILD_DIR)/bench_gpu
GPU_BENCH_SRCS := cpu/core.cpp bench/bench.cpp $(wildcard gpu/spr/*.cu)
GPU_SRCS := cpu/core.cpp tests/test_correctness.cpp $(wildcard gpu/spr/*.cu)
GPU_HEADERS := $(wildcard include/spr/*.hpp)

RUN_ENV := OMP_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS) VECLIB_MAXIMUM_THREADS=$(THREADS)

# Per-file flags: each wrapper / upstream file only sees the headers of the repo it comes from
# (the repos reuse names such as utils.h).
$(OBJ_DIR)/$(TCSC_DIR)/%.o $(OBJ_DIR)/$(SSR_DIR)/%.o:  EXTRA := -w $(AVX_COMPAT)
$(OBJ_DIR)/$(RSRC_DIR)/%.o:                             EXTRA := -w
$(OBJ_DIR)/cpu/orig/tcsc.o:  EXTRA := -w -I$(TCSC_DIR) $(AVX_COMPAT)
$(OBJ_DIR)/cpu/orig/ssr.o:   EXTRA := -w -I$(SSR_DIR) $(AVX_COMPAT)
$(OBJ_DIR)/cpu/orig/rsr.o:   EXTRA := -w -I$(RSR_DIR)
$(OBJ_DIR)/cpu/libs.o:       EXTRA := $(BLAS_FLAGS) $(EIGEN_FLAGS)

# Force a rebuild when configuration changes, even within one timestamp tick.
BUILD_STATE := $(CXX) | $(CC) | $(CPPFLAGS) | $(CXXFLAGS) | $(CFLAGS) | $(LDFLAGS) | $(OMP_FLAGS) | $(OMP_LIBS) \
               | $(BLAS_FLAGS) | $(BLAS_LIBS) | $(EIGEN_FLAGS) | $(AVX_COMPAT) | $(SOURCES) | $(CXX_ID)
BUILD_CHANGED :=
ifneq ($(BUILD_STATE),$(shell cat "$(BUILD_CONFIG)" 2>/dev/null))
  BUILD_CHANGED := FORCE
endif

# --- Build -------------------------------------------------------------------

.DEFAULT_GOAL := build
.PHONY: build test bench plots data clean help FORCE build-gpu test-gpu bench-gpu

build: $(TEST_BIN) $(BENCH_BIN) $(PEAKS_BIN)

$(BUILD_CONFIG): $(BUILD_CHANGED)
	@rm -rf "$(OBJ_DIR)" "$(TEST_BIN)" "$(BENCH_BIN)" "$(PEAKS_BIN)"
	@mkdir -p "$(@D)"
	@printf '%s' '$(BUILD_STATE)' > "$@"

$(OBJ_DIR)/%.o: %.cpp $(BUILD_CONFIG) $(BUILD_CHANGED)
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMP_FLAGS) $(EXTRA) -MMD -MP -c "$<" -o "$@"

$(OBJ_DIR)/%.o: %.c $(BUILD_CONFIG) $(BUILD_CHANGED)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(EXTRA) -MMD -MP -c "$<" -o "$@"

$(TEST_BIN): $(OBJECTS) $(TEST_MAIN) $(BUILD_CHANGED)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(LDFLAGS) $(filter %.o,$^) $(BLAS_LIBS) $(OMP_LIBS) -o "$@"

$(BENCH_BIN): $(BENCH_OBJECTS) $(BENCH_MAIN) $(BUILD_CHANGED)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(LDFLAGS) $(BENCH_TEAM_LINK) $(filter %.o,$^) $(BLAS_LIBS) $(OMP_LIBS) -o "$@"

$(PEAKS_BIN): bench/peaks.cpp $(BUILD_CONFIG) $(BUILD_CHANGED)
	@mkdir -p "$(@D)"
	$(CXX) $(PEAKS_FLAGS) $(OMP_FLAGS) "$<" $(OMP_LIBS) -o "$@"

-include $(OBJ_DIR)/bench/v8_adapter.d $(OBJECTS:.o=.d) $(TEST_MAIN:.o=.d) $(BENCH_MAIN:.o=.d)

build-gpu: $(GPU_TEST_BIN) $(GPU_BENCH_BIN)

$(GPU_TEST_BIN): $(GPU_SRCS) $(GPU_HEADERS) Makefile
	@mkdir -p "$(@D)"
	$(NVCC) -ccbin "$(CXX)" $(CPPFLAGS) $(NVCCFLAGS) \
		-arch=$(CUDA_ARCH) -Xcompiler=-fopenmp \
		$(GPU_SRCS) -o "$@"

$(GPU_BENCH_BIN): $(GPU_BENCH_SRCS) $(GPU_HEADERS) Makefile
	@mkdir -p "$(@D)"
	$(NVCC) -ccbin "$(CXX)" $(CPPFLAGS) $(NVCCFLAGS) \
		-arch=$(CUDA_ARCH) -Xcompiler=-fopenmp \
		$(GPU_BENCH_SRCS) -o "$@"

bench-gpu: $(GPU_BENCH_BIN)
	$(RUN_ENV) "$(GPU_BENCH_BIN)" $(BENCH_ARGS)

test-gpu: $(GPU_TEST_BIN)
	"$(GPU_TEST_BIN)" $(TEST_ARGS)

# --- Test and benchmark ------------------------------------------------------

# TEST_ARGS: kernel-name filters, e.g. TEST_ARGS="spr. tcsc."
test: $(TEST_BIN)
	"$(TEST_BIN)" $(TEST_ARGS)

# The raw benchmark driver, e.g. BENCH_ARGS="--M 64 --K 2048 --N 8192 --s 0.5 --kernels spr.v8.b32"
bench: $(BENCH_BIN)
	$(RUN_ENV) "$(BENCH_BIN)" $(BENCH_ARGS)

# --- Misc --------------------------------------------------------------------

# Offline plotting: no dependency on build, CPU ISA, or cluster libraries.
plots:
	@test -n "$(RESULTS)" || { echo 'Set RESULTS to a pilot or full campaign directory'; exit 2; }
	$(PYTHON) bench/campaign_plots.py "$(RESULTS)" $(if $(PLOT_DIR),--out "$(PLOT_DIR)")

data:
	$(PYTHON) scripts/generate_data.py --benchmark-suite --out "$(DATA_DIR)" $(if $(filter 1,$(OVERWRITE)),--overwrite) $(GENERATE_ARGS)

clean:
	rm -rf "$(BUILD_DIR)"

help:
	@printf '%s\n' \
		'make                 Build test_correctness, bench_cpu and peaks into build/<profile>/' \
		'make test            Correctness of every kernel vs an FP64 reference (TEST_ARGS="spr. tcsc." filters)' \
		'make bench           Run bench_cpu with BENCH_ARGS (see bench/bench.cpp)' \
		'make plots           Offline campaign plots (RESULTS=campaign directory, optional PLOT_DIR)' \
		'make data            Generate saved datasets (OVERWRITE=1 regenerates)' \
		'make clean           Remove the build directory of the selected profile' \
		'' \
		'PROFILE=release|sanitizer   separate build per profile' \
		'THREADS=n                   threads for make bench'
