# Run `make help` for the list of targets.

# --- Settings (override on the command line) --------------------------------

# release | native | scalar | sanitizer | openmp
PROFILE   ?= release
BUILD_DIR ?= build/$(PROFILE)
PYTHON    ?= python3
THREADS   ?= 1

TEST_DATA      ?=
BENCHMARK_ARGS ?=

RESULTS_DIR ?= build/results
PLOT_DIR    ?= out/plots
SWEEP_ARGS  ?= --width 6 --samples 15 --min-ms 10

# Sparsity sweep at SWEEP_M x SWEEP_K x SWEEP_N
SWEEP_M    ?= 16
SWEEP_K    ?= 1024
SWEEP_N    ?= 1024
SPARSITIES ?= 0.5 0.7 0.8 0.9 0.95 0.98 0.99

# Size sweep with K = N = each value
SIZES         ?= 256 512 1024 2048
SIZE_SPARSITY ?= 0.95

CONFIG        ?= configs/temporary_config.json
DATA_DIR      ?= data/sample_001
GENERATE_ARGS ?=

# --- Flags -------------------------------------------------------------------

OPT_FLAGS := -O3
AVX2      := 1
OPENMP    := 0

ifeq ($(PROFILE),release)
else ifeq ($(PROFILE),native)
  OPT_FLAGS += $(if $(filter arm64 aarch64,$(shell uname -m)),-mcpu=native,-march=native)
else ifeq ($(PROFILE),scalar)
  AVX2 := 0
else ifeq ($(PROFILE),sanitizer)
  OPT_FLAGS := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
else ifeq ($(PROFILE),openmp)
  OPENMP := 1
else
  $(error PROFILE must be release, native, scalar, sanitizer, or openmp)
endif

ifeq ($(OPENMP),1)
  OPT_FLAGS += -fopenmp
endif

# `:=` so pkg-config runs once, not on every use
DEP_CPPFLAGS ?= $(shell pkg-config --cflags-only-I eigen3 nlohmann_json openblas)
BLAS_LIBS    ?= $(shell pkg-config --libs openblas)
DEP_CPPFLAGS := $(DEP_CPPFLAGS)
BLAS_LIBS    := $(BLAS_LIBS)

CXXFLAGS ?= $(OPT_FLAGS) -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude $(DEP_CPPFLAGS)
LDLIBS   += $(BLAS_LIBS)

ifeq ($(AVX2),0)
  CPPFLAGS += -DTERNARY_BASELINE_DISABLE_AVX2
endif

# --- Files -------------------------------------------------------------------

OBJ_DIR    := $(BUILD_DIR)/obj
SOURCES    := $(wildcard src/baselines/*.cpp src/utils/*.cpp)
OBJECTS    := $(SOURCES:%.cpp=$(OBJ_DIR)/%.o)
TEST_MAIN  := $(OBJ_DIR)/src/test_baselines.o
BENCH_MAIN := $(OBJ_DIR)/src/benchmark.o

TEST_BIN     := $(BUILD_DIR)/test_baselines
BENCH_BIN    := $(BUILD_DIR)/benchmark
BUILD_CONFIG := $(BUILD_DIR)/config.txt
TESTS_PASSED := $(BUILD_DIR)/tests.passed

SPARSITY_CSV  := $(RESULTS_DIR)/sparsity.csv
SIZE_CSV      := $(RESULTS_DIR)/size.csv
SPARSITY_ARGS := $(RESULTS_DIR)/sparsity.args
SIZE_ARGS     := $(RESULTS_DIR)/size.args

RUN_ENV := OMP_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS)

# If compiler, flags, or source list changed since the last build, drop the old
# build. Checked here rather than via timestamps because make 3.81 (macOS) only
# compares whole seconds and can miss a change right after a build.
BUILD_STATE := $(CXX) | $(CPPFLAGS) | $(CXXFLAGS) | $(LDFLAGS) | $(LDLIBS) | $(SOURCES) | $(shell $(CXX) --version | head -1)
ifneq ($(BUILD_STATE),$(shell cat "$(BUILD_CONFIG)" 2>/dev/null))
  $(shell rm -rf "$(OBJ_DIR)" "$(TEST_BIN)" "$(BENCH_BIN)" "$(TESTS_PASSED)"; mkdir -p "$(BUILD_DIR)"; printf '%s' '$(BUILD_STATE)' > "$(BUILD_CONFIG)")
endif

# --- Helpers -----------------------------------------------------------------

# $(call write_if_changed,file,lines): only touch `file` when its content changes
define write_if_changed
@mkdir -p "$(dir $(1))"
@printf '%s\n' $(2) > "$(1).tmp"
@if cmp -s "$(1)" "$(1).tmp"; then rm "$(1).tmp"; else mv "$(1).tmp" "$(1)"; fi
endef

# $(call run_sweep,csv,values,args with $$x): one run per value, one CSV header
define run_sweep
@rm -f "$(1).tmp"
@for x in $(2); do \
	echo "benchmark $(3)"; \
	$(RUN_ENV) "$(BENCH_BIN)" $(3) $(SWEEP_ARGS) > "$(1).run" || exit 1; \
	if [ -f "$(1).tmp" ]; then tail -n +2 "$(1).run" >> "$(1).tmp"; \
	else mv "$(1).run" "$(1).tmp"; fi; \
done; \
rm -f "$(1).run"; mv "$(1).tmp" "$(1)"
endef

SPARSITY_LINES = "$(PROFILE)" "$(RUN_ENV)" "$(SWEEP_ARGS)" "$(SWEEP_M) $(SWEEP_K) $(SWEEP_N)" "$(SPARSITIES)"
SIZE_LINES     = "$(PROFILE)" "$(RUN_ENV)" "$(SWEEP_ARGS)" "$(SWEEP_M)" "$(SIZES)" "$(SIZE_SPARSITY)"

# --- Build -------------------------------------------------------------------

.DEFAULT_GOAL := build
.PHONY: build test benchmark sweep sweep-sparsity sweep-size plot data clean help FORCE

build: $(TEST_BIN) $(BENCH_BIN)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(TEST_BIN): $(OBJECTS) $(TEST_MAIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o "$@"

$(BENCH_BIN): $(OBJECTS) $(BENCH_MAIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o "$@"

-include $(OBJECTS:.o=.d) $(TEST_MAIN:.o=.d) $(BENCH_MAIN:.o=.d)

# --- Test and benchmark ------------------------------------------------------

test: $(TEST_BIN)
	$(RUN_ENV) "$(TEST_BIN)" $(TEST_DATA)

$(TESTS_PASSED): $(TEST_BIN)
	$(RUN_ENV) "$(TEST_BIN)"
	@touch "$@"

benchmark: $(TESTS_PASSED) $(BENCH_BIN)
	$(RUN_ENV) "$(BENCH_BIN)" $(BENCHMARK_ARGS)

# --- Sweeps and plots --------------------------------------------------------

# A CSV is re-measured when missing, when the benchmark was rebuilt, or when
# its sweep settings changed. `make sweep` always re-measures.

$(SPARSITY_ARGS): FORCE
	$(call write_if_changed,$@,$(SPARSITY_LINES))

$(SIZE_ARGS): FORCE
	$(call write_if_changed,$@,$(SIZE_LINES))

$(SPARSITY_CSV): $(BENCH_BIN) $(TESTS_PASSED) $(SPARSITY_ARGS)
	$(call run_sweep,$@,$(SPARSITIES),--m $(SWEEP_M) --k $(SWEEP_K) --n $(SWEEP_N) --sparsity $$x)

$(SIZE_CSV): $(BENCH_BIN) $(TESTS_PASSED) $(SIZE_ARGS)
	$(call run_sweep,$@,$(SIZES),--m $(SWEEP_M) --k $$x --n $$x --sparsity $(SIZE_SPARSITY))

sweep:          SWEEP_CSVS = $(SPARSITY_CSV) $(SIZE_CSV)
sweep-sparsity: SWEEP_CSVS = $(SPARSITY_CSV)
sweep-size:     SWEEP_CSVS = $(SIZE_CSV)

# Build in parallel, then measure with -j1 so sweeps never run concurrently.
sweep sweep-sparsity sweep-size: $(BENCH_BIN) $(TESTS_PASSED)
	@rm -f $(SWEEP_CSVS)
	@$(MAKE) --no-print-directory -j1 $(SWEEP_CSVS)

plot: $(BENCH_BIN) $(TESTS_PASSED)
	@$(MAKE) --no-print-directory -j1 $(SPARSITY_CSV) $(SIZE_CSV)
	$(PYTHON) scripts/plot_benchmarks.py "$(RESULTS_DIR)" "$(PLOT_DIR)"

# --- Misc --------------------------------------------------------------------

data:
	$(PYTHON) scripts/generate_data.py --config "$(CONFIG)" --out "$(DATA_DIR)" $(GENERATE_ARGS)

clean:
	rm -rf "$(BUILD_DIR)"

help:
	@printf '%s\n' \
		'make                 Build tests and benchmark' \
		'make test            Run correctness tests (TEST_DATA=dir adds dataset checks)' \
		'make benchmark       One run, CSV to stdout (BENCHMARK_ARGS="--m 1 --k 64 --n 64")' \
		'make sweep           Re-measure both sweeps into RESULTS_DIR' \
		'make sweep-sparsity  Re-measure only the sparsity sweep' \
		'make sweep-size      Re-measure only the size sweep' \
		'make plot            Plot into PLOT_DIR (runs sweeps if missing or stale)' \
		'make data            Generate optional file-based reference data' \
		'make clean           Remove the build directory of the selected profile' \
		'' \
		'PROFILE=release|native|scalar|sanitizer|openmp   separate build per profile' \
		'THREADS=n                                        threads for test/benchmark runs'
