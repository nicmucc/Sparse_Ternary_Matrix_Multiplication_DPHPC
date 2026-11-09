# Run `make help` for the list of targets.

# --- Settings (override on the command line) --------------------------------

# release | native | scalar | sanitizer | openmp
PROFILE   ?= release
BUILD_DIR ?= build/$(PROFILE)
PYTHON    ?= python3
THREADS   ?= 1

TEST_DATA      ?=
BENCHMARK_ARGS ?=
PYTHON_BENCHMARK_ARGS ?=

RESULTS_DIR ?= build/results
PLOT_DIR    ?= out/plots
SWEEP_ARGS  ?= --width 6 --samples 15 --min-ms 10

DATA_DIR      ?= data/benchmarks
GENERATE_ARGS ?=
OVERWRITE     ?= 0

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

SWEEP_CSV      := $(RESULTS_DIR)/benchmarks.csv
SWEEP_SETTINGS := $(RESULTS_DIR)/sweep.args
DATA_MANIFEST  := $(DATA_DIR)/manifest.json
DATA_FILES     := $(wildcard $(DATA_DIR)/*/*.bin $(DATA_DIR)/*/metadata.json)

RUN_ENV := OMP_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS)

# Force affected targets when configuration changes, even within one timestamp tick.
BUILD_STATE := $(CXX) | $(CPPFLAGS) | $(CXXFLAGS) | $(LDFLAGS) | $(LDLIBS) | $(SOURCES) | $(shell $(CXX) --version | head -1)
BUILD_CHANGED :=
ifneq ($(BUILD_STATE),$(shell cat "$(BUILD_CONFIG)" 2>/dev/null))
  BUILD_CHANGED := FORCE
endif

# --- Helpers -----------------------------------------------------------------

# $(call write_if_changed,file,lines): only touch `file` when its content changes
define write_if_changed
@mkdir -p "$(dir $(1))"
@printf '%s\n' $(2) > "$(1).tmp"
@if cmp -s "$(1)" "$(1).tmp"; then rm "$(1).tmp"; else mv "$(1).tmp" "$(1)"; fi
endef

SWEEP_LINES = "$(PROFILE)" "$(RUN_ENV)" "$(SWEEP_ARGS)" "$(abspath $(DATA_DIR))"
SWEEP_COMMAND = $(RUN_ENV) $(PYTHON) scripts/run_benchmarks.py "$(BENCH_BIN)" "$(DATA_DIR)" "$(SWEEP_CSV)" $(SWEEP_ARGS)

# --- Build -------------------------------------------------------------------

.DEFAULT_GOAL := build
.PHONY: build test benchmark python-extension python-test python-benchmark sweep plot data clean help FORCE

build: $(TEST_BIN) $(BENCH_BIN)

$(BUILD_CONFIG): $(BUILD_CHANGED)
	@rm -rf "$(OBJ_DIR)" "$(TEST_BIN)" "$(BENCH_BIN)" "$(TESTS_PASSED)"
	@mkdir -p "$(@D)"
	@printf '%s' '$(BUILD_STATE)' > "$@"

$(OBJ_DIR)/%.o: %.cpp $(BUILD_CONFIG) $(BUILD_CHANGED)
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(TEST_BIN): $(OBJECTS) $(TEST_MAIN) $(BUILD_CHANGED)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o,$^) $(LDLIBS) -o "$@"

$(BENCH_BIN): $(OBJECTS) $(BENCH_MAIN) $(BUILD_CHANGED)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o,$^) $(LDLIBS) -o "$@"

-include $(OBJECTS:.o=.d) $(TEST_MAIN:.o=.d) $(BENCH_MAIN:.o=.d)

# --- Test and benchmark ------------------------------------------------------

test: $(TEST_BIN)
	$(RUN_ENV) "$(TEST_BIN)" $(TEST_DATA)

$(TESTS_PASSED): $(TEST_BIN) $(BUILD_CHANGED)
	$(RUN_ENV) "$(TEST_BIN)"
	@touch "$@"

benchmark: $(TESTS_PASSED) $(BENCH_BIN)
	$(RUN_ENV) "$(BENCH_BIN)" $(BENCHMARK_ARGS)

python-extension:
	$(PYTHON) setup.py build_ext --inplace

python-test:
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests/python

python-benchmark:
	$(RUN_ENV) $(PYTHON) scripts/benchmark_python.py --threads $(THREADS) $(PYTHON_BENCHMARK_ARGS)

# --- Sweeps and plots --------------------------------------------------------

# Re-measure when inputs, executable, runner, or settings change.
# Explicit sweeps force the shared CSV target, so sweep and plot cannot race.
ifneq ($(filter sweep,$(MAKECMDGOALS)),)
$(SWEEP_CSV): FORCE
endif

$(SWEEP_SETTINGS): FORCE
	$(call write_if_changed,$@,$(SWEEP_LINES))

$(SWEEP_CSV): $(BENCH_BIN) $(TESTS_PASSED) $(SWEEP_SETTINGS) $(DATA_MANIFEST) $(DATA_FILES) scripts/run_benchmarks.py $(BUILD_CHANGED)
	$(SWEEP_COMMAND)

sweep: $(SWEEP_CSV)

plot: $(SWEEP_CSV)
	$(PYTHON) scripts/plot_benchmarks.py "$(RESULTS_DIR)" "$(PLOT_DIR)"

# --- Misc --------------------------------------------------------------------

data:
	$(PYTHON) scripts/generate_data.py --benchmark-suite --out "$(DATA_DIR)" $(if $(filter 1,$(OVERWRITE)),--overwrite) $(GENERATE_ARGS)

clean:
	rm -rf "$(BUILD_DIR)"

help:
	@printf '%s\n' \
		'make                 Build tests and benchmark' \
		'make test            Run correctness tests (TEST_DATA=dir adds dataset checks)' \
		'make benchmark       One saved dataset, CSV to stdout (BENCHMARK_ARGS="--dataset dir")' \
		'make python-extension  Build the PyTorch bindings for all C++ baselines' \
		'make python-test     Run Python benchmark-interface tests' \
		'make python-benchmark  One saved dataset through PyTorch (PYTHON_BENCHMARK_ARGS="--dataset dir")' \
		'make sweep           Benchmark every saved dataset in DATA_DIR/manifest.json' \
		'make plot            Plot into PLOT_DIR (runs sweeps if missing or stale)' \
		'make data            Generate benchmark datasets (OVERWRITE=1 regenerates)' \
		'make clean           Remove the build directory of the selected profile' \
		'' \
		'PROFILE=release|native|scalar|sanitizer|openmp   separate build per profile' \
		'THREADS=n                                        threads for test/benchmark runs'
