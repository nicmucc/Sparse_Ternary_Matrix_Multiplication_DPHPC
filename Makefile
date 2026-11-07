CXX ?= c++
PYTHON ?= python3
PROFILE ?= release
BUILD_DIR ?= build/$(PROFILE)
CONFIG ?= configs/temporary_config.json
DATA_DIR ?= data/sample_001
TEST_DATA ?=
BENCHMARK_ARGS ?=
GENERATE_ARGS ?=
AVX2 ?= 1
OPENMP ?= 0
THREADS ?= 1
RESULTS_DIR ?= build/results
PLOT_DIR ?= out/plots
SWEEP_ARGS ?= --width 6 --samples 15 --min-ms 10
# Sparsity sweep at a fixed M x K x N
SWEEP_M ?= 16
SWEEP_K ?= 1024
SWEEP_N ?= 1024
SPARSITIES ?= 0.5 0.7 0.8 0.9 0.95 0.98 0.99
# Size sweep: K = N = each value, at fixed M and sparsity
SIZES ?= 256 512 1024 2048
SIZE_SPARSITY ?= 0.95

ifeq ($(PROFILE),scalar)
AVX2 := 0
else ifeq ($(PROFILE),sanitizer)
CXXFLAGS ?= -O1 -g -std=c++17 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer
else ifeq ($(PROFILE),openmp)
OPENMP := 1
else ifneq ($(PROFILE),release)
$(error PROFILE must be release, scalar, sanitizer, or openmp)
endif

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
EIGEN_CPPFLAGS ?= $(shell pkg-config --cflags eigen3)
BLAS_CPPFLAGS ?= $(shell pkg-config --cflags-only-I openblas)
BLAS_LIBS ?= $(shell pkg-config --libs openblas)
CPPFLAGS += -Iinclude $(EIGEN_CPPFLAGS) $(BLAS_CPPFLAGS) $(shell pkg-config --cflags nlohmann_json)
ifeq ($(AVX2),0)
CPPFLAGS += -DTERNARY_BASELINE_DISABLE_AVX2
endif
ifeq ($(OPENMP),1)
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

SOURCES := $(wildcard src/baselines/*.cpp src/utils/*.cpp)
CONFIG_HASH := $(shell { printf '%s\n' "$(CXX)" "$(CPPFLAGS)" "$(CXXFLAGS)" "$(LDFLAGS)" "$(LDLIBS)" "$(BLAS_LIBS)" "$(SOURCES)"; $(CXX) --version; } | cksum | awk '{print $$1}')
OBJECT_DIR := $(BUILD_DIR)/obj/$(CONFIG_HASH)
OBJECTS := $(SOURCES:%.cpp=$(OBJECT_DIR)/%.o)
TEST_OBJECTS := $(OBJECTS) $(OBJECT_DIR)/tests/test_baselines.o
BENCHMARK_OBJECTS := $(OBJECTS) $(OBJECT_DIR)/src/benchmark.o
TARGET := $(BUILD_DIR)/test_baselines
BENCHMARK_TARGET := $(BUILD_DIR)/benchmark
BUILD_CONFIG := $(BUILD_DIR)/config.txt
DEPS := $(sort $(TEST_OBJECTS:.o=.d) $(BENCHMARK_OBJECTS:.o=.d))

.DEFAULT_GOAL := build
.PHONY: build test run benchmark sweep plot data clean help FORCE

build: $(TARGET) $(BENCHMARK_TARGET)

$(BUILD_CONFIG): FORCE
	@mkdir -p "$(BUILD_DIR)"
	@{ printf '%s\n' "$(CXX)" "$(CPPFLAGS)" "$(CXXFLAGS)" "$(LDFLAGS)" "$(LDLIBS)" "$(BLAS_LIBS)" "$(SOURCES)"; $(CXX) --version; } > "$@.tmp"
	@cmp -s "$@" "$@.tmp" || cp "$@.tmp" "$@"
	@rm -f "$@.tmp"

$(OBJECT_DIR)/%.o: %.cpp | $(BUILD_CONFIG)
	@mkdir -p "$(dir $@)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

$(TARGET): $(TEST_OBJECTS) FORCE | $(BUILD_CONFIG)
	@if ! cmp -s "$(BUILD_CONFIG)" "$@.config" || test -n "$(filter-out FORCE,$?)"; then \
		$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TEST_OBJECTS) $(LDLIBS) $(BLAS_LIBS) -o "$@" && cp "$(BUILD_CONFIG)" "$@.config"; \
	fi

$(BENCHMARK_TARGET): $(BENCHMARK_OBJECTS) FORCE | $(BUILD_CONFIG)
	@if ! cmp -s "$(BUILD_CONFIG)" "$@.config" || test -n "$(filter-out FORCE,$?)"; then \
		$(CXX) $(CXXFLAGS) $(LDFLAGS) $(BENCHMARK_OBJECTS) $(LDLIBS) $(BLAS_LIBS) -o "$@" && cp "$(BUILD_CONFIG)" "$@.config"; \
	fi

test run: $(TARGET)
	"$(TARGET)" $(TEST_DATA)

benchmark: $(TARGET) $(BENCHMARK_TARGET)
	"$(TARGET)" $(TEST_DATA)
	"$(BENCHMARK_TARGET)" $(BENCHMARK_ARGS)

# Runs the benchmark once per sweep point and concatenates the CSVs (header kept once).
BENCH_ENV := OMP_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS)
define append_run
	echo "$(1)"; $(BENCH_ENV) "$(BENCHMARK_TARGET)" $(1) $(SWEEP_ARGS) > "$$out.tmp" || exit 1; \
	if [ -f "$$out" ]; then tail -n +2 "$$out.tmp" >> "$$out"; else mv "$$out.tmp" "$$out"; fi
endef

sweep: $(TARGET) $(BENCHMARK_TARGET)
	"$(TARGET)"
	@mkdir -p "$(RESULTS_DIR)"
	@out="$(RESULTS_DIR)/sparsity.csv"; rm -f "$$out"; \
	for s in $(SPARSITIES); do \
		$(call append_run,--m $(SWEEP_M) --k $(SWEEP_K) --n $(SWEEP_N) --sparsity $$s); \
	done; rm -f "$$out.tmp"
	@out="$(RESULTS_DIR)/size.csv"; rm -f "$$out"; \
	for s in $(SIZES); do \
		$(call append_run,--m $(SWEEP_M) --k $$s --n $$s --sparsity $(SIZE_SPARSITY)); \
	done; rm -f "$$out.tmp"
	@echo "Results in $(RESULTS_DIR)/sparsity.csv and $(RESULTS_DIR)/size.csv"

plot:
	$(PYTHON) scripts/plot_benchmarks.py "$(RESULTS_DIR)" "$(PLOT_DIR)"

data:
	$(PYTHON) scripts/generate_data.py --config "$(CONFIG)" --out "$(DATA_DIR)" $(GENERATE_ARGS)

clean:
	@test ! -d "$(BUILD_DIR)" || find "$(BUILD_DIR)" -type f \( -name '*.o' -o -name '*.d' -o -name '*.config' -o -name 'benchmark' -o -name 'test_baselines' -o -name 'config.txt' \) -delete

help:
	@printf '%s\n' 'make build       Build tests and benchmark' 'make test        Run self-contained correctness tests' 'make benchmark   Print benchmark CSV (BENCHMARK_ARGS="--m 1 --k 64 --n 64")' 'make sweep       Run sparsity and size sweeps into RESULTS_DIR' 'make plot        Plot sweep results into PLOT_DIR''make data        Generate optional file-based reference data' 'make clean       Remove artifacts for the selected profile' 'PROFILE=release|scalar|sanitizer|openmp selects an isolated build' 'TEST_DATA=path runs additional dataset checks'

-include $(DEPS)
