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

CONFIG_HASH := $(shell { printf '%s\n' "$(CXX)" "$(CPPFLAGS)" "$(CXXFLAGS)" "$(LDFLAGS)" "$(LDLIBS)" "$(BLAS_LIBS)"; $(CXX) --version; } | cksum | awk '{print $$1}')
OBJECT_DIR := $(BUILD_DIR)/obj/$(CONFIG_HASH)
SOURCES := $(wildcard src/baselines/*.cpp src/utils/*.cpp)
OBJECTS := $(SOURCES:%.cpp=$(OBJECT_DIR)/%.o)
TEST_OBJECTS := $(OBJECTS) $(OBJECT_DIR)/tests/test_baselines.o
BENCHMARK_OBJECTS := $(OBJECTS) $(OBJECT_DIR)/src/benchmark.o
TARGET := $(BUILD_DIR)/test_baselines
BENCHMARK_TARGET := $(BUILD_DIR)/benchmark
BUILD_CONFIG := $(BUILD_DIR)/config.txt
DEPS := $(sort $(TEST_OBJECTS:.o=.d) $(BENCHMARK_OBJECTS:.o=.d))

.DEFAULT_GOAL := build
.PHONY: build test run benchmark data clean help FORCE

build: $(TARGET) $(BENCHMARK_TARGET)

$(BUILD_CONFIG): FORCE
	@mkdir -p "$(BUILD_DIR)"
	@{ printf '%s\n' "$(CXX)" "$(CPPFLAGS)" "$(CXXFLAGS)" "$(LDFLAGS)" "$(LDLIBS)" "$(BLAS_LIBS)"; $(CXX) --version; } > "$@.tmp"
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

benchmark: $(BENCHMARK_TARGET)
	"$(BENCHMARK_TARGET)" $(BENCHMARK_ARGS)

data:
	$(PYTHON) scripts/generate_data.py --config "$(CONFIG)" --out "$(DATA_DIR)" $(GENERATE_ARGS)

clean:
	@test ! -d "$(BUILD_DIR)" || find "$(BUILD_DIR)" -type f \( -name '*.o' -o -name '*.d' -o -name '*.config' -o -name 'benchmark' -o -name 'test_baselines' -o -name 'config.txt' \) -delete

help:
	@printf '%s\n' 'make build       Build tests and benchmark' 'make test        Run self-contained correctness tests' 'make benchmark   Print benchmark CSV (BENCHMARK_ARGS="--m 1 --k 64 --n 64")' 'make data        Generate optional file-based reference data' 'make clean       Remove artifacts for the selected profile' 'PROFILE=release|scalar|sanitizer|openmp selects an isolated build' 'TEST_DATA=path runs additional dataset checks'

-include $(DEPS)
