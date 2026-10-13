CXX ?= c++
PYTHON ?= python3
CONFIG ?= configs/temporary_config.json
DATA_DIR ?= data/sample_001
GENERATE_ARGS ?=
# Override with EIGEN_INCLUDE_DIR=/path/to/eigen (the directory containing Eigen/).
EIGEN_INCLUDE_DIR ?= $(firstword $(wildcard /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3 build/deps/eigen-3.4.0))
BLAS_CPPFLAGS ?= $(shell pkg-config --cflags-only-I openblas)
BLAS_LIBS ?= $(shell pkg-config --libs openblas)
CPPFLAGS += -Iinclude -isystem $(EIGEN_INCLUDE_DIR) $(BLAS_CPPFLAGS)
CPPFLAGS += $(shell pkg-config --cflags nlohmann_json)
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

BUILD_DIR := build
TARGET := $(BUILD_DIR)/rsr_baseline_test
EIGEN_TARGET := $(BUILD_DIR)/eigen_baseline_test

SOURCES := \
	src/naive.cpp \
	src/rsr.cpp \
	src/rsrpp.cpp \
	src/utils.cpp \
	src/eigen_baseline.cpp \
	src/blas_baseline.cpp \
	tests/data_reader.cpp \
	tests/test_extended.cpp

OBJECTS := $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(OBJECTS:.o=.d)
EIGEN_OBJECTS := $(BUILD_DIR)/src/eigen_baseline.o $(BUILD_DIR)/tests/test_eigen_baseline.o
DEPS += $(BUILD_DIR)/tests/test_eigen_baseline.d

.DEFAULT_GOAL := all

.PHONY: all build check-eigen generate run test clean clean-data clean-all help

all: build

build: $(TARGET) $(EIGEN_TARGET)

check-eigen:
	@test -f "$(EIGEN_INCLUDE_DIR)/Eigen/Core" || { \
		echo 'Eigen headers not found. On macOS: brew install eigen'; \
		echo 'Or set EIGEN_INCLUDE_DIR to the directory containing Eigen/.'; \
		exit 1; \
	}

$(TARGET): $(OBJECTS)
	mkdir -p "$(dir $@)"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(LDLIBS) $(BLAS_LIBS) -o "$@"

$(EIGEN_TARGET): $(EIGEN_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(EIGEN_OBJECTS) $(LDLIBS) -o "$@"

$(BUILD_DIR)/%.o: %.cpp | check-eigen
	mkdir -p "$(dir $@)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

generate:
	$(PYTHON) scripts/generate_data.py --config "$(CONFIG)" --out "$(DATA_DIR)" $(GENERATE_ARGS)

run test: $(TARGET) $(EIGEN_TARGET)
	"./$(TARGET)" "$(DATA_DIR)"
	"./$(EIGEN_TARGET)"

clean:
	find "$(BUILD_DIR)" -type f \( -name '*.o' -o -name '*.d' -o -name '$(notdir $(TARGET))' -o -name '$(notdir $(EIGEN_TARGET))' \) -delete

clean-data:
	find data -mindepth 1 -delete

clean-all: clean clean-data

help:
	@printf '%s\n' \
		'make             Build the executable' \
		'make build       Build the executable' \
		'                 EIGEN_INCLUDE_DIR=path selects Eigen headers' \
		'make generate    Generate sample data using config.json' \
 		'                 CONFIG=path.json selects another config' \
 		'                 DATA_DIR=path selects the output directory' \
 		'                 GENERATE_ARGS="--overwrite" replaces existing data' \
 		'                 GENERATE_ARGS="--debug-json" also writes expected.json' \
		'make run         Build and run the executable' \
		'make test        Build and run correctness tests' \
		'make clean       Remove build artifacts' \
		'make clean-data  Remove generated data' \
		'make clean-all   Clean build artifacts and data'

-include $(DEPS)
