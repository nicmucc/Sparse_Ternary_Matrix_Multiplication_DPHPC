CXX ?= c++
PYTHON ?= python3
CONFIG ?= configs/temporary_config.json
DATA_DIR ?= data/sample_001
GENERATE_ARGS ?=
CPPFLAGS += -Iinclude -include types.h
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

BUILD_DIR := build
TARGET := $(BUILD_DIR)/rsr_baseline_test

SOURCES := \
	src/naive.cpp \
	src/rsr.cpp \
	src/rsrpp.cpp \
	src/utils.cpp \
	tests/data_reader.cpp \
	tests/test_extended.cpp

OBJECTS := $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(OBJECTS:.o=.d)

.DEFAULT_GOAL := all

.PHONY: all build generate run test clean clean-data clean-all help

all: build

build: $(TARGET)

$(TARGET): $(OBJECTS)
	mkdir -p "$(dir $@)"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o "$@"

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p "$(dir $@)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

generate:
	$(PYTHON) scripts/generate_data.py --config "$(CONFIG)" --out "$(DATA_DIR)" $(GENERATE_ARGS)

run test: $(TARGET)
	"./$(TARGET)" "$(DATA_DIR)"

clean:
	rm -rf "$(BUILD_DIR)"

clean-data:
	find data -mindepth 1 -delete

clean-all: clean clean-data

help:
	@printf '%s\n' \
		'make             Build the executable' \
		'make build       Build the executable' \
		'make generate    Generate sample data' \
		'make run         Build and run the executable' \
		'make test        Build and run correctness tests' \
		'make clean       Remove build artifacts' \
		'make clean-data  Remove generated data' \
		'make clean-all   Clean build artifacts and data'

-include $(DEPS)