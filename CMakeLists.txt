cmake_minimum_required(VERSION 3.16)

project(sparse_ternary_matrix_multiplication LANGUAGES CXX)

add_executable(rsr_baseline_test
    tests/test_extended.cpp
    src/naive.cpp
    src/rsr.cpp
    src/rsrpp.cpp
    src/utils.cpp
)

target_compile_features(rsr_baseline_test PRIVATE cxx_std_17)

target_include_directories(rsr_baseline_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME rsr_baseline_correctness COMMAND rsr_baseline_test)
endif()
