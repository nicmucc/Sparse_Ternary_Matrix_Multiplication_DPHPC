cmake_minimum_required(VERSION 3.16)

project(sparse_ternary_matrix_multiplication LANGUAGES CXX)

set(RSR_BASELINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/scripts/baselines/rsr_extended")

add_executable(rsr_baseline_test
    "${RSR_BASELINE_DIR}/test_extended.cpp"
    "${RSR_BASELINE_DIR}/naive.cpp"
    "${RSR_BASELINE_DIR}/rsr.cpp"
    "${RSR_BASELINE_DIR}/rsrpp.cpp"
    "${RSR_BASELINE_DIR}/utils.cpp"
)

target_compile_features(rsr_baseline_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(rsr_baseline_test PRIVATE
        -iquote "${RSR_BASELINE_DIR}"
    )
else()
    target_include_directories(rsr_baseline_test PRIVATE
        "${RSR_BASELINE_DIR}"
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME rsr_baseline_correctness COMMAND rsr_baseline_test)
endif()
