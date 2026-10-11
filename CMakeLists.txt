cmake_minimum_required(VERSION 3.16)

project(sparse_ternary_matrix_multiplication LANGUAGES CXX)

find_package(BLAS REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(rsr_baseline_test
    tests/test_extended.cpp
    tests/data_reader.cpp
    src/naive.cpp
    src/rsr.cpp
    src/rsrpp.cpp
    src/utils.cpp
    src/blas_baseline.cpp
)

target_compile_features(rsr_baseline_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rsr_baseline_test PRIVATE
        -Wall
        -Wextra
        -include
        types.h
    )
endif()
target_include_directories(rsr_baseline_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(rsr_baseline_test PRIVATE
    nlohmann_json::nlohmann_json
    ${BLAS_LIBRARIES}
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME rsr_baseline_correctness
        COMMAND rsr_baseline_test "${CMAKE_CURRENT_SOURCE_DIR}/data/sample_001"
    )
endif()
