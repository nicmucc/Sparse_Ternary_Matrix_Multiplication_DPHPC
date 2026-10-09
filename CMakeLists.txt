cmake_minimum_required(VERSION 3.16)

project(sparse_ternary_matrix_multiplication LANGUAGES CXX)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(rsr_baseline_test
    tests/test_extended.cpp
    src/naive.cpp
    src/rsr.cpp
    src/rsrpp.cpp
    src/utils.cpp
    src/eigen_baseline.cpp
)

target_compile_features(rsr_baseline_test PRIVATE cxx_std_17)

target_include_directories(rsr_baseline_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(rsr_baseline_test PRIVATE Eigen3::Eigen)

add_executable(eigen_baseline_test tests/test_eigen_baseline.cpp src/eigen_baseline.cpp)
target_compile_features(eigen_baseline_test PRIVATE cxx_std_17)
target_include_directories(eigen_baseline_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(eigen_baseline_test PRIVATE Eigen3::Eigen)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME rsr_baseline_correctness COMMAND rsr_baseline_test)
    add_test(NAME eigen_baseline_correctness COMMAND eigen_baseline_test)
endif()
