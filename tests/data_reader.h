#ifndef DATA_READER_H
#define DATA_READER_H

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include "types.h"

struct SampleData {
    std::size_t M, K, N;

    vec_t X;
    tern_t W;

    vec_t bias;
    vec_t alpha;

    vec_t expected_linear;
    vec_t expected_prelu;

    double atol;
    double rtol;
};

SampleData read_sample(const std::string& directory);

#endif