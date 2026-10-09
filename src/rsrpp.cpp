#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include "utils.h"
#include "naive.h"
#include "rsrpp.h"

using namespace std;

vec_t step_three(vec_t u, int block_width) {
    vec_t result(block_width);
    float sum;
    for (int i = block_width; i > 0; i--) {
        sum = 0;
        for (int j = 1; j < pow(2, i); j += 2) {
            sum += u[j];
        }
        result[i - 1] = sum;
        for (int j = 0; j < pow(2, i - 1); j++) {
            u[j] = u[j * 2] + u[j * 2 + 1];
        }
    }

    return result;
}

vec_t rsr_pp_inference(const vec_t& v, const permutation_t& permutations,
                       const segment_t& segments, size_t K, int block_width) {
    // segmented sums
    const size_t pattern_count = size_t{1} << block_width;
    const size_t num_blocks = permutations.size() / K;
    vector<vec_t> us(num_blocks, vec_t(pattern_count, 0.0f));

    size_t start;
    size_t end;
    for (size_t i = 0; i < num_blocks; i++) {
        // Each block
        for (size_t j = 0; j < pattern_count; j++) {
            start = segments[i * pattern_count + j];
            if (j + 1 < pattern_count) {
                end = segments[i * pattern_count + j + 1];
            } else {
                end = K;
            }
            // Segmented sum
            for (size_t index = start; index < end; index++) {
                us[i][j] += v[permutations[i * K + index]];
            }
        }
    }

    vec_t result(num_blocks * block_width);

    // Block product to binary_patterns
    vec_t partial_result;
    for (size_t i = 0; i < us.size(); i++) {
        partial_result = step_three(us[i], block_width);
        for (int j = 0; j < block_width; j++) {
            result[i * block_width + j] = partial_result[j];
        }
    }
    return result;
}
