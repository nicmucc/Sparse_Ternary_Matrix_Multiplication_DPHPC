#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include "utils.h"
#include "naive.h"
#include "rsrpp.h"

using namespace std;

vec_t step_three(vec_t u, int k) {
    vec_t result(k);
    float sum;
    for (int i = k; i > 0; i--) {
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
                       const segment_t& segments, size_t N, int k) {
    // segmented sums
    const size_t block_size = size_t{1} << k;
    const size_t num_blocks = permutations.size() / N;
    vector<vec_t> us(num_blocks, vec_t(block_size, 0.0f));

    size_t start;
    size_t end;
    for (size_t i = 0; i < num_blocks; i++) {
        // Each block
        for (size_t j = 0; j < block_size; j++) {
            start = segments[i * block_size + j];
            if (j + 1 < block_size) {
                end = segments[i * block_size + j + 1];
            } else {
                end = N;
            }
            // Segmented sum
            for (size_t index = start; index < end; index++) {
                us[i][j] += v[permutations[i * N + index]];
            }
        }
    }

    vec_t result(num_blocks * k);

    // Block product to Bin_k
    vec_t partial_result;
    for (size_t i = 0; i < us.size(); i++) {
        partial_result = step_three(us[i], k);
        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }
    return result;
}
