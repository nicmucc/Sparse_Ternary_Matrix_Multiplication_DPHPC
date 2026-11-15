#include <vector>
#include <utility>
#include "naive.h"

using namespace std;

pair<vector<vector<int>>, vector<vector<int>>> binary_rsr_preprocess(vector<vector<int8_t>>& mat, int L);
pair<pair<vector<vector<int>>, vector<vector<int>>>, pair<vector<vector<int>>, vector<vector<int>>>> rsr_preprocess(vector<vector<int8_t>>& mat, int L);

pair<pair<pair<vector<vector<int>>, vector<vector<int>>>, pair<vector<vector<int>>, vector<vector<int>>>>, int> rsr_preprocess_float(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
pair<pair<pair<vector<vector<int>>, vector<vector<int>>>, pair<vector<vector<int>>, vector<vector<int>>>>, int> rsr_preprocess_int8(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);

template <typename T>
vector<T> binary_rsr_inference(vector<T> v, const vector<vector<int>>& permutations, const vector<vector<int>>& segments, vector<vector<int>> bin_L, int L) {
    int K = permutations[0].size();
    int N = permutations.size() * L;

    // segmented sums
    vector<vector<T>> us(permutations.size(), vector<T>(pow(2, L)));

    int start;
    int end;
    vector<int> segment;
    vector<int> permutation;
    for (int i = 0; i < permutations.size(); i++) {
        segment = segments[i];
        permutation = permutations[i];

        // Each block
        for (int j = 0; j < segment.size(); j++) {
            start = segment[j];
            if (j < segment.size() - 1) {
                end = segment[j + 1];
            } else {
                end = K;
            }
            // Segmented sum
            for (int index = start; index < end; index++) {
                us[i][j] += v[permutation[index]];
            }           
        }
    }

    vector<T> result(N);

    // Block product to Bin_L
    // TODO: change from here for RSR++
    vector<T> partial_result;
    for (int i = 0; i < us.size(); i++) {
        partial_result = vectorMatrixMultiply(us[i], bin_L);
        for (int j = 0; j < L; j++) {
            result[i * L + j] = partial_result[j];
        }
    }
    return result;
}

template <typename T>
vector<T> vector_rsr_inference(vector<T> v, const vector<vector<int>>& permutations_pos, const vector<vector<int>>& segments_pos, const vector<vector<int>>& permutations_neg, const vector<vector<int>>& segments_neg, vector<vector<int>> bin_L, int L) {
    int K = permutations_pos[0].size();
    int N = permutations_pos.size() * L;

    // segmented sums
    vector<vector<vector<T>>> us(2, vector<vector<T>>(permutations_pos.size(), vector<T>(pow(2, L))));

    int start_pos;
    int start_neg;
    int end_pos;
    int end_neg;
    vector<int> segment_pos;
    vector<int> segment_neg;
    vector<int> permutation_pos;
    vector<int> permutation_neg;
    for (int i = 0; i < permutations_pos.size(); i++) {
        segment_pos = segments_pos[i];
        segment_neg = segments_neg[i];
        permutation_pos = permutations_pos[i];
        permutation_neg = permutations_neg[i];

        // Each block
        for (int j = 0; j < segment_pos.size(); j++) {
            start_pos = segment_pos[j];
            start_neg = segment_neg[j];
            if (j < segment_pos.size() - 1) {
                end_pos = segment_pos[j + 1];
                end_neg = segment_neg[j + 1];
            } else {
                end_pos = K;
                end_neg = K;
            }
            // Segmented sum
            for (int index = start_pos; index < end_pos; index++) {
                us[0][i][j] += v[permutation_pos[index]];
            }    
            for (int index = start_neg; index < end_neg; index++) {
                us[1][i][j] += v[permutation_neg[index]];
            }          
        }
    }

    vector<T> result(N);

    // Block product to Bin_L
    // TODO: change from here for RSR++
    vector<vector<T>> partial_result(2);
    for (int i = 0; i < us[0].size(); i++) {
        partial_result[0] = vectorMatrixMultiply(us[0][i], bin_L);
        partial_result[1] = vectorMatrixMultiply(us[1][i], bin_L);
        for (int j = 0; j < L; j++) {
            result[i * L + j] = partial_result[0][j] - partial_result[1][j];
        }
    }
    return result;
}

template <typename T>
vector<vector<T>> rsr_inference(vector<vector<T>> v, const vector<vector<int>>& permutations_pos, const vector<vector<int>>& segments_pos, const vector<vector<int>>& permutations_neg, const vector<vector<int>>& segments_neg, vector<vector<int>> bin_L, int L) {
    int K = permutations_pos[0].size();
    int N = permutations_pos.size() * L;

    // segmented sums
    vector<vector<vector<T>>> us(2, vector<vector<T>>(permutations_pos.size(), vector<T>(pow(2, L))));

    int start_pos;
    int start_neg;
    int end_pos;
    int end_neg;
    vector<int> segment_pos;
    vector<int> segment_neg;
    vector<int> permutation_pos;
    vector<int> permutation_neg;
    vector<vector<T>> result(v.size(), vector<T>(N));
//#pragma omp parallel for 
    for (int row = 0; row < v.size(); row++) {
        for (int i = 0; i < permutations_pos.size(); i++) {
            segment_pos = segments_pos[i];
            segment_neg = segments_neg[i];
            permutation_pos = permutations_pos[i];
            permutation_neg = permutations_neg[i];

            // Each block
            for (int j = 0; j < segment_pos.size(); j++) {
                start_pos = segment_pos[j];
                start_neg = segment_neg[j];
                if (j < segment_pos.size() - 1) {
                    end_pos = segment_pos[j + 1];
                    end_neg = segment_neg[j + 1];
                } else {
                    end_pos = K;
                    end_neg = K;
                }
                // Segmented sum
                us[0][i][j] = 0;
                us[1][i][j] = 0;
                for (int index = start_pos; index < end_pos; index++) {
                    us[0][i][j] += v[row][permutation_pos[index]];
                }    
                for (int index = start_neg; index < end_neg; index++) {
                    us[1][i][j] += v[row][permutation_neg[index]];
                }          
            }
        }

        // Block product to Bin_L
        // TODO: change from here for RSR++
        vector<vector<T>> partial_result(2);
        for (int i = 0; i < us[0].size(); i++) {
            partial_result[0] = vectorMatrixMultiply(us[0][i], bin_L);
            partial_result[1] = vectorMatrixMultiply(us[1][i], bin_L);
            for (int j = 0; j < L; j++) {
                result[row][i * L + j] = partial_result[0][j] - partial_result[1][j];
            }
        }
    }
    return result;
}