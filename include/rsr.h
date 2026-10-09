#include <vector>
#include <utility>

using namespace std;

pair<vector<vector<int>>, vector<vector<int>>> preprocess(vector<vector<int>>& mat, int k);

vec_t rsr_inference(const vec_t& v, const permutation_t& permutations, const segment_t& segments, const binary_matrix_t& bin_k, size_t N, int k);
