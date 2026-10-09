#include <vector>
#include <utility>

using namespace std;

pair<vector<vector<int>>, vector<vector<int>>> preprocess(vector<vector<int>>& mat, int k);

vec_t rsr_inference(vector<int> v, const vector<vector<int>>& permutations, const vector<vector<int>>& segments, tern_t bin_k, int k);
