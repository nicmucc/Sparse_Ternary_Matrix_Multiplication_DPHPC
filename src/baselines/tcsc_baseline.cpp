// Contains code adapted from fpgasystems/ternaryLLM, commit
// 68d384425261d60bfbde5ff079c33a474c1eb28e.
// Copyright (c) 2025 FPGA @ Systems Group, ETH Zurich.
// Imported portions: MIT; see ../licenses/ternaryllm.LICENSE.
// Source mapping and local changes are documented in ../README.md.

#include "tcsc_baseline.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include "ternary_baseline_common.h"

#if TERNARY_BASELINE_HAS_AVX2
#include <immintrin.h>
#endif

namespace tcsc_baseline {
namespace {

using namespace std;

// Imported preprocessing and scalar implementation.
class SparseFormat {
public:
vector<int> col_start_pos, col_start_neg;
vector<int32_t> row_index_pos, row_index_neg;
	SparseFormat(const int8_t* matrix, int K, int N) { // K ROWS, N COLUMNS
		int column_start_pos = 0;
		int column_start_neg = 0;
// #pragma omp parallel for
		for (int n = 0; n < N; n++) {
			this->col_start_pos.push_back(column_start_pos);
			this->col_start_neg.push_back(column_start_neg);
			for (int k = 0; k < K; k++) {
				if (matrix[k * N + n] >= 1) {
					column_start_pos++;
					this->row_index_pos.push_back(k);
				}
				else if (matrix[k * N + n] <= -1) {
					column_start_neg++;
					this->row_index_neg.push_back(k);
				}
			}
		}
		this->col_start_pos.push_back(column_start_pos);
		this->col_start_neg.push_back(column_start_neg);
	}
};
class MergedTCSC_Group {
public:
vector<int> metadata;
vector<int32_t> row_index;
	MergedTCSC_Group(SparseFormat naiveTCSC, [[maybe_unused]] int K, int N, int group_size, string group_method, bool interleaved) {
		int align_pointer = 0; // to increase with the total number of non-zero elements
// #pragma omp parallel for
		for (int n = 0; n < N; n += group_size) {
			vector<int> col_pos(group_size, 0); // number of pos values at each column
			vector<int> col_neg(group_size, 0);
			int align_min = 65536; // Count the align of min(min(pos,neg)...group_size)
			int align_mid = 0; // Count the align of max(min(pos,neg)...group_size)
			int align_max = 0; // Count the align of max(max(pos,neg)...group_size)
			int align_chosen = 0; // The final chosen align method: one of three above.
			int mid_longest_align = 0; // Find the position of align parts, so other cols follow it in mid grouping method
			[[maybe_unused]] int max_longest_nzero = 0; // Find the position of nzero parts, so other cols follow it in max grouping method
			int longest_align = 0;
			int longest_nzero = 0;
			for (int i = 0; i < group_size; i++) {
				col_pos[i] = naiveTCSC.col_start_pos[n + i + 1] - naiveTCSC.col_start_pos[n + i];
				col_neg[i] = naiveTCSC.col_start_neg[n + i + 1] - naiveTCSC.col_start_neg[n + i];
				align_min = min(min(col_pos[i], col_neg[i]), align_min);
				align_mid = max(min(col_pos[i], col_neg[i]), align_mid);
				align_max = max(max(col_pos[i], col_neg[i]), align_max);
				if (min(col_pos[i], col_neg[i]) > longest_align) {
					longest_align = min(col_pos[i], col_neg[i]);
					mid_longest_align = i;
				}
				if (max(col_pos[i], col_neg[i]) > longest_nzero) {
					longest_nzero = max(col_pos[i], col_neg[i]);
					max_longest_nzero = i;
				}
			}
			// Copy data
			if (interleaved) {
				for (int k = 0; k < align_min; k++) {
					for (int i = 0; i < group_size; i++) {
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
					}
				}
			}
			else {
				for (int k = 0; k < align_min; k++) {
					for (int i = 0; i < group_size; i++)
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					for (int i = 0; i < group_size; i++)
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
				}
			}

			if (group_method == "min") { // min grouping copies all pos and neg values at each column to the row_index, each row needs two for() loops on remainders
				align_chosen = align_min;
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
				for (int i = 0; i < group_size; i++) {
					/* for (int k = align_chosen; k < naiveTCSC.col_start_pos[n + i + 1]; k++) {
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]);
					}*/
					// k: align_chosen ~ col_end // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[startPos], &naiveTCSC.row_index_pos[endPos]);
					int startPos = naiveTCSC.col_start_pos[n + i] + align_chosen;
					int endPos = naiveTCSC.col_start_pos[n + i + 1];
					for (int k = startPos; k < endPos; k++)
						row_index.push_back(naiveTCSC.row_index_pos[k]);
					align_pointer += endPos - startPos;
					metadata.push_back(align_pointer); // remain_end_pos
					/* for (int k = align_chosen; k < naiveTCSC.col_start_neg[n + i + 1]; k++) { // k: align_chosen ~ col_end
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					}*/
					int startNeg = naiveTCSC.col_start_neg[n + i] + align_chosen;
					int endNeg = naiveTCSC.col_start_neg[n + i + 1];
					for (int k = startNeg; k < endNeg; k++)
						row_index.push_back(naiveTCSC.row_index_neg[k]);
					// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[startNeg], &naiveTCSC.row_index_neg[endNeg]);
					align_pointer += endNeg - startNeg;
					metadata.push_back(align_pointer); // remain_end_neg
				}
			}
			else if (group_method == "mid") { // mid is put all aligned parts together, and put zeros to other columns, so that each col only has one for() loop on remainders
				align_chosen = align_mid;
				for (int k = align_min; k < align_chosen; k++) {
					for (int i = 0; i < group_size; i++) {
						if (k < min(col_pos[i], col_neg[i])) {
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
						}
						else {
							int same_as_longest_align = naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + mid_longest_align] + k]; // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(same_as_longest_align); // two same values produces 0: a-a=0
							row_index.push_back(same_as_longest_align);
						}
					}
				}
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
				// Copy one remainder per column
				for (int i = 0; i < group_size; i++) {
					// k: align_chosen ~ col_end
					if (col_pos[i] > col_neg[i]) {
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + col_neg[i]], &naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i + 1] -1]);
						int startPos = naiveTCSC.col_start_pos[n + i] + col_neg[i];
						int endPos = naiveTCSC.col_start_pos[n + i + 1];
						for (int k = startPos; k < endPos; k++)
							row_index.push_back(naiveTCSC.row_index_pos[k]);
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[startPos], &naiveTCSC.row_index_pos[endPos]);
						align_pointer += endPos - startPos;
						metadata.push_back(align_pointer); // remain_end
						metadata.push_back(+1); // remain_val
					}
					else {
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + col_pos[i]], &naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i + 1] -1]);
						int startNeg = naiveTCSC.col_start_neg[n + i] + col_pos[i];
						int endNeg = naiveTCSC.col_start_neg[n + i + 1];
						for (int k = startNeg; k < endNeg; k++)
							row_index.push_back(naiveTCSC.row_index_neg[k]);
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[startNeg], &naiveTCSC.row_index_neg[endNeg]);
						align_pointer += endNeg - startNeg;
						metadata.push_back(align_pointer); // remain_end
						metadata.push_back(-1);  // remain_val
					}
				}
			}
			else if (group_method == "max") { // max refer to the semi-structured sparsity: each column has the same number of 1s and -1s, no for() loop on remainders
				align_chosen = align_mid;
				for (int k = align_min; k < align_chosen; k++) {
					for (int i = 0; i < group_size; i++) {
						if (k < min(col_pos[i], col_neg[i])) {
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
						}
						else {
							int same_as_longest_align = naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + mid_longest_align] + k]; // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(same_as_longest_align); // two same values produces 0: a-a=0
							row_index.push_back(same_as_longest_align);
						}
					}
				}
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
			}
			else { // other cases returns a fully aligned matrix: same as the unified TCSC
				// Pass group_size == N. No additional processing is needed.
				if(n==0)
					metadata.push_back(align_min * 2); // as the non-zero elements per column
			}
		}
	}
};
void GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_OpenMP(const float* X, const int32_t* w_neg_row_ind, const int32_t* w_neg_col_ptr, const int32_t* w_pos_row_ind, const int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < columns; j++) {
        for (int i = 0; i < rows; i++) {
            float res = 0;

            int neg_start = w_neg_col_ptr[j];
            int pos_start = w_pos_col_ptr[j];

            int neg_end = w_neg_col_ptr[j + 1];
            int pos_end = w_pos_col_ptr[j + 1];

            int neg = neg_end - neg_start;
            int pos = pos_end - pos_start;

            int min_neg_pos = std::min(neg, pos);


            for (int k = 0; k < min_neg_pos; k++) {
                res -= X[i * inners + w_neg_row_ind[neg_start + k]];
                res += X[i * inners + w_pos_row_ind[pos_start + k]];
            }
            for (int k = min_neg_pos; k < pos; k++) {
                res += X[i * inners + w_pos_row_ind[pos_start + k]];
            }
            for (int k = min_neg_pos; k < neg; k++) {
                res -= X[i * inners + w_neg_row_ind[neg_start + k]];
            }

            result[i * columns + j] = res;
        }
    }
}
#if TERNARY_BASELINE_HAS_AVX2
// Imported AVX2 implementations; selected at runtime by the public functions.
TERNARY_BASELINE_AVX2_TARGET
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, int M_ROW, int N_COL, int K) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < N_COL / 8; j++) {
        for (int i = 0; i < M_ROW; i++) {

            /* Pointer to where a column starts and ends
            int align_start  = metadata[j * 10 + 0];
            int align_end    = metadata[j * 10 + 1];
            int +remain_end0 = metadata[j * 10 + 2];
            int -remain_end0 = metadata[j * 10 + 3];
            int +remain_end1 = metadata[j * 10 + 4];
            int -remain_end1 = metadata[j * 10 + 5];
            ...
            */
            // Group # = j, metadata per group = 18
            const int* groupData  = &metadata[j * 18];
            __m256 res0 = _mm256_setzero_ps();
            const float* Xbase = X + i * K;
            float* Ybase = result + i * N_COL + j * 8;
            for (int k = groupData[0]; k < groupData[1]; k += 16) {
                __m256i indices_p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_index + k + 0));
                __m256i indices_n = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_index + k + 8));
                //__m128i indices_p = _mm128_load_si128(reinterpret_cast<const __m128i*>(row_index + k + 0));
                //__m128i indices_n = _mm128_load_si128(reinterpret_cast<const __m128i*>(row_index + k + 8));
                __m256 pos0 = _mm256_i32gather_ps(Xbase, indices_p, 4);
                __m256 neg0 = _mm256_i32gather_ps(Xbase, indices_n, 4);
                res0 = _mm256_add_ps(res0, _mm256_sub_ps(pos0, neg0));
            }
            _mm256_storeu_ps(Ybase, res0);

            for (int g = 0; g < 8; g++) {
                float pos00 = 0;
                for (int k = groupData[2 * g + 1]; k < groupData[2 * g + 2]; k++) {
                    pos00 += Xbase[row_index[k]];
                }
                Ybase[g] += pos00;
                float neg00 = 0;
                for (int k = groupData[2 * g + 2]; k < groupData[2 * g + 3]; k++) {
                    neg00 += Xbase[row_index[k]];
                }
                Ybase[g] -= neg00;
            }
        }
    }
}
#endif


#if TERNARY_BASELINE_HAS_AVX2
void multiply_tail(const vec_t& X, const Weights& W, vec_t& Y, int M, int first) {
    for (int col = first; col < W.N; ++col) {
        for (int row = 0; row < M; ++row) {
            const float* input = X.data() + static_cast<std::size_t>(row) * W.K;
            float sum = 0.0f;
            for (int p = W.col_start_pos[col]; p < W.col_start_pos[col + 1]; ++p) {
                sum += input[W.row_index_pos[p]];
            }
            for (int p = W.col_start_neg[col]; p < W.col_start_neg[col + 1]; ++p) {
                sum -= input[W.row_index_neg[p]];
            }
            Y[static_cast<std::size_t>(row) * W.N + col] = sum;
        }
    }
}

#endif

} // namespace

Weights preprocess(const tern_t& W, int K, int N) {
    ternary_baseline_detail::validate_weights(W, K, N);
    if (N / 8 > 0) ternary_baseline_detail::checked_size(N / 8, 18);
    const std::vector<int8_t> matrix(W.begin(), W.end());
    SparseFormat sparse(matrix.data(), K, N);
    // Keep actual output stride N in inference; process partial groups with the
    // basic kernel instead of letting upstream read beyond the column pointers.
    MergedTCSC_Group merged(sparse, K, N - N % 8, 8, "min", false);
    return {K, N, std::move(sparse.col_start_pos), std::move(sparse.col_start_neg),
            std::move(sparse.row_index_pos), std::move(sparse.row_index_neg),
            std::move(merged.metadata), std::move(merged.row_index)};
}

void multiply_basic(const vec_t& X, const Weights& W, vec_t& Y, int M) {
    ternary_baseline_detail::validate_product(X, Y, M, W.K, W.N);
    Y.resize(static_cast<std::size_t>(M) * W.N);
    GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_OpenMP(
        X.data(), W.row_index_neg.data(), W.col_start_neg.data(),
        W.row_index_pos.data(), W.col_start_pos.data(), Y.data(), M, W.N, W.K);
}

void multiply_merged(const vec_t& X, const Weights& W, vec_t& Y, int M) {
    ternary_baseline_detail::validate_product(X, Y, M, W.K, W.N);
#if TERNARY_BASELINE_HAS_AVX2
    if (avx2_available()) {
        Y.resize(static_cast<std::size_t>(M) * W.N);
        GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP(
            X.data(), W.metadata.data(), W.row_index.data(), Y.data(), M, W.N, W.K);
        multiply_tail(X, W, Y, M, W.N - W.N % 8);
        return;
    }
#endif
    multiply_basic(X, W, Y, M);
}

bool avx2_available() { return ternary_baseline_detail::avx2_available(); }

} // namespace tcsc_baseline
