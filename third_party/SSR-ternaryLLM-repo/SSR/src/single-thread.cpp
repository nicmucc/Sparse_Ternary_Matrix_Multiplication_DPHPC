#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstdio>

#include "TCSC.hpp"
#include "initData.hpp"
#include "GEMM_CPU_INT8.hpp"
#include "GEMM_CPU_FP32.hpp"
#include "LlamaModel.hpp"

// OMP_NUM_THREADS(1)
#define OMP_NUM_THREADS = 1
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <Eigen/Sparse>


#include "time_compare.hpp"
using namespace std;

std::vector<int64_t>  record_time(vector<string> names, vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints, std::vector<int64_t> records, FILE* fptr) {
    for (int j = 0; j < names.size(); j++) {
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timePoints[j + 1] - timePoints[j]).count();
        records[j] += ns;
        fprintf(fptr, "%lld \t", ns);
    }
    fprintf(fptr, "\n");
    return records;
}

void record_config(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, FILE* fptr) {
    fprintf(fptr, "Benchmarking M=%d K=%d N=%d Sparsity=%f Variation=%f \n", M_ROW, K_LEN, N_COL, Sparsity, Variation);
    for (int n = 0; n < names.size(); n++) {
        fprintf(fptr, names[n].c_str());
    }
    fprintf(fptr, "\n");
}

std::vector<float> print_ms_speedup(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, std::vector<float> speedups, std::vector<int64_t> records, FILE* fptr) {
    std::cout << "M=" << M_ROW << ", K=" << K_LEN << ", N=" << N_COL << ", Sparsity=" << std::fixed << std::setprecision(2) << Sparsity << " +/- " << Variation << std::endl;
    int64_t baseline = records[0];
    for (int n = 0; n < names.size(); n++) {
        float speedup = (double)baseline / (double)records[n];
        speedups.push_back(speedup);
        std::cout << names[n] << records[n] / 1000000 << "\t ms, speedup = " << std::fixed << std::setprecision(2) << speedup << std::endl;
        fprintf(fptr, "%lld\t", records[n] / 1000000);
    }
    std::cout << "\n" << std::endl;
    fprintf(fptr, "ms\n");
    for (int n = 0; n < names.size(); n++) {
        float speedup = (double)baseline / (double)records[n];
        fprintf(fptr, "%.2f\t", speedup);
    }
    fprintf(fptr, "speedup\n");
    return speedups;
}

void print_speedup_summary(vector<string> names, std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV, std::vector<float> speedups) {
    std::cout << "speedup" << std::endl;
    for (int n = 0; n < names.size(); n++) {
        std::cout << names[n];
    }
    std::cout << std::endl;
    for (int i = 0; i < Config_MKNSV.size(); i++) {
        std::cout << "M=" << std::get<0>(Config_MKNSV[i]) << " K=" << std::get<1>(Config_MKNSV[i]) << " N=" << std::get<2>(Config_MKNSV[i]) << " S=" << std::fixed << std::setprecision(2) << std::get<3>(Config_MKNSV[i]) << " +/- " << std::get<4>(Config_MKNSV[i]) << "\t";
        for (int n = 0; n < names.size(); n++)
            std::cout << speedups[i * names.size() + n] << "\t";
        std::cout << std::endl;
    }
}

std::string time_string() {
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* p_tm = std::localtime(&now_c);
    if (p_tm == nullptr) {
        return "2025-01-01_00-00-00";
    }
    std::ostringstream oss;
    oss << std::put_time(p_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

int benchmark_FP32_Matrix_RSR_SSR(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_Matrix_RSR_SSR_1Thread_" + time_string() + ".txt";
    vector<string> names = { "EigenDense\t", "EigenSparse\t", "RSR_______\t", "RSRPP_____\t", "SSR_______\t", "SSR_VEC___\t", "SSR_VEC_TREE\t", };
    const int SSR_K = 6;
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        //  M    K      N    Sparsity, Data Variation
        {512, 1024,  4096, 0.45,0.05},
        {512, 1024,  4096, 0.50,0.05},
        {512, 1024,  4096, 0.55,0.05},
        {512, 1024,  4096, 0.60,0.05},
        {512, 1024,  4096, 0.65,0.05},
        {512, 1024,  4096, 0.70,0.05},
        {512, 1024,  4096, 0.75,0.05},
        {512, 1024,  4096, 0.80,0.05},
        {512, 1024,  4096, 0.85,0.05},
        {512, 1024,  4096, 0.90,0.05},
        {512, 1024,  4096, 0.95,0.05},
    };
    // X: MxK, W: KxN, Y=MxN, SSR_K is the k factor used in RSR and SSR methods.

    std::vector<float> speedups;
    FILE* fptr = fopen(file_name.c_str(), "w");
    std::cout << "Running " << file_name << std::endl;
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        // *********** Data Initialization *************
        std::vector<int8_t> X_INT8 = initX<int8_t>(M_ROW * K_LEN, 32);//Activation 
        std::vector<float>  X_FP32 = initX<float>(M_ROW * K_LEN, 512);//Activation   
        std::vector<int8_t> W_INT8 = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed 
        std::vector<float>  W_FP32(W_INT8.begin(), W_INT8.end());
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0)       
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<float> Y_FP32_SSR; // For SSR
        std::vector<int8_t> Y_INT8(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<int8_t> Y_INT8_SSR; // For SSR
        SparseFormat naiveTCSC = SparseFormat(W_INT8.data(), K_LEN, N_COL);
        SparseFormat naiveTCSC_Uniform = SparseFormat(WeightUniform.data(), K_LEN, N_COL);
        MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true);

        float* X_FP32_Pointer = X_FP32.data();// Convert vectors into *
        int8_t* W_INT8_Pointer = W_INT8.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32.data(), M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(W_FP32.data(), K_LEN, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();
        Eigen::Map < Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen_INT8(X_INT8.data(), M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen_INT8(W_INT8.data(), K_LEN, N_COL);
        Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen_INT8(M_ROW, N_COL);
        Eigen::SparseMatrix<int8_t> W_eigen_sparse_INT8 = W_eigen_INT8.sparseView();
        //Eigen::setNbThreads(1);
        std::cout << "Eigen Threads: " << Eigen::nbThreads() << std::endl;
        

        // RSR Related Method Initialization
        vector<vector<int8_t>> result_y_int;
        vector<vector<int8_t>> copied_x_int;
        vector<vector<int8_t>> copied_w;
        vector<vector<int8_t>> W_SSR = generateTernaryRandomMatrix(K_LEN, N_COL, Sparsity);
        vector<vector<int>>    B_SSR = generateBinaryMatrix(SSR_K);
        vector<vector<int8_t>> X_SSR_INT8 = generateRandomMatrix<int8_t>(K_LEN, M_ROW);
        vector<vector<float>>  X_SSR_FP32 = generateRandomMatrix<float >(K_LEN, M_ROW);
        // Generate filenames n=K_LEN, m=N_COL, k=M_ROW 
        string W_FileName = generateFilename(K_LEN, N_COL, Sparsity, "mat");
        string X_FileName = generateFilename(K_LEN, M_ROW, Sparsity, "v");
        // Save matrices to files (overwrites existing)
        saveMatrixToFile(W_SSR, W_FileName);
        saveMatrixToFile(X_SSR_INT8, X_FileName);
        // Preprocess  
        copied_w = copy(W_SSR);
        auto W_SSR_TRANS = ssr_preprocess(copied_w, SSR_K, K_LEN, N_COL);
        auto W_SSR_TRANS2 = ssr_preprocess2(copied_w, SSR_K, K_LEN, N_COL);
        //vector<int> SSR_PERMUTATION_POS = W_SSR_TRANS2.first.first;
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS = rsr_preprocess(copied_w, SSR_K);

        vector<vector<float>>  CP_X_SSR_FP32 = copy(X_SSR_FP32);
        vector<vector<int8_t>> CP_X_SSR_INT8 = copy(X_SSR_INT8);
        vector<vector<float>>  Y_SSR_FP32;
        vector<vector<int8_t>> Y_SSR_INT8;

        std::vector<int64_t> records(NUM_FUNCTIONS, 0);
        record_config(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, fptr);
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // *********** Benchmark Functions ************* 
            // Single Thread
            // Note that Eigen speed will be much lower if compiled with PyTorch...         
            Y_eigen = X_eigen * W_eigen;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_eigen = X_eigen * W_eigen_sparse;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // RSR & RSR++
            //cout << "RSR" << endl;
            Y_SSR_FP32 = rsr_inference(CP_X_SSR_FP32, W_RSR_TRANS.first.first, W_RSR_TRANS.first.second, W_RSR_TRANS.second.first, W_RSR_TRANS.second.second, B_SSR, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "RSR++" << endl;
            Y_SSR_FP32 = rsr_pp_inference(CP_X_SSR_FP32, W_RSR_TRANS.first.first, W_RSR_TRANS.first.second, W_RSR_TRANS.second.first, W_RSR_TRANS.second.second, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // SSR
            //cout << "SSR" << endl;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS.first.first, W_SSR_TRANS.first.second, W_SSR_TRANS.second.first, W_SSR_TRANS.second.second, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            Y_FP32_SSR = ssr_inference_float_vec_unroll16_v2(X_FP32, W_SSR_TRANS2.first.first, W_SSR_TRANS2.first.second, W_SSR_TRANS2.second.first, W_SSR_TRANS2.second.second, SSR_K, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            Y_FP32_SSR = ssr_inference_float_vec_k6(X_FP32, W_SSR_TRANS2.first.first, W_SSR_TRANS2.first.second, W_SSR_TRANS2.second.first, W_SSR_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // TCSC
            GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(X_FP32_Pointer, mergedTCSC_G8_Min.metadata.data(), mergedTCSC_G8_Min.row_index.data(), Rb, M_ROW, N_COL, K_LEN);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}

int main() {
    //verify_FP32_GEMM();
    benchmark_FP32_Matrix_RSR_SSR();
    //benchmark_FP32_Matrix_Torch_Eigen();
    //benchmark_FP32_MLP_Torch_Eigen();

    //verify_FP32_GEMM();
    //verify_FP32_GEMV();
    //verify_INT8_GEMM();
    //benchmark_FP32_GEMM_General_Optimizations();
    //benchmark_FP32_Matrix_Level();
    //benchmark_FP32_Layer_Level();
    //benchmark_FP32_Llama_Model();
    std::cin.get();
    return 0;
}