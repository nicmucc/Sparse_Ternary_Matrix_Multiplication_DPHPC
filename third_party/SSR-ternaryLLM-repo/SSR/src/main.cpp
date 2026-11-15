#include "util.h"

#include "initData.hpp" 
//#include "LlamaModel.hpp"
#include "TCSC.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <torch/torch.h>

#include "SSR.hpp"
using namespace std;


int benchmark_FP32_Matrix_Torch_Eigen(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_Matrix_Level_" + time_string() + ".txt";
    vector<string> names = { "PyTorchDense\t", "PyTorchSpCSR\t", "PyTorchSpCSC\t", "EigenSparse\t", "RSRPP____\t", "SSR_VEC_TREE\t", 
     "EigenSpars_INT8\t", "RSR_PP_INT8\t", "SSR_TREE_INT8\t",  };
    const int SSR_K = 4;
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        //  M    K      N    Sparsity, Data Variation
        {256, 1024,  4096, 0.45,0.05},
        {256, 1024,  4096, 0.50,0.05},
        {256, 1024,  4096, 0.55,0.05},
        {256, 1024,  4096, 0.60,0.05},
        {256, 1024,  4096, 0.65,0.05},
        {256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.75,0.05},
        {256, 1024,  4096, 0.80,0.05},
        {256, 1024,  4096, 0.85,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 1024,  4096, 0.95,0.05},
        {512, 2048,  8192, 0.45,0.05},
        {512, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.55,0.05},
        {512, 2048,  8192, 0.60,0.05},
        {512, 2048,  8192, 0.65,0.05},
        {512, 2048,  8192, 0.70,0.05},
        {512, 2048,  8192, 0.75,0.05},
        {512, 2048,  8192, 0.80,0.05},
        {512, 2048,  8192, 0.85,0.05},
        {512, 2048,  8192, 0.90,0.05},
        {512, 2048,  8192, 0.95,0.05},
        {512, 4096, 16384, 0.45,0.05},
        {512, 4096, 16384, 0.50,0.05},
        {512, 4096, 16384, 0.55,0.05},
        {512, 4096, 16384, 0.60,0.05},
        {512, 4096, 16384, 0.65,0.05},
        {512, 4096, 16384, 0.70,0.05},
        {512, 4096, 16384, 0.75,0.05},
        {512, 4096, 16384, 0.80,0.05},
        {512, 4096, 16384, 0.85,0.05},
        {512, 4096, 16384, 0.90,0.05},
        {512, 4096, 16384, 0.95,0.05},
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

        torch::Tensor tensorX = torch::from_blob(X_FP32.data(), { M_ROW, K_LEN }, torch::kFloat32);
        torch::Tensor tensorW = torch::from_blob(W_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorX_INT8 = torch::from_blob(X_INT8.data(), { M_ROW, K_LEN }, torch::kInt8);
        torch::Tensor tensorW_INT8 = torch::from_blob(W_INT8.data(), { K_LEN, N_COL }, torch::kInt8);
        torch::Tensor tensorWcsc = tensorW.to_sparse_csc();
        torch::Tensor tensorWcsr = tensorW.to_sparse_csr();
        torch::Tensor tensorY = torch::matmul(tensorX, tensorW);
        torch::Tensor tensorYcsr = torch::matmul(tensorX, tensorWcsr);
        torch::Tensor tensorYcsc = torch::matmul(tensorX, tensorWcsc);
        torch::Tensor tensorY_INT8 = torch::matmul(tensorX_INT8, tensorW_INT8);
        //torch::Tensor tensorWcsr_INT8 = tensorW_INT8.to_sparse_csc();
        //torch::Tensor tensorYcsr_INT8 = torch::matmul(tensorX_INT8, tensorWcsr_INT8);

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
        copied_w = copy(W_SSR);
        auto W_SSR_TRANS2 = ssr_preprocess(copied_w, SSR_K, K_LEN, N_COL);
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
            tensorY = torch::matmul(tensorX, tensorW); tensorY.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            tensorYcsr = torch::matmul(tensorX, tensorWcsr); tensorYcsr.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            tensorYcsc = torch::matmul(tensorX, tensorWcsc); tensorYcsc.sizes();
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // Note that Eigen speed will be much lower if compiled with PyTorch...  
            Y_eigen = X_eigen * W_eigen_sparse;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //cout << "RSR++" << endl;
            Y_SSR_FP32 = rsr_pp_inference(CP_X_SSR_FP32, W_RSR_TRANS.first.first, W_RSR_TRANS.first.second, W_RSR_TRANS.second.first, W_RSR_TRANS.second.second, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // SSR
            //cout << "SSR_unroll_tree" << endl;
            Y_FP32_SSR = ssr_inference_float_vec_L4(X_FP32, W_SSR_TRANS2.first.first, W_SSR_TRANS2.first.second, W_SSR_TRANS2.second.first, W_SSR_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // *********** Benchmark Functions ************* 
            // libTorch INT8 function are really slow... So I remove them in the benchmarking, otherwise it will take long time
            //tensorY_INT8 = torch::matmul(tensorX_INT8, tensorW_INT8); tensorY.sizes();
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++; 
            //tensorYcsr_INT8 = torch::matmul(tensorX_INT8, tensorWcsr_INT8); tensorYcsc.sizes();
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // Note that Eigen speed will be much lower if compiled with PyTorch...  
            Y_eigen_INT8 = X_eigen_INT8 * W_eigen_sparse_INT8;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            Y_SSR_INT8 = rsr_pp_inference(CP_X_SSR_INT8, W_RSR_TRANS.first.first, W_RSR_TRANS.first.second, W_RSR_TRANS.second.first, W_RSR_TRANS.second.second, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // SSR
            Y_INT8_SSR = ssr_inference_int8_vec_L4(X_INT8, W_SSR_TRANS2.first.first, W_SSR_TRANS2.first.second, W_SSR_TRANS2.second.first, W_SSR_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}

int benchmark_FP32_MLP_Torch_Eigen(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_Layer_Level_LlamaMLP_" + time_string() + ".txt";
    vector<string> names = { "PyTorchDense\t",  "PyTorchSpCSC\t", "EigenSparse\t", "RSRPP_FP32\t", "SSR_VEC_TREE\t", 
     "EigenSpars_INT8\t", "RSRPP_INT8\t", "SSR_TREE_INT8\t", };

    const int SSR_K = 8;
    //const int SSR_K8 = 8;
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        {256, 1024,  4096, 0.45,0.05},
        {256, 1024,  4096, 0.50,0.05},
        {256, 1024,  4096, 0.55,0.05},
        {256, 1024,  4096, 0.60,0.05},
        {256, 1024,  4096, 0.65,0.05},
        {256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.75,0.05},
        {256, 1024,  4096, 0.80,0.05},
        {256, 1024,  4096, 0.85,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 1024,  4096, 0.95,0.05},
        {512, 2048,  8192, 0.45,0.05},
        {512, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.55,0.05},
        {512, 2048,  8192, 0.60,0.05},
        {512, 2048,  8192, 0.65,0.05},
        {512, 2048,  8192, 0.70,0.05},
        {512, 2048,  8192, 0.75,0.05},
        {512, 2048,  8192, 0.80,0.05},
        {512, 2048,  8192, 0.85,0.05},
        {512, 2048,  8192, 0.90,0.05},
        {512, 2048,  8192, 0.95,0.05},
        {512, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.70,0.05},
        {512, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},
        {  1, 1024,  4096, 0.50,0.05}, // Different Sparsity
        {  1, 1024,  4096, 0.70,0.05},
        {  1, 1024,  4096, 0.90,0.05},
        {  1, 2048,  8192, 0.50,0.05},
        {  1, 2048,  8192, 0.70,0.05},
        {  1, 2048,  8192, 0.90,0.05},
        {  1, 4096, 16384, 0.50,0.05},
        {  1, 4096, 16384, 0.70,0.05},
        {  1, 4096, 16384, 0.90,0.05},
        {256, 1024,  4096, 0.50,0.05},
        {256, 1024,  4096, 0.55,0.05},
        {256, 1024,  4096, 0.60,0.05},
        {256, 1024,  4096, 0.65,0.05},
        {256, 1024,  4096, 0.70,0.05},
        {256, 1024,  4096, 0.75,0.05},
        {256, 1024,  4096, 0.80,0.05},
        {256, 1024,  4096, 0.85,0.05},
        {256, 1024,  4096, 0.90,0.05},
        {256, 1024,  4096, 0.95,0.05},
        {512, 2048,  8192, 0.45,0.05},
        {512, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.55,0.05},
        {512, 2048,  8192, 0.60,0.05},
        {512, 2048,  8192, 0.65,0.05},
        {512, 2048,  8192, 0.70,0.05},
        {512, 2048,  8192, 0.75,0.05},
        {512, 2048,  8192, 0.80,0.05},
        {512, 2048,  8192, 0.85,0.05},
        {512, 2048,  8192, 0.90,0.05},
        {512, 2048,  8192, 0.95,0.05},
        {512, 2048,  8192, 0.50,0.05},
        {512, 2048,  8192, 0.70,0.05},
        {512, 2048,  8192, 0.90,0.05},
        {256, 4096, 16384, 0.50,0.05},
        {256, 4096, 16384, 0.70,0.05},
        {256, 4096, 16384, 0.90,0.05},
        {  1, 1024,  4096, 0.50,0.05}, // Different Sparsity
        {  1, 1024,  4096, 0.70,0.05},
        {  1, 1024,  4096, 0.90,0.05},
        {  1, 2048,  8192, 0.50,0.05},
        {  1, 2048,  8192, 0.70,0.05},
        {  1, 2048,  8192, 0.90,0.05},
        {  1, 4096, 16384, 0.50,0.05},
        {  1, 4096, 16384, 0.70,0.05},
        {  1, 4096, 16384, 0.90,0.05},
    };

    std::vector<float> speedups;
    FILE* fptr = fopen(file_name.c_str(), "w");
    std::cout << "Running " << file_name << std::endl;
    for (const auto& [M_ROW, K_LEN, N_COL, Sparsity, Variation] : Config_MKNSV) {
        // *********** Data Initialization *************
        std::vector<float>  X_FP32 = initX<float>(M_ROW * K_LEN, 512);//Activation 
        std::vector<int8_t> X_INT8 = initX<int8_t>(M_ROW * K_LEN, 32);//Activation 
        std::vector<int8_t> Weight = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); //Weights not aligned, not uniformed 
        std::vector<int8_t> WeightG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, Variation, false, false); // Gate proj
        std::vector<int8_t> WeightD = sparseWeight<int8_t>(N_COL, K_LEN, Sparsity, Variation, false, false); // Down proj
        std::vector<int8_t> WeightUniform = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); //Weights uniformed (aligned and Variation ignored, Variation == 0) 
        std::vector<int8_t> WeightUniformG = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); // Gate proj
        std::vector<int8_t> WeightUniformD = sparseWeight<int8_t>(K_LEN, N_COL, Sparsity, 0, false, true); // Down proj

        std::vector<float> Weight_FP32(Weight.begin(), Weight.end());
        std::vector<float> Weight_FP32G(WeightG.begin(), WeightG.end());
        std::vector<float> Weight_FP32D(WeightD.begin(), WeightD.end());
        std::vector<float> Y_Ref(M_ROW * N_COL, 0);  //Baseline used for correctness
        std::vector<float> Y_Cal(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<int8_t> Y_INT8(M_ROW * N_COL, 0);  //Result of the multiplication
        std::vector<int8_t> Y_INT8G(M_ROW * N_COL, 0);  //Result of the multiplication

        SparseFormat naiveTCSC = SparseFormat(Weight.data(), K_LEN, N_COL);
        SparseFormat naiveTCSCG = SparseFormat(WeightG.data(), K_LEN, N_COL);
        SparseFormat naiveTCSCD = SparseFormat(WeightD.data(), N_COL, K_LEN);
        MergedTCSC_Group mergedTCSC_G8_Min = MergedTCSC_Group(naiveTCSC, K_LEN, N_COL, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G8_MinG = MergedTCSC_Group(naiveTCSCG, K_LEN, N_COL, 8, "min", true);
        MergedTCSC_Group mergedTCSC_G8_MinD = MergedTCSC_Group(naiveTCSCD, N_COL, K_LEN, 8, "min", true);

        float* X_FP32_Pointer = X_FP32.data();// Convert vectors into *
        int8_t* W_INT8 = Weight.data();
        float* Ra = Y_Ref.data();
        float* Rb = Y_Cal.data();
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen(X_FP32_Pointer, M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen(Weight_FP32.data(), K_LEN, N_COL);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenG(Weight_FP32G.data(), K_LEN, N_COL);
        Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenD(Weight_FP32D.data(), N_COL, K_LEN);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen(M_ROW, N_COL);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigenG(M_ROW, N_COL);
        Eigen::SparseMatrix<float> W_eigen_sparse = W_eigen.sparseView();
        Eigen::SparseMatrix<float> W_eigen_sparseG = W_eigenG.sparseView();
        Eigen::SparseMatrix<float> W_eigen_sparseD = W_eigenD.sparseView();
        Eigen::Map < Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X_eigen_INT8(X_INT8.data(), M_ROW, K_LEN);
        Eigen::Map < Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigen_INT8(Weight.data(), K_LEN, N_COL);
        Eigen::Map < Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenG_INT8(WeightG.data(), K_LEN, N_COL);
        Eigen::Map < Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_eigenD_INT8(WeightD.data(), N_COL, K_LEN);
        Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Y_eigen_INT8(M_ROW, N_COL);
        Eigen::SparseMatrix<int8_t> W_eigen_sparse_INT8 = W_eigen_INT8.sparseView();
        Eigen::SparseMatrix<int8_t> W_eigen_sparseG_INT8 = W_eigenG_INT8.sparseView();
        Eigen::SparseMatrix<int8_t> W_eigen_sparseD_INT8 = W_eigenD_INT8.sparseView();
        cout << "Eigen Nb Threads =" << Eigen::nbThreads() << endl; // Verified to be 16 threads = max threads available

        torch::Tensor tensorX = torch::from_blob(X_FP32.data(), { M_ROW, K_LEN }, torch::kFloat32);
        torch::Tensor tensorW = torch::from_blob(Weight_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorWcsc = tensorW.to_sparse_csc();
        torch::Tensor tensorWcsr = tensorW.to_sparse_csr();
        torch::Tensor tensorY = torch::matmul(tensorX, tensorW);
        torch::Tensor tensorYcsr = torch::matmul(tensorX, tensorWcsr);
        torch::Tensor tensorYcsc = torch::matmul(tensorX, tensorWcsc);

        torch::Tensor tensorWG = torch::from_blob(Weight_FP32.data(), { K_LEN, N_COL }, torch::kFloat32);
        torch::Tensor tensorWcscG = tensorWG.to_sparse_csc();
        torch::Tensor tensorWD = torch::from_blob(Weight_FP32.data(), { N_COL, K_LEN }, torch::kFloat32);
        torch::Tensor tensorWcscD = tensorWD.to_sparse_csc();

        torch::Tensor tensorX_INT8 = torch::from_blob(X_INT8.data(), { M_ROW, K_LEN }, torch::kInt8);
        torch::Tensor tensorW_INT8 = torch::from_blob(W_INT8, { K_LEN, N_COL }, torch::kInt8);
        torch::Tensor tensorWG_INT8 = torch::from_blob(WeightG.data(), { K_LEN, N_COL }, torch::kInt8);
        torch::Tensor tensorWD_INT8 = torch::from_blob(WeightD.data(), { N_COL, K_LEN }, torch::kInt8);
        torch::Tensor tensorY_INT8 = torch::matmul(tensorX_INT8, tensorW_INT8);

        // RSR Related Method Initialization
        vector<vector<int8_t>> result_y_int;
        vector<vector<int8_t>> copied_x_int;
        vector<vector<int8_t>> copied_w;
        vector<vector<int8_t>> W_SSR = generateTernaryRandomMatrix(K_LEN, N_COL, Sparsity);
        vector<vector<int8_t>> W_SSRG = generateTernaryRandomMatrix(K_LEN, N_COL, Sparsity);
        vector<vector<int8_t>> W_SSRD = generateTernaryRandomMatrix(N_COL, K_LEN, Sparsity);
        vector<vector<int>>    B_SSR = generateBinaryMatrix(SSR_K);
        vector<vector<int8_t>> X_SSR_INT8 = generateRandomMatrix<int8_t>(K_LEN, M_ROW);
        vector<vector<float>>  X_SSR_FP32 = generateRandomMatrix<float >(K_LEN, M_ROW);
        // Generate filenames n=K_LEN, m=N_COL, k=M_ROW 
        string W_FileName = generateFilename(K_LEN, N_COL, Sparsity, "mat");
        string X_FileName = generateFilename(K_LEN, M_ROW, Sparsity, "v");
        // Save matrices to files (overwrites existing)
        //saveMatrixToFile(W_SSR, W_FileName);
        //saveMatrixToFile(X_SSR_INT8, X_FileName);
        // Preprocess  
        copied_w = copy(W_SSR);
        auto W_SSR_TRANS = ssr_preprocess(copied_w, SSR_K, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS = rsr_preprocess(copied_w, SSR_K);
        copied_w = copy(W_SSRG);
        auto W_SSRG_TRANS = ssr_preprocess(copied_w, SSR_K, K_LEN, N_COL);
        copied_w = copy(W_SSRG);  // Reset for second preprocessing
        auto W_RSRG_TRANS = rsr_preprocess(copied_w, SSR_K);
        copied_w = copy(W_SSRD);
        auto W_SSRD_TRANS = ssr_preprocess(copied_w, SSR_K, N_COL, K_LEN);
        copied_w = copy(W_SSRD);  // Reset for second preprocessing
        auto W_RSRD_TRANS = rsr_preprocess(copied_w, SSR_K);
        copied_w = copy(W_SSR);
        auto W_SSR_TRANS2 = ssr_preprocess(copied_w, SSR_K, K_LEN, N_COL);
        copied_w = copy(W_SSRG);
        auto W_SSRG_TRANS2 = ssr_preprocess(copied_w, SSR_K, K_LEN, N_COL);
        copied_w = copy(W_SSRD);
        auto W_SSRD_TRANS2 = ssr_preprocess(copied_w, SSR_K, N_COL, K_LEN);

        vector<vector<float>>  CP_X_SSR_FP32 = copy(X_SSR_FP32);
        vector<vector<int8_t>> CP_X_SSR_INT8 = copy(X_SSR_INT8);
        vector<vector<float>>  Y_RSR_FP32;
        vector<vector<int8_t>> Y_RSR_INT8;
        vector<vector<float>>  Y_RSR_FP32G;
        vector<vector<int8_t>> Y_RSR_INT8G;
        std::vector<float> Y_SSR_FP32;  //Baseline used for correctness
        std::vector<float> Y_SSR_FP32G;  //Result of the multiplication
        std::vector<int8_t> Y_SSR_INT8;  //Result of the multiplication
        std::vector<int8_t> Y_SSR_INT8G;

        std::vector<int64_t> records(NUM_FUNCTIONS, 0);
        record_config(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, fptr);
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // *********** Benchmark Functions ************* 
            tensorY = torch::matmul(tensorX, tensorW);
            torch::Tensor tensorG = torch::matmul(tensorX, tensorWG);
            //tensorY = torch::mul(tensorY, torch::silu(tensorG));
            tensorY = torch::matmul(tensorY, tensorWD);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            tensorYcsc = torch::matmul(tensorX, tensorWcsc); tensorYcsc.sizes();
            torch::Tensor tensorGcsc = torch::matmul(tensorX, tensorWcscG);
            //tensorYcsc = torch::mul(tensorYcsc, torch::silu(tensorGcsc));
            tensorYcsc = torch::matmul(tensorYcsc, tensorWcscD);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //// Note that Eigen speed will be much lower if compiled with PyTorch...         
            Y_eigen = X_eigen * W_eigen_sparse;
            Y_eigenG = X_eigen * W_eigen_sparseG;
            //Y_eigenG = Y_eigenG.array() * Y_eigen.array() / (1.0f + (-Y_eigenG).array().exp());
            Y_eigen = Y_eigenG * W_eigen_sparseD;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //std::cout << "rsr_pp" << std::endl;
            Y_RSR_FP32 = rsr_pp_inference(CP_X_SSR_FP32, W_RSR_TRANS.first.first, W_RSR_TRANS.first.second, W_RSR_TRANS.second.first, W_RSR_TRANS.second.second, SSR_K);
            Y_RSR_FP32G = rsr_pp_inference(CP_X_SSR_FP32, W_RSRG_TRANS.first.first, W_RSRG_TRANS.first.second, W_RSRG_TRANS.second.first, W_RSRG_TRANS.second.second, SSR_K);
            Y_RSR_FP32 = rsr_pp_inference(Y_RSR_FP32G, W_RSRD_TRANS.first.first, W_RSRD_TRANS.first.second, W_RSRD_TRANS.second.first, W_RSRD_TRANS.second.second, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //std::cout << "ssr" << std::endl;
            Y_SSR_FP32 = ssr_inference_float_vec_L8(X_FP32, W_SSR_TRANS2.first.first, W_SSR_TRANS2.first.second, W_SSR_TRANS2.second.first, W_SSR_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            Y_SSR_FP32G = ssr_inference_float_vec_L8(X_FP32, W_SSRG_TRANS2.first.first, W_SSRG_TRANS2.first.second, W_SSRG_TRANS2.second.first, W_SSRG_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            Y_SSR_FP32 = ssr_inference_float_vec_L8(Y_SSR_FP32G, W_SSRD_TRANS2.first.first, W_SSRD_TRANS2.first.second, W_SSRD_TRANS2.second.first, W_SSRD_TRANS2.second.second, N_COL, K_LEN, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            ////cout << "TorchINT8" << endl;
            // libTorch INT8 is really slow on the tested CPU...
            //tensorY_INT8 = torch::matmul(tensorX_INT8, tensorW_INT8);
            //torch::Tensor tensorG_INT8 = torch::matmul(tensorX_INT8, tensorWG_INT8);
            ////tensorG_INT8 = torch::mul(tensorY_INT8, torch::silu(tensorG_INT8));
            //tensorY_INT8 = torch::matmul(tensorG_INT8, tensorWD_INT8);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //cout << "EIgen_INt8" << endl;
            Y_eigen_INT8 = X_eigen_INT8 * W_eigen_sparse_INT8;
            auto Y_eigenG_INT8 = X_eigen_INT8 * W_eigen_sparseG_INT8;
            // Y_eigenG_INT8 = Y_eigenG_INT8.array() * Y_eigen_INT8.array() / (1.0 + (-Y_eigenG_INT8).array().exp());
            Y_eigen_INT8 = Y_eigenG_INT8 * W_eigen_sparseD_INT8;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //cout << "RSR_INT8" << endl;
            Y_RSR_INT8 = rsr_pp_inference(CP_X_SSR_INT8, W_RSR_TRANS.first.first, W_RSR_TRANS.first.second, W_RSR_TRANS.second.first, W_RSR_TRANS.second.second, SSR_K);
            Y_RSR_INT8G = rsr_pp_inference(CP_X_SSR_INT8, W_RSRG_TRANS.first.first, W_RSRG_TRANS.first.second, W_RSRG_TRANS.second.first, W_RSRG_TRANS.second.second, SSR_K);
            Y_RSR_INT8 = rsr_pp_inference(Y_RSR_INT8G, W_RSRD_TRANS.first.first, W_RSRD_TRANS.first.second, W_RSRD_TRANS.second.first, W_RSRD_TRANS.second.second, SSR_K);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            //cout << "SSR_unroll_tree" << endl;
            Y_SSR_INT8 = ssr_inference_int8_vec_L8(X_INT8, W_SSR_TRANS2.first.first, W_SSR_TRANS2.first.second, W_SSR_TRANS2.second.first, W_SSR_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            Y_SSR_INT8G = ssr_inference_int8_vec_L8(X_INT8, W_SSRG_TRANS2.first.first, W_SSRG_TRANS2.first.second, W_SSRG_TRANS2.second.first, W_SSRG_TRANS2.second.second, K_LEN, N_COL, M_ROW);
            Y_SSR_INT8 = ssr_inference_int8_vec_L8(Y_SSR_INT8G, W_SSRD_TRANS2.first.first, W_SSRD_TRANS2.first.second, W_SSRD_TRANS2.second.first, W_SSRD_TRANS2.second.second, N_COL, K_LEN, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}

// Function used to determine the best L factor depending on size and sparsity of the input matrix.
int benchmark_FP32_Matrix_allL(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 40) {
    // *********** File Name and Function Names *************
    string file_name = "GEMM_CPU_FP32_Matrix_testK_" + time_string() + ".txt";
    vector<string> names = { "1\t", "2\t", "3\t", "4\t", "5\t", "6\t", "7\t", "8\t", "9\t", "10\t", "11\t", "12\t", "13\t", "14\t", "15\t", "16\t", "17\t", "18\t", "19\t", "20\t", "21\t", "22\t", "23\t", "24\t", };
    std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV = {
        //  M    K      N    Sparsity, Data Variation
        {512, 1024,  4096, 0.45,0.05},
        //{512, 1024,  4096, 0.50,0.05},
        //{512, 1024,  4096, 0.55,0.05},
        //{512, 1024,  4096, 0.60,0.05},
        //{512, 1024,  4096, 0.65,0.05},
        //{512, 1024,  4096, 0.70,0.05},
        //{512, 1024,  4096, 0.75,0.05},
        //{512, 1024,  4096, 0.80,0.05},
        //{512, 1024,  4096, 0.85,0.05},
        //{512, 1024,  4096, 0.90,0.05},
        //{512, 1024,  4096, 0.95,0.05},
        //{512, 2048,  8192, 0.45,0.05},
        //{512, 2048,  8192, 0.50,0.05},
        //{512, 2048,  8192, 0.55,0.05},
        //{512, 2048,  8192, 0.60,0.05},
        //{512, 2048,  8192, 0.65,0.05},
        //{512, 2048,  8192, 0.70,0.05},
        //{512, 2048,  8192, 0.75,0.05},
        //{512, 2048,  8192, 0.80,0.05},
        //{512, 2048,  8192, 0.85,0.05},
        //{512, 2048,  8192, 0.90,0.05},
        //{512, 2048,  8192, 0.95,0.05},
        //{512, 4096, 16384, 0.45,0.05},
        //{512, 4096, 16384, 0.50,0.05},
        //{512, 4096, 16384, 0.55,0.05},
        //{512, 4096, 16384, 0.60,0.05},
        //{512, 4096, 16384, 0.65,0.05},
        //{512, 4096, 16384, 0.70,0.05},
        //{512, 4096, 16384, 0.75,0.05},
        //{512, 4096, 16384, 0.80,0.05},
        //{512, 4096, 16384, 0.85,0.05},
        //{512, 4096, 16384, 0.90,0.05},
        //{512, 4096, 16384, 0.95,0.05},
    };
    // X: MxK, W: KxN, Y=MxN, SSR_L is the L factor used in RSR and SSR methods.

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
        std::vector<int8_t> Y_INT8(M_ROW * N_COL, 0);  //Result of the multiplication

        cout << "Init RSR" << endl;
        // RSR Related Method Initialization
        vector<vector<int8_t>> result_y_int;
        vector<vector<int8_t>> copied_x_int;
        vector<vector<int8_t>> copied_w;
        vector<vector<int8_t>> W_SSR = generateTernaryRandomMatrix(K_LEN, N_COL, Sparsity);
        cout << "Generating Matrix" << endl;
        vector<vector<int8_t>> X_RSR_INT8 = generateRandomMatrix<int8_t>(K_LEN, M_ROW);
        vector<vector<float>>  X_RSR_FP32 = generateRandomMatrix<float >(K_LEN, M_ROW);
        vector<int8_t> X_SSR_INT8 = generateRandomMatrix2<int8_t>(K_LEN, M_ROW);
        vector<float>  X_SSR_FP32 = generateRandomMatrix2<float >(K_LEN, M_ROW);
        cout << "Generating FileName" << endl;
        // Generate filenames n=K_LEN, m=N_COL, k=M_ROW 
        string W_FileName = generateFilename(K_LEN, N_COL, Sparsity, "mat");
        string X_FileName = generateFilename(K_LEN, M_ROW, Sparsity, "v");
        // Save matrices to files (overwrites existing)
        cout << W_FileName << X_FileName << endl;
        saveMatrixToFile(W_SSR, W_FileName);
        saveMatrixToFile(X_SSR_INT8, X_FileName);
        // Preprocess  
        cout << "Copy SSR" << endl;
        
        copied_w = copy(W_SSR);
        auto W_SSR_TRANS_L2 = ssr_preprocess(copied_w, 2, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L3 = ssr_preprocess(copied_w, 3, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L4 = ssr_preprocess(copied_w, 4, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L5 = ssr_preprocess(copied_w, 5, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L6 = ssr_preprocess(copied_w, 6, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L7 = ssr_preprocess(copied_w, 7, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L8 = ssr_preprocess(copied_w, 8, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L9 = ssr_preprocess(copied_w, 9, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L10 = ssr_preprocess(copied_w, 10, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L11 = ssr_preprocess(copied_w, 11, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L12 = ssr_preprocess(copied_w, 12, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L13 = ssr_preprocess(copied_w, 13, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L14 = ssr_preprocess(copied_w, 14, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L15 = ssr_preprocess(copied_w, 15, K_LEN, N_COL);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_SSR_TRANS_L16 = ssr_preprocess(copied_w, 16, K_LEN, N_COL);

        copied_w = copy(W_SSR);
        auto W_RSR_TRANS_L4 = rsr_preprocess(copied_w, 4);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS_L5 = rsr_preprocess(copied_w, 5);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS_L6 = rsr_preprocess(copied_w, 6);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS_L7 = rsr_preprocess(copied_w, 7);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS_L8 = rsr_preprocess(copied_w, 8);
        copied_w = copy(W_SSR);  // Reset for second preprocessing
        auto W_RSR_TRANS_L9 = rsr_preprocess(copied_w, 9);
        /*
        copied_w = copy(W_SSR);
        auto W_SSR_PAIR_F_U = ssr_preprocess_float_vec_unroll(copied_w, K_LEN, N_COL, Sparsity);
        int SSR_L_F_U = W_SSR_PAIR_F_U.second;
        auto W_SSR_TRANS_F_U = W_SSR_PAIR_F_U.first;
        copied_w = copy(W_SSR);
        auto W_SSR_PAIR_I8_U = ssr_preprocess_int8_vec_unroll(copied_w, K_LEN, N_COL, Sparsity);
        int SSR_L_I8_U = W_SSR_PAIR_I8_U.second;
        auto W_SSR_TRANS_I8_U = W_SSR_PAIR_I8_U.first;
        */

        vector<float>  CP_X_SSR_FP32 = copy(X_SSR_FP32);
        vector<int8_t> CP_X_SSR_INT8 = copy(X_SSR_INT8);
        vector<float>  Y_SSR_FP32;
        vector<int8_t> Y_SSR_INT8;
        vector<vector<float>>  CP_X_RSR_FP32 = copy(X_RSR_FP32);
        vector<vector<int8_t>> CP_X_RSR_INT8 = copy(X_RSR_INT8);
        vector<vector<float>>  Y_RSR_FP32;
        vector<vector<int8_t>> Y_RSR_INT8;

        cout << "Start Benchmarking" << endl;
        std::vector<int64_t> records(NUM_FUNCTIONS, 0);
        record_config(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, fptr);
        for (int i = 0; i < NUM_RUNS; i++) {
            vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
            int j = 0;
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            // SSR
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_F_U.first.first, W_SSR_TRANS_F_U.first.second, W_SSR_TRANS_F_U.second.first, W_SSR_TRANS_F_U.second.second, SSR_L_F_U, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            
            Y_RSR_FP32 = rsr_pp_inference(CP_X_RSR_FP32, W_RSR_TRANS_L4.first.first, W_RSR_TRANS_L4.first.second, W_RSR_TRANS_L4.second.first, W_RSR_TRANS_L4.second.second, 4);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_FP32 = rsr_pp_inference(CP_X_RSR_FP32, W_RSR_TRANS_L5.first.first, W_RSR_TRANS_L5.first.second, W_RSR_TRANS_L5.second.first, W_RSR_TRANS_L5.second.second, 5);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_FP32 = rsr_pp_inference(CP_X_RSR_FP32, W_RSR_TRANS_L6.first.first, W_RSR_TRANS_L6.first.second, W_RSR_TRANS_L6.second.first, W_RSR_TRANS_L6.second.second, 6);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_FP32 = rsr_pp_inference(CP_X_RSR_FP32, W_RSR_TRANS_L7.first.first, W_RSR_TRANS_L7.first.second, W_RSR_TRANS_L7.second.first, W_RSR_TRANS_L7.second.second, 7);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_FP32 = rsr_pp_inference(CP_X_RSR_FP32, W_RSR_TRANS_L8.first.first, W_RSR_TRANS_L8.first.second, W_RSR_TRANS_L8.second.first, W_RSR_TRANS_L8.second.second, 8);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_FP32 = rsr_pp_inference(CP_X_RSR_FP32, W_RSR_TRANS_L9.first.first, W_RSR_TRANS_L9.first.second, W_RSR_TRANS_L9.second.first, W_RSR_TRANS_L9.second.second, 9);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            
            //cout << "SSR" << endl;
            
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, 2, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, 3, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, 4, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, 5, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, 6, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, 7, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, 8, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            
            
            /*
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L9.first.first, W_SSR_TRANS_L9.first.second, W_SSR_TRANS_L9.second.first, W_SSR_TRANS_L9.second.second, 9, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L10.first.first, W_SSR_TRANS_L10.first.second, W_SSR_TRANS_L10.second.first, W_SSR_TRANS_L10.second.second, 10, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference(CP_X_SSR_FP32, W_SSR_TRANS_L11.first.first, W_SSR_TRANS_L11.first.second, W_SSR_TRANS_L11.second.first, W_SSR_TRANS_L11.second.second, 11, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, 2);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_L2(CP_X_SSR_FP32, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, 2, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, 3, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, 4, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, 5, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, 6, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, 7, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, 8, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            /*
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L9.first.first, W_SSR_TRANS_L9.first.second, W_SSR_TRANS_L9.second.first, W_SSR_TRANS_L9.second.second, 9, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L10.first.first, W_SSR_TRANS_L10.first.second, W_SSR_TRANS_L10.second.first, W_SSR_TRANS_L10.second.second, 10, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            /*
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L11.first.first, W_SSR_TRANS_L11.first.second, W_SSR_TRANS_L11.second.first, W_SSR_TRANS_L11.second.second, 11, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L12.first.first, W_SSR_TRANS_L12.first.second, W_SSR_TRANS_L12.second.first, W_SSR_TRANS_L12.second.second, 12, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_unroll8_v2(CP_X_SSR_FP32, W_SSR_TRANS_L13.first.first, W_SSR_TRANS_L13.first.second, W_SSR_TRANS_L13.second.first, W_SSR_TRANS_L13.second.second, 13, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            
            //cout << "SSR_unroll_tree" << endl;
            Y_SSR_FP32 = ssr_inference_float_vec_L3(CP_X_SSR_FP32, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, 4);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            Y_SSR_FP32 = ssr_inference_float_vec_L4(CP_X_SSR_FP32, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, 5);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            Y_SSR_FP32 = ssr_inference_float_vec_L5(CP_X_SSR_FP32, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, 6);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            /*
            Y_SSR_FP32 = ssr_inference_float_vec_L6(CP_X_SSR_FP32, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, 7);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_L7(CP_X_SSR_FP32, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, 8);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_FP32 = ssr_inference_float_vec_L8(CP_X_SSR_FP32, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_FP32 = ssr_inference_float_vec_unroll16_v2(CP_X_SSR_FP32, W_SSR_TRANS_L9.first.first, W_SSR_TRANS_L9.first.second, W_SSR_TRANS_L9.second.first, W_SSR_TRANS_L9.second.second, 9);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */

            // *********** Benchmark Functions ************* 

            // SSR
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_I8_U.first.first, W_SSR_TRANS_I8_U.first.second, W_SSR_TRANS_I8_U.second.first, W_SSR_TRANS_I8_U.second.second, SSR_L_I8_U, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            /*
            Y_RSR_INT8 = rsr_pp_inference(CP_X_RSR_INT8, W_RSR_TRANS_L4.first.first, W_RSR_TRANS_L4.first.second, W_RSR_TRANS_L4.second.first, W_RSR_TRANS_L4.second.second, 4);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_INT8 = rsr_pp_inference(CP_X_RSR_INT8, W_RSR_TRANS_L5.first.first, W_RSR_TRANS_L5.first.second, W_RSR_TRANS_L5.second.first, W_RSR_TRANS_L5.second.second, 5);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_INT8 = rsr_pp_inference(CP_X_RSR_INT8, W_RSR_TRANS_L6.first.first, W_RSR_TRANS_L6.first.second, W_RSR_TRANS_L6.second.first, W_RSR_TRANS_L6.second.second, 6);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            /*
            Y_RSR_INT8 = rsr_pp_inference(CP_X_RSR_INT8, W_RSR_TRANS_L7.first.first, W_RSR_TRANS_L7.first.second, W_RSR_TRANS_L7.second.first, W_RSR_TRANS_L7.second.second, 7);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            /*
            Y_RSR_INT8 = rsr_pp_inference(CP_X_RSR_INT8, W_RSR_TRANS_L8.first.first, W_RSR_TRANS_L8.first.second, W_RSR_TRANS_L8.second.first, W_RSR_TRANS_L8.second.second, 8);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_RSR_INT8 = rsr_pp_inference(CP_X_RSR_INT8, W_RSR_TRANS_L9.first.first, W_RSR_TRANS_L9.first.second, W_RSR_TRANS_L9.second.first, W_RSR_TRANS_L9.second.second, 9);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR" << endl;
            */
            /*
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, 2, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, 3, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            /*
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, 4, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, 5, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, 6, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, 7, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, 8, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference(CP_X_SSR_INT8, W_SSR_TRANS_L9.first.first, W_SSR_TRANS_L9.first.second, W_SSR_TRANS_L9.second.first, W_SSR_TRANS_L9.second.second, 9, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            //cout << "SSR_vec_unroll" << endl;
            /*
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, 2, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, 3, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, 4, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            /*
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, 5, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, 6, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, 7, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, 8, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L9.first.first, W_SSR_TRANS_L9.first.second, W_SSR_TRANS_L9.second.first, W_SSR_TRANS_L9.second.second, 9, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L10.first.first, W_SSR_TRANS_L10.first.second, W_SSR_TRANS_L10.second.first, W_SSR_TRANS_L10.second.second, 10, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L11.first.first, W_SSR_TRANS_L11.first.second, W_SSR_TRANS_L11.second.first, W_SSR_TRANS_L11.second.second, 11, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L12.first.first, W_SSR_TRANS_L12.first.second, W_SSR_TRANS_L12.second.first, W_SSR_TRANS_L12.second.second, 12, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L13.first.first, W_SSR_TRANS_L13.first.second, W_SSR_TRANS_L13.second.first, W_SSR_TRANS_L13.second.second, 13, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            /*
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L14.first.first, W_SSR_TRANS_L14.first.second, W_SSR_TRANS_L14.second.first, W_SSR_TRANS_L14.second.second, 14, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L15.first.first, W_SSR_TRANS_L15.first.second, W_SSR_TRANS_L15.second.first, W_SSR_TRANS_L15.second.second, 15, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            Y_SSR_INT8 = ssr_inference_int8_vec_unroll8_v2(CP_X_SSR_INT8, W_SSR_TRANS_L16.first.first, W_SSR_TRANS_L16.first.second, W_SSR_TRANS_L16.second.first, W_SSR_TRANS_L16.second.second, 16, K_LEN, N_COL, M_ROW);
            timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            */
            //cout << "SSR_unroll_tree" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L2(CP_X_SSR_INT8, W_SSR_TRANS_L2.first.first, W_SSR_TRANS_L2.first.second, W_SSR_TRANS_L2.second.first, W_SSR_TRANS_L2.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, 3);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L3(CP_X_SSR_INT8, W_SSR_TRANS_L3.first.first, W_SSR_TRANS_L3.first.second, W_SSR_TRANS_L3.second.first, W_SSR_TRANS_L3.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, 4);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L4(CP_X_SSR_INT8, W_SSR_TRANS_L4.first.first, W_SSR_TRANS_L4.first.second, W_SSR_TRANS_L4.second.first, W_SSR_TRANS_L4.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, 5);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L5(CP_X_SSR_INT8, W_SSR_TRANS_L5.first.first, W_SSR_TRANS_L5.first.second, W_SSR_TRANS_L5.second.first, W_SSR_TRANS_L5.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, 6);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_unroll_tree" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L6(CP_X_SSR_INT8, W_SSR_TRANS_L6.first.first, W_SSR_TRANS_L6.first.second, W_SSR_TRANS_L6.second.first, W_SSR_TRANS_L6.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, 7);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L7(CP_X_SSR_INT8, W_SSR_TRANS_L7.first.first, W_SSR_TRANS_L7.first.second, W_SSR_TRANS_L7.second.first, W_SSR_TRANS_L7.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, 8);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //Y_SSR_INT8 = ssr_inference_int8_vec_L8(CP_X_SSR_INT8, W_SSR_TRANS_L8.first.first, W_SSR_TRANS_L8.first.second, W_SSR_TRANS_L8.second.first, W_SSR_TRANS_L8.second.second, K_LEN, N_COL, M_ROW);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
            //cout << "SSR_vec_unroll" << endl;
            //Y_SSR_INT8 = ssr_inference_int8_vec_unroll32_v2(CP_X_SSR_INT8, W_SSR_TRANS_L9.first.first, W_SSR_TRANS_L9.first.second, W_SSR_TRANS_L9.second.first, W_SSR_TRANS_L9.second.second, 9);
            //timePoints[j] = std::chrono::high_resolution_clock::now(); j++;

            records = record_time(names, timePoints, records, fptr);
        }
        speedups = print_ms_speedup(names, M_ROW, N_COL, K_LEN, Sparsity, Variation, speedups, records, fptr);
    }
    fclose(fptr);
    print_speedup_summary(names, Config_MKNSV, speedups);
    return 0;
}


int main() {
    benchmark_FP32_Matrix_Torch_Eigen();
    benchmark_FP32_MLP_Torch_Eigen();
    // benchmark_FP32_Matrix_allL();


    std::cin.get();
    return 0;
}