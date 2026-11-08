#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

#include "baseline_suite.h"
#include "data_reader.h"

using namespace baseline_suite;

int main(int argc, char** argv) try {
    int width = 6, samples = 15;
    double min_ms = 10;
    std::string dataset;

    // Parse "--key value" options
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("Missing option value");
        const std::string value = argv[++i];

        if (key == "--width") width = std::stoi(value);
        else if (key == "--samples") samples = std::stoi(value);
        else if (key == "--min-ms") min_ms = std::stod(value);
        else if (key == "--dataset") dataset = value;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (dataset.empty())
        throw std::invalid_argument("Specify a saved input with --dataset");
    if (width < 1 || width > 8 || samples < 3 || !std::isfinite(min_ms) || min_ms <= 0)
        throw std::invalid_argument("Invalid benchmark configuration");

    // Load the saved matrices and their generation settings before timing
    auto data = read_sample(dataset);
    if (data.M > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        data.K > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        data.N > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Dataset dimensions exceed supported range");
    const int M = data.M, K = data.K, N = data.N;
    const int seed = data.seed;
    const double sparsity = data.target_zero_fraction;
    const auto& W = data.W;
    const std::vector<vec_t> inputs{std::move(data.X)};
    std::mt19937 rng(seed);

    Suite suite(inputs, W, M, K, N, width);

    // Validate every kernel on every input against an FP64 reference before timing
    for (std::size_t input = 0; input < inputs.size(); ++input) {
        const auto bounds = error_bounds(inputs[input], W, M, K, N);
        std::vector<double> reference(static_cast<std::size_t>(M) * N);
        for (int row = 0; row < M; ++row)
            for (int k = 0; k < K; ++k)
                for (int col = 0; col < N; ++col)
                    reference[row * N + col] += static_cast<double>(inputs[input][row * K + k]) * W[k * N + col];

        std::vector<double> tolerances(reference.size());
        for (std::size_t j = 0; j < reference.size(); ++j)
            tolerances[j] = tolerance(bounds[j], reference[j], K);

        for (auto& kernel : suite.kernels)
            validate(kernel, input, reference, tolerances);
    }

    // Times `count` calls; reading the output keeps the compiler from dropping them
    std::vector<std::size_t> iterations(suite.kernels.size(), 1);
    std::vector<std::vector<double>> timings(suite.kernels.size());
    volatile float checksum = 0;
    auto batch = [&](std::size_t impl, std::size_t count) {
        auto& kernel = suite.kernels[impl];
        const auto start = Clock::now();
        for (std::size_t i = 0; i < count; ++i) {
            kernel.run(i % inputs.size());
            checksum = kernel.output().data[(i * 17) % (static_cast<std::size_t>(M) * N)];
        }
        return microseconds(Clock::now() - start);
    };

    // Calibration doubles as warmup. Cover the complete input pool in each batch.
    for (std::size_t i = 0; i < suite.kernels.size(); ++i) {
        iterations[i] = inputs.size();
        while (batch(i, iterations[i]) < min_ms * 1000) {
            if (iterations[i] > std::numeric_limits<std::size_t>::max() / 2)
                throw std::runtime_error("Benchmark calibration overflow");
            iterations[i] *= 2;
        }
    }

    // Measure, shuffling kernel order every sample
    std::vector<std::size_t> order(suite.kernels.size());
    std::iota(order.begin(), order.end(), 0);
    for (int sample = 0; sample < samples; ++sample) {
        std::shuffle(order.begin(), order.end(), rng);
        for (auto i : order) timings[i].push_back(batch(i, iterations[i]) / iterations[i]);
    }

    std::vector<double> medians;
    for (auto times : timings) {
        std::sort(times.begin(), times.end());
        medians.push_back(samples % 2 ? times[samples / 2] : (times[samples / 2 - 1] + times[samples / 2]) / 2);
    }

    const auto blas = std::find_if(suite.kernels.begin(), suite.kernels.end(),
                                   [](const Kernel& k) { return k.name == "BLAS"; }) - suite.kernels.begin();
    if (blas == static_cast<std::ptrdiff_t>(suite.kernels.size()))
        throw std::runtime_error("BLAS reference is not registered");

    // Weight statistics: actual sparsity and distinct nonzero patterns per column block
    const auto nonzero = std::count_if(W.begin(), W.end(), [](int w) { return w != 0; });
    std::size_t pattern_count = 0;
    for (int col = 0; col < N; col += width) {
        std::set<std::vector<int>> patterns;
        for (int row = 0; row < K; ++row) {
            std::vector<int> pattern(W.begin() + row * N + col, W.begin() + row * N + std::min(N, col + width));
            if (std::any_of(pattern.begin(), pattern.end(), [](int w) { return w != 0; }))
                patterns.insert(std::move(pattern));
        }
        pattern_count += patterns.size();
    }

    // CSV output, one row per kernel
    const char* omp = std::getenv("OMP_NUM_THREADS");
    const char* blas_threads = std::getenv("OPENBLAS_NUM_THREADS");
    std::cout << "implementation,m,k,n,target_sparsity,sparsity,width,seed,nonzero_patterns,mode,inputs,input_pool_bytes,backend,omp_threads,blas_threads,preprocess_us,weight_bytes,median_us,p10_us,p90_us,speedup_blas,iterations,samples,scope,sample_us\n";
    for (std::size_t i = 0; i < suite.kernels.size(); ++i) {
        const auto& kernel = suite.kernels[i];
        auto t = timings[i];
        std::sort(t.begin(), t.end());

        std::cout << std::setprecision(10) << kernel.name << ',' << M << ',' << K << ',' << N << ',';
        std::cout << sparsity << ',' << 1.0 - static_cast<double>(nonzero) / W.size()
                  << ',' << width << ',' << seed << ',' << pattern_count << ",warm," << inputs.size()
                  << ',' << inputs.size() * M * K * sizeof(float)
                  << ',' << kernel.backend
                  << ',' << (omp ? omp : "unset") << ',' << (blas_threads ? blas_threads : "unset")
                  << ',' << kernel.preprocess_us << ',' << kernel.weight_bytes << ',' << medians[i]
                  << ',' << t[static_cast<std::size_t>((samples - 1) * 0.1)]
                  << ',' << t[static_cast<std::size_t>((samples - 1) * 0.9)]
                  << ',' << medians[blas] / medians[i] << ',' << iterations[i] << ',' << samples << ",api_call_xw,";
        for (std::size_t sample = 0; sample < timings[i].size(); ++sample)
            std::cout << (sample ? ";" : "") << timings[i][sample];
        std::cout << '\n';
    }

    (void)checksum;
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
