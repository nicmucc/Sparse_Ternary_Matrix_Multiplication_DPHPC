// Correctness test for EVERY registered kernel against the double-precision reference.
//
//   ./build/test_correctness              # all kernels
//   ./build/test_correctness ssr. rsr.    # only kernels whose name contains one of the arguments
//
// Each kernel is run on every (shape, sparsity, k) it supports(); shapes it does not support
// (unrolling / grouping / GEMV restrictions) are skipped. Shapes include ragged sizes and
// edge sparsities (0 % and 100 %) -- that's where bugs hide.
#include <cstdio>
#include <utility>

#include "spr/harness.hpp"

using namespace spr;

// Small exact examples exercise the sparse tree (shifting / merging) and scratch reset, for every spr.* kernel.
static std::pair<int, int> check_spr_examples(Kernel& kern) {
    struct Example { const char* name; Problem problem; };
    std::vector<Example> examples = {
        {"binary example", {1, 6, 3,
            {2, 4, 6, 8, 10, 12},
            {0,0,0, 0,0,1, 1,0,0, 1,0,1, 1,0,1, 1,1,0},
            {36, 12, 22}, 1}},
        {"ternary siblings", {2, 7, 3,
            {2,4,6,8,10,12,14, -1,2,-3,4,-5,6,-7},
            {0,0,0, 1,0,0, 1,0,1, 1,0,-1, 1,-1,0, 1,-1,0, -1,0,0},
            {26,-22,-2, 11,-1,-7}, 2}},
        {"empty block and ragged tail", {2, 3, 7,
            {2,-3,5, -1,4,7},
            {1,0,-1,0,0,0,-1, 0,1,1,0,0,0,1, -1,0,0,0,0,0,1},
            {-3,-3,-5,0,0,0,0, -8,4,5,0,0,0,12}, 2}},
        {"all zero", {2, 2, 3,
            {2,-3,4,5}, {0,0,0,0,0,0}, {0,0,0,0,0,0}, 2}},
    };

    // Exercise all 16 digits of the largest supported block, plus a short tail.
    const Problem& ternary = examples[1].problem;
    Problem wide = ternary;
    wide.N = 17;
    wide.W.resize((size_t)wide.K * wide.N);
    wide.Yref.resize((size_t)wide.M * wide.N);
    for (int r = 0; r < wide.K; ++r)
        for (int c = 0; c < wide.N; ++c)
            wide.W[(size_t)r * wide.N + c] = ternary.W[r * 3 + c % 3];
    for (int m = 0; m < wide.M; ++m)
        for (int c = 0; c < wide.N; ++c)
            wide.Yref[(size_t)m * wide.N + c] = ternary.Yref[m * 3 + c % 3];
    examples.push_back({"wide patterns", std::move(wide)});

    int pass = 0, fail = 0;
    for (const auto& example : examples) {
        const Problem& P = example.problem;
        for (int k : {1, 3, 16}) {
            Params p; p.k = k;
            kern.prepare(P.W.data(), P.K, P.N, p);
            Bound b(&kern, P);
            bool ok = true;
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::fill(b.Y.p, b.Y.p + b.Y.n, 123.f);
                b.run();
                kern.finish(b.io);
                // Y is in the kernel's layout(); Yref is row-major
                const bool row = kern.layout() == Layout::RowMajor;
                for (int m = 0; m < P.M; ++m)
                    for (int n = 0; n < P.N; ++n)
                        if (b.Y.p[row ? (size_t)m * P.N + n : (size_t)n * P.M + m] != P.Yref[(size_t)m * P.N + n]) ok = false;
            }
            // For the user's example: 4 patterns, 5 offsets, 5 rowids,
            // and 11 additions (5 segment + 4 output + 2 merge). Format-specific: spr.naive only.
            if (kern.name() == "spr.naive" && std::string(example.name) == "binary example" && k == 3) {
                ok = ok && kern.bytes() == 4 * sizeof(uint32_t) + 12 * sizeof(int);
                ok = ok && kern.adds_per_row() == 11;
            }
            if (ok) ++pass;
            else {
                ++fail;
                std::printf("    wrong: %s k=%d\n", example.name, k);
            }
        }
    }
    return {pass, fail};
}

int main(int argc, char** argv) {
    std::vector<std::string> filters(argv + 1, argv + argc);
    auto selected = [&](const std::string& name) {
        if (filters.empty()) return true;
        for (auto& f : filters) if (name.find(f) != std::string::npos) return true;
        return false;
    };

    const int SEED = 1234;
    const int THREADS = 4;
    struct Shape { int M, K, N; };
    const Shape shapes[] = {{1, 256, 128}, {5, 100, 37}, {16, 256, 128}, {64, 1024, 512}, {64, 2050, 264}, {128, 512, 256}};
    const double sparsities[] = {0.0, 0.3, 0.5, 0.8, 0.95, 1.0};
    const int ks[] = {2, 3, 4, 6, 8};

    std::vector<Problem> problems;   // built once, shared by all kernels
    for (const auto& sh : shapes)
        for (double s : sparsities) problems.push_back(make_problem(sh.M, sh.K, sh.N, s, SEED));

    set_threads(THREADS);
    int n_pass = 0, n_fail = 0, n_kernels_failed = 0;
    for (const auto& name : kernel_names()) {
        if (!selected(name)) continue;
        auto kern = make_kernel(name);
        int pass = 0, fail = 0;
        if (name.rfind("spr.", 0) == 0) {   // exact sparse-tree examples: every SPR kernel
            const auto examples = check_spr_examples(*kern);
            pass += examples.first;
            fail += examples.second;
        }
        bool not_impl = false;
        for (const auto& P : problems) {
            for (int k : ks) {
                if (!kern->uses_k() && k != ks[0]) continue;
                Params p; p.k = k; p.threads = THREADS;
                if (!kern->supports(P.M, P.K, P.N, p)) continue;
                try {
                    kern->prepare(P.W.data(), P.K, P.N, p);
                    Bound b(kern.get(), P);
                    double e = b.verify(P);
                    if (e <= tolerance(P.K)) ++pass;
                    else if (++fail <= 3)
                        std::printf("    wrong: M=%d K=%d N=%d %s  max_err=%.3g\n", P.M, P.K, P.N, p.str().c_str(), e);
                } catch (const NotImplemented&) {
                    not_impl = true;
                    break;
                }
            }
            if (not_impl) break;
        }
        const char* status = not_impl ? "TODO" : fail ? "FAIL" : pass ? "PASS" : "SKIP";
        std::printf("%s %-62s %4d passed %4d failed\n", status, name.c_str(), pass, fail);
        n_pass += pass; n_fail += fail; n_kernels_failed += fail > 0;
    }
    std::printf("\n%d cases passed, %d failed (%d kernels with failures)\n", n_pass, n_fail, n_kernels_failed);
    return n_fail ? 1 : 0;
}
