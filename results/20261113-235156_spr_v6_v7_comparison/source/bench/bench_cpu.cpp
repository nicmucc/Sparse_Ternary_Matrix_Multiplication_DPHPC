// CPU benchmark driver. One process = one shape, a list of sparsities, a list of kernels and
// a list of k values. All (kernel, k) configurations are timed INTERLEAVED (round-robin with a
// rotating start) so machine drift hits every kernel equally; the median is reported.
//
//   ./build/bench_cpu --M 256 --K 1024 --N 4096 --s 0.5,0.7,0.9 --k 4,6 --threads 6 \
//       --kernels ssr.ssr_inference_float_vec_L,tcsc.GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP \
//       --rounds 21 --tag S1 --out results/s1.csv
//
//   --kernels takes substrings ("tcsc.", "GroupMin_32x", ...) or exact names prefixed with '='
//   ("=tcsc.GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor"). Default: all.
//   A pattern can carry its own k list, which then replaces --k for it: "=rsr.rsr_pp_inference@6",
//   "spr.@4/6/8".
//   --verify PATTERNS   which kernels to check against the reference before timing (same pattern
//                       syntax, default: all; "none" = no check). Only the first --ref-rows rows of Y
//                       (default 16) are checked, to keep the reference cheap. Wrong results are
//                       written with ok=0 and not timed; unchecked configs have max_err = -1.
//   --budget-ms B       cap the timed runs per config at ~B ms (never fewer than 3 rounds), so slow
//                       configurations do not dominate the wall time. 0 = always --rounds.
//   --prep-rounds R     report median preparation time over R freshly constructed kernels.
//                       R > 1 also performs one untimed preparation to warm the runtime.
//   --sparsity-pattern  iid (default), or column_clustered: floor(s*N) leading
//                       output columns are zero and the remaining columns are dense +/-1.
//
// Unsupported shapes are skipped silently.
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>

#include "spr/harness.hpp"

using namespace spr;

namespace {

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) v.push_back(x);
    return v;
}
std::string clean(std::string s) {
    for (auto& c : s) if (c == ',' || c == '\n') c = ' ';
    return s;
}

// Does kernel `n` match pattern `p` ("=exact" or substring, optional "@k1/k2" suffix ignored here)?
bool matches(const std::string& n, std::string p) {
    p = p.substr(0, p.find('@'));
    return p[0] == '=' ? n == p.substr(1) : n.find(p) != std::string::npos;
}

struct Entry {
    std::unique_ptr<Kernel> kern;
    std::unique_ptr<Bound> bound;
    Params p;
    double prep_ms = 0, err = 0;
    bool ok = false;
    std::vector<double> times;
};

}  // namespace

int main(int argc, char** argv) {
    int M = 256, K = 1024, N = 4096, threads = 1, rounds = 21, warmup = 3, prep_rounds = 1;
    std::string kernels, s_list = "0.5", k_list = "4", tag = "run", out = "bench_cpu.csv", verify = "";
    std::string sparsity_pattern = "iid";
    int ref_rows = 16;
    double budget_ms = 0;
    uint64_t seed = 42;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--M") M = std::stoi(v);
        else if (a == "--K") K = std::stoi(v);
        else if (a == "--N") N = std::stoi(v);
        else if (a == "--s") s_list = v;
        else if (a == "--sparsity-pattern") sparsity_pattern = v;
        else if (a == "--k") k_list = v;
        else if (a == "--kernels") kernels = v;
        else if (a == "--threads") threads = std::stoi(v);
        else if (a == "--rounds") rounds = std::stoi(v);
        else if (a == "--warmup") warmup = std::stoi(v);
        else if (a == "--prep-rounds") prep_rounds = std::stoi(v);
        else if (a == "--tag") tag = v;
        else if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--verify") verify = v;
        else if (a == "--ref-rows") ref_rows = std::stoi(v);
        else if (a == "--budget-ms") budget_ms = std::stod(v);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    if (threads < 1 || rounds < 1 || prep_rounds < 1) {
        std::fprintf(stderr, "threads, rounds and prep-rounds must be positive\n");
        return 2;
    }

    // (kernel name, its k list): the first matching pattern decides the k list
    std::vector<int> ks_default;
    for (auto& x : split(k_list)) ks_default.push_back(std::stoi(x));
    std::vector<std::pair<std::string, std::vector<int>>> configs;
    auto pats = split(kernels);
    for (auto& n : kernel_names()) {
        if (pats.empty()) { configs.push_back({n, ks_default}); continue; }
        for (auto& p : pats) {
            if (!matches(n, p)) continue;
            std::vector<int> ks = ks_default;
            if (p.find('@') != std::string::npos) {
                ks.clear();
                std::stringstream ss(p.substr(p.find('@') + 1));
                for (std::string x; std::getline(ss, x, '/');) ks.push_back(std::stoi(x));
            }
            configs.push_back({n, ks});
            break;
        }
    }
    auto pats_v = split(verify);
    auto check = [&](const std::string& n) {
        if (pats_v.empty()) return true;
        if (verify == "none") return false;
        for (auto& p : pats_v) if (matches(n, p)) return true;
        return false;
    };
    bool any_check = false;
    for (auto& c : configs) any_check = any_check || check(c.first);
    set_threads(threads);

    std::ifstream probe(out);
    const bool new_file = !probe.good();
    probe.close();
    std::ofstream csv(out, std::ios::app);
    if (!csv) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    if (new_file)
        csv << "tag,machine,compiler,threads,M,K,N,sparsity,kernel,param,median_ms,min_ms,rounds,prep_ms,bytes,bits_per_weight,max_err,ok,adds_per_row\n";

    for (auto& s_str : split(s_list)) {
        const double s = std::stod(s_str);
        Problem P = make_problem(M, K, N, s, seed, any_check ? ref_rows : 0, sparsity_pattern);

        // ---- build + verify every configuration
        std::vector<Entry> E;
        for (auto& [name, ks] : configs) {
            for (int k : ks) {
                Entry e;
                e.kern = make_kernel(name);
                if (!e.kern->uses_k() && k != ks[0]) break;
                e.p.k = k; e.p.threads = threads;
                if (!e.kern->supports(M, K, N, e.p)) continue;
                try {
                    if (prep_rounds > 1) {
                        auto warm = make_kernel(name);
                        warm->prepare(P.W.data(), K, N, e.p);
                    }
                    std::vector<double> prep_times;
                    for (int r = 0; r < prep_rounds; ++r) {
                        // Construction and destruction are outside the preparation timer.
                        auto prepared = make_kernel(name);
                        double t0 = now_ms();
                        prepared->prepare(P.W.data(), K, N, e.p);
                        prep_times.push_back(now_ms() - t0);
                        e.kern = std::move(prepared);
                    }
                    std::sort(prep_times.begin(), prep_times.end());
                    e.prep_ms = prep_times[prep_times.size() / 2];
                } catch (const NotImplemented&) {
                    break;
                }
                e.bound = std::make_unique<Bound>(e.kern.get(), P);
                e.err = check(name) ? e.bound->verify(P) : -1;
                e.ok = e.err <= tolerance(K);
                if (!e.ok) std::fprintf(stderr, "WRONG RESULT %s %s err=%.3g (not timed)\n", name.c_str(), e.p.str().c_str(), e.err);
                E.push_back(std::move(e));
            }
        }

        // ---- interleaved timing
        // warm-up; the last warm-up run sets how many timed rounds this config gets under the budget
        std::vector<int> target(E.size(), rounds);
        for (size_t i = 0; i < E.size(); ++i) {
            if (!E[i].ok) continue;
            double t = 0;
            for (int w = 0; w < warmup; ++w) { double t0 = now_ms(); E[i].bound->run(); t = now_ms() - t0; }
            if (budget_ms > 0 && warmup > 0) target[i] = std::max(3, std::min(rounds, (int)std::ceil(budget_ms / std::max(t, 1e-3))));
        }
        for (int r = 0; r < rounds; ++r)
            for (size_t j = 0; j < E.size(); ++j) {
                const size_t i = (j + r) % E.size();
                auto& e = E[i];
                if (!e.ok || (int)e.times.size() >= target[i]) continue;
                double t0 = now_ms();
                e.bound->run();
                e.times.push_back(now_ms() - t0);
            }

        // ---- report
        for (auto& e : E) {
            double med = -1, mn = -1;
            if (!e.times.empty()) {
                std::sort(e.times.begin(), e.times.end());
                med = e.times[e.times.size() / 2];
                mn = e.times.front();
            }
            const size_t b = e.kern->bytes();
            csv << tag << ',' << clean(machine_name()) << ',' << clean(compiler_name()) << ',' << threads << ','
                << M << ',' << K << ',' << N << ',' << s << ',' << e.kern->name() << ',' << (e.kern->uses_k() ? e.p.str() : std::string("-")) << ','
                << med << ',' << mn << ',' << e.times.size() << ',' << e.prep_ms << ',' << b << ','
                << 8.0 * b / ((double)K * N) << ',' << e.err << ',' << e.ok << ',' << e.kern->adds_per_row() << '\n';
            std::printf("s=%.2f %-70s %-5s %10.3f ms  prep %9.1f ms %s\n", s, e.kern->name().c_str(),
                        e.kern->uses_k() ? e.p.str().c_str() : "", med, e.prep_ms, e.ok ? "" : "WRONG");
        }
        csv.flush();
    }
    return 0;
}
