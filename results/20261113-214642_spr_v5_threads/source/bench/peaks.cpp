// Measures the roofline ceilings of THIS machine (for bench/roofline.py):
//   - peak fp32 ADD throughput, SIMD (NEON / AVX2 / AVX-512) and scalar           [GFLOP/s]
//   - read bandwidth from L1, L2 and DRAM                                          [GB/s]
// with T threads (each thread runs the same loop; the numbers are the totals).
//
//   ./build/peaks --threads 1          -> JSON on stdout
//
// Compiled WITHOUT -ffast-math and without auto-vectorization (CMakeLists.txt), so the scalar
// loop really stays scalar and nothing is reassociated away; the SIMD loops use intrinsics.
// Every SPR operation is an fp32 add, so "peak" = add throughput (not FMA = 2 flops).
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#define ISA "avx512"
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define ISA "avx2"
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ISA "neon"
#else
#define ISA "none"
#endif

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static thread_local volatile float g_sink;   // keeps results alive without cross-thread writes

// Optimization barrier: the value must exist in a register here, so the compiler can neither drop nor
// merge the independent accumulator chains (it otherwise collapses most of them).
#if defined(__aarch64__)
#define KEEP(v) asm volatile("" : "+w"(v))
#elif defined(__x86_64__)
#define KEEP(v) asm volatile("" : "+x"(v))
#else
#define KEEP(v) ((void)0)
#endif

// Runs f(thread_id) on T threads, `reps` times; returns the best wall time of one repetition.
template <class F>
static double best_time(int T, int reps, F f) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = 0, t1 = 0;
#pragma omp parallel num_threads(T)
        {
#pragma omp barrier
#pragma omp master
            t0 = now();
#pragma omp barrier
            f(omp_get_thread_num());
#pragma omp barrier
#pragma omp master
            t1 = now();
        }
        best = std::min(best, t1 - t0);
    }
    return best;
}

// ---------------------------------------------------------------- compute peaks
// Many independent accumulators so the add LATENCY is hidden and only THROUGHPUT counts.
static const long ITERS = 50'000'000;
static const int ACC = 16;

static double scalar_adds(long iters) {   // returns flops done (per thread)
    float a[ACC];
    for (int i = 0; i < ACC; ++i) a[i] = (float)i;
    const float c = 1e-7f;
    for (long it = 0; it < iters; ++it)
        for (int i = 0; i < ACC; ++i) { a[i] += c; KEEP(a[i]); }   // unrolled by the compiler, not vectorized (-fno-vectorize -fno-slp-vectorize)
    float s = 0;
    for (int i = 0; i < ACC; ++i) s += a[i];
    g_sink = s;
    return (double)iters * ACC;
}

static double simd_adds(long iters) {
#if defined(__AVX512F__)
    __m512 a[ACC]; const __m512 c = _mm512_set1_ps(1e-7f);
    for (int i = 0; i < ACC; ++i) a[i] = _mm512_set1_ps((float)i);
    for (long it = 0; it < iters; ++it)
        for (int i = 0; i < ACC; ++i) { a[i] = _mm512_add_ps(a[i], c); KEEP(a[i]); }
    __m512 s = a[0]; for (int i = 1; i < ACC; ++i) s = _mm512_add_ps(s, a[i]);
    g_sink = _mm512_reduce_add_ps(s);
    return (double)iters * ACC * 16;
#elif defined(__AVX2__) || defined(__AVX__)
    const int A = 12;   // 16 ymm registers: 12 accumulators + the constant
    __m256 a[A]; const __m256 c = _mm256_set1_ps(1e-7f);
    for (int i = 0; i < A; ++i) a[i] = _mm256_set1_ps((float)i);
    for (long it = 0; it < iters; ++it)
        for (int i = 0; i < A; ++i) { a[i] = _mm256_add_ps(a[i], c); KEEP(a[i]); }
    __m256 s = a[0]; for (int i = 1; i < A; ++i) s = _mm256_add_ps(s, a[i]);
    float o[8]; _mm256_storeu_ps(o, s); g_sink = o[0] + o[7];
    return (double)iters * A * 8;
#elif defined(__ARM_NEON)
    float32x4_t a[ACC]; const float32x4_t c = vdupq_n_f32(1e-7f);
    for (int i = 0; i < ACC; ++i) a[i] = vdupq_n_f32((float)i);
    for (long it = 0; it < iters; ++it)
        for (int i = 0; i < ACC; ++i) { a[i] = vaddq_f32(a[i], c); KEEP(a[i]); }
    float32x4_t s = a[0]; for (int i = 1; i < ACC; ++i) s = vaddq_f32(s, a[i]);
    g_sink = vaddvq_f32(s);
    return (double)iters * ACC * 4;
#else
    return scalar_adds(iters);
#endif
}

// ---------------------------------------------------------------- read bandwidth
// Sum of a float array with 8 independent SIMD accumulators: a pure streaming read.
static float read_sum(const float* p, size_t n) {
#if defined(__AVX512F__)
    __m512 a[8]; for (auto& v : a) v = _mm512_setzero_ps();
    for (size_t i = 0; i + 128 <= n; i += 128)
        for (int j = 0; j < 8; ++j) a[j] = _mm512_add_ps(a[j], _mm512_loadu_ps(p + i + 16 * j));
    for (int j = 1; j < 8; ++j) a[0] = _mm512_add_ps(a[0], a[j]);
    return _mm512_reduce_add_ps(a[0]);
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 a[8]; for (auto& v : a) v = _mm256_setzero_ps();
    for (size_t i = 0; i + 64 <= n; i += 64)
        for (int j = 0; j < 8; ++j) a[j] = _mm256_add_ps(a[j], _mm256_loadu_ps(p + i + 8 * j));
    for (int j = 1; j < 8; ++j) a[0] = _mm256_add_ps(a[0], a[j]);
    float o[8]; _mm256_storeu_ps(o, a[0]); return o[0] + o[7];
#elif defined(__ARM_NEON)
    float32x4_t a[8]; for (auto& v : a) v = vdupq_n_f32(0);
    for (size_t i = 0; i + 32 <= n; i += 32)
        for (int j = 0; j < 8; ++j) a[j] = vaddq_f32(a[j], vld1q_f32(p + i + 4 * j));
    for (int j = 1; j < 8; ++j) a[0] = vaddq_f32(a[0], a[j]);
    return vaddvq_f32(a[0]);
#else
    float s[8] = {}; for (size_t i = 0; i + 8 <= n; i += 8) for (int j = 0; j < 8; ++j) s[j] += p[i + j];
    return s[0] + s[7];
#endif
}

// bytes_per_thread: working set of ONE thread (private, so L1/L2 tests stay in each core's cache).
static double read_bw(int T, size_t bytes_per_thread, double target_bytes) {
    const size_t n = bytes_per_thread / 256 * 256 / sizeof(float);   // multiple of 64 B (aligned_alloc) and of the unroll
    std::vector<float*> bufs(T);
#pragma omp parallel num_threads(T)
    {   // first touch by the owning thread
        int t = omp_get_thread_num();
        bufs[t] = (float*)std::aligned_alloc(64, n * sizeof(float));
        if (!bufs[t]) std::abort();
        for (size_t i = 0; i < n; ++i) bufs[t][i] = 1.0f;
    }
    const long passes = std::max<long>(1, (long)(target_bytes / bytes_per_thread));
    double t = best_time(T, 5, [&](int tid) {
        float s = 0;
        for (long p = 0; p < passes; ++p) {
            // Each pass must read memory again, even though the buffer is unchanged.
            asm volatile("" : : "r"(bufs[tid]) : "memory");
            s += read_sum(bufs[tid], n);
        }
        g_sink = s;
    });
    for (auto b : bufs) std::free(b);
    return (double)T * passes * n * sizeof(float) / t / 1e9;
}

int main(int argc, char** argv) {
    int T = 1;
    for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], "--threads")) T = std::atoi(argv[i + 1]);

    const long iters = ITERS / 4;
    double flops = 0;   // per thread
    const double t_simd = best_time(T, 7, [&](int tid) { double f = simd_adds(iters); if (tid == 0) flops = f; });
    const double simd_gflops = T * flops / t_simd / 1e9;
    const double t_scal = best_time(T, 7, [&](int tid) { double f = scalar_adds(iters); if (tid == 0) flops = f; });
    const double scalar_gflops = T * flops / t_scal / 1e9;

    const double l1 = read_bw(T, 16 << 10, 4e9);                         // 16 KB per thread: L1 everywhere
    const double l2 = read_bw(T, 512 << 10, 4e9);                        // 512 KB per thread: L2, not L1
    const double dram = read_bw(T, ((size_t)1 << 30) / T, 4e9);          // 1 GB in total: beyond any LLC

    std::printf("{\n  \"threads\": %d,\n  \"isa\": \"%s\",\n", T, ISA);
    std::printf("  \"peak_simd_gflops\": %.2f,\n  \"peak_scalar_gflops\": %.2f,\n", simd_gflops, scalar_gflops);
    std::printf("  \"bw_l1_gbs\": %.2f,\n  \"bw_l2_gbs\": %.2f,\n  \"bw_dram_gbs\": %.2f\n}\n", l1, l2, dram);
    return 0;
}
