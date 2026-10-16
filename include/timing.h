#ifndef TIMING_H
#define TIMING_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
    #include <x86intrin.h>
#endif

typedef struct {
    uint64_t os_start_nsec;
    uint64_t os_stop_nsec;
    uint64_t hw_start_tick;
    uint64_t hw_stop_tick;
} timer_context_t;


/* ============================================================================
 * Compiler barrier
 * Prevents compiler reordering across timing boundaries.
 * ============================================================================ */
static inline void _compiler_barrier(void) {
    __asm__ volatile("" ::: "memory");
}


/* ============================================================================
 * OS timer
 * ============================================================================ */
static inline uint64_t _get_os_nsec(void) {
    struct timespec ts;

#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}


/* ============================================================================
 * Hardware timer: start
 *
 * x86:
 *   LFENCE
 *   RDTSC
 *   LFENCE
 *
 * AArch64:
 *   keep previous M1 behavior
 * ============================================================================ */
static inline uint64_t _get_hw_ticks_start(void) {
#if defined(__aarch64__)

    uint64_t val;
    __asm__ volatile("isb" : : : "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;

#elif defined(__x86_64__) || defined(__i386__)

    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;

#else

    return 0;

#endif
}


/* ============================================================================
 * Hardware timer: stop
 *
 * x86:
 *   RDTSCP
 *   LFENCE
 *
 * RDTSCP waits for previous instructions before reading the timestamp.
 * LFENCE prevents following instructions from being moved before the timestamp.
 *
 * AArch64:
 *   keep previous M1 behavior
 * ============================================================================ */
static inline uint64_t _get_hw_ticks_stop(void) {
#if defined(__aarch64__)

    uint64_t val;
    __asm__ volatile("isb" : : : "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;

#elif defined(__x86_64__) || defined(__i386__)

    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;

#else

    return 0;

#endif
}


/* ============================================================================
 * Timer controls
 * ============================================================================ */
static inline void start_timer(timer_context_t* ctx) {
    _compiler_barrier();

    ctx->os_start_nsec = _get_os_nsec();
    ctx->hw_start_tick = _get_hw_ticks_start();

    _compiler_barrier();
}

static inline void stop_timer(timer_context_t* ctx) {
    _compiler_barrier();

    ctx->hw_stop_tick = _get_hw_ticks_stop();
    ctx->os_stop_nsec = _get_os_nsec();

    _compiler_barrier();
}


/* ============================================================================
 * Elapsed time calculators
 * ============================================================================ */
static inline double get_elapsed_os_sec(const timer_context_t* ctx) {
    return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
}

static inline double get_elapsed_os_nsec(const timer_context_t* ctx) {
    return (double)(ctx->os_stop_nsec - ctx->os_start_nsec);
}

static inline double get_elapsed_hw_cycles(
    const timer_context_t* ctx,
    double tick_freq_mhz,
    double cpu_freq_ghz
) {
    uint64_t elapsed_ticks = ctx->hw_stop_tick - ctx->hw_start_tick;

#if defined(__x86_64__) || defined(__i386__)

    /*
     * On x86, this returns serialized TSC ticks.
     * For this project, keep calling them cycles because the benchmarking
     * and plots already use that terminology.
     */
    (void)tick_freq_mhz;
    (void)cpu_freq_ghz;
    return (double)elapsed_ticks;

#else

    /*
     * Keep previous M1/AArch64 behavior:
     * convert architectural counter ticks to estimated CPU cycles.
     */
    return (double)elapsed_ticks * (cpu_freq_ghz * 1000.0) / tick_freq_mhz;

#endif
}

static inline double get_elapsed_os_cycles(
    const timer_context_t* ctx,
    double cpu_freq_ghz
) {
    return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) * cpu_freq_ghz;
}

#endif

