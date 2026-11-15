// Cache sizes of the machine, detected at runtime (once), so kernels can size their tiles
// for any cache level without hard-coded numbers.
//
//   const CacheInfo& c = cache_info();
//   size_t bytes = c.size(CacheLevel::L1);     // or L2, LLC
//
// Detection:
//   macOS : sysctl hw.perflevel0.* (performance cores; falls back to hw.* on single-cluster chips)
//   Linux : sysconf(_SC_LEVEL*_CACHE_SIZE), then /sys/devices/system/cpu/cpu0/cache/index*/
// Overrides (bytes, e.g. to model another machine or a share of a shared cache):
//   SPR_L1_BYTES, SPR_L2_BYTES, SPR_LLC_BYTES, SPR_LINE_BYTES
// Missing levels: LLC falls back to L2 (e.g. Apple M1 reports no L3), L2 to L1.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace spr {

enum class CacheLevel { L1 = 1, L2 = 2, LLC = 3 };

inline const char* cache_level_name(CacheLevel l) {
    switch (l) {
        case CacheLevel::L1: return "l1";
        case CacheLevel::L2: return "l2";
        default:             return "llc";
    }
}

struct CacheInfo {
    size_t l1d = 0;     // L1 data cache of one core
    size_t l2 = 0;      // L2 (per core or per cluster, as the OS reports it)
    size_t llc = 0;     // last-level cache (L3, or L2 if there is no L3)
    size_t line = 0;    // cache line size

    size_t size(CacheLevel l) const {
        switch (l) {
            case CacheLevel::L1: return l1d;
            case CacheLevel::L2: return l2;
            default:             return llc;
        }
    }
};

namespace detail {

inline size_t env_bytes(const char* name) {
    const char* v = std::getenv(name);
    return v ? (size_t)std::strtoull(v, nullptr, 10) : 0;
}

#if defined(__APPLE__)
inline size_t sysctl_bytes(const char* name) {
    long long v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0 || v <= 0) return 0;
    return (size_t)v;
}
#elif defined(__linux__)
// /sys/devices/system/cpu/cpu0/cache/indexN/{level,type,size}; size is like "48K" or "2M"
inline size_t sysfs_cache_bytes(int level) {
    for (int i = 0; i < 8; ++i) {
        const std::string dir = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        auto read = [&](const char* f) {
            std::string s;
            if (FILE* fp = std::fopen((dir + f).c_str(), "r")) {
                char buf[64];
                if (std::fgets(buf, sizeof buf, fp)) s = buf;
                std::fclose(fp);
            }
            while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
            return s;
        };
        const std::string lvl = read("level");
        if (lvl.empty()) break;
        if (std::atoi(lvl.c_str()) != level) continue;
        const std::string type = read("type");
        if (type == "Instruction") continue;   // L1i
        const std::string sz = read("size");
        if (sz.empty()) continue;
        size_t v = (size_t)std::strtoull(sz.c_str(), nullptr, 10);
        if (sz.back() == 'K') v <<= 10;
        else if (sz.back() == 'M') v <<= 20;
        return v;
    }
    return 0;
}
inline size_t sysconf_bytes(int name) {
    const long v = sysconf(name);
    return v > 0 ? (size_t)v : 0;
}
#endif

inline CacheInfo detect_caches() {
    CacheInfo c;
#if defined(__APPLE__)
    c.l1d = sysctl_bytes("hw.perflevel0.l1dcachesize");
    c.l2 = sysctl_bytes("hw.perflevel0.l2cachesize");
    if (!c.l1d) c.l1d = sysctl_bytes("hw.l1dcachesize");
    if (!c.l2) c.l2 = sysctl_bytes("hw.l2cachesize");
    c.llc = sysctl_bytes("hw.l3cachesize");
    c.line = sysctl_bytes("hw.cachelinesize");
#elif defined(__linux__)
#ifdef _SC_LEVEL1_DCACHE_SIZE
    c.l1d = sysconf_bytes(_SC_LEVEL1_DCACHE_SIZE);
    c.l2 = sysconf_bytes(_SC_LEVEL2_CACHE_SIZE);
    c.llc = sysconf_bytes(_SC_LEVEL3_CACHE_SIZE);
    c.line = sysconf_bytes(_SC_LEVEL1_DCACHE_LINESIZE);
#endif
    if (!c.l1d) c.l1d = sysfs_cache_bytes(1);
    if (!c.l2) c.l2 = sysfs_cache_bytes(2);
    if (!c.llc) c.llc = sysfs_cache_bytes(3);
#endif
    // environment overrides
    if (size_t v = env_bytes("SPR_L1_BYTES")) c.l1d = v;
    if (size_t v = env_bytes("SPR_L2_BYTES")) c.l2 = v;
    if (size_t v = env_bytes("SPR_LLC_BYTES")) c.llc = v;
    if (size_t v = env_bytes("SPR_LINE_BYTES")) c.line = v;
    // conservative fallbacks for anything still unknown
    if (!c.l1d) c.l1d = 32 * 1024;
    if (!c.l2) c.l2 = c.l1d;
    if (!c.llc) c.llc = c.l2;
    if (!c.line) c.line = 64;
    return c;
}

}  // namespace detail

// Detected once per process.
inline const CacheInfo& cache_info() {
    static const CacheInfo info = detail::detect_caches();
    return info;
}

}  // namespace spr
