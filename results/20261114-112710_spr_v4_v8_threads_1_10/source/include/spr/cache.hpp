#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace spr {

enum class CacheLevel { L1, L2, LLC };

inline const char* cache_level_name(CacheLevel level) {
    switch (level) {
        case CacheLevel::L1: return "l1";
        case CacheLevel::L2: return "l2";
        case CacheLevel::LLC: return "llc";
    }
    return "unknown";
}

struct CacheInfo {
    // Conservative defaults when the OS does not expose cache information.
    size_t line = 64;
    size_t l1 = 32 * 1024;
    size_t l2 = 256 * 1024;
    size_t llc = 256 * 1024;

    size_t size(CacheLevel level) const {
        switch (level) {
            case CacheLevel::L1: return l1;
            case CacheLevel::L2: return l2;
            case CacheLevel::LLC: return llc;
        }
        return l1;
    }
};

inline const CacheInfo& cache_info() {
    static const CacheInfo info = [] {
        CacheInfo c;
#if defined(__APPLE__)
        auto read = [](const char* key, size_t fallback) {
            uint64_t value = 0;
            size_t bytes = sizeof(value);
            return sysctlbyname(key, &value, &bytes, nullptr, 0) == 0 && value > 0
                ? static_cast<size_t>(value) : fallback;
        };
        c.line = read("hw.cachelinesize", c.line);
        // On Apple Silicon, perflevel0 describes the performance cores.
        c.l1 = read("hw.perflevel0.l1dcachesize", read("hw.l1dcachesize", c.l1));
        c.l2 = read("hw.perflevel0.l2cachesize", read("hw.l2cachesize", c.l2));
        c.llc = read("hw.l3cachesize", c.l2);
#elif defined(__linux__)
        auto read = [](int key, size_t fallback) {
            const long value = sysconf(key);
            return value > 0 ? static_cast<size_t>(value) : fallback;
        };
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
        c.line = read(_SC_LEVEL1_DCACHE_LINESIZE, c.line);
#endif
#ifdef _SC_LEVEL1_DCACHE_SIZE
        c.l1 = read(_SC_LEVEL1_DCACHE_SIZE, c.l1);
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
        c.l2 = read(_SC_LEVEL2_CACHE_SIZE, c.l2);
#endif
        c.llc = c.l2;
#ifdef _SC_LEVEL3_CACHE_SIZE
        c.llc = read(_SC_LEVEL3_CACHE_SIZE, c.llc);
#endif
#endif
        auto override_bytes = [](const char* key, size_t detected) {
            const char* value = std::getenv(key);
            if (!value || *value < '0' || *value > '9') return detected;
            char* end = nullptr;
            const auto bytes = std::strtoull(value, &end, 10);
            return end != value && *end == '\0' && bytes > 0
                ? static_cast<size_t>(bytes) : detected;
        };
        c.l1 = override_bytes("SPR_L1_BYTES", c.l1);
        c.l2 = override_bytes("SPR_L2_BYTES", c.l2);
        c.llc = override_bytes("SPR_LLC_BYTES", c.llc);
        return c;
    }();
    return info;
}

}  // namespace spr
