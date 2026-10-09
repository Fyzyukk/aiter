#pragma once
#include <cstdint>

inline bool private9030_shape_valid(int m, int n, int k, bool fixed_only) {
    constexpr int64_t input_limit = 2147483647;
    if (m <= 0 || n <= 0 || k <= 0 || m % 64 || n % 256 || k % 128 || k > 16384)
        return false;
    if (fixed_only && k != 1536)
        return false;
    // Divide before multiplying C: even signed-int dimensions can overflow 2*M*N.
    return int64_t(m) * k <= input_limit && int64_t(n) * k <= input_limit &&
           int64_t(191) * n + 256 <= input_limit / 2 &&
           int64_t(m) > (input_limit / 2) / n &&
           int64_t(m) <= (INT64_MAX / 2) / n;
}

inline bool private9030_pointer_valid(const void* a, const void* b, const void* sfa,
                                     const void* sfb, const void* c) {
    return a && b && sfa && sfb && c && reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(b) % 16 == 0 && reinterpret_cast<uintptr_t>(sfa) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(c) % 16 == 0;
}
