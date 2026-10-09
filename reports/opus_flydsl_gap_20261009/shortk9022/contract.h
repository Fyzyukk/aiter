#pragma once
#include <cstdint>

// Native E8M0 scales, contiguous FP8 A/B, standard (16,16) preshuffled B,
// column-major logical [M,K/128] SFA, row-major [N/128,K/128] SFB, BF16 C.
// The public 9022 parent permits M%16, N%128 and signed-int tensor byte extents.
inline bool shortk9022_shape_valid(int m, int n, int k, bool fixed_only) {
    constexpr int64_t limit = 2147483647;
    if (m <= 0 || n <= 0 || k <= 0 || m % 16 || n % 128 || k % 128 || k > 16384)
        return false;
    if (fixed_only && k != 384 && k != 768)
        return false;
    return int64_t(m) * k <= limit && int64_t(n) * k <= limit && int64_t(m) * n * 2 <= limit;
}

inline bool shortk9022_pointer_valid(const void* a, const void* b, const void* sfa,
                                    const void* sfb, const void* c) {
    return a && b && sfa && sfb && c && reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(b) % 16 == 0 && reinterpret_cast<uintptr_t>(sfa) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(c) % 16 == 0;
}
