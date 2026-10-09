#pragma once
#include <cstdint>

// Raw FP8 A/B, native E8M0 SFA [M,K/128] (M contiguous), SFB
// [N/128,K/128], standard N16/K16 B preshuffle, contiguous BF16 C.
// Actual family IDs are explicit: this private entry never reroutes a parent.
inline bool opus_private_small_shape_valid(int actual, int m, int n, int k) {
    const int max_m = actual == 9047 || actual == 9049 ? 512 : 2048;
    if (actual != 9043 && actual != 9044 && actual != 9045 && actual != 9046 &&
        actual != 9047 && actual != 9049 && actual != 9055 && actual != 9056)
        return false;
    if (m <= 0 || m > max_m || n <= 0 || n % 128 || k <= 0 || k % 128 || k > 16384)
        return false;
    constexpr int64_t limit = 2147483647;
    return int64_t(m) * k <= limit && int64_t(n) * k <= limit && int64_t(m) * n * 2 <= limit;
}
inline bool opus_private_small_pointer_valid(const void* a, const void* b, const void* sfa,
                                            const void* sfb, const void* c) {
    return a && b && sfa && sfb && c && reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(b) % 16 == 0 && reinterpret_cast<uintptr_t>(c) % 8 == 0;
}
