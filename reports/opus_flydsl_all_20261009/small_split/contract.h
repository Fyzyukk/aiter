#pragma once
#include <cstdint>

struct PrivateConfig {
    int bm, bn, waves, split, local_k;
    bool narrow;
};

inline bool private_config(int id, bool candidate, int m, int n, int k, PrivateConfig& cfg) {
    constexpr int64_t limit = 2147483647;
    if (m <= 0 || m > 2048 || n <= 0 || k <= 0 || n % 128 || k % 128 || k > 16384 ||
        int64_t(m) * k > limit || int64_t(n) * k > limit || int64_t(m) * n * 2 > limit)
        return false;
    if (id == 110 || id == 111 || id == 120 || id == 121) {
        const bool n32 = id >= 120;
        cfg = {16, n32 ? 32 : 16, candidate ? (id % 10 == 0 ? 1 : 2) : 8,
               candidate ? 4 : 1, candidate ? (id % 10 == 0 ? 1 : 2) : 8, false};
    } else if (candidate && (id == 130 || id == 140)) {
        cfg = {32, id == 130 ? 32 : 64, 1, 4, 1, false};
    } else if (id == 210 || id == 211 || id == 220 || id == 221 || id == 230 || id == 231) {
        const int bm = id >= 230 ? 96 : id >= 220 ? 64 : 48;
        cfg = {bm, bm == 48 ? 64 : 128, 4, id % 10 == 0 ? 1 : 2, 1, true};
        if (!candidate) return false;
    } else if (!candidate && id >= 9040 && id <= 9054 && id != 9048 && id != 9049 && id != 9050) {
        switch (id) {
        case 9040: cfg = {16,32,1,1,1,false}; break;
        case 9041: cfg = {16,16,8,1,8,false}; break;
        case 9042: cfg = {32,32,4,1,4,false};
            if (k == 7168) cfg = ((m + 15) / 16) * ((n + 47) / 48) <= 256
                ? PrivateConfig{16,48,4,1,4,false} : PrivateConfig{32,48,4,1,4,false}; break;
        case 9051: cfg = {16,32,4,1,4,false}; break;
        case 9052: cfg = {16,32,8,1,8,false}; break;
        case 9053: cfg = {32,32,k == 7168 ? 4 : 8,1,k == 7168 ? 4 : 8,false}; break;
        case 9054: cfg = {32,64,4,1,4,false}; break;
        default: return false;
        }
    } else if (!candidate && id >= 9060 && id <= 9063) {
        cfg = {id == 9061 ? 96 : 80,128,id == 9061 ? 8 : 4,id == 9062 ? 2 : id == 9063 ? 4 : 1,1,true};
        if (id == 9062 && (m + 95) / 96 < (m + 79) / 80)
            cfg = {96,128,8,2,1,true};
        if (id == 9063) {
            if (k >= 8192 && m <= 48) cfg = {48,128,4,4,1,true};
            else if (k >= 8192 && m > 80 && m <= 96) cfg = {96,128,4,4,1,true};
            else if (k >= 8192 && m > 96 && m <= 112) cfg = {112,128,4,4,1,true};
            else if (k >= 8192 && m > 112 && m <= 128) cfg = {128,128,4,4,1,true};
            else if (!(k >= 8192 && m <= 128) && ((m + 79) / 80) * (n / 128) * 4 > 256)
                cfg = {96,128,8,4,1,true};
        }
    } else return false;
    if (m <= 0 || n <= 0 || k <= 0 || n % 128 || k % 128 || k > 16384)
        return false;
    if (cfg.narrow ? (m > 2048) : (m > 512))
        return false;
    return int64_t(m) * k <= limit && int64_t(n) * k <= limit &&
           int64_t(m) * n * (cfg.split > 1 ? 4 : 2) <= limit;
}

inline uint64_t private_workspace_bytes(const PrivateConfig& cfg, int m, int n) {
    return cfg.split > 1 ? uint64_t(cfg.split) * m * n * sizeof(float) : 0;
}

inline bool private_ranges_overlap(const void* first, uint64_t first_bytes,
                                   const void* second, uint64_t second_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(first), b = reinterpret_cast<uintptr_t>(second);
    return a >= b ? uint64_t(a - b) < second_bytes : uint64_t(b - a) < first_bytes;
}

inline bool private_pointers_valid(const PrivateConfig& cfg, const void* a, const void* b,
    const void* sa, const void* sb, const void* c, void* workspace, uint64_t capacity,
    int m, int n, int k) {
    if (!a || !b || !sa || !sb || !c || reinterpret_cast<uintptr_t>(a) % 16 ||
        reinterpret_cast<uintptr_t>(b) % 16 || reinterpret_cast<uintptr_t>(c) % 16)
        return false;
    // Native scales are loaded as bytes in the register path. The fine loader
    // also uses vector SFA loads and therefore requires its parent alignment.
    if (cfg.narrow && reinterpret_cast<uintptr_t>(sa) % 16) return false;
    const void* inputs[] = {a, b, sa, sb, c};
    const uint64_t sizes[] = {uint64_t(m) * k, uint64_t(n) * k,
        uint64_t(m) * (k / 128), uint64_t(n / 128) * (k / 128), uint64_t(m) * n * 2};
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (private_ranges_overlap(inputs[i], sizes[i], inputs[j], sizes[j])) return false;
    const uint64_t required = private_workspace_bytes(cfg, m, n);
    if (!required) return workspace == nullptr && capacity == 0;
    if (!workspace || reinterpret_cast<uintptr_t>(workspace) % 16 || capacity < required) return false;
    for (int i = 0; i < 5; ++i)
        if (private_ranges_overlap(workspace, required, inputs[i], sizes[i])) return false;
    return true;
}
