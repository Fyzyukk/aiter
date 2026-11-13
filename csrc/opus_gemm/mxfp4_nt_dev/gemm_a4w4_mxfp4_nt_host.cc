// Host driver for the MXFP4 x MXFP4 (NT, 1x32 UE8M0 scales) GEMM.
//   A : [batch, M, K] fp4 E2M1, K-contiguous, 2 per byte (even k in low nibble)
//   B : [batch, N, K] fp4 E2M1, K-contiguous (NT), same packing
//   SFA: [batch, M, K/32] UE8M0, SFB: [batch, N, K/32] UE8M0 (native 1x32)
//   C : [batch, M, N] bf16 (fp32 accumulation, round-to-nearest-even on store)
// 4-wave, tile 256x256x256: the kernel reads the native row-major scales directly.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <omp.h>

#include "opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh"

template<class Traits>
__global__ void gemm_a4w4_mxfp4_nt_kernel(opus_gemm_mxfp4_scale_kargs_gfx950 kargs);

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

template<int BM, int BN> using TileTraits = opus_mxfp4_nt_traits<BM, BN>;
using fp32_t = float;
using e8m0_t = uint8_t;

__host__ __device__ inline float e2m1_to_f32(unsigned code) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[code & 7];
    return (code & 8) ? -v : v;
}

inline double e8m0_to_f64(e8m0_t e) { return std::ldexp(1.0, static_cast<int>(e) - 127); }

void rand_bytes(uint8_t* ptr, std::size_t size, unsigned seed) {
    #pragma omp parallel
    {
        std::mt19937 gen(seed * 7919u + omp_get_thread_num());
        std::uniform_int_distribution<int> dis(0, 255);
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) ptr[i] = static_cast<uint8_t>(dis(gen));
    }
}

void rand_scale_e8m0(e8m0_t* ptr, std::size_t size, unsigned seed, int lo = 124, int hi = 130) {
    #pragma omp parallel
    {
        std::mt19937 gen(seed * 104729u + omp_get_thread_num());
        std::uniform_int_distribution<int> dis(lo, hi);
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) ptr[i] = static_cast<e8m0_t>(dis(gen));
    }
}

// Naive GPU reference: one thread per C element, fp64 accumulate.  Products of
// E2M1 values and power-of-two scales are exact, so this is an exact reference
// up to the final fp32 rounding.  Also emits sum(|term|) for the tolerance.
__global__ void gemm_ref_gpu(const uint8_t* a, const uint8_t* b, const e8m0_t* sfa, const e8m0_t* sfb,
                             float* c, float* mag, int m, int n, int k) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    const int kb = k / 2, kg = k / 32;
    const uint8_t* ar = a + static_cast<size_t>(i) * kb;
    const uint8_t* br = b + static_cast<size_t>(j) * kb;
    double sum = 0.0, sabs = 0.0;
    for (int g = 0; g < kg; ++g) {
        float part = 0.0f, pabs = 0.0f; // exact: <= 32 products of E2M1 values
        for (int p = 0; p < 16; ++p) {
            const unsigned x = ar[g * 16 + p], y = br[g * 16 + p];
            const float t0 = e2m1_to_f32(x & 15) * e2m1_to_f32(y & 15);
            const float t1 = e2m1_to_f32(x >> 4) * e2m1_to_f32(y >> 4);
            part += t0 + t1;
            pabs += fabsf(t0) + fabsf(t1);
        }
        const double s = ldexp(1.0, static_cast<int>(sfa[static_cast<size_t>(i) * kg + g]) +
                                    static_cast<int>(sfb[static_cast<size_t>(j) * kg + g]) - 254);
        sum += part * s;
        sabs += pabs * s;
    }
    c[static_cast<size_t>(i) * n + j] = static_cast<float>(sum);
    mag[static_cast<size_t>(i) * n + j] = static_cast<float>(sabs);
}

// CPU reference (small shapes only) to cross-check the GPU reference itself.
void gemm_ref_cpu(const uint8_t* a, const uint8_t* b, const e8m0_t* sfa, const e8m0_t* sfb,
                  float* c, int m, int n, int k) {
    const int kb = k / 2, kg = k / 32;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int p = 0; p < k; ++p) {
            const unsigned x = a[static_cast<size_t>(i) * kb + p / 2];
            const unsigned y = b[static_cast<size_t>(j) * kb + p / 2];
            const unsigned xa = (p & 1) ? (x >> 4) : (x & 15);
            const unsigned yb = (p & 1) ? (y >> 4) : (y & 15);
            sum += static_cast<double>(e2m1_to_f32(xa)) * e2m1_to_f32(yb)
                 * e8m0_to_f64(sfa[static_cast<size_t>(i) * kg + p / 32])
                 * e8m0_to_f64(sfb[static_cast<size_t>(j) * kg + p / 32]);
        }
        c[static_cast<size_t>(i) * n + j] = static_cast<float>(sum);
    }
}

struct cmp_stats { long errors; double max_diff; double max_ratio; double calc_diff; };

cmp_stats compare(const float* ref, const float* got, const float* mag, size_t cnt,
                  float rel_mag = 5e-5f, float abs_floor = 1e-4f, bool print = true, double out_rel = 0.0) {
    long errors = 0; double max_diff = 0, max_ratio = 0, xy = 0, xx_yy = 0;
    #pragma omp parallel for reduction(+:errors,xy,xx_yy) reduction(max:max_diff,max_ratio)
    for (size_t i = 0; i < cnt; ++i) {
        const double r = ref[i], g = got[i];
        const double d = std::fabs(r - g);
        const double tol = abs_floor + rel_mag * (mag ? mag[i] : std::fabs(r)) + out_rel * std::fabs(r);
        max_diff = std::max(max_diff, d);
        max_ratio = std::max(max_ratio, d / tol);
        if (!(d <= tol)) ++errors;
        xy += r * g; xx_yy += r * r + g * g;
    }
    const double cd = xx_yy > 0 ? 1.0 - 2.0 * xy / xx_yy : 0.0;
    if (print && errors) {
        int shown = 0;
        for (size_t i = 0; i < cnt && shown < 8; ++i) {
            const double tol = abs_floor + rel_mag * (mag ? mag[i] : std::fabs(ref[i])) + out_rel * std::fabs(ref[i]);
            if (!(std::fabs(ref[i] - got[i]) <= tol)) {
                std::printf("  mismatch at %zu: ref=%.6f got=%.6f tol=%.6f\n", i, ref[i], got[i], tol);
                ++shown;
            }
        }
    }
    return {errors, max_diff, max_ratio, cd};
}

int main(int argc, char** argv) {
    int M = 256, N = 512, K = 512, batch = 2, verify = 1, warmup = 50, iters = 100, cpu_check = -1;
    int tile_m = 256, tile_n = 256;   // -t 256x256 | 256x128 | 128x256 | 128x128
    for (int i = 1; i + 1 < argc; i += 2) {
        const char* f = argv[i]; const int v = std::atoi(argv[i + 1]);
        if (!std::strcmp(f, "-t")) {
            if (std::sscanf(argv[i + 1], "%dx%d", &tile_m, &tile_n) != 2) { std::fprintf(stderr, "bad -t\n"); return 1; }
            continue;
        }
        if (!std::strcmp(f, "-m")) M = v;
        else if (!std::strcmp(f, "-n")) N = v;
        else if (!std::strcmp(f, "-k")) K = v;
        else if (!std::strcmp(f, "-b")) batch = v;
        else if (!std::strcmp(f, "-v")) verify = v;
        else if (!std::strcmp(f, "-w")) warmup = v;
        else if (!std::strcmp(f, "-i")) iters = v;
        else if (!std::strcmp(f, "-c")) cpu_check = v;
        else { std::fprintf(stderr, "unknown flag %s\n", f); return 1; }
    }
    using T = TileTraits<256, 256>;   // tile-independent constants
    const bool tile_ok = (tile_m == 256 || tile_m == 128 || tile_m == 192) && (tile_n == 256 || tile_n == 128 || tile_n == 192) && !(tile_m == 192 && tile_n != 256) && !(tile_n == 192 && tile_m != 256);
    if (!tile_ok) { std::fprintf(stderr, "tile must be 256x256 / 256x128 / 128x256 / 128x128 / 192x256 / 256x192\n"); return 1; }
    // Any M; N a multiple of 16 (shuffle_weight); K a multiple of one K tile (256).
    if (M < 1 || N % 16 || K % T::B_K || K < T::B_K) {
        std::fprintf(stderr, "need M >= 1, N %% 16 == 0, K %% %d == 0\n", T::B_K);
        return 1;
    }
    if (cpu_check < 0) cpu_check = (static_cast<double>(M) * N * K * batch <= 4.0e9);

    const size_t a_bytes = static_cast<size_t>(batch) * M * K / 2;
    const size_t b_bytes = static_cast<size_t>(batch) * N * K / 2;
    const size_t c_cnt = static_cast<size_t>(batch) * M * N;
    const int kg = K / 32;
    const size_t sfa_cnt = static_cast<size_t>(batch) * M * kg;
    const size_t sfb_cnt = static_cast<size_t>(batch) * N * kg;

    std::vector<uint8_t> h_a(a_bytes), h_b(b_bytes);
    std::vector<e8m0_t> h_sfa(sfa_cnt), h_sfb(sfb_cnt);
    rand_bytes(h_a.data(), a_bytes, 1);
    rand_bytes(h_b.data(), b_bytes, 2);
    rand_scale_e8m0(h_sfa.data(), sfa_cnt, 3);
    rand_scale_e8m0(h_sfb.data(), sfb_cnt, 4);
    if (std::getenv("MXFP4_UNIT_SCALE")) {
        std::fill(h_sfa.begin(), h_sfa.end(), 127);
        std::fill(h_sfb.begin(), h_sfb.end(), 127);
    }
    if (std::getenv("MXFP4_SFA_K_PATTERN"))   // per-K-group pattern, catches group mix-ups
        for (size_t r = 0; r < static_cast<size_t>(batch) * M; ++r)
            for (int g = 0; g < kg; ++g) h_sfa[r * kg + g] = static_cast<e8m0_t>(120 + (g % 13));

    uint8_t *d_a, *d_b; e8m0_t *d_sfa_raw, *d_sfb_raw; uint16_t* d_c;   // C: bf16 bits
    CHECK_HIP(hipMalloc(&d_a, a_bytes));
    CHECK_HIP(hipMalloc(&d_b, b_bytes));
    CHECK_HIP(hipMalloc(&d_c, c_cnt * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&d_sfa_raw, sfa_cnt));
    // sc: kernel scales in aiter e8m0_shuffle (shuffle_scale, gfx950) layout: rows padded to 32, K/32 to 8 (pad 0x7F),
    // [rows/32][kg/8][k_lane 4][row16 16][kk 2][row-pair 2]; stride_sf* = bytes per 32-row block.
    const int kg8 = (kg + 7) / 8 * 8;
    // Rows padded to 256 as aiter's e8m0_shuffle; its padding is torch.empty, so pad with 0xFF (NaN) by default to make sure
    // padded scales only reach rows / columns that are never written (MXFP4_SF_PAD overrides the pad byte).
    const int sf_pad = std::getenv("MXFP4_SF_PAD") ? std::atoi(std::getenv("MXFP4_SF_PAD")) : 0xFF;
    auto rows_pad = [](int rows) { return (rows + 255) / 256 * 256; };
    auto shuffle_sf = [&](const std::vector<e8m0_t>& raw, int rows) {
        const int rows32 = rows_pad(rows);
        std::vector<e8m0_t> out(static_cast<size_t>(batch) * rows32 * kg8, static_cast<e8m0_t>(sf_pad));
        for (int b = 0; b < batch; ++b) for (int r = 0; r < rows; ++r) for (int g = 0; g < kg; ++g) {
            const size_t o = static_cast<size_t>(b) * rows32 * kg8 + (static_cast<size_t>(r / 32) * (kg8 / 8) + g / 8) * 256
                           + (g % 4) * 64 + (r % 16) * 4 + ((g % 8) / 4) * 2 + (r % 32) / 16;
            out[o] = raw[(static_cast<size_t>(b) * rows + r) * kg + g];
        }
        return out;
    };
    const std::vector<e8m0_t> h_sfa_s = shuffle_sf(h_sfa, M), h_sfb_s = shuffle_sf(h_sfb, N);
    e8m0_t *d_sfa_s, *d_sfb_s;
    CHECK_HIP(hipMalloc(&d_sfa_s, h_sfa_s.size())); CHECK_HIP(hipMalloc(&d_sfb_s, h_sfb_s.size()));
    CHECK_HIP(hipMemcpy(d_sfa_s, h_sfa_s.data(), h_sfa_s.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfb_s, h_sfb_s.data(), h_sfb_s.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc(&d_sfb_raw, sfb_cnt));
    CHECK_HIP(hipMemcpy(d_a, h_a.data(), a_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), b_bytes, hipMemcpyHostToDevice));
    // bs: kernel B in aiter shuffle_weight(w, layout=(16, 16)) layout (byte permutation of each [N, K/2] batch):
    // (n, kb) -> (n/16)*16*Kb + (kb/32)*512 + ((kb/16)%2)*256 + (n%16)*16 + kb%16.  The reference keeps the raw B.
    if (N % 16 != 0 || (K / 2) % 32 != 0) { std::printf("bs: N %% 16 and (K/2) %% 32 must be 0\n"); return 1; }
    std::vector<uint8_t> h_b_s(b_bytes);
    {
        const size_t Kb = static_cast<size_t>(K) / 2;
        for (int bb = 0; bb < batch; ++bb) for (int n = 0; n < N; ++n) for (size_t kb = 0; kb < Kb; ++kb) {
            const size_t o = (static_cast<size_t>(n) / 16) * 16 * Kb + (kb / 32) * 512 + ((kb / 16) % 2) * 256 + (n % 16) * 16 + kb % 16;
            h_b_s[static_cast<size_t>(bb) * N * Kb + o] = h_b[(static_cast<size_t>(bb) * N + n) * Kb + kb];
        }
    }
    uint8_t* d_b_s; CHECK_HIP(hipMalloc(&d_b_s, b_bytes));
    CHECK_HIP(hipMemcpy(d_b_s, h_b_s.data(), b_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfa_raw, h_sfa.data(), sfa_cnt, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfb_raw, h_sfb.data(), sfb_cnt, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_c, 0xFF, c_cnt * sizeof(uint16_t))); // NaN poison

    const int num_tiles_m = (M + tile_m - 1) / tile_m, num_tiles_n = (N + tile_n - 1) / tile_n;
    opus_gemm_mxfp4_scale_kargs_gfx950 kargs{};
    kargs.ptr_a = d_a; kargs.ptr_b = d_b_s; kargs.ptr_c = d_c;   // bs: shuffled B
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
    kargs.stride_a = K / 2; kargs.stride_b = K / 2; kargs.stride_c = N;
    kargs.stride_a_batch = M * K / 2; kargs.stride_b_batch = N * K / 2; kargs.stride_c_batch = M * N;
    kargs.ptr_sfa = d_sfa_s; kargs.ptr_sfb = d_sfb_s;   // sc: e8m0_shuffle layout
    kargs.stride_sfa = 32 * kg8; kargs.stride_sfb = 32 * kg8;
    kargs.stride_sfa_batch = rows_pad(M) * kg8; kargs.stride_sfb_batch = rows_pad(N) * kg8;

    // Persistent: one block per CU (LDS limits occupancy to 1), each block walks the
    // (batch, m, n) tile list with stride gridDim.x.  PERSIST_GRID overrides the block count.
    int num_cu = 0; CHECK_HIP(hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, 0));
    const int total_tiles = num_tiles_m * num_tiles_n * batch;
    int grid_x = std::min(total_tiles, num_cu);
    if (std::getenv("PERSIST_GRID")) grid_x = std::min(total_tiles, std::atoi(std::getenv("PERSIST_GRID")));
    dim3 grid(grid_x, 1, 1), block(T::BLOCK_SIZE);
    kargs.grid_blocks = grid_x;
    // MXFP4_CO_DIR=<dir>: launch the pre-compiled gen_co image <dir>/opus_gemm_gfx950_a4w4_mxfp4_nt_co_<BM>x<BN>x256.co instead.
    hipFunction_t co_fn = nullptr;
    if (const char* co_dir = std::getenv("MXFP4_CO_DIR")) {
        char sym[128]; std::snprintf(sym, sizeof(sym), "opus_gemm_gfx950_a4w4_mxfp4_nt_co_%dx%dx256", tile_m, tile_n);
        const std::string path = std::string(co_dir) + "/" + sym + ".co";
        hipModule_t mod; CHECK_HIP(hipModuleLoad(&mod, path.c_str()));
        CHECK_HIP(hipModuleGetFunction(&co_fn, mod, sym));
        std::printf("using code object %s\n", path.c_str());
    }
    size_t kargs_size = sizeof(kargs);
    void* co_cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kargs, HIP_LAUNCH_PARAM_BUFFER_SIZE, &kargs_size, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() {
        if (co_fn) { CHECK_HIP(hipModuleLaunchKernel(co_fn, grid.x, 1, 1, block.x, 1, 1, 0, nullptr, nullptr, co_cfg)); return; }
        if (tile_m == 256 && tile_n == 256)      gemm_a4w4_mxfp4_nt_kernel<TileTraits<256, 256>><<<grid, block>>>(kargs);
        else if (tile_m == 256 && tile_n == 128) gemm_a4w4_mxfp4_nt_kernel<TileTraits<256, 128>><<<grid, block>>>(kargs);
        else if (tile_m == 128 && tile_n == 256) gemm_a4w4_mxfp4_nt_kernel<TileTraits<128, 256>><<<grid, block>>>(kargs);
        else if (tile_m == 192)                  gemm_a4w4_mxfp4_nt_kernel<TileTraits<192, 256>><<<grid, block>>>(kargs);
        else if (tile_n == 192)                  gemm_a4w4_mxfp4_nt_kernel<TileTraits<256, 192>><<<grid, block>>>(kargs);
        else                                     gemm_a4w4_mxfp4_nt_kernel<TileTraits<128, 128>><<<grid, block>>>(kargs);
    };
    std::printf("MXFP4xMXFP4 NT 4-wave (1x32 UE8M0) tile %dx%d M=%d N=%d K=%d batch=%d grid=(%u,1,%u) block=%d\n",
                tile_m, tile_n, M, N, K, batch, grid.x, grid.z, T::BLOCK_SIZE);
    launch();
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if (verify) {
        std::vector<float> h_c(c_cnt), h_ref(c_cnt), h_mag(c_cnt);
        {
            std::vector<uint16_t> h_c16(c_cnt);
            CHECK_HIP(hipMemcpy(h_c16.data(), d_c, c_cnt * sizeof(uint16_t), hipMemcpyDeviceToHost));
            for (size_t i = 0; i < c_cnt; ++i) { const uint32_t u = static_cast<uint32_t>(h_c16[i]) << 16; std::memcpy(&h_c[i], &u, 4); }
        }
        float *d_ref, *d_mag;
        CHECK_HIP(hipMalloc(&d_ref, c_cnt * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_mag, c_cnt * sizeof(float)));
        for (int b = 0; b < batch; ++b) {
            dim3 rb(16, 16), rg((N + 15) / 16, (M + 15) / 16);
            gemm_ref_gpu<<<rg, rb>>>(d_a + static_cast<size_t>(b) * M * K / 2, d_b + static_cast<size_t>(b) * N * K / 2,
                                     d_sfa_raw + static_cast<size_t>(b) * M * kg, d_sfb_raw + static_cast<size_t>(b) * N * kg,
                                     d_ref + static_cast<size_t>(b) * M * N, d_mag + static_cast<size_t>(b) * M * N, M, N, K);
        }
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipMemcpy(h_ref.data(), d_ref, c_cnt * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_mag.data(), d_mag, c_cnt * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipFree(d_ref)); CHECK_HIP(hipFree(d_mag));

        if (cpu_check) {
            std::vector<float> h_cpu(c_cnt);
            for (int b = 0; b < batch; ++b)
                gemm_ref_cpu(h_a.data() + static_cast<size_t>(b) * M * K / 2, h_b.data() + static_cast<size_t>(b) * N * K / 2,
                             h_sfa.data() + static_cast<size_t>(b) * M * kg, h_sfb.data() + static_cast<size_t>(b) * N * kg,
                             h_cpu.data() + static_cast<size_t>(b) * M * N, M, N, K);
            auto s = compare(h_cpu.data(), h_ref.data(), h_mag.data(), c_cnt, 1e-6f, 1e-6f, false);
            std::printf("[gpu-ref vs cpu-ref] errors=%ld max_diff=%.3e calc_diff=%.3e %s\n",
                        s.errors, s.max_diff, s.calc_diff, s.errors ? "FAIL" : "OK");
            ok = ok && !s.errors;
        }
        auto s = compare(h_ref.data(), h_c.data(), h_mag.data(), c_cnt, 5e-5f, 1e-4f, true, 1.0 / 256);   // + bf16 output rounding
        const bool pass = !s.errors && s.calc_diff < 0.02;
        std::printf("[kernel vs ref] errors=%ld/%zu max_diff=%.3e max_ratio=%.3f calc_diff=%.3e (gate 0.02) %s\n",
                    s.errors, c_cnt, s.max_diff, s.max_ratio, s.calc_diff, pass ? "VALID" : "FAIL");
        ok = ok && pass;
    }

    if (iters > 0) {
        for (int i = 0; i < warmup; ++i) launch();
        hipEvent_t e0, e1;
        CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) launch();
        CHECK_HIP(hipEventRecord(e1));
        CHECK_HIP(hipEventSynchronize(e1));
        float ms = 0; CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
        const double avg_ms = ms / iters;
        const double tflops = 2.0 * M * N * static_cast<double>(K) * batch / (avg_ms * 1e-3) / 1e12;
        std::printf("Kernel Performance: avg_time=%.4f ms (%.1f us), %.1f TFlops\n", avg_ms, avg_ms * 1e3, tflops);
    }

    CHECK_HIP(hipFree(d_a)); CHECK_HIP(hipFree(d_b)); CHECK_HIP(hipFree(d_c));
    CHECK_HIP(hipFree(d_sfa_raw)); CHECK_HIP(hipFree(d_sfb_raw));
    return ok ? 0 : 2;
}
