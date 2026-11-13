// Host driver / test harness for the small-tile MXFP4 GEMM (../include/gfx950/opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh).  Same data / reference as work_m10's host; inputs in the
// aiter gemm_a4w4(bpreshuffle=True) layout.  Arbitrary M, N (N % 16 == 0), K % 256 == 0.
//   -a 0: B_DIRECT (-t 16x32 / 16x64 / 32x32 / 32x64, -s KS 2/4)   -a 1: B_LDS (-t 32x128 / 64x128 / 128x128 / 32x256 / 64x256)
//   -l 0: shuffled inputs (gemm_a4w4 bpreshuffle)   -l 1: row-major inputs (Triton gemm_afp4wfp4: B, scales row-major, N % 8)
//   -D dir: dump raw inputs + C   -d prefetch depth (ring)  -p split-K over blocks
//   -m -n -k -b   -v verify   -i graph iterations (0 = no timing).  Default 16x32 KS4 D3.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <omp.h>
#include "opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh"
using smallm_kargs = opus_gemm_mxfp4_small_kargs_gfx950;
#define CHECK_HIP(call) do { hipError_t s_ = call; if (s_ != hipSuccess) { fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(s_)); exit(1); } } while (0)
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

template<int BM, int BN, int D, int KS, int BMODE, int L> void launch_k(const smallm_kargs& ka, hipStream_t st) {
    dim3 grid((ka.n + BN - 1) / BN, (ka.m + BM - 1) / BM, ka.batch * ka.splitk);
    gemm_a4w4_mxfp4_nt_small_kernel<opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950<BM, BN, D, KS, BMODE, L>><<<grid, 256, 0, st>>>(ka);
}
static int g_layout = OPUS_MXFP4_SMALL_SHUFFLED;
template<int BM, int BN, int D> void launch_lr(const smallm_kargs& ka, hipStream_t st) {   // row-major-only LDS tiles
    launch_k<BM, BN, D, 1, OPUS_MXFP4_SMALL_B_LDS, OPUS_MXFP4_SMALL_ROWMAJOR>(ka, st);
}
template<int BM, int BN, int D, int KS> void launch_t(const smallm_kargs& ka, hipStream_t st) {
    if (g_layout == OPUS_MXFP4_SMALL_ROWMAJOR) launch_k<BM, BN, D, KS, OPUS_MXFP4_SMALL_B_DIRECT, OPUS_MXFP4_SMALL_ROWMAJOR>(ka, st);
    else                                      launch_k<BM, BN, D, KS, OPUS_MXFP4_SMALL_B_DIRECT, OPUS_MXFP4_SMALL_SHUFFLED>(ka, st);
}
template<int BM, int BN, int D> void launch_l(const smallm_kargs& ka, hipStream_t st) {
    if (g_layout == OPUS_MXFP4_SMALL_ROWMAJOR) launch_k<BM, BN, D, 1, OPUS_MXFP4_SMALL_B_LDS, OPUS_MXFP4_SMALL_ROWMAJOR>(ka, st);
    else                                      launch_k<BM, BN, D, 1, OPUS_MXFP4_SMALL_B_LDS, OPUS_MXFP4_SMALL_SHUFFLED>(ka, st);
}
using launch_fn = void (*)(const smallm_kargs&, hipStream_t);
// Instantiated configurations (the small-tile pipeline's tile set):
//   B_DIRECT: 16x32 KS4, 16x64 KS2/KS4, 32x32 KS4, 32x64 KS2/KS4, D 2-4    B_LDS: 32x128 / 64x128 / 128x128 D 2-4, 32x256 D 2-3, 64x256 D 2
launch_fn pick_lds(int bm, int bn, int d) {
#define SL3(BM, BN) if (bm == BM && bn == BN) { if (d == 2) return launch_l<BM, BN, 2>; if (d == 3) return launch_l<BM, BN, 3>; }
#define SL4(BM, BN) SL3(BM, BN) if (bm == BM && bn == BN && d == 4) return launch_l<BM, BN, 4>;
    SL4(32, 128) SL4(64, 128) SL4(128, 128) SL3(32, 256)
    if (bm == 64 && bn == 256 && d == 2) return launch_l<64, 256, 2>;
    // row-major only (-l 1): BM 16 / BN 64 LDS tiles
    if (g_layout == OPUS_MXFP4_SMALL_ROWMAJOR) {
        if (bm == 16 && bn == 64) { if (d == 2) return launch_lr<16, 64, 2>; if (d == 3) return launch_lr<16, 64, 3>; if (d == 4) return launch_lr<16, 64, 4>; }
        if (bm == 32 && bn == 64) { if (d == 2) return launch_lr<32, 64, 2>; if (d == 3) return launch_lr<32, 64, 3>; if (d == 4) return launch_lr<32, 64, 4>; }
        if (bm == 16 && bn == 128) { if (d == 2) return launch_lr<16, 128, 2>; if (d == 3) return launch_lr<16, 128, 3>; if (d == 4) return launch_lr<16, 128, 4>; }
        if (bm == 64 && bn == 64) { if (d == 2) return launch_lr<64, 64, 2>; if (d == 3) return launch_lr<64, 64, 3>; }
    }
#undef SL3
#undef SL4
    return nullptr;
}
launch_fn pick(int bm, int bn, int d, int ks) {
#define SM_D(BM, BN, KS) if (bm == BM && bn == BN && ks == KS) { if (d == 2) return launch_t<BM, BN, 2, KS>; if (d == 3) return launch_t<BM, BN, 3, KS>; if (d == 4) return launch_t<BM, BN, 4, KS>; }
    SM_D(16, 32, 4) SM_D(16, 64, 4) SM_D(16, 64, 2) SM_D(32, 32, 4) SM_D(32, 64, 4) SM_D(32, 64, 2)
#undef SM_D
    return nullptr;
}

int main(int argc, char** argv) {
    int M = 16, N = 1024, K = 4096, batch = 1, verify = 1, iters = 0, depth = 0, bm = 16, bn = 32, ks = 4, splitk = 1, alds = 0;
    const char* dump_dir = nullptr;   // default: best overall (52 of 60 target shapes)
    for (int i = 1; i + 1 < argc; i += 2) {
        const char* f = argv[i];
        if (!std::strcmp(f, "-t")) { std::sscanf(argv[i + 1], "%dx%d", &bm, &bn); continue; }
        if (!std::strcmp(f, "-D")) { dump_dir = argv[i + 1]; continue; }
        const int v = std::atoi(argv[i + 1]);
        if (!std::strcmp(f, "-m")) M = v; else if (!std::strcmp(f, "-n")) N = v; else if (!std::strcmp(f, "-k")) K = v;
        else if (!std::strcmp(f, "-b")) batch = v; else if (!std::strcmp(f, "-v")) verify = v; else if (!std::strcmp(f, "-i")) iters = v;
        else if (!std::strcmp(f, "-d")) depth = v;
        else if (!std::strcmp(f, "-s")) ks = v;
        else if (!std::strcmp(f, "-p")) splitk = v;
        else if (!std::strcmp(f, "-a")) alds = v;
        else if (!std::strcmp(f, "-l")) g_layout = v;
        else { std::fprintf(stderr, "unknown flag %s\n", f); return 1; }
    }
    if (depth == 0) depth = (bm * bn >= 64 * 256) ? 2 : 3;
    launch_fn launch = alds ? pick_lds(bm, bn, depth) : pick(bm, bn, depth, ks);
    if (!launch) { std::fprintf(stderr, "unsupported tile %dx%d depth %d ks %d\n", bm, bn, depth, ks); return 1; }
    const bool RM = g_layout == OPUS_MXFP4_SMALL_ROWMAJOR;
    if (N % (RM ? 8 : 16) || K % 256 || M < 1) { std::fprintf(stderr, "need N %% %d == 0, K %% 256 == 0\n", RM ? 8 : 16); return 1; }

    const size_t a_bytes = (size_t)batch * M * K / 2, b_bytes = (size_t)batch * N * K / 2, c_cnt = (size_t)batch * M * N;
    const int kg = K / 32, kg8 = (kg + 7) / 8 * 8;
    const size_t sfa_cnt = (size_t)batch * M * kg, sfb_cnt = (size_t)batch * N * kg;
    std::vector<uint8_t> h_a(a_bytes), h_b(b_bytes);
    std::vector<e8m0_t> h_sfa(sfa_cnt), h_sfb(sfb_cnt);
    rand_bytes(h_a.data(), a_bytes, 1); rand_bytes(h_b.data(), b_bytes, 2);
    rand_scale_e8m0(h_sfa.data(), sfa_cnt, 3); rand_scale_e8m0(h_sfb.data(), sfb_cnt, 4);
    if (std::getenv("MXFP4_SFA_K_PATTERN"))
        for (size_t r = 0; r < (size_t)batch * M; ++r) for (int g = 0; g < kg; ++g) h_sfa[r * kg + g] = (e8m0_t)(120 + (g % 13));
    // e8m0_shuffle: rows padded to 256 (as aiter), K/32 to 8; padding poisoned with 0xFF (e8m0 NaN) -- must never reach C.
    auto shuffle_sf = [&](const std::vector<e8m0_t>& raw, int rows, int& rows_pad) {
        rows_pad = (rows + 255) / 256 * 256;
        std::vector<e8m0_t> out((size_t)batch * rows_pad * kg8, (e8m0_t)0xFF);
        for (int b = 0; b < batch; ++b) for (int r = 0; r < rows; ++r) for (int g = 0; g < kg; ++g) {
            const size_t o = (size_t)b * rows_pad * kg8 + ((size_t)(r / 32) * (kg8 / 8) + g / 8) * 256
                           + (g % 4) * 64 + (r % 16) * 4 + ((g % 8) / 4) * 2 + (r % 32) / 16;
            out[o] = raw[((size_t)b * rows + r) * kg + g];
        }
        return out;
    };
    int ma_pad, nb_pad;
    const std::vector<e8m0_t> h_sfa_s = shuffle_sf(h_sfa, M, ma_pad), h_sfb_s = shuffle_sf(h_sfb, N, nb_pad);
    const size_t bs_bytes = (size_t)batch * ((N + 15) / 16 * 16) * K / 2;   // shuffled B: rows padded to 16
    std::vector<uint8_t> h_b_s(bs_bytes);
    { const size_t Kb = (size_t)K / 2;
      for (int bb = 0; bb < batch; ++bb) for (int n = 0; n < N; ++n) for (size_t kb = 0; kb < Kb; ++kb) {
          const size_t o = ((size_t)n / 16) * 16 * Kb + (kb / 32) * 512 + ((kb / 16) % 2) * 256 + (n % 16) * 16 + kb % 16;
          h_b_s[(size_t)bb * ((N + 15) / 16 * 16) * Kb + o] = h_b[((size_t)bb * N + n) * Kb + kb];
      } }
    uint8_t *d_a, *d_b, *d_b_s; e8m0_t *d_sfa, *d_sfb, *d_sfa_s, *d_sfb_s; uint16_t* d_c;
    CHECK_HIP(hipMalloc(&d_a, a_bytes)); CHECK_HIP(hipMalloc(&d_b, b_bytes)); CHECK_HIP(hipMalloc(&d_b_s, bs_bytes));
    CHECK_HIP(hipMalloc(&d_sfa, sfa_cnt)); CHECK_HIP(hipMalloc(&d_sfb, sfb_cnt));
    CHECK_HIP(hipMalloc(&d_sfa_s, h_sfa_s.size())); CHECK_HIP(hipMalloc(&d_sfb_s, h_sfb_s.size()));
    CHECK_HIP(hipMalloc(&d_c, c_cnt * 2));
    CHECK_HIP(hipMemcpy(d_a, h_a.data(), a_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), b_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b_s, h_b_s.data(), bs_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfa, h_sfa.data(), sfa_cnt, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfb, h_sfb.data(), sfb_cnt, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfa_s, h_sfa_s.data(), h_sfa_s.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfb_s, h_sfb_s.data(), h_sfb_s.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_c, 0xFF, c_cnt * 2));   // NaN poison: every C element must be written
    // row-major scales (Triton gemm_afp4wfp4 layout): exactly [batch*rows, K/32], followed by a 256-row 0xFF (NaN) guard
    // that the kernel must never read (buffer ranges stop at row M / N).
    e8m0_t *d_sfa_r, *d_sfb_r;
    CHECK_HIP(hipMalloc(&d_sfa_r, sfa_cnt + 256 * kg)); CHECK_HIP(hipMalloc(&d_sfb_r, sfb_cnt + 256 * kg));
    CHECK_HIP(hipMemset(d_sfa_r + sfa_cnt, 0xFF, 256 * kg)); CHECK_HIP(hipMemset(d_sfb_r + sfb_cnt, 0xFF, 256 * kg));
    CHECK_HIP(hipMemcpy(d_sfa_r, h_sfa.data(), sfa_cnt, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfb_r, h_sfb.data(), sfb_cnt, hipMemcpyHostToDevice));

    smallm_kargs ka{};
    ka.ptr_a = d_a; ka.ptr_b = d_b_s; ka.ptr_c = d_c; ka.ptr_sfa = d_sfa_s; ka.ptr_sfb = d_sfb_s;
    ka.m = M; ka.n = N; ka.k = K; ka.batch = batch; ka.stride_a = K / 2; ka.stride_c = N;
    ka.stride_sfa = 32 * kg8; ka.stride_sfb = 32 * kg8;
    ka.stride_a_batch = (long long)M * K / 2; ka.stride_b_batch = (long long)N * K / 2; ka.stride_c_batch = (long long)M * N;
    ka.stride_sfa_batch = (long long)ma_pad * kg8; ka.stride_sfb_batch = (long long)nb_pad * kg8;
    if (RM) {
        ka.ptr_b = d_b; ka.ptr_sfa = d_sfa_r; ka.ptr_sfb = d_sfb_r;
        ka.stride_sfa = kg; ka.stride_sfb = kg; ka.stride_sfa_batch = (long long)M * kg; ka.stride_sfb_batch = (long long)N * kg;
    }
    ka.splitk = std::max(1, std::min(std::min(splitk, 8), K / 256));
    { const size_t tm = (M + bm - 1) / bm, tn = (N + bn - 1) / bn;
      CHECK_HIP(hipMalloc(&ka.cnt, batch * tm * tn * sizeof(int))); CHECK_HIP(hipMemset(ka.cnt, 0, batch * tm * tn * sizeof(int)));
      CHECK_HIP(hipMalloc(&ka.ws, ka.splitk > 1 ? (size_t)batch * ka.splitk * tm * bm * tn * bn * sizeof(float) : 4)); }
    std::printf("%s ", RM ? "rowmajor" : "shuffled");
    std::printf("smallm split%d ks%d tile %dx%d d%d M=%d N=%d K=%d batch=%d grid=(%d,%d,%d)\n", ka.splitk, ks, bm, bn, depth, M, N, K, batch,
                (N + bn - 1) / bn, (M + bm - 1) / bm, batch);
    launch(ka, 0); CHECK_HIP(hipGetLastError()); CHECK_HIP(hipDeviceSynchronize());

    if (dump_dir) {   // raw inputs (row-major / unshuffled) and the kernel's C, for an external comparison (Triton)
        auto dump = [&](const char* name, const void* dptr, size_t bytes) {
            std::vector<uint8_t> h(bytes); CHECK_HIP(hipMemcpy(h.data(), dptr, bytes, hipMemcpyDeviceToHost));
            const std::string path = std::string(dump_dir) + "/" + name;
            FILE* f = std::fopen(path.c_str(), "wb"); std::fwrite(h.data(), 1, bytes, f); std::fclose(f);
        };
        dump("a.bin", d_a, a_bytes); dump("b.bin", d_b, b_bytes); dump("sfa.bin", d_sfa, sfa_cnt); dump("sfb.bin", d_sfb, sfb_cnt);
        dump("c.bin", d_c, c_cnt * 2);
        const std::string meta = std::string(dump_dir) + "/meta.txt";
        FILE* f = std::fopen(meta.c_str(), "w"); std::fprintf(f, "%d %d %d %d\n", M, N, K, batch); std::fclose(f);
    }
    bool ok = true;
    if (verify) {
        std::vector<float> h_c(c_cnt), h_ref(c_cnt), h_mag(c_cnt);
        { std::vector<uint16_t> c16(c_cnt); CHECK_HIP(hipMemcpy(c16.data(), d_c, c_cnt * 2, hipMemcpyDeviceToHost));
          for (size_t i = 0; i < c_cnt; ++i) { const uint32_t u = (uint32_t)c16[i] << 16; std::memcpy(&h_c[i], &u, 4); } }
        float *d_ref, *d_mag; CHECK_HIP(hipMalloc(&d_ref, c_cnt * 4)); CHECK_HIP(hipMalloc(&d_mag, c_cnt * 4));
        for (int b = 0; b < batch; ++b) {
            dim3 rb(16, 16), rg((N + 15) / 16, (M + 15) / 16);
            gemm_ref_gpu<<<rg, rb>>>(d_a + (size_t)b * M * K / 2, d_b + (size_t)b * N * K / 2, d_sfa + (size_t)b * M * kg,
                                     d_sfb + (size_t)b * N * kg, d_ref + (size_t)b * M * N, d_mag + (size_t)b * M * N, M, N, K);
        }
        CHECK_HIP(hipMemcpy(h_ref.data(), d_ref, c_cnt * 4, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_mag.data(), d_mag, c_cnt * 4, hipMemcpyDeviceToHost));
        CHECK_HIP(hipFree(d_ref)); CHECK_HIP(hipFree(d_mag));
        auto s = compare(h_ref.data(), h_c.data(), h_mag.data(), c_cnt, 5e-5f, 1e-4f, true, 1.0 / 256);
        const bool pass = !s.errors && s.calc_diff < 0.02;
        std::printf("[kernel vs ref] errors=%ld/%zu max_diff=%.3e calc_diff=%.3e %s\n", s.errors, c_cnt, s.max_diff, s.calc_diff, pass ? "VALID" : "FAIL");
        ok = pass;
    }
    if (iters > 0) {   // hipGraph of `iters` back-to-back launches (no CPU launch overhead), replayed 3x after warmup
        hipStream_t st; CHECK_HIP(hipStreamCreate(&st));
        hipGraph_t g; hipGraphExec_t ge;
        CHECK_HIP(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
        for (int i = 0; i < iters; ++i) launch(ka, st);
        CHECK_HIP(hipStreamEndCapture(st, &g));
        CHECK_HIP(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int w = 0; w < 5; ++w) CHECK_HIP(hipGraphLaunch(ge, st));
        CHECK_HIP(hipStreamSynchronize(st));
        hipEvent_t e0, e1; CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
        std::vector<float> r;
        for (int rep = 0; rep < 5; ++rep) {
            CHECK_HIP(hipEventRecord(e0, st)); CHECK_HIP(hipGraphLaunch(ge, st)); CHECK_HIP(hipEventRecord(e1, st));
            CHECK_HIP(hipEventSynchronize(e1)); float ms; CHECK_HIP(hipEventElapsedTime(&ms, e0, e1)); r.push_back(ms * 1e3f / iters);
        }
        std::sort(r.begin(), r.end());
        const double us = r[r.size() / 2];
        std::printf("time %.3f us  B %.2f TB/s  %.1f TFlops\n", us, (double)b_bytes / (us * 1e-6) / 1e12, 2.0 * M * N * (double)K * batch / (us * 1e-6) / 1e12);
    }
    return ok ? 0 : 2;
}
