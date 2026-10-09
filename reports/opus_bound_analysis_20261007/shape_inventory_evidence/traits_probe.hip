#include <cstdio>
#include <typeinfo>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_fine_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
template<class T> long long get_B_M() { if constexpr(requires{T::B_M;}) return T::B_M; else return -1; }
template<class T> long long get_B_N() { if constexpr(requires{T::B_N;}) return T::B_N; else return -1; }
template<class T> long long get_B_K() { if constexpr(requires{T::B_K;}) return T::B_K; else return -1; }
template<class T> long long get_BLOCK_SIZE() { if constexpr(requires{T::BLOCK_SIZE;}) return T::BLOCK_SIZE; else return -1; }
template<class T> long long get_T_M() { if constexpr(requires{T::T_M;}) return T::T_M; else return -1; }
template<class T> long long get_T_N() { if constexpr(requires{T::T_N;}) return T::T_N; else return -1; }
template<class T> long long get_T_K() { if constexpr(requires{T::T_K;}) return T::T_K; else return -1; }
template<class T> long long get_NUM_WAVES() { if constexpr(requires{T::NUM_WAVES;}) return T::NUM_WAVES; else return -1; }
template<class T> long long get_NUM_STAGES() { if constexpr(requires{T::NUM_STAGES;}) return T::NUM_STAGES; else return -1; }
template<class T> long long get_CLUSTER() { if constexpr(requires{T::CLUSTER;}) return T::CLUSTER; else return -1; }
template<class T> long long get_OUTPUT() { if constexpr(requires{T::OUTPUT;}) return T::OUTPUT; else return -1; }
template<class T> long long get_PREFETCH() { if constexpr(requires{T::PREFETCH;}) return T::PREFETCH; else return -1; }
template<class T> long long get_FIXED_K() { if constexpr(requires{T::FIXED_K;}) return T::FIXED_K; else return -1; }
template<class T> long long get_SPLIT_K() { if constexpr(requires{T::SPLIT_K;}) return T::SPLIT_K; else return -1; }
template<class T> long long get_REDUCE_VEC() { if constexpr(requires{T::REDUCE_VEC;}) return T::REDUCE_VEC; else return -1; }
template<class T> long long get_REDUCE_BLOCK() { if constexpr(requires{T::REDUCE_BLOCK;}) return T::REDUCE_BLOCK; else return -1; }
template<class T> long long get_STORE_CACHE() { if constexpr(requires{T::STORE_CACHE;}) return T::STORE_CACHE; else return -1; }
template<class T> long long get_SCALE_PANEL() { if constexpr(requires{T::SCALE_PANEL;}) return T::SCALE_PANEL; else return -1; }
template<class T> long long get_SCALE_PANEL_K_CAPACITY() { if constexpr(requires{T::SCALE_PANEL_K_CAPACITY;}) return T::SCALE_PANEL_K_CAPACITY; else return -1; }
template<class T> long long get_A_STAGE() { if constexpr(requires{T::A_STAGE;}) return T::A_STAGE; else return -1; }
template<class T> long long get_B_STAGE() { if constexpr(requires{T::B_STAGE;}) return T::B_STAGE; else return -1; }
template<class T> long long get_B_GROUPS() { if constexpr(requires{T::B_GROUPS;}) return T::B_GROUPS; else return -1; }
template<class T> long long get_REGISTER_SCALES() { if constexpr(requires{T::REGISTER_SCALES;}) return T::REGISTER_SCALES; else return -1; }
template<class T> long long get_XOR_LDS() { if constexpr(requires{T::XOR_LDS;}) return T::XOR_LDS; else return -1; }
template<class T> long long get_EARLY_SCALE_LOADS() { if constexpr(requires{T::EARLY_SCALE_LOADS;}) return T::EARLY_SCALE_LOADS; else return -1; }
template<class T> long long get_FINE_M_LOADS() { if constexpr(requires{T::FINE_M_LOADS;}) return T::FINE_M_LOADS; else return -1; }
template<class T> long long get_LDS_BYTES() { if constexpr(requires{T::LDS_BYTES;}) return T::LDS_BYTES; else return -1; }
template<class T> long long get_WAVE_K() { if constexpr(requires{T::WAVE_K;}) return T::WAVE_K; else return -1; }
template<class T> long long get_BLOCK_GROUP_M() { if constexpr(requires{T::BLOCK_GROUP_M;}) return T::BLOCK_GROUP_M; else return -1; }
template<class T> void show(int id) { std::printf("%d",id);
std::printf(",%lld",get_B_M<T>());
std::printf(",%lld",get_B_N<T>());
std::printf(",%lld",get_B_K<T>());
std::printf(",%lld",get_BLOCK_SIZE<T>());
std::printf(",%lld",get_T_M<T>());
std::printf(",%lld",get_T_N<T>());
std::printf(",%lld",get_T_K<T>());
std::printf(",%lld",get_NUM_WAVES<T>());
std::printf(",%lld",get_NUM_STAGES<T>());
std::printf(",%lld",get_CLUSTER<T>());
std::printf(",%lld",get_OUTPUT<T>());
std::printf(",%lld",get_PREFETCH<T>());
std::printf(",%lld",get_FIXED_K<T>());
std::printf(",%lld",get_SPLIT_K<T>());
std::printf(",%lld",get_REDUCE_VEC<T>());
std::printf(",%lld",get_REDUCE_BLOCK<T>());
std::printf(",%lld",get_STORE_CACHE<T>());
std::printf(",%lld",get_SCALE_PANEL<T>());
std::printf(",%lld",get_SCALE_PANEL_K_CAPACITY<T>());
std::printf(",%lld",get_A_STAGE<T>());
std::printf(",%lld",get_B_STAGE<T>());
std::printf(",%lld",get_B_GROUPS<T>());
std::printf(",%lld",get_REGISTER_SCALES<T>());
std::printf(",%lld",get_XOR_LDS<T>());
std::printf(",%lld",get_EARLY_SCALE_LOADS<T>());
std::printf(",%lld",get_FINE_M_LOADS<T>());
std::printf(",%lld",get_LDS_BYTES<T>());
std::printf(",%lld",get_WAVE_K<T>());
std::printf(",%lld",get_BLOCK_GROUP_M<T>());
std::printf("|%s\n",typeid(T).name()); }
int main() {
show<opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950>(0);
show<opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950>(1);
show<opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950>(2);
show<opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<4, 64, 7168>>(3);
show<opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950>(4);
show<opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64, 7168, 4>>(5);
show<opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950>(6);
show<opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950>(7);
show<opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950>(8);
show<opus_gemm_mxscale_bpreshuffle_8wave_192x256_traits_gfx950>(9);
show<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128, 128, 64>>(10);
show<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 32, 1536>>(11);
show<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 32, 3072>>(12);
show<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 64, 7168>>(13);
show<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 8, 384>>(14);
show<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 8, 768>>(15);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<112, 1, 4, 4, 1, 4, 4, 128, 0, 0>>(16);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<128, 2, 2, 4, 1, 4, 4, 128, 0, 0>>(17);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<48, 1, 4, 4, 1, 4, 4, 128, 0, 0>>(18);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<48, 1, 4, 6, 2, 4, 4, 128, 0, 7168>>(19);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 1, 4, 128, 0, 0>>(20);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 1, 4, 128, 0, 3072>>(21);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 1, 4, 128, 0, 7168>>(22);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 2, 4, 128, 0, 0>>(23);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 2, 4, 128, 0, 16384>>(24);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 4, 16, 128, 2, 0>>(25);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 4, 4, 128, 0, 16384>>(26);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 2, 4, 1, 4, 16, 64, 0, 0>>(27);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 4, 4, 1, 1, 4, 128, 0, 0>>(28);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 4, 4, 1, 2, 16, 128, 2, 0>>(29);
show<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 4, 4, 1, 4, 16, 128, 2, 0>>(30);
show<opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950>(31);
show<opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950>(32);
show<opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950>(33);
show<opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950>(34);
show<opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950>(35);
show<opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950>(36);
show<opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950>(37);
show<opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950>(38);
show<opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950>(39);
show<opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950>(40);
show<opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950>(41);
show<opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950>(42);
show<opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950>(43);
show<opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950>(44);
show<opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950>(45);
show<opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950>(46);
show<opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 3, 8, 4, 3, 7168, false, true>>(47);
show<opus_gemm_small_register_traits_gfx950<16, 48, 1, 1, 4, 4, 4, 3, 7168, true, false>>(48);
show<opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 4, 4, 4, 3, 7168, false, true>>(49);
show<opus_gemm_small_register_traits_gfx950<32, 48, 1, 1, 3, 4, 4, 0, 7168, true, false>>(50);
}
