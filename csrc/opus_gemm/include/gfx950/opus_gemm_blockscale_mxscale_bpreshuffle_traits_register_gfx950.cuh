#ifndef OPUS_BP_REGISTER_TRAITS_BASE
#define OPUS_BP_REGISTER_TRAITS_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::registers::schedule_0 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
int WaveK = 1, int Output = 0, int BCache = 0,
int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
static constexpr int E_M = BlockM / (WaveM * W_M);
static constexpr int E_N = BlockN / (WaveN * W_N);
static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
static constexpr int B_CACHE = BCache, OUTPUT = Output;
static constexpr int FIXED_K = FixedK;
static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
static_assert(Output == 0 || Output == 3 || Output == 4);
static_assert(BCache >= 0 && BCache <= 3);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
static_assert(!PadN || Output == 4);
static_assert(!ReuseBScale || 128 % BlockN == 0);
static_assert(Output != 4 || WaveK > 1);
static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};
}

namespace opus_bpreshuffle_gfx950::registers::schedule_1 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0
template<int BM,int BN,int WM,int WN,int Prefetch,int WK,int BCache,int Chains,
int UniformScale,int InterleavedK,int ExactM,int FixedN,int FixedK>
struct small260_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=WM,T_N=WN,T_K=WK;
static constexpr int W_M=16,W_N=16,W_K=128,E_M=BM/(16*WM),E_N=BN/(16*WN);
static constexpr int BLOCK_SIZE=WM*WN*WK*64,PREFETCH=Prefetch,WAVE_K=WK;
static constexpr int B_CACHE=BCache,CHAINS=1,FIXED_K=FixedK,FIXED_N=FixedN,EXACT_M=ExactM;
static constexpr int LDS_M=ExactM ? ExactM : BM;
static constexpr bool PIN_C=WK==4 && WM==1 && WN==1;
static constexpr bool N_TAIL=false,REUSE_B_SCALE=true;
static constexpr bool UNIFORM_SCALE=UniformScale,INTERLEAVED_K=InterleavedK;
static_assert(BM%(16*WM)==0 && BN%(16*WN)==0 && 128%BN==0);
static_assert(BM<=64 && WK>=4 && WK<=16 && BLOCK_SIZE<=1024);
static_assert(Chains==1 && FixedN%BN==0 && FixedK%128==0);
static_assert(WK*LDS_M*BN*4<=160*1024);
static_assert(!ExactM || (ExactM<=4 && BM==16 && WM==1));
};
}

namespace opus_bpreshuffle_gfx950::registers::schedule_2 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0



#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_blockscale_mxscale_bpreshuffle_mfma16_output {
template<class T, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(
const Accumulators& c, Gmem gc, int lane, int wm, int wn, int stride_c) {
using namespace opus;
static_assert(T::W_M == 16 && T::W_N == 16);
const int side = (lane / 16) % 2;
static_for<T::E_M * T::E_N / 2>([&](auto pair) {
constexpr int ci = decltype(pair)::value * 2;
const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci + 1]));
const unsigned own0 = side ? second[0] : first[0];
const unsigned own1 = side ? second[1] : first[1];
const unsigned src0 = side ? first[0] : second[0];
const unsigned src1 = side ? first[1] : second[1];
const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src0);
const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src1);
vector_t<unsigned, 4> packed;
packed[0] = side ? other0 : own0;
packed[1] = side ? other1 : own1;
packed[2] = side ? own0 : other0;
packed[3] = side ? own1 : other1;
const int m = (ci + side) / T::E_N, n = (ci + side) % T::E_N;
const int rr = (m * T::T_M + wm) * 16 + lane % 16;
const int nn = (n * T::T_N + wn) * 16 + (lane / 32) * 8;
store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0, number<2>{});
});
if constexpr (T::E_M * T::E_N % 2 != 0) {
constexpr int ci = T::E_M * T::E_N - 1;
const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[0]);
const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[1]);
if (side == 0) {
vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
const int rr = ((ci / T::E_N) * T::T_M + wm) * 16 + lane % 16;
const int nn = ((ci % T::E_N) * T::T_N + wn) * 16 + (lane / 32) * 8;
store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0, number<2>{});
}
}
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::registers::schedule_3 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_packed_sfb_traits_gfx950
: BaseTraits {
static_assert(BaseTraits::B_M == 16 && BaseTraits::B_N == 32);
static_assert(BaseTraits::T_M == 1 && BaseTraits::T_N == 1);
static_assert(BaseTraits::PREFETCH == 3 && BaseTraits::WAVE_K == 8);
static_assert(BaseTraits::OUTPUT == 4 && BaseTraits::B_CACHE == 3);
static_assert(BaseTraits::FIXED_K == 7168 && !BaseTraits::N_TAIL);
static_assert(BaseTraits::REUSE_B_SCALE);
};
}

namespace opus_bpreshuffle_gfx950::registers::schedule_4 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_3;
using namespace opus_bpreshuffle_gfx950::registers::schedule_2;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::registers::schedule_5 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0, bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static constexpr bool SFA_WAVEORDER = false;
static_assert((StaticN == 0) == (StaticStrideK == 0));
static_assert(StaticN == 0 || (StaticN > 0 && StaticN % 128 == 0));
static_assert(StaticStrideK == 0 || (StaticStrideK > 0 && StaticStrideK <= 16384 &&
StaticStrideK % BaseTraits::B_K == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 || StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma, "Register supply supports static matrix strides");
};
}

namespace opus_bpreshuffle_gfx950::registers::schedule_6 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_5;
using namespace opus_bpreshuffle_gfx950::registers::schedule_2;
// SPDX-License-Identifier: Apache-2.0
}

#endif

#if defined(OPUS_BP_INCLUDE_REGISTER_7) && !defined(OPUS_BP_INCLUDE_REGISTER_7_DONE)
#define OPUS_BP_INCLUDE_REGISTER_7_DONE
namespace opus_bpreshuffle_gfx950::registers::schedule_7 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
int WaveK = 1, int Output = 0, int BCache = 0,
int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
static constexpr int E_M = BlockM / (WaveM * W_M);
static constexpr int E_N = BlockN / (WaveN * W_N);
static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
static constexpr int B_CACHE = BCache, OUTPUT = Output;
static constexpr int FIXED_K = FixedK;
static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
static_assert(Output == 0 || Output == 3 || Output == 4);
static_assert(BCache >= 0 && BCache <= 3);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
static_assert(!PadN || Output == 4);
static_assert(!ReuseBScale || 128 % BlockN == 0);
static_assert(Output != 4 || WaveK > 1);
static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
int WaveK = 1, int Output = 0, int BCache = 0,
int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
static constexpr int E_M = BlockM / (WaveM * W_M);
static constexpr int E_N = BlockN / (WaveN * W_N);
static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
static constexpr int B_CACHE = BCache, OUTPUT = Output;
static constexpr int FIXED_K = FixedK;
static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
static_assert(Output == 0 || Output == 3 || Output == 4);
static_assert(BCache >= 0 && BCache <= 3);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
static_assert(!PadN || Output == 4);
static_assert(!ReuseBScale || 128 % BlockN == 0);
static_assert(Output != 4 || WaveK > 1);
static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_REGISTER_11) && !defined(OPUS_BP_INCLUDE_REGISTER_11_DONE)
#define OPUS_BP_INCLUDE_REGISTER_11_DONE
namespace opus_bpreshuffle_gfx950::registers::schedule_11 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_packed_sfb_traits_gfx950
: BaseTraits {
static_assert(BaseTraits::B_M == 16 && BaseTraits::B_N == 32);
static_assert(BaseTraits::T_M == 1 && BaseTraits::T_N == 1);
static_assert(BaseTraits::PREFETCH == 3 && BaseTraits::WAVE_K == 8);
static_assert(BaseTraits::OUTPUT == 4 && BaseTraits::B_CACHE == 3);
static_assert(BaseTraits::FIXED_K == 7168 && !BaseTraits::N_TAIL);
static_assert(BaseTraits::REUSE_B_SCALE);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_packed_sfb_traits_gfx950
: BaseTraits {
static_assert(BaseTraits::B_M == 16 && BaseTraits::B_N == 32);
static_assert(BaseTraits::T_M == 1 && BaseTraits::T_N == 1);
static_assert(BaseTraits::PREFETCH == 3 && BaseTraits::WAVE_K == 8);
static_assert(BaseTraits::OUTPUT == 4 && BaseTraits::B_CACHE == 3);
static_assert(BaseTraits::FIXED_K == 7168 && !BaseTraits::N_TAIL);
static_assert(BaseTraits::REUSE_B_SCALE);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_REGISTER_15) && !defined(OPUS_BP_INCLUDE_REGISTER_15_DONE)
#define OPUS_BP_INCLUDE_REGISTER_15_DONE
namespace opus_bpreshuffle_gfx950::registers::schedule_15 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0, bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static constexpr bool SFA_WAVEORDER = false;
static_assert((StaticN == 0) == (StaticStrideK == 0));
static_assert(StaticN == 0 || (StaticN > 0 && StaticN % 128 == 0));
static_assert(StaticStrideK == 0 || (StaticStrideK > 0 && StaticStrideK <= 16384 &&
StaticStrideK % BaseTraits::B_K == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 || StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma, "Register supply supports static matrix strides");
};
#else

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0, bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_register_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static constexpr bool SFA_WAVEORDER = false;
static_assert((StaticN == 0) == (StaticStrideK == 0));
static_assert(StaticN == 0 || (StaticN > 0 && StaticN % 128 == 0));
static_assert(StaticStrideK == 0 || (StaticStrideK > 0 && StaticStrideK <= 16384 &&
StaticStrideK % BaseTraits::B_K == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 || StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma, "Register supply supports static matrix strides");
};

#endif
}
#endif
