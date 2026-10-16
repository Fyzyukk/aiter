#ifndef OPUS_BP_LDS_RING_TRAITS_BASE
#define OPUS_BP_LDS_RING_TRAITS_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_panel_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_register_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_0 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN, int WaveM, int WaveN,
int Stages, int Cluster = 1, int Output = 2,
bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
static constexpr int OUTPUT = Output;
static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
static constexpr bool FINE_M_LOADS = FineMLoads;
static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
static constexpr int B_GROUPS = (BlockN + 127) / 128;
static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
(RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16);
static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
static_assert(Output == 1 || Output == 2);
static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
BlockM * BlockN % (BLOCK_SIZE * 8) == 0));
static constexpr int lds_bytes(int k) {
const int loops = (k / B_K + SplitK - 1) / SplitK;
const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
}
};
template<int FixedK = 7168>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_4wave_traits_gfx950 {
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=1, T_N=4, T_K=1;
static constexpr int BLOCK_SIZE=256, NUM_WAVES = 4, WARP_SIZE = 64;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = 12, E_N = 4, E_K = 1;
static constexpr int NUM_STAGES = 4, FIXED_K = FixedK, LOOPS = FixedK / B_K;
static constexpr int A_STAGE = B_M * B_K;
static constexpr int OFF_SA = NUM_STAGES * A_STAGE;
static constexpr int SA_BYTES = 11264, SB_BYTES = 1024;
static constexpr int OFF_SB = OFF_SA + SA_BYTES;
static constexpr int LDS_BYTES = OFF_SB + SB_BYTES + 16;
static constexpr int SPLIT_K = 1;
static_assert(FixedK == 7168 && LOOPS == 56 && LDS_BYTES == 110608);
static_assert(BLOCK_SIZE == T_M * T_N * WARP_SIZE);
static_assert(E_M * E_N * 4 == 192);
static_assert(B_M * (B_N + 8) * 2 <= LDS_BYTES);
static constexpr int lds_bytes(int) { return 0; }
};
}

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_1 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_4;
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_0;
using namespace opus_bpreshuffle_gfx950::registers::schedule_2;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_2 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_b_nt_traits_gfx950
: BaseTraits {
static_assert(!BaseTraits::XOR_LDS);
};
}

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_3 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_1;
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_2;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_4 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0,
bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_supply_traits_gfx950
: BaseTraits {
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static_assert((StaticN == 0 && StaticStrideK == 0) ||
(StaticN > 0 && StaticN % 128 == 0 &&
StaticStrideK > 0 && StaticStrideK <= 16384 &&
StaticStrideK % 128 == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 ||
StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma || !BaseTraits::REGISTER_SCALES);
};
}

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_5 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_1;
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_4;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::lds_ring::schedule_6 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_5;
// SPDX-License-Identifier: Apache-2.0
}

#endif

#if defined(OPUS_BP_INCLUDE_LDS_RING_7) && !defined(OPUS_BP_INCLUDE_LDS_RING_7_DONE)
#define OPUS_BP_INCLUDE_LDS_RING_7_DONE
namespace opus_bpreshuffle_gfx950::lds_ring::schedule_7 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN, int WaveM, int WaveN,
int Stages, int Cluster = 1, int Output = 2,
bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
static constexpr int OUTPUT = Output;
static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
static constexpr bool FINE_M_LOADS = FineMLoads;
static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
static constexpr int B_GROUPS = (BlockN + 127) / 128;
static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
(RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16);
static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
static_assert(Output == 1 || Output == 2);
static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
BlockM * BlockN % (BLOCK_SIZE * 8) == 0));
static constexpr int lds_bytes(int k) {
const int loops = (k / B_K + SplitK - 1) / SplitK;
const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
}
};
template<int FixedK = 7168>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_4wave_traits_gfx950 {
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=1, T_N=4, T_K=1;
static constexpr int BLOCK_SIZE=256, NUM_WAVES = 4, WARP_SIZE = 64;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = 12, E_N = 4, E_K = 1;
static constexpr int NUM_STAGES = 4, FIXED_K = FixedK, LOOPS = FixedK / B_K;
static constexpr int A_STAGE = B_M * B_K;
static constexpr int OFF_SA = NUM_STAGES * A_STAGE;
static constexpr int SA_BYTES = 11264, SB_BYTES = 1024;
static constexpr int OFF_SB = OFF_SA + SA_BYTES;
static constexpr int LDS_BYTES = OFF_SB + SB_BYTES + 16;
static constexpr int SPLIT_K = 1;
static_assert(FixedK == 7168 && LOOPS == 56 && LDS_BYTES == 110608);
static_assert(BLOCK_SIZE == T_M * T_N * WARP_SIZE);
static_assert(E_M * E_N * 4 == 192);
static_assert(B_M * (B_N + 8) * 2 <= LDS_BYTES);
static constexpr int lds_bytes(int) { return 0; }
};
#else

// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN, int WaveM, int WaveN,
int Stages, int Cluster = 1, int Output = 2,
bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
static constexpr int OUTPUT = Output;
static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
static constexpr bool FINE_M_LOADS = FineMLoads;
static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
static constexpr int B_GROUPS = (BlockN + 127) / 128;
static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
(RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16);
static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
static_assert(Output == 1 || Output == 2);
static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
BlockM * BlockN % (BLOCK_SIZE * 8) == 0));
static constexpr int lds_bytes(int k) {
const int loops = (k / B_K + SplitK - 1) / SplitK;
const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
}
};
template<int FixedK = 7168>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_4wave_traits_gfx950 {
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=1, T_N=4, T_K=1;
static constexpr int BLOCK_SIZE=256, NUM_WAVES = 4, WARP_SIZE = 64;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = 12, E_N = 4, E_K = 1;
static constexpr int NUM_STAGES = 4, FIXED_K = FixedK, LOOPS = FixedK / B_K;
static constexpr int A_STAGE = B_M * B_K;
static constexpr int OFF_SA = NUM_STAGES * A_STAGE;
static constexpr int SA_BYTES = 11264, SB_BYTES = 1024;
static constexpr int OFF_SB = OFF_SA + SA_BYTES;
static constexpr int LDS_BYTES = OFF_SB + SB_BYTES + 16;
static constexpr int SPLIT_K = 1;
static_assert(FixedK == 7168 && LOOPS == 56 && LDS_BYTES == 110608);
static_assert(BLOCK_SIZE == T_M * T_N * WARP_SIZE);
static_assert(E_M * E_N * 4 == 192);
static_assert(B_M * (B_N + 8) * 2 <= LDS_BYTES);
static constexpr int lds_bytes(int) { return 0; }
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_LDS_RING_9) && !defined(OPUS_BP_INCLUDE_LDS_RING_9_DONE)
#define OPUS_BP_INCLUDE_LDS_RING_9_DONE
namespace opus_bpreshuffle_gfx950::lds_ring::schedule_9 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_b_nt_traits_gfx950
: BaseTraits {
static_assert(!BaseTraits::XOR_LDS);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_b_nt_traits_gfx950
: BaseTraits {
static_assert(!BaseTraits::XOR_LDS);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_LDS_RING_13) && !defined(OPUS_BP_INCLUDE_LDS_RING_13_DONE)
#define OPUS_BP_INCLUDE_LDS_RING_13_DONE
namespace opus_bpreshuffle_gfx950::lds_ring::schedule_13 {
using namespace opus_bpreshuffle_gfx950::lds_ring::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0,
bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_supply_traits_gfx950
: BaseTraits {
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static_assert((StaticN == 0 && StaticStrideK == 0) ||
(StaticN > 0 && StaticN % 128 == 0 &&
StaticStrideK > 0 && StaticStrideK <= 16384 &&
StaticStrideK % 128 == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 ||
StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma || !BaseTraits::REGISTER_SCALES);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0,
bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_supply_traits_gfx950
: BaseTraits {
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static_assert((StaticN == 0 && StaticStrideK == 0) ||
(StaticN > 0 && StaticN % 128 == 0 &&
StaticStrideK > 0 && StaticStrideK <= 16384 &&
StaticStrideK % 128 == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 ||
StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma || !BaseTraits::REGISTER_SCALES);
};

#endif
}
#endif
