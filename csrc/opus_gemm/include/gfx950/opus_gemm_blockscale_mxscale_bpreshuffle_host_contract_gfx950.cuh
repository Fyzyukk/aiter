
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#ifndef OPUS_BP_FORMAL_HOST_CONTRACT
#define OPUS_BP_FORMAL_HOST_CONTRACT
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "opus_gemm_common.cuh"
#include <optional>
namespace opus_bpreshuffle_host {
struct ShapeContract { int64_t m,n,k; uint64_t workspace; };
inline aiter_tensor_t make_tensor(void* pointer, AiterDtype dtype,
    std::initializer_list<int64_t> dimensions, std::initializer_list<int64_t> strides) {
    aiter_tensor_t tensor{};
    tensor.ptr=pointer; tensor.dtype_=dtype; tensor.ndim=dimensions.size(); tensor.numel_=1;
    int i=0;
    for(auto dimension:dimensions) { tensor.shape[i++]=dimension; tensor.numel_*=dimension; }
    i=0; for(auto stride:strides) tensor.strides[i++]=stride;
    HIP_CALL(hipGetDevice(&tensor.device_id));
    return tensor;
}
template<size_t Count>
inline uint64_t validate(const ShapeContract (&contracts)[Count],
    aiter_tensor_t& x, aiter_tensor_t& w, aiter_tensor_t& sfa, aiter_tensor_t& sfb,
    aiter_tensor_t& out, std::optional<aiter_tensor_t>& workspace) {
    constexpr auto entry="opus_gemm_gfx950_bpreshuffle";
    AITER_CHECK((x.dim()==2 || x.dim()==3) && w.dim()==x.dim() && out.dim()==x.dim(),entry,": matrix ranks differ");
    AITER_CHECK(x.dim()==2 || (x.size(0)==1 && w.size(0)==1 && out.size(0)==1),entry,": requires batch=1");
    const int64_t m=x.size(-2),n=w.size(-2),k=x.size(-1);
    const ShapeContract* contract=nullptr;
    for(const auto& shape:contracts) if(shape.m==m && shape.n==n && shape.k==k) {contract=&shape;break;}
    AITER_CHECK(contract,entry,": shape is outside this configuration");
    AITER_CHECK(x.dtype()==AITER_DTYPE_fp8 && w.dtype()==AITER_DTYPE_fp8 && out.dtype()==AITER_DTYPE_bf16,entry,": requires FP8/BF16");
    AITER_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() && w.size(-1)==k && out.size(-2)==m && out.size(-1)==n,entry,": matrix layout differs");
    const auto e8m0=[](const aiter_tensor_t& t) {return t.dtype()==AITER_DTYPE_fp8_e8m0 || t.dtype()==AITER_DTYPE_u8;};
    AITER_CHECK(e8m0(sfa) && e8m0(sfb) && sfa.dim()==2 && sfb.dim()==2 && sfa.size(0)==m && sfa.size(1)==k/128 && sfa.stride(0)==1 && sfa.stride(1)==m && sfb.size(0)==n/128 && sfb.size(1)==k/128 && sfb.is_contiguous(),entry,": scale layout differs");
    AITER_CHECK(bool(contract->workspace)==workspace.has_value(),entry,": workspace presence differs");
    if(contract->workspace) opus_validate_workspace(workspace.value(),out,AITER_DTYPE_fp32,(contract->workspace+3)/4,16,entry);
    return contract->workspace;
}
}
#endif
#endif
#ifndef OPUS_BP_HOST_CONTRACT_PIPELINE_BASE
#define OPUS_BP_HOST_CONTRACT_PIPELINE_BASE
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"

#endif

#if defined(OPUS_BP_INCLUDE_HOST_CONTRACT_0) && !defined(OPUS_BP_INCLUDE_HOST_CONTRACT_0_DONE)
#define OPUS_BP_INCLUDE_HOST_CONTRACT_0_DONE
namespace opus_bpreshuffle_gfx950::host_contract::schedule_0 {
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0




struct FixedMContract {
int m, n, k, split, reduce_vec, reduce_block, sfb_alignment;
bool pad_n;
};
static bool fixed_m_overlap(const void* a, uint64_t as, const void* b, uint64_t bs) {
const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
return aa >= bb ? aa - bb < bs : bb - aa < as;
}
static int fixed_m_validate(const FixedMContract& c, int variant,
const void* a, const void* b, void* out,
const void* sfa, const void* sfb, void* workspace,
uint64_t workspace_bytes, int m, int n, int k) {
if (variant < 0 || variant > 1) return -1;
if (m != c.m || n != c.n || k != c.k || m <= 0 || n <= 0 || k <= 0 || n % 128 || k % 128) return -2;
if (!a || !b || !out || !sfa || !sfb) return -3;
const uint64_t um = static_cast<uint64_t>(m), un = static_cast<uint64_t>(n), uk = static_cast<uint64_t>(k);
const uint64_t ab = um * uk, bb = un * uk, elements = um * un;
const uint64_t cb = elements * 2, sab = um * (uk / 128), sbb = (un / 128) * (uk / 128);
const uint64_t partition_bytes = elements * (c.split > 1 ? 4 : 2);
if (ab > INT_MAX || bb > INT_MAX || partition_bytes > INT_MAX ||
sab > INT_MAX || sbb > INT_MAX || elements > INT_MAX) return -4;
if ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(sfa)) % 16) return -5;
if (reinterpret_cast<uintptr_t>(sfb) % c.sfb_alignment) return -6;
const void* ptrs[] = {a, b, out, sfa, sfb};
const uint64_t lengths[] = {ab, bb, cb, sab, sbb};
for (int i = 0; i < 5; ++i)
for (int j = i + 1; j < 5; ++j)
if (fixed_m_overlap(ptrs[i], lengths[i], ptrs[j], lengths[j])) return -7;
if (c.split > 1) {
const uint64_t required = static_cast<uint64_t>(c.split) * elements * 4;
if (!workspace || workspace_bytes < required || reinterpret_cast<uintptr_t>(workspace) % 16) return -8;
for (int i = 0; i < 5; ++i)
if (fixed_m_overlap(workspace, required, ptrs[i], lengths[i])) return -9;
} else if (workspace || workspace_bytes) {
return -10;
}
return 0;
}
static opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 fixed_m_args(
const FixedMContract& c, const void* a, const void* b, void* out,
const void* sfa, const void* sfb, void* workspace) {
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
const uint64_t m = c.m, n = c.n, k = c.k;
args.ptr_a = a; args.ptr_b = b; args.ptr_c = c.split > 1 ? workspace : out;
args.ptr_sfa = sfa; args.ptr_sfb = sfb;
args.m = c.m; args.n = c.n; args.k = c.k; args.batch = 1;
args.stride_a = c.k; args.stride_b = c.k; args.stride_c = c.n;
args.stride_a_batch = static_cast<int>(m * k);
args.stride_b_batch = static_cast<int>(n * k);
args.stride_c_batch = static_cast<int>(m * n);
args.stride_sfa = c.m; args.stride_sfb = c.k / 128;
args.stride_sfa_batch = static_cast<int>(m * (k / 128));
args.stride_sfb_batch = static_cast<int>((n / 128) * (k / 128));
return args;
}
#else

// SPDX-License-Identifier: Apache-2.0




struct FixedMContract {
int m, n, k, split, reduce_vec, reduce_block, sfb_alignment;
bool pad_n;
};
static bool fixed_m_overlap(const void* a, uint64_t as, const void* b, uint64_t bs) {
const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
return aa >= bb ? aa - bb < bs : bb - aa < as;
}
static int fixed_m_validate(const FixedMContract& c, int variant,
const void* a, const void* b, void* out,
const void* sfa, const void* sfb, void* workspace,
uint64_t workspace_bytes, int m, int n, int k) {
if (variant < 0 || variant > 1) return -1;
if (m != c.m || n != c.n || k != c.k || m <= 0 || n <= 0 || k <= 0 || n % 128 || k % 128) return -2;
if (!a || !b || !out || !sfa || !sfb) return -3;
const uint64_t um = static_cast<uint64_t>(m), un = static_cast<uint64_t>(n), uk = static_cast<uint64_t>(k);
const uint64_t ab = um * uk, bb = un * uk, elements = um * un;
const uint64_t cb = elements * 2, sab = um * (uk / 128), sbb = (un / 128) * (uk / 128);
const uint64_t partition_bytes = elements * (c.split > 1 ? 4 : 2);
if (ab > INT_MAX || bb > INT_MAX || partition_bytes > INT_MAX ||
sab > INT_MAX || sbb > INT_MAX || elements > INT_MAX) return -4;
if ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(sfa)) % 16) return -5;
if (reinterpret_cast<uintptr_t>(sfb) % c.sfb_alignment) return -6;
const void* ptrs[] = {a, b, out, sfa, sfb};
const uint64_t lengths[] = {ab, bb, cb, sab, sbb};
for (int i = 0; i < 5; ++i)
for (int j = i + 1; j < 5; ++j)
if (fixed_m_overlap(ptrs[i], lengths[i], ptrs[j], lengths[j])) return -7;
if (c.split > 1) {
const uint64_t required = static_cast<uint64_t>(c.split) * elements * 4;
if (!workspace || workspace_bytes < required || reinterpret_cast<uintptr_t>(workspace) % 16) return -8;
for (int i = 0; i < 5; ++i)
if (fixed_m_overlap(workspace, required, ptrs[i], lengths[i])) return -9;
} else if (workspace || workspace_bytes) {
return -10;
}
return 0;
}
static opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 fixed_m_args(
const FixedMContract& c, const void* a, const void* b, void* out,
const void* sfa, const void* sfb, void* workspace) {
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
const uint64_t m = c.m, n = c.n, k = c.k;
args.ptr_a = a; args.ptr_b = b; args.ptr_c = c.split > 1 ? workspace : out;
args.ptr_sfa = sfa; args.ptr_sfb = sfb;
args.m = c.m; args.n = c.n; args.k = c.k; args.batch = 1;
args.stride_a = c.k; args.stride_b = c.k; args.stride_c = c.n;
args.stride_a_batch = static_cast<int>(m * k);
args.stride_b_batch = static_cast<int>(n * k);
args.stride_c_batch = static_cast<int>(m * n);
args.stride_sfa = c.m; args.stride_sfb = c.k / 128;
args.stride_sfa_batch = static_cast<int>(m * (k / 128));
args.stride_sfb_batch = static_cast<int>((n / 128) * (k / 128));
return args;
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_HOST_CONTRACT_1) && !defined(OPUS_BP_INCLUDE_HOST_CONTRACT_1_DONE)
#define OPUS_BP_INCLUDE_HOST_CONTRACT_1_DONE
namespace opus_bpreshuffle_gfx950::host_contract::schedule_1 {
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0



struct ReducerStoreContract { int m, n, k, split, vec, block; };
static bool reducer_store_overlap(const void* a, uint64_t as, const void* b, uint64_t bs) {
const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
return aa >= bb ? aa - bb < bs : bb - aa < as;
}
static int reducer_store_validate(const ReducerStoreContract& c, int variant,
const void* a, const void* b, void* out, const void* sfa, const void* sfb,
void* workspace, uint64_t workspace_bytes, int m, int n, int k) {
if (variant < 0 || variant > 1) return -1;
if (m != c.m || n != c.n || k != c.k || m <= 0 || n <= 0 || k <= 0 ||
n % 128 || k % 128 || c.split <= 1) return -2;
if (!a || !b || !out || !sfa || !sfb || !workspace) return -3;
const uint64_t um = m, un = n, uk = k, elements = um * un;
const uint64_t ab = um * uk, bb = un * uk, cb = elements * 2;
const uint64_t sab = um * (uk / 128), sbb = (un / 128) * (uk / 128);
const uint64_t required = static_cast<uint64_t>(c.split) * elements * 4;
if (ab > INT_MAX || bb > INT_MAX || elements * 4 > INT_MAX ||
sab > INT_MAX || sbb > INT_MAX || elements + c.vec * c.block - 1 > INT_MAX ||
elements % c.vec) return -4;
if ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(sfa) |
reinterpret_cast<uintptr_t>(workspace)) % 16) return -5;
if (workspace_bytes < required) return -6;
const void* pointers[] = {a, b, out, sfa, sfb, workspace};
const uint64_t lengths[] = {ab, bb, cb, sab, sbb, required};
for (int i = 0; i < 6; ++i) {
if (lengths[i] > UINTPTR_MAX - reinterpret_cast<uintptr_t>(pointers[i])) return -7;
for (int j = i + 1; j < 6; ++j)
if (reducer_store_overlap(pointers[i], lengths[i], pointers[j], lengths[j])) return -8;
}
return 0;
}
static opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 reducer_store_args(
const ReducerStoreContract& c, const void* a, const void* b,
const void* sfa, const void* sfb, void* workspace) {
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = a; args.ptr_b = b; args.ptr_c = workspace;
args.ptr_sfa = sfa; args.ptr_sfb = sfb;
args.m = c.m; args.n = c.n; args.k = c.k; args.batch = 1;
args.stride_a = c.k; args.stride_b = c.k; args.stride_c = c.n;
args.stride_a_batch = c.m * c.k; args.stride_b_batch = c.n * c.k;
args.stride_c_batch = c.m * c.n;
args.stride_sfa = c.m; args.stride_sfb = c.k / 128;
args.stride_sfa_batch = c.m * (c.k / 128);
args.stride_sfb_batch = (c.n / 128) * (c.k / 128);
return args;
}
#else

// SPDX-License-Identifier: Apache-2.0



struct ReducerStoreContract { int m, n, k, split, vec, block; };
static bool reducer_store_overlap(const void* a, uint64_t as, const void* b, uint64_t bs) {
const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
return aa >= bb ? aa - bb < bs : bb - aa < as;
}
static int reducer_store_validate(const ReducerStoreContract& c, int variant,
const void* a, const void* b, void* out, const void* sfa, const void* sfb,
void* workspace, uint64_t workspace_bytes, int m, int n, int k) {
if (variant < 0 || variant > 1) return -1;
if (m != c.m || n != c.n || k != c.k || m <= 0 || n <= 0 || k <= 0 ||
n % 128 || k % 128 || c.split <= 1) return -2;
if (!a || !b || !out || !sfa || !sfb || !workspace) return -3;
const uint64_t um = m, un = n, uk = k, elements = um * un;
const uint64_t ab = um * uk, bb = un * uk, cb = elements * 2;
const uint64_t sab = um * (uk / 128), sbb = (un / 128) * (uk / 128);
const uint64_t required = static_cast<uint64_t>(c.split) * elements * 4;
if (ab > INT_MAX || bb > INT_MAX || elements * 4 > INT_MAX ||
sab > INT_MAX || sbb > INT_MAX || elements + c.vec * c.block - 1 > INT_MAX ||
elements % c.vec) return -4;
if ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(sfa) |
reinterpret_cast<uintptr_t>(workspace)) % 16) return -5;
if (workspace_bytes < required) return -6;
const void* pointers[] = {a, b, out, sfa, sfb, workspace};
const uint64_t lengths[] = {ab, bb, cb, sab, sbb, required};
for (int i = 0; i < 6; ++i) {
if (lengths[i] > UINTPTR_MAX - reinterpret_cast<uintptr_t>(pointers[i])) return -7;
for (int j = i + 1; j < 6; ++j)
if (reducer_store_overlap(pointers[i], lengths[i], pointers[j], lengths[j])) return -8;
}
return 0;
}
static opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 reducer_store_args(
const ReducerStoreContract& c, const void* a, const void* b,
const void* sfa, const void* sfb, void* workspace) {
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = a; args.ptr_b = b; args.ptr_c = workspace;
args.ptr_sfa = sfa; args.ptr_sfb = sfb;
args.m = c.m; args.n = c.n; args.k = c.k; args.batch = 1;
args.stride_a = c.k; args.stride_b = c.k; args.stride_c = c.n;
args.stride_a_batch = c.m * c.k; args.stride_b_batch = c.n * c.k;
args.stride_c_batch = c.m * c.n;
args.stride_sfa = c.m; args.stride_sfb = c.k / 128;
args.stride_sfa_batch = c.m * (c.k / 128);
args.stride_sfb_batch = (c.n / 128) * (c.k / 128);
return args;
}

#endif
}
#endif
