# OPUS GEMM C++ and code generation

The user-facing guide is
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). This document
describes the canonical registry, family-specific C++ entries, generated
exact-kid tables and physical launch validation.

## Architecture and family capability

Kernel identity is always `(arch, logical family, kid, Y dtype)`. A bare kid is
not a cross-architecture capability.

| Logical family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `a16w16` | non-workspace and two-stage workspace | non-workspace and two-stage workspace | two-stage and fused workspace |
| `a8w8` | empty | kid 2, FP32 Y | empty |
| `a8w8_blockscale` | empty | kid 1, FP32 Y, plain WQ | empty |
| `a8w8_blockscale_bpreshuffle` | kid 11000, BF16 Y | explicit empty table | explicit empty table |

Empty A8 tables are valid capability states. The stable public symbol still
exists and reports no registered kernel for the current architecture rather
than trying another architecture's table.

## Canonical C++ entries

```cpp
void opus_gemm_a16w16_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);

void opus_gemm_a8w8_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    int kid);

void opus_gemm_a8w8_blockscale_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    int kid);

void opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    aiter_tensor_t& Y,
    int kid);
```

The production Python a16w16 path enters the same launcher through the exported
status-returning C ABI `opus_gemm_a16w16_launch_cabi`. Optional tensors are
nullable pointers and the caller supplies the live HIP stream. The wrapper
checks integer ranges, switches/restores the XQ HIP device and thread-local
stream, bridges exceptions through thread-local error text, and then calls
`opus_gemm_a16w16_launch`; it owns no policy, dispatch table, Tensor, pointer,
or workspace. The original pybind canonical entry remains available for
compatibility and A/B testing.

The C++ entries never choose a default kid, query CSV data, allocate a Tensor
or run a shape heuristic. They validate the common device/family/dtype surface,
select only the current runtime architecture and output-dtype table, then
perform strict exact-kid lookup.

Three failure classes remain distinct:

1. the module was not built for the runtime architecture;
2. the selected architecture/family/dtype table has no registered kernel;
3. a non-empty table does not contain the requested kid.

## Runtime policy and build-time CSV

All runtime policy is Python-owned:

```text
explicit -> OPUS tuned row -> Python heuristic/default -> fallback/error
```

Build-time CSV processing remains intentionally separate. `gen_instances.py`
extracts OPUS `solidx`/`kernelId` values and combines them with the sidecar,
Python heuristic-default kids and mandatory A8 kids to form the subset compile
set. Shape rows do not become C++ runtime data.

Current mandatory A8 kids are gfx950 1/2 and gfx942 11000. The generator
asserts that every Python a16 heuristic default is also compiled.

## Generated exact-kid tables

Generated roots are:

```text
opus_gemm_a16w16_kid_dispatch.h
opus_gemm_a8w8_kid_dispatch.h
opus_gemm_manifest.h
opus_build_archs.h
```

A16 macros are:

```text
GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_<ARCH>_<DTYPE>
GENERATE_A16W16_WORKSPACE_KID_DISPATCH_<ARCH>
```

A8 macros are:

```text
GENERATE_A8W8_NOSCALE_KID_DISPATCH_GFX950
GENERATE_A8W8_BLOCKSCALE_KID_DISPATCH_GFX950
GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_<ARCH>_<DTYPE>
```

Every base macro has a `_SIZE`. Rows are sorted by kid and use one typed
function-pointer ABI per table. Empty tables use `std::array<Entry,0>` and do
not reference an absent launcher symbol.

The full canonical A16 table sizes are:

| Architecture | Non-workspace BF16 | Non-workspace FP32 | Workspace |
|---|---:|---:|---:|
| gfx942 | 14 | 1 | 8 |
| gfx950 | 92 | 92 | 48 |
| gfx1250 | 0 | 0 | 1874 |

## A16 requested/actual kid and workspace

The Python selector resolves requested kid to actual kid before workspace
allocation. This is required for the gfx942 non-exact-N redirects
`10210 -> 10200` and `10213 -> 10203`; kid 10216 rejects non-exact N.

The A16 function-pointer ABIs remain separate:

```cpp
using OpusA16W16Kernel = void (*)(
    XQ, WQ, Y, optional_bias, split_k);

using OpusA16W16WorkspaceKernel = void (*)(
    XQ, WQ, Y, workspace, optional_bias, split_k);
```

The router first performs workspace-table membership. A hit requires a
workspace Tensor; a miss requires `workspace=None` before the non-workspace
table is queried.

Task1 workspace ownership remains entirely in Torch. Let
`padded_M=ceil_div(M,B_M)*B_M` and
`padded_N=ceil_div(N,B_N)*B_N`:

| Architecture/family | Physical Tensor shape | Storage distribution |
|---|---|---|
| gfx950 two-stage | `[allocation_split_k, batch, padded_M, padded_N]` | 48 FP32 kids |
| gfx942 two-stage | `[allocation_split_k, batch, padded_M, padded_N]` | 3 BF16 + 5 FP32 kids |
| gfx1250 two-stage | `[allocation_split_k, padded_M, padded_N]` | 496 BF16 kids |
| gfx1250 fused | `[tiles_m, tiles_n, fuse_split_k - 1, B_M, B_N]` | 780 BF16 + 598 FP32 kids, 1378 total |

gfx1250 fused is an active registry/codegen family. It uses the exact kid's
compile-time `fuse_split_k`; a runtime `split_k` cannot change its workspace
shape or launch schedule.

Each generated workspace launcher repeats the final physical checks after its
local split clamp:

- XQ/WQ/Y shape, dtype, stride and batch contract;
- workspace device, exact instance dtype and contiguous storage;
- pointer alignment;
- checked extent and byte-span arithmetic;
- final required element capacity;
- bias rules owned by that exact architecture and kid.

C++ neither allocates nor retains the workspace Tensor or pointer.

## A8 validation ownership

The public C++ router checks only rules common to the logical family: GPU
device agreement, FP8 XQ/WQ, output-dtype table selection and exact kid. Scale
arguments are mandatory references for both blockscale APIs.

Architecture-specific generated launchers own physical rules.

### gfx950 no-scale kid 2

- XQ/WQ/Y are matching 3D `[B,M,K]`, `[B,N,K]`, `[B,M,N]` tensors;
- XQ/WQ are FP8 and K-contiguous; Y is FP32 and contiguous;
- `ceil_div(K,B_K) >= 2`, K-tile loops are even, and K is even.

### gfx950 plain-WQ blockscale kid 1

- the no-scale tensor contract still applies;
- x/w scales are both FP32, contiguous and on the XQ device;
- N and K follow the 128 group contract;
- x scale is `[B,M,K/128]`, w scale is `[B,N/128,K/128]`, with 2D forms
  accepted for batch one;
- the host launcher rejects one K tile or an odd K-tile count before the
  device prefetch pipeline runs;
- this ABI has no bias or `group_layout` parameter.

### gfx942 blockscale-bpreshuffle kid 11000

- XQ/WQ/Y may be 2D or explicit 3D, but batch must be one;
- XQ/WQ are FP8, Y is BF16, and scales are FP32;
- N and K are divisible by 128;
- tensors, WQ shape/stride and scale shapes are explicitly checked;
- x scale has shape `[M,K/128]` with the required transposed physical storage;
- w scale is row-major `[N/128,K/128]`.

WQ bpreshuffle is a content semantic. Metadata checks cannot prove that the
bytes were produced by `shuffle_weight(..., layout=(16,16))`; callers and
numerical tests must supply genuinely shuffled storage.

The public bpreshuffle router contains none of the gfx942-only BF16, FP32
scale, batch-one or 128-tile rules. Future gfx950/gfx1250 implementations can
populate their own registry/emitter/table without changing the stable public
ABI.

## gfx942 direct workspace pointer

gfx942 device code continues to uniformize both halves of the direct 64-bit
workspace pointer with `__builtin_amdgcn_readfirstlane` in main and reduce
kernels. The obsolete handle indirection is absent, but wave-uniform address
semantics remain part of the verified Task1 implementation.

## Source layout

| Path | Role |
|---|---|
| `opus_gemm.cu` | Family routers, common validation and strict current-arch dispatch |
| `opus_gemm_common.py` | Canonical registry, family mappings and compile invariants |
| `gen_instances.py` | Cross-architecture generation, manifests and typed dispatch tables |
| `codegen/gen_instances_gfx*.py` | Exact-instance host launcher and physical validation emission |
| `include/opus_gemm_common.cuh` | Checked extent/workspace helpers |
| `include/gfx*/opus_gemm_arch_*.cuh` | Per-arch sorted exact-kid tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | Kernel argument and trait definitions |
| `include/gfx*/**/splitk_reduce*.cuh` | Architecture-specific reduce paths |

## Generation and tests

```bash
GPU_ARCHS='gfx942;gfx950;gfx1250' \
python csrc/opus_gemm/gen_instances.py -w /tmp/opus-generated

pytest -q \
  op_tests/test_opus_interfaces.py \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py
```

Generation must be byte-stable across repeated runs. Single-arch builds must
emit only that architecture's symbols, while every public bpreshuffle table
slot still has a valid zero-size representation where the capability is empty.
