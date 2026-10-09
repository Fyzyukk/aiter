# OPUS GEMM C++ and code generation

The public Python contract is documented in
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). C++ keeps family launch ABIs and a bpreshuffle workspace entry. They are shared private implementation boundaries for the
Python `opus_gemm(..., kid=...)` and `opus_bmm(..., kid=...)` entries; the
public operation split does not duplicate C++ launchers or kernels.

MXFP8 B-preshuffle is organized as **five parameterized compute pipelines**:
`pin`, `tiled`, `register`, `lds`, and `large_output`. The scalar
[configuration catalog](opus_gemm_bpreshuffle_config.py) exposes pipeline names
and tile/wave/stage/load-policy parameters. Normal tuning retains 89 compile
configurations; 105 numeric IDs remain as internal compatibility launcher keys,
including 16 historical entries. A compile configuration is a specialization
of a pipeline, rather than another source file or public pipeline candidate.

The [five-pipeline report](../../reports/opus_pipeline5_20261009/README.md) and
[configuration map](../../reports/opus_pipeline5_20261009/configurations.csv)
record the current layout. The earlier
[runtime split-K](../../reports/opus_runtime_splitk_20261009/README.md) and
[92-configuration registration](../../reports/opus_register92_20261009/README.md)
reports are frozen evidence for earlier implementations. GPU tests remain
stopped; the refactor has no new numerical or performance result.

## Runtime global split-K

Register IDs 92310, 92311, 92320, 92321, 92330, and 92340, and fine LDS IDs
92410, 92420, and 92430 use a shape-dependent launch plan. The framework resolves
the split count once for the producer grid, dynamic LDS, workspace, and reducer.
Tile geometry, MFMA layout, operand queues, and local WaveK remain compile-time
parameters. The six register geometries originally all used global split four;
their tile/local-WaveK differences remain distinct configurations.

For those nine IDs, `split_k` and the tuned CSV `splitK` column have these meanings:

| Value | Behavior |
|---|---|
| `1..16` | Literal global partition count, at most `K / 128`, subject to the shape and byte/workspace limits |
| `0` | Historical default: one for fine LDS and four for register; short-K register defaults may include empty partitions |
| `-1` | Optional grid/CU heuristic calculated from M/N/K and the supplied CU count; this policy is not a measured tuning result |

Split one writes BF16 directly. Larger splits write one FP32 workspace plane
per global partition after its local-wave sum, and the shared runtime reducer
converts the final FP32 sum to BF16. The fine producer compiles two output modes;
the register producer compiles once per static geometry. There is no producer
specialization for each runtime split count. Fine IDs 92411, 92421, and 92431
retain their fixed-split-two compatibility paths.

The tuner enumerates every legal positive split for a runtime ID and records
`(kid, splitK)` for exact replay. Kernel registration count and launch-parameter
search count are different: reducing split-only IDs keeps the launch choices
available for tuning. Other fixed-split candidates retain `splitK=0` replay.

The [frozen header checks](../../reports/opus_runtime_splitk_20261009/headers/receipt.json)
passed offline compilation and host syntax for fourteen runtime entries, with
zero scratch/spills, and CPU partition/layout/LDS checks. All 105 actual generated HIP TUs were freshly compiled; fused host, actual router, and the complete pybind TU also compiled. The bpreshuffle shared link with `--no-undefined` and all 127 host launch references passed. The [build scope](../../reports/opus_runtime_splitk_20261009/full_build/verification_scope.json) records the separate new-binding symbol check and zero scratch/spills for the runtime kernels. GPU tests remain stopped; these checks provide no new performance result.

The [historical gfx950 results](../../reports/opus_current745_tables_20260930/REPORT.md)
list the then-current 26 default MXFP8 B-preshuffle candidates and their win counts, with
per-shape timings in milliseconds against the original CSV. The
[same-run comparison](../../reports/opus_current745_tables_20260930/SAME_RUN.md)
compares OPUS with the fastest valid CK/CKTile/ASM candidate for all 745 shapes.
The reports also quantify timing changes for matching historical configurations.

## MXFP8 B-preshuffle pipeline and traits headers

The current device source lives in `include/gfx950/`. All geometry and schedule
traits are in [one traits header](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh).
Five compute headers replace the previous 24 compute headers; the complete
B-preshuffle header set, including layout/ABI/output/reduction helpers, is
11 files instead of 47. Compute source shrank from 8,223 to 3,015 lines.

| Pipeline | Default compile configurations | Device source | Retained policies |
|---|---:|---|---|
| `pin` | 15 | [pin](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_pin_gfx950.cuh) | AGPR placement, padded M, fixed K, panel size, scale reset, unroll |
| `tiled` | 20 | [tiled](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_tiled_gfx950.cuh) | Main/narrow geometry, stage count, scale representation, wait schedule, tile ordering |
| `register` | 13 | [register](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_gfx950.cuh) | Register queues, local K waves, fixed/runtime K, output/tail policy, runtime global split |
| `lds` | 38 | [lds](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_lds_gfx950.cuh) | A LDS ring, B LDS/direct registers, queue depth, scale/XOR/drain policy, fine M, split |
| `large_output` | 3 | [large_output](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_large_output_gfx950.cuh) | Wide C addressing, panel size, direct B, chunked output |

Each pipeline has one producer entry name. Register and LDS have typed legacy
and runtime-kargs overloads. Internal `if constexpr` policies preserve different
load and wait schedules without separate global kernels per schedule. Shared
layouts are in the [base layout helper](include/gfx950/opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh)
and [tiled layout helper](include/gfx950/opus_gemm_mxscale_bpreshuffle_tiled_layout_gfx950.cuh).
A [shared runtime reducer](include/gfx950/opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh)
handles FP32 partial planes. Historical traits names are retained within the
single traits file for saved launcher configurations.

This follows FlyDSL gfx950 BMM's source organization:
[one JIT factory and kernel body](../../aiter/ops/flydsl/kernels/bmm_a8w8_mxscale_gfx950.py)
accept compile-time tile/wave/buffer/B-load parameters. Its
[host wrapper](../../aiter/ops/flydsl/batched_gemm_a8w8_gfx950.py) selects and caches
specializations. Its tuned configurations number in the hundreds; its seven
fallback M tiers do not limit it to seven compiled configurations. FlyDSL BMM's
split count is also compile-time, whereas these nine OPUS runtime configurations
accept a launch-time global split count.

The public [Python wrapper](../../aiter/ops/opus/README.md) accepts pipeline and
configuration parameters. Numeric IDs are resolved before the unchanged
internal ABI. Unsupported or ambiguous parameter combinations are rejected;
this catalog currently exposes the registered configurations and does not JIT
an arbitrary new tuple supplied by a caller.

All 105 configurations compiled offline, fused host/router/pybind compiled,
and the shared link resolved all 127 distinct launch references. The
[verification scope](../../reports/opus_pipeline5_20261009/full_build/verification_scope.json)
records changed instruction/register allocations and the pending GPU checks.
Earlier source layouts and validation history are preserved in the
[pre-refactor README](../../reports/opus_pipeline5_20261009/before/csrc/opus_gemm/README.md)
and their original reports.

## Current MXFP8 tuning entry

Use [`opus_gemm_mxscale_bpreshuffle_tune.py`](opus_gemm_mxscale_bpreshuffle_tune.py)
from the checkout root. The latest remote setup, source map, full-shape command,
and replay instructions are at the top of [HANDOFF_MXFP8.md](../../HANDOFF_MXFP8.md).
For all 745 gfx950/256-CU shapes:

The current compiler policy uses the minimal local Clang 23 at
`/opt/rocm-llvm23-46fcb339/bin` for CK, CKTile, the ASM host wrapper, and
74 unpinned active OPUS configurations. Set both `HIP_CLANG_PATH` and
`OPUS_BASELINE_HIP_CLANG_PATH` to this directory. The 15 explicit pin-AGPR
configurations (9000/9001/9010/9011 and 92100–92104/92110–92114/92120)
use `OPUS_HIP_CLANG_PATH`, the verified pin-AGPR branch
`yuyzhang512/llvm-project` at `49c41889681640665400cb01c9fbb4c0a024cde4`
(Clang 24). ASM device kernels retain their precompiled code. Compiler probes
and resource-header settings are restored after the OPUS build, including on
failure. With Clang 24 and ROCm 7.0, pin kernels use installed ROCm
resource headers or `OPUS_HIP_RESOURCE_DIR`; the baseline compiler uses its
own headers or `OPUS_BASELINE_HIP_RESOURCE_DIR`. Shared unpinned layout helpers
are independent of pin-AGPR declarations.

Use a fresh `AITER_JIT_DIR` when changing a backend's compiler; existing
libraries are not recompiled merely by changing an environment variable.
Without `OPUS_BASELINE_HIP_CLANG_PATH`, the previous all-OPUS pin-compiler
build remains available. The original upstream tune compiler is unconfirmed;
Clang 23 establishes a new measured baseline.

```bash
ROCR_VISIBLE_DEVICES=0 \
HIP_CLANG_PATH=/opt/rocm-llvm23-46fcb339/bin \
OPUS_BASELINE_HIP_CLANG_PATH=/opt/rocm-llvm23-46fcb339/bin \
OPUS_HIP_CLANG_PATH=/absolute/path/to/llvm-pin-build/bin \
AITER_JIT_DIR=/absolute/path/to/fresh-jit-dir \
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  -i aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv \
  -o /tmp/dsv4_opus_tuned.csv -o2 /tmp/dsv4_opus_profile.csv \
  --libtype all --splitK --shape_grouped --mp 1 --warmup 5 --iters 51 --all
```

The [complete tuned CSV](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv)
contains the latest 745-shape compiler-split all-backend results from
[tuned_all_config.csv](../../reports/opus_clang23_mixed_retune_20261008/tuned_all_config.csv):
693 OPUS, 8 CK, 10 CKTile, and 34 ASM rows.
OPUS uses native E8M0 scales; CK/CKTile/ASM use FP32 scales; all outputs are BF16.
The production table reports bandwidth in TB/s; raw reports retain GB/s.
The 13-column schema is retained. See the [complete measured report](../../reports/opus_clang23_mixed_retune_20261008/README.md)
for compiler receipts, same-run comparison, historical timings, and raw accuracy results.
The original upstream tune compiler remains unconfirmed; this Clang23 run is a new baseline.

The earlier [Clang20 external-backend run](../../reports/opus_retune28_20261008/README.md)
is retained as historical evidence; its anomalous CKTile timings and mixed
five-round/screening output do not define the current table. The fresh table
uses the complete new screening profile for all 745 selections.

The table can be passed to `-i` for shape enumeration, which ignores its timings
and candidate choices, or to `--run_config` for explicit backend replay. The tuner
does not publish native E8M0 results into the FP32-scale production dispatcher.
`-o` saves the fastest valid candidate per shape and `-o2` saves the candidate
profile. OPUS candidates are compiled before the sweep; external backends use
their existing JIT paths. `--shape_grouped` measures each shape's candidates
on the same GPU. Use `--opus_pipelines register,lds` to restrict the five
pipeline candidates; each selected pipeline enumerates its supported compile
configurations and legal runtime split counts. The legacy `--opus-kids` and
`--opus-families` filters remain accepted for tuning and debugging.
`--run_config` replays the saved rows and ignores candidate filters, including
`--opus_pipelines`.

New `-o` and `-o2` files add `pipeline` and canonical JSON `config` columns.
Runtime split remains in `splitK`, and `kernelId` remains a compatibility ABI
key. Old CSVs acquire metadata when read; new metadata is validated against
the exact registered configuration before compilation. The measured production
CSV remains the existing 13-column historical dataset. Rebuild the OPUS JIT
after changing the kernel sources; historical binaries do not become the new
implementation merely because their IDs are unchanged.

| Default ID | Configurations consolidated into it |
|---:|---|
| 9041 | 9041 / 9050, register-prefetch depth |
| 9042 | runtime-K fallback plus 9071 / 9073 N48 tiles at K=7168 |
| 9044 | 9044 / 9056, LDS queue depth |
| 9047 | 9048 removed from default tuning; 9047 retained |
| 9052 | runtime-K fallback plus 9070 at K=7168 |
| 9053 | runtime-K fallback plus 9072 at K=7168 |
| 9062 | 9062 / 9064, two FP32 partitions, M80 / M96 tiles |
| 9063 | 9063 / 9065–9069, four FP32 partitions, M48–M128 tiles |
| 92410 | 48×64 fine geometry; 92411 retains the fixed-split-two compatibility path |
| 92420 | 64×128 fine geometry; 92421 retains the fixed-split-two compatibility path |
| 92430 | 96×128 fine geometry; 92431 retains the fixed-split-two compatibility path |

9043–9046, 9055 and 9060–9063 accept M <= 2048; the other default 904x/905x
configurations accept M <= 512. 9000/9010 retain their configurations in the shared pin pipeline.

An earlier migration, before the five-pipeline refactor, mapped then-current
9020 to 9010 and 9060–9064 to 9020–9024; 9000 stayed unchanged. Its historical
CSVs and binaries retain those earlier meanings. This refactor does not
renumber the current 105 compatibility IDs. Use the current production CSV
and rebuild the OPUS JIT after source changes.

CK/CKTile/ASM reuse the original blockscale tuner's input generator: FP16
uniform random operands divided by 10 and cast to FP8, plus independently
random FP32 scales. OPUS uses the same operand generation with independently
random native E8M0 scales. External backends do not decode OPUS scales.
Tuning and replay compute a reference for each dataset and retain the existing
error checks. The adapter selects the existing reference-recomputation path in
the shared tuner; `aiter/utility/mp_tuner.py` remains unchanged.

Use a ROCm/PyTorch environment with native `torch.float8_e8m0fnu`, initialize
the CK submodule, and point `OPUS_HIP_CLANG_PATH` to a clang `bin/` supporting
`clang::amdgpu_pin_agpr`. The known compiler source is `yuyzhang512/llvm-project`
at `49c41889681640665400cb01c9fbb4c0a024cde4`.

## Exact-id architecture

Kernel identity is `(arch, logical family, kid, Y dtype)`. Python resolves a
bare final id through the merged `kernels_list`; C++ receives an already
resolved family call and performs strict lookup in the current architecture's
typed table.

```text
caller final kid
  -> strict 2D opus_gemm or batch-first 3D opus_bmm
  -> Python canonical registry route and family adapter
  -> family C++ entry
  -> runtime architecture + output-dtype table
  -> exact kid lookup
  -> generated launcher checks
```

C++ does not choose a default kid, read a CSV, run a shape heuristic, redirect
an id, allocate a workspace, or fall back to another backend.

The Python layer retains two distinct A16 shape-driven flows. The generic
`aiter.gemm_a16w16` dispatcher uses the global multi-backend tuned result and,
on a miss or invalid OPUS row, keeps its original skinny, gfx1250 Triton, or
PyTorch fallback. It does not run an OPUS heuristic. The OPUS-only
`gemm_a16w16_opus` compatibility entry instead uses an explicit id when
provided, otherwise tries a tuned row and falls back to its heuristic for a
missing or invalid row. All selections pass through legacy compatibility
resolution before the local exact launcher; the strict `opus_gemm`/`opus_bmm`
APIs never redirect. Reusable policy helpers live in
`aiter/ops/opus/policy.py`.

## Family entries

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

void opus_gemm_a8w8_mxscale_bmm_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);
```

## Registry and capability

| Family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `a16w16` | direct + two-stage | direct + two-stage | two-stage + pre-built BF16 direct; fused source retained but unregistered |
| `a8w8` | empty | kid 2, FP32 Y | empty |
| `a8w8_blockscale` | empty | kid 1, FP32 Y | empty |
| `a8w8_blockscale_bpreshuffle` | kid 11000, BF16 Y | empty | empty |
| `a8w8_mxscale_gemm_bpreshuffle` | empty | five pipelines, 89 active compile configurations, 105 compatibility ids; native E8M0 scales, BF16 Y | empty |
| `a8w8_mxscale_bmm` | empty | 45 exact ids in 8000--8653, BF16/FP32 Y | empty |

Empty tables are explicit capability states. The merged registry currently
contains 1032 final ids, including 221 pre-built gfx1250 A16W16 CO ids. Those CO
ids currently occupy 21016--21315 inside the reserved `[21000,27000)` band.
The MXFP8 BMM ids are
`8000 + family_local_kid`, which places them in an unused global band while
preserving family-local tuning/debug correlation. Historical child-dictionary
collisions are resolved by the final merge; runtime routing always follows the
resulting `kernels_list[kid]` instance and never a numeric interval.

The gfx942 BF16-workspace A16 exact kids (`10210`, `10213`, `10216`) are the
one workspace-output exception: their current exact-N reducer requires BF16
`Y`. The canonical Python registry rejects FP32 `Y` before launch, matching the
generated host guard.

## Generated tables

Generated roots are:

```text
opus_gemm_a16w16_kid_dispatch.h
opus_gemm_a8w8_kid_dispatch.h
opus_bmm_mxscale_kid_dispatch.h
opus_gemm_manifest.h
opus_build_archs.h
```

A16 tables separate direct BF16/FP32 launchers from workspace launchers. A8
tables are family and output-dtype scoped. Every macro has a `_SIZE`; an empty
capability produces `std::array<Entry,0>` without referencing a missing
launcher.

Full canonical A16 counts are:

| Architecture | Direct BF16 | Direct FP32 | Workspace |
|---|---:|---:|---:|
| gfx942 | 14 | 1 | 8 |
| gfx950 | 92 | 92 | 48 |
| gfx1250 | 221 | 0 | 496 |

`gen_instances.py` treats tuned CSV ids, the sidecar, the per-architecture
default compile floor, and mandatory A8 ids as build availability. It emits no
runtime shape table. All 45 gfx950 MXFP8 BMM ids are emitted as one family and
deduplicated by generated symbol name rather than entering the ordinary
per-kid subset. All available gfx1250 CO ids are in the gfx1250 compile floor;
codegen emits their five-argument host launchers but no device translation
units. The device bodies come from `gen_co/gfx1250/<symbol>.co`.

## A16 workspace checks

Torch owns every workspace Tensor. Generated launchers validate the final
launch inputs after architecture-specific split resolution:

- XQ/WQ/Y shape, dtype, stride and batch rules;
- exact instance workspace dtype;
- same device, contiguous storage and 16-byte alignment;
- overflow-checked extent and byte-span arithmetic;
- sufficient capacity for the final effective split;
- exact-kid bias support.

Two-stage layouts are split-major. gfx1250 exact kids currently use BF16
workspace storage; the generated launcher/reducer ABI remains typed for either
BF16 or FP32. C++ never owns or retains a Tensor or pointer.

The gfx1250 TDM pipelines use the policy-tag, element-unit API. Clusterlaunch
rounds only the physical grid to `(cluster_wg_m, cluster_wg_n)` multiples;
surplus workgroups arrive at the required cluster barrier and leave through the
uniform `tile_oob` path. Logical tile counts and workspace strides remain
unrounded. The separate reducer dispatches runtime split-K to compile-time
specializations `SPLIT_K_=1..16`, with `SPLIT_K_=0` as the runtime fallback,
using the VEC=8/BLOCK=128 geometry.

The fused gfx1250 factory, emitter and device pipeline remain in-tree for repair,
but `GFX1250_SPLITK_FUSE_ENABLED` is `False`. No fused kid is registered, the
unified capability tables cannot return one, and its `[27000,30000)` band is
unclaimed. The preceding `[21000,27000)` band is reserved for CO ids.

gfx942 continues to wave-uniformize both halves of the direct 64-bit workspace
pointer with `__builtin_amdgcn_readfirstlane` in main and reduce kernels.

## A8 input checks

The family router owns common device/dtype checks. Generated exact-instance
launchers own tile and storage details:

- gfx950 no-scale kid 2: matching 3D FP8 inputs, contiguous FP32 output and
  valid K-loop depth/parity;
- gfx950 blockscale kid 1: the same tensors plus contiguous FP32 1x128x128
  scales and exact scale shapes;
- gfx942 bpreshuffle kid 11000: batch one, BF16 output, exact 128-wide N/K
  tiles, registered scale layouts and truly pre-shuffled WQ content.

MXFP8 BMM is gfx950-only. `opus_bmm.cu` first applies the shared FP8/E8M0
shape, stride, device and output checks, then performs an exact lookup in
`opus_bmm_mxscale_kid_dispatch.h`. Unknown ids fail immediately. Generated
launchers enforce their own M/tile/K restrictions; they never redirect to kid
8000 or another family.

For two-stage BMM split-K, the caller supplies a direct FP32 partial-buffer
pointer. Fused split-K stores partials and aligned tile counters in the same
caller Tensor. The reduce kernel also receives the direct pointer. No BMM
launcher allocates, frees, registers or retains workspace memory.

For global kid 8326 (family-local kid 326), codegen sets
`PRELOAD_SF_LDS=false` only on the `split_k > 1`, `D_OUT=void` workspace
specialization that writes partial sums. Its direct BF16/FP32
`split_k == 1` specializations keep `PRELOAD_SF_LDS=true`.

## Source layout

| Path | Role |
|---|---|
| `opus_bmm.cu` / `include/opus_bmm.h` | MXFP8 BMM exact-kid family entry and Torch-workspace forwarding |
| `opus_gemm_common.py` | canonical registry, unique route map and compile-floor constants |
| `gen_instances.py` | subset selection, manifests and typed dispatch generation |
| `codegen/gen_instances_gfx*.py` | exact-instance host launchers and generated input checks |
| `gen_co/` | offline CO manifest/builder, build metadata and packaged gfx1250 ELF images |
| `include/gfx950/opus_bmm_*` | MXFP8 BMM traits, launchers and pipelines |
| `include/gfx1250/opus_co_launch_gfx1250.cuh` | first-use CO loader and cluster launcher |
| `include/gfx*/opus_gemm_arch_*.cuh` | sorted exact-kid tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | kernel arguments and traits |


## Fine-M tiles and global split-K (9060–9069)

These configurations use the common LDS pipeline through
[shared traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh).
Normal tuning enumerates 9060–9063. The table lists the individual configurations;
9064–9069 remain available for explicit calls and historical-table replay.
They accept M ≤ 2048, K ≤ 16384, N divisible by 128 and K divisible by 128,
including arbitrary positive M tails. Each ID fixes its global K partition count.

| ID | M×N tile | Waves | Global split-K |
|---|---|---:|---:|
| 9060 | 80×128 | 4 | 1 |
| 9061 | 96×128 | 8 | 1 |
| 9062 | 80×128 | 4 | 2 |
| 9063 | 80×128 | 4 | 4 |
| 9064 | 96×128 | 8 | 2 |
| 9065 | 96×128 | 8 | 4 |
| 9066 | 96×128 | 4 | 4 |
| 9067 | 128×128 | 4 | 4 |
| 9068 | 112×128 | 4 | 4 |
| 9069 | 48×128 | 4 | 4 |

Use the existing `opus_gemm(..., kid=9063, layout="bpreshuffle", x_scale=..., w_scale=...)`
route. A scales remain native E8M0 logical `[M,K/128]` in dense column-major storage;
B scales remain contiguous `[N/128,K/128]`. K128 tiles are balanced over partitions,
including uneven and empty partitions. Partial sums and the reduction stay FP32
until the final BF16 conversion.

For split-K IDs, Python allocates a temporary FP32 workspace through PyTorch's
stream-aware allocator. A caller may supply `workspace=` with at least
`global_split_k * M * N` contiguous FP32 elements on the same GPU, aligned to 16
bytes and disjoint from inputs/output. Each invocation overwrites every partial;
no initialization or persistent counter is needed. Separate concurrent invocations
must use separate workspaces. Both automatic allocation and caller-owned storage
support graph capture/replay. The original raw launch ABI remains available;
workspace candidates use a separate checked raw entry.

The public `split_k` argument and tuned CSV `splitK` column remain zero for these
fixed-ID candidates; the actual partition count is encoded in the kernel name.
9060 specializes K=3072/7168, 9062/9063 specialize K=16384, and 9069 specializes
K=7168. Their other legal K values use the general pipeline. Performance comparisons
include both the producer and reduction kernels. See the
[latest 745-shape all-backend comparison](../../reports/opus_current745_tables_20260930/SAME_RUN.md).

## Fixed-K register tiles (9070–9073)

These historical single-launch IDs are now private choices of the general
9042/9052/9053 families and use the common register pipeline. Explicit calls to
9070–9073 still require `1 <= M <= 512`, `N % 128 == 0`, and
`K == 7168`. K waves write FP32 partials to LDS; each output fragment is reduced
and converted to BF16 by one wave. They use no global workspace.

| ID | M×N tile | K waves | Register prefetch | B cache |
|---:|---:|---:|---:|---:|
| 9070 | 16×32 | 8 | 3 | 3 |
| 9071 | 16×48 | 4 | 4 | 3 |
| 9072 | 32×32 | 4 | 4 | 3 |
| 9073 | 32×48 | 4 | 3 | 0 |

N32 tiles reuse the B scale across their N16 fragments. N48 tiles load each
fragment's scale separately and mask the last N tile, including when `N` is not
divisible by 48. Use the same native E8M0 `opus_gemm(..., layout="bpreshuffle")`
entry; public `split_k` remains zero. Their parent families support other legal K
values through the existing runtime-K configurations.
