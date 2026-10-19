# OPUS GEMM C++ and code generation

The public Python contract is documented in
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). C++ keeps family launch ABIs and a bpreshuffle workspace entry. They are shared private implementation boundaries for the
Python `opus_gemm(..., kid=...)` and `opus_bmm(..., kid=...)` entries; the
public operation split does not duplicate C++ launchers or kernels.

The [2026-09-30 gfx950 results](../../reports/opus_current745_tables_20260930/REPORT.md)
list the 26 default MXFP8 B-preshuffle candidates of that revision and their win
counts, with per-shape timings in milliseconds against the original CSV. The
[same-run comparison](../../reports/opus_current745_tables_20260930/SAME_RUN.md)
compares OPUS with the fastest valid CK/CKTile/ASM candidate for all 745 shapes.
The reports also quantify timing changes for matching historical configurations.
Their IDs and measurements are historical; they do not validate the current
parameterized catalog or provide replayable choices for its sequential IDs.

## Current MXFP8 status: 2026-10-10

The registry contains **140 configurations, IDs 9000–9139**, in the existing
six pipeline/traits header pairs. New 9139 is the FixedK7168 version of
LDS-ring 9030: 64×128×128, 4×2 waves, 6 stages, cluster 2. The existing catalog/codegen
tests passed **17/17**. The independent
[`jit_optimized140` build](../../reports/opus_parameterized745_20261009T164517Z/jit_optimized140/build_result.json)
and [CPU device-code audit](../../reports/opus_parameterized745_20261009T164517Z/jit_optimized140/public9139_code_audit.json)
passed: public9139's actual function bytes, descriptor and resources match the
local FixedK prototype, and old public 9030 is unchanged. **Public 9139 GPU
numerical and performance acceptance is still pending.** The completed local
[13-shape screen](../../reports/opus_parameterized745_20261009T164517Z/lds_ring64x128_w4x2_fixedk7168_delta13/analysis.json)
contains 335 measurements and 17085 retained-output checks; it is not public-entry
acceptance. Its FixedK candidate improves the runtime 9030 median on 13 shapes,
beats the strongest fixed BMM median on 2 and the BMM round-envelope median on 1,
and wins no shape in every round against that envelope.

The full 745-shape tune and conditioned five-round comparison are **complete**:
[the frozen same-run result](../../reports/opus_parameterized745_20261009T164517Z/full745_revalidation_conditioned/analysis/README.md)
is **332 OPUS median wins / 413 losses**. Reclassifying those existing timings
against `min(current strongest BMM median, original CSV us)` gives **323 wins**;
this is CPU analysis, not new GPU evidence. The current strongest FlyDSL remains
slower than the original CSV on 132 shapes, including 19 by more than 5%. The older
445 within ±5% / 260 faster >5% / 40 slower >5% counts refer to the same original
kernel name/configuration. Five percent is descriptive, not a relaxed acceptance
rule. Local optimization results do not change these full 745-shape counts; see the
[historical-ceiling analysis](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/deltas/README.md).

**Correction: the earlier claim that the baseline was aligned incorrectly
presented configuration/CSV identity checks as measured performance reproduction.**
The [original PR #5896 source](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/origin/README.md)
already contained the table at `ef0297e4`, before source changes retained by the
merge; its documented CK/asm companion protocol and full original environment
have not been recreated. The [current compiler identities](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/compiler_identity.json)
also differ: OPUS uses LLVM `49c41889`, while the installed FlyDSL wheel embeds
LLVM `e2a39f50`. `HIP_CLANG_PATH` does not select FlyDSL's embedded backend.
An independent MLIR build now uses the original OPUS LLVM library. Three slow
representatives (1536×7168×384, 64×6144×7168, 8×65536×1536) have identical kernel
functions, descriptors and resources across initial/current source and the two
backend/linker combinations. This excludes those code-generation differences
for these configurations; it does not establish the original run environment.
The frontend and original compiler options remain distinct; changing final
backend O2 to O3 also leaves the 1536 case unchanged. See the
[compiler investigation](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/README.md).
The broader comparison covers 621 unique compile specializations from 1490
shape/role mappings: 479 have identical executable code, ABI and loading
semantics; 142 have different functions, including 25 resource changes. The
changed products touch 17 of the 132 slower shapes, and only 2 of the 19
slowdowns over 5%. Four controlled representatives reproduce the old cache and
isolate actual backend differences with the same prebinary IR, final pass and
LLD23, excluding dump/linker effects for those four. This CPU evidence alone
does not establish numerical correctness or performance. See the
[four-case controls](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/differing_backend_controls4/summary.json)
and [slow-shape mapping](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/common_backend_slow_mapping/README.md).
Changing only the common final backend from O2 to O3 also preserves the entire
HSACO for all four; this does not equate HIP/DSL frontends or all compiler flags.
The subsequent four-case GPU comparison passed 48 seed gates, 80 prechecks,
80 timing rows and 4080 retained outputs, with actual A/D loading and call
bindings verified. Paired common49/original-backend changes were −1.447%,
−0.049%, +0.636% and −0.151% for 256×2048×7168, 64×65536×1536,
480×7168×3072 and 16384×6144×7168. The first two retain the original CSV
configuration and remain 6.999% and 2.039% slower than it under common49;
neither independent process recovers the gap. The other two use different
current configurations and were already faster than the CSV under the original
backend. See the [measured historical comparison](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/backend_gpu_probe4/historical_comparison/README.md).
These four cases do not update the full745 score. Sparse occupancy/clock
samples do not certify the state during each microsecond kernel.
The [six-arm timer diagnostic](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/timer_probe/analysis/README.md)
passed 120 timing rows and 9120 retained-output checks; the original timer's
default 2 warmups/101 iterations did not recover that case's CSV time.
The CK companion diagnostic subsequently completed both reversed-order
processes: 40 timing rows, 40 prechecks and 2040 retained outputs passed with
zero error ratios. CK preceding BMM improves the paired median by only 0.3202%;
the per-process BMM medians remain 10.71%–12.80% slower than the original10.82us.
That companion choice does not recover this gap. The initial external
interruption remains preserved. See the [companion result](../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/companion_probe/performance_resume_after_external_20261010T0248Z/analysis/README.md).
Historical performance reproduction and the all-745-shape OPUS optimization
target remain unfinished.

## Blockscale / MX-scale B-preshuffle pipeline and traits headers

The current large-shape worktree retains four gfx950 B-preshuffle pipelines:
`pin_agpr`, `prefetch_b_vgpr`, `pingpong_lds` and `prefetch_b_vgpr_overlap`. They serve 136, 81,
69 and 8 of the 294 final selections in round58, respectively. Each has an
explicit device entry and pipeline header; legal geometry, register placement,
cache policy and output layouts remain configuration parameters.

The unused `pin_agpr_early`, `register_prefetch` and `lds_ring` implementations,
exclusive traits and registrations were removed. Remaining IDs and launch
contracts are unchanged; B-direct JIT IDs stay at 39585–42089. All retained
configurations use WaveK=1. The registry contains 161 CO and 2505 JIT entries.

See [the pipeline reference](include/gfx950/bpreshuffle_templates.md) for source
mapping and parameter boundaries. All four pipelines use
[`opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh`](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh)
for shared configuration and helpers. The kernel-argument ABI remains 96 bytes.
The structural changes make no new performance claim.

## Historical validation and performance records

The following records describe the IDs, source organizations and validation
performed at their recorded revisions. They make no validation or performance
claim for the current registry.

The 2026-09-28 complete all-backend sweep found valid winners for all 305 input
shapes: OPUS won 299, CK 2, and CKTile 4. All 1,955 OPUS measurements passed
the numerical check; the ten shapes omitted from the historical 295-shape
subset all selected 9030. See the
[full sweep](../../reports/opus_full305_after9020_20260928/RESULTS.md) and
[candidate optimization details](../../reports/opus_full305_after9020_20260928/CANDIDATES.md).
The source organization above is newer than that sweep. The initial organization
change passed [CPU validation for all eight candidates](../../reports/opus_layout_9000_20260928/validation.json).
The subsequent split of 9021–9024 compiled those four actual device TUs once each;
their machine instructions match the previous version byte for byte, and their
complete resource metadata matches after excluding symbol names. Public names,
registry entries, and host guards are unchanged; see the
[four-candidate split validation](../../reports/opus_split_9021_9024_20260928/validation.json).
The subsequent internal pipeline reorganization also passed
[CPU validation for all six independent bodies](../../reports/opus_pipeline_structure_20260928/attempt2/validation.json):
instruction bytes and complete resource metadata match their previous versions.
Wave-coordinate and runtime-K calculations retain their original evaluation
points to preserve compiler lowering. No source refactor ran another GPU tune.
The [per-candidate optimization summary](../../reports/opus_pipeline_structure_20260928/OPTIMIZATION_SUMMARY.md)
describes each candidate's target, retained optimization steps, and measured scope.
A later change in that historical sequence separated 9000's unpadded body from
9010's padded-M body.
Both actual device TUs compiled once each; their instruction bytes and complete
resource metadata (excluding symbol names) match the prior implementations.
Public names, registry entries, and host guards also match; see the
[9000/9010 separation validation](../../reports/opus_9000_9010_separate_20260928/validation.json).
The earlier optimization summary records 9010's former shared-body organization.
9010's subsequent spill fix moves the SFA per-pass zero initialization before
the K guard, preventing unused raw-scale values from remaining live across the
matrix loop. The final production TU has zero VGPR/SGPR spills, zero private
segment bytes, and no scratch load/store instructions; LDS remains 152064 bytes.
This passed CPU compilation and source review only, with no new GPU numerical
or performance run; see the [9010 spill report](../../reports/opus_9010_spill_20260928/summary.json).
The subsequent expansion of 9020/9030 traits into complete independent structs
compiled both device TUs once each. Their instruction bytes and full metadata,
including names, match the previous versions; see the
[traits expansion validation](../../reports/opus_9020_9030_traits_20260928/validation.json).
Moving 9030's A-address helper into its pipeline also passed one device-TU
CPU compilation: all 7884 instruction bytes and the complete metadata are
identical. 9020's unused helper was removed without recompiling its pipeline.
See the [helper relocation validation](../../reports/opus_9030_lds_helper_20260928/validation.json).
9020's subsequent restoration of upfront global views and layouts compiled
successfully with identical full metadata (VGPR 208, SGPR 52, LDS 143360 bytes,
zero spills/private storage). Its instruction bytes changed from 7816 to 7880;
no GPU correctness or timing run was performed for this source reorganization.
See the [final frontmatter compilation](../../reports/opus_9020_frontmatter_20260928/attempt2/validation.json).
The later restoration of A's non-XOR layout and regular Scale A producer layout
passed one device-TU CPU compilation. VGPR usage decreased from 208 to 204;
SGPR 52, LDS 143360 bytes, zero spills/private storage, and the public ABI remain
unchanged. The instruction body is 7656 bytes. This layout change has no new GPU
numerical or timing result; see the
[A layout restoration validation](../../reports/opus_9020_restore_a_layout_20260928/validation.json).

## Current MXFP8 tuning entry

Use the shared
[`gemm_a8w8_mxfp8_bpreshuffle_tune.py`](../gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py)
with `--mxscale_table`. It searches OPUS, FlyDSL GEMM and FlyDSL BMM on
A 1x128 / B 128x128 native E8M0 scales and saves the fastest valid candidate
in the existing mxscale table:

```bash
python -u csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype both \
  -i aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
  -o /tmp/dsv4_mxscale_tuned.csv -o2 /tmp/dsv4_mxscale_profile.csv \
  --shape_grouped --mp 8 --warmup 5 --iters 101 --all
```

The [source CSV](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv)
uses the original 13-column format and contains 861 rows: 745 gfx950/256-CU shapes,
105 gfx942/80-CU shapes and 11 gfx1250/256-CU shapes. The 1042-row count in
dated records describes an earlier dataset. This CSV can be passed to `-i`; the tuner selects the
current gfx/CU shapes and ignores the input timings and candidate choices.
`-o` saves the fastest valid candidate per shape and `-o2` saves the candidate
profile. OPUS rows retain `libtype=opus`, the exact numeric `kernelId` and
its fixed `splitK`. The ordinary GPU candidates use JIT, pin-AGPR candidates
use pre-built CO, and their host launchers are prepared before the sweep.
`--shape_grouped` measures each shape's candidates on the same GPU.

Omitting `--opus-kids` searches the complete saved shape candidate pool.
`--opus-kids` restricts the search to specified registered IDs that fit the
shape. Use `--libtype opus` or `--libtype flydsl` for a single backend family.

One ID fixes tile geometry, prefetch/stages, scale policy, FixedK, global Split-K
and reduction settings. A FixedK ID requires its exact K; runtime-K variants
have separate IDs. Shape checks derive from the selected configuration's
layout, scale capacity and addressing limits. The former manual M≤512/M≤2048
and short-K tuning caps do not constrain this registry.

Retune old CSV choices and rebuild the OPUS JIT for the new IDs. Historical CSVs
remain usable as shape inputs through `-i`; their IDs and names cannot be
replayed as current configurations. `--all` forces remeasurement of input
shapes and does not request exhaustive configuration enumeration.

OPUS and FlyDSL BMM use FP8 operands and the same native E8M0 scale layouts;
BMM adds batch-one views. Tuning and replay retain each backend's reference and
error checks. Use a ROCm/PyTorch environment with native
`torch.float8_e8m0fnu` and the FlyDSL BMM dependencies. Build the pin-AGPR
images with `gen_co/build_co.py` before tuning; ordinary JIT candidates use
the normal compiler. `OPUS_HIP_CLANG_PATH` optionally selects a JIT compiler.

The completed full-shape tune and subsequent optimization are recorded in
`reports/opus_parameterized745_20261009T164517Z`. The current status above
separates completed same-run measurements, unresolved original-baseline
reproduction, and pending public 9139 acceptance. Registering 140 configurations
does not mean that this newest catalog has undergone another complete sweep.

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
| `a8w8_mxscale_bmm` | empty | registered BMM ids, including compact ids 8470--8484; BF16/FP32 Y | empty |

Empty tables are explicit capability states. `kernels_list` is the source of
truth for registered ids. The pre-built gfx1250 A16W16 CO ids occupy the
reserved `[21000,27000)` band.
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
| gfx1250 | 219 | 0 | 496 |

`gen_instances.py` treats tuned CSV ids, the sidecar, the per-architecture
default compile floor, and mandatory A8 ids as build availability. It emits no
runtime shape table. All gfx950 MXFP8 BMM ids in `BMM_MXSCALE_KIDS` are emitted
as one family and deduplicated by generated symbol name rather than entering the ordinary
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
| `opus_gemm_common.py` | canonical registry, B-preshuffle configurations, catalog, CPU candidate ranking and exact metadata, unique route map and compile-floor constants |
| `../gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py` | Shared native-E8M0 GEMM tuning and mxscale table replay |
| `gen_instances.py` | subset selection, manifests and typed dispatch generation |
| `codegen/gen_instances_gfx*.py` | exact-instance host launchers and generated input checks |
| `gen_co/` | offline CO manifest/builder, build metadata and packaged gfx1250 ELF images |
| `include/gfx950/opus_bmm_*` | MXFP8 BMM traits, launchers and pipelines |
| `include/gfx1250/opus_co_launch_gfx1250.cuh` | first-use CO loader and cluster launcher |
| `include/gfx*/opus_gemm_arch_*.cuh` | sorted exact-kid tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | kernel arguments and traits |


## Fine-M tiles and global split-K

These configurations use the common LDS pipeline through
[LDS-ring traits](include/gfx950/opus_gemm_blockscale_mxscale_bpreshuffle_traits_lds_ring_gfx950.cuh).
`fine_m_loads` handles partial A-copy passes; `split_k` independently sets
the global partition count. The catalog contains multiple M tiles, wave
layouts, stages and reduction policies, each under its own sequential ID.
These configurations require K ≤ 16384, N divisible by 128 and K divisible by
128, with arbitrary positive M tails subject to addressing limits. FixedK
configurations additionally require exact K equality.

Select the desired ID using the candidate report, then use
`opus_gemm(..., kid=kid, layout="bpreshuffle", x_scale=..., w_scale=...)`.
A scales remain native E8M0 logical `[M,K/128]` in dense column-major storage;
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
fixed-ID candidates; the actual partition count is part of the configuration
and saved metadata. Runtime-K and K=3072/7168/16384 specializations are separate
configurations. Comparisons for global Split-K must include both the producer
and reduction kernels. The
[2026-09-30 comparison](../../reports/opus_current745_tables_20260930/SAME_RUN.md)
records historical measurements only.

## Fixed-K register tiles

FixedK configurations use the common register pipeline and have separate IDs
from runtime-K configurations. The following K=7168 parameter combinations are
included in the catalog. They require `N % 128 == 0` and `K == 7168`, with M
subject to their memory-addressing limits. K waves write FP32 partials to LDS;
each output fragment is reduced and converted to BF16 by one wave. They use no
global workspace.

| M×N tile | K waves | Register prefetch | B cache |
|---:|---:|---:|---:|
| 16×32 | 8 | 3 | 3 |
| 16×48 | 4 | 4 | 3 |
| 32×32 | 4 | 4 | 3 |
| 32×48 | 4 | 3 | 0 |

N32 tiles reuse the B scale across their N16 fragments. N48 tiles load each
fragment's scale separately and mask the last N tile, including when `N` is not
divisible by 48. Use the same native E8M0 `opus_gemm(..., layout="bpreshuffle")`
entry; public `split_k` remains zero. Use a runtime-K configuration's own ID for
other supported K values.
## Compact MXFP8 BMM on gfx950

Kids 8470--8476 retain their original `N=1024, K=4096` device specialization.
Other supported N/K shapes launch the same kid's general compact kernel. It
stages at most 4096 K elements of E8M0 scales at a time and retains FP32
accumulators across segments, so LDS usage does not grow with total K.
Short and final segments reread their last valid A/B tile to maintain the
prefetch depth, then drain outstanding copies before reusing LDS.

The compact contract is:

- `1 <= M < 2048`, `1 <= batch <= 16`;
- positive N divisible by both 128 and the kid's `B_N`;
- positive K divisible by the kid's `B_K` (128, 256, or 512);
- ordinary 128x128 weight scales and 1x128 activation scales;
- contiguous token-major A/scales/Y storage and `(16,16)` preshuffled B;
- BF16 or FP32 output, split-K 0/1, no workspace;
- 16-byte A/B/Y pointer alignment and 4-byte scale pointer alignment;
- A's total bytes, each B batch's bytes, and Y's total bytes fit signed int32.

The public `opus_bmm` interface still uses batch-first transpose views and
`layout="mxscale_bmm"`. The C++ launcher validates physical strides. Shape
support is not a performance selection: existing general pipelines remain
available, and the tuner compares eligible compact kids without changing
existing tuned CSV rows. Regression coverage is in
`op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py`.

Kid 8477 adds a `16x32x512` compact tile for small M: 128 threads (two
compute waves in a 1x2 grid), three A/B LDS slots, and 73 KiB of LDS.
It supports both the fixed N=1024/K=4096 and general paths, and requires K
divisible by 512. All waves stage A/B and compute; this differs
from kid 8179's same-size `bdirect` tile, where two producer waves stage A
and two consumer waves load B directly into registers and perform MFMA.

Additional compact candidates use four compute waves and the same group128
scale contract. The LDS column includes A/B ring and scale panels; it alone
does not determine actual occupancy, which also depends on registers.

| Kid | Tile MxNxK | Wave grid MxN | A/B slots | LDS KiB |
|---|---|---|---|---|
| 8478 | 32x32x512 | 2x2 | 2 | 66 |
| 8479 | 16x64x512 | 1x4 | 2 | 82 |
| 8480 | 16x64x256 | 1x4 | 3 | 62 |
| 8481 | 32x64x256 | 2x2 | 3 | 74 |
| 8482 | 64x32x256 | 2x2 | 3 | 75 |
| 8483 | 64x64x128 | 2x2 | 3 | 51 |
| 8484 | 128x128x128 | 2x2 | 2 | 69 |

These candidates cover intermediate tile sizes and shallower rings. They are
additional tuner options, with no changes to existing kids or tuned CSV rows.

## Tuning MXFP8 BMM

See [MXFP8 BMM tuning](../bmm_a8w8_mxscale/README.md) for the header-only
input/output CSV templates, input schema, OPUS/FlyDSL tuning commands,
configuration verification and performance comparison workflow.

The joint tuner is `csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py`.
The OPUS-only entry is `csrc/opus_gemm/opus_bmm_mxscale_tune.py`; it uses the
same CSV schema and shared tuner options. Pass `--bpreshuffle` when tuning
for the preshuffled-weight interface.
