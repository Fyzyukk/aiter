# OPUS GEMM C++ and code generation

The public Python contract is documented in
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). C++ keeps family launch ABIs and a bpreshuffle workspace entry. They are shared private implementation boundaries for the
Python `opus_gemm(..., kid=...)` and `opus_bmm(..., kid=...)` entries; the
public operation split does not duplicate C++ launchers or kernels.

The [latest gfx950 results](../../reports/opus_current745_tables_20260930/REPORT.md)
list all 26 default MXFP8 B-preshuffle candidates and their win counts, with
per-shape timings in milliseconds against the original CSV. The
[same-run comparison](../../reports/opus_current745_tables_20260930/SAME_RUN.md)
compares OPUS with the fastest valid CK/CKTile/ASM candidate for all 745 shapes.
The reports also quantify timing changes for matching historical configurations.

## MXFP8 B-preshuffle pipeline and traits headers

The original eight gfx950 candidates follow 9000's `template<class Traits>` structure:
pipeline headers contain device execution, and traits headers contain geometry,
storage sizes, and layout constants. The files live in `include/gfx950/`.

| ID | Tile M×N×K / waves | Pipeline | Traits |
|---|---|---|---|
| 9000 | 256×256×128 / 4 | [4wave](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh) | [original traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh) |
| 9010 | 256×256×128 / 4, padded M | [4wave_256x256_padded_m](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh) | [padded-M traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh) |
| 9020 | 192×256×128 / 8 | [8wave_192x256](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh) | [192×256 traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh) |
| 9021 | 128×128×128 / 4 | [4wave_128x128](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh) | [128×128 traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh) |
| 9022 | 160×128×128 / 4 | [4wave_160x128](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh) | [160×128 traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh) |
| 9023 | 64×128×128 / 4 | [4wave_64x128](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh) | [64×128 traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh) |
| 9024 | 64×64×128 / 4 | [4wave_64x64](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh) | [64×64 traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh) |
| 9030 | 192×256×128 / 8, large output | [8wave_192x256_large_output](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh) | [large-output traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh) |

Except for 9000's original naming, the same suffix identifies each
pipeline, traits file, traits type, and device kernel:

```text
opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_<suffix>_gfx950.cuh
opus_gemm_traits_a8w8_mxscale_bpreshuffle_<suffix>_gfx950.cuh
opus_gemm_mxscale_bpreshuffle_<suffix>_traits_gfx950
gemm_a8w8_mxfp8_scale_<suffix>_kernel
```

9000's `gemm_a8w8_mxfp8_scale_kernel` contains only the unpadded 256×256 flow.
9010 owns `gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel`, including bounded
A/C views, checked prefetch offsets, A-scale tail handling, and guarded output.
Its geometry traits and layout/AGPR helpers are shared with 9000; its complete
device body is defined in the padded-M pipeline. The 9000 pipeline and traits
contain no `PAD_M` switch.

Each of 9021, 9022, 9023, and 9024 owns an independent pipeline body and a
fixed-geometry traits type. Their matrix rings retain three, two, three, and
four stages respectively. This gives eight pipeline entry headers and eight
device template bodies for eight registered candidates.
Every pipeline uses runtime K and the shared 96-byte kargs ABI. Existing layout
and AGPR helpers remain in the original 9000 header.

The independent pipeline bodies follow 9000's internal organization:

```text
types and tile/thread coordinates
  -> global-memory views
  -> matrix layouts
  -> LDS views
  -> MMA and register fragments
  -> address, prefetch, scale and operand helpers
  -> Prologue
  -> Main loop
  -> Epilogue
  -> Output writeback
```

Type and coordinate names use `D_A`, `D_B`, `D_C`, `D_ACC`, `D_SF`,
`D_SF_PACK`, `wave_id`, and `lane_id`. The reorganized candidates use `v_c`
for accumulators and `p_coord_c`, `u_gc`, and `gc_offsets` for C layout.
Matrix address helpers use `ga_offset`, `gb_offset`, `sa_offset`, and
`sb_offset`; the four scale offsets are grouped beside them. Matrix prefetch
uses `issue_matrix_prefetch`. Setup and loading helpers precede the Prologue.
All five global views include their own batch stride and tile base offset.

9021/9022 place their advancing schedule directly in the runtime main loop.
9020/9030 keep the two static-stage calls in their U2 loop, with `advance_tile`
defined in the helper section. 9023/9024 use operand-load helpers at the same
last-use replacement points as before. Stage counts, wait instructions,
barriers, AGPR pins, scale layouts, and output strategies remain specific to
each geometry. In particular, 9020/9030 retain fused final MFMA/BF16 staging,
and 9023/9024 retain direct global output stores.

9020 and 9030 each define their complete fixed 192×256 geometry, storage
constants, and static assertions in their own traits file, without inheritance.
Their declarations follow 9000's grouping: threads/waves, B/T/W geometry,
half tiles and geometry checks, E/VEC/GROUP constants, then LDS and scale
storage. Each constant has its own declaration. Address computations belong
to the pipeline's layout helpers, not traits. 9020 reuses 9000's `make_layout_ga_scale`,
`make_layout_sa_scale`, and bulk `async_load` for A. Its eight-wave register
reader uses a matching affine layout with the original 32-byte LDS padding.
Scale A's global and LDS copies share a regular layout: four threads per K128
group cover 192 rows in three 64-row passes, with only the group stride differing.
Its reader and scale packing retain the existing three M-repeat scales.
These local layouts live in `opus_gemm_8wave_192x256_layout`. Its kernel
declares all five `g_*` views and twelve `u_*` layouts before LDS/register setup;
the loading helpers consume these layouts. Its obsolete traits helper has been
removed. B's global view uses the same typed-pointer construction as 9000.
9020 specifies packed B scales (`SFB_BYTES=512`, `B_SCALE_PACKS=1`,
`LDS_BYTES=143360`). 9030 retains its byte scale panel (`256`, `2`, `143104`)
and extends C addressing with a 64-bit base and a buffer range restricted to
the current tile. Its A/B extents and tile-local offsets keep signed 32-bit
limits. The older generic `192x256` traits template remains for retained
experimental sources; neither current candidate includes or inherits it.

9021, 9022, 9023, 9024, and 9030 also define their local layouts with explicit
shape/dim/unfold expressions and lane/wave coordinates. All matrix and scale
layouts are declared before the loading helpers. 9021/9022 retain their A/C
AGPR pins. 9023/9024 retain XOR A addressing, prepacked u16 A scales, replicated
u32 B scales, and direct global C stores. 9030 retains its byte B scales and
its own U2 prefetch schedule. Their source cleanup and CPU address/resource
checks are recorded in the [five-candidate report](../../reports/opus_9021_9030_style_20260928/RESULTS.md).

9021/9022 reuse 9000's RA helper directly. 9023/9024 reuse 9000's GA helper
with the original lane XOR. 9030 reuses 9020's eight-wave RA, and
9021/9022/9030 share 9020's parameterized raw-byte SFA reader. Their includes
expose the existing helpers without merging their independent kernel bodies.
For 9023/9024, scale layouts describe positions within a K128 group, while
the four scale offset helpers advance the group/panel; this preserves the
original dynamic address computation and avoids retaining complete scale
byte addresses across the main loop.

Codegen maps the existing registry tags to these internal names. Candidate IDs,
public `kernelName` values, and support guards are unchanged, so existing
current-ID tuning CSVs remain valid. Historical reports retain their original
source paths and symbol names.

The latest complete all-backend sweep found valid winners for all 305 input
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
The latest change separates 9000's unpadded body from 9010's padded-M body.
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

Use [`opus_gemm_mxscale_bpreshuffle_tune.py`](opus_gemm_mxscale_bpreshuffle_tune.py)
from the checkout root. The latest remote setup, source map, full-shape command,
and replay instructions are at the top of [HANDOFF_MXFP8.md](../../HANDOFF_MXFP8.md).
For all 745 gfx950/256-CU shapes:

```bash
ROCR_VISIBLE_DEVICES=0 \
OPUS_HIP_CLANG_PATH=/absolute/path/to/llvm-pin-build/bin \
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  -i aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv \
  -o /tmp/dsv4_opus_tuned.csv -o2 /tmp/dsv4_opus_profile.csv \
  --libtype all --splitK --shape_grouped --mp 1 --warmup 5 --iters 51 --all
```

The [complete tuned CSV](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv)
contains the fastest valid candidate per shape, using the same 13-column format as
the [original CSV](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv).
The original CSV contains 1042 rows across architectures, including these 745
gfx950/256-CU shapes. Either CSV can be passed to `-i`; the tuner selects the
current gfx/CU shapes and ignores the input timings and candidate choices.
`-o` saves the fastest valid candidate per shape and `-o2` saves the candidate
profile. OPUS candidates are compiled before the sweep; external backends use
their existing JIT paths. `--shape_grouped` measures each shape's candidates
on the same GPU. Omitting `--opus-kids` includes 26 consolidated MXFP8
B-preshuffle candidates: 9000, 9010, 9020–9024, 9030, 9040–9047, 9049,
9051–9055, and 9060–9063. The later candidates share two device templates:
[register](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh)
and [LDS](include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh).
K length and tile-grid size select prefetch, tile geometry, and reduction
configurations inside each family; K remains runtime up to 16384. This reduces
tuning IDs and duplicated implementation, while retaining useful compiled variants.
The 13 historical IDs remain callable with their original names and shape/workspace
contracts, and can be explicitly selected by `--opus-kids`. Saved tables can still
be replayed. Rebuild the OPUS JIT after changing the candidate set.

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

9043–9046, 9055 and 9060–9063 accept M <= 2048; the other default 904x/905x
candidates accept M <= 512. 9000/9010 retain their original implementations.

The renumbering maps old 9020 to 9010 and old 9060–9064 to 9020–9024;
9000 is unchanged. Historical CSVs and JIT binaries retain their old meanings.
Use the current-ID CSV and rebuild the OPUS JIT before executing these new IDs.

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
| `a8w8_mxscale_bmm` | empty | 45 exact ids in 8000--8653, BF16/FP32 Y | empty |

Empty tables are explicit capability states. The merged registry currently
contains 925 final ids, including 219 pre-built gfx1250 A16W16 CO ids. Those CO
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
| gfx1250 | 219 | 0 | 496 |

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
[fine traits](include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_fine_gfx950.cuh).
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
