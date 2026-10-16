# gfx950 blockscale / MX-scale B-preshuffle templates

All calculation headers and internal template symbols use the prefix
`opus_gemm_blockscale_mxscale_bpreshuffle_`. `blockscale` describes the logical
scale groups (A: 1×128; B: 128×128), `mxscale` describes the native E8M0 scaled
MFMA path, and `bpreshuffle` identifies the required B layout. The logical K128
scale is reused across the hardware MFMA's K32 scale groups.

The configuration catalog contains 140 unique configurations with sequential
IDs 9000–9139. Each ID selects one resolved kernel/traits configuration.
Runtime-K and FixedK variants have separate IDs; launchers do not choose a
different tile, prefetch policy or specialization based on the input shape.
Old manual IDs, aliases, generated names and saved CSV choices have no
compatibility mapping. Retune old choices before replaying them.

Maintain calculation pipelines separately from their compiled configurations.
Tile dimensions, supported wave layouts, prefetch/stages, scale panels, FixedK,
Split-K, reduction and swizzle settings are compile-time parameters. Tuning
selects generated instances. More configurations do not require more
handwritten pipeline files.

## Shared calculation templates

| Configuration pipeline | Pipeline / traits suffix | Supported parameter axes | Catalog count |
| --- | --- | --- | ---: |
| `quadrant_4wave` | `4wave_pin_agpr` | `pad_m`; fixed 256×256×128 tile, 2×2 wave layout and pinned schedule | 2 |
| `register` | `register_prefetch` | M/N tile, M/N/K waves, register prefetch, FixedK, B cache, scale reuse, output packing/N tails | 23 |
| `lds_ring` | `lds_ring` | M/N tile, M/N waves, stages, cluster, scale placement, drain/read policy, Fine-M loads, FixedK, global Split-K and FP32 reduction | 54 |
| `packed_4wave` | `4wave`, packed kernel | N tile, stages, scale panel, FixedK, block grouping and fragment-read interleaving | 16 |
| `lds_4wave` | `4wave`, LDS-output kernel | M tile, stages, scale panel, FixedK and scale-tail load policy | 15 |
| `panel_8wave` | `8wave` | M/N tile, scale panel, FixedK, schedule and block swizzle | 27 |
| `b_direct_4wave` | stored in `lds_ring`, independent type/kernel | Fixed 192×256×128, 1×4 waves, four A LDS slots/four B register sets, FixedK7168 | 1 |
| `panel_2wave` | `2wave` | M tile, scale panel, FixedK; two LDS slots and a 1×2 wave layout; opt-in for tuning | 2 |

The 4-wave packed and LDS-output kernels have different scale and output
layouts, so they retain two calculation bodies. The 8-wave template shares its
calculation body and selects `interleaved` or `synchronized` scheduling with
`if constexpr`, including the prefetch/wait/barrier order. The latter also
retains large-output addressing and its supported 192×256/panel128 layout.

IDs 9000 and 9001 instantiate the pinned kernel with
`4wave_pin_agpr_traits_gfx950<false>` and `<true>`, respectively. This template
uses four Wave64 waves, a 256×256×128 block tile, four 128×128 subtiles, and
pinned AGPR operands/accumulators. `if constexpr (T::PAD_M)` selects bounded A/C
views, matrix-prefetch address formation, A-scale panel initialization and row
guards, prologue A addresses, and C row guards. The main loop and pinned-compiler
requirement are shared.

There are eight GEMM bodies in six pipeline headers, with six matching traits
headers (12 B-preshuffle headers total). Shared layout/output functions and
argument definitions live in these existing files. Pipeline and traits headers
share one suffix: `2wave`, `4wave_pin_agpr`, `4wave`, `8wave`,
`register_prefetch`, or `lds_ring`. Their names follow these forms:

```text
opus_gemm_blockscale_mxscale_bpreshuffle_pipeline_<suffix>_gfx950.cuh
opus_gemm_blockscale_mxscale_bpreshuffle_traits_<suffix>_gfx950.cuh
opus_gemm_blockscale_mxscale_bpreshuffle_<suffix>_traits_gfx950
```

The LDS-ring header pair also owns the independent `b_direct_4wave` type and
kernel, preserving the existing LDS-ring definitions. Its traits suffix is
`b_direct_4wave`, while its header suffix remains `lds_ring`. It loads A into
four LDS slots and B into four register sets; each B column is reused for a
future K tile after its 12 MFMA consumers. K is fixed at 7168, M must be a
multiple of 4 and N a multiple of 256, with the signed-32 tensor extent guard.
The host emits a flattened grid rounded to eight workgroups. Static LDS is
110608 bytes and dynamic LDS is zero. Performance varies with the shape;
ordinary tuning determines whether this candidate is useful.

The pinned M contracts remain 256 for `pad_m=false` and 64 for `pad_m=true`.
The former uses a floor-sized M grid and unguarded full-tile loads/stores. The
latter uses a ceiling-sized grid and guards A/C rows, but A scales still load
16 consecutive bytes after checking only the starting row. M divisible by 16
keeps each active scale vector within its column and preserves alignment; the
current 64-row contract is more conservative. Arbitrary M would require a
byte-safe scale-tail path and boundary validation before relaxing this guard.

Shared code is owned once inside the existing pipeline/traits headers:

| Shared definitions | Owning header | Consumers |
| --- | --- | --- |
| Fragment reads, scale layouts and safe A-scale tail loads | [`pipeline_4wave`](opus_gemm_blockscale_mxscale_bpreshuffle_pipeline_4wave_gfx950.cuh) | Ordinary 4-wave, 2-wave and LDS ring |
| A-fragment and panel scale layouts | [`pipeline_8wave`](opus_gemm_blockscale_mxscale_bpreshuffle_pipeline_8wave_gfx950.cuh) | 8-wave, 2-wave and ordinary 4-wave |
| MFMA16 BF16 output packing | [`pipeline_register_prefetch`](opus_gemm_blockscale_mxscale_bpreshuffle_pipeline_register_prefetch_gfx950.cuh) | Register prefetch and LDS ring |
| Shared constants, kernel arguments and ceiling division | [`traits_4wave_pin_agpr`](opus_gemm_blockscale_mxscale_bpreshuffle_traits_4wave_pin_agpr_gfx950.cuh) | All six traits/pipeline families |

Consumers include the owning header. Each kernel retains its computation and
synchronization schedule. The common A/B matrix layouts remain in the pinned
pipeline. The traits owner contains no pinned device code, so including shared
arguments does not require the pinned-AGPR compiler attribute. The common
kernel-argument ABI is 96 bytes.

## Tile size and calculation flow

All current register-prefetch and LDS-ring configurations use BK=128. Their
parameter tables include these tile geometries:

| Template | Registered BM×BN | Waves per workgroup |
| --- | --- | --- |
| `register_prefetch` | 16×16, 16×32, 16×48, 32×32, 32×48, 32×64 | 1, 4 or 8, depending on configuration |
| `lds_ring` | {32,64,96}×64; {32,48,64,80,96,112,128}×128 | 4 or 8, depending on configuration |

The register template loads A/B into a register prefetch queue, then executes
MFMA. When `wave_k>1`, waves compute different K partitions and combine their
FP32 partials through LDS before BF16 conversion. The LDS-ring template instead
copies A/B into reusable LDS slots, waits for the required data, reads fragments
into registers and executes MFMA while prefetching future K tiles.

With the wave layout and policies held fixed, increasing a tile changes the
fragment counts `E_M=BM/(WaveM*16)` and `E_N=BN/(WaveN*16)`. Each wave executes
`E_M*E_N` native 16×16×128 MFMA operations per K128 tile assigned to it. Operand
storage, accumulator count and memory-copy work grow accordingly; the template's
prefetch/compute/reduction structure remains shared.

For example, register configurations with the same 1×1×8 wave layout,
`prefetch=2` and `output=3` can use 16×32 or 32×32 tiles. Increasing BM raises
E_M from 1 to 2, so each wave executes four instead of two MFMA operations per
assigned K128 tile. Both retain the same register prefetch and final WaveK
reduction flow. Other table rows may change scheduling or output policies
alongside geometry; those choices remain explicit configuration parameters.

Stages and cluster control queue depth and barrier cadence; `fine_m_loads`
handles a final A copy with fewer active waves. Scale placement and output
policies select their own compile-time paths. Global `split_k` partitions work
between workgroups and adds an FP32 workspace reduction. It is distinct from
`wave_k`, which cooperates within one workgroup. The same tile can have several
such configurations without a new handwritten calculation template.

## Parameter schema and catalog enumeration

[`opus_gemm_common.py`](../../opus_gemm_common.py) owns `BpreshuffleConfig`, the
pipeline-specific parameter domains, validation, instance factory, enumeration,
shape ranking and exact configuration metadata. All registrations come from
these parameters in the single
`a8w8_mxscale_gemm_bpreshuffle_kernels_list`. There is no second catalog registry
or manual 9xxx dictionary.

| Parameter group | `BpreshuffleConfig` fields |
| --- | --- |
| Geometry | `pipeline`, `block_m`, `block_n`, `wave_m`, `wave_n`, `wave_k` |
| K specialization | `fixed_k` (`0` for runtime K) |
| Register queue and output | `prefetch`, `output`, `b_cache`, `pad_n`, `reuse_b_scale` |
| LDS queue and scheduling | `stages`, `cluster`, `register_scales`, `xor_lds`, `early_scale_loads`, `prefetch_before_read`, `read_only_drain`, `fine_m_loads` |
| Global Split-K and reduction | `split_k`, `reduce_vec`, `reduce_block`, `store_cache` |
| Compact/panel layouts | `scale_panel`, `group_m`, `interleave_matrix_reads`, `byte_safe_scale_tail` |
| Eight-wave scheduling/swizzle | `schedule`, `n_swizzle`, `swizzle_group_n`, `swizzle_group_m` |
| Pinned M tails | `pad_m` |

`generate_configs` expands coupled geometry and policy rows, rather than a
Cartesian product of every field. `build_catalog` validates and deduplicates
resolved configurations, then assigns `9000 + i` in the canonical
`ALL_PIPELINES` order. This follows the parameter-table/factory/enumeration
pattern used by other OPUS families, such as persistent IDs `300 + i`.
Enumeration covers all pipelines before any pipeline or shape filter is
applied. Selecting fewer pipelines or a different shape cannot change an ID.
The reserved range is 9000–9999; catalog overflow is rejected.

The domain includes the previous 74 generated configurations plus 63 unique
configurations from the manual base and specialization variants, for the initial
137. Later append-only entries added short-K 8-wave swizzle 9137 and B-direct 9138,
bringing that stage to 139. The current 140th entry, 9139, adds FixedK7168 to the
existing runtime 9030 configuration: 64×128×128, 4×2 waves, 6 stages, cluster 2,
output 2, early scale loads and SplitK1. Other policies and old IDs/keys remain
unchanged. It uses the existing LDS-ring body and adds no header or calculation
body.
Equivalent resolved kernel/traits combinations appear once. The family counts
above describe this domain, not every theoretically legal combination of
parameters or a promise that each configuration is numerically validated.
Changing the parameter tables or their order can change subsequent IDs; there
is no compatibility promise for old tuning outputs.

`config_key` hashes the actual kernel symbol and resolved traits arguments,
including resolved `None` defaults. It supports metadata and deduplication;
the integer ID itself is sequential. Generated names use configuration identity
without translating symbols back to historical aliases. Metadata uses schema
version 2 and describes exactly the configuration launched by the selected ID.

A parameter row can be constructed and passed through the same instance factory:

```python
from opus_gemm_common import config_from_dict, instance_from_config

config = config_from_dict(dict(
    pipeline="lds_ring",
    block_m=64, block_n=64, wave_m=2, wave_n=2,
    stages=8, cluster=2, early_scale_loads=True,
))
candidate = instance_from_config(config)
```

To extend the registered domain, add the desired parameter row to the existing
pipeline domain in `generate_configs`; enumeration assigns its ID. Constructing
an instance alone does not register it. The generator emits thin traits aliases,
explicit instantiations and a launcher for the selected configuration using the
existing template headers. No handwritten ID entry, copied kernel, per-tile C++
alias or shape-dispatch wrapper is needed.

Python rejects parameters that the chosen pipeline does not consume and checks
layout, resource and launch-contract agreement. C++ `static_assert`s enforce
detailed layout and LDS constraints. Stage count is tunable only where the
schedule supports it. Eight-wave scales load once, so runtime K must fit in its
scale panel; smaller panel pipelines refill scales for long K. FixedK IDs require
exact K equality. Shape eligibility uses intrinsic alignment, scale capacity and
addressing restrictions; the former manual M≤512/M≤2048 and short-K tuning caps
are removed.

## Candidate selection and replay

The default selector ranks legal configurations and returns at most ten IDs per
shape, with at most two per pipeline. It considers the seven `DEFAULT_PIPELINES`;
`panel_2wave` is opt-in. This bounded policy does not exhaustively tune all 140
configurations. `candidate_kids_for_shape(..., all_candidates=True)` bypasses
ranking and candidate-count limits for explicit IDs and exact replay while
retaining shape legality. There is no `include_legacy` selection mode.

[`opus_gemm_mxscale_bpreshuffle_tune.py`](../../opus_gemm_mxscale_bpreshuffle_tune.py)
provides CPU candidate reports through `--explain-candidates`. From the
repository root, inspect candidate selection without loading the GPU runtime:

```bash
python csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py --explain-candidates --shape 64 256 384 --explain
```

The report accepts `--pipelines`, `--max-candidates` and `--per-pipeline`.
GPU tuning uses `--opus-pipelines`, `--opus-max-candidates` and optional explicit
`--opus-kids`. Explicit IDs and replay select the same single configuration
regardless of ranking policy. Old CSVs can supply shapes through `-i`, but their
IDs/names cannot be replayed against the new registry; save new tuning outputs
with complete configuration metadata and rebuild the OPUS JIT.

## Validation and subsequent optimization

The initial parameterization changed registration and selection without changing
the six pipeline or six traits headers. It does not establish GPU
correctness or performance for the enumerated configurations. Historical
reports retain their original IDs and measured scope, including the numerical
issues recorded for experimental variants. The parameterization itself made no GPU validation or speedup claim. Subsequent
full745 tuning and comparison are recorded in
[the conditioned audit](../../../../reports/opus_parameterized745_20261009T164517Z/full745_revalidation_conditioned/analysis/README.md):
332 current same-run median OPUS wins and 413 losses. CPU reclassification of
those same measurements against `min(current strongest BMM median, original
CSV us)` gives 323 wins, excluding 9 of the same-run wins. This is not a new GPU
result. Current strongest FlyDSL is still slower than the original CSV on 132
shapes, including 19 by more than 5%; 5% is descriptive, not permission to regress
or use a slower comparator. See the
[historical-ceiling analysis](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/deltas/README.md).

**Correction: earlier baseline-alignment wording incorrectly presented
configuration/CSV identity as measured performance reproduction.** Original
PR #5896 initial source `ef0297e4` differs from merge/current source even though
its CSV rows survived unchanged. The actual current compiler backends also
differ: OPUS LLVM `49c41889` versus the installed FlyDSL wheel's `e2a39f50`.
The independent same-49c MLIR build is complete. Three slow representatives
(1536×7168×384, 64×6144×7168, 8×65536×1536) have identical kernel functions,
descriptors and resources under initial/current source and both backend/linker
combinations. The frontend and original options remain distinct; the 1536
final-backend O2→O3 check also produces identical code. The six-arm timer probe
passed 120 measurements/9120 retained-output checks without recovering that
case's CSV time. These bounded negative results do not identify the original
environment or settle the historical gap. See the
[PR source/protocol record](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/origin/README.md),
[compiler investigation](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/README.md),
and [timer diagnostic](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/timer_probe/analysis/README.md).

The subsequent 621-specialization comparison found 479 identical executable
code/ABI/loading products and 142 different functions, including 25 resource
changes. These are compile specializations, not shape counts. Differences touch
17/132 slower shapes and 2/19 slowdowns over 5%. Four same-input/final-pass/linker
controls isolate actual LLVM backend differences for those representatives. See
the [four-case controls](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/differing_backend_controls4/summary.json)
and [shape mapping](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/common_backend_slow_mapping/README.md).
All four also retain byte-identical whole HSACO when only the common final
backend optimization level changes from O2 to O3; their frontends and remaining
compiler options are not asserted to be identical.
Their later GPU comparison passed 48 seed gates, 80 prechecks, 80 timing rows
and 4080 retained outputs, including actual loaded-code/call-binding checks.
Paired common49/original-backend changes were −1.447%, −0.049%, +0.636% and
−0.151% for 256×2048×7168, 64×65536×1536, 480×7168×3072 and
16384×6144×7168. The first two retain the historical configuration and remain
6.999% and 2.039% slower than the CSV; the other two change configuration and
were already faster with the original backend. This does not resolve historical
reproduction or update745 scores. See the [GPU/history comparison](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/compilers/backend_gpu_probe4/historical_comparison/README.md).
The later CK/OPUS companion comparison passed 40 timing rows and2040 retained
outputs, but its 0.3202% paired-median improvement did not recover the original
1536×7168×384 CSV time. See the [companion result](../../../../reports/opus_parameterized745_20261009T164517Z/baseline_cause_investigation/companion_probe/performance_resume_after_external_20261010T0248Z/analysis/README.md).

The current 140-configuration registry passed 17 CPU catalog/codegen tests.
[`jit_optimized140`](../../../../reports/opus_parameterized745_20261009T164517Z/jit_optimized140/build_result.json)
has built successfully. Its [public9139 CPU code audit](../../../../reports/opus_parameterized745_20261009T164517Z/jit_optimized140/public9139_code_audit.json)
matches the local FixedK prototype's 15920 function bytes, descriptor and
resources: 86 VGPR, 47 SGPR, zero spills/scratch, 96-byte kargs, and 155704 bytes
dynamic LDS at K7168. Old public 9030 code is unchanged. **Public 9139 GPU numerical
and performance acceptance is pending.** The local
[delta13 screen](../../../../reports/opus_parameterized745_20261009T164517Z/lds_ring64x128_w4x2_fixedk7168_delta13/analysis.json)
passed 335 measurement prechecks and 17085 retained-output checks. It improved
runtime 9030 medians on 13/13 shapes but beat the strongest fixed BMM median on 2,
the BMM envelope median on 1, and the envelope in all five rounds on none.
Local-prototype measurements and CPU byte identity do not substitute for the
public-entry gate or update the full 745-shape tally.

At the earlier 139-configuration stage, public 9138 had matching device instruction bytes and
resources to its validated report prototype. Its public-launch acceptance passed
228 numerical checks, three unsupported-shape rejections, and all 9996 retained
outputs in 196 timing rows. Both representative shapes beat their old public
OPUS candidate in all 14 rounds, but beat the BMM round envelope in only 10/14
and 13/14 rounds. These local results do not update the full745 tally; see
[public-launch audit](../../../../reports/opus_parameterized745_20261009T164517Z/hybrid_192x256_public139/performance_all_bmm/audit.md).

Performance work follows separately: capture PMC and ATT on representative
shapes, identify the limiting resource, change one configuration/policy at a
time, then validate numerical results and compare timings with FlyDSL BMM using
the same measurement method.
