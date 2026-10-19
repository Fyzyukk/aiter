# gfx950 MX-scale B-preshuffle pipelines

Each pipeline has an explicit execution schedule, device entry and header.
Legal tile dimensions, M/N wave layouts, storage depth, cache policy, addressing
and output choices remain compile-time parameters. All retained configurations
use WaveK=1. Shared traits and layout/fragment helpers are allowed; complete
alternate main loops have distinct pipeline identities.

## Current large-shape pipelines

The latest round58 batch contains 294 shapes with M > 1024. Its selected kernel
IDs use the four pipelines below. Counts are selections, including both wins
and losses against the external references.

| Pipeline | Header suffix | Fixed execution schedule | Selected shapes | Registered configurations |
| --- | --- | --- | ---: | ---: |
| `pin_agpr` | [`4wave_pin_agpr`](opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh) | Pinned A/B/C quadrant schedule; groups 2–4 before the matrix wait, group 5 before the barrier, groups 6–7 after it | 136 | 53 CO |
| `prefetch_b_vgpr` | [`prefetch_b_vgpr`](opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_prefetch_b_vgpr_gfx950.cuh) | A LDS ring and B register ring; read current A, issue future A, consume current operands | 81 | 56 CO + 2505 JIT |
| `pingpong_lds` | [`pingpong_lds`](opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_pingpong_lds_gfx950.cuh) | A/B LDS panels with operand reads overlapped with MFMA | 69 | 46 CO |
| `prefetch_b_vgpr_overlap` | [`prefetch_b_vgpr_overlap`](opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_gfx950.cuh) | Retain A across iterations and replace fragments after their last MFMA consumer | 8 | 6 CO |

Headers use the prefix `opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_` and suffix
`_gfx950.cuh`. Each pipeline has a matching `opus_gemm_traits_...` header;
its layout, fragment-loading and output helpers live in the pipeline header.
The launch ABI is shared in
[`opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh`](opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh).
The overlap traits inherit the ordinary B-prefetch geometry and add their own
publication-point constraint.

`pin_agpr_early`, `register_prefetch` and `lds_ring` had zero final selections
in round58. Their headers, exclusive traits and 3597 registered configurations
were removed from this worktree, together with the four retired early-MFMA CO
images. The resulting registry has 161 CO and 2505 JIT configurations. Registration
is not a claim that every configuration has passed numerical validation; known
invalid, unselected ID 9085 is unchanged.

## Flow identity and tune parameters

Historical `quadrant` and `tiled` labels now classify the retained regular pin
schedule as `pin_agpr`. The overlap entries are IDs 9457–9462. The `EarlyMfma`
and `OperandOverlap` schedule switches are absent. The registry selects an
explicit header and device body.

Overlap traits derive the geometry/resource contract from B-direct traits and
add `PublishM`, a legal publication point within that schedule. Pin A/C in
B-direct remains a register-placement parameter. Stage depth and publication
point obey each flow's legality constraints. Changing the complete load/MFMA/
barrier schedule requires a distinct pipeline identity, not a pipeline per shape.

IDs 9000/9001 retain the 256×256×128, 2×2-wave regular pin configurations,
without/with padded M. The generalized body also supports other registered
configurations; it is not a verbatim copy of the specialized original reference.

## Registry, build and replay

[`co_kernels_gfx950.json`](../../gen_co/co_kernels_gfx950.json) binds each CO ID
through `pipeline`, `pipeline_header`, `body_func`, traits and body arguments.
`build_co.py` consumes that binding directly. The remaining JIT domain in
[`opus_gemm_common.py`](../../opus_gemm_common.py) keeps B-direct IDs 39585–42089.
Removed IDs stay vacant. Every retained ID, symbol, launch contract and parameter
configuration is unchanged, including all 294 round58 selections.

`name_family` and exported symbols retain historical spellings for replay; use
`pipeline` / `bpreshuffle_pipeline(instance)` to classify execution flows. Frozen
reports and historical CO images retain their original content. No new tuning
or performance result is implied by removing unused pipelines.

The shared tuner continues to group candidates by pipeline and geometry. Unused
per-source compiler patterns for retired B-preshuffle generated filenames were
removed; effective flags for retained configurations are unchanged.

## Historical validation before the current registry

The following record describes earlier registry revisions and their original
measurements. Its old configuration counts, names and file layouts do not describe
the current organization above.

### Validation and subsequent optimization

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
