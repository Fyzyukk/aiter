# MXFP8 B-preshuffle: five parameterized pipelines

The production source now exposes five pipeline candidates: `pin`, `tiled`,
`register`, `lds`, and `large_output`. Geometry, stage/queue depth, scale layout,
B movement, output policy and schedule are compile configuration parameters.
This implements the requested few-pipeline organization while preserving the
existing parameter search space and saved tuning tables.

| Pipeline | Active compile configurations | Registered compatibility configurations |
|---|---:|---:|
| pin | 15 | 15 |
| tiled | 20 | 20 |
| register | 13 | 18 |
| lds | 38 | 49 |
| large_output | 3 | 3 |
| Total | 89 | 105 |

Five source pipelines do not imply five compiled specializations. The 16
additional configurations remain available for historical exact-ID replay.
Nine runtime configurations accept multiple global split counts without a
producer specialization for each count. Static split configurations keep their
historical behavior. Old IDs are internal ABI compatibility keys; new callers
can use pipeline and parameters.

The kernel/traits source is in [include/gfx950](../../csrc/opus_gemm/include/gfx950).
Compute headers decreased from **24 to 5** and traits headers from **19 to 1**.
Including shared ABI/layout/output/reduction helpers, B-preshuffle headers
went from **47 to 11**. Compute source decreased from **8,223 to 3,015 lines**;
the complete header source decreased from **9,838 to 4,950 lines**. The new
headers contain shared computations and compile-time branches; they do not
include the deleted compute files. The tiled pipeline retains three distinct
internal schedule helpers and five compile schedule policies because its
load/scale/wait representation varies.

## Configuration selection and tuning

The scalar [configuration catalog](../../csrc/opus_gemm/opus_gemm_bpreshuffle_config.py)
provides `pipeline_configs((M,N,K), pipelines=...)`, `resolve_config(...)`, and
frozen `BpreshuffleConfig` values. The [configuration map](configurations.csv)
contains all 105 parameter payloads and their compatibility keys.

```python
from aiter.ops.opus import opus_gemm_bpreshuffle

opus_gemm_bpreshuffle(
    XQ, WQ_shuffled, Y, x_scale, w_scale,
    pipeline="register", tile_m=16, tile_n=16, wave_k=1, split_k=3,
)
```

The example resolves a registered tuple without requiring the caller to look
up its numeric ID. It requires a supported shape and at least three K128 tiles.
A selected config object, full mapping or canonical JSON payload can be passed
with `config=...`. Partial named parameters must identify exactly one tuple;
unsupported or ambiguous tuples are rejected. The catalog exposes existing
compiled configurations and does not create arbitrary new JIT tuples.

The [tuner](../../csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py) supports
`--opus_pipelines register,lds`. It filters compile configurations by shape,
measures each legal configuration/split combination and saves the winning
`pipeline`, canonical JSON `config`, and literal `splitK`. It retains
`kernelId` for internal ABI replay. Old CSVs are backfilled with metadata;
new metadata is checked against the exact configuration before compilation.
The historical 745-shape production CSV was not rewritten or remeasured.

## FlyDSL BMM reference

The gfx950 [FlyDSL BMM kernel](../../aiter/ops/flydsl/kernels/bmm_a8w8_mxscale_gfx950.py)
has one JIT factory at line 268 and one nested kernel body at line 400.
N/K/batch/tile/wave/buffers/split/B movement/scale/layout are `Constexpr`
parameters; M and pointers are runtime arguments. Its
[wrapper](../../aiter/ops/flydsl/batched_gemm_a8w8_gfx950.py) caches compiled
launchers by the complete constexpr tuple, with tuned selection or shape
fallback. Its seven fallback M tiers are not seven total configurations.

Static inspection of `dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv`
found 680 gfx950 FlyDSL BMM rows and 212 distinct configuration names.
`dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv` contains 1,055 BMM rows
and 408 distinct names. Source reuse and compile specialization counts are
separate there as well. FlyDSL split is compile-time; these nine OPUS runtime
configurations retain launch-time global split-K.

## Validation and limits

**100 CPU regression checks passed** across the configuration catalog,
registry, runtime split, mixed compiler and transactional JIT cache suites.
The [summary](summary.json) records the exact command. Frozen launchers compare
byte-for-byte after only the intended compute-header names, producer symbols,
traits wrappers and explicit runtime ABI include changes. Shape, pointer,
workspace, grid, dispatch and split contracts remain checked.

The [tiled source review](tiled_source_review.md) and
[repeatable projection checks](check_tiled_source_projection.py) preserve
independent frozen source oracles. Eight layout/body projections passed;
narrow schedule 3/4 also passed independent source review. Source review
caught and fixed a `v_sfb_next` shadowing bug before final validation. The
fused-host compilation caught a missing explicit runtime-kargs include, which
was fixed and recompiled. Intermediate receipts remain available.

All **105 configurations compiled as fresh HIP objects** with the existing
90 LLVM23 / 15 pin-AGPR LLVM24 compiler split. The affected 20 tiled
configurations were recompiled after the fixes. Fused host, actual router and
complete pybind translation unit compiled. A shared `--no-undefined` link
resolved all **127 distinct launch stubs**. The pybind object was checked
separately because unrelated BMM bindings are outside this B-preshuffle link.
The [final build scope](full_build/verification_scope.json) records source and
receipt hashes.

For 157 emitted kernel instances, 39 instruction byte arrays and 128 metadata
entries excluding private names are identical to the prior offline artifacts;
29 instances change resource allocations. All have zero private scratch and
zero VGPR spill. Existing 9023 and 9024 base instances each retain 30 SGPR
spills; the runtime producer/reducer instances have zero spills. Source
consolidation can change scheduling and register allocation, so it does not
establish numerical equivalence or a performance improvement.

The separate historical retained experiments also passed a compile-only
check of nine translation units. All 38 original manifest source hashes
match, and the historical builder, audit and manifests are unchanged. The
[snapshot rebuild wrapper](rebuild_retained_snapshot.py) uses frozen headers
ahead of the current include directory; its [receipt](retained_compile/verification_receipt.json)
records the result. The original compiler and nine historical shared
libraries are absent here, so this check establishes source compatibility
with the available compiler and makes no historical binary equivalence claim.

**GPU tests remain stopped at the user's request.** No device query, GPU
initialization, shared-library loading, kernel execution, numerical test,
benchmark or tuning was performed in this refactor. The examples are future
calls and were not executed. Numerical/performance validation is pending.
Prior reports, measured CSVs and historical artifacts are preserved.

The local report includes offline objects and linked artifacts. The Git
publication contains source, metadata, receipts and text logs; binary objects
and shared libraries are kept locally. The [publication manifest](publication_manifest.json)
lists published text and excluded binaries. Receipt artifact hashes describe
the local verification outputs.
