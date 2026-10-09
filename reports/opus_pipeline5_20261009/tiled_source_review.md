Tiled source review completed on 2026-10-09 against the frozen sources in
`before/csrc/opus_gemm/include/gfx950`. No additional semantic mismatch was
found after the narrow B-scale variable scope fix and the restoration of
immediate scale loads for schedule 2.

This review used file reads and pure Python source projection only. It did
not import a GPU package, query a device, compile C++, load a generated
library, or execute a kernel. The build and device-resource checks performed
by the separate header auditor are documented in `full_build/`.

The repeatable check is `python3 reports/opus_pipeline5_20261009/check_tiled_source_projection.py`.
Its captured output is `tiled_source_projection.json`; the file hashes bind
this receipt to the reviewed production sources. All eight projections
passed. The checks ignore comments and whitespace, select the active
compile-time schedule, remove unused policy lambdas, and normalize the
approved fixed/runtime loop and equivalent per-pass tuple declarations.
They retain all active loads, guards, waits, barriers, MFMA instructions,
register assignments, and epilogues.

| Schedule | Frozen producer body | Review result |
| --- | --- | --- |
| 0 | 8wave 192x256, historical 9020 | Entire compute body is token-exact: 3,901 tokens. |
| 1 | 4wave 128x128, historical 9021 | Active projected body is token-exact: 3,064 tokens. Tail-safe byte loads and issue/publish scheduling remain intact. |
| 2 | Geometry, 4wave 160x128, and short-K | Each active projected body is token-exact after approved loop/pass substitutions: 2,865 tokens. Immediate scale loads are retained. |
| 3 | 4wave 64x128 | Independent read-only review by the config agent found matching guards, packing, waits, fragment scheduling, and epilogue. |
| 4 | 4wave 64x64 | Independent read-only review by the config agent found matching guards, packing, waits, vector scheduling, and epilogue. |

The three extracted layout namespaces are also token-exact against their
frozen definitions: 8wave 192x256 (1,447 tokens), 4wave 128x128 (1,387), and
4wave 64x128 (468). Schedule 1's single frozen pass and schedule 2's explicit
one/two-pass tuple are generalized to a compile-time tuple of the same
per-pass layout helpers. Those helpers preserve the original pass, wave,
K-group, and row-vector starts.

Scale issue and publish functions use identical panel guards. Schedule 1
still calls the frozen `load_sfa_vector` helper, which fills invalid rows
with raw E8M0 byte `0x7f` and uses scalar reads for partial or unaligned
vectors. Schedule 2's active immediate load/store lambdas retain the frozen
row guard and source/load/store sequence. No shared raw scale arrays are
used on that active path.

For the narrow schedules, the issue/publish choice remains schedule 3 with
runtime K and schedule 4 with fixed K 7168; the complementary cases retain
the immediate path. The next B scale is assigned to one outer
`v_sfb_next` variable in either compile-time branch before the subsequent
iteration. Schedule 3's interleaved fragment replacement and schedule 4's
full-vector replacement match the frozen bodies. The schedule 3 swizzle
uses the new default `BLOCK_GROUP_M=0`, preserving its original mapping.
Schedule 4 keeps unroll count 4 through the equivalent Clang pragma.

The combined pure CPU suite passed all 40 tests: registry 12, runtime
split-K 20, and pipeline configuration 8. The suites cover historical
catalog/source contracts, launch metadata, split partitioning, CSV replay,
public-wrapper forwarding, and the five-header/five-producer invariant.
They do not validate numerical output or performance on hardware.
