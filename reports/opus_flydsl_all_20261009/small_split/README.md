Private small register split-K and fine geometry experiment, prepared on 2026-10-09. All files stay in this directory. The main Clang23 build and CPU audits pass; GPU numerical and timing validation remain pending under the GPU stop. These are candidates for future complete-call measurement, with no speedup claim or production registration.

The exact pool is 90 unique historical loser shapes: 52 register shapes from parents 9040/9041/9042/9051/9052/9053/9054 and all 38 fine shapes from parents 9060/9061/9062/9063. `coverage.csv` lists each assigned shape, current resolved actual kernel, baseline tile/waves/split/fixed-K, and six legal new candidates. `candidate_cases.csv` expands this to 540 shape/candidate cases with launch geometry, dynamic LDS, workspace, and call count. The separate compiler9070 pair covers three shapes already in this pool.

| ID | Tile M×N×K | Wave M×N×K | Global split-K | Producer VGPR / SGPR | Static LDS bytes |
|---|---|---|---|---|---|
| 110 | 16×16×128 | 1×1×1 | 4 | 66 / 54 | 0 |
| 111 | 16×16×128 | 1×1×2 | 4 | 66 / 58 | 1024 |
| 120 | 16×32×128 | 1×1×1 | 4 | 100 / 53 | 0 |
| 121 | 16×32×128 | 1×1×2 | 4 | 104 / 56 | 2048 |
| 130 | 32×32×128 | 1×1×1 | 4 | 140 / 56 | 0 |
| 140 | 32×64×128 | 1×1×1 | 4 | 215 / 61 | 0 |
| 210 | 48×64×128 | 1×4×1 | 1 | 44 / 56 | 0 + dynamic |
| 211 | 48×64×128 | 1×4×1 | 2 | 44 / 56 | 0 + dynamic |
| 220 | 64×128×128 | 2×2×1 | 1 | 90 / 65 | 0 + dynamic |
| 221 | 64×128×128 | 2×2×1 | 2 | 90 / 65 | 0 + dynamic |
| 230 | 96×128×128 | 2×2×1 | 1 | 108 / 73 | 0 + dynamic |
| 231 | 96×128×128 | 2×2×1 | 2 | 108 / 77 | 0 + dynamic |

All 12 producers and both candidate reducers have zero scratch and zero VGPR/SGPR spills. SK4/Vec16/Block128 reducer resources are VGPR59/SGPR20; SK2 uses VGPR35/SGPR18. Fine maximum dynamic LDS over the assigned cases is 65,408 / 62,272 / 109,696 / 105,536 / 130,688 / 124,480 bytes for 210 / 211 / 220 / 221 / 230 / 231 respectively. Metadata's zero fixed LDS for these producers does not include dynamic LDS.

`candidate/register_global_split.cuh` is a private copy of the frozen register pipeline; its exact change is in `candidate/register_global_split.diff`. It balances K128 tiles across four workgroups first, then one or two local K waves. It retains the P3 register queue, native K128 scale addresses, direct preshuffled B loads, and local FP32 reduction. Only the surviving local wave writes the FP32 partial. The matching frozen reducer sums four partials and converts to BF16 once. Empty partitions and empty local waves write initialized zero accumulators. No atomic counter or persistent state is used.

The six fine candidates instantiate new geometry/split combinations through the existing common fine-M LDS pipeline. They retain S4/C1, fine A loads, raw SFA/SFB handoff, early scale loads, prefetch before read, read-only drain, and output packing. These geometry candidates do not implement a direct-B hybrid; that is a separate experiment. All arithmetic and scale grouping stays K128.

`build/baseline/experiments.so` recompiles the current source choices for the eleven assigned parents, including actual 9050 and fixed 9070/9071/9072/9073 register choices, fine fixed-K 3072/7168/16384 choices, and current fine M-tail/split dispatch. It contains 26 producers and five reducers. This frozen-source baseline uses the same private raw ABI and local flags as the candidates; it is not an assertion that its code object is byte-identical to an already registered production library.

The exported `launch` ABI in `launch.hip` is `(id, a, b, sfa, sfb, c, workspace, workspace_capacity_bytes, M, N, K, stream)`, returning a HIP status. It accepts positive N/K divisible by 128, K≤16384, register M≤512 or fine M≤2048, and checked 32-bit buffer extents. A/B/C must be 16-byte aligned; fine SFA also requires 16-byte alignment, while register byte scales may be unaligned. The launcher rejects null pointers, pairwise tensor overlap, workspace overlap, short capacity, and workspace misalignment. Input A/B are FP8 bytes; output C is BF16. SFA is raw `[K/128,M]`; SFB is raw `[N/128,K/128]`; B is the existing preshuffled byte layout.

For split-K>1 the caller provides 16-byte-aligned FP32 workspace of `splitK*M*N*4` bytes. It is fully overwritten on every call and reused only after the caller's normal stream ordering completes prior users. Workspace initialization is unnecessary. The producer and matching reducer launch on the caller's stream, so a complete call includes two kernels. Split-K1 takes null workspace and zero capacity and launches one kernel. No allocation, memset, synchronization, or timing wrapper runs inside the ABI. Workspace maximum in this pool is 7,340,032 bytes for register SK4 and 49,545,216 bytes for fine SK2. Later timing must include the complete call and use an output correctness check; producer-only timing would omit required work.

`compiler9070_v2/` contains two additional unregistered controls, each instantiating exactly `<16,32,1,1,3,8,4,3,7168,false,true>`. Both use the same private `compiler9070.hip` and the production defines/codegen flags copied from the old compiler review's kid9052 build manifest, with source includes redirected to the frozen tree. Clang23 is `/opt/rocm-llvm23-46fcb339/bin/clang++`; Clang24 is the pinned `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++` with `/opt/rocm/lib/llvm/lib/clang/20`. The single-producer raw ABI accepts parent id9052 with K7168 and no workspace. Its three coverage shapes are (112,768,7168), (16,6144,7168), and (16,7168,7168). The same kernel symbol has different ISA hashes: Clang23 VGPR58/SGPR38 and Clang24 VGPR96/SGPR47, both static LDS16,384 and scratch/spills0. This establishes build/resource differences, without a performance conclusion.

Validation evidence is in `cpu_audit.json`, `cpu_audit_no_runtime/`, `coverage_audit.json`, and `compiler9070_v2/cpu_audit.json`. The CPU executable exercises the frozen C layout expressions for 12 variants over 14 M tails, the frozen fine B producer/consumer layouts, fine A/scale coverage, and actual workspace/alignment/overlap/dimension guards. Its HIP host-only compile emits an ordinary CPU object, then plain clang++ links the executable. `readelf -d` is checked before execution; dependencies are only libstdc++, libm, libgcc_s, and libc, with no HIP/HSA dependency. The host attribute adapter changes pure layout helper attributes in a separate audit include; device inputs remain frozen. A separate arithmetic audit checks K128 totals 1..128 with global SK1/2/4 and local WK1/2, including empty partitions, and the P3/S4 ring indices.

CPU layout and index checks do not establish GPU synchronization, numerical correctness, or speed. The altered split order changes FP32 accumulation order, so numerical validation is required. Added workspace traffic and a reducer launch can outweigh producer gains. Larger register tiles can increase register pressure; lower VGPR counts alone do not prove improvement. All future comparison should use the exact listed baseline dispatch and frozen input layout.

`source_manifest.json` freezes 84 current source files and records their hashes. `build_receipt.json` records commands, compiler hash/version/resource directory, source hashes, logs, object/library hashes, and production-source checks. No production source was edited. Earlier `cpu_audit_final/` is a failed helper compile, `cpu_audit_v2/` is superseded because its HIP driver link added an implicit HIP dependency, and `compiler9070/` is a retained failed full-runtime-header compile. The final evidence is the no-runtime CPU audit and compiler9070_v2 pair.

Reproduction is CPU-only: `prepare.py` and the build scripts refuse existing output directories; `audit.py --output-name fresh_audit_name` supports a new audit directory. `coverage.py` regenerates tables from the frozen comparison CSV. Do not execute or load either HIP experiment library while the GPU stop remains active.
