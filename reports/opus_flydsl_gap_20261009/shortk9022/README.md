# Private 9022 short-K panel8 prototype

Status: **offline build and CPU audit passed; numerical and performance validation not run**.
GPU testing remains stopped. This candidate is private and unregistered; it does not change
production headers, dispatch, the formal tuning CSV, or any previous report/library.

The candidate preserves the current 160×128×128, four-wave, two-matrix-stage 9022
pipeline and its ordering. It makes K384 and K768 independent template instantiations
(three and six K128 iterations), replaces the 32-column resident scale panel with eight
columns, and removes the unused second SFA layout/pass construction. The original
runtime pipeline is frozen byte-for-byte under `frozen/gemm_include/`.

This is not the previously rejected global scale issue/publish candidate. Matrix issue,
scale request/publication ordering, MFMA ordering, LDS barriers, and BF16 output shuffle
remain those of the current runtime pipeline. A selectable runtime fallback remains in
the separate local control library; the private candidate's C ABI rejects any K other
than 384 or 768.

| Offline Clang23 kernel | VGPR | SGPR | AGPR | LDS bytes | Scratch bytes | VGPR/SGPR spills | ISA bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| Current runtime 9022, rebuilt local control | 194 | 62 | 0 | 81,184 | 0 | 0 / 0 | 6,696 |
| Private fixed K384, panel8 | 176 | 46 | 0 | 77,320 | 0 | 0 / 0 | 4,928 |
| Private fixed K768, panel8 | 176 | 46 | 0 | 77,320 | 0 | 0 / 0 | 4,908 |

These are emitted static resource metadata and complete ELF function sizes. They do
not measure speedup, dynamic occupancy, bandwidth, or which pipeline stage limits
execution. The original runtime already fits the 80 KiB LDS budget; reducing LDS
does not by itself prove a new residency level. The runtime guards already skip inactive
K columns, so panel8 does not reduce the active scale bytes fetched at a fixed K.

The local baseline is **the current frozen runtime source recompiled with the same
Clang23 flags as the candidate**. It is not the historical formal `.so`, and no machine-code
identity with that formal module is claimed. The existing formal Clang23 library is only
a read-only hash reference in `source_manifest.json` and `build_receipt.json`.

`launch.hip` exposes the existing private-builder-style C ABI
`launch(9022, a, b, sfa, sfb, c, m, n, k, stream)`. It uses contiguous FP8 A/B, the standard
16×16 B preshuffle, column-major logical `[M,K/128]` native E8M0 A scales, row-major
`[N/128,K/128]` B scales, and BF16 output. It enforces M divisible by 16, N divisible by
128, exact candidate K, positive dimensions, A/B/C signed-32-bit byte extents, nonnull
pointers, and 16-byte A/B/SFA/C alignment. SFB retains the parent byte-alignment contract.
Caller-owned device pointers and nonoverlapping buffers remain required; this raw private
ABI does not provide tensor metadata checks.

Offline compilation uses `/opt/rocm-llvm23-46fcb339/bin/clang++`, revision
`46fcb339fb61119b337f973c7ca9e710a319fdd0`, its own resource directory, ROCm SDK
`/opt/rocm`, and the production backend scheduling flags. The complete command lists,
compiler version/SHA, frozen source hashes, object/library hashes and logs are in
[build_receipt.json](build_receipt.json). The two local libraries are
`build/baseline/experiments.so` and `build/candidate/experiments.so`; neither was loaded.

The [CPU audit](cpu_audit.json) parses the generated gfx950 ELF metadata and function
bytes, then runs a host-only executable against the actual frozen SFA/SFB address-layout
expressions. It checks both fixed K values, 14 M sizes including 16/144/160/176/304/320/336
tail boundaries and 544/672/800/864/992/1024/1088 representatives, producer once-coverage,
consumer addresses, identity-scale tail bytes, SFB once-coverage, and actual launcher
guards. A separate audit-only header adds host attributes to the pure layout helpers;
its recorded diff changes attributes only, not arithmetic. The HIP build uses the unchanged
frozen OPUS headers. The host check does not call HIP runtime functions or dereference
synthetic device-pointer values.

Artifacts for the final successful audit are in `cpu_audit_final/`. Preparation failures
are preserved in the earlier `cpu_audit*` directories: the first parser expected the old
`hip-` bundle spelling instead of Clang23's `hipv4-`; the next check assumed rounded LDS
metadata instead of the emitted exact 77,320 bytes; the third host-only attempt lacked
host attributes on pure layout helpers. `cpu_audit_v4/` passed; the final audit narrows
the host attribute adapter and adds SFB once-coverage. These setup corrections never
changed either compiled HIP library.

Review entries are [candidate/pipeline.diff](candidate/pipeline.diff),
[candidate/traits.cuh](candidate/traits.cuh), [launch.hip](launch.hip),
[contract.h](contract.h), [source_manifest.json](source_manifest.json), and
[cpu_audit.json](cpu_audit.json). `prepare.py` and `build.py` refuse existing outputs;
`audit.py --output-name <fresh-local-name>` can create an independent CPU audit.
No script in this directory schedules or runs GPU validation.

Before use, both specializations still require signed/cancellation numerical checks,
output/repeat/guard validation, and matched complete-call timing against the frozen
runtime and corresponding FlyDSL configurations on an authorized exclusive GPU.
Until GPU testing is resumed explicitly, the prototype remains unvalidated and cannot
support a performance claim or production selection.
