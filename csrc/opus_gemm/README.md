# OPUS GEMM C++ and code generation

The user-facing API guide lives in
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). This document
describes the generated launch ABI and the C++ side of a16w16 dispatch.

## Supported architecture families

The a16w16 path is implemented for gfx950, gfx942, and gfx1250. Kernel ids are
architecture-scoped; an integer id is never sufficient without
`(arch, family)`.

| Architecture | a16w16 implementation | External workspace |
|---|---|---|
| gfx950 | split-barrier, flatmm, persistent, two-stage flatmm split-K | fp32 for two-stage kids |
| gfx942 | WKC/direct and several two-stage split-K pipelines | fp32 or bf16, declared by the actual instance |
| gfx1250 | cluster/TDM two-stage split-K | fp32; batch must be 1 |

The generic `opus_gemm()` entry still owns the existing gfx950 a8w8 paths.
Its bf16 a16w16 branch is intentionally disabled: a16w16 must first resolve an
actual kid in Python so a correctly typed Torch workspace can be supplied.
The gfx942 a8w8 tune entry and the separately built a8w4 MoE modules are not
part of the a16w16 workspace ABI.

## Source layout

| Path | Role |
|---|---|
| `opus_gemm.cu` | Raw entries and strict architecture routers |
| `opus_gemm_common.py` | Canonical per-architecture instance registry and subset-compile kid sets |
| `gen_instances.py` | Cross-architecture codegen driver, manifests, and dispatch-table emission |
| `codegen/gen_instances_gfx*.py` | Architecture-specific host launcher emission |
| `include/opus_gemm_common.cuh` | Family-neutral checked extent and physical workspace validation helpers |
| `include/gfx*/opus_gemm_arch_*.cuh` | Per-architecture non-workspace/workspace id tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | Direct-pointer kernel argument types |
| `include/gfx*/**/splitk_reduce*.cuh` | Architecture-specific reduce kernels |

Generated output contains one fused host TU per selected architecture, one
device TU per `(kid, output specialization)`, and one reduce TU per
architecture that has two-stage kernels.

## a16w16 dispatch contract

Selection policy is Python-owned and runs in this order:

```text
explicit kid
  -> tuned CSV row
  -> per-architecture Python heuristic
  -> framework fallback
```

The selected object records requested kid, actual kid, allocation split-K,
and launch split-K. In particular, gfx942 resolves legacy bf16-workspace
redirects before allocation:

```text
non-exact N: 10210 -> 10200, 10213 -> 10203, 10216 -> error
exact N:     10210 / 10213 / 10216 remain unchanged
```

The C++ raw tune entry is:

```cpp
void opus_gemm_a16w16_tune(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kernelId,
    int splitK);
```

Codegen emits two distinct function-pointer tables:

- non-workspace launchers retain the five-argument launcher ABI
  `(XQ, WQ, Y, bias, splitK)`;
- workspace launchers use the six-argument launcher ABI
  `(XQ, WQ, Y, workspace, bias, splitK)`.

The tables are never type-punned or merged. The architecture router asks the
generated workspace table whether a kid is a workspace kid. A workspace-table
hit requires `workspace`; a non-workspace-table hit requires
`workspace=None`. Shape lookup tables are policy probes that return a kid,
not launch-ready five-argument pointers for split-K kernels.

Subset compilation always unions CSV ids with
`HEURISTIC_DEFAULT_KIDS_BY_ARCH`; therefore Python cannot select a heuristic
kid omitted from the shared object.

## Caller-owned Torch workspace

Workspace ownership is entirely outside C++:

```text
Python actual-kid resolution
  -> family planner computes typed extents
  -> torch.empty for this call, or validate explicit Tensor
  -> raw C++ entry
  -> main kernel receives ptr_ws directly
  -> reduce kernel receives the same direct pointer
```

There is no registry, handle mirror, stream-keyed scratch map, raw HIP
allocation, capture query, or device synchronization in this path. C++ never
retains the Tensor or pointer after the call. PyTorch's allocator owns stream
safety and graph-private memory.

Each generated workspace launcher independently enforces the final physical
contract after its local split-K clamp:

- workspace is present;
- workspace device id equals `XQ.device_id`;
- dtype equals the instance's `D_WS` (`bf16` or `fp32`);
- storage is contiguous;
- `data_ptr()` satisfies the instance alignment;
- checked `size_t` extent products fit, and `required_numel <= numel`.

Element count is not used as a substitute for dtype validation. Every extent
product and required byte-span product is overflow-checked before launch.
gfx1250 also rejects `batch != 1` in the generated launcher as a final C++
defense.

## gfx942 uniform direct pointer

gfx942 retains a device helper that splits the 64-bit direct pointer and uses
`__builtin_amdgcn_readfirstlane` for both halves. Main and reduce kernels call
this helper; it no longer reads a handle or device mirror. Do not remove the
uniformization merely because the kernel argument itself is direct. Any such
optimization requires a separate ISA, register, and performance study.

The Step 6 cross-compile comparison against the pre-direct-pointer source
confirmed:

- the representative main kernel keeps five `v_readfirstlane_b32`
  instructions, identical before and after;
- all 134 generated reduce kernels keep 276 total `v_readfirstlane_b32`
  instructions;
- main VGPR/SGPR use remains 169/96, while the obsolete handle dereference
  load disappears;
- no reduce kernel consumes a larger hardware VGPR allocation block (124 are
  unchanged and 10 improve by one four-VGPR block).

Performance still must be measured on gfx942 hardware; a gfx950 system cannot
produce a meaningful gfx942 split-K timing.

## Graph capture and concurrent streams

No OPUS-specific prewarm is required. A fresh `torch.empty` executed during
`torch.cuda.graph` capture belongs to the graph pool and is reused by replay.
Two concurrent streams receive separate call-scoped Tensor objects; OPUS has
no process-global scratch pointer to share between them.

`opus_gemm_workspace_init()` remains only as a deprecated Python no-op so old
callers do not fail during migration. It has no C++ implementation or pybind
entry.

## Code generation and validation

Generate a default multi-architecture subset:

```bash
GPU_ARCHS='gfx942;gfx950;gfx1250' \
python csrc/opus_gemm/gen_instances.py -w /tmp/opus-generated
```

The generated manifests declare six launcher parameters only for workspace
kids and five for all other a16w16 kids. Build or syntax-check every emitted
host/device TU for its target architecture after changing traits, launcher
signatures, or reduce overloads.

Run the focused regressions with:

```bash
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py
```

Runtime tests select only cases matching the installed GPU. On a single-arch
host, skipped gfx942/gfx1250 cases are coverage definitions, not evidence of
hardware validation.

Before landing workspace changes, also verify that no legacy allocator symbols
have returned:

```bash
rg -n 'SplitkWsRegistry|opus_splitk_ws_|ws_handle' \
  csrc/opus_gemm aiter/ops/opus aiter/tuned_gemm.py

rg -n 'hipMalloc|hipFree|hipHostMalloc' \
  csrc/opus_gemm/opus_gemm.cu \
  csrc/opus_gemm/codegen/gen_instances_gfx950.py \
  csrc/opus_gemm/codegen/gen_instances_gfx942.py \
  csrc/opus_gemm/codegen/gen_instances_gfx1250.py
```

README prose may mention removed names when describing migration history; code
paths must not contain them.
