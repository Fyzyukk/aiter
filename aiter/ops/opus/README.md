# OPUS operators

OPUS exposes architecture-specific GEMM and MoE kernels. The GEMM surface is
family-specific: one a16w16 launch and three A8W8 launches represent four
different physical contracts.

## Current GEMM capability

| Canonical Python API | Current architecture/kid | Contract |
|---|---|---|
| `opus_gemm_a16w16_launch` | gfx942, gfx950, gfx1250 | BF16 inputs, BF16/FP32 output, optional bias and per-call Torch workspace |
| `opus_gemm_a8w8_launch` | gfx950 kid 2 | FP8 inputs, FP32 output, no scale |
| `opus_gemm_a8w8_blockscale_launch` | gfx950 kid 1 | FP8 inputs, FP32 output, plain WQ and two FP32 block scales |
| `opus_gemm_a8w8_blockscale_bpreshuffle_launch` | gfx942 kid 11000 | FP8 inputs, BF16 output, pre-shuffled WQ and two FP32 block scales |

The blockscale-bpreshuffle Python, pybind and C++ symbols also exist on gfx950
and gfx1250. Their OPUS capability tables are currently empty, so calling that
API reports that the current architecture has no registered kernel. This does
not affect the high-level AITER router's CK, CKTile, ASM, FlyDSL, Triton or
Gluon backends.

## a16w16 high-level API

```python
import torch
from aiter.ops.opus import gemm_a16w16_opus

A = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
B = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

Y_bf16 = gemm_a16w16_opus(A, B)
Y_fp32 = gemm_a16w16_opus(A, B, dtype=torch.float32)
```

The shape-driven API is:

```python
gemm_a16w16_opus(
    A,
    B,
    bias=None,
    dtype=torch.bfloat16,
    *,
    kernelId=None,
    splitK=None,
    out=None,
)
```

Its input rules are:

- `A` is BF16 `[M,K]` or `[batch,M,K]`;
- `B` is BF16 `[N,K]` for batch one, or a real `[batch,N,K]` allocation;
- an expanded batch-stride-zero weight is rejected;
- output is BF16 or FP32 and has shape `[M,N]` or `[batch,M,N]`;
- `out`, when supplied, must have the exact contiguous output shape;
- XQ/WQ launch tensors are K-contiguous and may use a padded leading row
  stride where the launcher passes that stride explicitly;
- current a16w16 launchers reject odd K, and individual instances may impose
  stricter tile or prefetch requirements;
- current gfx1250 a16w16 launchers require batch one.

`kernelId` and `splitK` are compatibility names on this high-level API. An
explicit request is strict and passes through the same selector legality rules
as tuned and heuristic choices.

## Canonical family launch APIs

The canonical explicit interfaces use `kid` and `split_k` terminology:

```python
opus_gemm_a16w16_launch(
    XQ, WQ, Y, bias=None, *, kid, split_k=0, workspace=None
)

opus_gemm_a8w8_launch(
    XQ, WQ, Y, *, kid=2
)

opus_gemm_a8w8_blockscale_launch(
    XQ, WQ, Y, x_scale, w_scale, *, kid=1
)

opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    XQ, WQ, x_scale, w_scale, Y, *, kid=None
)
```

The a16w16 public explicit and shape-driven paths use a private ctypes C ABI
binding in the same mixed module as the retained pybind raw endpoint. The C
ABI forwards to the same canonical checked C++ launcher; it does not duplicate
selection, dispatch, or workspace validation. It switches to the XQ HIP device
and live PyTorch stream for the call, restores the previous device/stream, and
converts C++ exceptions to Python `RuntimeError`. The pybind raw remains an
internal compatibility and A/B endpoint.

All private raw bindings receive a resolved integer kid. In particular,
`opus_gemm_a16w16_launch` does not bypass gfx942 requested-to-actual kid
redirects, and the bpreshuffle wrapper resolves `kid=None` before entering C++.

The A8 family adapters cache only immutable metadata: the gfx architecture is
keyed by the explicit `torch.device`, and successful exact
`(arch,family,kid,Y.dtype)` capability checks are memoized. Failed capability
checks are not cached. Tensor objects, data pointers and streams are never
cached; Tensor type/same-device checks still run on every public call, and the
C++/generated launcher remains the final physical-contract validator.

## Runtime selection policy

The a16w16 runtime policy lives only in Python:

```text
explicit kid
  -> OPUS tuned CSV row
  -> per-architecture Python heuristic
  -> framework fallback
```

Every successful OPUS choice becomes one `LaunchConfig` containing requested
and actual kid plus requested, allocation and launch `split_k`. Explicit,
tuned and heuristic choices all call the same legality function. A stale tuned
row is discarded atomically with its split value.

gfx942 resolves legacy BF16-workspace redirects before allocation:

| Requested kid | N outside `{64,128,256,384,512,1024,2048}` |
|---:|---:|
| 10210 | actual kid 10200 |
| 10213 | actual kid 10203 |
| 10216 | rejected |

The blockscale-bpreshuffle wrapper uses a narrower Python policy:

```text
explicit kid
  -> current-shape tuned row when libtype == "opus"
  -> per-architecture Python default
  -> no-registered-kernel error
```

The current defaults are gfx942 `11000`, gfx950 `None`, and gfx1250 `None`.
C++ does exact-kid table lookup only; it does not read shape rows or run a
heuristic.

## Build-time subset compile

Tuned CSV files still participate at build time. The generator extracts OPUS
kids from `solidx`/`kernelId` and unions them with the compiled-kids sidecar,
Python heuristic defaults and mandatory A8 kids. This set controls which
instances are compiled; CSV shapes are not emitted as C++ runtime policy.

Generated dispatch is written to:

```text
opus_gemm_a16w16_kid_dispatch.h
opus_gemm_a8w8_kid_dispatch.h
```

Tables are architecture-, family- and output-dtype-scoped and perform exact
kid lookup.

## Task1 Torch workspace contract

Workspace ownership is call-scoped and Python/Torch-owned:

```text
resolve actual kid and split_k
  -> derive shape/dtype from the canonical instance
  -> use caller Tensor or torch.empty for this call
  -> generated launcher validates the final physical contract
  -> launch
```

There is no OPUS Tensor cache or raw HIP allocation. Let
`padded_M=ceil_div(M,B_M)*B_M` and
`padded_N=ceil_div(N,B_N)*B_N`. Current layouts are:

| Architecture/family | Workspace shape | Workspace dtype | Kids |
|---|---|---|---:|
| gfx950 two-stage | `[allocation_split_k, batch, padded_M, padded_N]` | FP32 | 48 |
| gfx942 two-stage | `[allocation_split_k, batch, padded_M, padded_N]` | exact instance: 3 BF16, 5 FP32 | 8 |
| gfx1250 two-stage | `[allocation_split_k, padded_M, padded_N]` | BF16 | 496 |
| gfx1250 fused | `[tiles_m, tiles_n, fuse_split_k - 1, B_M, B_N]` | exact instance: 780 BF16, 598 FP32 | 1378 |

gfx1250 fused is already registered and generated. Its tile-major layout and
compile-time `fuse_split_k` are not interchangeable with the two-stage
split-major layout or a runtime CSV split value.

An explicit caller workspace must be on the XQ device, contiguous, aligned,
of the exact instance dtype, and large enough after the launcher's final
split clamp. A non-workspace kid requires `workspace=None`. Bias behavior is
unchanged from Task1:

- bias is contiguous `[N]` or `[batch,N]`;
- gfx950/gfx942 accept bias only on bias-aware kids;
- gfx1250 reduce accepts FP32 bias with BF16 Y as well as bias matching Y;
- unsupported bias is never silently dropped.

## gfx950 A8W8 contracts

### No scale, kid 2

```text
XQ: FP8 [batch,M,K]
WQ: FP8 [batch,N,K]
Y:  FP32 [batch,M,N]
```

XQ/WQ are K-contiguous and Y is contiguous. The generated launcher verifies
matching batch/M/N/K and preserves its existing requirements:
`ceil_div(K,B_K) >= 2`, an even number of K-tile loops, and even K.

### Plain-WQ blockscale, kid 1

This family has no bias or `group_layout` argument. Both scales are mandatory,
FP32, contiguous and on the XQ device. The group contract is 1x128x128:

```text
x_scale: [batch,M,K/128]
w_scale: [batch,N/128,K/128]
```

For batch one, `[M,K/128]` and `[N/128,K/128]` are accepted. N and K must be
divisible by 128. The launcher rejects fewer than two K tiles and an odd K-tile
count before device launch so the pipeline cannot form a negative final tile
or perform an out-of-range prefetch.

## gfx942 blockscale-bpreshuffle contract

Current kid 11000 accepts 2D tensors or explicit 3D tensors with batch one:

```text
XQ:      FP8  [M,K] or [1,M,K]
WQ:      FP8  [N,K] or [1,N,K], already pre-shuffled
Y:       BF16 [M,N] or [1,M,N]
x_scale: FP32 [M,K/128], transposed physical scale storage
w_scale: FP32 [N/128,K/128], row-major
N % 128 == 0
K % 128 == 0
```

All tensors are contiguous and on one device. Crucially, bpreshuffle is a WQ
content semantic: shape, dtype and strides cannot prove that the bytes were
actually shuffled. Callers must pass the result of the required weight
transformation, currently `shuffle_weight(WQ, layout=(16, 16))`. Numerical
tests compare that real shuffled input against an unshuffled reference weight.

## Compatibility window

Two old Python names remain for one migration window:

- `opus_gemm_a16w16_tune` parses legacy positional arguments and
  `kernelId`/`splitK`, emits one `DeprecationWarning`, then calls
  `opus_gemm_a16w16_launch`;
- `opus_gemm_a8w8_blockscale_bpreshuffle_tune` preserves the gfx942
  `Y=None, kernelId=11000` behavior, emits one warning, then calls the
  canonical bpreshuffle launch.

There are no corresponding C++ or pybind compatibility entries.
`opus_gemm_workspace_init()` is a deprecated Python no-op and is not required
for graph capture or normal launch.

## Graphs, streams and testing

`torch.empty` during CUDA graph capture uses the graph-private pool, and each
concurrent stream call owns its workspace Tensor. OPUS keeps no process-global
scratch pointer.

Focused tests:

```bash
pytest -q \
  op_tests/test_opus_interfaces.py \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py
```

Architecture-gated skips define coverage but are not evidence that another
GPU architecture passed.

## File map

| Path | Purpose |
|---|---|
| `_selector_a16w16.py` | Python runtime policy, redirects and shared legality rules |
| `heuristics/a16w16_gfx*.py` | Per-architecture Python heuristic implementations |
| `gemm_op_a16w16.py` | High-level API, canonical launch and direct Task1 workspace allocation |
| `gemm_op_a8w8.py` | Three canonical A8W8 family wrappers and gfx942 compatibility shim |
| `../../../csrc/opus_gemm/` | Canonical registry, C++ family routers, codegen, traits and pipelines |
