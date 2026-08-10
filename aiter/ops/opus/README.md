# OPUS operators

OPUS provides architecture-specific GEMM and MoE kernels. This guide focuses
on the plain-layout bf16-input a16w16 GEMM and its call-scoped Torch workspace.

## a16w16 quick start

```python
import torch
from aiter.ops.opus import gemm_a16w16_opus

A = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
B = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

Y_bf16 = gemm_a16w16_opus(A, B)
Y_fp32 = gemm_a16w16_opus(A, B, dtype=torch.float32)
```

Supported a16w16 architectures are gfx950, gfx942, and gfx1250. Unsupported
architectures receive a callable stub with a clear runtime error rather than
an import-time failure.

The high-level API is:

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

Inputs and output follow these rules:

- `A` is bf16 `[M, K]` or `[batch, M, K]`.
- `B` is bf16 `[N, K]` when `batch == 1`, or a real contiguous
  `[batch, N, K]` allocation. An `expand()` view with batch stride zero is not
  safe for the launcher and is rejected.
- output dtype is bf16 or fp32.
- the return shape is `[M, N]` for 2D `A`, otherwise `[batch, M, N]`.
- `out`, when supplied, must have the exact contiguous output layout.
- current a16w16 launchers reject odd `K`; individual pipelines may impose a
  larger tile or prefetch minimum.
- gfx1250 currently requires `batch == 1` in both Python selection and the
  generated C++ launcher.

`kernelId` and `splitK` are advanced overrides. An explicit kid is validated
strictly for the current architecture and shape; it never silently falls back
to a different family. `splitK` is meaningful only with an explicit kid or a
tuned CSV row.

## Selection happens before allocation

Every high-level call follows one policy sequence:

```text
explicit kid
  -> tuned CSV row
  -> architecture-specific Python heuristic
  -> framework fallback
```

The result is a `LaunchConfig` containing:

- requested and actual kid;
- requested, allocation, and launch split-K;
- architecture and family;
- selection source or framework fallback reason.

This distinction matters on gfx942. Its legacy bf16-workspace ids have an
exact-N reduce path for `N` in `{64, 128, 256, 384, 512, 1024, 2048}`. Outside
that set, resolution occurs before workspace planning:

| Requested kid | Non-exact-N result |
|---|---|
| 10210 | 10200 (fp32 workspace) |
| 10213 | 10203 (fp32 workspace) |
| 10216 | rejected |

Inside the exact-N set, including `N=384`, all three ids remain unchanged.
The planner therefore always reads the actual instance and cannot allocate a
bf16 buffer for a redirected fp32 launcher.

Tuned rows are accepted atomically: a stale kid discards its paired split-K as
well. Each architecture's heuristic kids are force-included in subset builds,
so a CSV miss cannot select an uncompiled default.

## Torch workspace ownership

Two-stage split-K kernels no longer allocate or cache scratch memory in C++.
The execution path is:

```text
resolve actual kid and split-K
  -> build typed WorkspacePlan
  -> torch.empty for this call
  -> raw C++ launch with Tensor
```

`WorkspacePlan` is dynamic and records shape, dtype, required element count,
and alignment. Architecture- and kid-specific policy lives in
`_workspace_a16w16.py`; the shared `_workspace.py` module only represents,
allocates, and validates a plan.

Current planned layouts are:

| Architecture | Logical workspace shape | Dtype |
|---|---|---|
| gfx950 | `[allocation_split_k, batch, padded_M, padded_N]` | fp32 |
| gfx942 | `[allocation_split_k, batch, padded_M, padded_N]` | instance-declared bf16 or fp32 |
| gfx1250 | `[allocation_split_k, padded_M, padded_N]` | fp32 |

Padding uses the actual launcher's `B_M` and `B_N`. The planner rejects a
split-K larger than `ceil(K / B_K)` before calling `torch.empty`.

There is deliberately no Python global Tensor cache. PyTorch may cache the
underlying storage in its device allocator, but OPUS retains no Tensor object
or raw pointer after a call.

## Explicit workspace reuse

The low-level public wrapper accepts an optional keyword-only workspace:

```python
from aiter.ops.opus import opus_gemm_a16w16_tune

opus_gemm_a16w16_tune(
    XQ, WQ, Y,
    bias=None,
    kernelId=200,
    splitK=2,
    workspace=my_workspace,
)
```

`XQ`, `WQ`, and `Y` are 3D tensors for this entry. If the selected kid needs
scratch and `workspace` is omitted, the wrapper allocates one fresh Tensor. If
it is supplied, the shared validator returns the same object unchanged and no
allocation occurs. A non-workspace kid requires `workspace=None`.

Caller-owned workspace shape is flexible because kernels consume a flat
address range. It must satisfy all of the following:

- same device as `XQ`;
- exact planned dtype;
- contiguous storage;
- at least `required_numel` elements;
- address aligned to the plan requirement (currently 16 bytes).

The generated C++ launcher repeats device, dtype, contiguity, alignment,
overflow, and final post-clamp capacity checks. Exact capacity succeeds; one
element short fails before a device kernel is launched.

The underlying pybind ABI is ordered as
`(XQ, WQ, Y, bias, workspace, kernelId, splitK)`. It is intentionally private;
use the Python wrapper unless testing the C++ guard itself.

## Graph capture, streams, and lifecycle

No OPUS workspace initialization or shape prewarm is required. `torch.empty`
is capture-aware: an allocation made during `torch.cuda.graph` capture belongs
to the graph-private pool and the captured Tensor address is reused on replay.

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    Y = gemm_a16w16_opus(A, B, kernelId=200, splitK=2)

graph.replay()
```

Concurrent streams receive separate call-scoped workspace Tensor objects.
There is no stream registry or process-global scratch buffer, so two-batch
overlap does not need an OPUS-specific lock, handle registration, or warmup.

`opus_gemm_workspace_init()` remains available only as a deprecated Python
no-op. Remove it from new code.

## Bias behavior

Bias follows `torch.nn.functional.linear`'s output-feature convention:
`[N]` broadcasts across the batch and `[batch, N]` supplies one vector per
batch. It must be contiguous.

Architecture rules remain launcher-specific:

- gfx950 bias-aware launchers require bias dtype to match `Y`; unsupported
  kids reject bias.
- gfx942 automatic requests that need a split-K bias launcher retain the
  framework fallback. An explicit such kid is a strict error under the
  current tune ABI.
- gfx1250 reduce accepts fp32 bias even when `Y` is bf16, as well as bias that
  matches `Y`. Bias is accumulated in fp32 before the final cast.

The shared workspace validator does not inspect or alter bias behavior.

## Architecture notes

### gfx950

Workspace kids include the two-stage flatmm split-K family, with fp32
partials. Non-workspace split-barrier, flatmm, mono-tile, and persistent kids
remain in the separate five-argument dispatch table. A common explicit
regression fixture is kid 200 with `(M, N, K, splitK) = (64, 64, 512, 2)`;
its exact workspace capacity is 8192 fp32 elements.

### gfx942

The selector separately mirrors the generated split-K auto-pick and clamp,
including even-loop constraints. bf16-workspace and fp32-workspace launchers
are distinct actual instances.

The device path still uniformizes both halves of the 64-bit direct workspace
pointer with `readfirstlane` in main and reduce kernels. The old handle load is
gone, but the wave-uniform address semantics are retained. See the C++ README
for the cross-compiled ISA/register comparison.

### gfx1250

Cluster/TDM split-K always uses fp32 workspace. The planner omits a batch
extent and rejects `batch > 1`; the generated launcher repeats that check as a
raw C++ defense. `Y=bf16` with `bias=fp32` is a supported reduce combination.

## Tuning

The existing tuner still calls the public wrapper, which now allocates the
correct workspace automatically. Callers in
`csrc/opus_gemm/opus_gemm_tune.py`,
`csrc/gemm_a16w16/gemm_a16w16_tune.py`, and `aiter/ops/deepgemm.py` must not
add their own scratch allocator.

Generate or rebuild with an explicit architecture set when needed:

```bash
GPU_ARCHS='gfx942;gfx950;gfx1250' \
python csrc/opus_gemm/gen_instances.py -w /tmp/opus-generated
```

Generated a16w16 dispatch has separate non-workspace and workspace function
pointer tables. Existing a8w8 output and the separately generated a8w4 MoE
modules are outside this split and retain their prior API and launcher ABI.

## Tests

Run the focused suite:

```bash
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py
```

The suite contains CPU selector/planner tests plus architecture-gated GPU
tests for numerical output, raw C++ validation, graph replay, concurrent
streams, bias, and batch rules. A test skipped for a missing architecture is
not a hardware pass for that architecture.

For a CLI performance/sweep run:

```bash
python op_tests/test_opus_a16w16_gemm.py -m 64 -n 64 -k 512 -b 1
python op_tests/test_opus_a16w16_gemm.py --csv_file /path/to/shapes.csv
```

## Troubleshooting

### `workspace kernel id ... requires a workspace tensor`

The private raw binding was called directly. Use
`opus_gemm_a16w16_tune`, or pass a Tensor that satisfies the actual kid's
plan when intentionally testing the raw ABI.

### `workspace dtype must be ...`

Do not infer dtype from output dtype or requested kid. Build the plan from the
resolved actual instance. gfx942 can require bf16 or fp32 scratch; gfx950 and
gfx1250 require fp32.

### `workspace capacity ... elements ... required`

Capacity is computed after the launcher's split-K clamp using checked padded
extents. Allocate at least the plan size; byte-equivalent storage with the
wrong dtype is not accepted.

### `kid 10216 requires exact-N`

Use one of the shared exact-N values or select a compatible fp32-workspace
launcher. Unlike 10210 and 10213, kid 10216 has no non-exact redirect.

### `gfx1250 ... requires batch=1`

Split the batch into individual calls. Do not bypass the Python error: the C++
launcher enforces the same limitation.

### Layout or broadcast-view errors

Materialize `A`, `B`, or `out` with the documented strides. In particular,
replace a batch-broadcast `B.expand(...)` view with `.contiguous()`.

## File map

| Path | Purpose |
|---|---|
| `_selector_a16w16.py` | Actual-kid-first selection and redirects |
| `heuristics/a16w16_gfx*.py` | Python parity ports of per-architecture C++ heuristics |
| `_workspace.py` | Family-neutral `WorkspacePlan`, allocation, and validation |
| `_workspace_a16w16.py` | a16w16 instance-to-plan adapter |
| `gemm_op_a16w16.py` | Public APIs and centralized launch path |
| `gemm_op_a8w8.py` | Separate gfx942 a8w8 tune API |
| `moe_stage1_a8w4.py`, `moe_stage2_a8w4.py` | Separately built a8w4 MoE APIs |
| `../../../csrc/opus_gemm/` | C++ entry, codegen, traits, pipelines, and reduce kernels |

`_workspace.py` is intentionally family-neutral: it accepts a completed plan
and never selects an architecture, kid, dtype policy, redirect, or launcher
ABI.  `_workspace_a16w16.py` is the family adapter that owns those a16w16
decisions.  A new adapter is warranted only after another family acquires an
external two-stage workspace kernel; a8w8 and a8w4 MoE do not have one, and no
a4w4 implementation exists in this tree.
