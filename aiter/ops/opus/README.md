# OPUS GEMM and BMM Python interfaces

OPUS exposes strict exact-kid functions for logical 2D GEMM and batch-first 3D
BMM, plus the parameterized `opus_gemm_bpreshuffle` interface and the retained
shape-driven `gemm_a16w16_opus` compatibility entry.
The exact functions never select a kernel from the shape; the compatibility
entry resolves an A16W16 kid before entering the same exact path.

## Public API

```python
import torch

from aiter.ops.opus import gemm_a16w16_opus, opus_bmm, opus_gemm

opus_gemm(  # XQ [M,K], WQ [N,K], Y [M,N]
    XQ,
    WQ,
    Y,
    *,
    kid,
    layout="plain",
    x_scale=None,
    w_scale=None,
    bias=None,
    split_k=0,
    workspace=None,
)

opus_bmm(  # XQ [B,M,K], WQ [B,N,K], Y [B,M,N]
    XQ,
    WQ,
    Y,
    *,
    kid,
    layout="plain",
    x_scale=None,
    w_scale=None,
    bias=None,
    split_k=0,
    workspace=None,
)

# Retained A16W16 shape-driven API. An explicit kernelId wins; otherwise this
# performs OPUS-only tuned lookup followed by the per-architecture heuristic.
result = gemm_a16w16_opus(A, B, bias=None, dtype=torch.bfloat16)
```

For `opus_gemm` and `opus_bmm`, `kid` is mandatory. `Y` is caller-owned and is
returned unchanged after the launch. The selected exact function determines
the logical rank, while the resolved family must support that operation;
dtype does not determine it.
Among A8 families, no-scale, blockscale, and both FP32-scale/native-E8M0
bpreshuffle families are GEMM-only, while `a8w8_mxscale_bmm` is BMM-only.
The disjoint architecture id bands and the
merged `kernels_list` form the canonical registry. Both entries call
`kernels_list.get(kid)`; they do not introduce or renumber ids. The returned
instance tag plus the dtype/layout arguments determine the private family
adapter, rather than a second selector or a numeric-range guess.

## Dispatch model

```text
caller-resolved final kid
  -> opus_gemm (strict 2D) or opus_bmm (strict batch-first 3D)
  -> existing kernels_list.get(kid)
  -> instance arch/tag metadata
  -> family-local dtype/layout/scale checks
  -> A16W16 or A8W8 family adapter
  -> shared immutable A16 launch plan or A8 family planner
  -> family executor
  -> unchanged exact-kid C++ family table
```

The public operation split does not duplicate kernels, workspace allocation,
or raw bindings. A logical GEMM becomes a batch-one view at the family
boundary. A logical BMM keeps its batch-first public layout; the MXScale
adapter alone converts activation/output tensors to the raw kernel's existing
M-major views with `transpose(0, 1)`, which does not copy storage. The physical
3D raw ABI used by a non-MX A8 GEMM is not exposed as public BMM.

Kernel-ID selection inside either exact public path has no tuned-CSV lookup,
architecture heuristic, redirect, or framework fallback: the caller supplies
the exact `kid`. Runtime split-K for nine gfx950 native-E8M0 bpreshuffle kids
can use a separate optional shape/CU heuristic to choose the partition count
within that selected kid. The two shape-driven A16 callers remain intentionally
different:

```text
aiter.gemm_a16w16
  -> global multi-backend tuned row -> selected backend
  -> no valid row -> skinny (eligible gfx90a/gfx942/gfx950)
                  -> gfx1250 Triton
                  -> otherwise PyTorch

gemm_a16w16_opus
  -> explicit kernelId -> legacy requested-to-actual resolution
  -> otherwise OPUS-only tuned row -> the same compatibility resolution
  -> missing/invalid OPUS row -> per-arch OPUS heuristic
                               -> validate -> local exact A16 GEMM/BMM launcher
```

The shared OPUS candidate helpers are isolated in `policy.py`.
`tuned_gemm.py` validates tuned candidates and keeps its normal framework
fallback. The OPUS-only compatibility entry warns once and uses its heuristic
for a stale tuned row, but rejects an invalid explicit id. It applies legacy
gfx942 requested-to-actual resolution before calling the local exact launcher;
the strict `opus_gemm`/`opus_bmm` APIs never redirect.

There is currently no high-level A16W16 BF16 BMM wrapper. In particular,
`aiter/ops/batched_gemm_op_bf16.py` contains the existing CK entry points but
does not define `batched_gemm_bf16_OPUS` or a tuned CK/OPUS dispatcher. A16W16
BMM therefore starts at exact-kid `opus_bmm`: its caller owns `Y`, resolves the
final `kid`/`split_k`, and may provide a Torch workspace. The public router
calls `_launch_a16w16_bmm`, which preserves the batch dimension and forwards
to the same `_execute_a16w16` planner/executor used by A16W16 GEMM.

## Current families

| Registry family | Current route | Public operation and dtype rules |
|---|---|---|
| `a16w16` | gfx942, gfx950, gfx1250 | GEMM or BMM; BF16 `XQ/WQ`, normally BF16 or FP32 `Y`, plain WQ, optional bias/split-K/Torch workspace; gfx942 BF16-workspace exact kids require BF16 `Y` |
| `a8w8` | gfx950 kid 2 | GEMM only; FP8 `XQ/WQ`, FP32 `Y`, plain WQ, no scales |
| `a8w8_blockscale` | gfx950 kid 1 | GEMM only; FP8 `XQ/WQ`, FP32 `Y`, plain WQ, two FP32 scales |
| `a8w8_blockscale_bpreshuffle` | gfx942 kid 11000 | GEMM only; FP8 `XQ/WQ`, BF16 `Y`, pre-shuffled WQ, two FP32 scales |
| `a8w8_mxscale_gemm_bpreshuffle` | gfx950, five pipelines / 89 compile configurations / 105 compatibility IDs | GEMM only; FP8 E4M3FN inputs, BF16 `Y`, pre-shuffled WQ, native E8M0 scales; nine IDs accept runtime global split-K and optional FP32 workspace |
| `a8w8_mxscale_bmm` | gfx950 global kids 8000--8653 (45 registered ids) | BMM only; batch-first FP8 inputs, E8M0 scales, BF16 or FP32 output, optional split-K Torch workspace |

Empty family tables on another architecture are valid capability states. A
kid registered for another architecture is rejected before family launch.

## Examples

### A16W16 GEMM and BMM

```python
import torch
from aiter.ops.opus import opus_bmm, opus_gemm

XQ = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
WQ = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
Y = torch.empty((64, 64), device="cuda", dtype=torch.bfloat16)

# The caller/tuner has already chosen gfx950 kid 200 and split_k 2.
opus_gemm(XQ, WQ, Y, kid=200, split_k=2)

XQ_b = torch.randn((8, 64, 512), device="cuda", dtype=torch.bfloat16)
WQ_b = torch.randn((8, 64, 512), device="cuda", dtype=torch.bfloat16)
Y_b = torch.empty((8, 64, 64), device="cuda", dtype=torch.bfloat16)
opus_bmm(XQ_b, WQ_b, Y_b, kid=200, split_k=2)
```

`opus_gemm` requires 2D tensors; `opus_bmm` requires batch-first 3D tensors.
Inputs are K-contiguous and `Y` is N-contiguous. gfx1250 two-stage workspace
kernels require BMM batch one; pre-built CO kernels support batched inputs.
Exact instances can impose additional tile, output
dtype, bias, or K-loop constraints. The BMM example is a direct exact-API call;
there is no current `batched_gemm_bf16_OPUS` high-level wrapper.

A16 bias follows the `F.linear` output-feature convention: `[N]` broadcasts
across batch and `[batch,N]` supplies a separate bias for each batch.

### gfx950 A8W8 without scales

```python
from aiter.ops.opus import opus_gemm

Y = torch.empty((M, N), device=XQ.device, dtype=torch.float32)
opus_gemm(XQ, WQ, Y, kid=2)
```

The general `aiter.gemm_a8w8` API remains the scaled CK/Triton operation and
requires both `x_scale` and `w_scale`; omitting scales does not select OPUS.

### gfx950 A8W8 blockscale

```python
from aiter.ops.opus import opus_gemm

opus_gemm(
    XQ,
    WQ,
    Y,
    kid=1,
    x_scale=x_scale,
    w_scale=w_scale,
)
```

The group contract is 1x128x128. GEMM scales are contiguous FP32
`[M,K/128]` and `[N/128,K/128]` tensors. This family does not accept
`opus_bmm`. The general `aiter.gemm_a8w8_blockscale` dispatcher remains a
BF16/FP16 CK/CKTile/ASM/Triton API; FP32 output is available only through the
explicit OPUS exact-kid call above.

### gfx950 native-E8M0 MXFP8 bpreshuffle GEMM

The public wrapper selects a configuration by `(gfx, cu_num, M, N, K)`. It first
reads a valid OPUS row from the tuned CSV and then tries legal default
configurations if no matching row exists. A pipeline name limits this choice.
An explicit configuration or named compile parameters bypass the table and
construct traits for that tuple, including legal combinations with no numeric
ID. Unsupported geometry and inactive parameters are rejected.

There are five compute pipelines, 89 default configurations, and 105 historical
compatibility IDs. Legal new tuples instantiate these same pipelines. Each
configuration has an independent JIT module; its cache identity includes the
complete parameters, source contents, compiler, resource headers, and build
settings. First use prepares that module, and subsequent launches reuse it.

A scales have logical shape `[M,K/128]` with dense column-major strides `(1,M)`;
B scales are contiguous `[N/128,K/128]`. Both are one-byte E8M0. FP8 inputs
and BF16 output are contiguous and on the same device. N and K must be divisible
by 128. Configuration-specific
shape, alignment, and byte limits apply.

```python
from aiter.ops.opus import opus_gemm_bpreshuffle
from aiter.ops.shuffle import shuffle_weight
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config

# Existing tensors, with native E8M0 scales in the layouts described above.
WQ_shuffled = shuffle_weight(WQ, layout=(16, 16))
Y = torch.empty((M, N), device=XQ.device, dtype=torch.bfloat16)

# Tuned CSV first, then a legal shape-based default.
opus_gemm_bpreshuffle(XQ, WQ_shuffled, Y, x_scale, w_scale)

# Restrict automatic selection to one pipeline.
opus_gemm_bpreshuffle(
    XQ, WQ_shuffled, Y, x_scale, w_scale,
    pipeline="register",
)

# New register queue depth: no ID registration or extra kernel source file.
# This tuple retains the register seed's M <= 512 and K <= 16384 limits.
config = construct_config(
    "register", tile_m=16, tile_n=32, prefetch=5, runtime_split_k=True,
)
opus_gemm_bpreshuffle(
    XQ, WQ_shuffled, Y, x_scale, w_scale,
    config=config, split_k=3,  # K >= 384 for this literal split
)
```

`config` also accepts its complete parameter mapping or `config.to_json()`;
these require a pipeline name. Named partial axes inherit a compatible default;
save the complete canonical payload to replay the same tuple. The default table
is `aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv`.
Use `tuned_file=...` or `OPUS_BPRESHUFFLE_TUNED_CONFIG` to select another CSV.
An invalid explicit file raises an error with its path and line; a missing or
stale default row falls back. Rows from other backends do not select an OPUS
configuration. The default heuristic has no measured speed guarantee.

The lower-level `opus_gemm(..., kid=..., layout="bpreshuffle")` remains available
for existing callers. Kernel source, parameter mapping, tuning CSV metadata and
verification are documented in the
[configuration flow report](../../../reports/opus_configflow_20261009/README.md).

Runtime split-K is supported by register IDs 92310/92311/92320/92321/92330/92340
and fine LDS IDs 92410/92420/92430, and legal parameter configurations that enable
`runtime_split_k`. `split_k=None` uses the tuned count or a grid/CU heuristic.
Positive `split_k` values are literal counts
in `1..min(16,K/128)`. Zero preserves the historical default: register four,
fine one; the register default may include empty partitions for short K.
Minus one chooses an optional grid/CU heuristic using M/N/K and the input
device CU count. It has not been measured as the fastest split. Tile geometry
and local WaveK remain static. Global runtime split counts share a producer
specialization; selecting another count does not create a configuration ID.

Split one writes BF16 directly and requires `workspace=None`. Larger splits
use `split_k * M * N` FP32 workspace elements and a shared reducer. Python
allocates call-scoped storage if omitted, or validates caller-owned contiguous,
same-device, 16-byte-aligned FP32 storage that is large enough and disjoint
from input/output. Fine IDs 92411/92421/92431 retain their fixed-split-two
compatibility paths and require public `split_k=0`, as do other fixed-split
bpreshuffle IDs.

All 105 compatibility configurations compiled offline through the parameter
codegen path. Five new tuples, one per pipeline, also compiled and linked as
independent modules. CPU checks cover selection, two configurations in one
process, and tuning CSV replay. GPU numerical and performance tests remain
stopped. These examples were not executed during this update.

### gfx950 MXFP8 BMM

```python
Y = torch.empty((G, M, N), device=XQ.device, dtype=torch.bfloat16)
opus_bmm(
    XQ,                    # [G,M,K], batch-first and K-contiguous
    WQ,                    # [G,N,K], batch-first and K-contiguous
    Y,
    kid=8311,              # exact global id; family-local id 311
    layout="mxscale_bmm",
    x_scale=x_scale,       # [G,M,K/128], one-byte E8M0
    w_scale=w_scale,       # [G,N/128,K/128], one-byte E8M0
    split_k=1,
)
```

The 45 MXFP8 BMM kernels use global ids `8000 + family_local_kid`; the
family-local ids remain recognizable while sharing the canonical
`kernels_list` without colliding with existing GEMM ids. The public layout
name is strictly `mxscale_bmm`. Internally, the family adapter passes zero-copy
`[M,G,*]` transpose views to the unchanged raw kernel ABI.
The high-level tuned caller remains
`aiter.batched_gemm_a8w8_mxscale`. Its cold-path tuned-row, padded-M,
local-to-global-id and heuristic selection live in `policy.py` beside the A16
caller policy.
The caller caches the final id/split pair per shape: split-one enters the
checked raw launcher directly, while workspace launches call `opus_bmm`.

### gfx942 blockscale bpreshuffle

```python
from aiter.ops.shuffle import shuffle_weight

WQ_shuffled = shuffle_weight(WQ, layout=(16, 16))
opus_gemm(
    XQ,
    WQ_shuffled,
    Y,
    kid=11000,
    layout="bpreshuffle",
    x_scale=x_scale,
    w_scale=w_scale,
)
```

`layout="bpreshuffle"` is a declaration of WQ content, not something Tensor
shape or strides can prove. Kid 11000 requires batch one, exact 128-wide N/K
tiles, BF16 output, and its registered scale storage contracts. The high-level
`gemm_a8w8_blockscale_bpreshuffle` dispatcher enters this OPUS route only when
the tuned row has `libtype=opus`; CK, CKTile, ASM and Triton rows remain on
their respective backends. gfx950 currently registers zero OPUS kids for this
family, so a gfx950 OPUS validation must report it unavailable rather than run
a non-OPUS fallback as coverage.

## Tuning compatibility

The exact public APIs execute a caller-selected kid. A16 production tuning
continues through `csrc/gemm_a16w16/gemm_a16w16_tune.py`; plain A8W8 and
MXFP8 BMM use `csrc/opus_gemm/opus_gemm_a8w8_tune.py` and
`csrc/opus_gemm/opus_bmm_mxscale_tune.py`, respectively.

Native-E8M0 gfx950 bpreshuffle GEMM uses
`csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py`. Its default set contains
89 default configurations. `--opus_configs path.json` adds legal parameter
tuples without registering IDs; `--opus_pipelines register,lds` filters the
pipelines. For runtime configurations, tuning enumerates every legal positive
split. Output saves canonical `pipeline/config`, `splitK`, and a compatibility
`kernelId` (`-1` for new tuples). Replay also accepts parameter-only CSVs with no
ID column. Configurations are prepared before timing. The sixteen
historical compatibility/internal IDs remain callable when explicitly selected.

The CK-owned blockscale tuner remains unchanged. Its legacy
`opus_gemm_a8w8_blockscale_bpreshuffle_tune(...)` import is retained in
`gemm_op_a8w8.py` and calls the bpreshuffle family launcher directly.

### Plain A8W8 GEMM

`csrc/opus_gemm/opus_gemm_a8w8_tune.py` tunes the gfx950 no-scale and ordinary
blockscale GEMMs. It accepts the existing GEMM tuner options; CSV `scaleAB`
selects the scale mode, so there is no `--family` option.

```bash
# Run from the repository root.
export PYTHONPATH="$PWD"
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

cat > /tmp/opus_a8w8_shapes.csv <<'CSV'
M,N,K,dtype,outdtype,bias,scaleAB,bpreshuffle
64,4096,4096,fp8,fp32,False,False,False
128,4096,4096,fp8,fp32,False,False,False
64,4096,4096,fp8,fp32,False,True,False
128,4096,4096,fp8,fp32,False,True,False
CSV

python3 csrc/opus_gemm/opus_gemm_a8w8_tune.py \
  --input_file /tmp/opus_a8w8_shapes.csv \
  --tuned_file /tmp/opus_a8w8_tuned.csv \
  --libtype opus --mp 1

python3 csrc/opus_gemm/opus_gemm_a8w8_tune.py \
  --run_config /tmp/opus_a8w8_tuned.csv --libtype opus --mp 1
```

`-i/--untune_file` and `-o/--tune_file` are equivalent aliases. Missing
`dtype`, `outdtype`, and `scaleAB` columns default to FP8, FP32, and `False`.
`scaleAB=True` uses FP32 scales with the registered 1x128x128 group contract.
Both modes require contiguous inputs/output, no bias or preshuffle, and
`splitK=0`. Candidates come from the canonical registry and are checked against
an independent FP32 dequantize-and-matmul reference. The default `--errRatio 0`
rejects any element outside `rtol=atol=1e-2`.

The output key includes `gfx,cu_num,M,N,K,dtype,outdtype,scaleAB`, so both
scale modes can coexist for the same shape. `--all` retunes the input rows;
`--profile_file` records all candidates. `--run_config` executes the saved
`kernelId` through `opus_gemm` with preallocated FP32 output; without a path it
reads `--tuned_file`. Callers own this CSV lookup. For M padding, allocate and
zero-pad XQ, pad `x_scale` rows with finite scales (for example 1), call the
saved kid with padded output, then slice back to the original M.

## A16 Torch workspace

Workspace ownership is call-scoped and remains in Torch:

```text
validate exact kid and split_k
  -> derive immutable workspace plan from the exact instance
  -> reuse caller workspace or torch.empty for this call
  -> _launch_a16w16_backend
```

There is no process-global Tensor, pointer registry, HIP allocator, or prewarm
API. The bounded public-contract and A16 launch-plan caches store only registry
metadata, integers, dtypes, option-presence flags and shapes; they never retain
Tensor objects, data pointers, devices, streams or workspaces.

Let `padded_M=ceil_div(M,B_M)*B_M` and
`padded_N=ceil_div(N,B_N)*B_N`:

| Architecture/family | Workspace shape | Instance storage |
|---|---|---|
| gfx950 two-stage | `[workspace_capacity_split_k,batch,padded_M,padded_N]` | FP32 |
| gfx942 two-stage | `[workspace_capacity_split_k,batch,padded_M,padded_N]` | exact BF16/FP32 dtype |
| gfx1250 two-stage | `[workspace_capacity_split_k,padded_M,padded_N]` | FP32 |
| gfx1250 pre-built CO direct | none | none |
| gfx1250 fused | not publicly registered | factory/emitter/source retained for repair |

For gfx942, `abi_split_k` records the converged value passed to the launcher.
`workspace_capacity_split_k` uses the same value, so automatic allocation
reserves one workspace slice per launched split.

An explicit workspace must be on the XQ device, contiguous, 16-byte aligned,
of the exact instance dtype, and large enough for the final split. Larger
caller-provided workspaces are also accepted. A direct kid requires
`workspace=None`.

gfx1250 two-stage kids require `M <= 65535`, including `split_k=0/1`, because
their separate reducer places one logical row in each `grid.y` block. The
tuner excludes larger M, policy rejects stale split-K rows, and exact launch
checks the limit before either kernel runs. Larger M can use a compatible
tuned CO kid; the gfx1250 heuristic only selects two-stage kids and requests
tuning when this limit is exceeded.

gfx1250 clusterlaunch exact kids round the physical launch grid up to complete
clusters; tile-less workgroups exit inside the pipeline. This does not change
the logical workspace shape above. The experimental fused family is disabled,
so no fused kid in `[27000,30000)` can be resolved through the public registry.
The `[21000,27000)` band belongs to pre-built CO kids, which are direct and
therefore never request a workspace.

gfx942 BF16-workspace kids `10210`, `10213`, and `10216` are exact ids. Their
registered exact-N contract is `{64,128,256,384,512,1024,2048}`. A different N
or an FP32 `Y` is rejected by the strict `opus_gemm`/`opus_bmm` exact APIs; they
never redirect. To preserve the former shape-driven API, `gemm_a16w16_opus`
maps `10210` to `10200` and `10213` to `10203` for a non-exact N. Kid `10216`
has no FP32-workspace sibling and remains rejected.

## MXFP8 BMM Torch workspace

MXFP8 BMM follows the same ownership rule: Python either uses the caller's
contiguous FP32 Tensor or creates a call-scoped `torch.empty`; C++ receives a
direct pointer and never retains it. Two-stage split-K uses
`split_k * G * padded_M * padded_N` FP32 elements. The fused family stores its
partials and aligned tile counters in one FP32 Tensor. `split_k == 1` and
families that do not consume workspace reject a supplied Tensor.

`launch_plan.py` owns the shared immutable workspace specification plus the
family-specific A16W16 and A8W8 plans. Its `A8W8MxscaleBMMPlan` records the
resolved exact kid, the split-K value passed to the ABI and an optional
`WorkspaceSpec`. `gemm_op_a8w8.py` only adapts logical layouts, materializes
that workspace and invokes `_launch_a8w8_backend`.

For gfx950 kid 8326, only the `split_k > 1`, `D_OUT=void` workspace
specialization sets `PRELOAD_SF_LDS=false` to avoid the ROCm 7.2.4 compiler
failure. Its direct BF16/FP32 `split_k == 1` specializations keep
`PRELOAD_SF_LDS=true`.

## A8 pybind backend

The original A8 GEMM adapters and the MXFP8 BMM executor use the shared
low-level facade. The native-E8M0 bpreshuffle adapter adds a checked runtime
split entry for its nine runtime kids:

```text
validated family + resolved kid + physical Tensor views
  -> _launch_a8w8_backend
       -> no-scale pybind raw launcher
       -> plain blockscale pybind raw launcher
       -> blockscale-bpreshuffle pybind raw launcher
       -> MXFP8 BMM pybind raw launcher
native-E8M0 bpreshuffle runtime plan
  -> checked bpreshuffle runtime split pybind raw launcher
```

## Graphs and streams

Automatic `torch.empty` during graph capture uses the graph-private pool.

A CO image is opened and registered on the first call to its launcher. That
first load must happen before graph capture; warm-up followed by capture/replay
is supported, while first-ever loading inside capture is not. `import aiter`
sets `OPUS_GEN_CO_DIR` to the packaged `csrc/opus_gemm/gen_co` directory, and an
explicit environment value overrides it for testing locally rebuilt images.

## Build-time subset compile

Tuned CSVs, the last successful compiled-kids sidecar, and additional tuner
candidates passed through `--extra_kids` are build inputs only. Their valid
non-BMM OPUS ids are unioned with:

- `DEFAULT_COMPILED_KIDS_BY_ARCH`, the exact-id compile floor containing every
  A16 caller-side heuristic result;
- mandatory A8 ids (`gfx950: {1,2}`, `gfx942: {11000}`).

This controls which launchers enter a subset `.so`.  The high-level A16 caller
may read a tuned row at runtime, but the public/C++ path receives only its
resolved id. Calling a known non-BMM registry kid that was omitted from a
subset build produces an uncompiled-id error. A gfx950 build emits all 45
MXFP8 BMM routes as one deduplicated family so every registered BMM id remains
exact-routable. A gfx1250 build keeps all 219 available CO host launchers in its
default compile floor; their device code remains in the packaged `.co` files.
The sidecar records all emitted ids, including the deduplicated BMM family.
An explicit `--extra_kids` request that is unknown, outside the target
architectures, or excluded by `--kernel_tag` fails codegen before the sidecar
is updated.

The canonical sidecar is `{bd_dir}/compiled_kids_opus.json`, outside the
per-module build directory so it survives `clear_build`. Tuners synchronously
build candidates before spawning workers. They pass requests through
`--extra_kids` without expanding the canonical sidecar in advance. JIT uses
`blob.staging` for generated working files, installs the binary, then publishes
the generated sidecar and a receipt binding its contents to that binary.
Runtime exact dispatch does not read this sidecar.

A tuner skips rebuilding only when the sidecar and its receipt match the
required kids and installed binary. Missing or stale metadata triggers a
rebuild. An explicit `AITER_REBUILD` request runs once in the parent even on
a cache hit; successful preparation sets `AITER_REBUILD=0` for workers. A
failed compile restores the original environment and preserves the previous
successful metadata. See [transactional JIT cache](../../../docs/jit_cache.md)
for recovery and storage requirements.

## Migration

For new exact-id integrations, allocate `Y`, resolve the final id in the
caller, use `opus_gemm` for logical 2D calls, and use `opus_bmm` for
batch-first 3D calls. The retained `gemm_a16w16_opus` entry preserves the
former A16W16 shape-driven behavior: explicit id, then OPUS-only tuned lookup,
then the migrated per-architecture heuristic. Do not infer the operation from
dtype or expose the physical 3D raw ABI of a GEMM-only A8 family as public
BMM. The A8 family module exports only its legacy tuner compatibility name.

## Validation

Run the retained OPUS numerical tests on matching target GPUs rather than
treating architecture skips as coverage:

```bash
pytest -q op_tests/test_opus_a16w16_gemm.py
PYTHONPATH=. python3 op_tests/test_opus_a8w8_bmm.py \
  -g 2 -s 16,1024,4096 -d bf16
```

In particular, gfx942 and gfx1250 validation must run on matching hardware; a
skip on another architecture is not a pass for that target.

## Source map

| Path | Role |
|---|---|
| `__init__.py` | thin public `opus_gemm`/`opus_bmm` delegates and lazy `gemm_a16w16_opus` compatibility entry |
| `dispatch.py` | public contract validation and strict exact-kid family routing |
| `_arch.py` | per-explicit-device architecture/CU scalar cache |
| `policy.py` | A16 tuned/heuristic candidate selection plus MXFP8 tuned CSV discovery, padded-M lookup, local-to-global kid normalization and heuristic fallback |
| `launch_plan.py` | shared `WorkspaceSpec`, A16 exact-kid/split-K planning, and A8 family contract/MXFP8 BMM planning |
| `gemm_op_a8w8.py` | A8 GEMM adapters including native-E8M0 runtime bpreshuffle plan/workspace materialization, the legacy tuner compatibility entry, MXFP8 BMM execution, and shared raw-binding facade |
| `csrc/opus_gemm/opus_gemm_a8w8_tune.py` | plain A8W8 no-scale/blockscale tuner and saved-kid CSV replay |
| `moe_stage1_a8w4.py` | A8W4 MoE stage-1 runtime binding and launcher |
| `moe_stage2_a8w4.py` | A8W4 MoE stage-2 runtime bindings and launchers |
| `../gemm_op_a8w8.py` | general scaled CK/CKTile/ASM/Triton A8 dispatchers plus the tuned-row OPUS bpreshuffle route |
| `../batched_gemm_op_bf16.py` | existing high-level CK BF16 BMM path; it is not an OPUS A16W16 BMM wrapper |
| `../batched_gemm_op_a8w8.py` | MXFP8 high-level caller, scalar launch cache, output allocation and split-one/workspace execution choice |
| `../../../csrc/opus_gemm/` | canonical registry, C++ family launchers, codegen, traits and pipelines |
