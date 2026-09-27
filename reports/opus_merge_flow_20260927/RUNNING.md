# Single-flow consolidation in progress

## 2026-09-27 recovery and source-style update

The old queued pilot and continuation processes no longer exist. The third pilot
started at 08:32 UTC but all eight summary CSVs have zero data rows; its `running`
and `launching` JSON states are stale. A fresh measurement batch is required.

The main_v1 / small_v4 / tiny_v3 implementations now also have canonical
pipeline/Traits headers in `csrc/opus_gemm/include/gfx950`, named with the
`main`, `small`, and `narrow` family suffixes. All five instantiated kernels
compile with byte-identical device instructions and normalized descriptors.
The 9000/9020 dependencies, registry, codegen, and original candidate files remain
unchanged. See [the source-style report](../opus_native_style_20260927/README.md).
Full295 selection, pruning, and installation are still pending. The old tools
below do not automatically consume the new header/symbol mapping.

The remainder of this file is the earlier handoff snapshot.

Scope: keep registered 9000/9020 and their shared helpers unchanged. Replace all
other fourteen retained candidates, including 9010/9011/9012, with one computation
flow per geometry. Final target is at most five new geometry entries in three
shared families: 192x256, {128,160}x128, and 64x{128,64}.

Completed measurement:
- Pilot 1: 62 existing target shapes, eight GPUs, three rounds, zero new numerical
  failures; reduced pool geometric mean time +3.962% versus sixteen old controls.
- Pilot 2: same representative target set, fresh same-batch old/new measurements,
  zero new numerical failures. With main21000/small21110+21111/tiny21120+21121,
  reduced pool geometric mean time +1.043% versus old sixteen. The main21001
  policy is +1.122%, so expanded workgroup mapping is not automatically better.
- Remaining largest gap is low-CTA-count, narrow-N long-K cases formerly won by
  9010. Long-K support in the new small kernels improves several wide-N targets.

Prepared next candidates:
- Small v3 (21210/21211): one U1 loop, partial VMEM wait, no new K-specific flow.
- Tiny v3 (21220/21221): one shared flow, v1 instructions except fixing N128 grid.
- Small v4 compiled and frozen (21310/21311): one shared ring loop, geometry-constant S3 for
  128x128 and S2 for 160x128, larger scale panel.

2026-09-27 08:02 UTC: two launch attempts stopped before GPU allocation because
all eight physical GPUs have outside activity. Latest sample shows ~18% VRAM
allocated on every card. No third-pilot timings exist yet. Production and the old
sixteen-candidate baseline remain frozen until the full 295-shape comparison ends.

Finalization tools in `retained_tools/` are being reviewed. The final measurement
must precede production pruning and old JIT replacement. Delete unselected source
and binaries from this campaign after successful install; preserve measured CSV,
JSON, and textual assembly audit records. Do not delete unrelated historical work.


Automatic continuation is now running:
1. `pilot62_v34_queue.json`: idle-gated third pilot, all v2/v3/v4 small versions,
   three main grid policies, tiny v3, old16 controls, fresh external-finalist timing.
2. `continuation.json` / `continue_measurements.py`: after the pilot passes, run
   `analyze_flow.py` and `analyze_pools.py`; then wait for idle GPUs and run the full
   295-shape, three-round comparison using this exact frozen candidate population.
3. After full completion, the continuation writes audited pool comparisons and
   stops at `measurements_complete_pending_final_selection_and_install`. Production
   changes, retained geometry selection, final JIT replacement and deletion still
   require the active agent to review the completed results and execute the tools.

Active exec sessions at handoff: idle/pilot worker 37617, continuation 52752.
Do not start another pilot or full batch while these jobs are active. Do not edit
any candidate directory or current production dependencies before they finish.
The optional user question about when the outside GPU task ends is pending; no
permission is needed to continue the already authorized tuning when GPUs are idle.

Finalization tools reviewed, actual use pending full295:
- `retained_tools/README.md`: prepare selected families, byte-identical rebuild,
  production prune proof, installation, then explicit cleanup plan/apply. Review
  details in `retained_tools/REVIEW.md`. All actions still unexecuted.
- `finalize_tools/README.md`: fresh JIT build/install after production prune;
  registered population 87 IDs including preserved other families, 208 device
  symbols. Updates default and measurement paths; retires old retained_jit.
- Execute final JIT replacement before deleting rollback/candidate sources needed
  by any remaining proof. Preserve CSV/JSON/textual ISA evidence; do not delete
  unrelated historical reports. Both families of tools must finish for the user's
  requested deletion to be complete.
