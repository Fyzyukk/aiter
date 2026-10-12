# Latest recovery: v13, 2026-10-09 17:22 UTC

Goal remains active: all745 shapes must pass current-source numerical/guard checks
and both timing protocols, then production integration/public replay if any private
winner is selected. Historical BMM-only results remain428 wins/317 losses.
Current valid GPU receipts:0. One earlier contaminated smoke is retained/excluded.

Latest watcher PID2037733, exec session98791, log`watch_stdout_v13.log`.
Read`status.json` and check the live PID before starting any watcher.
PhysicalSMI0-3 remain the only permitted devices. They have external owners
2256588/2257233/2256872/2256371 and100% activity at the latest check.
PhysicalSMI4 was owner-free at17:18-17:22;SMI7 briefly idle then busy again.
An optional async question asks whether to expand to all8 cards. No answer yet;
do not infer permission from elapsed time. Retain0-3 unless user changes it.

This continuation added:

- PublicAPI validation for every public task with seed29 and every public finalist
  with seed101, including selected replay parameters;`public_api_checks` receipts.
  Baseline failure detection uses its actualsplit tag.
- `audit_public_api.py`: AST-extracted actual adapters on CPU; public/prepared
  argument order, exactworkspace slice, directNone, numerical/guard/extent rejection.
- `fused_lds/`:6 private last-arrival kernels copied from current LDS compute,
  M48xN64/M64xN64/M64xN128,stages3/4,256threads. Existing204 modules already
  include matching public controls. All6 compile,scratch/spill0,VGPR46-102.
  Separate`build_fused_lds.py` and source-bound`build.json`,private.so,deviceISA.
- Two private libraries use tags`fused_0..9` and`fused_lds_0..5`.
  Benchmark validates metadata/library SHA and selects the correctC ABI library.
  Watch receipts hash bothprivate source/library/build sets.
- Fusion searches unresolvedM<=2048; config_supports_shape filters each config.
  Register remainsM<=512,LDSM<=2048. Private rows remain opus_private and
  private_register_fused/private_lds_fused,not consumable by production API yet.
- Fusion smoke10shapes,all7K lengths,M1088 included;1562OPUS tasks,
  776private/786public. `fused_smoke_coverage_v13.json` is the currentcount.
- `audit_fused_candidates.py`:16configs,2496full-grid enumerations,
  83936legal split/resource checks,disjoint complete fragments,2actualAST
  C ABI adapters,release/atomic/bypass/reset staticISAchecks. GPU pending.

All production source/input hashes stillmatch originalplan. No production kernel
or formalCSV changed. AST/pyflakes/diffchecks pass. Script revisions before
changes are SHA-preserved in`harness_revisions/`. v11/v12coverage files retained.

Next: inspect watcher/status. When permitted card becomesidle, inspect first
smoke log immediately,fix realadapter/numerical failures if needed,preserve all
failedreceipts. Then finishbaseline745,parameters,allsplits,fusion,replay. If
`more_kernel_optimization_required`,continue mechanisms from measuredunresolved.
If allprivatewins,integrate parameterizedproductionpipeline/workspacecontract and
rerun fullpublicAPI745. A runningwatcher or CPUbuilds do not complete the goal.
