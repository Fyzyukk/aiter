# Latest recovery: verified wait, 2026-10-09 18:41 UTC

Goal remains active and incomplete: all 745 OPUS shapes must beat FlyDSL BMM
stably after signed/reference/guard validation and production integration.
Current-source valid performance remains 0/745. Private GPU results remain zero.

Live watcher v19 PID 2059925, exec session 23410, log `watch_stdout_v19.log`.
The actual `/proc` command line and fresh status were verified again. The watcher
was not restarted. Only physical AMD-SMI 0–3 are authorized; all eight monitored.
Four permitted cards remained externally owned throughout this recovery wait:
SMI0/HIP1 PID3297704, SMI1/HIP3 PID3298723, SMI2/HIP2 PID3298108,
SMI3/HIP0 PID3297438. Each sampled gfx 100% and about 246GiB used.
`verified_wait_1841.json` stores exact BDFs, owners, queue counts and hashes.

No new GPU worker or valid result in this turn. Normal smoke remains 6/9:
124 numerical variants,118 public API variants,1364 signed checks. Nine receipts
exist:six valid,three contaminated and excluded. Recovery evidence audit passed
all existing valid receipts with no audit failures; exit2 correctly indicates
incomplete 745 evidence. See `evidence_audit_recovery_1829.json`.

The previous numerical progress turn produced four valid workers, followed by
one verified-wait turn and this verified-wait turn. This is the second consecutive
unchanged GPU wait since that progress; do not classify it as new kernel/GPU
progress or incorrectly infer an established three-turn impasse.

README stale 2/9 current-progress wording was corrected; validation metadata now
matches the six audited receipts. Prior validation saved separately. No benchmark,
watcher, private library, production kernel/source or formal tuned CSV changed.

Resume automatically at normal smoke1024x7168x16384,then1088x7168x768 and
16384x65536x1536; then private10shape smoke,baseline745,measured-loss tuning,
all legal splits,legal-domain private fusion and full745 replay. Inspect actual
failures/results when windows arrive. Do not add speculative candidates without
measured gaps. If private winners are selected,integrate into the parameterized
production/workspace contract and perform a fresh public745 replay before
completion. Terminal unresolved state also requires continued mechanism work.

Detailed GPU evidence: `RESUME_GPU_PROGRESS_1816.md`.
Current code/build/host-contract state: `RESUME_V19.md`.
Do not restart a live watcher or stop external owners.
