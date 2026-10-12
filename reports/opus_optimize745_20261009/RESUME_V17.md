# Latest recovery: v17, 2026-10-09

The goal remains active and incomplete: OPUS must beat FlyDSL BMM stably on all
745 shapes, with signed/reference/guard validation and production integration.
Historical BMM-only comparison is 428 wins / 317 losses. Current-source valid
performance shapes: 0.

Watcher restarted and verified alive: PID 2050917, exec session 46407,
log `watch_stdout_v17.log`. It holds the experiment watcher lock. Check the
live PID and `status.json` before any restart. Only physical AMD-SMI GPUs 0–3
are authorized; monitor all eight. The earlier optional GPU range question
remains unanswered.

v17 adds `kfd_owners.py`: maps physical BDF to KFD GPU ID and reads per-process
VRAM accounting plus compute/DMA queues. Any queue or VRAM above 8 MiB counts
as ownership. Observed 2 MiB registrations with no queues are ignored. Missing
accounting for a live process is unknown, never an empty owner list.

The watcher requires both AMD-SMI and KFD ownership to be empty, checks again
before claiming, and monitors external KFD owners during workers. Unknown KFD
reads interrupt our worker as contaminated. Performance receipts bind the KFD
helper SHA and watcher SHA; performance reuse requires the same helper SHA.
No change to `benchmark.py`, so the two validated numerical smokes are reused.

`audit_kfd_owners.py` passed registration-only, 38 GiB VRAM without AMD-SMI,
queue without VRAM, own PID filtering, unknown reads, and physical 0–3 limits.
Actual topology/owner mapping matched the live front-four processes
2931581 / 2932814 / 2932123 / 2929501. Front four were gfx 100% with about
284 GiB allocated each at restart. All eight GPUs had external owners.
AST, pyflakes (existing `/tmp/opus745_pyflakes`), and git diff whitespace passed.

Valid numerical evidence:

- `smoke_0000_8aec31a05369.1791566980047369445`: 1×768×7168,
  18 variants, 17 public API variants, 198 signed checks.
- `smoke_0001_779287cc56cb.1791566991159003697`: 16×2048×7168,
  23 variants, 22 public API variants, 253 signed checks.
- Both owner-isolated receipts passed all numerical and output/workspace
  guards; benchmark/plan/output hashes verified. Total 41 variants,
  39 public API variants, 451 signed checks. Earlier contaminated worker excluded.
- Normal smoke progress is 2/9. Private fusion dynamic validation is pending.

Prepared experiment and pending workflow remain as documented in
`RESUME_V16.md`: 204 public configurations, 453 cached BMM controls,
22 private fusion configurations with exact public controls. Normal smoke,
private smoke, baseline 745, measured-loss parameter tuning, all legal splits,
legal-domain private fusion, and replay each persist one shape per receipt.
On new GPU evidence inspect logs/raw JSON/receipts immediately and fix failures.

Private winners require production parameterized integration and a fresh full
745 public API replay. If terminal status is `more_kernel_optimization_required`,
continue from measured unresolved gaps. A running watcher or CPU compilation
does not satisfy the performance goal. No production kernels or formal tuned
CSV were modified in this continuation.

`audit_results.py` was added while waiting. It rechecked the two live-source
smoke receipts (41 variants / 39 public API variants / 451 signed checks) and
excluded the contaminated worker. It audits all replay raw rounds, medians,
public seeds and selected CSV once they exist. Synthetic corruption checks
rejected missing rounds, output failures, forged medians and missing API seeds.
It only certifies candidate evidence; production integration/tuned dispatch
replay must be validated separately before the goal can be complete.
