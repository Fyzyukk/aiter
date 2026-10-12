# Latest recovery: v18, 2026-10-09

Goal remains active and incomplete. OPUS must beat FlyDSL BMM stably on every
one of the 745 shapes, pass signed/reference/guard validation, and be integrated
into production parameterized dispatch. Current-source valid performance: 0/745.
Historical BMM-only 428 wins / 317 losses is unchanged historical evidence.

Live watcher PID 2052695, exec session 39225, log `watch_stdout_v18.log`.
Verified alive after restart. Previous v17 watcher PID 2050917 was stopped only
while idle, exit 130 confirmed. Check `status.json` and live PID before restart.
Only physical AMD-SMI GPUs 0–3 authorized; all eight monitored. The optional
question about extending the allowed device range remains unanswered.

v18 follows the v17 KFD ownership gate from `RESUME_V17.md`, with one correction:
a queue directory that still exists but lacks gpuid/type metadata is unknown,
not a removed queue. Only a disappeared directory can be skipped. Regression
`incomplete_live_queue_rejected` passed with the other six KFD ownership cases.
AST, pyflakes and diff whitespace checks passed. No GPU contexts or kernels
were created by these CPU checks.

The watcher requires empty AMD-SMI process and KFD owner lists, low activity
and memory, and locks. It checks ownership immediately before claiming and
every two seconds while a worker runs. External ownership or unknown KFD data
stops our worker, preserves invalid receipts, and waits again. No external
process is stopped. Front-four owners at restart were
2931581 / 2932814 / 2932123 / 2929501, gfx100%, about284GiB each.

Benchmark source unchanged. Two owner-isolated numerical smokes remain valid:
1×768×7168 and16×2048×7168;41 variants/39 public API variants/451 signed checks,
invalid0, guards passed. Normal smoke2/9; private10shape smoke pending.
`audit_results.py` revalidated them and excluded the earlier contaminated worker.
It audits hashes/physical mapping/public seeds/raw timing rounds/post-timing
checks/medians/selected CSV, including actual scalar tuned policy selection.
It only certifies candidate evidence; production integration/tuned dispatch
GPU replay needs separate final validation. Never complete the goal from its
exit code alone.

Continue the prepared experiment described in `RESUME_V16.md`: normal smoke,
private smoke,745 baseline, measured-loss parameter tuning, all legal splits,
legal-domain private fusion tuning,745 replay. Each shape has its own worker
receipt. Inspect new worker logs/results immediately when an idle window arrives.
Private winner integration and a fresh public full745 run are still required.
If the round reports unresolved shapes, continue mechanisms from measured gaps.
No production kernel or formal tuned CSV changed this continuation.
