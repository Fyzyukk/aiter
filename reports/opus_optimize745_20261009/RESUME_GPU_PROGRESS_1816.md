# Latest GPU recovery: 2026-10-09 18:16 UTC, watcher v19

Goal remains active and incomplete: all745 OPUS shapes must beat FlyDSL BMM
stably after signed/reference/guard validation and production integration.
Current-source valid performance remains0/745.

Live watcher PID2059925, exec session23410, log`watch_stdout_v19.log`.
Confirmed live after this idle window. Do not restart a live watcher.
Only physical AMD-SMI0–3 authorized; all8monitored. Optional range question
unanswered. Latest front-four owners3297704/3298723/3298108/3297438, initially
gfx0%/~67GiB as new tasks initialized. Four other cards remain occupied.

New owner-free window around18:14–18:15 UTC produced four valid numerical
workers on physicalGPU0/BDF0000:05:00.0. Normal smoke now6/9:

| Shape | Numerical variants | Public API variants | Signed checks |
|---|---:|---:|---:|
| 1×768×7168 | 18 | 17 | 198 |
| 16×2048×7168 | 23 | 22 | 253 |
| 48×7168×16384 | 20 | 19 | 220 |
| 112×768×7168 | 24 | 23 | 264 |
| 256×7168×7168 | 27 | 26 | 297 |
| 672×7168×384 | 12 | 11 | 132 |

All valid workers completed with invalid0, signed/reference/output/workspace
guards passed;124numericalvariants/118publicAPIvariants/1364signedchecks total.
`validated_smoke_first6.json` records receipts. `evidence_audit_smoke6.json`
rechecks hashes/physical mapping/checks and excludes three contaminated workers.
Total receipts9,valid6. No performance measurements or private numerical results.

Fourth smoke's first attempt was stopped when KFD-only externalPID3290590
allocated41190162432bytes (~38GiB) without an AMD-SMI process entry or queue.
Automatic retry subsequently completed112×768×7168. Seventh smoke's first
attempt was stopped when externalPID3297704 created a queue and440004608byte
allocation; new tasks then owned allfrontfour. Preserve/exclude both receipts.
Do not loosen KFD/memory gates because activity or AMD-SMI processlist is empty.

Remaining normal smokes:1024×7168×16384,1088×7168×768,16384×65536×1536.
Watcher automatically resumes the seventh shape after an allowed card is free.
Then private10shape smoke,745baseline,measuredloss tune,allsplits,legalMfusion,
full745replay. Inspectnewlog/results promptly;fixactual failures when observed.
Private winners still require parameterizedproductionintegration+freshpublic745.

Latest code/build state remains`RESUME_V19.md`:privatehostguardsfixed,
22deviceimagesunchanged,88C++validplans/279rejectedcalls passed.
Benchmark/watcher/KFDhelper/plan unchanged in this continuation;validsmokes
reused. No production source or formal tuned CSV changed.
