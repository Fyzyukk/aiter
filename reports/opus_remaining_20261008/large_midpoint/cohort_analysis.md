# 9030 midpoint scheduling evidence

Clean Oct8 recollect: 688 complete waves, 86 publication-release cohorts, 288 MFMA per wave.

Interior: 774 cohort/tile observations, 6192 correlated wave/tile observations. Units are shader clocks.

| Metric | Median | p90 |
|---|---:|---:|
| Cohort end-barrier arrival spread | 908.0 | 928 |
| Cohort last-MFMA issue spread | 900.0 | 904 |
| Latest arrival to earliest release | 4.0 | 8 |
| All-wave barrier event duration | 356.0 | 900 |
| Latest-arrival barrier event duration | 8 | 8 |
| All-wave combined VMEM/LDS wait event duration | 76.0 | 104 |
| Latest-arrival combined VMEM/LDS wait event duration | 84 | 84 |
| All-wave last MFMA to barrier attempt | 104.0 | 140 |
| Release to next first MFMA | 374.0 | 1180 |

Interior end-barrier durations correlate with inter-wave MFMA/arrival skew; the latest arrival-to-release residual is reported separately.
The barrier publishes K+2 and retires all K+1 operand readers. Its wave share cannot be deleted as whole-kernel loss.
Midpoint moves publication of K+1 before its reads and delays K+2 overwrite until all K readers retire, permitting the first eight current MFMAs to overlap pending K+1 supply.
Final advance must retain a terminal C-alias retirement barrier; loops=1 retains the original post-operand prologue barrier.
Wait duration combines VMEM and LDS queue dependencies; no individual memory return time is inferred.
Cohorts come only from synchronized initial publication release and are verified at every interior release; no physical SIMD mapping is claimed.

Retained large-spread observations: 1. Group8 tile6 has first/last-MFMA spreads 5388/5396 clocks while end-barrier arrival spread is680; four earlier waves spend about4830 clocks in the combined wait and the later waves spend about5500 clocks in async issue events. No samples are discarded or reclassified as ownership contamination.

The isolated candidate changes only the 9030 advance schedule. The scale-first prologue, loop predicates, matrix/scale layouts, C64 base, row guards, and output code are preserved.

The existing end barrier is moved after the first M repeat for ordinary transitions. Matrix K+2 is issued after this barrier. The last transition retains a second terminal barrier before C staging.

Candidate ISA review, numerical guards and no-profiler Event measurements are still required. Full cohort and wave data: [cohort_analysis.json](cohort_analysis.json).
