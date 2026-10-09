# Narrow candidate retention recommendation

Recommend keeping the tested issue-before-publish change for the existing9023 runtime and9024 fixed7168 entries. The merge patch is prepared and production remains untouched by this script.

| Cohort | Shapes | Event geometric-mean speedup | Positive medians | Five-of-five faster |
| --- | --- | --- | --- | --- |
| 9023_runtime_winners | 4 | +1.1173% | 3 | 2 |
| 9024_fixed_winners | 9 | +1.2745% | 9 | 9 |
| all13_narrow_winners | 13 | +1.2261% | 12 | 11 |

9023 measured K7168/K16384 cohort speedups are +1.9162% /+0.3246%. All five pooled paired rounds are positive for the four winners and also after adding theM16 boundary. No actual winner is5/5 slower. This supports retaining the existing runtime entry under the same finite-cost standard used for9021; it does not justify a new K threshold.

| Explicit finite cost | Baseline / candidate Event us | Call-time increase | Paired faster rounds |
| --- | --- | --- | --- |
| 9023 / 544×7168×16384 | 99.5282 / 100.0678 | +0.5422% | 2/5 |
| 9023 / 16×128×384 | 5.4558 / 5.4785 | +0.4165% | 1/5 |

9023VGPR increases232→251 and SGPR lane spill46→48; its ISA grows1.36%. LDS and private/VGPR spill/AGPR remain unchanged. This is a long-term compiler headroom risk and an unmeasured support-domain risk. The nine9024 winners keep110VGPR and zero spill while SGPR rises58→60 and ISA grows1.81%. The unchanged runtime control itself moves0.49%, so small individual differences should not be overinterpreted.

Independent CPU replay of the result analysis passed the strict owner/claim/fingerprint check, all15 signed eight-repeat reference/guard checks, and all five alternating Event rounds on the same51-address pools. No new GPU test or build was run by this analysis.

Exact tested production-target patches: [formal_merge/narrow.patch](formal_merge/narrow.patch). Before/after hashes and scoped snapshots: [formal_merge/manifest.json](formal_merge/manifest.json). Full statistics, finite costs, resources and remaining official gates: [retention_recommendation.json](retention_recommendation.json).

Root should coordinate the source application with the other candidates and complete the official linked identity/API gates before final adoption. Shared helpers,9000,9021 and dispatch thresholds are not part of this patch.
