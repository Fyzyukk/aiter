# Scale issue/publish final scope decision

Retain only the already existing9020 fixed384 variant. Reject global9022. Production integration and official gates remain with root.

The seven fixed384 winner medians are all positive, all seven paired medians are positive, and34/35 paired observations are faster. Six shapes win5/5 and one wins4/5; no actual winner has a stable regression. Equal-shape Event gain is+2.5715%.

The retained12288×7168×384 round2 speedup is0.834618: baseline67.4383us versus candidate80.8014us. It remains in every summary. It makes that pooled round across seven shapes−0.4930% and that shape AB geomean−4.1741%; the whole seven-shape AB aggregate remains+1.6154%. Its cause is unverified, and it is not deleted, assigned zero cost or rerun until positive.

Global9022 has+0.8208% equal-shape aggregate gain, but six actual winners lose5/5 and32 medians are negative. The20 longK16384 winners have12 negative paired medians and three stable losers; their+0.0956% aggregate is too weak to offset recurrent costs under the established global standard. No newK threshold is introduced.

The original mechanical analyzer recommendation is preserved in [coverage_independent_review.json](coverage_independent_review.json). Its every-pooled-round9020 rule is stricter than the final root decision, which explicitly uses per-shape stability while retaining the complete negative-round risk.

Full decisions, scope, rawcosts and remaining official gates: [retention_decision.json](retention_decision.json).
