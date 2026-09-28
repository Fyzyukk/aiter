# Latest completed 87-shape result: v11

The complete five-round collection has **87 OPUS wins**, with no CK, CKTile, ASM,
or tied winners. All 87 winning OPUS kernels are independent experimental
implementations. The original registered kernel 9000 and production defaults
remain unchanged.

The collection has 57 shapes that win every paired round and 30 that win by
median without winning every round. Its per-shape geometric mean latency
reduction is 4.64346% versus the same-batch external reference and 13.01541%
versus the same-batch old OPUS reference.

The formerly remaining shape `(1536,7168,384)` now selects OPUS kernel **15940**
from the complete `targeted1_v11_r5` result: about **12.51579 µs** versus
**12.64447 µs** for `cktile_11_split0`, an improvement of about **1.02%**.

- [All comparisons](comparison.csv) retain OPUS identity, external reference,
  recorded rounds, and source batch, and explicitly name the overall algorithm.
- [Overall winners](overall_winners.csv) list winning algorithm, implementation,
  library, kernel ID, and split-K alongside the best OPUS and external candidates.
- [Display summary](summary.json) contains algorithm counts and input hashes.
- [Underlying collection](../coverage87_v11/summary.json) contains the full
  ordered provenance and comparison statistics.

This derives from the original `coverage87` ordered batches followed by
`targeted1_v11_r5`. The complete latest row replaces the earlier result regardless
of latency. Previous `coverage87` and `display_v10` files are unchanged. This
snapshot covers only the original 87 shapes; the 295-shape retune is separate.
