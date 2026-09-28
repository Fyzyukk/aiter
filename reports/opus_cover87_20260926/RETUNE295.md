# 295-shape retune data flow

`shapes295.csv` is the exact 295-member supported catalog copied from
`../opus_local_gap_current_20260925/shapes.csv`, with SHA256
`20b92df1957f00766b309153cafd137d01c40aeb941c6d61c2b62c052d3ff9b5`.
The ten addressing-excluded shapes remain outside this catalog; the 295 supported
and ten excluded shapes partition the original 305. `plan295.json` retains the
original seven 87-shape cohorts and adds the remaining 208 members.

The measurement is complete: `full295_r3` passed all 295 shapes, followed by
`close295_r5` for the 58 close or round-inconsistent shapes. Final results are in
[`display295_v1/README.md`](display295_v1/README.md): 293 OPUS median wins and
two CKTile wins. The final 295 OPUS choices use 30 IDs, with 196 registered and
99 experimental choices. The commands below document the completed data flow;
use fresh output/batch names for any future rerun.

## Summarize the full three-round measurement

```bash
python reports/opus_cover87_20260926/summarize.py \
  --batch reports/opus_cover87_20260926/full295_r3
```

This creates `full295_r3/results/close_shapes.csv` for shapes with a median gap
within ±3%, or with paired round wins other than zero or all three. If this list
is empty, no close-shape confirmation is needed.

An optional display of the complete initial three-round result can be generated
without additional measurements:

```bash
python reports/opus_cover87_20260926/collect_scope.py \
  --shapes reports/opus_cover87_20260926/shapes295.csv \
  --cohorts reports/opus_cover87_20260926/plan295.json \
  --batches reports/opus_cover87_20260926/full295_r3 \
  --output-dir reports/opus_cover87_20260926/coverage295_r3

python reports/opus_cover87_20260926/make_scope_report.py \
  --input-dir reports/opus_cover87_20260926/coverage295_r3 \
  --output-dir reports/opus_cover87_20260926/display295_r3
```

## Confirm the close subset for five rounds

```bash
python -u reports/opus_cover87_20260926/launch.py \
  --shapes reports/opus_cover87_20260926/full295_r3/results/close_shapes.csv \
  --batch close295_r5 \
  --external-mode finalists \
  --full-batch reports/opus_cover87_20260926/full295_r3 \
  --rounds 5 \
  --experiments reports/opus_cover87_20260926/experiments295.json

python reports/opus_cover87_20260926/summarize.py \
  --batch reports/opus_cover87_20260926/close295_r5
```

If the close subset is effectively the full catalog, the same finalists command
may use `shapes295.csv` instead. Finalists preserve the same GPU as the full sweep,
retain its valid external candidates within 5% of the fastest external candidate,
and compare all legal configured OPUS candidates. Frozen measurement and
experimental sources, configuration files, and binaries must remain unchanged
throughout the measurements.

## Collect the latest complete row per shape

```bash
python reports/opus_cover87_20260926/collect_scope.py \
  --shapes reports/opus_cover87_20260926/shapes295.csv \
  --cohorts reports/opus_cover87_20260926/plan295.json \
  --batches reports/opus_cover87_20260926/full295_r3 \
            reports/opus_cover87_20260926/close295_r5 \
  --output-dir reports/opus_cover87_20260926/coverage295_v1

python reports/opus_cover87_20260926/make_scope_report.py \
  --input-dir reports/opus_cover87_20260926/coverage295_v1 \
  --output-dir reports/opus_cover87_20260926/display295_v1
```

`collect_scope.py` requires an explicit catalog and accepts complete three- and
five-round input batches by default. Input order is oldest to newest. A newer
shape replaces its entire earlier row even if its latency is worse; no result is
selected by comparing latency across batches. More complete five-round subsets
can be appended in chronological order and collected into a new output directory.
`--min-rounds 5` is available only when every supplied input batch has five rounds.

Every row retains `rounds`, `external_mode`, GPU identity, source batch, selected
OPUS library/kernel ID, and same-batch external comparison. The collection must
cover the catalog exactly once. Its summary reports `rounds_by_shape` as counts,
for example `{"3": 208, "5": 87}`; `rounds` is null for a mixed collection.
Five-round stability counts include only shapes with five recorded rounds.

The display distinguishes `algorithm=OPUS` from `implementation=experimental` or
`registered`. `experiment` never means CK. `comparison.csv` retains every raw
comparison column and adds the overall winning algorithm. `overall_winners.csv`
includes the winning algorithm, implementation, kernel/library, kernel ID,
split-K, actual rounds, best OPUS candidate, and external reference. Exact ties
name both algorithms without choosing a single winning kernel. `summary.json`
counts OPUS/CK/CKTile/ASM wins overall and separately for three- and five-round
shapes. Missing old OPUS baselines, if any, are counted and omitted from old OPUS
geometric means.

Both scripts read CSV/JSON and generate derived files only. They require new
output directories and do not edit measured files. Original `collect_latest.py`,
`make_opus_report.py`, and all previous collection/display directories remain
available for the historical 87-shape results.

## Latest completed 87-shape snapshot

`coverage87_v11` reapplies the original ordered batches from `coverage87` and then
overlays the complete `targeted1_v11_r5` row. `display_v11` shows **OPUS 87 wins,
CK 0, CKTile 0, ASM 0**, with all 87 winning OPUS implementations experimental.
All shapes have five recorded rounds; 57 shapes win all five paired rounds.
The same-batch median geometric mean latency reduction is 4.64346% versus the
external reference and 13.01541% versus old OPUS. This 87-shape snapshot does not
stand in for the new 295-shape measurement.
