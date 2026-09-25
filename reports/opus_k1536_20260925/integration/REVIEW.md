# K1536 CPU-only integration review

Status: **passed** (127/127 checks).

- All eight frozen K1536 target shapes retain their old candidates and add 9050 (`kunroll12`) plus 9051 (`kunroll2`).
- The first 25 short-K targets keep exactly their existing 9040/9041/9042 fixed-K match: {9040: 12, 9041: 12, 9042: 1}; neither new ID accepts those K values.
- All 295 supported shapes retain their pre-edit 12-ID candidate sets, including the three existing fixed-K IDs. The historical 1,275 records for IDs 9000/9010/9011/9012/9020 also match.
- The 10 pre-existing address-range exclusions remain excluded. 4322 scalar boundary cases check exact K, M64/N256 alignment, positive dimensions, and int32 byte limits.
- Frozen old kernel names/fields, protected source/helper hashes, and all old per-kernel generated files remain unchanged. `k_loop_unroll` is `None` on every old ID.
- Generated wrappers bind distinct K1536 traits/device symbols and enforce exact K before constructing kernel arguments. Aggregate generated files preserve old declarations.
- Default compile sets match commit `10ab50645d1f25e11844b814b66002b27181dfbf`; all 230 tracked production configuration files match its bytes. New IDs are absent from defaults.

Re-run from the repository root after the offline codegen artifacts have been generated:

```sh
python -B reports/opus_k1536_20260925/integration/review.py
```

Only Python standard-library code and isolated scalar AST functions execute. Runtime imports are blocked; no GPU discovery, runtime compilation, or kernel launch occurs. Full input hashes and per-shape records are in `review.json`. GPU numerical correctness and performance remain unmeasured.
