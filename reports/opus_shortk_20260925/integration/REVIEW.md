# Short-K CPU-only integration review

Status: **passed** (103/103 checks).

- All first 25 frozen target shapes admit exactly their planned new ID: {9040: 12, 9041: 12, 9042: 1}.
- All 295 supported shapes retain their pre-edit 9-ID candidate sets. The historical 1,275 candidate records for IDs 9000/9010/9011/9012/9020 also match exactly.
- The 10 pre-existing address-range exclusions remain excluded.
- 4098 scalar cases cover exact K, M64/N256 alignment, invalid dimensions, and signed-int byte-limit boundaries. Unsupported architecture/output dtype is rejected.
- Frozen old names/metadata, protected source/helper hashes, and old per-kernel generated files remain unchanged. Aggregate generated files preserve existing declarations and add only the new family.
- Default compile sets match commit `10ab50645d1f25e11844b814b66002b27181dfbf`; all 230 tracked production configuration files match its bytes. New IDs are not defaults.
- New generated wrappers select the correct fixed-K traits/device symbol and enforce the exact-K/shape/scale/byte-range contract before constructing kernel arguments.

Re-run from the repository root after `offline_compile.py` has regenerated codegen artifacts:

```sh
python -B reports/opus_shortk_20260925/integration/review.py
```

The script uses only the standard library, reads old adapter functions through AST extraction, and blocks runtime imports. It does not discover a GPU, launch a kernel, or import `aiter`/`torch`. Full records and input hashes are in `review.json`. This is integration evidence; GPU numerical correctness and performance remain unmeasured.
