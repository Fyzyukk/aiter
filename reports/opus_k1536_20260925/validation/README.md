# K=1536 CPU validation

Run from the repository root:

```sh
python3 reports/opus_k1536_20260925/validation/check_layout.py
```

The script extracts the current production code on every run, compiles an x86
host executable with the pinned Clang using `--offload-host-only
--offload-arch=gfx950`, inspects its ELF dependencies and undefined symbols, and
runs it. It does not import `aiter` or `torch`, discover GPUs, call HIP/HSA, or
execute GPU code. `layout_check.json` records the source and generated-file
SHA-256 hashes, exact compiler command, adaptations, scope, and process output.
The script verifies that sources did not change during the run and removes any
previous result before beginning, so a failed rerun cannot retain a stale pass.

Both production traits, `LoopUnroll=12` and `LoopUnroll=2`, passed:

| Check per specialization | Result |
| --- | ---: |
| Matrix producer byte coordinates | 688,128 |
| Matrix consumer byte coordinates | 2,162,688 |
| A coordinate checks against M tails | 3,538,944 |
| B coordinate checks against N bounds | 18,874,368 |
| Output coordinates | 294,912 |
| Output vector bounds checks | 884,736 |
| Scale cases / exact scale values | 144 / 4,423,680 |
| Matrix prefetches | 12 |
| Logical MFMA calls per wave | 288 |
| Barriers | 13 |
| Complete scale-panel loads | 1 |
| Scale panel / total LDS bytes | 2,328 / 120,600 |

Coverage uses all 512 threads and all 12 K groups. M tiles have 64, 128, or
192 valid rows, with origins 0 and 192. N is 256, 512, 6144, 7168, 8192, or
16384, checking both the first and last N tile. Scale cases also use two byte
patterns that jointly distinguish all tested global scale addresses. Missing
A-scale rows must contain `0x7f`, and every scale-panel byte has one producer.

The matrix check executes the actual production matrix-layout declarations,
`issue_matrix_prefetch`, `load_a`, and `load_b` lambdas. Integer address tokens
preserve full coordinates through bounded CPU LDS arrays, checking unique
production, correct consumption, and both physical slots. Output coordinates
execute the actual production output-partition declarations. The existing
layout helpers and OPUS adaptor definitions are copied with host/device
qualifiers adapted; production files are unchanged by validation.

The scale check executes the actual `load_scale_panel` and `read_scales`
lambdas. Their indexing and branching are unchanged; thread IDs are enumerated
on the CPU, and memory operations are replaced by bounds-checked wrappers.
The pair variant exercises `int` K indices for the same groups used by its
production loop, and `number<>` indices for its prologue and tail.

The schedule check executes the extracted prologue, `advance` lambda, both
`LOOP_UNROLL` branches, and final K11 compute. Its instruments check LDS
publication, barriers before slot reuse, completion of all uses before
register replacement, and exactly one visit to each K/M/N combination.
The actual `compute` lambda calls a CPU MFMA oracle that checks A/B groups,
packed A-scale selectors, B-scale halves, accumulator indices, and logical
LDS readiness. Both variants produce identical 483-event traces. Only the pair
variant supplies runtime indices for its ten refills and ten scale reads.

`build.log` contains one existing OPUS deduction-guide attribute deprecation
warning. `elf_dynamic.txt` and `undefined_symbols.txt` show no HIP/HSA dynamic
dependencies or unresolved HIP/HSA symbols.

This checks coordinates and logical scheduling. It does not measure GPU
performance, test device numerical output, simulate hardware latency, or prove
ISA hazard behavior. Hardware OOB zero-fill, store discard, and async Wave64
lane placement semantics are assumed by the CPU memory model.
