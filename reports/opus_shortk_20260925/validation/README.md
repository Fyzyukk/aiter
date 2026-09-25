# CPU-only short-K validation

Run from the repository root:

```sh
python reports/opus_shortk_20260925/validation/check_layout.py
```

The checked implementation passed for all three fixed K values on 2026-09-25.
The complete compiler command, source hashes, host adaptations, and stdout are
recorded in [layout_check.json](layout_check.json).

| Fixed K | K128 groups | Matrix LDS bytes | Scale LDS bytes | Total LDS bytes | MFMA calls per wave | Barriers |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 3 | 118272 | 582 | 118854 | 72 | 4 |
| 768 | 6 | 118272 | 1164 | 119436 | 144 | 7 |
| 1024 | 8 | 118272 | 1552 | 119824 | 192 | 9 |

The checker executes the production matrix layout helpers with host qualifiers
and instantiates the production short-K traits. For each K128 group it checks
every cooperative A/B byte write, every register-fragment byte read, padding
exclusion, and the two-slot LDS allocation. Across the three variants, 974848
global A/B bytes are covered exactly once and 3063808 consumer bytes match their
expected coordinates.

The production output partition layout covers each 192x256 element exactly once
for strides 256, 512, 6144, 7168, and 8192. The checker verifies 737280 output
coordinates and 1105920 vector-bound cases across the three variants. Valid-row
counts 64, 128, and 192, combined with the first and last N256 tile positions,
exercise the M64 tail contract. Every four-element store is entirely valid or
entirely outside the row bound.

The actual scale producer and consumer lambda bodies are extracted from the
short-K pipeline and compiled with bounded CPU memory operations. Each variant
passes 72 parameter combinations covering valid-row counts 64/128/192, block
row origins 0/192, N values 256/6144/7168, first/last N256 tile positions, and two
byte patterns that identify source coordinates. The 3133440 decoded scale values
match their matrix coordinates. Every resident scale byte has exactly one
producer, every source read stays within its exact input extent, missing A rows
receive neutral `0x7f`, and unused A-scale packing bytes remain zero. The compact
B scale halves match the N128 groups consumed by each MFMA repeat.

The actual prologue, compile-time `static_for` schedule, and final compute are
also extracted and executed with instrumented operations. The checker verifies
exactly 3/6/8 matrix prefetches, exactly one compute per K/M/N repeat, exactly one
scale-panel load, no out-of-range prefetch, no LDS slot reuse before reads finish
and a barrier occurs, and no register replacement before its final use. The
expected scale group accompanies every compute.

Compilation uses the pinned clang directly with `--offload-host-only`, explicit
`--offload-arch=gfx950`, and `OPUS_ENABLE_RUNTIME_QUERY=0`. It does not link the HIP
or HSA runtime. [elf_dynamic.txt](elf_dynamic.txt) and
[undefined_symbols.txt](undefined_symbols.txt) record the inspected dependencies
and undefined symbols before the CPU executable runs. No GPU discovery or GPU
calls occur.

These results establish coordinate and logical schedule consistency. They do
not provide GPU numerical correctness or performance results, simulate hardware
latency, or prove emitted ISA hazards. Buffer OOB zero-fill and store-discard
behavior are hardware semantics assumed by the coordinate-bound checks.
