# Independent 192×256 wide-N MXFP8 experiment

Status: **compiled and CPU layout checks passed; GPU correctness and performance
are pending. No speedup or correctness acceptance is claimed.** GPU 7 became
occupied before the first smoke test could initialize a GPU context. Its failed
attempt is retained as `smoke_r1_run.json` / `smoke_r1.log`.

The frozen `wide_r5` worker was launched with PID **17829**. It waits for GPU 7
to produce three idle samples five seconds apart (utilization 0%, VRAM ≤1%,
stable GFX activity), then runs a finite pilot. The wait expires after 12 hours.
Read [queue status](wide_r5_queue.json), [launch record](wide_r5_launch.json),
and, once started, `wide_r5_run.json` / `wide_r5.log`. A recorded PID/status
alone does not prove that the process is still alive.

## Scope and source

This is a separate experimental family for the five remaining large-N losses:

| M | N | K |
|---:|---:|---:|
| 1088 | 6144 | 7168 |
| 1152 | 6144 | 7168 |
| 1088 | 7168 | 3072 |
| 1024 | 7168 | 16384 |
| 1088 | 7168 | 16384 |

- [Pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh)
- [Traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh)
- [Isolated launch ABI](launch.hip), [build](build.py), [benchmark](bench.py)

The launch contract is positive M divisible by 64, N divisible by 256, K
divisible by 128; contiguous native FP8 matrices, existing B preshuffle,
column-major native E8M0 A scales, row-major E8M0 B scales, BF16 output, and
the existing signed-32-bit tensor byte limit. M is rounded up only in the grid.
Bounded buffer resources zero-fill missing A rows and discard missing output
rows. Missing A scale rows are explicitly set to `0x7f` (1.0), so they never
alias the next scale column. There is no external padding allocation, split-K,
or output workspace.

These IDs belong only to the experimental `launch` ABI. They are **not added
to the public tuner registry or production selection CSVs**. Existing
9000/9010/9011/9012/9020 source and selection behavior remain as before.

## Four initial configurations

All use a 192×256×128 tile and two matrix LDS slots. K0 lives in operand
registers while K1 is ready in LDS; K+2 replaces the consumed LDS slot during
the current MFMA work. Each operand is replaced after its final use. One
end-of-step full wait/barrier protects both reuse and producer publication.
The scale panel is refilled only after current scales have reached registers
on every wave.

| Experimental ID | Waves | Scale capacity in K128 groups | LDS bytes | Combined VGPR metadata | AGPR metadata |
|---:|---:|---:|---:|---:|---:|
| 9030 | 4 | 64 | 130688 | 374 | 202 |
| 9031 | 8 | 64 | 130688 | 232 | 0 |
| 9032 | 4 | 128 | 143104 | 387 | 207 |
| 9033 | 8 | 128 | 143104 | 236 | 0 |

All four compile with machine-instruction verification using the accepted
compiler flags, with no scratch or SGPR/VGPR spills. See [resources](resources.json),
[build manifest](build_manifest.json), and `experiments.disasm`.
`vgpr_count` includes the AGPR reservation; the two counts must not be added.

The four-wave configuration pins accumulator results and A loads; the
eight-wave configuration uses compiler register allocation. Pinning the
initial eight-wave operand/accumulator arrangement exceeded its available
register allocation and did not compile. The comparison therefore evaluates
two viable implementations of the same geometry, **not an isolated causal
measurement of wave count**. This first eight-wave version uses the common
interleaved loop; it does not implement CKTile's two-group ping-pong pipeline.

## CPU verification

The [layout checker](check_layout.py) executes the actual layout/adaptor
definitions on the CPU, adapting only host/device qualifiers in a local copy.
It checks every A/B fragment byte against its logical matrix coordinate,
unique output coverage, and bounded 64/128-row tails at output strides
256/6144/7168. [Results](layout_check.json) passed for both wave counts.

The check caught a real portability issue: the imported A register layout
assumes `T_M=2`. The new traits invert the cooperative producer layout for
both `T_M=2` and `T_M=4`. The old and new offsets are identical for four waves;
2816 chunk offsets differ for eight waves. B layout and output partitioning
remain shared. This establishes address mapping; it does not substitute for
GPU MFMA numerical or synchronization validation.

## Pending GPU checks

The benchmark tests 39 boundary shapes before the five targets, including all
M64 tail residues, K128/K256/K384 short loops, K values on both sides of
64/128-group scale boundaries, and multiple scale panels. Each new variant
must meet the original FP32 accumulation bounds and leave NaN output canaries
intact. A failing variant is excluded from all final summaries.

For each target the same run compares the new variants, legal original OPUS
9000/9010/9011, CKTile 11/27/28/29, and any saved final CK finalists. Reference
scales are exact FP32 decodes of the same native E8M0 inputs. Five rounds rotate
candidate order, with `run_perftest`, warmup=5, iters=51, and automatic argument
rotation. The entire guarded output allocation is rotated, preserving canaries.
Inputs, binaries and source hashes are checked before/after the run. This is a
bounded comparison with the established rivals, not a new exhaustive sweep.

Reproduction (use a fresh benchmark prefix and do not rebuild while the worker
is waiting/running):

```bash
python reports/opus_9030_wide_n_20260924/check_layout.py
python reports/opus_9030_wide_n_20260924/build.py
python reports/opus_9030_wide_n_20260924/audit.py
python reports/opus_9030_wide_n_20260924/bench.py --prefix fresh_r5 --rounds 5
```

Next decision: inspect numerical results and same-run medians, then retain a
useful configuration or revise this independent pipeline. A completed pilot
does not automatically install a production kernel or change any winner.
