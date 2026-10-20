# gfx950 small MXFP8 bpreshuffle GEMM results, 2026-10-20

The final frozen replay covers all **451 small shapes**. OPUS beats the
same-batch best of BMM and PR6603 on **404/451**, with a geometric mean speedup
of **1.066983×** across all 451 shapes. There are 47 slower or equal results.
The default table adopts **400 OPUS rows** that both win on median latency
and win at least two of three paired rounds. The remaining 51 rows select
the faster measured external backend.

Compared with the previously saved best BMM/PR6603 latencies, the final OPUS
results win on **360/451**. This historical comparison is distinct from the
404 same-batch wins; timing drift changes the count. External known-best
configurations were remeasured, including historical and more recent table
choices. The entire external candidate spaces were not exhaustively searched
again.

## Measurement and implementation

All 451 OPUS shapes underwent a finite seeded search, followed by a 125-shape
refinement and a uniform frozen replay. The final replay uses three rounds,
five warmup calls, and 101 iterations per measured configuration. Calls pass
tensor arguments to `run_perftest` so its automatic input rotation remains
active. Timings are GPU operation durations for ordinary public `opus_gemm`
and external calls; they are not end-to-end Python wall-clock latency. No
graph performance timing is used.

The production OPUS path automatically reuses fused split-K workspace on the
current stream. Its eight-entry cache holds at most 32 MiB per entry. Each
kernel overwrites its partials and resets its trailing counters, so only the
counters need initialization when the buffer is allocated. The warmup
initializes this cache before measurement; no workspace is manually prepared
by this benchmark. Cold calls include allocation and counter initialization.
Graph capture allocates graph-owned scratch and initializes its counters
inside capture; larger allocations bypass the cache.

Native gfx950 pipelines, traits, and codegen follow the remote branch layout.
New entries retain the `950` ID prefix. Twenty-eight additional composable
geometry/policy configurations were retained after screening and repeated
validation. They share the existing compute body and use runtime M/N with
generic shape constraints. New native families and seven COs are included;
the prior remote kernel records, COs, and 785 tuning rows outside the small
scope remain unchanged. The remote base is
`72b24a7ec5c4d403948a42317a60193b4ebdf618`.

## Validation

| Check | Result |
|---|---|
| Final small replay | 5,412 full-output checks, zero errors |
| Native workspace/Tensor ABI tests | 34 passed |
| CPU interface tests | 10 passed |
| General production entry | 36 checks, zero errors |
| Final kernel resource metadata | 147 distinct configs, no scratch or VGPR/SGPR spills |
| Seven new CO rebuilds | 7 ready, 0 rejected |
| gfx942 shared host ABI | Generated host launchers and shared dispatch compile |
| All 294 retained large choices | 1,176 output checks, zero errors |
| Paired large OPUS follow-up | 126 checks, zero errors |

The large regression replays the remote's 294 default choices. Its geometric
mean duration is 1.01295× the saved historical duration, with 26 entries more
than 5% slower. The 21 OPUS entries in that group were compared with the
remote build on the same GPU using identical descriptor-ABI calls, input
rotation and three alternating rounds. Their geometric mean duration ratio
is 0.99688×, ranging from 0.97220× to 1.01038×. No substantial OPUS regression
was found in those paired checks. The five external entries use unchanged
source and configurations; their historical timing drift is recorded.

Production entry checks include byte-view and typed E8M0 scales, allocated
and caller-owned output, new derived configs, automatic scratch, a CO, short
K, retained large choices, and arbitrary M. A padded lookup whose config
does not support the original M correctly takes the external fallback.

## Result files and reproduction

- `final_comparison451.csv`: every final OPUS/external latency, historical
  comparison, per-round values and default adoption decision.
- `final_trials.jsonl`: all raw frozen replay observations, one shape per line.
- `production451.csv`: the 451 updated production rows.
- `summary.json`, `resources_final.csv`, `build_fingerprint.json`: aggregate
  counts, compiled kernel resources and published source hashes.
  `measured_build_fingerprint.json` preserves the measured hashes; only
  surplus trailing newlines in two traits headers were removed during final
  cleanup, with no change to kernel code.
- `large_regression294.csv`, `large_pair.json`, `large_summary.json`: retained
  large-choice regression and paired follow-up results.
- XML and JSON validation files preserve the checks listed above.

`environment.json` records this machine's paths and settings. The helper
`report_environment.py` relocates checkout-owned paths and accepts compiler
and CK overrides from the environment. Initialize the repository's CK
submodule or supply `CK_DIR`, and supply a suitable `HIP_CLANG_PATH` and
`OPUS_HIP_CLANG_PATH` if the recorded compiler is unavailable. The provided
CO images use the existing remote loader convention.

This machine assigned ROCR GPUs 4–7 to the task. `device_occupancy.py` records
their verified ROCR/DRM/PCI mapping and checks occupancy. On a different
machine, update that mapping and the worker device assignment before running.
On this machine, rebuild and replay with:

```bash
python reports/opus_small400_20261020/build.py
python reports/opus_small400_20261020/tune.py \
  --plan final_plan.json --output replay451 --mode confirm
python reports/opus_small400_20261020/production_smoke.py
```

`tune_plan.json` records the first seeded search, and `refine_plan.json`
records the follow-up candidate set; both can be used with `--mode search`.
`final_plan.json` freezes the final candidates and references. External
backends and their compiler dependencies must be available to reproduce the
comparison. Runtime build artifacts and exploratory results stay outside
the commit.
