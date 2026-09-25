Both K=1536 device translation units and the generated host translation unit
compiled successfully with the pinned amdgpu-pin-op-dst clang toolchain. The
full-unroll candidate has register spills; compilation success does not imply
that its performance is acceptable.

| Kernel ID | K-loop unroll | VGPR | SGPR | LDS bytes | Private bytes | VGPR spills | SGPR spills | Static MFMA | Static waits | Static barriers | Backward branches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9050 | 12 | 256 | 41 | 120600 | 76 | 22 | 0 | 288 | 47 | 13 | 0 |
| 9051 | 2 | 206 | 42 | 120600 | 0 | 0 | 0 | 96 | 16 | 5 | 1 |

Both candidates report 0 AGPRs, Wave64, and a maximum workgroup size of 512.
MFMA instructions are `v_mfma_scale_f32_16x16x128_f8f6f4`; wait counts refer to
textual `s_waitcnt` instructions in each assembly file.

The 9051 backward edge is `s_cbranch_scc0 .LBB0_7`. Its body contains 48 MFMA
instructions and 2 barriers and runs five times, leaving 48 MFMA instructions
and 3 barriers outside the loop. Its dynamic counts are therefore 288 MFMA
instructions per wave and 13 barriers, matching 9050. The loop counter starts
at -2, increments by 2, and leaves after the unsigned comparison against 7
becomes true. Static instruction counts must not be compared as dynamic work.

9050 contains 7 scratch-load instructions and 7 scratch-store instructions;
9051 contains none. This is a resource observation only. No timing or GPU
numerical result is produced by this offline check.

The generator emits all 14 optional bpreshuffle candidates. The 12 pre-existing
names and frozen metadata fields remain unchanged, and their newly added
`k_loop_unroll` field defaults to `None`. All 24 pre-existing implementation and
device files match their frozen SHA-256 hashes, as do all 19 protected source
files. The default compiled set and tuned configuration are unchanged.

Reproduce with:

```sh
python reports/opus_k1536_20260925/offline_compile.py
```

The script uses explicit `--offload-arch=gfx950`, disables runtime queries, and
installs an import guard against aiter, torch, triton, and cupy. It calls the
generator API directly and never loads or launches the generated output.
[offline_compile.json](../offline_compile.json) records complete compiler
commands, source/generated/assembly hashes, resources, and preservation checks.
The assembly and compiler logs are [9050.s](9050.s), [9050.log](9050.log),
[9051.s](9051.s), [9051.log](9051.log), and
[generated_host.log](generated_host.log).
