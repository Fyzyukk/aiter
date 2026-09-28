The saved 9010 kernel has spills, but no scratch load/store inside its hot loop or its periodic scale-refill path. All 17 static scratch sites are outside every control-flow cycle, so none executes repeatedly with each K tile. This was a read-only analysis of the saved object; no compilation or GPU execution occurred.

| Metadata field | 9000 | 9010 |
|---|---:|---:|
| `.vgpr_count` | 477 | 512 |
| `.agpr_count` | 221 | 256 |
| `.sgpr_count` | 72 | 76 |
| `.vgpr_spill_count` | 0 | 14 |
| `.sgpr_spill_count` | 0 | 0 |
| `.private_segment_fixed_size` | 0 | 44 bytes |
| `.group_segment_fixed_size` | 152064 | 152064 bytes |

Both use Wave64 and a 256-thread workgroup. The metadata spill count is a compiler statistic, not a count of scratch operations per loop iteration.

| ISA addresses | Static sites | Role |
|---|---:|---|
| `0x2460–0x2488` | 6 stores | Save state before the main-loop register allocation region |
| `0x2c2c–0x2c44` | 4 loads | Restore state into the loop's register assignments, then branch to `0x32e4` |
| `0x47dc` | 1 store | Save the four scale words on main-loop exit |
| `0x54d4–0x54fc` | 6 loads | Restore state before the epilogue and its optional scale refill |

Control-flow evidence: the repeated region is `0x2c60–0x3ecc` (578 instruction nodes; zero scratch instructions). The main-loop exit branch at `0x32e0` targets `0x47dc`. The ordinary loop backedge at `0x3950` targets `0x2c68`; the scale-refill path finishes at `0x3ecc` and targets `0x2c60`. Epilogue scratch restoration precedes its refill test at `0x5518–0x551c`, so those restores are not conditional on taking a refill. At `0x2458`, `s_cmpk_lt_u32 s38, 0x180` tests K < 384; its branch bypasses the scratch-saving path for K=128/256.

The slots primarily carry scale-panel data and derived addresses across the high-pressure region:

- Offsets 0–15: four SFA raw scale words, initially `v[16:19]`, used as `v[28:31]` inside the loop and restored to `v[16:19]` for the epilogue. The loop refill updates these words using a global load and DPP transpose, without scratch.
- Offsets 16, 20, 24–31: SFA pass-1/pass-2/pass-3 address calculations (stride multipliers 16, 32, 48), used again at scale refill.
- Offset 32: B-operand LDS address base, reused by the epilogue's `ds_read_b128` sequence at `0x5a00`.
- Offset 36: the SFA LDS scale-read lane offset, reused by the epilogue's `ds_read_b64` at `0x57c8`.

Representative ISA:

```text
0x2488  scratch_store_dwordx4 off, v[16:19], off
0x2c2c  scratch_load_dwordx4 v[28:31], off, off
0x2c4c  s_branch 421                 // -> 0x32e4
0x32e0  s_cbranch_scc0 1342          // -> 0x47dc, loop exit
0x47dc  scratch_store_dwordx4 off, v[28:31], off
0x54d4  scratch_load_dwordx4 v[16:19], off, off
0x5518  s_and_b32 s0, s28, 63
0x551c  s_cbranch_scc1 155           // -> 0x578c, skip epilogue refill
```

The relevant optimization target is state lifetime and register reassignment across loop boundaries, especially the retained SFA panel and address values. The saved ISA does not show a per-K scratch bandwidth cost, so removal alone does not establish a performance improvement.
