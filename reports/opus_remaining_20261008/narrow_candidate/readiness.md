# Narrow candidate CPU readiness

Every candidate full/full SFA path issues low/high16B loads and the guarded SFB byte before its first VMEM completion wait. The baseline waits and publishes A before B issue. All remaining forward execz edges through the matching final wait/barrier are enumerated. The9024 fixed LGKM-only wait before SFB issue waits for SMEM address inputs and does not drain VMEM.

| Variant | Static publication copies | Baseline VGPR / SGPR spill | Candidate VGPR / SGPR spill | LDS | ISA bytes |
| --- | --- | --- | --- | --- | --- |
| 9023_runtime | 5 | 232 / 46 | 251 / 48 | 78112 | 25976→26328 |
| 9024_fixed | 1 | 110 / 0 | 110 / 0 | 71744 | 7276→7408 |

All four private baseline kernels match the current Oct8 official instructions, metadata and normalized descriptors. Candidate9023 fixed and9024 runtime remain identical. Private/AGPR/VGPR spill are0 for all variants. SGPR spill counts describe lane storage, not HBM traffic.

The audit reconstructs every FUNC byte from the disassembly and checks it against the audited CO. It computes branch targets from machine words, takes the full-vector helper edges and enumerates every remaining forward branch through each publication barrier. Complete path decisions and PC traces are in [isa_order_audit.json](isa_order_audit.json).

Baseline library: [baseline/experiments.so](baseline/experiments.so). Candidate library: [candidate/experiments.so](candidate/experiments.so). Both retain the experiment launcher ABI.

- This is an ISA/CFG order check, not a hardware latency or performance measurement.
- Conditional partial/default helper paths retain their byte/default/control code and VMEM waits; no issue-overlap claim applies to these paths or to mixed partial/full EXEC.
- M544 last WG has invalid-high chunks. Its default path is excluded from the full/full precondition and must be covered by the numerical/guard GPU screen.
- Resources increased for9023 runtime; no occupancy or throughput inference is made.
- Numerical/guard and shared-address AB/BA Event screens remain required before adoption.
