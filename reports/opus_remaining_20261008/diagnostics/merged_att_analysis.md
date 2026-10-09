# Current Oct8 9020 / 9022 / 9030 ATT

gfx9 event [time,type,stall,duration,code_line]: successful issue = time+stall; duration includes stall. All event times are shader clocks. Wait dependencies identify queue members, not individual memory-return latency. Wave/event shares are not full-kernel wall-time shares.

Each accepted capture passed module/CO/FUNC/metadata/descriptor, current PC mapping, clean owner claim, complete wave MFMA, and code/stat/wave aggregate checks.

| Target | Parent / shape | Waves / WGs | First MFMA median | Wave duration median | Last MFMA to end median |
|---|---|---:|---:|---:|---:|
| 0 | 9020 / 1536×7168×384 | 8 / 1 | 3492.0 | 13876.0 | 3948.0 |
| 1 | 9020 / 1536×7168×16384 | 8 / 1 | 9162.0 | 278388.0 | 3670.0 |
| 2 | 9020 / 8192×768×7168 | 16 / 2 | 7130.0 | 81576.0 | 1452.0 |
| 3 | 9020 / 1344×16384×1536 | 16 / 2 | 5764.0 | 34986.0 | 4494.0 |
| 4 | 9020 / 1728×7168×3072 | 8 / 1 | 5680.0 | 60802.0 | 5330.0 |
| 5 | 9020 / 6144×2048×7168 | 8 / 1 | 6926.0 | 128332.0 | 4462.0 |
| 6 | 9020 / 1728×7168×768 | 8 / 1 | 5188.0 | 22338.0 | 4924.0 |
| 7 | 9022 / 800×7168×384 | 4 / 1 | 5016.0 | 10578.0 | 1868.0 |
| 8 | 9022 / 800×7168×16384 | 4 / 1 | 7022.0 | 213372.0 | 1698.0 |
| 13 | 9030 / 65536×16384×1536 | 688 / 86 | 8034.0 | 36834.0 | 4484.0 |

All numbers are shader clocks from correlated captured waves. Output begins at the first BF16 conversion and overlaps final MFMAs where the epilogue interleaves output. Full event, dependency, phase, and tile records are in [merged_att_analysis.json](merged_att_analysis.json).

Status: passed; pending 0; errors 0.
