# Current Oct8 narrow ATT analysis

Shaderclocks; successfulissue=time+stall, duration includesstall. Wait queue members are not individualmemory-returntimestamps. Exclusive partitions cap decoder overlap; rawstataggregates preserved.

| Target / shape | Complete waves | MFMA each | Prologue median | Wave life median | Readlane each |
| --- | --- | --- | --- | --- | --- |
| 9 / 544×7168×16384 | 8 | 1024 | 6356.0 | 131804.0 | 6 |
| 10 / 576×7168×7168 | 8 | 448 | 6714.0 | 64506.0 | 6 |
| 11 / 1344×768×7168 | 4 | 224 | 5218.0 | 33644.0 | 0 |

All times are shader clocks from finite correlated CU0 waves. For accepted targets9/10/11, exact current module/CO/FUNC/metadata/descriptor, relative PC ISA, corrected clean claim, complete MFMA count and code/CSV/wave aggregates passed.

Target12 is excluded: Monitor contains a process other than the recorded owner. During epoch1791425408.647696..1791425419.0551922, owner hostPID2299436 and unowned hostPID(s)[2300083] appear in the same GPU monitor. The end record says contamination:false, but the strict owner check rejects the capture. Original evidence is retained and no timing or decoded-wave finding from this capture is used.

- Runtime9023 each of eight complete waves executes46 scalar lane saves but only6 lane restores: three remainder control and three output address restores. None of the captured ordinary-tile or aligned refill paths executes the static120/readlane branch body. Unroll1 is not justified by a steady restore bottleneck.
- Runtime9023 refills occur after MFMA ordinals248/504/760 (tile31/63/95 before final eight MFMA of that tile). The7→0 gaps containing producer wait/pack/publish/barrier are far wider than ordinary7→0 gaps. Full-vector SFA waits precede SFBissue and its wait; this is a concrete local serialization candidate.
- Accepted fixed9024 target11 contains224 MFMA per wave and zero lane operations, with no panel refills. Startup median is5218 shaderclocks, about15% of its33644-clock median wave life. Two wave timelines execute SFA vectors, one executes SFB; source preserves matching guards and byte/u16 layout.
- Both runtime9023 captures show two synchronized4wave publication cohorts in slots0/1. Slot1 life is longer in these captures; the exporter does not identify true WG coordinates and this does not establish whole-device dispatch fill.
- Fixed9024 252/264WG targets use the same fixed/group4 entry in the source/ELF review. Target12 is excluded for an observed unowned GPU process; these captures do not provide a clean252/264WG timing comparison or evidence of a256 source boundary.

Full queue sets, dynamic PCs, phase partitions, producer events and ordinary/refill/drain gap records are in [narrow_att_analysis.json](narrow_att_analysis.json). Original durations are retained;4clock overlapping decoder scalar/nop events are capped only for exclusive time partitions. Tail coordinates and EXEC masks are not exported, so no tail-WG or active-byte claim follows.

No candidate adoption; isolated issue-before-publish candidate requires exact baseline/ISA, guard/numerical and shared-address AB/BA Event.
