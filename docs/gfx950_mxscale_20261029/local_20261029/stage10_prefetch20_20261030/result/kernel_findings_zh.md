第十轮已完成，326领先/95落后。20个窄Nshape做新的六轮，401行历史数据保持，1516条测试正确、pending0。新增1项M14336/N1536/K7168为原kid48078复测越线，无新CO整合；第九轮2.0786%的实际新kernel收益保留。

提前B预取相对同几何、A stages、B深度、streaming的对照有44组成对六轮数据，只有2组达到0.5%和4/6更快门槛（M1152/N768：0.7733%、4/6；M14336/N768：0.6175%、5/6），但都未超过本shape当前公开最优，不能记为新的公开kernel收益。四组matched CO的ISA不同，改动确实到达机器码，原硬件wait/barrier及地址/descriptor边界未变。完整明细在matched_schedule_comparisons.csv、matched_schedule_metrics.json，co_experiment/isa保留反汇编和diff。

下一轮转向76个宽N剩余项：原先pin_AGPR主循环只有unroll2/4，扩展到unroll1/8与已有schedule policy0/2组合；保持原内存流水、正确性和大M地址限制。
