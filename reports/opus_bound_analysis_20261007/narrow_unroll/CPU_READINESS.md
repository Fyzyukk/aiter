# 9023 / 9024 Runtime narrow unroll 实验

2026-10-07。CPU构建和机器码审查已完成；随后完成四个当前9023赢家和两个9024 runtime对照的GPU数值及5轮共享池Event验证，未证明稳定收益，本轮不采用全局runtime展开改动。生产源码未修改。结果见 [逐轮分析](../results/narrow_runtime_window_analysis.json) 和 [Compute文档9.2.1](../COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md)。

## 单项改动

9023 runtime main loop的`unroll_count(NUM_STAGES=3)`改为1；9024 runtime的unroll4改为1。模板条件保留`FIXED_K`路径原来的3/4展开策略。ring slots、scale panel、prefetch距离、wait/barrier、输出、cache及grid swizzle均沿用原实现。

固定K7168内部分支的机器码、完整metadata及归一化descriptor在baseline/candidate中完全一致。两个私有baseline的全部四个device variants与本轮正式JIT对应variant具有相同指令字节、metadata和归一化descriptor。

## CPU结果

| Runtime | VGPR | SGPR | SGPR spill slots | 静态writelane/readlane | ISA bytes | 全函数静态MFMA |
|---|---:|---:|---:|---:|---:|---:|
| 9023 baseline→candidate | 232→121 | 106→106 | 46→21 | 46/166→21/21 | 25976→10928 | 40→16 |
| 9024 baseline→candidate | 152→87 | 106→106 | 44→22 | 44/132→22/22 | 21736→10776 | 24→8 |

LDS分别78112B、69664B不变；AGPR、private bytes、VGPR spill均0，没有scratch load/store。**这里SGPR spill通过VGPR lane保存和取回，不是HBM/scratch内存访问。** 即使无scratch，它仍增加lane move、依赖和寄存器占用。

编译用指定LLVM、gfx950、与compute_prologue相同冻结flags，补充本机ROCm20 resource目录并启用`-verify-machineinstrs`。两个构建通过，命令/哈希在`build_manifest.json`。

## 动态频率线索

`isa_comparison.json`列出了每条lane move的PC和每条向后分支覆盖区间。runtime9023原主循环三个K128展开窗口共24MFMA、120条静态readlane，另有单K128余数循环8MFMA/40readlane；candidate单K128循环8MFMA/16readlane。主循环区间里的write为0，主要spill槽在进入循环前安装。

runtime9024原四K128展开窗口共16MFMA、88条readlane，另有单K128余数循环4MFMA/44readlane；candidate单K128循环4MFMA/22readlane。其外层回跳还覆盖scale panel生产和44→22条writelane安装。不同M尾部、scale panel换代、loop余数会选择不同控制流，完整静态数量不能直接当作动态执行次数。

减少展开缩短live range并压缩代码，同时会增加loop控制频率，可能改变MFMA/load重叠。它是否属于计算发射、局部供数、latency或dispatch改善，需用完整计时与counter确认；CPU结果不证明加速。

## GPU验证计划

- `timing_plan.json`：9023/9024，M1536/N768，K384/768/1536/3072/16384；另有K7168不变分支对照，共12目标。
- `correctness_plan.json`：short K、展开余数、scale panel边界、M尾部、swizzle范围及固定/非固定K7168分支，共52目标；signed输入和多次重复由统一runner执行。
- 采用相同`launch` C ABI、标准B `(16,16)`预排、SFA原生E8M0 `[K/128,M]`、SFB `[N/128,K/128]`、BF16输出、batch1；parent dispatch条件与正式一致。
- GPU绑定、物理设备锁和空闲判定由root控制。真实shape六目标已完成8次原buffer检查及51池逐buffer数值/重复/guard、5轮Event；未完成52机制目标或718runtime支持域回归。六项Event吞吐变化依次−0.390%、+0.291%、+0.070%、−1.161%、+0.109%、−1.311%，9024最大K对照5/5轮退步，因此停止该候选的扩大计时并保留基线。

CPU复现：`python3 reports/opus_bound_analysis_20261007/narrow_unroll/build.py`，然后`python3 reports/opus_bound_analysis_20261007/narrow_unroll/audit.py`。
