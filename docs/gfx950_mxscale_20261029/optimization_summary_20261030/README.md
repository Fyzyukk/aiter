# Opus MXFP8 NT 优化结果整理

截至第17轮：**421项中337领先、84落后**。按用户要求停止继续优化，保留当前可用最优配置与已完成的实验。第16、17轮均未产生满足稳定标准的新CO；两轮共9694条测量全部正确，controller均退出0。

从第四步261/160推进到337/84，新增76项领先：53项在当轮原最优仍落后时由更快候选越线，23项当轮原最优自身复测也越线。53项包含既有候选重新选择，不能全部归因于新kernel源码。

当前421行来自各轮已确认数据合并。最后一轮只对54项pin_agpr落后做新六轮，其余367行逐字段保留；全部421项的最后复测来源在[shape_evidence421.csv](shape_evidence421.csv)。FlyDSL使用固定CSV，按gfx/CU/M/N/K/scale-block六字段匹配。

## 剩余落后

落后定义：`(Opus_us / 固定FlyDSL_us - 1) * 100%`。84项中位落后3.52%，最大13.25%。以下每个shape只按当前最优pipeline计一次。

| Pipeline | 项数 | 中位落后 | 最大落后 |
|---|---:|---:|---:|
| pin_agpr | 54 | 3.55% | 7.59% |
| pingpong_lds | 20 | 2.17% | 9.34% |
| prefetch_b_vgpr_overlap | 4 | 8.71% | 13.25% |
| prefetch_b_vgpr | 3 | 2.51% | 2.68% |
| wave_interleave | 3 | 9.08% | 10.03% |

| 落后区间 | 项数 |
|---|---:|
| 0-1% | 6 |
| 1-3% | 29 |
| 3-5% | 19 |
| 5-10% | 28 |
| >=10% | 2 |

完整降序明细：[remaining_losses84.csv](remaining_losses84.csv)。按pipeline拆分的CSV与[remaining_by_pipeline_N_K.csv](remaining_by_pipeline_N_K.csv)同时保存。最近两轮定向处理最大类别pin_agpr，仍有54项；这表示当前尝试未见稳定收益，不代表已证明所有方向都无法提升。

## 已保留的实现

- [best_opus421.csv](best_opus421.csv)：当前可用的421项最优配置；领先与落后均保留。
- [integrated_co_inventory.csv](integrated_co_inventory.csv)：第5至17轮整合的27个CO，含当前选中shape数、traits、pipeline、shape范围、镜像路径及SHA256。历史CO继续保留。
- [integrated_kernel_gains.csv](integrated_kernel_gains.csv)：各轮整合新CO的逐shape收益与复测证据；同一shape可在多轮出现。
- [co_kernels_gfx950_snapshot.json](co_kernels_gfx950_snapshot.json)：本次停止时的公共CO清单快照。公共registry为48261，JIT编号起点固定780，旧配置身份保留。

近期有效改动包括：紧凑B预取流水；pin主循环展开8；SFA每次补载使用局部临时值；scale refill移至按缓存边界分段的外层；个别shape使用展开2。第13轮SFA改动带来两个shape收益，第14轮分段循环带来六个，第15轮展开2带来两个；对应源码与CO见清单，各轮之间存在shape重叠。

近期未采用的实验：第12轮缩小scale缓存无新最优CO，panel16曾出现大量spill；第16轮提前矩阵请求及缩短尾部临时值未达到稳定收益；第17轮暴露偏移不变量减少了部分寄存器，但伴随spill，同样无稳定新最优。完整实验、初筛明显慢候选、未通过编译的诊断都保存在原campaign目录；未作为当前最优整合。

## 轮次与归因

新增领先的两类分别为“更快候选越线 / 原control复测越线”。只列各轮新增数。

| 轮次 | 方向 | 领先/落后 | 新CO数 | 新领先归因 |
|---:|---|---:|---:|---:|
| 5 | 候选扩展与复测 | 304/117 | 6 | 39/4 |
| 6 | 扩候选、kernel参数与调度 | 318/103 | 6 | 8/6 |
| 7 | B-VGPR及pin主循环 | 321/100 | 4 | 2/1 |
| 8 | 长K候选 | 323/98 | 0 | 0/2 |
| 9 | 窄N紧凑B流水 | 325/96 | 1 | 1/1 |
| 10 | 提前B预取 | 326/95 | 0 | 0/1 |
| 11 | 主循环展开 | 332/89 | 3 | 2/4 |
| 12 | scale缓存缩小 | 333/88 | 0 | 0/1 |
| 13 | SFA局部临时值 | 336/85 | 1 | 1/2 |
| 14 | 按scale边界分段主循环 | 336/85 | 4 | 0/0 |
| 15 | pin_agpr展开2/4 | 337/84 | 2 | 0/1 |
| 16 | 矩阵请求提前、尾部临时值缩短 | 337/84 | 0 | 0/0 |
| 17 | 偏移不变量与标量传播 | 337/84 | 0 | 0/0 |

原始目录、复测数量、测量数及新CO编号：[round_history.csv](round_history.csv)。最新完成凭据：[latest_completion_receipt.json](latest_completion_receipt.json)。

## 复现与范围

在本workspace中使用已验证的公共模块和custom pin-AGPR LLVM wrapper：

```bash
source docs/gfx950_mxscale_20261029/optimization_summary_20261030/run_env.sh
```

环境为gfx950/256 CU、Opus NT；逻辑卡0/1/2/3经PCI核对对应物理0/3/6/5。该环境指向第17轮公开缓存，并设置`OPUS_COMPILE_FAMILY_ONLY=a8w8_blockscale_bpreshuffle`。后续本任务构建已限制为MXFP8 family，实验和公开生成清单均没有a16w16路由；其他任务未设置该变量时保持原构建范围。

A输入侧2GiB guard保持。输出使用已验证的tile寻址路径；421项中最大A输入为1GiB、最大C输出为8GiB，实际覆盖了超过4GiB的输出。此整理没有扩大A输入支持范围，也没有改动已有buffer描述符、硬件wait或线程同步。

[summary.json](summary.json)记录校验后的结果、来源哈希及公共模块哈希。整理过程仅作CPU读取、校验与文件汇总，没有新增GPU测量。当前优化进程均已退出。
