# MXFP8 B-preshuffle：kernel优化与完整295项调优

## 最新：295项全量调优完成

**295/295已完整重新tune：OPUS293胜，CKTile2胜，CK/ASM均0胜。** 全部295项重新做三轮完整候选扫描；其中58项接近或轮次胜负不一致，随后同GPU五轮确认。最终237项保留三轮结果、58项采用较新五轮结果。两批次各8个worker均passed，测量已结束。

相对最快有效外部对照，几何平均耗时下降 **13.01%**；相对同批旧OPUS下降 **4.34%**。最佳OPUS选型最终涉及 **30个ID**：5个原注册ID覆盖196项，25个实验ID覆盖99项。9000选中127项；9000及其共享helpers、注册和默认配置未修改。`experiment`仍属于OPUS，只表示独立实验库实现。

- [最终295项结果说明及30个ID次数表](display295_v1/README.md)
- [295项逐shape OPUS选型](display295_v1/usage/opus_choices.csv)
- [OPUS候选使用次数](display295_v1/usage/opus_candidate_usage.csv)
- [整体最快实现](display295_v1/overall_winners.csv)
- [完整比较](display295_v1/comparison.csv)、[统计与来源](coverage295_v1/summary.json)
- [最终精选实验候选配置](selected_experiments295.json)；完整扫描使用[25库109实验ID](experiments295.json)，并另行保留所有合法注册OPUS。

两个未胜shape均有五轮记录：`(1600,7168,384)`，OPUS14540为12.784568us，CKTile11为12.760023us，慢0.1924%；`(1536,7168,768)`，OPUS12641为17.179160us，CKTile30为17.168143us，慢0.0642%。不将之前更快的记录混入本次295结果。最终原87项为85/87领先，另外208项为208/208领先。

58项五轮确认中56项中位数领先、37项每轮均领先。所有入选OPUS均通过随测数值检查；全扫描2626个注册OPUS候选记录和5650个实验OPUS候选记录无拒绝。完整外部扫描CK/CKTile有效候选为5160/9535；ASM11370个候选因超原FP32误差界剔除。详细口径与原始批次入口见最终报告。

## 以下为原87项优化阶段历史记录

后面的v10/v11统计保留用于追溯，不能替代上面的295项重新调优结果。


用户要求以每个 shape 上数值正确的 CK / CKTile / ASM 中最快者为目标，使用表现最好的旧或新 OPUS kernel 继续覆盖下一组。本轮只进行 kernel 修改、编译、目标 shape 的随测数值检查和性能比较。

原87项优化阶段结果曾更新到 [v11 的87项快照](display_v11/README.md)：**OPUS 87胜，CK / CKTile / ASM各0胜**；全部87项的赢家均为OPUS实验kernel，全部有五轮记录。外部几何平均耗时下降4.64%，57项五轮均领先。下面保留v10结果说明；295项全量retune的数据流程另见 [RETUNE295.md](RETUNE295.md)。

## v10 历史结果

原87项全部完成测量与选型。按每个shape最新完整五轮批次，**86项的整体赢家是OPUS实验kernel，余下1项由CKTile获胜**；CK和ASM均无胜项。原始记录中的 `experiment` 表示OPUS的独立实验实现，算法归属仍为 `OPUS`，不是CK。相对最快有效外部候选的几何平均耗时下降 **4.63%**，相对同批旧正式OPUS下降 **13.00%**，单shape最大外部改善 **22.48%**。这些新OPUS实现在独立实验目录中，原注册kernel **9000未修改**。

其中 **56项五轮均领先**，另有30项中位数领先但未在每一轮都胜出，需按表中的实际差距理解。后续54项全部中位数领先。

| 组别 | 领先/总数 | 比外部耗时下降（几何平均） | 比旧OPUS耗时下降（几何平均） |
|---|---:|---:|---:|
| 短 K：K384/768/1024 | 24/25 | 11.18% | 21.74% |
| K1536，N16384 | 8/8 | 3.21% | 13.44% |
| K3072，N7168 | 12/12 | 1.31% | 10.93% |
| K7168，N6144/7168 | 26/26 | 1.18% | 7.98% |
| K16384，N7168 | 10/10 | 3.16% | 8.78% |
| K7168，N768 | 3/3 | 2.38% | 7.85% |
| K7168，N2048 | 3/3 | 1.32% | 3.83% |

唯一未胜项是 **(1536,7168,384)**：OPUS **15040** 为 **12.665778 µs**，**cktile_11_split0** 为 **12.660383 µs**，OPUS慢 **0.0426%**。该项接近持平，仍如实列为未胜；不把此前较小的单次/单轮耗时替换进本批结果。

可直接查阅：

- [算法归属明确的87项展示](display_v10/comparison.csv)：`algorithm=OPUS`、`implementation=experimental`，另列每项的 `overall_winner_algorithm`；[整体赢家选型](display_v10/overall_winners.csv)直接标出OPUS 86项、CKTile 1项，[展示汇总](display_v10/summary.json)保存分类说明和原始来源哈希。这些是派生文件，未改动已有测量记录。
- [逐shape最佳OPUS选型](coverage87/best_opus_selection.csv)：kernel ID、实验库、设备、同批对照、轮次和来源。
- [87项逐shape比较](coverage87/comparison.csv)与[完整来源和汇总](coverage87/summary.json)。
- [各shape最快OPUS候选的配置](selected_experiments.json)：保留每个shape最佳OPUS；最后未胜项的选择状态仍见CSV，不表示已经超过外部。
- [剩余项](coverage87/remaining.csv)。

本轮所有参与最终计时的OPUS候选均通过目标shape数值检查。删除重复同步、tile切换与output融合通过目标测量选优；完整tile分支、2CTA XOR及9000派生候选未带来明确的外部领先，其原始证据均保留。

## 比较口径

- `full87_r3` 已完整枚举原 87 项的 CK / CKTile / ASM 候选，全部 8 个 worker 成功完成。该三轮初筛为 OPUS 72 胜 / 15 负。
- 后续五轮批次保留所有合法注册 OPUS、配置中的实验候选，以及同一 GPU 全扫描中距最快有效外部候选不超过 5% 的对手。实际最快有效外部候选均为 CKTile。
- ASM 的 3294 个候选均未通过原 FP32 数值误差界，保护区完整，未进入计时。它们不能作为数值正确的性能胜项；详细记录见 [全扫描分析](full87_analysis.md)。
- 保留原 `run_perftest`：warmup=5、iters=51、自动输入轮换、候选顺序轮换。每个 shape 的各候选在同一实际 GPU 上测量。
- 保留原 FP32 逐元素误差界、BF16 区间端点、NaN 输出预填及上下 64 行保护区。没有增加边界、接口或单元测试。
- 汇总按每个 shape 最新的完整五轮批次整行替换旧结果；不跨批次选更小耗时。五轮中位数领先不等于每一轮都领先，逐 shape 表保留 `opus_faster_rounds`。

## 已实施的 kernel 改动

- 将成功的 BF16 LDS 输出重排扩展到通用 S64/S128 和 K3072/K7168/K16384 的固定 K U2 流水，继续覆盖原前 33 项之外的 54 项。
- 添加 128×256、64/96×256、64/128/160/192×128 tile；保留每个 shape 上表现更好的几何。128×128 和160×128解决了不同 M 区间的并行工作量台阶。
- 在最后一次 advance 已等待全部矩阵/scale LDS 读取并同步的实现中，删除 final MFMA 与输出 staging 之间重复的 wait/barrier；输出 LDS 写入后、跨 wave 读取前的同步仍保留。
- 比较输出 pitch、cache policy、CTA 顺序和向量搬运宽度，并保留旧候选参与同批比较。参数组合仅对相关剩余目标继续测量。
- 完整tile专用路径、交错final MFMA/BF16 staging、旧9000独立副本等也在剩余shape上比较。没有将无收益的候选替换为生产默认；9000独立副本已从后续候选配置移除。
- 新实现均位于 `reports/opus_resume_20260926/` 的独立实验目录；原 9000、生产注册和默认调度未修改。

所有批次的 `launch.json`、各 GPU `run/check/reject/raw/summary/comparison` 文件、配置和构建 manifest 均保留。原87项成员没有删减；该历史阶段未重测295项。完整295重新调优现已完成，结果见本文顶部。
