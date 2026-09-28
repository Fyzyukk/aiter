# 完整295项OPUS重新调优结果

**295/295全部完成：OPUS 293胜，CKTile 2胜，CK与ASM均0胜。** 全部295项重新做三轮全候选调优，再对差距±3%或轮次不一致的58项做同GPU五轮确认。最终237项保留三轮结果、58项使用较新的完整五轮结果。

相对每个shape最快有效外部对照，几何平均耗时下降 **13.01%**（1.1496×）；相对同批旧OPUS下降 **4.34%**。99项选用更快的实验OPUS，其余196项选用原注册OPUS。原87项目标本次为85胜/2负，其余208项全部领先；历史v11的87/87只代表当时批次，未混入本次295结果。

**9000及其共享helpers、注册、默认配置未修改。** 本次最终有127个shape继续选中9000。所有标为`experiment`的候选仍属于OPUS，`experimental`只表示独立实验库实现。

## 结果入口

- [295项逐shape最佳OPUS选择](usage/opus_choices.csv)：M/N/K、实现类型、库、kernel ID、耗时、外部对照、实际轮次、来源和二进制哈希。
- [30个入选OPUS候选及使用次数](usage/opus_candidate_usage.csv)：按库和ID统计选中shape数、整体胜出数与3/5轮分布。
- [295项整体最快实现](overall_winners.csv)：明确标出OPUS或CKTile整体赢家，同时保留最佳OPUS。
- [完整逐shape比较](comparison.csv)、[汇总与全部来源](../coverage295_v1/summary.json)、[分组结果](../coverage295_v1/cohort_summary.csv)。
- [选中实验候选配置](../selected_experiments295.json)是调优后的精选池；完整调优输入仍为[25库109候选配置](../experiments295.json)。

## 最终入选的OPUS候选

5个原注册ID覆盖196项；25个实验ID覆盖99项，合计30个ID。这里“选中”表示该shape中最快的有效OPUS，包含下表列出的两项外部仍领先的shape。

| OPUS ID | 实现 / 库 | 选中shape数 | 整体胜出数 |
|---:|---|---:|---:|
| 9000 | registered_opus | 127 | 127 |
| 9020 | registered_opus | 29 | 29 |
| 9011 | registered_opus | 23 | 23 |
| 12971 | long_epilogue_sync | 17 | 17 |
| 9010 | registered_opus | 10 | 10 |
| 9672 | fixed_long_lds | 9 | 9 |
| 9670 | fixed_long_lds | 8 | 8 |
| 9671 | fixed_long_lds | 8 | 8 |
| 12071 | long_grid | 7 | 7 |
| 9012 | registered_opus | 7 | 7 |
| 12340 | shortk_n128 | 7 | 7 |
| 13163 | long_epilogue_sync | 5 | 5 |
| 12840 | shortk_m160n128 | 5 | 5 |
| 12843 | shortk_m160n128 | 5 | 5 |
| 12841 | shortk_m160n128 | 4 | 4 |
| 12341 | shortk_n128 | 3 | 3 |
| 9663 | generic_lds | 2 | 2 |
| 11971 | long_grid | 2 | 2 |
| 12641 | shortk_epilogue_sync | 2 | 1 |
| 13440 | shortk_sync_tuned | 2 | 2 |
| 13441 | shortk_sync_tuned | 2 | 2 |
| 13540 | shortk_sync_tuned | 2 | 2 |
| 13541 | shortk_sync_tuned | 2 | 2 |
| 10272 | fixed_long_tuned | 1 | 1 |
| 9661 | generic_lds | 1 | 1 |
| 9651 | k1536_tuned | 1 | 1 |
| 14540 | shortk_fused_output | 1 | 0 |
| 14840 | shortk_fused_tuned | 1 | 1 |
| 12342 | shortk_n128 | 1 | 1 |
| 15940 | shortk_n224 | 1 | 1 |

注册几何：9000/9020为256×256、4wave；9010为128×128、4wave；9011为64×128、4wave；9012为64×64、4wave。实验族主要为192×256、8wave的BF16 LDS输出流水，128×128/160×128、4wave的短K实现，以及15940的192×224、8wave实现。具体候选名及元数据见使用次数CSV。

## 两项仍略慢于外部的shape

| (M,N,K) | 最佳OPUS ID | OPUS µs | 外部候选 | 外部 µs | OPUS慢 | 五轮胜出数 |
|---|---:|---:|---|---:|---:|---:|
| (1600,7168,384) | 14540 | 12.784568 | cktile_11_split0 | 12.760023 | 0.1924% | 0/5 |
| (1536,7168,768) | 12641 | 17.179160 | cktile_30_split0 | 17.168143 | 0.0642% | 2/5 |

保留这两项为未胜，不用此前批次中更小的耗时替换本次结果。58项五轮确认中56项中位数领先、37项每轮均领先；19项中位数领先但未每轮领先。全体295项中274项在各自全部记录轮次中领先。

## 分组结果

| 范围 | OPUS领先 / 总数 | 比外部耗时下降（几何平均） | 比旧OPUS耗时下降（几何平均） |
|---|---:|---:|---:|
| short_k_3_6_8_steps | 23/25 | 11.12% | 22.39% |
| k1536_wide_n | 8/8 | 3.27% | 13.37% |
| k3072_n7168 | 12/12 | 2.65% | 11.88% |
| k7168_wide_n | 26/26 | 1.63% | 8.27% |
| k16384_scale_panel | 10/10 | 3.14% | 9.02% |
| narrow_n768 | 3/3 | 5.14% | 11.36% |
| narrow_n2048 | 3/3 | 4.90% | 7.74% |
| other_supported_model_shapes | 208/208 | 16.11% | 0.15% |

## 测量范围和记录

- 精确复用原295个受支持shape，SHA256：`20b92df1957f00766b309153cafd137d01c40aeb941c6d61c2b62c052d3ff9b5`。原10项寻址排除保持在此目录之外。
- [full295_r3](../full295_r3/launch.json)：8个worker全部passed，完整枚举合法注册OPUS、109个实验ID中对该shape合法的候选，以及CK/CKTile/ASM；三轮初筛293/295领先。
- [close295_r5](../close295_r5/launch.json)：8个worker全部passed，仅对筛出的58项同GPU五轮确认；保留全部合法OPUS及全扫描中距最快有效外部候选5%内的对手。
- 全扫描OPUS：2626个注册候选记录、5650个实验候选记录全部通过目标shape数值检查及3轮计时后检查。这里候选记录按shape×kernel计数。
- CK 5160个、CKTile 9535个有效候选；150个CK、200个CKTile因参数不支持被拒绝。ASM的11370个候选均超出原FP32误差界而被拒绝，不能计作数值正确的性能结果。每项仍有有效外部对照；最终45项对照为CK，250项为CKTile。
- 五轮确认无候选拒绝，全部随测数值检查通过。使用原warmup=5、iters=51、自动输入轮换和候选顺序轮换；保留FP32误差界、BF16端点、NaN预填和既有保护区。没有新增边界、接口或单元测试。
- 最终汇总按shape整行采用较新完整批次，不跨批选更小耗时。各批次源码、配置和二进制哈希已由原测量流程核对；计时期间未编译或修改候选。
- 本次测量已结束，无后台GPU任务待收尾。kernel源码与改进过程见[优化记录](../RESULTS.md)。
