# 28候选全量 retune：745 shapes，8卡并行

全部 **28 个 OPUS 候选**已经纳入：保留9000、9010，加入9001、9011，并覆盖9020–9024、9030及9040+等所有公开候选。全745个gfx950/256CU shape完成，**13,027/13,027个合法 OPUS配置均已执行，errRatio=0**；全部后端合计79,504条原始记录。最优后端是OPUS720项、ASM24项、CK1项。

新增9001/9011的全量筛选增量很小：相对**本轮同shape、同GPU测量中原26候选的最优值**，完整28候选几何平均加速 **0.036518%**，各shape单次时间之和下降 **0.039968%**。这是单轮筛选的最优值统计，包含选型噪声；多数新赢家在五轮复测时会翻转，不能宣称已得到普遍稳定收益。

## 同批原26 vs 完整28

原26候选也使用当前已接受的9021等优化，所以这组比较只衡量**增加9001/9011**的效果，不是对Sep30冻结二进制的A/B。

| 项目 | 原26候选最优 | 新28候选最优 |
| --- | ---: | ---: |
| 745shape单次时间之和 | 146523.9434 µs | 146465.3802 µs |
| 相同数据中新增候选被选中的shape | 0 | 78 |
| 9000选中 | 127 | 68 |
| 9001选中 | 0 | 59 |
| 9010选中 | 29 | 11 |
| 9011选中 | 0 | 19 |

78项改善、667项最优值不变；5项筛选时的延迟降幅超过1%，最大1.371630%。78个新增选中项自身的几何平均加速0.349335%，全集合几何平均加速0.036518%。单次时间之和节省58.5632 µs，不能当作某个模型端到端时间。由于原26始终留在集合内，同一raw数据取最优的28结果不会比26结果更慢，这本身不是重复运行无回退的证据。

## 新候选五轮复测

全78个新增筛选赢家都复测。每个shape重测原最优、全量筛选前三、接近的竞争者和原/新family成员，合计**257次独立signed数据数值校验、1,285条性能记录**。每项五轮、随机候选顺序、同一组8个地址、warmup5/iters51，直接调用正式OPUS接口。复测输入改为seed29正负FP8；仍为native E8M0和完整FP32/BF16 accumulation interval零outlier。

| 指标 | 9001 | 9011 |
| --- | ---: | ---: |
| 全量单轮筛选选中 | 59 | 19 |
| 原选中项在复测中位数继续胜出 | 29 | 12 |
| 五轮都比复测原最优更快 | 2 | 3 |
| 五轮都比复测原最优更慢 | 0 | 2 |
| 原选中子集相对复测原最优GM | +0.014012% | +0.845944% |
| 复测重新选型后的新ID数量 | 30 | 12 |

复测重新选择后是9001共30项、9011共12项；14336×2048×7168从筛选的9011转为9001。其余37项回选原候选。中位数胜出也不等于5/5稳定胜出，所有轮次已保留。

| M×N×K | 新ID | 复测中位数加速 | 五轮更快 |
| --- | ---: | ---: | ---: |
| 1856×7168×768 | 9011 | 7.9178% | 5/5 |
| 1920×7168×768 | 9011 | 5.7285% | 5/5 |
| 32768×7168×384 | 9001 | 1.6048% | 5/5 |
| 1856×7168×384 | 9011 | 1.1793% | 5/5 |
| 16384×7168×3072 | 9001 | 0.5620% | 5/5 |

上述时间来自单独的shared-pool profiler复测，部分差距比筛选更大。地址池、输入seed、GPU分配改变，**不能将这张表的时间混入筛选全集合GM**，也不能直接替代此前Event实验的结论。9011在40960×7168×768和14336×2048×7168中5/5慢，复测选型已回到其他候选；9001的59项几何平均只有+0.0140%，证据仍弱。

## 全后端结果与旧记录

OPUS28相对**本轮CK/CKTile/ASM最快有效者**：720项更快、25项更慢，几何平均 **1.255110×**。全后端最优选择OPUS720、ASM24、CK1；本次枚举不含Triton。CK/CKTile/ASM保留FP32scale，OPUS为native E8M0；公共shuffle和reference不计时，后端内转换按正式tuner计时。

与Sep30历史745个OPUS最优值作描述性比较：649项更快、96项更慢，GM **1.041807×**。历史记录使用不同环境、多卡分配和计时地址池，本次还重建了外部模块；这个+4.1807%不能全部归因于新增候选。Sep30的旧原始表和manifest保留。

无OPUS配置数值失败。外部raw中CKTile874项不支持/执行无效；CK498项无效，其中408项没有有效计时、90项有计时但未通过数值门槛。这些记录已保留并排除选型，所有745shape仍有有效最优项。

## 候选与bound

本次新增性能数据不是新的ATT/counter证据。类型沿用[上轮完整分类](../opus_9000_9010_bound_20261008/BOUND_CLASSIFICATION.md)的配置/shape/阶段范围；详细58个device配置连同本轮parent胜出数量见[完整分类附retune数量](bound_classification_with_retune.csv)。下表只是方向速查，不能将整个parent归为单一瓶颈。

| ID | 合法配置数 | OPUS内胜出 | 全后端胜出 | 已有bound证据方向 |
| --- | ---: | ---: | ---: | --- |
| 9000 | 207 | 68 | 68 | 长K compute/issue＋供数；短K scale latency＋固定成本 |
| 9001 | 207 | 59 | 59 | 优化后主瓶颈未确认；SFA清零降VGPR |
| 9010 | 449 | 11 | 11 | 长K compute/issue＋LDS；短K固定成本 |
| 9011 | 449 | 19 | 19 | 优化后主瓶颈未确认；padded-M unroll4 |
| 9020 | 691 | 89 | 89 | 按实际body：latency 或 compute/供数混合 |
| 9021 | 735 | 46 | 46 | 短K scale latency；长K compute/供数混合 |
| 9022 | 691 | 154 | 154 | 短K memory latency；长K混合 |
| 9023 | 735 | 4 | 4 | runtime VMEM依赖＋compute；fixed support未确认 |
| 9024 | 735 | 9 | 9 | compute/issue＋VMEM/LDS供数 |
| 9030 | 10 | 10 | 10 | compute/issue＋输出store成本；无整系列HBM饱和证明 |
| 9040 | 290 | 13 | 13 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9041 | 290 | 18 | 18 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9042 | 290 | 17 | 9 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9043 | 612 | 2 | 2 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9044 | 612 | 30 | 29 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9045 | 612 | 13 | 13 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9046 | 612 | 30 | 30 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9047 | 290 | 25 | 24 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9049 | 290 | 6 | 6 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9051 | 290 | 11 | 11 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9052 | 290 | 26 | 22 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9053 | 290 | 10 | 7 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9054 | 290 | 4 | 4 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9055 | 612 | 14 | 12 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9060 | 612 | 10 | 8 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9061 | 612 | 18 | 16 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9062 | 612 | 10 | 8 | 按配置：VMEM请求/依赖 或 local LDS/同步 |
| 9063 | 612 | 19 | 19 | 按配置：VMEM请求/依赖 或 local LDS/同步 |

9000/9010长K倾向compute/issue加供数，短K有scale/SMEM依赖及setup/output固定成本；9020等按实际body分类；9040+主要区分VMEM请求依赖与local LDS同步。**没有证据确认任何整个系列为纯HBM bandwidth bound或dispatch bound；9001/9011优化后的主瓶颈仍未重采确认。**

## 运行身份、方法及文件

开始单卡完成150shape；用户要求8卡后，剩余595shape按计算量分成74–75shape/卡。8张MI355X都通过物理设备锁与逐进程KFD握手。每个shape的全部候选在同一卡内计时；GPU间只拆shape。中断中的batch03只作历史raw，不纳入结果，最终无重复shape。正式OPUS SO SHA256：`e18fe59bf6b06ddc53349df5eaa5a3f5d31d3b00b75627705c26477d09db00eb`。

使用正式tuner的候选枚举、worker与torch profiler（warmup5/iters51），以单进程per GPU适配严格owner。shape的OPUS顺序由固定seed打乱。外部模块用当前源码/ROCm重建，记录[external_build.json](external_build.json)与完整构建参数。

输出超过32M elements的shape，地址池上限8，reference及comparison按256行分块；FP32/BF16表达式和零outlier门槛保留，CK全局allclose比例/catastrophic策略保持。最大输出8GiB，全部完成，未因内存跳shape。各shape实际池大小保存在summary与run JSON。这一有界轮换是相对旧自动轮换的计时差异，旧时间变化只能描述性列出。

所有用于结论的运行均逐条检查监测PID等于所属GPU owner，不能仅凭wrapper的contamination=false。筛选共4,549个监测样本；复测另附审计。每个GPU的KFD host PID、PCI与完成状态见[final_review.json](final_review.json)。

- [全量原始profile](profile.csv)：79,504条，含无效记录。
- [原26 OPUS最优](tuned_opus26.csv)、[新28 OPUS筛选最优](tuned_opus28.csv)、[全后端筛选最优](tuned_all.csv)。这三份使用同轮筛选时间，可直接比较。
- [745shape逐项比较](shape_comparison.csv)、[候选胜出数](candidate_wins.csv)、[新增78项筛选清单](new_candidate_selections.csv)。
- [五轮复测逐shape比较](confirmation_comparison.csv)、`confirm_shard_0–7.json`保留全部轮次与数值校验。
- [OPUS复测选型CSV](tuned_opus28_rechecked_config.csv)、[全后端复测选型CSV](tuned_all_rechecked_config.csv)，分别附[timing来源](tuned_opus28_rechecked.csv)和[timing来源](tuned_all_rechecked.csv)。这两份仅78shape使用复测中位数，其余667保留筛选时间，所以不能按混合时间重新声称全集合GM。
- [共享计划](plan.json)、[8卡执行计划](execution_plan_eight.json)、[筛选统计](summary.json)、[最终审核](final_review.json)、[最终文件哈希](final_manifest.json)。

新结果全部写在此报告目录。仓库原生产默认tuning config保留；新CSV可用于显式tuner replay，OPUS仍是native E8M0输入合约。没有commit/push。
