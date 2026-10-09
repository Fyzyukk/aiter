# 9000/9010 独立候选与全系列 bound 分类

当前状态为 **applied_verified_not_committed**。按用户最终要求，保留原9000/9010，各新增一个独立候选，共 **9000、9001、9010、9011** 四项。没有增加尺寸硬编码dispatch；新ID进入现有tuner，保存的旧ID继续调用原版。此前已采用9020 fixed384、9021 scale issue/publish、9023 runtime、9024 fixed7168和9042/9053/9054 runtime B-scale alias均保持。

| ID | 实现 | 自身性能证据 | 最终取舍 |
| --- | --- | --- | --- |
| 9000 | 原256×256 / unroll2 | 原始基线 | 保留 |
| 9001 | SFA raw先清零，VGPR477→469 / AGPR221→213，LDS152064与无scratch保持 | 全129历史winner GM **0.999512877×（−0.0487%）**，49正median、7个5/5 loser；两原5/5正样本新seed仍微正但仅3/5轮快 | 仅作为可选tuner候选；证据较弱，不全局替换，不保证获益 |
| 9010 | 原padded-M 256×256 / unroll2 | 原始基线 | 保留 |
| 9011 | padded-M unroll4，VGPR497→492 / AGPR241→236，无scratch | 全38历史winner GM **1.002808271×（+0.2808%）**，34正median，10项5/5快，0项5/5慢；4项negative median | 保留独立tuner候选，原版可回选 |

9001的两个原5/5正样本为`1792×7168×384`与`2048×7168×3072`：第一次median加速约0.5912%/0.6633%；独立seed29复核约0.1114%/0.0387%，各仅3/5轮快，不能描述为稳定的半个百分点收益。另`512×512×8320`复核median约+0.2841%、5/5快，但它是非winner control。减寄存器本身不证明性能提高或dispatch瓶颈。

9011四个negative median分别为`448×65536×1536`、`1920×7168×3072`、`1920×7168×16384`、`1984×6144×7168`，最差ratio约0.996876（时间约+0.3134%）。收益很小，仍须以当前机器的tuner结果选择，不把等shape GM当作生产负载收益。

9000 unroll4也完成全129实测，GM **0.996996820×（−0.3003%）**，25正median、46个5/5 loser。独立seed复核没有恢复清晰收益，已淘汰；不会再增加第三个9000类候选。其他matrix-first、SFB-first、unroll1/8、scale-release分支的负样本或资源失败均保留，`padded_sparse_clear`是未测试草稿。

全系列分类见[BOUND_CLASSIFICATION.md](BOUND_CLASSIFICATION.md)，覆盖原 **26公开parent/56entry** 和新增后 **28parent/58entry**。9040+的33个历史winner producer配置均有自身ATT（旧7＋新26），不是只给9000/9010贴标签。完整逐symbol证据在[JSON](bound_classification.json)和[CSV](bound_classification.csv)。四support body和五reducer明确标为未确认；新增候选没有post-change ATT，不能直接套用原版主瓶颈。

分类主要结论：长K的大tile偏compute/issue和供数混合；register队列偏VMEM请求与依赖等待；LDS/fine偏local LDS依赖与同步；短K启动与输出固定成本显著。9051有较强memory request/latency与可能的带宽成分。**现有证据没有证明任何整个系列为纯HBM bandwidth bound或dispatch bound。** CPU注册/分支审计不等于dispatch瓶颈测量，局部LDS issue也不等于LDS带宽饱和。

## 正式应用和验证

最终正式模块：[module_deepgemm_opus.so](jit_split_formal_scale_reset/module_deepgemm_opus.so)，SHA256 `e18fe59bf6b06ddc53349df5eaa5a3f5d31d3b00b75627705c26477d09db00eb`。正式构建28个公开parent/58entry；完整linked module为204个gfx950 bundle/265 entry。所有原263 linked entry的FUNC、完整metadata和normalized descriptor与上一版逐项相等。两新增entry精确匹配各自已测private候选；metadata中的`.name/.symbol`随新traits改变，其余完全匹配。见[identity_audit](split_formal_scale_reset/identity_audit.json)。

最终14个official API target/112次数值调用全部通过signed8、独立FP32/BF16区间reference、repeat、output/workspace guards及实际加载module SHA；9001/9011各6target，9000/9010各1control。严格GPU owner确认clean，见[集成记录](split_formal_scale_reset/integration_manifest.json)。其余原entry没有重新做全parent数值门，以精确机器身份保持及上轮正式验证为依据。

源码应用六个文件，全部tracked `aiter/csrc`输入逐字等于已构建worktree；先前五个已采用header保持。原先短暂应用到9010本体的unroll4决定已被此次独立9011方案取代，旧[中间记录](formal_selected/integration_manifest.json)保留为历史。未commit/push，未改保存的tuning CSV，待执行实验为0。

已有保存结果不会自动选择9001/9011；下次用原tuner重新调优或在官方API显式指定新ID时才能使用。本轮没有重调全部745个历史winner归属，也不宣称相对其他所有parent/外部backend取得新全局winner。

## 性能口径与排除记录

9000全129分为25项51地址池和104项8地址池，均51次调用/graph、五轮AB/BA Event；只在各自同批基线和候选之间算ratio，按shape等权汇总。大输出参考按256行分块；原FP32区间契约保持。9011全38用51地址池和51调用/graph；profiler计数器与Event性能判断分开。

第一轮大输出性能及第一轮9010替换API出现非owner PID，已经排除，由独立串行复测替代。原raw与[owner_exclusion](scale_reset/large9000_event.owner_exclusion.json)保持；[full9000_analysis.owner_excluded.json](scale_reset/full9000_analysis.owner_excluded.json)不作决策证据，最终[full9000_analysis.json](scale_reset/full9000_analysis.json)只用clean25＋clean串行104。

[owner_audit.json](owner_audit.json)的全历史aggregate为`failed`，因为它有意包含两条已排除attempt；各个最终选用epoch的`strict_clean`均为true，并绑定在集成/性能记录。不能据aggregate错误地忽略有效串行复测，也不能凭wrapper的`contamination:false`接受非owner记录。

## 主要记录

- [9000全129 scale-reset决定](scale_reset/full9000_analysis.json)
- [9001独立seed复核](split_formal_scale_reset/confirm9000_event.json)
- [9000 unroll4淘汰](loop_unroll4/full9000_split_analysis.json)
- [9011全38性能](loop_unroll4/full9010_analysis.json)
- [最终正式应用manifest](split_formal_scale_reset/integration_manifest.json)
- [全部entry分类](BOUND_CLASSIFICATION.md)

`prepare.py`、`finalize_optimization.py`、`apply_split.py`是本次一次性执行记录，含历史绝对路径与阶段前置断言，不作为新机器自动重跑入口。原Compute/Memory文档新增第13节，旧字节前缀与历史证据保持；当前文件哈希见[final_manifest.json](final_manifest.json)。
