# Oct8 剩余 kernel 定位、有限优化和集成记录

本轮处理用户列出的9020、9022、9023、9024、9030及全部公开9040+：23个parent、44个与历史winner相交的实际producer配置、536个历史winner形状。基线是已验收9021的Oct8正式模块，旧记录保持。9000、共享helper和9021保持；未commit/push。

最终性能取舍为保留已有9020 fixed384、9023 runtime、9024 fixed7168；其余本轮候选拒绝或保持当前配置。正式CPU身份检查与44项正式API验证全部通过，三处精确源码已应用，状态`applied_verified_not_committed`。352次official加120次private数值调用通过，生产2691个构建输入与正式验证树逐字一致。最终机器状态见[集成记录](formal_selected/integration_manifest.json)。

| 系列 | 实际配置 / 历史winner | 新候选验证与结果 | 取舍 |
| --- | ---: | --- | --- |
| 9020 | 7 / 76 | 七分支代表screen；fixed384全部7项median正、34/35配对正，geomean+2.571542% | 仅保留已有fixed384；其余六分支保持 |
| 9022 | 1 / 159 | 全159项geomean+0.820755%，六个真实winner五轮均慢 | 拒绝global issue/publish |
| 9023 | 1 / 4 | 全4项geomean+1.117277%，一项longK median成本+0.5422% | 保留runtime；fixed支持域对照保持 |
| 9024 | 1 / 9 | 全9项geomean+1.274511%，均5/5快 | 保留fixed7168；runtime支持域对照保持 |
| 9030 | 1 / 10 | 全10项median负、4/50配对正，geomean−0.220256% | 拒绝midpoint |
| 9040+ | 33 / 278 | 四实际配置/21winner新候选Event；七配置有限ATT；33配置各自数值门 | 拒绝新9062/9046/9055/9051候选；其余各自keep，原三runtime alias保留 |

性能均使用同一shape的同共享地址池、五轮AB/BA HIP graph Event，含完整调用、signed参考/重复/guard和严格物理GPU owner记录；新机与旧机耗时没有拼接。普通目标为51个地址，大输出9030预算后为八个地址、每graph51调用，明确分开。集合geomean按shape等权，不代表生产batch或未测全部支持域。

9020 `12288×7168×384`第2轮speedup0.834618、时间增加19.815%完整保留；它使该shape AB geomean−4.174%、全七shape该轮−0.493%，没有删除或重测到正。9023另保留M16非winner时间+0.4165%，VGPR232→251、scalar lane spill46→48，记录资源余量风险。9022实际winner稳定退步与有限非winner代价分开判断，不新增K阈值。

诊断按最新Compute/Memory总结展开：9020/9022/9023/9024有scale请求与LDS publish串行；9030/9046的barrier和tile gap要区分供数与到达差；9062有scale尾producer启动延迟；9040/9047短K主要为startup/SMEM设置；9051的issue/wait可以结合旧LFIFO/translation线索，但全17实际winner不支持新类型全局替换。局部ATT、资源和counter用于解释机制，性能由完整Event决定。GRBM未校准，不声明绝对MFMA利用率、动态occupancy或cycles→ns。

每个配置的决定和证据范围见[family_progress.json](family_progress.json)；small的18parent/33config/五reducer表见[small_family_review.md](small_family_review.md)。44配置的数值检查不等于536winner的新性能全覆盖，未做745盲扫。

可审核的入口：

- [全范围清单](inventory.json)和[44配置基线数值结果](diagnostics/all_config_gate_results.json)
- [scale 166项严格Event审查](scale_issue/coverage_independent_review.json)与[最终人工取舍](scale_issue/retention_decision.json)
- [narrow全13winner取舍](narrow_candidate/retention_recommendation.json)
- [9030全十winner拒绝](large_midpoint/global_decision.json)与[独立审查](large_midpoint/coverage_independent_review.json)
- [9051全17winner拒绝](register_issue_order/coverage_analysis.json)和[small范围最终核验](small_final_closure_audit.json)
- [正式身份审计](formal_selected/identity_audit.json)、[独立复核](formal_selected/independent_identity_plan_review.json)和[正式API结果](formal_selected/api_gate/gpu_api_analysis.json)
- 原有[Compute总结](../opus_bound_analysis_20261007/COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md)与[Memory总结](../opus_bound_analysis_20261007/MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md)第12节

准备时的source manifest与机械analyzer推荐保持历史字节；当前取舍通过新decision衔接。首次无波形ATT、非owner窗口、失败raw-byte revision、9030不合法M plan和停止的9071草稿均保留并标明排除。最终文档和证据SHA由[final_manifest.json](final_manifest.json)绑定。
