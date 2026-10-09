# 剩余23个parent与44个实际配置的最终结果

当前状态为 **applied_verified_not_committed**：本轮新增采用并应用的device entry只有9020原fixed384、9023原runtime和9024原fixed7168，共三个源码文件。其余配置保留当前selected，已测候选按自身证据拒绝；此前已采用9021与9042/9053/9054 runtime B-scale alias保持。没有commit或push，待执行实验为0。逐symbol状态依据 [family_progress.json](family_progress.json)，源码应用和验证依据 [integration_manifest.json](formal_selected/integration_manifest.json)。

本表的44项是与536个历史winner相交的**实际producer symbol配置**，分属23个public parent；同ID的runtime/fixedK或不同traits单独成行。它们与正式构建的26parent/56entry、正式API的26parent/44target分别计数。全部44配置各取一个自身winner完成原Oct8 baseline signed2repeat/reference/guard；最终正式模块再完成44 API target、352次official加120次private，共472次数值调用。两组数值门都没有性能结论，未测配置的keep不表示全winner提速。

[最终正式模块](jit_formal_selected/module_deepgemm_opus.so)的SHA256为 `4be55119596805e3d2967a00c3cbf70129f280094d14d43fbbb2461f91aee1bd`。[正式身份审计](formal_selected/identity_audit.json)确认56entry中3改变精确匹配已测候选、53不变；313生成文件、206build object、202linked gfx950 bundle/263device entry核对通过。[正式API审查](formal_selected/api_gate/gpu_api_analysis.json)确认44target均通过signed8/reference/repeat/guard、实际加载SHA和物理GPU owner；[原44配置数值门](diagnostics/all_config_gate_results.json)用于各配置baseline正确性。small为33个winner-producer配置、另外2个无历史winnerproducer及5个reducer；本轮small候选性能范围严格为4config/21winner、ATT为7config、各自数值门为33config，5reducer保留身份且仅matching调用计时按实际包含。

## 23个public parent汇总

| parent | 实际配置数 | 历史winner数 | 本轮候选性能实测范围 | 最终状态 | 依据 |
| --- | ---: | ---: | --- | --- | --- |
| 9020 | 7 | 76 | fixed384全部7winner；其余6body各1screen Event | 采用原fixed384；其余6配置保留 | [记录](scale_issue/retention_decision.json) |
| 9022 | 1 | 159 | 全部159winner Event | 拒绝global候选；保留原配置 | [记录](scale_issue/retention_decision.json) |
| 9023 | 1 | 4 | 全部4winner Event；M16非winner | 采用原runtime；fixed支持域不变 | [记录](narrow_candidate/retention_recommendation.json) |
| 9024 | 1 | 9 | 全部9winner Event；unchanged runtime control | 采用原fixed7168；runtime不变 | [记录](narrow_candidate/retention_recommendation.json) |
| 9030 | 1 | 10 | 全部10winner Event；8地址池，另screen/2非winner | 拒绝midpoint；保留原配置 | [记录](large_midpoint/global_decision.json) |
| 9040 | 1 | 11 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9041 | 2 | 17 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9042 | 3 | 15 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | 保留此前runtime alias；其余keep | [记录](small_family_review.json) |
| 9043 | 1 | 2 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9044 | 2 | 26 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9045 | 1 | 17 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9046 | 1 | 32 | 1个自身winner候选Event | 拒绝自身有限候选；保留原配置 | [记录](ring_overlap/results_analysis.json) |
| 9047 | 1 | 27 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9049 | 1 | 6 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9051 | 1 | 17 | 17个自身winner候选Event | 拒绝自身有限候选；保留原配置 | [记录](register_issue_order/coverage_analysis.json) |
| 9052 | 2 | 24 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9053 | 2 | 9 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | 保留此前runtime alias；其余keep | [记录](small_family_review.json) |
| 9054 | 1 | 3 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | 保留此前runtime alias；其余keep | [记录](small_family_review.json) |
| 9055 | 1 | 17 | 1个自身winner候选Event | 拒绝自身有限候选；保留原配置 | [记录](ring_overlap/results_analysis.json) |
| 9060 | 3 | 9 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9061 | 1 | 19 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |
| 9062 | 2 | 6 | 2个自身winner候选Event | 拒绝自身有限候选；保留原配置 | [记录](fine_startup/results_analysis.json) |
| 9063 | 7 | 21 | 本轮新候选性能未测；自身数值/identity/存在时历史证据 | keep自身当前配置 | [记录](small_family_review.json) |

采用三entry的等shape geomean收益为9020 fixed384 **+2.571542%**、9023 runtime **+1.117277%**、9024 fixed7168 **+1.274511%**。9020的`12288,7168,384` round2时间增加19.815%仍保留，原因未确认；9023的`544,7168,16384` winner median时间增加0.542%、M16非winner增加0.417%，VGPR232→251和scalar lane spill46→48风险也保留。9022虽整体GM+0.821%，六winner稳定5/5更慢；9030全十ratio中位数负；9051全17为8正8退1平，统一拒绝。这些有限集合统计不是生产batch或全部支持域收益。

## 全部44个actual producer配置

短traits标签仅省略类型固定前缀：`merged8`、`merged160`、`narrow64x128`、`narrow64x64`、`large_output`、`register`、`lds`、`fine`对应原traits家族，尖括号内参数原样保留。完整mangled symbol、selected traits、状态原文和所有winner shape见 [family_progress.json](family_progress.json) 与 [inventory.json](inventory.json)。9042/9053/9054旧inventory descriptive false标签已按实际selected symbol/SHA纠正为true，历史文本单列保存。

| # | parent | actual ID | selected traits短标签 | 历史winner数 | 最终state | 性能实测范围 | 证据与适用限制 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 1 | 9020 | 9020 | `merged8<128, 128, 64, 0>` | 3 | 拒绝本轮候选；保留原配置 | 1个自身配置screen Event 8192,768,7168（自身winner）；未覆盖全部winner | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；单代表mixed；不从其他body失败冒称自身全winner性能 |
| 2 | 9020 | 9020 | `merged8<192, 256, 128, 0>` | 31 | 拒绝本轮候选；保留原配置 | 1个自身配置screen Event 1536,7168,16384（自身winner）；未覆盖全部winner | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；单代表mixed；不从其他body失败冒称自身全winner性能 |
| 3 | 9020 | 9020 | `merged8<192, 256, 32, 1536>` | 9 | 拒绝本轮候选；保留原配置 | 1个自身配置screen Event 1344,16384,1536（自身winner）；未覆盖全部winner | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；单代表mixed；不从其他body失败冒称自身全winner性能 |
| 4 | 9020 | 9020 | `merged8<192, 256, 32, 3072>` | 5 | 拒绝本轮候选；保留原配置 | 1个自身配置screen Event 1728,7168,3072（自身winner）；未覆盖全部winner | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；单代表mixed；不从其他body失败冒称自身全winner性能 |
| 5 | 9020 | 9020 | `merged8<192, 256, 64, 7168>` | 16 | 拒绝本轮候选；保留原配置 | 1个自身配置screen Event 6144,2048,7168（自身winner）；未覆盖全部winner | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；单代表mixed；不从其他body失败冒称自身全winner性能 |
| 6 | 9020 | 9020 | `merged8<192, 256, 8, 384>` | 7 | 采用候选；已应用并验证 | 全部7/7自身winner Event；另1screen（去重） | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；仅existing fixed384；保留round2时间+19.815%异常 |
| 7 | 9020 | 9020 | `merged8<192, 256, 8, 768>` | 5 | 拒绝本轮候选；保留原配置 | 1个自身配置screen Event 1728,7168,768（自身winner）；未覆盖全部winner | [screen](scale_issue/screen_analysis.json)、[取舍](scale_issue/retention_decision.json)；单代表mixed；不从其他body失败冒称自身全winner性能 |
| 8 | 9022 | 9022 | `merged160` | 159 | 拒绝本轮候选；保留原配置 | 全部159/159自身winner Event | [166覆盖](scale_issue/coverage_independent_review.json)、[取舍](scale_issue/retention_decision.json)；GM+0.821%仍有6winner稳定5/5更慢；全局拒绝，无新K阈值 |
| 9 | 9023 | 9023 | `narrow64x128<3, 32, 0>` | 4 | 采用候选；已应用并验证 | 全部4/4自身winner Event；另M16非winner | [13winner取舍](narrow_candidate/retention_recommendation.json)；保留longK winner median成本+0.542%、M16+0.417%；VGPR232→251/lane spill46→48 |
| 10 | 9024 | 9024 | `narrow64x64<64, 7168, 4>` | 9 | 采用候选；已应用并验证 | 全部9/9自身winner Event；另unchanged runtime control | [13winner取舍](narrow_candidate/retention_recommendation.json)；9winner45/45pair更快；仅原fixed7168，runtime不变 |
| 11 | 9030 | 9030 | `large_output` | 10 | 拒绝本轮候选；保留原配置 | 全部10/10自身winner Event（8地址池）；另screen/2非winner | [10覆盖](large_midpoint/coverage_independent_review.json)、[拒绝](large_midpoint/global_decision.json)；10个ratio中位数全负、4/50pair快；22%barrier波份额不是可删墙钟 |
| 12 | 9040 | 9040 | `register<16, 32, 1, 1, 6, 1, 3, 3, 0, false, false>` | 11 | keep自身当前配置 | 本轮新候选性能未测；自身ATT1形状；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 13 | 9041 | 9041 | `register<16, 16, 1, 1, 2, 8, 0, 0, 0, false, false>` | 1 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 14 | 9041 | 9050 | `register<16, 16, 1, 1, 3, 8, 3, 3, 0, false, false>` | 16 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 15 | 9042 | 9071 | `register<16, 48, 1, 1, 4, 4, 4, 3, 7168, true, false>` | 7 | keep自身当前配置 | 本轮新候选性能未测；自身ATT1形状；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身旧N32失败保留N48；9071草稿未build，9071ATT不是9073性能 |
| 16 | 9042 | 9042 | `register<32, 32, 1, 1, 3, 4, 3, 0, 0, false, true>` | 1 | 保留此前已采用runtime alias | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；采用依据为此前自身winner Event；本轮仅保留，未新增采用 |
| 17 | 9042 | 9073 | `register<32, 48, 1, 1, 3, 4, 4, 0, 7168, true, false>` | 7 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身旧N32失败保留N48；9071草稿未build，9071ATT不是9073性能 |
| 18 | 9043 | 9043 | `lds<32, 64, 1, 4, 8, 2, 2, false, false, true, false, false, 1, 4, 128, 0, 0, false>` | 2 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 19 | 9044 | 9044 | `lds<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true, 1, 4, 128, 0, 0, false>` | 9 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 20 | 9044 | 9056 | `lds<64, 64, 2, 2, 8, 2, 2, false, false, true, false, false, 1, 4, 128, 0, 0, false>` | 17 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 21 | 9045 | 9045 | `lds<96, 64, 2, 2, 4, 1, 1, false, false, false, false, true, 1, 4, 128, 0, 0, false>` | 17 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 22 | 9046 | 9046 | `lds<64, 128, 4, 2, 6, 2, 2, false, false, false, false, false, 1, 4, 128, 0, 0, false>` | 32 | 拒绝本轮候选；保留原配置 | 1/32自身winner候选Event 256,7168,16384 | [自身Event](ring_overlap/results_analysis.json)、[配置/历史证据](small_family_review.json)；自身有限候选失败；不宣称全部winner或别的配置有性能结果 |
| 23 | 9047 | 9047 | `lds<32, 64, 2, 2, 4, 1, 2, true, false, false, false, false, 1, 4, 128, 0, 0, false>` | 27 | keep自身当前配置 | 本轮新候选性能未测；自身ATT1形状 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 24 | 9049 | 9049 | `lds<32, 128, 2, 2, 4, 2, 2, true, true, false, true, false, 1, 4, 128, 0, 0, false>` | 6 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 25 | 9051 | 9051 | `register<16, 32, 1, 1, 3, 4, 3, 3, 0, false, false>` | 17 | 拒绝本轮候选；保留原配置 | 17/17自身winner候选Event | [自身Event](register_issue_order/coverage_analysis.json)、[配置/历史证据](small_family_review.json)；17项8正8退1平，仅3项5/5快；统一reject，不按shape切scope |
| 26 | 9052 | 9052 | `register<16, 32, 1, 1, 2, 8, 3, 3, 0, false, false>` | 7 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 27 | 9052 | 9070 | `register<16, 32, 1, 1, 3, 8, 4, 3, 7168, false, true>` | 17 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 28 | 9053 | 9053 | `register<32, 32, 1, 1, 2, 8, 3, 3, 0, false, true>` | 2 | 保留此前已采用runtime alias | 本轮新候选性能未测；自身历史exact-config Event 2行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；采用依据为此前自身winner Event；本轮仅保留，未新增采用 |
| 29 | 9053 | 9072 | `register<32, 32, 1, 1, 4, 4, 4, 3, 7168, false, true>` | 7 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 30 | 9054 | 9054 | `register<32, 64, 1, 1, 2, 4, 3, 3, 0, false, true>` | 3 | 保留此前已采用runtime alias | 本轮新候选性能未测；自身历史exact-config Event 3行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；采用依据为此前自身winner Event；本轮仅保留，未新增采用 |
| 31 | 9055 | 9055 | `lds<32, 64, 1, 4, 12, 4, 2, false, false, true, false, false, 1, 4, 128, 0, 0, false>` | 17 | 拒绝本轮候选；保留原配置 | 1/17自身winner候选Event 64,7168,7168 | [自身Event](ring_overlap/results_analysis.json)、[配置/历史证据](small_family_review.json)；自身有限候选失败；不宣称全部winner或别的配置有性能结果 |
| 32 | 9060 | 9060 | `fine<80, 1, 4, 4, 1, 1, 4, 128, 0, 0>` | 2 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 33 | 9060 | 9060 | `fine<80, 1, 4, 4, 1, 1, 4, 128, 0, 3072>` | 2 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 34 | 9060 | 9060 | `fine<80, 1, 4, 4, 1, 1, 4, 128, 0, 7168>` | 5 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 35 | 9061 | 9061 | `fine<96, 2, 4, 4, 1, 1, 4, 128, 0, 0>` | 19 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 36 | 9062 | 9062 | `fine<80, 1, 4, 4, 1, 2, 4, 128, 0, 16384>` | 2 | 拒绝本轮候选；保留原配置 | 2/2自身winner候选Event，含matching split2 reducer | [自身Event](fine_startup/results_analysis.json)、[配置/历史证据](small_family_review.json)；本配置2/2 winner已完整覆盖，候选失败；不推广为parent全部6个winner或其他配置的性能结果 |
| 37 | 9062 | 9064 | `fine<96, 2, 4, 4, 1, 2, 16, 128, 2, 0>` | 4 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 38 | 9063 | 9068 | `fine<112, 1, 4, 4, 1, 4, 4, 128, 0, 0>` | 1 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 39 | 9063 | 9067 | `fine<128, 2, 2, 4, 1, 4, 4, 128, 0, 0>` | 1 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 40 | 9063 | 9069 | `fine<48, 1, 4, 4, 1, 4, 4, 128, 0, 0>` | 1 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 41 | 9063 | 9063 | `fine<80, 1, 4, 4, 1, 4, 16, 128, 2, 0>` | 7 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 42 | 9063 | 9063 | `fine<80, 1, 4, 4, 1, 4, 4, 128, 0, 16384>` | 2 | keep自身当前配置 | 本轮新候选性能未测；自身历史exact-config Event 1行 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 43 | 9063 | 9066 | `fine<96, 2, 2, 4, 1, 4, 16, 64, 0, 0>` | 1 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |
| 44 | 9063 | 9065 | `fine<96, 2, 4, 4, 1, 4, 16, 128, 2, 0>` | 8 | keep自身当前配置 | 本轮新候选性能未测 | [自身配置/历史范围](small_family_review.json)、[自身数值门](small_config_gate_audit.json)；自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能 |

本轮新候选有完整自身winner集合性能的配置是9020 fixed384、9022、9023 runtime、9024 fixed7168、9030、runtime9051和fixed9062；9046/9055各只测一个自身winner，fixed9062测其两个winner，其余9020 body各只一个screen。其余small配置未测本轮新候选性能，自身历史exact-config Event或局部ATT分别列出，共享机制说明和别的配置reject不算其性能实测。

44配置表不包含没有历史winner的合法support body：9023 fixed7168、9024 runtime，以及small的两个兼容/runtime producer；正式56entry由本表44配置、这四body、5个reducer和范围外已冻结9000/9010/9021三个entry组成，均逐项检查身份。5reducer无逐个独立性能声明，fixed9062 initial handoff的完整Event包含实际matching split2 reducer；split1的9046/9055/9051包含producer内部处理，不因runner通用includes文字额外算一次reducer。所有local ATT与counter仍限自身shape/CO/PC/物理owner；GRBM未校准，不声明绝对MFMA利用率、动态occupancy或shader clocks→ns。

生成依据SHA256：

- [family_progress](family_progress.json)：`08b365a467aeaf7387b53d1780914cfe1d186336c033cc29c3c45e85e8b10902`
- [inventory](inventory.json)：`1144b4750389fe5381cfc7318e52f87009e26c2cc715105f2bae2967e957ba7b`
- [integration](formal_selected/integration_manifest.json)：`ccc070d6a8f77b40dd294a1b68bd9004d041bf33ca79ab088467f21c6d3d398d`
- [正式API](formal_selected/api_gate/gpu_api_analysis.json)：`21410deb2eafce419245b90049b6cf3b6499301211c24ddcc0d71e49b2512e40`
- [baseline44gate](diagnostics/all_config_gate_results.json)：`88ab58a5113ba20b7d96800c9cad76b83c87e9b39bf58e1f38c0712735eb6134`
