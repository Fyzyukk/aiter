## 12. Oct8 剩余系列：按供数层次定位，再由完整调用取舍

本轮继续处理9020、9022、9023、9024、9030和全部公开9040+：共23个parent、44个与历史赢家相交的实际producer配置、536个历史winner形状。以已验收9021的Oct8正式模块为基线，9000、共享helper、9021和此前已采用的三个runtime B-scale alias保持。公开编号不等于单一device body；9040+的18个parent包含33个实际配置，另有两个无历史winner的producer和五个reducer。精确对应见[完整清单](../opus_remaining_20261008/inventory.json)、[逐配置处理状态](../opus_remaining_20261008/family_progress.json)。44个配置各自一个真实winner的正式API signed2/reference/repeat/guard检查全部通过；这项数值覆盖不等于536项新性能覆盖。

依据[最新合并总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)，分别检查VMEM请求、返回队列、VGPR/LDS消费与同步到达。没有把短K启动、长K普通tile、scale-panel refill、最终输出混成一个带宽瓶颈。ATT以当前CO/FUNC/PC重定位；成功issue=time+stall，duration已经含stall，wait依赖只标识队列成员。所有clocks均为shader clocks，不换算ns。GRBM窗口仍未校准，不报告绝对MFMA utilization或动态occupancy，也不套固定倍率。

### 12.1 证据有效范围和物理流量

[SQ/EA分析](../opus_remaining_20261008/diagnostics/sq_ea_analysis.json)保留20个有限pass，严格owner检查接受10个、排除10个；即使wrapper写contamination:false，发现非owner也排除。没有为补齐表格全量重采。9030必要的ATT/counter单独串行重采；9055原CU0 decoder code:null捕获保留并排除，CU1补采成功。9024 target12有非owner，不能用其与target11构造252/264WG对照。第一次ATT filter误用mangled symbol导致无波形的记录保留，修正为SDK要求的demangled函数名后才分析。

9030的clean SQ/EA在同一有限采集口径中给出median read **2,118,863,744B**、write **2,148,535,936B**、total **4,267,483,264B**；写量约2.149GB/2.001GiB。它确认大输出服务必须进入预算，但单独请求量不证明HBM已饱和，profiled时间不代替无profilerEvent。9062的split2 producer和reducer分开保留counter，完整候选计时包含两者；不能拿producer局部等待解释整个API时间。

### 12.2 Scale issue/publish：相同字节量，不同请求次序

9020 fixed384的scale producer处于起步关键路径，其他wave在publication barrier等待8–1268 clocks；long runtime代表startup只占约3.3%。9022短K有SFA load→wait/store→SFB load→wait/store串行，长K的panel32 refill也串行。9023 runtime的动态普通tile/对齐refill不执行静态大批readlane，实际每wave仅六次恢复用于remainder/output，不能把静态spill数量直接当稳态HBM损失；9024九个真实winner均fixed7168，无spill，startup约15%。对应[merged ATT](../opus_remaining_20261008/diagnostics/merged_att_analysis.json)和[narrow ATT](../opus_remaining_20261008/narrow_att_analysis.json)。

候选保留E8M0、标准(16,16) B preshuffle、K128 scale、FP32累加/BF16输出、u16 panel布局、尾界和0x7f填充。9020仅将SFB先issue，再执行原SFA load/publish，最后SFB publish。最初raw byte尝试被编译器提前消费，数值失败后修正为同一volatile asm中的vmcnt(0)+pack；失败revision和所有SHA保留，失败版本不进入性能选型。最终scoped源码只作用已有fixed384实例，其余六body在正式编译后精确不变。没有减少必要barrier或改变ring出版/退休。

完整[166项Event审查](../opus_remaining_20261008/scale_issue/coverage_independent_review.json)含9020 fixed384全部七winner及9022全部159winner。每项同一共享51地址池、五轮AB/BA HIP graph Event、signed8/reference/repeat/guards，通过strict物理GPU claim。9020七项median全正，geomean **+2.571542%**，34/35配对更快，保留已有fixed384；`12288×7168×384`第2轮speedup0.834618、调用时间增加19.815%完整保留，该shape AB geomean −4.174%、全七shape该轮geomean −0.493%。没有删值、认定它已证明是噪声或重测到正。机械analyzer的严格all-round占位推荐保留，由[最终取舍](../opus_remaining_20261008/scale_issue/retention_decision.json)说明人工判据。

9022整体geomean **+0.820755%**，127项median正、32项负，但六个真实winner五轮一致变慢；K16384的20项中12个paired median负。拒绝全局候选，保留原scale-first矩阵prologue，不用短K收益隐藏稳定winner回归，也不新增未经测试的K阈值。9020其他分支screen混合，其中fixed3072 speedup0.972995、0/5更快，所以同样保留原实现。

[narrow完整13winner](../opus_remaining_20261008/narrow_candidate/retention_recommendation.json)支持9023 runtime geomean **+1.117277%**、9024 fixed7168 **+1.274511%**；9024九项均5/5更快。9023保留`544×7168×16384`median时间+0.5422%与M16非winner+0.4165%的有限代价。VGPR232→251、scalar lane spill46→48代表资源余量减少，不能把零scratch写成无资源风险；收益机制由baseline ATT和candidate ISA支持，没有candidate ATT时间收益声明。

### 12.3 Ring、register和大输出：局部等待不等于可删耗时

9030 interior barrier累计share约22%，但cohort arrival skew median908 clocks、最后MFMA到达跨度900、last arrival→release仅4。它反映供数/计算到达不齐，不是可直接删除的22%墙钟。midpoint候选保留所有publication/retirement及最后alias-C保护；完整十winner使用预算后的同八地址池、51调用graph、五轮AB/BA和256-row FP32区间参考，不能混称自动51地址池。最终十项median全负，geomean **−0.220256%**，仅4/50配对更快；[拒绝global9030](../opus_remaining_20261008/large_midpoint/global_decision.json)。早期代表+0.261%和两项support控制回归均保留。

9046 cluster2的MFMA边界gap median648 clocks、普通边界316，VMEM wait约20，主要线索在LDS/issue/barrier而非单纯HBM等待。9062首barrier到达差2308，慢wave的SFA尾路径使初始matrix晚发；提前矩阵候选两个winner均退。9040/9047短K startup约76%/66%，SMEM waits约1048/1019；9040起步VMEM wait很短，9047另有约169 clocks的混合VMEM/LGKM wait，不能全归VMEM。当前捕获仍不支持更深queue或扩大reuse。9051 startup约87.8%，matrix/byte issue和VMEM等待可以结合旧exact-config LFIFO/translation线索，但这些不能给单请求HBM latency。其新scale-first候选两个代表正后扩大一次完整17winner，结果八正、八负、一平，仅三项5/5更快，也有三项5/5负；[统一拒绝9051](../opus_remaining_20261008/register_issue_order/coverage_analysis.json)。

small/fine本轮候选实测范围是四配置/21个真实winner：9062两项、9046/9055各一项、9051全部17项；四候选都拒绝，分别见[small最终范围审计](../opus_remaining_20261008/small_final_closure_audit.json)。七配置各有自身有限ATT，33配置各有自身数值门；其余keep依据自身type/source/机器身份、数值和存在时的历史exact-config证据。共享机制解释或其他配置的失败不是其性能实测。9071保留N48，旧N32失败仍有效，新草稿在build前停止。18parent逐配置表和五reducer范围见[small总结](../opus_remaining_20261008/small_family_review.md)。

### 12.4 正式集成状态

正式CPU身份审查已通过：26parent/56entry仅9020 fixed384、9023 runtime、9024 fixed7168三个entry改变且精确匹配已测候选；其余53entry、313生成文件、206build objects、202linked gfx950 bundles/263device entries符合Oct8当前基线。9000、9021、共享helper和全部small entry保持，见[identity审计](../opus_remaining_20261008/formal_selected/identity_audit.json)。正式44项API检查进行中，成功后才应用三个源码文件；最终状态以[本轮集成记录](../opus_remaining_20261008/formal_selected/integration_manifest.json)为准。

旧Oct7/Oct8构建、原始结果和SHA snapshot保留；本节新增改变原文档全文SHA，由remaining最终manifest绑定。收益是同一形状等权集合的有限实验，不是生产应用batch收益或全部支持域保证。本轮未执行全745重tune。
