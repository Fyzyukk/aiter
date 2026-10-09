

## 12. 2026-10-08剩余23个parent的有限优化与正式门禁

### 12.1 恢复基线、实际配置与证据范围

本节承接第11节的已采用9021，按[剩余inventory](../opus_remaining_20261008/inventory.json)收口23个public parent、44个有历史winner的实际device配置及536个历史winner。9020本轮已获用户授权；9000、共享helper、已采用9021保持冻结。基线仍是Oct8正式selected `module_deepgemm_opus.so`，SHA256 `6f6a0117b122f06b4802833c020effb4b24ff9fea56f20c8c16b9ced9787173f`，对应HEAD `b152ab834e68f8fde18adbd7b200c587770b9fb5`及先前small runtime B-scale alias。这里的actual配置以实际producer symbol/完整traits为准，同一parent的fixedK与runtime配置分别记录。

| 剩余范围 | public parent数 | 自身实际配置数 | 历史winner数 | 本轮最终决定 |
| --- | ---: | ---: | ---: | --- |
| 9020 | 1 | 7 | 76 | 仅保留已有fixed384配置候选，其余6配置保留当前版本 |
| 9022 | 1 | 1 | 159 | 拒绝global issue/publish候选，保留当前版本 |
| 9023 | 1 | 1 | 4 | 保留已有runtime配置候选 |
| 9024 | 1 | 1 | 9 | 保留已有fixed7168配置候选 |
| 9030 | 1 | 1 | 10 | 拒绝global midpoint候选，保留当前版本 |
| small/fine，详见12.7的18-parent表 | 18 | 33 | 278 | 四项候选拒绝，本轮没有新增采用 |
| 合计 | 23 | 44 | 536 | 本轮仅选定3个新device entry |

[当前44配置数值门](../opus_remaining_20261008/diagnostics/all_config_gate_results.json)已通过：每个配置各取一个自身actual winner，使用原Oct8正式selected、signed输入、2次repeat/reference/output及workspace guard，并记录实际加载的module SHA；该检查没有Event性能结果。small的33配置另有[逐配置审计](../opus_remaining_20261008/small_config_gate_audit.json)。**全部44配置均获得自身数值覆盖，不等于全部536历史winner获得本轮性能覆盖**，也不等于新正式selected的API门禁已结束。新候选完整winner性能覆盖分别为scale166、narrow13、9030的10以及small中9051的17；其他screen及局部ATT按自身有限范围单列，不能用共享pipeline的结论代替未测配置。

### 12.2 先定位startup及wave到达差，再判断可优化空间

[本轮ATT汇总](../opus_remaining_20261008/diagnostics/merged_att_analysis.json)核准当前module、CO、FUNC、full metadata、normalized descriptor、PC映射、完整wave及code/stat聚合，并使用clean同mm owner身份。gfx9成功issue仍按`time+stall`取值，duration已包含stall；shader clocks及wave份额不能换成整核墙钟损失，wait dependency只标记队列成员，EXEC未导出。原9030 capture的owner之外额外PID没有证明来源，不能事后猜为rocminfo而洗成clean；该记录保留为拒绝证据，采用后来独立clean补采。尚未核准的counter时钟窗口也不支持绝对MFMA利用率或动态occupancy。

| 当前exact body / 代表 | 完整wave数 | 首MFMA中位数 / clocks | wave时长中位数 / clocks | 局部机制含义 |
| --- | ---: | ---: | ---: | --- |
| 9020 fixed384 `[1536,7168,384]` | 8 | 3492 | 13876 | raw-scale启动链占较大比例，可检验更早issue |
| 9020 runtime `[1536,7168,16384]` | 8 | 9162 | 278388 | startup约3.3%，不能外推短K收益 |
| 9022 `[800,7168,384]` | 4 | 5016 | 10578 | 同一body短K的producer与发布等待明显 |
| 9022 `[800,7168,16384]` | 4 | 7022 | 213372 | 长K需区分普通边界与scale refill |
| 9030 `[65536,16384,1536]` | 688 | 8034 | 36834 | 86个八wave发布组，startup约21.8% |

9020 fixed384的producer先SFA wait496 clocks，再SFB wait376，其余wave的publication barrier可到1268；它支持raw-scale请求序列化诊断，但这些wait可能包含矩阵队列，不能按scale独立返回延迟计账。9020长K的普通interior窗口中位数2072 clocks，barrier wave份额20.13%，不表示可删除20.13%的整核时间。

9022短K只有producer wave执行SFA/SFB，相关wait为196/444 clocks，矩阵wait436，producer barrier8；其他wave barrier408–460。原SFA第二pass的合法guard在K384跳过，不能把静态存在的第二pass当作动态工作。长K每wave2560条MFMA，普通与refill混合窗口中位数1556，scale queue wait wave份额约1.59%；refill在tile30/62/94边界，需保留panel32的访问及发布同步。静态VGPR/SGPR/LDS计数只用于风险审查，不作为瓶颈证明。

### 12.3 Raw-scale issue/publish：安全版本与166项固定覆盖

[安全source/ISA复核](../opus_remaining_20261008/scale_issue/scale_source_review.json)确认9020把raw SFB提前到原SFA之前发出，`vmcnt(0)`与pack位于同一volatile asm block，等待之前不消费raw VGPR；9022先issue原SFA各pass及SFB，再按原guard发布。原始读取字节、SFA/SFB布局、scale-first矩阵prologue、panel32/refill、尾行guard、barrier及输出语义保留。两次拒绝revision继续留档：第一版编译器插入早wait而没有得到目标issue顺序；第二版把standalone wait与C++ pack分开后，编译器把pack提前到wait之前，数值失败。该unsafe版本不代表后来安全版本仍失败，也不能仅凭源码顺序宣称异步重排成立。

最终安全候选library SHA256 `10aabbbeef67cae83c9c18a0d0c307a5b5353dc836111cf546d949951c5ee7ea`、CO `706f335722d4d42ecfdf909ceb9e6fe16b9f706b44c971ebf98dc153fcff7702`。[166项独立审计](../opus_remaining_20261008/scale_issue/coverage_independent_review.json)覆盖已有9020 fixed384的全部7个winner及9022全部159个winner，均完成signed8/reference/guard/repeatability和5轮AB/BA、51次完整调用的共同自动地址池Event。Baseline使用未经改变的OfficialRunner包装，精确加载原Oct8正式module；此前私有9020 baseline有20字节差异，已排除出该覆盖依据。原结果没有逐项序列化`actual_module`字段；原OfficialRunner首次调用的已加载SHA核对通过，这个记录限制仍保留，不能补写成原始JSON已有该字段。

### 12.4 9020只保留已有fixed384，9022全局拒绝

[最终scope决定](../opus_remaining_20261008/scale_issue/retention_decision.json)保留9020**已有fixed384**，不新增shape或K阈值。7/7 shape Event中位数及paired中位数正，6个shape为5/5轮更快，另一个4/5，共34/35个positive pair；等shape几何平均speedup为+2.571542%，shape耗时中位数之和的speedup为+2.800398%。当前没有actual winner稳定5/5退化。该entry VGPR216、SGPR54、LDS119840、scratch及spill均保持，ISA6284→6312字节；这些资源没有变化本身不证明性能收益。

必须保留的负样本是`[12288,7168,384]`的**round2**：baseline `67.43829390581917 μs`，candidate `80.80135607251934 μs`，speedup `0.8346183428566867`，即**该轮耗时增加19.8152435%**。其余4轮改善约2.55%–3.00%，该shape总体中位数仍正；但该shape AB几何平均为−4.1740986%，全部7个shape的round2 pooled几何平均为−0.493013%。全部7个shape的AB/BA分别+1.6153806%/+2.6637477%。该轮处于clean epoch，原因未知，没有丢弃、归零、认定contamination或重跑寻找正结果。

机械分析器要求每一pooled round皆正，因此曾建议拒绝9020 fixed384；**原机械JSON保持不变，最终root scope决定明确以逐shape中位数/paired稳定性取代该机械建议，并接受上述孤立大代价风险**。这属于有限证据下的显式决定，不是改写异常。9020其余6个已有body只有各自一个screen代表，结果mixed；fixed3072代表speedup退化−2.70047%且0/5轮更快。因此global9020候选拒绝，不能把fixed384的完整7项收益扩大为9020全部76个winner收益。

9022全159项等shape几何平均虽为+0.820755%，shape中位耗时和speedup仅+0.429064%，127个ratio中位数正、32个负，79个5/5更快，却有6个actual winner **5/5更慢**。VGPR194、LDS81184不变，SGPR66→64、scratch及VGPR/SGPR spill为0，不抵消稳定退化。

| 9022 K cohort | winner数 | 等shape GM speedup | 稳定5/5更慢数 |
| --- | ---: | ---: | ---: |
| 384 | 22 | +1.786601% | 0 |
| 768 | 22 | +1.763129% | 0 |
| 1536 | 30 | +0.309205% | 1 |
| 3072 | 22 | +0.415938% | 1 |
| 7168 | 43 | +0.752891% | 1 |
| 16384 | 20 | +0.095636% | 3 |

K16384还存在10/20个ratio中位数负、12/20个paired中位数负及pooled round2负。六个稳定loser的shape及ratio speedup为`[512,16384,1536]` −1.153093%、`[704,7168,3072]` −0.556446%、`[896,7168,16384]` −0.406775%、`[1024,7168,16384]` −0.830989%、`[1280,7168,16384]` −0.818175%、`[4096,2048,7168]` −0.157735%；第一项paired中位数为−0.239276%，ratio与paired统计分开保留。已有9021/narrow接受有限代价时没有actual winner稳定5/5退化，本次不能用整体GM覆盖global9022的稳定loser。最终拒绝整个原global entry候选，保留当前9022，不发明短K阈值，不事后挑positive subset，不追加重测。

### 12.5 Narrow：保留9023 runtime与9024 fixed7168，并列资源代价

[独立narrow决定](../opus_remaining_20261008/narrow_candidate/retention_recommendation.json)与[所有静态副本CFG审计](../opus_remaining_20261008/narrow_candidate/isa_order_audit.json)支持保留原9023 runtime entry及原9024 fixed7168 entry。候选在完整vector路径先issue原SFA low/high及SFB，再publish；原guarded helper、显式wait/barrier、尾helper及epilogue保留。9023五份full-vector静态copy及9024一份均核对实际ISA顺序；partial byte helper仍可能序列化，不声称全支持域都重叠。

9023全部4个actual winner，含K7168/K16384、M544尾行及M576整行，加M16非winner边界；9024 fixed全部9个actual winner，另带unchanged runtime control，共15个target完成signed8/reference/repeat/guards及clean共同51地址池、5轮AB/BA完整调用Event。9023四winner GM speedup为+1.117277%，两个K组的聚合分别+1.916205%/+0.324612%，全四shape的五个pooled round均正，但longK分shape及其pooled round仍mixed，不把“两个K组总体正”写成“longK每轮正”。接受`[544,7168,16384]`中位耗时+0.542160%、2/5轮更快，以及M16边界+0.416545%、1/5轮更快；没有actual winner 0/5更快。

9024 fixed九winner GM为+1.274511%，九项中位数正且45/45 pair更快。未修改的9024 runtime control变化+0.486590%属于过程/噪声证据，不归因源码。9023 fixed及9024 runtime均保持原device身份，二者是合法support control，不属于本节44个历史winner配置。

**9023资源风险随决定保留**：VGPR232→251（+8.19%），SGPR lane spill46→48，ISA25976→26328（+1.36%）；LDS78112、private、VGPR spill及AGPR不变。额外live寄存器收窄后续编译器余量，本次没有证明动态occupancy变化。9024 fixed VGPR110及zero spill/LDS71744保持，SGPR58→60，ISA7276→7408（+1.81%）。有限call收益与这些风险分别记录，不把静态资源当作饱和证据。

### 12.6 9030 midpoint：同步证明成立，完整10-winner覆盖拒绝

[八wave cohort分析](../opus_remaining_20261008/large_midpoint/cohort_analysis.json)在clean补采中核准688个完整wave、86个发布组，每wave288条MFMA。774个interior group/tile、6192个相关wave/tile中，arrival spread中位数908 clocks，last MFMA spread900；latest arrival→release仅4，latest barrier8，而all-wave barrier duration中位数356。barrier的wave份额约22.15%，矩阵VMEM issue等待4.14%、LDS issue等待1.89%，与wave到达差一起判断，不能宣称删除barrier就收回22.15%整核时间。组合VMEM/LDS wait中位数76、latest84，latest matrix issue764且stall688；它们不是独立访存返回时间。

异常group8 tile6仍保留：first/last MFMA spread5388/5396 clocks，早四wave组合wait约4830、晚四waveasync issue约5500。没有排除样本。候选只把普通advance的`vmcnt(0)/lgkmcnt(0)+barrier`移到当前前8条MFMA之后，再issue K+2；同步证明要求先publish K+1、退休所有K读者后覆盖K+2，并保留最后transition的C-alias retirement barrier及loops1原prologue barrier。原scale-first prologue、loop/layout/guard、C64资源和output保持，[source/ISA审计](../opus_remaining_20261008/large_midpoint/isa_source_review.json)通过。VGPR203/LDS143360不变，SGPR50→53。

原M16385计划违反M%64，由baseline API先拒绝；保留在`rejected_prepare_revision_misaligned_M/`，随后合法`[16448,65536,K128..640]`五种terminal guard通过。候选library SHA256 `2bf61c8d18bb444fe9c71a9ff8de6cf5d42c1a16cfcb21e0797a8286e9f5efb9`，CO `5cee4b4df6b4083de16cca1839e9b1d3aa2f1c1a006480236c570310b07f1309`。大输出采用256-row分块FP32 reference，8个shared地址、5轮AB/BA、51次完整调用，记录actual module、完成状态及helper SHA。

首个true-winner screen曾有+0.261% ratio/+0.226% paired中位数、5/5更快；预声明的全部10个K1536 actual winner覆盖却不复现：[独立覆盖审计](../opus_remaining_20261008/large_midpoint/coverage_independent_review.json)数值/guard/clean identity通过，但10/10 ratio中位数负、9/10 paired中位数负、0/10同时在AB与BA正，仅4/50 pair更快。等shape GM ratio为−0.220256%、paired为−0.166456%。两个合法非winner control继续保留：K384 ratio−0.550244%、paired−0.587315%、0/5；K16384 ratio−0.933701%、paired−0.069249%、2/5，AB约−2.55%/BA约+0.28%。[全局决定](../opus_remaining_20261008/large_midpoint/global_decision.json)拒绝该midpoint，保留当前9030，没有新增K gate、posthoc subset或重新测试到正。

### 12.7 Small/fine：按自身33配置收口，四候选全部拒绝

以下18-parent表和对应范围从本轮新[small_append](../opus_remaining_20261008/small_final_document_append.md)引用。其逐symbol记录区分自身数值门、局部CU ATT、本轮candidate Event及历史exact-config Event。机制诊断仍有限：9040 startup纯`vmcnt`仅4 clocks；9047的PC `0x2540` 是`vmcnt(0)+lgkmcnt(0)`混合wait，平均约169 clocks（128–196），不能概括为VMEM等待近零。runtime9051本轮全部17个winner中另有3个shape稳定5/5更慢，故早期正screen不支持采用或事后切shape。

small与fine分支本轮按33个有历史winner的实际producer symbol配置收口，映射18个public parent、30个actual ID和278个历史winner。候选性能实测严格为4个配置、21个actual-winner形状：fixed9062 initial matrix-before-scale覆盖M144/M160两个winner且完整调用包含matching split2 reducer；9046与9055 steady ring issue-before-read各测一个自身winner；runtime9051 scale-first覆盖其全部17个winner。四个exact配置的候选均拒绝并保留原selected：fixed9062两项speedup0.996596/0.998854，9046/9055为0.980340/0.991871；9051完整17项为8项median正、8项退、1项平，仅3项全部5轮更快。所有自身signed8repeat、reference、重复及guard和clean sharedpool完整Event审查通过。两个9051早期正代表仅是进入固定覆盖的screen依据，全17 mixed结果作统一拒绝，不按shape新增阈值或重跑寻找正结果。本轮small没有新增source/device entry采用，9042/9053/9054此前已采用runtime B-scale alias保留，9071草稿在build前停止。

证据范围分别保存：7个exact配置各有一个自身有限形状的局部CU ATT，33个配置各自一个actualwinner的signed2repeat/reference/output/workspace guard数值门；它们分别证明局部机制和正确性，均不扩大为全部winner性能覆盖。9055原CU0 code=null/零wave记录排除，CU1补采独立核过身份；9051的32个ATT wave跨多个WG，聚合同PC barrier跨度不用于WG内到达差。该small正式入口集合为35个producer与5个独立reducer，其中33个producer与历史winner相交、两个兼容/runtime producer无历史winner；5个reducer保留机器身份，未声称每个reducer有独立性能实测。各配置的keep依据自身selected source/type、正式FUNC/fullmetadata/descriptor、数值门及存在时的自身历史exact-config Event；共享ring/register/fine机制说明、静态资源或其他配置reject都不算该配置性能。完整配置记录见 [small审查](../opus_remaining_20261008/small_family_review.json)，范围核验见 [small最终审计](../opus_remaining_20261008/small_final_closure_audit.json)。

| public parent | actual配置（同ID按fixedK区分） | 历史winner数 | 自身本轮候选Event形状 | 自身ATT配置 | 自身数值门配置 | 最终状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 9040 | 9040 | 11 | 0 | 1 | 1 | keep自身当前配置 |
| 9041 | 9041, 9050 | 17 | 0 | 0 | 2 | keep自身当前配置 |
| 9042 | 9042, 9071, 9073 | 15 | 0 | 1 | 3 | 保留已采用runtime alias；其余配置keep；9071保留N48 |
| 9043 | 9043 | 2 | 0 | 0 | 1 | keep自身当前配置 |
| 9044 | 9044, 9056 | 26 | 0 | 0 | 2 | keep自身当前配置 |
| 9045 | 9045 | 17 | 0 | 0 | 1 | keep自身当前配置 |
| 9046 | 9046 | 32 | 1 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9047 | 9047 | 27 | 0 | 1 | 1 | keep自身当前配置 |
| 9049 | 9049 | 6 | 0 | 0 | 1 | keep自身当前配置 |
| 9051 | 9051 | 17 | 17 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9052 | 9052, 9070 | 24 | 0 | 0 | 2 | keep自身当前配置 |
| 9053 | 9053, 9072 | 9 | 0 | 0 | 2 | 保留已采用runtime alias；其余配置keep |
| 9054 | 9054 | 3 | 0 | 0 | 1 | 保留已采用runtime alias；其余配置keep |
| 9055 | 9055 | 17 | 1 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9060 | 9060/K0, 9060/K3072, 9060/K7168 | 9 | 0 | 0 | 3 | keep自身当前配置 |
| 9061 | 9061/K0 | 19 | 0 | 0 | 1 | keep自身当前配置 |
| 9062 | 9062/K16384, 9064/K0 | 6 | 2 | 1 | 2 | 保留；拒绝自身有限候选 |
| 9063 | 9063/K0, 9063/K16384, 9065/K0, 9066/K0, 9067/K0, 9068/K0, 9069/K0 | 21 | 0 | 0 | 7 | keep自身当前配置 |

表中0项candidate Event表示该parent本轮保留自身配置，没有冒称该parent全部winner已测或已获得新性能增益；历史exact-config Event另存逐symbol记录。


### 12.8 当前3个selected entry、完整正式identity及API pending

本轮[最终selection](../opus_remaining_20261008/formal_selected/final_selection.json)只有**9020 existing fixed384、9023 existing runtime、9024 existing fixed7168**三个新entry；先前9021及small alias继续保留。原有guard/dispatch阈值沿用，没有把候选推广到support-only或其他配置。[正式CPU identity审计](../opus_remaining_20261008/formal_selected/identity_audit.json)已通过：26个public parent、56个device variant中3 changed/53 unchanged；三个changed的FUNC/full metadata/normalized descriptor精确匹配已测private candidate，所有其余entry精确匹配Oct8，9000/9010/9021冻结。313个generated file、206个build object（202个device object）、202个linked gfx950 bundle及263个linked device entry核对，0 failure。新正式module SHA256 `4be55119596805e3d2967a00c3cbf70129f280094d14d43fbbb2461f91aee1bd`；此identity通过本身不产生新性能结论。

**正式API gate当前pending**：[44-target API计划](../opus_remaining_20261008/formal_selected/api_gate/official_api_plan.json)正在root所有的GPU队列执行，覆盖26个public parent、8次signed/reference/repeat/guards，以及3个selected entry的实际winner/K组和必要panel32/33、M17及grouped尾边界。计划SHA256 `89e7582436378d43584b579b52a2b30324ff44d63c2e943ffaf26750aaba6669`；这里44个API target与12.1的44个历史winner实际配置是两个不同集合，不能混称全配置门禁或536项性能覆盖。API完成、实际加载module核对及root最终应用收口仍待完成，当前不得写成`applied_verified`。

[独立CPU dispatch复核](../opus_remaining_20261008/formal_selected/api_gate/independent_dispatch_review.json)确认44个合法且唯一target/26个parent、逐项branch与CPU计划一致，恰好三个changed symbol；changed target15个，9020=3、9023=8、9024=4，9023 fixed与9024 runtime control unchanged。target32 `[9020,16,256,384]`的purpose称fixed384 minimum tail，但M<1024实际落unchanged runtime branch6，故只算合法control；已有changed fixed384 `[1472,7168,384]`覆盖M%192=128尾行。该purpose问题非阻断，原plan/audit保持，不把它充作changed-entry证据。

本节保存机械建议被最终scope决定取代、两次scale坏revision、9030非法M计划、ownership歧义及所有负样本。当前性能决定基于各配置自身的有限clean Event，仍不声称全支持域获益、全部536 winner已测、绝对MFMA饱和或动态occupancy已知。
