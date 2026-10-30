最新正式结果已合并整理：**338领先 / 83落后**。远端54bef351的236领先全部包含在原本地337领先，101本地独有领先、84共同落后；421六字段集合和基线原字节SHA完全一致。最后第18轮只处理共同落后84项：9096条性能测量全正确，最终接受8742条；6项GPU5波动换卡补做整组六轮并替换，原354条只归档。20项稳定候选提升，其中新interleave CO101312使2项快1.16%/1.69%但仍落后；新增1领先为原control复测越线。原337领先逐字段保留，最后领先338已独立汇总。剩余落后中位3.3152%、最大8.8026%。公共JIT、421身份、旧48261配置和新CO正确性全部核验，registry48262/JIT780；controller_verified退出0。最终入口：[合并结果](../final_aligned_results_20261030/README.md)，领先CSV：[leading_tuned.csv](../final_aligned_results_20261030/leading_tuned.csv)。本次最后一轮已结束，不启动第19轮。

以下为历史执行记录，当前状态以上段和workflow_state.json为准。

当前按用户要求已停止继续优化，结果已整理：../optimization_summary_20261030/README.md。最新337领先/84落后，落后中位3.52%、最大13.25%；pin_agpr54、pingpong_lds20、B-overlap4、wave_interleave3、B-VGPR3。第16/17轮共9694条正确，无新增稳定新CO，均退出0。最后一轮54项新六轮，367历史行逐字段保持；当前421配置、baseline六字段、公共模块hash及CO清单已核验。当前registry48261/JIT780；第5~17轮27个整合CO完整清单保存，shape_evidence421.csv记录每项最后复测来源。当前无本任务优化进程。第18轮未启动。构建已限定MXFP8 family，未引入a16w16路由。

第十七轮已正式完成：stage17_pin_invariants54_20261030，4634条性能测量正确（初筛1940+最终2694）；偏移不变量优化无稳定新赢家，公共JIT和421身份核验完成。

第十六轮已正式完成：337领先/84落后，54项pin_agpr新六轮，367历史行逐字段保持。5060条性能记录全正确（初筛1940+最终3120），无新增稳定kernel收益和新CO，registry48261/JIT780。提前未来matrix请求和缩短tail staging未达到稳定整合标准；CPU数据流、旧硬件同步、公开JIT与完整421身份均通过。构建已限定MXFP8家族，实验41routes/公共156routes无a16w16配置。结果 stage16_pin_compute54_20261030/result/public_result_zh.md；失败方向保留。继续pin_agpr偏移不变量优化。

第十五轮已正式完成：337领先/84落后。55项pin_agpr落后做新的六轮，366项历史行逐字段保留；5006条实验性能记录全部正确，controller退出0。两项真实kernel收益：M1920/N7168/K3072耗时下降0.814%，M12288/N7168/K4096下降0.487%；成对中位收益0.704%/0.587%，5/6和6/6轮更快。同配置unroll8对照也分别5/6和6/6更快，成对收益1.134%/1.030%。两项均仍落后；唯一新增领先为原control复测越线。新CO101011/101020已整合，registry48261、JIT780。最新落后分类pin_agpr54、pingpong_lds20、prefetch_b_vgpr_overlap4、wave_interleave3、prefetch_b_vgpr3；继续优先优化pin_agpr。结果 stage15_pin55_20261030/result/public_result_zh.md。

第十四轮已正式完成：336领先/85落后，66项新六轮、5446条正确，355历史行逐字段保持，controller退出0，公开421配置及4个新CO正确性验证通过。6个shape真实kernel收益0.60%~3.98%，都仍落后；新CO100901/100906/100908/100911整合，公共registry48259。scale refill从inner unroll loop移到cache-boundary chunks外层，Panel16 spill从785~795降0~2，但没有Panel16版本成为公开最优。结果 `stage14_chunk_scale_20261030/result/public_result_zh.md`；分类pipeline_priority85.csv。


第十三轮已正式完成：336领先/85落后，69项新六轮、5622条正确，352历史行逐字段保持，controller退出0，公开421配置/新CO正确性均通过。新CO100808整合，两个shape真正改kernel收益：M32768/N6144/K7168快3.4941%仍落后；M40960/N7168/K3072快2.4438%并越线；均6/6快，相比同panel32旧body也分别1.3028%/1.4964% paired收益，均6/6。另2项新领先为原control复测越线。公共registry48255、JIT base780；旧48254配置保持。结果 `stage13_local_scale69_20261030/result/public_result_zh.md`。


第十二轮已正式完成：333领先/88落后；70项新六轮、4546条正确，351历史行逐字段保持、完整88项落后表验证，controller退出0。缩小scale缓存没有新最优CO，新增1项为原100083在M32768/N7168/K4096复测越线。12个编译均ready，Panel16有785~832 VGPR spill，诊断与慢候选记录保留。4个原panel64仅作对照未集成。结果 `stage12_scale70_20261030/result/public_result_zh.md`，registry48254、JIT base780。


最新正式结果：第十一轮已完整完成，332领先/89落后。`stage11_unroll76_20261030/result/public_result_zh.md` 和public_best_opus421.csv为最新结果，public_env.sh为公开运行环境。76个宽N项新的最终六轮、3946条性能记录全部正确，345个此前行逐字段保持，公开421项身份/artifact membership、3个新CO独立正确性均通过，controller退出0。新CO100602/100606/100610已整合，公共registry48254、JIT base780；全部旧48251配置保留。

本次第九至十一轮：323/98→332/89，共7132条性能记录正确。真实kernel收益5个shape（1.37%~4.67%耗时下降），4个新CO整合；新增9个领先中3项靠新kernel、6项原control复测越线，归因分别保存。第十一轮主要有效方向为unroll8、policy0：M20480/N7168/K3072快3.408%、M65536/N7168/K3072快4.666%，两者均6/6快并越线；另外2个宽Nshape新kernel提升约1.37%但仍落后。第九轮M1536/N768/K7168快2.079%并越线。只tune OPUS NT，FlyDSL固定CSV，实际物理0/3/5/6。当前这批测试已全部完成。


第十轮已正式完成：326领先/95落后。20个窄Nshape的新六轮、1516条记录全部正确，401历史行逐字段不变，完整95项落后表通过，controller退出0，公开421项身份/缓存校验完成。新增1项为原48078在M14336/N1536/K7168复测越线，无新CO整合。提前B预取的44组成对对照中2组有0.6~0.8%的单因素收益，但未成为公开最优。结果 `stage10_prefetch20_20261030/result/public_result_zh.md`，kernel_findings_zh.md记录实际收益归因。


第九轮已正式完成：325领先/96落后。定向22项做六轮，399项此前行逐字段保持，完整剩余96项表已核验。1670条记录正确、pending0、controller退出0。新CO100413整合并公开正确性通过：M1536/N768/K7168，六轮6/6更快，耗时下降2.0786%、成对中位speedup2.0425%，原control仍落后，靠新kernel越线。另1项M10240/N1536/K7168是原control复测越线。3项stable收益中只有1项属于新kernel，另2项为既有候选。公共registry48251、JIT base780。结果 `stage9_compact22_20261030/result/public_result_zh.md`，环境public_env.sh。


第八轮已完成：323领先/98落后。100项全部新的六轮、1786条性能记录正确、公开421项编译和identity检查完成，registry48250、JIT base780未变。新增2项均原control自身复测越线；20个长K新CO无最终胜者，没有新CO集成。结果 `stage8_longk100_20261030/result/public_result_zh.md`，receipt为completion_receipt.json。

第七轮已正式完成：`stage7_kernel103_20261030/result/public_result_zh.md`，321领先/100落后。原318领先保留，原103项全部新六轮，3项越线：2项靠更快候选、1项原control自身也越线。6项稳定收益，中位1.68%、最大2.75%；4个新kernel CO成为4个shape公开最优，原48246配置完整保留，当前48250，JIT base780。公开缓存及4个新CO独立公开正确性均通过。3688条性能测量全部正确，最终1866条；无待确认。初筛明显慢候选380个shape-candidate组合已保存，最终最优没有>=20%落后项。kernel_findings_zh.md记录有效和未选方向。controller正常退出0。

第六轮已正式完成：`stage6_optimize117_20261030/result/public_result_zh.md`，318领先/103落后。原304领先保留，117落后全部新的四卡最终六轮；14越线，其中8项原control仍落后、靠候选越线，6项原control本轮自身也越线。52项稳定候选收益，中位耗时下降1.09%、最大6.04%；其中42项更快但仍落后。6个新CO已整合，成为25个shape最优，原48240配置完整保留，当前48246，JIT base780。公开CSV、421项JIT缓存及6个新CO独立公开正确性检查均完成。35742条实验/确认性能测量全部正确；最终117共2496条，严格审计通过。完整receipt为completion_receipt.json。

新CO与最终四卡环境的ROCR为 `1,0,6,7`，HIP逻辑0/1/2/3经PCI核对实际对应物理0/3/6/5。搜索沿用ROCR `4,5,6,7`，只使用逻辑2/3，实际物理6/5。每条raw都记录实际PCI和物理卡号，不能用ROCR序号推断rocm-smi卡号。

上一轮正式结果：304领先/117落后；原160项全部新最终六轮，43项越线（39项靠候选，4项原control也越线），105项稳定候选收益。6个新CO已集成，20个shape选中。公用结果：stage5_optimize160_20261030/result/public_result_zh.md；CSV：public_best_opus421.csv；运行环境：../public_env.sh。第四步旧261/160报告和合并前manifest已保留。

此前第五轮优化剩余160项已完成，当前第六轮继续优化剩余117项。第四步261/160与第五轮304/117正式结果均保留为参照。

# 当前执行记录（接续用）

## 最新最终结果：第四步已完整完成

最终目录 `step4_final_recovery_gpu56_v2_20261030/full421_final_report/`。
421shape全部六轮确认：261中位领先 / 160落后，244每轮领先；无待确认、无正确性失败。
5732个候选组合、34392条最终raw，225known全部保留；严格审计通过，controller退出0。
仅物理GPU5/6，PCI映射核对；保留207完整shape、214重测或补候选整shape复测。
此前后96初筛29/67，最终59/37；其中44明显偏慢项最终25领先/19落后，均慢不到20%。
完整初筛104个明显偏慢shape均已核对，最终最优全部慢不到20%；原104个初筛最优候选有62自身仍慢至少20%，最终采用更快候选。
剩余160项落后中位3.19%，最大16.90%。
人读报告 `final_result_zh.md`，全421最优 `best_opus.csv`，落后排序 `remaining_losses_sorted.csv`。
最终阶段GPU4/7外部任务占用已核实；原受占用影响的失败分片不进入最终数据。

## 完成前的执行记录：最终六轮恢复到物理GPU5、GPU6

421个Top8初筛和37个历史赢家几何补测均已结束并退出0。
完整Top8初筛181领先/240落后；合并focused与补测候选后的初筛194领先/227落后。
以上仍为初筛，受到设备占用影响的计时不能作为最终稳定结论。
原最终六轮分片0在物理GPU7受到其他任务占用、已终止；分片1实际物理GPU4发生OOM；
分片2/3在实际物理GPU6/5完成210shape。原controller退出1，所有原始文件保留。
恢复目录 `step4_final_recovery_gpu56_v2_20261030/`，controller `recover_final_gpu56.py --run`。
已核对PCI实际映射：HIP逻辑0/1/2/3=rocm-smi物理7/4/6/5；原先按4/5/6/7顺序标记错误。
仅在物理5/6运行；运行时核对PCI并写入raw。
原210中3个shape需加回初筛最优候选后整shape重测，保留207完整shape、16956条真实本次final raw。
恢复214shape、2906候选；最终扩展计划421shape/5732候选，225known候选全部保留。
所有104个初筛慢至少20%的shape的原最优已加入最终计划；补回6个被后续筛选挤掉的候选。
恢复结束后合并本次final已保留的完整shape和新测完整shape，严格审计，再生成104项偏离核对和完整421最终报告。
旧step1/2/3轮次仅用于选择候选，不进入最终统计。
先看恢复目录phase/exit_code/segments/gpu2,gpu3日志与raw；原运行目录只用于初筛与历史依据。
初次准备目录 `step4_final_recovery_gpu56_20261030/` 未通过计数断言，仅保留准备失败证据，不作为输入。

## 原执行：用户指定 GPU4–7，只 tune OPUS

用户已明确改用后4张物理GPU4、5、6、7，后续只测OPUS，FlyDSL仅使用保存CSV基线。实际运行目录为 `step4_resume_gpu47_20261030/`；controller `run_step4_resume_gpu47.sh` 已启动。ROCR物理4–7，HIP逻辑0–3，未重新JIT，复用已核对hash的26322配置编译产物。首轮保留前100shape完整profile，再对其余321shape做Top8/opus/mp4/21iters/batch4。之后自动接37个Top0赢家补测、完整421的最终六轮101iters和严格最终报告。恢复时优先查看该新目录phase/exit_code及full421_top8/progress.jsonl，旧目录仅作历史记录。

CPU已校验候选分片和物理映射4–7；最终raw将同时记录logical与physical GPU IDs，最终audit要求物理4–7。

## 用户指定的待核对事项：初筛明显偏慢计时（完整初筛104项）

用户要求将明显偏离项记录下来，后续核对。清单与机器记录位于
`step4_screen208_corrected_losses80_20261030/pending_outlier_checks_zh.md` 和
`step4_screen208_corrected_losses80_20261030/pending_outlier_checks.json`。
当前均为 pending，需在新一批最终六轮 raw 完成并审计后逐项核对。
最初208shape快照的5项已保留；272shape快照共33项，304shape快照扩展到49项，
筛查条件为正确设备基线下单轮慢至少20%，不据此判定稳定退化或原因。
目标 M/N/K：4096/7168/7168、1408/7168/2048、1536/7168/2048、
1472/7168/2048、1792/7168/2048。确认最终计划保留这5项初筛最优，
以及第一项历史六轮最优 kid100003（178.6429us，仅慢2.56%；本次同候选1379.1976us）。
若最终结果仍明显偏离，检查设备负载、频率、计时与候选覆盖并记录证据。
清单包含逐项状态，旧轮次不能替代新最终六轮。

208shape快照汇总已纠正为128领先/80落后；先前108/100的临时汇总仅用M/N/K匹配，
误用了gfx1250基线导致20项误判。之后必须使用完整六字段匹配gfx950/256 CU。
完整纠正明细及初筛快照在上述目录，不作为最终raw输入。


## 2026-10-30T02:19:03.933751+00:00 恢复核对（覆盖下文旧的运行中状态）

第四步全量任务已停止；188/421进度，187正确OPUS，单轮119胜CSV/68落后/1无有效结果。完整profile只有100shape，续跑输入为其余321个，位于 `recovery_20261030T021903Z/resume_shapes_321.csv`。37个补测和最终六轮未启动。原主进程和worker均不在当前容器；GPU1–3为当前容器外进程，0与4–7空闲。用户已提出扩到8卡，但续跑尚未启动。


用户授权完成四步 NT gfx950 MXFP8 优化，只使用 GPU 0–3；不优化 TT/NN/TN。
仓库 `/root/workspace/aiter-fyz-10-29-aiter`，分支 `fyz/10-29-aiter`，
原始提交 `39cdf47d864c463b08f0cf9be146595db755578d`。
先读上一级 `handoff_4steps_zh.md`。

截至本记录：公共源码已冻结JIT base780，manifest已整合22稳定新CO（现802 CO）。
原0–48211全部身份保持；新CO镜像均已复制并核对hash，来源receipt见
`co_rounds_family_integration/receipt.json`（status=applied）。

## 已完成

- 第1步32shape及六轮复测：20 median胜CSV/12落后。
- 第2步46shape及六轮复测：42 median胜CSV/4落后；38旧CO赢家恢复36。
- 前两步唯一76shape：60胜/16落后；历史另69落后尚待本轮全量确认。
- 第3步代表19shape，指定几何Top0/101iters：63965候选全部正确；
  单轮4胜/15落后。原profile/tuned及profile_confirmable均完成，无GPU恢复。
- 第3步旧配置六轮：479候选/2874测量全正确；2胜/17落后。
  `step3_existing_round1/repeat_report/report.json`，raw在`step3_existing_repeats/gpu*.jsonl`。
- 第3步第1轮有限CO：`co_round1`隔离副本，24新CO100000–100023全ready；
  8 GroupM4/8、6 XCD、6 NB3、4实际BD，原780+JIT全部身份保持。
  17目标shape定向GPU+6轮294候选/1764测量全正确，但0胜CSV/17落后。
  注意这17只测B/overlap/4wave指定几何，不能代替旧m-repeat总榜。
  `co_round1/repeat_report/report.json`，raw=`co_round1/repeats/gpu*.jsonl`。
- `co_round1_integration_review/receipt.json`只CPU提案，未apply：16新CO对13shape
  有稳定parent收益，45稳定shape/candidate pairs，但尚未新增CSV胜。
  GroupM把长K差距从6–8%降至约2.1–2.5%，M32768/N7168/K16384只差0.51%。
  关闭XCD的B/overlap普遍退化，应停止该方向。
- 宽N1664×65536×1536：新增GN32/Boundedfalse JIT48193单轮167.46us，
  六轮179.88us仍胜CSV183.49；该shape旧panel16 JIT20381六轮176.39us更快。

## 最新用户 steering

用户问“是重新优化到再有效果更多就全量tune，还是已经要全量tune了”。
已明确答：仍第3步，收益不足继续针对性优化，稳定方向扩测同类剩余落后，
更新落后集合后才第4步421 Top8。因此不要只因第1轮CO结束就立即全量。

第3步第2轮已完成：隔离clone追加100024–100035，现816 CO；49候选/294次六轮
101iters测量全部正确，6 shape中3胜CSV、3仍落后。稳定CSV方向是NB2/BD1
100030/100031与4wave GM16 100024/100025，加第1轮GM8 wide 100003。
4096×1536×7168最佳57.02542us(CSV57.7197)；6144×6144×7168最佳232.43107us
(CSV233.6307)；8192×7168×7168最佳326.63178us(CSV333.3864)。三长K16384仍落后。
Panel32与BD3没有可靠新增收益；不把单轮或parent提升计作CSV胜。

目前正在准备family定向扩测，确认这5个CSV获益CO在同类shape的收益；随后第4步421
Top8。已一次公共整合两轮22稳定CO，保留合法长K等parent收益配置；GPU扩测主要
5个CSV获益CO。`co_rounds_family_review`为22CO CPU提案，
`co_rounds_csv_family_review`为5CO提案；实际apply见`co_rounds_family_integration`。
`co_rounds_family_compatibility.json`确认22新增几何不改变Top8几何集合，旧候选不会
被挤掉，3新CSV shape的新CO均保留。

`focused_expansion`已按最终public准备：56shape=原落后12+历史pending27+同类17；
18已有本轮六轮、38新待测，44个M≥8192；108geometry行、2704candidate-shape、
986uniqueunion、212forced新/known行，镜像/身份/真实device guards核对0失败。
统一JIT session76332已退出0，日志`full421_union_compile.log`，构建803.3秒，完成于
2026-10-29 17:09:25 UTC。Top8完整421+旧37fallback+focused56，union26322kids
（706CO/25616JIT），receipt=`full421_union/compile_receipt.json`。
第三轮focused v2 GPU筛选与六轮均完成（筛选42695、repeat32784均退出0）。
实际目录`focused_expansion_known_complete`：56shape、109geometry、2716candidate-shape、
986unique。v2比v1补1个known geometry及13个曾best候选（含48096），225 forced known
完整保留；`v1_v2_union_difference.json`证实v2新增编译ID=0，旧hash输入未改变。
运行日志`tune.log`、结果`profile.csv`/`progress.jsonl`、shell最终写`exit_code`。
编译成功的严格guard后才启动GPU；等待shell38906已终止并被42695替代，不要重启。
GPU筛选结束后prepare_final_repeat.py --shapes v2/shapes.csv --known-csv v2/known_winners.csv
--known-kids 389 100001 100003 100022 100023 100024 100025 100030 100031，结合旧完整
repeats和公共family receipt，完成558候选/3348次六轮101iters，全部正确；225known无遗漏。
初筛56=19CSV/37loss；严格本批六轮56=16CSV/40loss，0单轮正式best、0待确认。
官方报告=`focused_expansion_known_complete/repeat_report/report.json`。
原85集合本轮43已六轮验证：8恢复CSV/35确认loss/42历史pending；16原observed=4/12，
69原historical=4/23/42。`step3_initial85_progress/latest_shape_batch_report.json`，focused
覆盖原集合39按最新整批覆盖旧报告；不能沿用4096×1536和6144×6144上批小幅胜。
GM16扩展稳定有效（7CSV实例），NB2未能泛化此前单点CSV收益。第3步定向已有效，
已进入第4步完整421；其余42 pending在全量阶段确认，不需等待再一轮有限CO。

第4步当前运行：`run_full421.sh`，exec session97950，tuner pid1399270，目录
`step4_full421_top8`，Top8/both/mp4/21iters/timeout7200，不传局部几何，GPU0–3。
完整结束后run_winner_fallback.sh做旧37shape/geometry Top0补测；全部788kid已编译。
最终prepare_final_repeat使用full/fallback/focused profiles+225known，做完整421六轮；
正式finish_workflow必须--require-repeat-winners --finalize，repeat仅传本次final raw，
确保421最终best都本次六轮，不让未复测单轮进入正式CSV统计。

Agent `/root/stage3_analysis`准备focused_expansion的shapes/geometries/known winners；
`/root/winner_audit`CPU审阅22CO公共整合兼容性；`/root/full421_reporting`支持定向
六轮和Top8+fallback+focused一次JIT union。Agent不运行GPU/JIT，root统一调度。

## 后续工具与计划

- `run_repeat_shards.py --plan PLAN --out-dir FRESH`：source env.sh后使用GPU0–3，
  4份按成本分片，六轮101iters交替，先CPU完整身份dry-run。
- `prepare_existing_repeat.py`与`summarize_existing_repeat.py`：在实验副本同路径的
  helper中运行以使ROOT指向clone；全部新CO和seed parent强制加入复测。
- `integrate_co_rounds.py --check-only/--apply --expand-scope family`：多轮raw6轮、
  至少4/6胜immediate parent、原48212身份、源码镜像hash、备份/rollback。公共一次
  整合后JIT base固定780；新CO为独立镜像，实验parent无需运行时存在。
- `finish_workflow.py`：多profile/progress/repeat按完整shape+name合并；完整六轮
  中位覆盖单轮；`--current-registry`过滤未集成/名字错误/scope外实验候选。
- `prepare_full421_union.py`：原Top8合计838597候选、26300不同kid。
  历史37个赢家fallback所有788不同kid已在Top8 union内，无额外JIT需求。
  新CO集成后重新生成或直接`--compile`一次JIT，AITER_MAX_JOBS96。
- `run_full421.sh`：已准备，尚未执行。421完全相同shape按估计耗时降序，
  Top8、both、mp4、首轮21iters、timeout7200；之后最终复测101iters6轮。
- `top8_current_audit/`：前两步76完整配置全部进Top8，60赢家无裁剪。
  历史37赢家有旧几何裁剪，winner_fallback_shapes/geometries.csv供Top0补测。
  新CO收益后重跑audit_progress.py top8检查当前赢家。
- `run_winner_fallback.sh`：37历史赢家几何Top0定向/21iters，root执行；
  完整421首轮不传局部几何文件。
- `prepare_final_repeat.py`：从完整421+fallback全部profile，保留globalTop5、
  每pipelineTop2、远端完整名赢家、所有历史本轮六轮最佳及新CO赢家；
  仅CPU，自动过滤未集成实验ID。配run_repeat_shards做421六轮最终确认。
- `finish_workflow.py --finalize --current-registry`：完整Top8/fallback覆盖和每shape
  正确OPUS验证后才mark第4步完成；输出best_opus.csv、remaining_losses.csv与中文报告。

持续用commentary汇报当前步、完成shape数、胜CSV/落后数；对状态问题简短回答后继续。
胜负固定使用handoff_data/csv_baseline.csv的完整六字段，不用同轮FlyDSL最佳或best_libtype。
