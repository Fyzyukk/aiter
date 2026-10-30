# gfx950 MXFP8 NT 第3步：两轮 CO 优化与56形扩测记录

统计截至 `focused_expansion_known_complete` 的真实六轮复测。
只优化NT，GPU使用0–3。56个shape、558个候选组、3348次测量全部正确，
**16个胜CSV、40个仍落后**；首轮单次筛选的19胜/37落后不作为最终确认。
4wave的GroupM16在多个大M、N=K=7168的shape确认有效，已进入完整421形
Top8及已知赢家复核。NB2在扩测范围没有CSV赢家，不能泛化第二轮的单点收益。

第三步原始待处理集合共85个，包括16个本机已测落后及69个历史落后、
当时尚待本机确认的shape。按原集合去重，当前**8个恢复胜CSV、35个有
本机六轮证据但仍落后、42个尚未补测**，共43/85完成本机六轮验证。
原16个中恢复4个、仍落后12个；原69个中确认胜4个、确认落后23个、待测42个。
本批覆盖原集合39个，结果7胜/32落后；其余46个已有1胜/3落后，42个待测。
56形包含17个额外同家族对照，故其16个CSV赢家不能全部计为原85的恢复数。

两轮共新增36个CO并完成build，公共分支集成其中22个稳定CO。
下面两轮的3胜/3落后等统计保留为各自批次的实验结果，后续扩测结论另列，
旧批次的小幅CSV胜出不覆盖同shape在最新完整批次的落后。

胜CSV沿用固定 `../handoff_data/csv_baseline.csv`：完整六字段shape匹配，
六轮正确，`errRatio<=0.05`，`rotate_args=0`，中位OPUS耗时严格小于CSV。
新增CO的稳定准入还要求在同批次、同shape/GPU的直接parent配对中，
中位数更低且至少4/6轮更快；每个偶/奇轮对反转执行顺序。
不同批次的耗时不混在一起充当配对证据。

| 有限实验轮次 | 新增CO | 已完成shape | 六轮候选全部正确 | 胜CSV shape | 本批次剩余落后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第1轮 | 24，100000–100023 | 17/17 | 294/294 | 0 | 17 |
| 第2轮 | 12，100024–100035 | 6/6 | 49/49 | 3 | 3 |

第一轮8个4wave点为4个原parent各测试GroupM4/8，保持runtimeK和Unroll2。
另16个B/overlap点覆盖XCD切换、NB3以及BD1，每点只改单变量，普通B
与overlap按17个完整基础traits严格配对。GroupM4/8先在大M、长K对照
中形成稳定收益，因此第二轮才测试合法的GroupM16；C++不允许GroupM32。

第二轮为2个GroupM16、4个Panel32、NB2/BD1普通与overlap一对、
NB3/BD1一对、BD3一对，共12点。4wave继续保持runtimeK、Unroll2。
窄N的旧JIT赢家kid48096及其完整几何仍作为对照，未用仅CO子集的
最佳值替代整体最佳。

第二轮各shape的最佳六轮中位值如下。CSV收益率定义为 `CSV/us-1`。

| M×N×K | 最佳CO | OPUS中位µs | CSV µs | 结果 |
| --- | ---: | ---: | ---: | --- |
| 4096×1536×7168 | 100031 | 57.0254 | 57.7197 | 胜CSV，+1.22% |
| 6144×6144×7168 | 100003 | 232.4311 | 233.6307 | 胜CSV，+0.52% |
| 8192×7168×7168 | 100024 | 326.6318 | 333.3864 | 胜CSV，+2.07% |
| 6144×7168×16384 | 100000 | 547.6027 | 534.8130 | 仍慢2.39% |
| 8192×7168×16384 | 100024 | 726.6057 | 705.6902 | 仍慢2.96% |
| 32768×7168×16384 | 100001 | 2685.0689 | 2679.1350 | 仍慢0.22% |

五个重点CO对应三个CSV shape，不应重复计为五个shape赢家。

| CO | 来源与直接parent | 已确认的收益 |
| --- | --- | --- |
| 100003 | 第1轮，389→GroupM8，Panel64、WideOutput=true | 第一轮在6144×6144×7168为235.7626对240.8535µs，6/6快于parent；第二轮该相同CO复测232.4311µs胜CSV |
| 100024 | 第2轮，100001→GroupM16，Panel64、WideOutput=false | 8192×7168×7168为326.6318对338.8661µs，6/6快于parent；parent/us-1为3.75% |
| 100025 | 第2轮，100003→GroupM16，Panel64、WideOutput=true | 同shape为329.6120对338.5555µs，6/6快于parent；parent/us-1为2.71% |
| 100030 | 第2轮，100022→NB2，普通B，BD1 | 4096×1536×7168为57.2494对59.8631µs，6/6快于parent；parent/us-1为4.57% |
| 100031 | 第2轮，100023→NB2，overlap，BD1/PublishM0 | 同shape为57.0254对59.4065µs，6/6快于parent；parent/us-1为4.18% |

100003的稳定直接parent证据来自第一轮，第二轮389未被同步复测。
发布记录分别保存 `stable_basis_round=1` 和 `current_timing_round=2`；
240.8535与232.4311不能作为第二轮同批配对速度比。100024/25在目标
shape的最大耗时329.9568/330.6908µs也低于CSV。100003及NB2普通/overlap
按中位数胜CSV，其全部六轮并非都低于CSV。

NB2收益对应实际build report的资源变化：

| 配置对照 | LDS字节 | VGPR | SGPR | AGPR | 实际B_AHEAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100022/23，NB4、BD1 | 74752 | 208 | 45 | 64 | 2 |
| 100030/31，NB2、BD1 | 41984 | 184 | 42 | 64 | 2 |

该几何的B_AHEAD保持2，LDS下降32768字节，VGPR下降24，因此这是保持
实际B预取距离的stage比较。两者build report均无VGPR/SGPR spill，
private segment为0。GroupM8→16对照的LDS均为152064字节、VGPR456、
AGPR212，normal/wide的SGPR分别102/104，均无spill/scratch；已确认的
收益没有伴随这些资源数量下降。

Panel32的100026–100029均未形成稳定准入。100027在
6144×6144×7168的中位233.1772µs虽然低于CSV233.6307µs，但只在2/6轮
快于直接parent；LDS152064→143616字节不足以证明稳定收益，因此不发布。
第一轮XCD切换100008–100013也未入选。第二轮BD3的100034/35将实际
B_AHEAD从4增到6，但中位60.2806/59.8763µs均慢于对应parent，只有3/6、
1/6轮更快，未入选。NB3/BD1的100032/33有有限parent收益，仍未胜CSV。
这些结论只对应已测几何和shape，不能外推为该轴普遍无效。

公共分支已集成22个稳定CO，现有802个CO，JIT base固定780，registry
为原48212个ID加22个稀疏ID。原780条manifest/suffix及原48212个完整
名字、traits和dataclass配置逐项保持不变；新ID沿用实验ID，不重新压缩。
准入按稳定parent收益，CSV胜数另计，因此22个CO并非22个CSV赢家。
共56个已验证稳定candidate×shape、13个去重shape，其中3个胜CSV。

发布时的family范围另有**1338个CPU合法、尚待GPU验证的candidate×shape**。
全部1394条候选scope覆盖166个去重shape。这些pending项不计正确性
完成数或CSV胜数；已实测不稳定的同一CO/shape不会通过扩展重新加入。
窄N、FixedK7168的CO可扩到N768/1536/2048；4wave K7168可扩到
N6144/7168，其他已经有稳定收益的K按同N/K扩展，均检查真实host guards。

下表为已发布22-CO manifest中五个重点CO的实际scope，
不是较小的“仅CSV赢家”审阅提案范围。validated/pending是发布时receipt的
状态快照，后续56形GPU测量见下面的当前复测结果；不重写冻结发布证据。

| CO | 合法scope条数 | 已验证稳定条数 | pending条数 | 实际N/K范围 |
| --- | ---: | ---: | ---: | --- |
| 100003 | 55 | 7 | 48 | N6144/K7168、N7168/K7168、N7168/K16384 |
| 100024 | 55 | 3 | 52 | 同上；6144×6144×7168实测不稳定，已排除 |
| 100025 | 55 | 3 | 52 | 同上；6144×6144×7168实测不稳定，已排除 |
| 100030 | 79 | 1 | 78 | K7168，N768/1536/2048 |
| 100031 | 79 | 1 | 78 | K7168，N768/1536/2048 |

完整M列表与每条validated/pending状态以
`co_rounds_family_integration/new_kernel_scopes.csv`及公共manifest为准。
NB2跨N扩展在发布时尚未GPU验证，现已有24个shape的六轮对照；
来源parent100022/23的原N1536限制仍完整
保留在隔离快照中，独立发布child的新增host scope不改变旧parent身份。

22个新配置沿用原有合法几何；Top8成本只取几何，不取配置数量，
不会挤掉原Top8候选。实际tuner纯函数确认三个CSV目标的五个重点CO
都被Top8保留。历史已知赢家的原有fallback需要仍须保留。
56形定向扩测已完成，主测五个CSV获益CO、全部已知赢家及必要JIT。
冻结v2输入为 `focused_expansion_known_complete`，56形、109条指定几何、
2716个初筛候选×shape、986个唯一候选ID。225条新配置/已知赢家强制
保留项全部测到且正确。六轮计划使用Top5整体、Top2每管线、所有已知赢家
及必要直接parent，共558个候选组；只采用本计划本批次的六轮，不补入旧轮。

| 56形本轮分组 | 已完成 | 胜CSV | 仍落后 |
| --- | ---: | ---: | ---: |
| N1536/K7168 | 16/16 | 3 | 13 |
| N2048/K7168 | 4/4 | 2 | 2 |
| N6144/K7168 | 16/16 | 1 | 15 |
| N7168/K7168 | 16/16 | 8 | 8 |
| N768/K7168 | 4/4 | 2 | 2 |
| M≥8192子集 | 44/44 | 11 | 33 |

16个中位CSV赢家中，13个的最佳候选六轮耗时均低于CSV；3个仅中位数
胜出。按用户要求中位OPUS耗时低于保存CSV即算胜，但小幅点在完整421
仍需保留并复核。此前已测18个本轮6胜/12落后，包含4个恢复、2个保持、
2个旧胜转落后、10个持续落后；首次当前GPU确认38个中10胜/28落后。

| 新CO方向 | 六轮正确shape | 中位胜CSV shape | 快于同批直接parent | 6/6快于parent |
| --- | ---: | ---: | ---: | ---: |
| 100003，GM8 wide | 30 | 1 | 15/30 | 10/30 |
| 100024，GM16 normal | 31 | 7 | 25/29 | 20/29 |
| 100025，GM16 wide | 31 | 8 | 24/29 | 21/29 |
| 100030，NB2普通B | 24 | 0 | 3/24 | 2/24 |
| 100031，NB2 overlap | 24 | 0 | 4/24 | 2/24 |

100024/25各有2个扩展shape的精确parent未测，父配置不在已发布合法范围，
这些点仅统计正确性和CSV比较，不混用旧批parent中位数或单轮耗时。
这两条GM16方向在可比shape上的parent收益中位数为2.600%/2.706%。
两ID合计为16个shape的最佳候选，其中7个为CSV赢家，避免按配置重复计数。

| M×N×K | 本轮最佳候选 | OPUS中位µs | CSV µs | 六轮CSV情况 |
| --- | ---: | ---: | ---: | --- |
| 8192×7168×7168 | 100025 | 329.1256 | 333.3864 | 6/6胜 |
| 28672×7168×7168 | 100025 | 1046.2612 | 1091.5874 | 6/6胜 |
| 49152×7168×7168 | 100024 | 1746.3708 | 1823.0221 | 6/6胜 |
| 32768×7168×7168 | 100025 | 1206.2768 | 1207.2079 | 中位胜，未全轮胜 |

两处此前CSV小幅赢家本轮转为落后，完整421必须保留原候选：

- 4096×1536×7168：100031为57.9740µs，CSV57.7197µs，慢0.441%；
  仍比同批直接parent100023的60.3944µs快4.175%，6/6更快，NB2单点parent
  优化仍成立，但本轮不能计CSV赢家。
- 6144×6144×7168：整体最佳17910为239.2027µs，CSV233.6307µs，慢2.385%；
  原100003本轮245.2619µs，比同批parent389慢0.670%，仅2/6更快，
  不沿用第二轮232.4311µs的CSV胜出结论。

完整421首轮已启动 `step4_full421_top8`，继续只用GPU0–3，Top8加已知赢家
检查。原85中的42个未补测shape保持pending，直到本次完整验证获得当前
正确性和性能证据。

复现不需要提交隔离repo里的额外核心源码。标准
`csrc/opus_gemm/gen_co/build_co.py`直接读取公共manifest中的完整traits、
body_args、flags和image_suffixes，可重生相同body、extern symbol和文件名；
支持scope不进入device stub，发布镜像SHA已逐项匹配source build。
两轮使用同一pin compiler wrapper，路径和版本见 `TOOLCHAIN_EVIDENCE.md`。
`co_publication_reproducibility.json`保存22个标准stub文本/SHA、资源、
镜像SHA及精确rebuild命令。本项只做CPU代码生成核对，未重新编译，
不同编译器不保证相同二进制或性能。

需保留公共common/manifest/22个镜像，两轮selection plan、独立build
report、六轮report/raw和发布receipt。
`gen_co_kernels_gfx950.py`的CLI已修复：重新生成原表后，严格确认原780
entries/defaults/flags/名字后缀完整一致，再保留现存稀疏CO、host scope、
`jit_kid_base`和集成元数据。原base不同会拒绝写入；纯
`generate_document`与`validate_document`未改。临时文件验证当前802表
及无扩展780表均完全保持，见 `co_manifest_generator_cpu_report.json`。
新增配置继续以已发布manifest及保存的实验计划/receipt为source of truth。
当前 `integrate_co_rounds.py`也限定从原780公共基线进行一次发布；
发布后新增CO或调整scope需要增量机制和新快照，不能修改旧实验证据。

证据入口：

- [第1轮六轮报告](co_round1/repeat_report/report.json)；[第2轮六轮报告](co_round2/repeat_report/report.json)。
- [22-CO发布receipt](co_rounds_family_integration/receipt.json)；[逐shape范围](co_rounds_family_integration/new_kernel_scopes.csv)。
- [Top8及范围兼容性](co_rounds_family_compatibility.json)；[标准builder复现核对](co_publication_reproducibility.json)。
- [多轮CPU筛选fixture](co_rounds_cpu_fixture_report.json)；[冻结56形v2计划](focused_expansion_known_complete/receipt.json)。
- [56形真实六轮严格报告](focused_result_analysis/repeat/report.json)；[同批parent比较](focused_result_analysis/repeat/direction_comparisons.csv)。
- [原85去重进展](stage3_original85_status/report.json)；[8个恢复](stage3_original85_status/recovered_wins.csv)；[35个落后](stage3_original85_status/remaining_losses.csv)；[42个待测](stage3_original85_status/pending_shapes.csv)。
