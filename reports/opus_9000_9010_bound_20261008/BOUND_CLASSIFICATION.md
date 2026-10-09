# 全系列 bound 分类：按实际配置、形状和阶段

原有 **26 个公开 parent / 56 个 device entry** 均已列出，含47个历史winner producer配置、四个support producer和五个reducer。本次新增候选计入后为 **28 parent / 58 entry**。745为历史winner归属数，不是本次745次性能重测。

最新依据为[带宽文档第4.2节](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md:789)。结论是：长K大tile更偏compute/issue与供数混合，register队列多偏VMEM请求/依赖，LDS/fine多偏local LDS依赖与同步；短K常有明显启动与输出固定成本。**现有证据没有确认任何整个系列属于纯HBM bandwidth bound或dispatch bound。**

memory latency在表中标注VMEM、SMEM或LDS；它表示供数依赖贡献，不等于已测出单次HBM返回延迟。LDS issue多也不能自动定为LDS bandwidth饱和。compute/issue是方向判断，尚未由校准全卡利用率证明达到计算峰值。

分类证据与优化收益分别记录：同一个parent可以随K、grid和实际body变化；采用前baseline的ATT只说明优化贡献，不能冒称采用后的主瓶颈仍完全相同。新增9001/9011采用版没有重采ATT，新增行的代表是正式API验证尺寸。

9040+的33个实际winner配置均已有自身ATT：旧7个加本次补26个，补采逐项核对CO/FUNC/full metadata/normalized descriptor和严格GPU owner；每个配置只采一个自身代表。[新26项分析](remaining_att_analysis.json)、[旧7项分析](../opus_remaining_20261008/small_att_analysis.json)。

## 按公开 parent 速查

| parent | 原始/候选配置数 | 类型判断 | 关键机制与适用范围 |
| --- | ---: | --- | --- |
| 9000 | 1 | 长K compute/issue；短K memory latency贡献＋setup/output | 原版3个自身代表；新9001分别选型 |
| 9001 | 1 | 未确认优化后主瓶颈 | SFA raw清零减少VGPR；自身性能独立；原版ATT只作方向 |
| 9010 | 1 | 长K compute/issue＋LDS；短K memory latency贡献＋setup/output | 原版2个自身代表；新9011分别选型 |
| 9011 | 1 | 未确认优化后主瓶颈 | 循环展开4；自身性能独立；原版ATT只作方向 |
| 9020 | 7 | 按body：fixed384 latency；其他compute/供数混合 | 7个body各有ATT；128x128 runtime更偏VMEM等待 |
| 9021 | 1 | 短K scale latency；长K compute/供数混合 | scale请求与publish重叠已验证贡献，采用后长K未重采 |
| 9022 | 1 | 短K memory latency；长K混合 | SFA/SFB串行、panel32 refill、matrix/LDS issue |
| 9023 | 2 | runtime memory latency贡献＋compute/issue | 采用前refill VMEM机制；fixed support未确认 |
| 9024 | 2 | fixed7168 memory latency贡献＋compute/issue | 启动约15.5%；runtime control未确认 |
| 9030 | 1 | compute＋memory latency＋输出服务混合 | 4.267GB/2.923TB/s不是HBM饱和证据 |
| 9040 | 1 | memory latency＋启动设置 | 短K约76% startup；不是dispatch证据 |
| 9041 | 2 | memory latency / VMEM request issue | runtime与实际9050各自ATT |
| 9042 | 3 | memory latency / VMEM request issue | runtime、实际9071/9073分开判断 |
| 9043 | 1 | memory latency贡献 / local LDS issue | 短K约42% startup |
| 9044 | 2 | memory latency / local LDS；部分compute混合 | runtime与实际9056各自ATT |
| 9045 | 1 | memory latency / local LDS＋issue混合 | LDS依赖7562、issue8736 clocks |
| 9046 | 1 | memory latency / local LDS同步＋issue | cluster2 gap648 vs316；非LDS带宽饱和结论 |
| 9047 | 1 | memory latency＋启动设置 | 短K约66% startup |
| 9049 | 1 | memory latency / local LDS＋启动设置 | 短K约47% startup |
| 9051 | 1 | memory request/latency压力；可能带宽成分 | 5.962TB/s与LFIFO/UTCL1线索；未证明独占HBM带宽限制 |
| 9052 | 2 | memory latency / VMEM request issue | runtime与实际9070各自ATT |
| 9053 | 2 | 短K memory latency＋setup；fixed VMEM issue | runtime与实际9072分开 |
| 9054 | 1 | 短K memory latency＋setup | startup约65%；wave计数8/8/16/16 |
| 9055 | 1 | memory latency / scale发布与local LDS | scale producer启动到达差约2300 clocks |
| 9060 | 3 | memory latency / local LDS＋compute混合 | runtime/K3072/K7168各自ATT |
| 9061 | 1 | memory latency / local LDS与同步 | main LDS依赖15666、barrier7030 clocks |
| 9062 | 5 | producer memory latency / local LDS与同步 | 两winner body各自ATT；另兼容producer与2reducer未独立分类 |
| 9063 | 11 | producer memory latency / local LDS＋compute混合 | 7winner body各自ATT；兼容producer与3reducer未独立分类 |

## 全部实际 entry

完整symbol、machine身份、owner与原始报告哈希见[JSON](bound_classification.json)；可筛选表见[CSV](bound_classification.csv)。`未确认`是完整清单中的明确状态，不借别的body测量填标签。

| # | parent / actual | traits | 自身代表 M×N×K | 类型与置信范围 | 关键依据 |
| --- | --- | --- | --- | --- | --- |
| 1 | 9000 / 9000 | `4wave` | 1792×7168×384；8192×8192×8192；1792×7168×16384 | shape_dependent；中等；按阶段判断；原版自身短/长K代表；不是所有winner统一标签 | 短K：启动与写回固定工作占大头，SMEM/scale依赖是memory latency贡献；长K：MFMA发射/依赖占主要局部时间，倾向compute/issue，同时有LDS/VMEM供数成本。未证明全卡计算峰值。 [att_analysis.json](att_analysis.json) |
| 2 | 9010 / 9010 | `4wave_256x256_padded_m` | 1856×7168×384；1856×7168×16384 | shape_dependent；中等；按阶段判断；原版自身短/长K代表；不是所有winner统一标签 | 短K：启动与写回固定工作占大头，SMEM/scale依赖是memory latency贡献；长K：MFMA发射/依赖占主要局部时间，倾向compute/issue，同时有LDS/VMEM供数成本。未证明全卡计算峰值。 [att_analysis.json](att_analysis.json) |
| 3 | 9020 / 9020 | `8wave<192, 256, 128, 0>` | 1536×7168×16384 | mixed；中等；自身局部ATT；自身代表形状；与最终device相同 | 稳态MFMA发射与matrix VMEM issue、LDS/barrier共同占时；倾向compute/issue加供数，尚无绝对执行管线饱和证据。固定K和runtime各用自己的ATT。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 4 | 9020 / 9020 | `8wave<128, 128, 64, 0>` | 8192×768×7168 | memory_latency；中等；自身局部ATT；自身代表形状；与最终device相同 | 128x128 runtime代表VMEM依赖约12026、MFMA事件12406 clocks，另有LDS/barrier；供数与计算混合，不能套192x256长runtime结论。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 5 | 9020 / 9020 | `8wave<192, 256, 8, 384>` | 1536×7168×384 | memory_latency；中等；自身局部ATT；自身代表形状；已采用改动之前的baseline机制 | fixed384启动publication/scale producer延迟与固定输出成本显著；采用请求重排，全7winner GM+2.5715%。采用版总瓶颈未重新采集。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 6 | 9020 / 9020 | `8wave<192, 256, 8, 768>` | 1728×7168×768 | mixed；中等；自身局部ATT；自身代表形状；与最终device相同 | fixed768启动5188/总22338、after-last4924 clocks，MFMA4456与barrier2822；设置/输出与计算供数混合。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 7 | 9020 / 9020 | `8wave<192, 256, 32, 1536>` | 1344×16384×1536 | mixed；中等；自身局部ATT；自身代表形状；与最终device相同 | 稳态MFMA发射与matrix VMEM issue、LDS/barrier共同占时；倾向compute/issue加供数，尚无绝对执行管线饱和证据。固定K和runtime各用自己的ATT。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 8 | 9020 / 9020 | `8wave<192, 256, 32, 3072>` | 1728×7168×3072 | mixed；中等；自身局部ATT；自身代表形状；与最终device相同 | 稳态MFMA发射与matrix VMEM issue、LDS/barrier共同占时；倾向compute/issue加供数，尚无绝对执行管线饱和证据。固定K和runtime各用自己的ATT。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 9 | 9020 / 9020 | `8wave<192, 256, 64, 7168>` | 6144×2048×7168 | mixed；中等；自身局部ATT；自身代表形状；与最终device相同 | 稳态MFMA发射与matrix VMEM issue、LDS/barrier共同占时；倾向compute/issue加供数，尚无绝对执行管线饱和证据。固定K和runtime各用自己的ATT。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 10 | 9021 / 9021 | `4wave_128x128` | 480×7168×384；480×7168×16384 | shape_dependent；scale因果较强；总瓶颈中等；短K采用版ATT与前后Event；长K为采用前自身baseline | 短K的scale请求串行与LDS发布等待是已验证贡献，重叠请求后全42winner GM+1.5756%；长K仍混合matrix issue、LDS和panel refill。不能据此称采用版全部latency瓶颈已消除。 [scale_issue_publish_att_analysis.json](../../reports/opus_resume_20261008/diagnostics/scale_issue_publish_att_analysis.json) |
| 11 | 9022 / 9022 | `4wave_160x128` | 800×7168×384；800×7168×16384 | shape_dependent；中等；自身局部ATT；自身代表形状；与最终device相同 | 短K启动5016/总10578 clocks，SFA→wait/store→SFB串行；长K混合MFMA、matrix issue、LDS及panel32 refill。全局候选虽GM微正，有6个5/5 loser而拒绝。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 12 | 9023 / 9023 | `4wave_64x128_traits_base_gfx950<3, 32, 0>` | 544×7168×16384；576×7168×7168 | memory_latency；机制中等；采用前baseline；已采用改动之前自身baseline；不宣称采用后独占主瓶颈 | runtime对齐refill VMEM串行、发布依赖；动态readlane每wave仅6次用于remainder/output，静态spill不是稳态120次读。采用版全4winner GM+1.1173%。 [narrow_att_analysis.json](../../reports/opus_remaining_20261008/narrow_att_analysis.json) |
| 13 | 9023 / 9023 | `4wave_64x128_traits_base_gfx950<4, 64, 7168>` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 合法support/control body，无历史winner；数值或control Event不构成本body独立瓶颈分类。  |
| 14 | 9024 / 9024 | `4wave_64x64_traits_base_gfx950<32, 0, 0>` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 合法support/control body，无历史winner；数值或control Event不构成本body独立瓶颈分类。  |
| 15 | 9024 / 9024 | `4wave_64x64_traits_base_gfx950<64, 7168, 4>` | 1344×768×7168 | memory_latency；机制中等；采用前baseline；已采用改动之前自身baseline；不宣称采用后独占主瓶颈 | fixed7168启动约15.5%，scale producer/publication延迟是贡献，稳态计算与LDS issue仍重要。采用版全9winner GM+1.2745%。 [narrow_att_analysis.json](../../reports/opus_remaining_20261008/narrow_att_analysis.json) |
| 16 | 9030 / 9030 | `8wave_192x256_large_output` | 65536×16384×1536 | mixed；中等；自身局部ATT；自身代表形状；与最终device相同 | 大输出的计算/供数/写回混合；barrier cohort到达差908 clocks，而last-arrival到release仅4。clean DRAM4.267GB/2.923TB/s确认输出流量重要，不能证明HBM饱和；midpoint全10winner失败。 [merged_att_analysis.json](../../reports/opus_remaining_20261008/diagnostics/merged_att_analysis.json) |
| 17 | 9040 / 9040 | `register<16, 32, 1, 1, 6, 1, 3, 3, 0, false, false>` | 8×7168×384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 短K startup2452/3224 clocks约76%，主要SMEM参数/scale供给与固定设置；小grid不等于dispatch已证实。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 18 | 9041 / 9041 | `register<16, 16, 1, 1, 2, 8, 0, 0, 0, false, false>` | 32×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命29.3%；main VMEM依赖4032、issue1952，MFMA事件54 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 19 | 9041 / 9050 | `register<16, 16, 1, 1, 3, 8, 3, 3, 0, false, false>` | 80×768×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命39.6%；main VMEM依赖2376、issue2160，MFMA事件56 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 20 | 9042 / 9042 | `register<32, 32, 1, 1, 3, 4, 3, 0, 0, false, true>` | 32×7168×3072 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命52.1%；main VMEM依赖1346、issue1980，MFMA事件454 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 21 | 9042 / 9073 | `register<32, 48, 1, 1, 3, 4, 4, 0, 7168, true, false>` | 64×6144×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命25.4%；main VMEM依赖3954、issue10384，MFMA事件1506 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 22 | 9042 / 9071 | `register<16, 48, 1, 1, 4, 4, 4, 3, 7168, true, false>` | 192×768×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 实际9071 fixed7168 register队列有VMEM请求/依赖等待；这是9071自身证据，不能代替9073或runtime。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 23 | 9043 / 9043 | `lds<32, 64, 1, 4, 8, 2, 2, false, false, true, false, false, 1, 4, 128, 0, 0, false>` | 32×16384×1536 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命42.0%；main LDS依赖338、issue1370，MFMA事件476 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 24 | 9044 / 9044 | `lds<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true, 1, 4, 128, 0, 0, false>` | 192×6144×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命11.5%；main LDS依赖10956、issue3714，MFMA事件3472 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 25 | 9044 / 9056 | `lds<64, 64, 2, 2, 8, 2, 2, false, false, true, false, false, 1, 4, 128, 0, 0, false>` | 64×16384×1536 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命37.8%；main LDS依赖306、issue1854，MFMA事件1268 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 26 | 9045 / 9045 | `lds<96, 64, 2, 2, 4, 1, 1, false, false, false, false, true, 1, 4, 128, 0, 0, false>` | 768×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命13.5%；main LDS依赖7562、issue8736，MFMA事件5548 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 27 | 9046 / 9046 | `lds<64, 128, 4, 2, 6, 2, 2, false, false, false, false, false, 1, 4, 128, 0, 0, false>` | 256×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | cluster2 MFMA间隔648 vs普通316 clocks，VMEM wait约20；偏local LDS依赖/同步和issue，未证明LDS带宽饱和。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 28 | 9047 / 9047 | `lds<32, 64, 2, 2, 4, 1, 2, true, false, false, false, false, 1, 4, 128, 0, 0, false>` | 64×7168×384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 短K startup约66%，SMEM wait约1019 clocks；设置/内存依赖占主，未证明dispatch。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 29 | 9049 / 9049 | `lds<32, 128, 2, 2, 4, 2, 2, true, true, false, true, false, 1, 4, 128, 0, 0, false>` | 224×7168×768 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命47.1%；main LDS依赖1450、issue308，MFMA事件272 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 30 | 9051 / 9051 | `register<16, 32, 1, 1, 3, 4, 3, 3, 0, false, false>` | 1×65536×1536 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | M1代表VMEM请求issue/返回等待显著；旧4组clean PMC约5.962TB/s、LFIFO24.03%、UTCL1在途6.18%，支持memory request/latency压力，可能有bandwidth成分；M方向16x padding浪费。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 31 | 9052 / 9052 | `register<16, 32, 1, 1, 2, 8, 3, 3, 0, false, false>` | 16×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命12.3%；main VMEM依赖12056、issue14372，MFMA事件498 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 32 | 9052 / 9070 | `register<16, 32, 1, 1, 3, 8, 4, 3, 7168, false, true>` | 64×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命37.9%；main VMEM依赖2782、issue1858，MFMA事件136 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 33 | 9053 / 9053 | `register<32, 32, 1, 1, 2, 8, 3, 3, 0, false, true>` | 32×7168×1024 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命75.4%；main VMEM依赖106、issue0，MFMA事件76 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 34 | 9053 / 9072 | `register<32, 32, 1, 1, 4, 4, 4, 3, 7168, false, true>` | 128×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命11.5%；main VMEM依赖4448、issue11504，MFMA事件694 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 35 | 9054 / 9054 | `register<32, 64, 1, 1, 2, 4, 3, 3, 0, false, true>` | 64×7168×768 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命64.7%；main VMEM依赖206、issue0，MFMA事件270 clocks。请求供给/返回等待主导，未证明外部带宽饱和。 K768四wave计数为8/8/16/16，总48；12仅均值，不作为每wave期望。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 36 | 9055 / 9055 | `lds<32, 64, 1, 4, 12, 4, 2, false, false, true, false, false, 1, 4, 128, 0, 0, false>` | 64×7168×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | scale producer让startup barrier到达延后约2300 clocks，SFA/SFB wait892/944；稳态仍有deep-ring LDS issue。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 37 | 9060 / 9060 | `fine<80, 1, 4, 4, 1, 1, 4, 128, 0, 0>` | 160×16384×1536 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命28.5%；main LDS依赖3860、issue1424，MFMA事件2340 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 38 | 9060 / 9060 | `fine<80, 1, 4, 4, 1, 1, 4, 128, 0, 3072>` | 320×7168×3072 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命15.4%；main LDS依赖7072、issue2926，MFMA事件4978 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 39 | 9060 / 9060 | `fine<80, 1, 4, 4, 1, 1, 4, 128, 0, 7168>` | 1280×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命8.8%；main LDS依赖17304、issue6164，MFMA事件11352 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 40 | 9061 / 9061 | `fine<96, 2, 4, 4, 1, 1, 4, 128, 0, 0>` | 1536×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命8.2%；main LDS依赖15666、issue7468，MFMA事件6626 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 41 | 9062 / 9062 | `fine<80, 1, 4, 4, 1, 2, 4, 128, 0, 0>` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 无历史winner的兼容producer；保留身份，不能借同parent其他配置的ATT给它定性。  |
| 42 | 9062 / 9062 | `fine<80, 1, 4, 4, 1, 2, 4, 128, 0, 16384>` | 144×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 实际80/fixed16384/split2 producer scale尾使startup到达延后2308 clocks；steady local LDS/同步。完整API含matching reducer，producer分类不能代替完整API。 [small_att_analysis.json](../../reports/opus_remaining_20261008/small_att_analysis.json) |
| 43 | 9062 / 9064 | `fine<96, 2, 4, 4, 1, 2, 16, 128, 2, 0>` | 1920×768×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命17.6%；main LDS依赖7796、issue3544，MFMA事件3458 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 44 | 9062 / 9062 | `void reduce_kernel<2, 4, 128>(float const*, bool _Accum*)` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 保留身份与完整调用支持；没有每个reducer独立的带宽/latency/dispatch诊断。  |
| 45 | 9062 / 9062 | `void reduce_kernel<2, 16, 128>(float const*, bool _Accum*)` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 保留身份与完整调用支持；没有每个reducer独立的带宽/latency/dispatch诊断。  |
| 46 | 9063 / 9063 | `fine<80, 1, 4, 4, 1, 4, 16, 128, 2, 0>` | 320×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命24.8%；main LDS依赖4520、issue1698，MFMA事件2730 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 47 | 9063 / 9063 | `fine<80, 1, 4, 4, 1, 4, 4, 128, 0, 16384>` | 80×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命11.5%；main LDS依赖10296、issue3086，MFMA事件6372 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 48 | 9063 / 9065 | `fine<96, 2, 4, 4, 1, 4, 16, 128, 2, 0>` | 384×2048×7168 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命25.4%；main LDS依赖3774、issue1710，MFMA事件1712 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 49 | 9063 / 9067 | `fine<128, 2, 2, 4, 1, 4, 4, 128, 0, 0>` | 128×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命12.9%；main LDS依赖12232、issue2942，MFMA事件11712 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 50 | 9063 / 9068 | `fine<112, 1, 4, 4, 1, 4, 4, 128, 0, 0>` | 112×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命11.2%；main LDS依赖13302、issue5044，MFMA事件9020 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 51 | 9063 / 9066 | `fine<96, 2, 2, 4, 1, 4, 16, 64, 0, 0>` | 96×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命12.9%；main LDS依赖10908、issue3008，MFMA事件8266 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 52 | 9063 / 9069 | `fine<48, 1, 4, 4, 1, 4, 4, 128, 0, 0>` | 48×7168×16384 | memory_latency；中等；自身局部ATT；自身代表；最终device身份相同 | 启动到first MFMA占局部寿命16.7%；main LDS依赖7528、issue2334，MFMA事件3402 clocks。local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。 [remaining_att_analysis.json](remaining_att_analysis.json) |
| 53 | 9063 / 9063 | `fine<48, 1, 4, 6, 2, 4, 4, 128, 0, 7168>` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 无历史winner的兼容producer；保留身份，不能借同parent其他配置的ATT给它定性。  |
| 54 | 9063 / 9063 | `void reduce_kernel<4, 4, 128>(float const*, bool _Accum*)` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 保留身份与完整调用支持；没有每个reducer独立的带宽/latency/dispatch诊断。  |
| 55 | 9063 / 9063 | `void reduce_kernel<4, 16, 64>(float const*, bool _Accum*)` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 保留身份与完整调用支持；没有每个reducer独立的带宽/latency/dispatch诊断。  |
| 56 | 9063 / 9063 | `void reduce_kernel<4, 16, 128>(float const*, bool _Accum*)` | — | unconfirmed；未确认；未作本配置独立瓶颈测量 | 保留身份与完整调用支持；没有每个reducer独立的带宽/latency/dispatch诊断。  |
| 57 | 9001 / 9001 | `void gemm_a8w8_mxfp8_scale_kernel<4wave_traits_scale_reset_gfx950>(kargs_gfx950)` | 512×512×128；512×512×256；1792×7168×384；2048×7168×3072；512×512×8320；1792×7168×16384 | unconfirmed；最终候选瓶颈未重采；原版诊断用于选方向；优化版只有自身性能/数值和machine身份 | SFA raw临时变量清零；VGPR477→469。原版的短/长K类型不能直接当作本候选最终主瓶颈。保留为可选tuner候选：两原5/5正样本在新seed仍median微正；全129 GM略负，禁止全局替换或宣称稳定普遍收益。 [integration_manifest.json](split_formal_scale_reset/integration_manifest.json) |
| 58 | 9011 / 9011 | `void gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel<4wave_256x256_padded_m_traits_unroll4_gfx950>(kargs_gfx950)` | 64×256×128；320×256×256；1856×7168×384；512×512×8320；1920×7168×3072；1856×7168×16384 | unconfirmed；最终候选瓶颈未重采；原版诊断用于选方向；优化版只有自身性能/数值和machine身份 | 循环展开4；VGPR497→492。原版的短/长K类型不能直接当作本候选最终主瓶颈。保留展开4为独立tuner候选：全38历史winner GM+0.2808%，34正median，0个5/5 loser；原9010供4个negative median形状回选。 [integration_manifest.json](split_formal_scale_reset/integration_manifest.json) |

## 证据边界与后续优先级

Memory bandwidth需要同层级的可达带宽、请求/通道分布与回压；多数代表缺少完整的匹配窗口。9051外部请求压力较强，仍无法把LFIFO/翻译队列与HBM带宽限制唯一分开。9030写流量大，但仅凭bytes/time不能定为bandwidth bound。

Dispatch需要动态活跃wave、SPI/资源分配stall和完成时间影响。静态VGPR/LDS大、224个WG少于256CU、短K启动长，均只提供线索；当前没有可靠dispatch判定。CPU注册/分支身份核对与dispatch瓶颈测量是不同事情。

下一步若继续针对机制优化，register优先查请求并发/等待位置，LDS/fine优先查供数依赖与barrier到达差，大tile长K优先查MFMA issue与matrix ring重叠；每项都需要自身完整调用性能决定。当前分类所需的有限采集已收口，不把全系列分类写成全系列已提速。
