from pathlib import Path
import hashlib,json,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
review=json.loads((OUT/'final_review.json').read_text());s=review['screening'];c=review['confirmation'];wins=pd.read_csv(OUT/'candidate_wins.csv');comp=pd.read_csv(OUT/'shape_comparison.csv');confirm=pd.read_csv(OUT/'confirmation_comparison.csv')
base=ROOT/'reports/opus_9000_9010_bound_20261008/bound_classification.csv';bound=pd.read_csv(base)
# Retain full per-device classifications and attach the new measured selection
# count separately. Count is per parent, so do not sum repeated device rows.
joined=bound.merge(wins.rename(columns={'kernelId':'parent_id'}),on='parent_id',how='left',validate='many_to_one')
joined['retune_count_scope']='parent-level counts repeated per device row; classification evidence inherited Oct8 ATT, not new profiling'
joined.to_csv(OUT/'bound_classification_with_retune.csv',index=False)
headers={9000:'长K compute/issue＋供数；短K scale latency＋固定成本',9001:'优化后主瓶颈未确认；SFA清零降VGPR',9010:'长K compute/issue＋LDS；短K固定成本',9011:'优化后主瓶颈未确认；padded-M unroll4',9020:'按实际body：latency 或 compute/供数混合',9021:'短K scale latency；长K compute/供数混合',9022:'短K memory latency；长K混合',9023:'runtime VMEM依赖＋compute；fixed support未确认',9024:'compute/issue＋VMEM/LDS供数',9030:'compute/issue＋输出store成本；无整系列HBM饱和证明'}
for kid in wins.kernelId:
 if kid not in headers:headers[int(kid)]='按配置：VMEM请求/依赖 或 local LDS/同步'
table=['| ID | 合法配置数 | OPUS内胜出 | 全后端胜出 | 已有bound证据方向 |','| --- | ---: | ---: | ---: | --- |']
for r in wins.itertuples(index=False):table.append(f'| {r.kernelId} | {r.enumerated} | {r.opus_wins} | {r.all_backend_wins} | {headers[r.kernelId]} |')
strong=confirm[confirm.selected_faster_rounds.eq(5)].sort_values('selected_median_speedup',ascending=False)
strong_table=['| M×N×K | 新ID | 复测中位数加速 | 五轮更快 |','| --- | ---: | ---: | ---: |']
for r in strong.itertuples(index=False):strong_table.append(f'| {r.M}×{r.N}×{r.K} | {r.screen_selected_id} | {(r.selected_median_speedup-1)*100:.4f}% | 5/5 |')
gain=(s['same_run_26_vs28']['geomean_speedup']-1)*100;sum_saved=s['same_run_26_vs28']['time_sum26_us']-s['same_run_26_vs28']['time_sum28_us'];sum_pct=100*(1-s['same_run_26_vs28']['time_sum28_us']/s['same_run_26_vs28']['time_sum26_us'])
report=f'''# 28候选全量 retune：745 shapes，8卡并行

全部 **28 个 OPUS 候选**已经纳入：保留9000、9010，加入9001、9011，并覆盖9020–9024、9030及9040+等所有公开候选。全745个gfx950/256CU shape完成，**13,027/13,027个合法 OPUS配置均已执行，errRatio=0**；全部后端合计79,504条原始记录。最优后端是OPUS720项、ASM24项、CK1项。

新增9001/9011的全量筛选增量很小：相对**本轮同shape、同GPU测量中原26候选的最优值**，完整28候选几何平均加速 **{gain:.6f}%**，各shape单次时间之和下降 **{sum_pct:.6f}%**。这是单轮筛选的最优值统计，包含选型噪声；多数新赢家在五轮复测时会翻转，不能宣称已得到普遍稳定收益。

## 同批原26 vs 完整28

原26候选也使用当前已接受的9021等优化，所以这组比较只衡量**增加9001/9011**的效果，不是对Sep30冻结二进制的A/B。

| 项目 | 原26候选最优 | 新28候选最优 |
| --- | ---: | ---: |
| 745shape单次时间之和 | {s['same_run_26_vs28']['time_sum26_us']:.4f} µs | {s['same_run_26_vs28']['time_sum28_us']:.4f} µs |
| 相同数据中新增候选被选中的shape | 0 | 78 |
| 9000选中 | 127 | 68 |
| 9001选中 | 0 | 59 |
| 9010选中 | 29 | 11 |
| 9011选中 | 0 | 19 |

78项改善、667项最优值不变；5项筛选时的延迟降幅超过1%，最大1.371630%。78个新增选中项自身的几何平均加速0.349335%，全集合几何平均加速{gain:.6f}%。单次时间之和节省{sum_saved:.4f} µs，不能当作某个模型端到端时间。由于原26始终留在集合内，同一raw数据取最优的28结果不会比26结果更慢，这本身不是重复运行无回退的证据。

## 新候选五轮复测

全78个新增筛选赢家都复测。每个shape重测原最优、原前三、接近的竞争者和原/新family成员，合计**257次独立signed数据数值校验、1,285条性能记录**。每项五轮、随机候选顺序、同一组8个地址、warmup5/iters51，直接调用正式OPUS接口。复测输入改为seed29正负FP8；仍为native E8M0和完整FP32/BF16 accumulation interval零outlier。

| 指标 | 9001 | 9011 |
| --- | ---: | ---: |
| 全量单轮筛选选中 | 59 | 19 |
| 原选中项在复测中位数继续胜出 | 29 | 12 |
| 五轮都比复测原最优更快 | 2 | 3 |
| 五轮都比复测原最优更慢 | 0 | 2 |
| 原选中子集相对复测原最优GM | +0.014012% | +0.845944% |
| 复测重新选型后的新ID数量 | 30 | 12 |

复测重新选择后是9001共30项、9011共12项；14336×2048×7168从筛选的9011转为9001。其余37项回选原候选。中位数胜出也不等于5/5稳定胜出，所有轮次已保留。

{chr(10).join(strong_table)}

上述时间来自单独的shared-pool profiler复测，部分差距比筛选更大。地址池、输入seed、GPU分配改变，**不能将这张表的时间混入筛选全集合GM**，也不能直接替代此前Event实验的结论。9011在40960×7168×768和14336×2048×7168中5/5慢，复测选型已回到其他候选；9001的59项几何平均只有+0.0140%，证据仍弱。

## 全后端结果与旧记录

OPUS28相对**本轮CK/CKTile/ASM最快有效者**：720项更快、25项更慢，几何平均 **1.255110×**。全后端最优选择OPUS720、ASM24、CK1；本次枚举不含Triton。CK/CKTile/ASM保留FP32scale，OPUS为native E8M0；公共shuffle和reference不计时，后端内转换按正式tuner计时。

与Sep30历史745个OPUS最优值作描述性比较：649项更快、96项更慢，GM **1.041807×**。历史记录使用不同环境、多卡分配和计时地址池，本次还重建了外部模块；这个+4.1807%不能全部归因于新增候选。Sep30的旧原始表和manifest保留。

无OPUS配置数值失败。外部raw中CKTile874项不支持/执行无效；CK498项无效，其中408项没有有效计时、90项有计时但未通过数值门槛。这些记录已保留并排除选型，所有745shape仍有有效最优项。

## 候选与bound

本次新增性能数据不是新的ATT/counter证据。类型沿用[上轮完整分类](../opus_9000_9010_bound_20261008/BOUND_CLASSIFICATION.md)的配置/shape/阶段范围；详细58个device配置连同本轮parent胜出数量见[完整分类附retune数量](bound_classification_with_retune.csv)。下表只是方向速查，不能将整个parent归为单一瓶颈。

{chr(10).join(table)}

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
'''
(OUT/'README.md').write_text(report)
# Preserve prior documentation bytes and append a self-contained new section.
append_records=[]
text=f'''\n\n## 14. Oct8 全28候选、745shape重新tune（8卡）

全量结果见[retune报告](../opus_retune28_20261008/README.md)，状态completed。原9000/9010保留，9001/9011与其他全部公开候选纳入，总28parent。745shape完成、13,027/13,027合法OPUS配置errRatio=0，全后端79,504条raw。开始150shape单卡，用户要求8卡后其余595shape按shape分组并行；每个shape的全部候选同卡计时，全部KFD握手、物理锁和严格非owner PID审计通过。

同轮原26最优vs完整28最优：9001选59、9011选19，全集合GM+{gain:.6f}%，单次时间和−{sum_pct:.6f}%；78项单轮改善，最大延迟降幅1.371630%。这是新增候选的筛选增量，原26也使用当前已采用的其他优化；min统计有选型噪声，不等于稳定收益。

78项五轮shared-eight-address profiler复测，seed29 signed native E8M0，257次数值校验和1,285条计时。9001原59中29项中位数继续胜，9011原19中12项继续胜；仅2/3项分别5/5快，9011有2项5/5慢。重选后的新ID数量9001为30、9011为12（包含1项9011转9001）；证据来源分别保留，不把不同地址池/seed复测时间混入筛选GM。raw与复测config都在新目录，旧生产默认CSV保持。

OPUS28对同轮CK/CKTile/ASM最快者为720快/25慢、GM1.255110×；全后端选择OPUS720/ASM24/CK1。与Sep30旧OPUS表649快/96慢、GM1.041807×仅描述跨轮变化，不能归因于新两项。大输出>32M elements地址池上限8、reference/comparison256行分块；最大8GiB输出全部完成，旧自动轮换时间不能用于因果A/B。外部无效CK498/CKTile874记录保留，不参与选型。

bound仍按第13节的实际配置/阶段证据；本轮没有新的ATT/counter，不证明任何整个系列为纯HBM bandwidth或dispatch bound，9001/9011优化后的主瓶颈仍未确认。[完整58配置分类附本轮数量](../opus_retune28_20261008/bound_classification_with_retune.csv)明确parent计数不能跨重复device行求和。正式OPUS SO SHA仍为`e18fe59bf6b06ddc53349df5eaa5a3f5d31d3b00b75627705c26477d09db00eb`，最终审核和新增文件哈希见retune报告。原第14节前全部字节保持，不改变旧manifest所描述的历史快照。没有commit/push。
'''
for name in ['COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md','MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md']:
    path=ROOT/'reports/opus_bound_analysis_20261007'/name;old=path.read_bytes();assert b'## 14. Oct8' not in old
    path.write_bytes(old+text.encode());assert path.read_bytes()[:len(old)]==old
    append_records.append({'path':str(path),'old_prefix_bytes':len(old),'old_prefix_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
p=ROOT/'HANDOFF_MXFP8.md';old=p.read_bytes();heading=b'# MXFP8 B-preshuffle \xe4\xbc\x98\xe5\x8c\x96\xe4\xba\xa4\xe6\x8e\xa5\n';assert old.startswith(heading)
latest=f'''\n## 2026-10-08：28候选全745shape retune完成（8卡）

最新入口为[完整retune报告](reports/opus_retune28_20261008/README.md)，状态`completed_verified_not_committed`。原9000/9010与新增9001/9011及全部9020+/9030/9040+共28公开候选都纳入；745shape、13,027合法OPUS配置全部errRatio=0，79,504全后端raw。用户要求8卡后595shape并行，已完成150shape保持，同shape全部候选同卡计时；8卡KFD owner/物理锁/严格PID审计通过。

同轮原26最优vs完整28最优，9001选59、9011选19；全集合GM+{gain:.6f}%，合成单次时间和−{sum_pct:.6f}%，大部分微小收益有选型噪声。78项五轮复测原选项9001有29/59继续胜、9011有12/19继续胜，5/5快仅2/3项；复测重选数量30/12含一个9011转9001。OPUS对同轮外部最快者720快/25慢、GM1.255110×。Sep30旧时间GM1.041807×仅描述跨轮，不能因果归于两新ID。

[筛选OPUS CSV](reports/opus_retune28_20261008/tuned_opus28.csv)、[全后端CSV](reports/opus_retune28_20261008/tuned_all.csv)、[复测选型config](reports/opus_retune28_20261008/tuned_all_rechecked_config.csv)均独立保存；混合来源复测CSV不能用于新的全集合GM。原默认生产配置未覆盖，正式SO SHA不变。Compute/Memory追加第14节保留旧字节，分类没有新ATT证明。见[最终审核](reports/opus_retune28_20261008/final_review.json)，没有commit/push，待实验0。
'''
p.write_bytes(heading+latest.encode()+old[len(heading):]);append_records.append({'path':str(p),'old_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'method':'prepend latest section after title; original content preserved'})
(OUT/'document_append_manifest.json').write_text(json.dumps(append_records,indent=2)+'\n')
print('wrote report, per-device classification+counts, appended Compute/Memory14, prepended HANDOFF')
