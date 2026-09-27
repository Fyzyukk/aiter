# MXFP8 B-preshuffle 优化交接

## 当前入口：原专用 tuner + M >= 1024 untuned CSV

运行文件恢复为
[`csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py`](csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py)。
原始输入和历史基线是
[`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`](aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv)，
本次未修改其内容。原表共 1042 条数据，其中 gfx950/256 CU 共 745 个唯一 shape。
按当前要求，已将其中 `M >= 1024` 的全部 **305 个唯一 shape** 提取到
[`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv`](aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv)。
新文件只含 `gfx,cu_num,M,N,K` 五列，保留旧 295 项子集之外的全部 10 项。
本次只提取并核对 CSV，未运行 GPU 调优。

输入按后端生成：CK/CKTile/ASM 直接复用原 blockscale tuner 的
`generate_data`，A/B 为 `rand(FP16) / 10` 后转 FP8，scale 独立随机生成 FP32；
OPUS 使用相同的 A/B 生成方式，scale 独立生成原生 E8M0 指数字节。
外部后端不使用 E8M0 解码的 scale。调优和回放均使用各自数据计算参考结果，
保留现有误差判定；同 shape 切换输入生成器时刷新数据和参考缓存。

在仓库根目录执行，将编译器路径换成目标机器的实际路径：

```bash
ROCR_VISIBLE_DEVICES=0 \
OPUS_HIP_CLANG_PATH=/absolute/path/to/llvm-pin-build/bin \
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  -i aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv \
  -o /tmp/dsv4_m_ge1024_tuned.csv \
  -o2 /tmp/dsv4_m_ge1024_profile.csv \
  --opus-kids 9000,9020,9060,9061,9062,9063,9064 \
  --libtype all --splitK --shape_grouped --mp 1 \
  --warmup 5 --iters 51 --all
```

这是原 tuner 的单命令入口，OPUS JIT 在正式扫描前自动补编候选，CK/CKTile/ASM
沿用原后端的 JIT 路径。八卡时使用物理 0–7 卡，将环境变量改为 `ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，
并将参数改为 `--mp 8`。
`--shape_grouped` 将每个 shape 的候选放在同一张卡比较。

新五项使用正式 ID，避免旧独立库 ID 与其它架构的全局 ID 区间冲突：

| Family | 正式 ID | 历史独立库 ID | Tile M×N×K |
|---|---:|---:|---|
| main | 9060 | 21000 | 192×256×128 |
| small | 9061 | 21310 | 128×128×128 |
| small | 9062 | 21311 | 160×128×128 |
| narrow | 9063 | 21220 | 64×128×128 |
| narrow | 9064 | 21221 | 64×64×128 |

上面的命令只在 OPUS 侧选择 9000/9020 + 新五项，并与 CK/CKTile/ASM 比较。
旧 9010/9011/9012 注册仍在；省略 `--opus-kids` 时原 tuner 枚举全部已注册项。
不支持某个 shape 的 OPUS 候选按原规则跳过；原始 shape 仍由有效的外部候选参与比较。

`-i` 读取新 untuned CSV 的 shape；`-o` 保存新的逐 shape 最优选择，`-o2` 保存
各候选记录。原始基线的历史微秒数不会参与本轮选型。若需要扫描完整 745 项，
将 `-i` 换回原始基线 CSV；tuner 会忽略其中已有的 `libtype/kernelId/us`。

环境需要 gfx950 ROCm/PyTorch（原生 `torch.float8_e8m0fnu`）、已初始化的 CK submodule，
以及支持 `clang::amdgpu_pin_agpr` 的定制 clang；已验证工具链源为
`yuyzhang512/llvm-project` 提交 `49c41889681640665400cb01c9fbb4c0a024cde4`。

以下为历史阶段记录。此前新增的 `reports/opus_remote_tune_20260927/` 仅保留为
独立实验工具；它的 prepare/build/launch 和 295 项子集不是当前推荐入口。

## 2026-09-27 最新：合并 kernel 统一为 9000/9020 的源码样式

按用户最新要求，三个新 family 已整理成 `include/gfx950/` 下的正式
`opus_gemm_{pipeline,traits}_a8w8_mxscale_bpreshuffle_{main,small,narrow}_gfx950.cuh`。
入口统一为 `template<class Traits>`；small 用 M128/160 两个 Traits 实例，
narrow 用 N128/64 两个实例，main 也由固定入口改为 Traits 模板。
计算流程、runtime K、私有 ID 和原启动约束保持。来源为 main_v1/21000、
small_v4/21310+21311、tiny_v3/21220+21221；加原9000/9020，最终目标仍最多7候选。

三库五个kernel已CPU编译通过，与来源版本逐ID比较，指令字节和归一化资源描述
全部一致；94个既有文件（9000/9020、helpers、注册、codegen及来源文件）哈希未变。
新薄测量adapter和验证证据在
[opus_native_style_20260927](reports/opus_native_style_20260927/README.md)。
未运行GPU，也未把私有ID加入全局注册。

**合并调优尚未完成。** 当前可用完整基线仍是下方16候选、OPUS291/295。
第三批 `pilot62_v34_r3` 于08:32 UTC启动后中断：恢复检查时相关进程已不存在，
八份summary均0数据行，JSON的running/launching为残留状态。须以新batch重启测量，
完成合并版295项比较，再最终选型、替换和删除。原自动续跑/安装工具仍绑定旧版本
路径和符号，使用新正式头前需更新其来源映射。

## 2026-09-27 最新完成：8 卡全量重新 tune，仅保留实际中选 kernel

按最新要求，从头完成完整 295 项、8 卡、三轮的全后端扫描（CK/CKTile/ASM 全枚举），耗时22.20分钟；没有复用旧计时或1%子集阈值。8个worker均passed。**OPUS291胜、CKTile4胜；实际中选16个OPUS（原5+11通用）**。相对有效外部几何平均耗时降低12.9453%，相对同批原5降低4.2688%；4575条合法OPUS候选均通过目标数值检查。

保留原5：9000、9010、9011、9012、9020。保留通用：13163、20000、20010、20011、20020、20100、20124、20125、20126、20128、20131。移除未选通用9661、9663、20120、20121、20122、20123、20127、20129、20130，并删除旧9030–9051的9个注册和对应生成分支/5个无依赖头。9000及共享helpers未改，原5的生成代码逐字节一致。

中选通用已迁入 `csrc/opus_gemm/mxfp8_bpreshuffle_retained/`，独立C ABI构建，9库仅11个device kernel，机器码与实测版本完全相同。共清理50个纯MXFP8实验目录的380个实现/生成器/二进制文件，并替换旧测量JIT为原5子集、删除788个旧生成缓存。历史CSV/JSON/日志/文档和更早混合快照保留；下方旧实验源码链接可能因本次按要求删除而失效。

入口：[本次完成报告](reports/opus_retune_prune_20260927/README.md)、[16候选模板清单](reports/opus_retune_prune_20260927/SELECTED_CANDIDATES.md)、[现存候选配置](reports/opus_retune_prune_20260927/selected_experiments.json)、[295整体选择](reports/opus_retune_prune_20260927/full295_r3/results/overall_selection.csv)、[删除清单](reports/opus_retune_prune_20260927/deleted_experiment_files.json)。

4个外部负项仍是N7168/K384、M1536/1600/1664/1728，慢1.569%/3.237%/1.442%/0.376%，均0/3轮胜。不额外追加测试，不修改默认dispatch CSV；本次无后台GPU工作，未提交/推送。下文原8候选结论属于前一轮历史。

## 2026-09-27 本轮完成：原5保留，其余实验收敛为8个通用候选

用户要求保留 **9000、9010、9011、9012、9020**，将其余实验合并成真正的runtime-K候选，不要求恰好4个。已恢复昨天记录、完成295项重新tune，并继续做了短K循环、融合输出和cache策略优化。9000及影响它的共享helpers、生产注册和默认dispatch均未改；本轮只做kernel、目标shape数值检查及性能比较。

**最终结果：8个通用候选+原5，共291/295项快于同卡重测的有效外部finalist。** 原5选中200项，8个通用选中95项。相同shape等权几何平均耗时：相对原5降低4.147%，相对外部降低12.973%；相对旧25实验+原5增加0.114637%，最坏增加3.951%。固定K对照没有进入精简集合。20个通用候选的精确子集搜索证明：逐shape相对完整通用池最多1%损失、且保留其全部外部胜项时，最少需要8个；实际几何损失0.008106%、最坏0.956643%。

最终数据来自完整v6的178行与后续v7的117行，均为三轮；每个shape整行选择其最新完整worker，包含新旧候选、外部参考、来源和轮次。所有shape保持v6的物理GPU4–7分配，**两批全部passed，数值检查0拒绝**。两个队列均complete、benchmark/analysis退出码均0，本轮没有后台GPU工作。较早v5的119项五轮也全部通过，但不混入最终计时。

| 通用ID | 实现 | runtime K | 最终选择次数 |
|---|---|---|---:|
| 20000 | 192×256、8 wave、N-first | 128–16384，步长128 | 12 |
| 20100 | 同主体，窄N/大M使用group-M4 | 128–16384，步长128 | 31 |
| 20128 | 20000融合最终MFMA/BF16输出，vec8/cache2 | 128–16384，步长128 | 21 |
| 20010 | 128×128、4 wave | 128–1536，步长128 | 7 |
| 20011 | 160×128、4 wave | 128–1536，步长128 | 14 |
| 20020 | 192×224、8 wave、融合输出 | 128–1536，步长128 | 2 |
| 20125 | 192×256统一U2/drain、非融合vec8输出 | 128–1536，步长128 | 6 |
| 20131 | 192×256统一U2/drain、fused vec4、group-M4/cache2 | 128–1536，步长128 | 2 |

原5的使用次数：9000×129、9010×10、9011×23、9012×7、9020×31。各通用ID对应单个runtime-K入口，不按K实例化。

**有效与无效改动：** 20100的通用网格策略和20010/20011小tile保留主要贡献；20125在部分大M短K有用。20128相对20000在同源295项中200项更快、几何平均快0.165%，属于小幅互补收益。20131相对20127在117合法项上几何平均快2.305%，但只58项严格更快，不能全局替换；它在(1472,7168,384)为12.280889µs，快于同场外部12.379533µs，3/3轮胜，解决一个原通用池外部负项。同shape去掉20131的最快通用为20128，12.691872µs。

20129/20130统一小tile循环后，实际VGPR分配升至264/320；虽无spill，117项全部比20010/20011慢，几何平均慢55.76%/49.47%，已剔除。20120–20123无最终贡献；20124/20126等局部小收益在精简约束下可移除。所有失败实验、源代码和中间结果保留。

**剩余4个外部负项**都为N7168/K384，且本批各0/3轮胜：M1536慢2.888%、M1600慢2.882%、M1664慢2.110%、M1728慢0.905%。另有8项相对旧25+原5慢超过3%，最坏为(1152,7168,384)的3.951%；完整列表在最终报告。未把这些差距当作已解决，也未反复测到获胜。

最终交付入口：

- [完整中文结果与剩余慢项](reports/opus_generalize_20260926/refine117_20260927_v7_r3/review_results/RESULTS.md)
- [8候选配置及原5声明](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/selected_experiments.json)
- [295项选择、GPU与轮次来源](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/choices295.csv)
- [候选用量及移除影响](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/candidate_usage.csv)
- [同场优化配对结果](reports/opus_generalize_20260926/refine117_20260927_v7_r3/review_results/candidate_effects.csv)
- [精确子集与0%/1%/2%取舍](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/report.md)
- [测量与来源审计](reports/opus_generalize_20260926/refine117_20260927_v7_r3/runtime_analysis/summary.json)
- [最终独立选型复核](reports/opus_generalize_20260926/v7_final_selection_review.json)、[原始数据与结果复核](reports/opus_generalize_20260926/v7_results_review.json)：最小集合、同卡来源、固定K隔离及原5保护均通过。

配置是实测候选池，库目录相对配置文件解析；registered_opus_ids只声明原5成员，旧loader不消费该字段，不是已计时验证的生产dispatch。外部对照只从历史全后端扫描取有效finalist身份，全部耗时在当前worker重新测量；本轮没有重扫全部CK/CKTile/ASM。

阶段记录均保留：v3完整295三轮；v4短117三轮；v5短117+长2共119五轮；v6再做完整295三轮；v7最后117三轮。v6/v7使用同一物理GPU分配，v7以v6为base，禁止跨批次为各候选取最小值。源码、生成器、metadata、构建二进制和保护文件在各次计时期间冻结；无额外边界、接口或单元测试。本轮未提交或推送。

## 2026-09-26 最新恢复：减少实际候选，runtime-K 通用化

用户在完整295项调优后明确要求：把大量实验候选做成少量通用 kernel，**不是合并源文件**。最新工作入口为 [opus_generalize_20260926](reports/opus_generalize_20260926/README.md)。已编译4个实际runtime-K kernel：20000（192×256长短K）、20010（128×128短K）、20011（160×128短K）、20020（192×224融合输出）。仍冻结9000，只做kernel及目标shape数值/性能比较。

最新 `generalize99_v1_r3` 在99项目标测试中断：0–3因外部占用失败，4–7各仅完成1项；恢复时旧进程均已不存在。旧记录原样保留，不把其残留 `running` 当作当前任务。已准备 [resume_launch.py](reports/opus_generalize_20260926/resume_launch.py)，把99项重新分片到物理GPU4–7；复用历史有效外部候选身份，所有对照耗时在实际新卡重测，源卡与实际卡分别记录。新的完整通用化测量尚未完成，不能引用下文293/295作为通用候选池结果。

恢复时GPU0–3满载、4–7原本空闲；启动前复查发现后四卡也新增外部任务。08:38 UTC已启动独立等待队列PID11966，自动等待GPU4–7连续空闲后运行99项并汇总；当时状态为waiting，尚无新GPU结果。实时状态见 [queue_state.json](reports/opus_generalize_20260926/queue_resume99_v1_r3/queue_state.json)。不使用前四卡，不终止他人任务，不删除占用检测。

## 2026-09-26 最新完成：完整295项重新调优

**任务已完成：295/295全量三轮调优，加58项同GPU五轮确认；最终OPUS293胜、CKTile2胜、CK/ASM均0胜。** `full295_r3`与`close295_r5`两批各8个worker全部passed，已无后台GPU工作。最终237项采用三轮、58项采用五轮，每个shape取较新完整批次整行，不跨批次选最小耗时。

**9000、共享helpers、注册和默认配置完全未改。** 本次最终127项仍选中9000。`experiment`表示独立编译的OPUS实验实现，算法仍为OPUS。最终选型共30个ID：原注册9000×127、9020×29、9011×23、9010×10、9012×7，共196项；25个实验ID覆盖99项（其中97项整体胜出）。相对最快有效外部几何平均耗时下降13.01294%，相对同批旧OPUS下降4.34455%。

入口：

- [最终报告](reports/opus_cover87_20260926/display295_v1/README.md)
- [295项逐shape最佳OPUS选择](reports/opus_cover87_20260926/display295_v1/usage/opus_choices.csv)
- [30个候选的使用次数](reports/opus_cover87_20260926/display295_v1/usage/opus_candidate_usage.csv)
- [295项整体最快实现](reports/opus_cover87_20260926/display295_v1/overall_winners.csv)
- [汇总与来源](reports/opus_cover87_20260926/coverage295_v1/summary.json)
- [精选25实验ID配置](reports/opus_cover87_20260926/selected_experiments295.json)；全扫描实际输入为[25库109实验ID](reports/opus_cover87_20260926/experiments295.json)，另行保留全部合法注册OPUS。

剩余两项（均五轮）：`(1600,7168,384)` OPUS14540 12.784568us vs CKTile11 12.760023us，慢0.192362%、0/5轮胜；`(1536,7168,768)` OPUS12641 17.179160us vs CKTile30 17.168143us，慢0.064172%、2/5轮胜。本轮原87项为85胜/2负，其余208项全胜；历史v11的87/87只代表之前目标批次，不能替代本次全量结果。没有把噪声反复复测到获胜，也没有额外边界/接口/单元测试。

最后一次kernel改进为OPUS15940（192×224、cache0、融合最终MFMA/BF16 LDS输出）。完整295调优后，15940仍在`(1536,7168,384)`选中；相关源码在`reports/opus_resume_20260926/shortk_n224_exp/`。新候选全部独立实验实现；生产注册未改。

全部OPUS全扫描候选记录（注册2626、实验5650）及五轮确认候选均通过原目标数值检查。58项确认中56项中位数领先，37项每轮都领先。精确295成员复用旧`shapes.csv`，另10项原寻址排除未纳入。旧阶段记录、失败实验和所有候选源/二进制哈希继续保留。

后续若继续优化，优先上述两项；沿用已编译JIT及目标shape检查，不改9000。不得使用旧`selected_experiments.json`替代新全量候选配置，也不得用旧87快照替代295结果。

## 2026-09-26 v10历史阶段：原87项优化

**9000保持冻结。** 用户再次明确不得修改9000；本轮没有修改9000本体、共享helpers、生产注册或默认调度。新实现均位于 `reports/opus_resume_20260926/` 的独立实验目录。14640只是隔离的9000派生副本，无收益，已从后续候选配置移除。

按每个shape最新完整五轮批次，原87项 **86胜 / 1未胜**。相对最快有效CK/CKTile/ASM候选，几何平均耗时下降 **4.63%**；相对同批旧正式OPUS下降 **13.00%**。后续54项全部中位数领先。56项五轮均领先，另30项中位数领先但未每轮胜出，不能把全部中位数胜项称为稳定大幅领先。

唯一未胜项 `(1536,7168,384)`：本批最佳OPUS15040 **12.665778us**，cktile_11_split0 **12.660383us**，差 **0.0426%**，接近持平但保留为未胜。所有测量已经结束，最后批次为 `targeted1_v10_r5`，status=passed；没有后台GPU工作待收尾。

入口：[完整结果](reports/opus_cover87_20260926/RESULTS.md)、[87项选型](reports/opus_cover87_20260926/coverage87/best_opus_selection.csv)、[逐shape比较](reports/opus_cover87_20260926/coverage87/comparison.csv)、[汇总与来源](reports/opus_cover87_20260926/coverage87/summary.json)、[精选候选配置](reports/opus_cover87_20260926/selected_experiments.json)。每个shape整行使用最新完整批次；不跨批选最小耗时。

已完成顺序：`finalists87_r5` → `targeted35_v3_r5` → `targeted27_v4_r5` → `targeted3_v5_r5` → `targeted2_v6_r5` → `targeted2_v7_r5` → `targeted1_v8_r5` → `targeted1_v9_r5` → `targeted1_v10_r5`。全扫描来源为 `full87_r3`：完整枚举CK/CKTile/ASM，数值不合格候选剔除；后续只同卡重测有效外部最快者5%内对手。原87成员和295项历史基线保留。

本轮只进行kernel优化、编译、目标shape随测数值检查和性能比较，没有新增边界、接口、单元测试或295项全量回归。下文33项结果和额外验证流程为历史记录。

## 2026-09-26 前一阶段：33 项 kernel 实验

本轮按用户最新要求，只优化 kernel、测目标 shape、比较效果；停止扩展边界/接口测试和全后端扫描。完成四轮 kernel 实验，均在原 **25 个短 K + 8 个 K1536** 目标上同卡同批测五轮，随目标计时检查原 FP32 误差界。

**有效突破是 8-wave BF16 LDS 输出重排**：先按 pitch=264 将累加结果转为 BF16 放入复用的 LDS，再连续 `load/store<8>` 写回。最终候选 **9640/9641/9642/9651** 对应 K384/768/1024/1536；本轮 **33/33 超过同批旧最快 OPUS，18/33 超过同批 CKTile 对照**。相对原固定 K kernel，各组几何平均耗时下降 **31.29% / 24.10% / 23.42% / 23.83%**；相对旧最快 OPUS 下降 **10.98% / 13.07% / 15.32% / 10.75%**。CKTile 的多数胜负差距较小，18/33 是本批五轮中位数结果。

最新结果和复跑入口见 **[RESULTS.md](reports/opus_resume_20260926/RESULTS.md)**；最终逐 shape 比较在 [epilogue_r5](reports/opus_resume_run_20260926/epilogue_r5/variant_comparison/variants_by_shape.csv)。新 kernel 为独立实验实现，位于 [shortk_epilogue_exp](reports/opus_resume_20260926/shortk_epilogue_exp/) 和 [k1536_epilogue_exp](reports/opus_resume_20260926/k1536_epilogue_exp/)，保留同库原版控制组；生产注册、9000 和默认选型均未修改。

本轮 CKTile 对照是同批重测的 27/28/29 和原 shape 的最快 CKTile，不是重新穷举所有后端。未重测完整 87/295 项，不能将 18 项直接相加得到新的全量胜负。下文的 **209/86 完整基线**及原 87 项成员继续保留；旧“33 项未完成”指当时中断的全后端 sweep，新完成的是上述目标 kernel 比较。下一步应沿用已经证明有效的输出重排方向，勿把无收益的 scale/XOR 微调当作主线。

## 2026-09-25 迁移交接记录

本分支：`Fyzyukk/aiter:aiter-opus-mxfp8-bpreshuffle`。实验源码提交为 `b7df6147`。本次交接把当前实验源码、最新完整基线、未完成实验记录及复跑工具提交到同一分支，供换服务器继续优化。本次没有启动 GPU 测试，也没有等待原机空闲。

## 先看结论和未完成事项

- **最新完整基线为 209 胜 / 86 负**：295 个支持的 M≥1024 模型 shape，另 10 项超出既有单张量有符号 32 位字节寻址范围。基线代码是 `10ab50645d1f25e11844b814b66002b27181dfbf`，已同步 upstream `b3d0cf4e`。此前 208/87、206/89 属于旧批次。
- **优化范围仍保留原 87 项**：上述 86 个负项，以及 `(4096,2048,7168)` 这个仅领先约 0.033% 的近平局项。不能因为最新基线少了一个负项就删除它。
- **9030–9033** 已接入注册和代码生成，GPU 边界检查通过；87 项五轮全后端扫描未完成，不能据此给出完整胜负统计。
- **9040–9042、9050–9051** 已实现，离线编译、CPU 检查和首次 GPU 正确性验证均通过；33 项五轮性能扫描未完成，尚未关闭任何优化目标。
- 9000/9010/9011/9012/9020 保持原实现；新 ID 均为可显式调用的实验候选，未加入默认编译集合，未更改生产调度 CSV。

最先补测 **25 个短 K + 8 个 K1536**；再按原 87 项分组继续，最终完整复核 295 项及原有胜项。要分别记录“超过旧 OPUS”和“超过最快有效外部后端”。

## 已恢复的证据

| 记录 | 当前可用结论 | 入口 |
|---|---|---|
| 当前分支全量重建、扫描、确认、回放 | 295 项；OPUS 209 胜、CKTile 86 胜；16 项追加五轮确认；295 项选择回放通过 | [完整结果](reports/opus_local_gap_current_20260925/RESULTS.md)、[最终比较](reports/opus_local_gap_current_20260925/final_comparison.csv)、[落后清单](reports/opus_local_gap_current_20260925/final_remaining_shapes.csv) |
| 当前全量候选数 | 27,690；OPUS 1,275 全有效；CK 5,160/5,310 有效；CKTile 9,535/9,735 有效；ASM 0/11,370 有效 | [全部候选记录](reports/opus_local_gap_current_20260925/profile.csv)、[拒绝记录](reports/opus_local_gap_current_20260925/rejected_candidates.csv) |
| 9030–9033 GPU 验证 | 156 项数值/保护区检查通过，12 项非法对齐被拒绝 | [validation.json](reports/opus_9030_targets87_20260925/validation.json) |
| fixed-K GPU 验证 | 96 项数值检查通过：41 项目标调用、50 项边界、5 项 raw E8M0；40 项非法 shape 被拒绝 | [validation.json](reports/opus_fixedk_gpu_20260925/validation.json) |
| fixed-K CPU/编译 | 短 K 集成检查 103/103；K1536 集成检查 127/127；布局/调度检查及 gfx950 编译通过 | [短 K](reports/opus_shortk_20260925/README.md)、[K1536 集成](reports/opus_k1536_20260925/integration/REVIEW.md)、[K1536 编译资源](reports/opus_k1536_20260925/offline/README.md) |

短 K / K1536 目录内较早文档的“GPU 待验证”描述对应当时的 CPU 阶段。以较新的 `opus_fixedk_gpu_20260925/validation.json` 和本交接页为准；性能验收仍未完成。本次交接重新执行 K1536 标准库集成检查，127/127 通过。

**中断记录的解释：** `opus_9030_targets87_20260925` 的四个 `gpu*_r5_run.json`，以及 `opus_fixedk_gpu_20260925` 的三个同名文件都残留 `status=running`。本次恢复时当前进程空间没有对应测试进程；这些是未正常收尾的记录，不能当作仍在后台执行或已完成。

- 903x 原始计时各卡分别为 103/100/99/100 行，都只触及分片的第一个 shape，未完成五轮。
- fixed-K 原始计时各卡分别为 327/315/361 行，每卡触及两个 shape；每卡仅首个 shape 有五轮汇总。不要把三个局部结果外推成 33 项结论。
- 原机器持续有外部占用，保留部分记录仅用于追溯。新服务器从全新 prefix 开始，同一个 shape 的所有候选在同卡同批重新测量。

## 源码入口和优化方向

注册：[opus_gemm_common.py](csrc/opus_gemm/opus_gemm_common.py)；生成：[gen_instances_gfx950.py](csrc/opus_gemm/codegen/gen_instances_gfx950.py)；调用：`aiter.ops.opus.opus_gemm(..., kid=..., layout="bpreshuffle", x_scale=..., w_scale=...)`。

pipeline/traits 位于 `csrc/opus_gemm/include/gfx950/`，公共文件名前缀为 `opus_gemm_{pipeline,traits}_a8w8_mxscale_bpreshuffle_`。

| ID | 几何/调度 | 文件后缀和说明 |
|---|---|---|
| 9000 | 256×256×128，4 wave | `4wave_gfx950.cuh`；冻结基准，保留 pipeline、traits 和影响它的共享 helper |
| 9010 / 9011 / 9012 | 128×128 / 64×128 / 64×64；K128 | 对应尺寸后缀；正式配置分别为 S3/K+2、S3/K+2、S4/K+3，直接 BF16 |
| 9020 | padded M | `padded_m_gfx950.cuh`，复用 9000 主体 |
| 9030 / 9031 | 192×256×128；4 / 8 wave；S64 | `192x256_gfx950.cuh` |
| 9032 / 9033 | 同几何；4 / 8 wave；S128 | 同上，scale panel 更大 |
| 9040 / 9041 / 9042 | 192×256×128，8 wave；固定 K384 / K768 / K1024 | `shortk_gfx950.cuh`；完整展开 3/6/8 步，只准备实际 scale |
| 9050 / 9051 | 同几何；固定 K1536 | `k1536_gfx950.cuh`；完整展开12步 / 两步循环 |

固定 K 入口要求精确 K，正 M/N、M%64=0、N%256=0，保留 FP8/BF16、原生 E8M0 布局、对齐和字节范围检查。固定 K 不能用于其他 K；过滤逻辑为 `a8w8_mxscale_bpreshuffle_supports_shape`，generated launcher 也有精确 K guard。

优先关注寄存器压力：9041 为 256 VGPR、无 spill；9050 为 256 VGPR、22 VGPR spills、76 private bytes；9051 为 206 VGPR、无 spill。编译资源只提供优化线索，不能替代 GPU 性能比较。不要为了减少等待而删除必要同步，也不要从短 K 结果外推长 K。

原 87 项分组为 **25 + 8 + 12 + 26 + 10 + 3 + 3**：短 K、K1536、N7168/K3072、N6144或7168/K7168、N7168/K16384、窄 N768、大 M/N2048。精确成员与旧基线冻结在 [分组台账](reports/opus_shortk_20260925/plan/targets87_ledger.csv) 和 [cohorts.csv](reports/opus_shortk_20260925/plan/cohorts.csv)。

## 新服务器准备

```bash
git clone --branch aiter-opus-mxfp8-bpreshuffle --single-branch \
  git@github.com:Fyzyukk/aiter.git
cd aiter
git submodule update --init --recursive
git rev-parse HEAD
git -C 3rdparty/composable_kernel rev-parse HEAD
```

CK submodule 应为 `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`。本交接已把当前修改提交进分支，不需要再应用旧 overlay 或旧 patch。

历史环境是 MI355X / gfx950 / 256 CU，Python 3.12、PyTorch `2.11.0+rocm7.14.0`、HIP `7.14.60850`。需要原生 `torch.float8_e8m0fnu`、可用的 ROCm/PyTorch 开发环境，以及 `rocm-smi`。新机环境不同，应保存版本并重建该环境的基线。已有匹配的 ROCm/PyTorch/Triton 环境中，可按仓库安装方式执行：

```bash
BUILD_TARGET=rocm AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  python -m pip install -e . --no-build-isolation
```

**OPUS 需要定制 clang。** 9000 使用 `clang::amdgpu_pin_agpr`；新候选引用相关布局 helper，也会遇到该编译器要求。普通 ROCm clang 不能直接代替。历史 OPUS 编译器为 `https://github.com/yuyzhang512/llvm-project.git` 的 `49c41889681640665400cb01c9fbb4c0a024cde4`（clang 24）；CK/CKTile/ASM 使用 ROCm clang 23。可迁移已构建工具链，或在新机按旧配置构建：

```bash
git clone https://github.com/yuyzhang512/llvm-project.git llvm-pin-src
git -C llvm-pin-src checkout 49c41889681640665400cb01c9fbb4c0a024cde4
cmake -G Ninja -S llvm-pin-src/llvm -B llvm-pin-build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_PROJECTS='clang;lld' -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build llvm-pin-build --target clang lld --parallel 12
```

保留 `aiter/jit/optCompilerConfig.json` 的编译语义。完整编译器版本记录见 [build.json](reports/opus_fixedk_gpu_20260925/build.json)。工具链、ROCm/PyTorch 和 `.so` 不随 Git 推送，必须在目标机器准备。

## 生成适配新机的入口

使用 [prepare_remote.py](reports/opus_remote_handoff_20260925/prepare_remote.py)，只读归档模板，在 `reports/` 下生成新目录。它不导入 torch/aiter、不查询 GPU、不编译、不测试，也不会覆盖已有目录。

先在新机核对同一张空闲卡的 **rocm-smi 物理编号、ROCr UUID 和 PCI bus**。ROCr 数字编号可能与物理编号不同。下面三个设备值必须替换；`85` 是 PCI `0000:85:00.0` 的十六进制 bus 字节。当前 harness 限定 PCI domain/device 为 0、gfx950/256CU。

```bash
python reports/opus_remote_handoff_20260925/prepare_remote.py \
  --output-dir reports/opus_remote_run \
  --gpu-index 0 --gpu-uuid GPU-0123456789abcdef --pci-bus 85 \
  --opus-clang-path /absolute/path/to/llvm-pin-build/bin \
  --stock-clang-path /opt/rocm/llvm/bin

# 仅列清单，不访问 GPU。
python reports/opus_remote_run/benchmark.py \
  --shapes reports/opus_remote_run/shapes33.csv --list-shapes
```

生成目录包含全新 `build.py`、`validate.py`、`benchmark.py`、最新 tuner adapter，以及 25/8/33/87/295 项 shape CSV。构建脚本从源码构建五个后端模块，OPUS 显式包含全部 14 个候选；不会复制旧机的 `.so`。计时脚本保留空闲检测、实际加载路径和源码/二进制哈希检查。

这些迁移入口已做 CPU 生成、语法、shape 清单检查；新机编译和 GPU 运行仍需执行。不要直接启动归档目录里的 `launch.py`，也不要直接跑归档 `build.py`：旧脚本有固定设备/路径，部分还需要旧 JIT 缓存。

## 新机重建、正确性和计时

在仓库根目录依次执行；每个命令成功后再继续。构建较重，编译时不要同时计时。

```bash
python -u reports/opus_remote_run/build.py
python -u reports/opus_remote_run/validate.py

# 先补完整 25+8 项，每个候选先检查数值，合格才计时。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 5 \
  --shapes reports/opus_remote_run/shapes33.csv \
  --jit-dir reports/opus_remote_run/jit --prefix fixedk_r5

# 后续覆盖原87项，包括全部合法旧/新OPUS和参考后端。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 5 \
  --shapes reports/opus_remote_run/shapes87.csv \
  --jit-dir reports/opus_remote_run/jit --prefix targets87_r5

# 候选优化完成后，完整回归，避免丢掉已有胜项。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 3 \
  --shapes reports/opus_remote_run/shapes295.csv \
  --jit-dir reports/opus_remote_run/jit --prefix all295_r3
```

`validate.py` 覆盖 fixed-K 的目标、边界、raw E8M0 和非法 K/对齐拒绝。benchmark 对每个合法候选独立检查数值并复查输出；修改 903x 或其他旧 kernel 时，还应适配其专项边界回归，不能只看性能目标。旧 903x 验证脚本在 [validate.py](reports/opus_9030_targets87_20260925/validate.py)，使用前需另行调整其固定 GPU 和输出目录。

若 GPU 空闲检查失败，保留此次输出，以新 prefix 重跑该 shape 的完整批次；不要删除检测或混合两次计时。prefix 相对路径落在新生成目录下，输出文件若已存在会拒绝覆盖。`validate.py` 和 `build.py` 的结果也应使用新目录保存，避免覆盖上一版控制组。

接近 ±3% 或轮次胜负不一致的 shape，从 `*_choices.csv` / `*_raw.csv` 另建包含 M/N/K 的 CSV，然后用 **同一次完整 sweep、同卡、同一套源码/二进制** 追加五轮：

```bash
python -u reports/opus_remote_run/benchmark.py \
  --final reports/opus_remote_run/all295_r3_run.json --rounds 5 \
  --shapes reports/opus_remote_run/close_shapes.csv \
  --jit-dir reports/opus_remote_run/jit --prefix close_r5
```

`--final` 保留全部合法 OPUS，以及有效外部对手中距最快者≤5%的候选。确认批次整体替换该 shape 的旧批次，不能挑两批最小值。完成性能选型后，还需把最终选择通过原生 E8M0 接口回放；旧 [replay.py](reports/opus_local_gap_current_20260925/replay.py) 可作实现参考，其历史路径和 GPU 映射需要适配。

## 必须保留的比较口径

- 原 FP32 逐元素误差界：`1e-4 + 5e-5 * sum(abs(A_i*B_i))`，区间端点舍入 BF16，`error==0`，保留 NaN 预填和输出保护区。数值失败不参与性能选优。
- 使用原 `run_perftest` profiler，warmup=5、iters=51、自动输入轮换，轮换候选顺序。公共 B shuffle、scale 解码和参考计算不计时；后端内部转换计时；CPU launch 开销不计时。
- 每个 shape 的全部对比在同一张实际空闲 GPU 上完成。新机器的耗时不与旧机器逐项拼接；旧值只作参考。
- 差距为 `(OPUS_us / reference_us - 1) * 100%`，正值表示 OPUS 更慢。上游 CSV 的历史 us 不直接替代同批重测值。计时条件差异分析见 [调查记录](reports/cktile_timing_diagnosis_20260925/README.md)。
- 9000 保持冻结；新方案用独立候选和独立 JIT 目录。新候选通过前不改默认选型；最终保留原 87 项成员、10 项寻址排除和全 295 项覆盖。

## 交回时保留什么

源码 commit、环境/编译命令、新控制组与候选的源码/二进制 SHA256、设备 UUID/PCI、完整 raw/correctness/summary/run 文件、最终 295 项比较与选择、87 项进度和仍未解决项。报告区分数值通过、超过旧 OPUS、超过最快外部对手三种结论。

本分支保存 2026-09-25 的文本报告、原始 CSV/JSON、验证源码及此前 9030 实验记录；更早的完整本地实验目录不在本次交接范围。文件清单和 SHA256 见 [manifest.json](reports/opus_remote_handoff_20260925/manifest.json)。`reports/.gitattributes` 保留证据原始字节，避免 CSV 换行归一化破坏历史哈希。

旧 JSON 内的绝对路径与二进制哈希描述原机，不要求新编译产物匹配旧哈希。未上传 JIT 缓存、编译二进制及完整工具链；新生成的运行入口以新哈希记录本次执行。


## Historical snapshot: 2026-09-27 single-flow queue before interruption

Latest scope: leave 9000/9020 and shared helpers unchanged; consolidate every other
retained candidate, including 9010/9011/9012, into one main loop and output flow per
geometry. Work and live status: `reports/opus_merge_flow_20260927/RUNNING.md`.
Two 62-shape/3-round pilots passed numerical checks. Five new geometries in three
families are implemented; small v3/v4 and tiny v3 await numerical/performance data.
At that earlier handoff, eight cards had outside activity and a queue was waiting.
Recovery subsequently confirmed that those processes no longer exist and all
eight pilot v34 summary files contain zero data rows. The JSON queue states are
historical remnants. Use the new-machine entry at the top of this document;
full295 selection and final cleanup remain pending.
