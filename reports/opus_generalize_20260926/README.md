# 通用 OPUS kernel：最终结果（2026-09-27）

本轮完成。保留原有 **9000、9010、9011、9012、9020**，其余25个实验候选收敛为 **8个runtime-K候选**：20000、20010、20011、20020、20100、20125、20128、20131。没有预设必须剩4个；9000、共享helpers、生产注册及默认dispatch均未改。

[完整中文结果](refine117_20260927_v7_r3/review_results/RESULTS.md)：**291/295项快于同卡外部finalist**。等权几何平均耗时相对原5降低4.147%、相对外部降低12.973%；相对旧25+原5增加0.115%，最坏增加3.951%。原5选择200项，8个通用选择95项。8是逐shape相对完整通用池允许1%损失、同时保留所有外部胜项时的精确最少候选数。

最后测量为v6完整295项三轮与v7追加117项三轮；均使用物理GPU4–7，全部passed、数值检查0拒绝。最终保留178项v6整行和117项v7整行，每个shape的候选及外部耗时同卡同批，不混用历史最小值。两个队列已complete，本轮没有后台GPU工作。历史全后端扫描只提供外部候选身份，耗时全部重测；本轮不是新的全后端穷举。

主要产物：

- [候选配置](refine117_20260927_v7_r3/exact_generic_selection/selected_experiments.json)
- [295项选择和来源](refine117_20260927_v7_r3/exact_generic_selection/choices295.csv)
- [候选用量](refine117_20260927_v7_r3/exact_generic_selection/candidate_usage.csv)
- [同场新旧候选配对效果](refine117_20260927_v7_r3/review_results/candidate_effects.csv)
- [精确选择与候选数量取舍](refine117_20260927_v7_r3/exact_generic_selection/report.md)
- [完整测量审计](refine117_20260927_v7_r3/runtime_analysis/summary.json)
- [最终独立选型复核](v7_final_selection_review.json)、[结果复核](v7_results_review.json)

有效改动包括20100的窄N网格策略、20010/20011的小tile、20125的大M短K路径，以及20128的小幅融合输出收益。最终20131（20127恒cache2版本）在(1472,7168,384)为12.280889µs，外部12.379533µs，3/3轮胜。20129/20130虽无spill，但寄存器分配增加导致全部117项退化，已剔除。

剩余外部负项为N7168/K384、M1536/1600/1664/1728，分别慢2.888%/2.882%/2.110%/0.905%，均0/3轮胜。相对旧25+原5慢超过3%的8项也完整列在中文报告。性能结论限于实际模型shape；运行范围和入口约束见各library metadata。

这是已测候选的离线选择，生产dispatch未计时或修改。配置中的registered_opus_ids只声明原5成员；现有loader不消费该字段。完整恢复与阶段记录见根目录[HANDOFF_MXFP8.md](../../HANDOFF_MXFP8.md)。

下面保留2026-09-26最初的恢复记录；其中运行状态和首轮4候选说明均属于历史，以本页顶部最终结果为准。

## 恢复点

`../opus_cover87_20260926/generalize99_v1_r3/` 是最新中断批次。GPU0–3 因外部占用而失败；GPU4–7 各完成一个 shape 后中断。恢复时旧进程均已不存在，残留的 `running` 状态不代表后台仍在工作。原目录保留，不覆盖、不拼接其部分计时。

新测试限定物理 GPU4–7。ROCr UUID 与物理 PCI bus 已由 KFD topology 核对：

| 物理 GPU | ROCr UUID | PCI bus |
|---|---|---|
| 4 | GPU-23a6cd0d658d72b6 | 0000:85:00.0 |
| 5 | GPU-2d57f9bd7c2ee0fe | 0000:95:00.0 |
| 6 | GPU-56f0ab624008ec65 | 0000:E5:00.0 |
| 7 | GPU-5ff36708541c8ec0 | 0000:F5:00.0 |

原来分配到 GPU0–3 的 shape 会重新分片到后四卡。仅复用历史完整扫描中有效、距最快外部对手不超过 5% 的**候选身份**，其耗时全部在新卡重测；同一 shape 的新旧 OPUS 与外部对手始终在同卡同批比较。源 GPU 和实际测量 GPU 分别记录，不用旧卡耗时替代本次对照。

## 已编译的通用候选

| ID | 几何 | runtime K 支持 | 实际 kernel 数 |
|---|---|---|---|
| 20000 | 192×256，8 wave | 128–16384，步长128 | 1 |
| 20010 | 128×128，4 wave | 128–1536，步长128 | 1 |
| 20011 | 160×128，4 wave | 128–1536，步长128 | 1 |
| 20020 | 192×224，8 wave，融合最后 MFMA/BF16 输出 | 128–1536，步长128 | 1 |

源文件和已通过构建的二进制分别位于 `long_runtime/`、`short_runtime/`、`n224_runtime/`。全部要求 M%64=0；N 的要求分别为 256、128、128、896 的倍数，另保留既有字节寻址范围与接口要求。表中的支持范围描述入口约束，性能和数值结论仅覆盖实际测量的目标 shape。

对比配置 `experiments_v1.json` 包含旧25实验候选与新4候选，另由原 harness 枚举合法注册 OPUS。分析时单独计算通用候选池，不能把对照中的固定 K 候选计作新的通用成果。已有 runtime 候选 9661/9663/13163 也作为可选的精简池成员。

## 自动续跑

2026-09-26 08:38 UTC 已启动独立等待队列，PID 为11966。启动后的状态是 `waiting`，后四卡有外部任务；完整通用候选比较尚未完成。后续以 [queue_state.json](queue_resume99_v1_r3/queue_state.json) 的实时状态为准，日志为 [queue.log](queue_resume99_v1_r3/queue.log)。

队列每5秒只查询GPU4–7。连续3次四卡使用率均为0、GFX Activity计数不变、各卡已用显存小于2GiB后，自动运行新批次 `resume99_v1_r3`，再进行CPU结果汇总。计时过程仍保留原占用检测，失败后保存记录并退出，不反复重测。

实际启动命令：

```bash
/opt/python/bin/python -u reports/opus_generalize_20260926/wait_resume.py \
  --batch resume99_v1_r3 --gpus 4,5,6,7 --rounds 3
```

该队列为历史中断批次；当前续跑状态见本文顶部。完整批次成功后，`runtime_analysis/report.md` 和 `summary.json` 会给出通用池与旧25候选、有效外部对手的同批比较。分析穷举7个runtime候选的128种子集，并分别计算保留旧5个注册OPUS时的结果，列出最少候选数、使用次数、胜负与性能退化。

本次恢复只做了源码/编译产物静态审查、续跑计划与既有数据的CPU校验；尚未产生新的GPU性能结论。静态审查和待实测的优化方向见 [RUNTIME_REVIEW.md](RUNTIME_REVIEW.md)。
