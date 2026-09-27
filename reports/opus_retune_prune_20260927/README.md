# 2026-09-27：8 卡完整重新 tune 与候选裁剪

本页记录旧16候选池的历史全量结果。当前main/small/narrow合并版的新机器构建、
295项调优命令和历史基线入口见
[可移植调优说明](../opus_remote_tune_20260927/README.md)。
下方旧launch命令描述原机器运行环境，不是新机器入口；
summary中的旧绝对路径和已清理文件哈希仅保留历史来源。

任务完成。完整 295 个 shape、8 张 MI355X/gfx950、每个合法候选 3 轮，22.20 分钟。每个 shape 的全部候选在同一卡同批测量，完整枚举 CK、CKTile、ASM，不复用历史耗时。

整体选择：**OPUS 291 项、CKTile 4 项**。OPUS 相对最快有效外部的等权几何平均耗时降低 **12.9453%**，相对本批原 5 降低 **4.2688%**。279 项 3/3 轮领先、12 项 2/3 轮领先。全部 4575 条合法 OPUS 候选记录通过原 FP32 误差界和 NaN 输出保护区检查。

最终保留 **16 个 OPUS：原 5 + 11 个 runtime-K**，逐 shape 直接采用本批最快有效候选，没有使用上一轮 1% 子集损失阈值。

- [16 个候选的模板类型、几何、K 范围、选中次数和源码](SELECTED_CANDIDATES.md)
- [最终候选配置（指向现存源码包）](selected_experiments.json)
- [保留的 kernel 源码与独立构建入口](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/README.md)
- [本批 295 项整体选择](full295_r3/results/overall_selection.csv)、[25 候选用量](full295_r3/results/candidate_usage.csv)
- [完整计时报告](full295_r3/results/RESULTS.md)、[原始数据与来源审计](full295_r3/results/summary.json)

未选通用 ID **9661、9663、20120、20121、20122、20123、20127、20129、20130** 已移除。旧的 **9030–9033、9040–9042、9050–9051** 注册、构造器、生成分支和 5 个无依赖头文件一并删除。

清理当前通用化实验及其固定 K 前身的 50 个独立实验目录，共删除 **380 个源码、生成器、二进制与构建产物**；原测量 CSV/JSON/日志/文档保留。含退役 ID 的测量 JIT 已换为新构建的原 5 子集，移除其 788 个旧生成缓存文件。更早混合仓库快照作为历史档案保留，不是当前候选池。

[删除文件清单](deleted_experiment_files.json)、[生产裁剪记录](production_prune_applied/prune_manifest.json)、[JIT 替换记录](pruned_jit_install.json)。原 5 全部生成文件在裁剪前后逐字节相同；9000 本体及共享 helpers 未改。新通用源码包的 9 个库实际仅有 11 个 device kernel，全部机器码与本次实测版本逐字节一致，见 [device_audit.json](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/device_audit.json)。

4 个外部胜项均为 N=7168、K=384：M=1536/1600/1664/1728，最快 OPUS 分别慢 1.569%/3.237%/1.442%/0.376%；本批各 0/3 轮领先，[详情](full295_r3/results/remaining_losses.csv)。

11 个通用 ID 使用独立 C ABI 库；原 5 保留既有生产注册。未把 20000 段实验 ID 写入全局注册（该范围与 gfx1250 ID 重叠），未改默认 dispatch CSV。没有额外 GPU 测试或重复测到获胜。本次没有提交或推送。

原始 full295_r3 下的 manifest、配置与来源路径描述清理前的冻结测量；需要使用现存 kernel 时用本页 selected_experiments.json。复跑保留池可执行：

```bash
python reports/opus_retune_prune_20260927/launch.py \
  --experiments reports/opus_retune_prune_20260927/selected_experiments.json \
  --batch retained_new_r3 --rounds 3 --external-mode full --gpus 0,1,2,3,4,5,6,7
```

该命令要求 GPU 空闲，输出使用新目录；此次完成的 295 项结果无需重跑。
