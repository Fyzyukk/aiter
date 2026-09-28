> 最新87项优化与外部基准结果见 [后续报告](../opus_cover87_20260926/RESULTS.md)。本页保留前33项历史实验。

8-wave BF16 输出重排已把这 33 个固定 K 目标的性能提升到本批 CKTile 对手附近：相对原固定 K 控制，几何平均耗时下降 **26.71%**，33/33 更快；相对旧最快 OPUS，下降 **11.82%**，33/33 更快。与同批最快 CKTile 总体基本持平，几何平均耗时下降 **0.05%**，18 胜、15 负。结论来自已完成的 [epilogue_r5/launch.json](../opus_resume_run_20260926/epilogue_r5/launch.json)，没有拼接其他轮次的耗时。

| K | 8-wave 新 ID → 原固定 K 控制 | 目标数 | 相对控制：耗时降幅 / 更快项 | 相对旧最快 OPUS：耗时降幅 / 更快项 | 相对同批 CKTile：耗时降幅 / 更快项 |
|---:|---|---:|---:|---:|---:|
| 384 | 9640 → 9040 | 12 | **31.29% / 12** | **10.98% / 12** | −0.76% / 5 |
| 768 | 9641 → 9041 | 12 | **24.10% / 12** | **13.07% / 12** | +0.28% / 7 |
| 1024 | 9642 → 9042 | 1 | **23.42% / 1** | **15.32% / 1** | +4.07% / 1 |
| 1536 | 9651 → 9051 | 8 | **23.83% / 8** | **10.75% / 8** | +0.39% / 5 |

表内降幅为 `100 × (1 − 几何平均(新耗时 / 对手耗时))`；正数表示更快。“更快项”按每个 shape 的五轮中位数比较。控制取同一实验动态库中的对应 `base_kid`；旧 OPUS 取 `{9000, 9010, 9011, 9012, 9020}` 的同批最优；CKTile 取本批测到的候选最优。所有比值先在同 shape、同 GPU、同批次内计算，再汇总，不跨 shape 或 GPU 平均绝对耗时。8-wave 新版相对控制的逐项降幅为 21.67%～33.96%，相对旧 OPUS 为 2.65%～20.41%；相对 CKTile 为 −3.40%～+4.07%，目前仍有 15 项稍慢。K1024 仅有一个目标，表中结果只代表该项。

这次有效改动集中在输出阶段：计算结束后复用已有 LDS，把 MFMA 分散的 BF16 结果重排到 pitch=264 的输出 tile，再由线程连续写出 8 个 BF16，即每次 16 字节。192×264×2=101376 字节的输出 tile 没有增加原 LDS 分配；8-wave 对照保持原计算、scale 布局和同步语义，区别是输出重排。同轮也测了 4-wave 重排版，全部结果保留在逐变体 CSV 中。

四轮分别比较了 scale 调度/读取、4-wave 几何、XOR LDS 布局和 BF16 输出重排；主要提升来自第四轮输出重排。四轮记录分别保存在 `experiments_r5/`、`fourwave_r5/`、`xor_r5/`、`epilogue_r5/`，上表只使用最后一轮。

本批 **33/33 目标全部完成，所有参与比较的候选每个 shape 均有五轮计时**：549 个候选与 shape 的组合，2745 条计时记录；已有记录中的 3294 次正确性检查全部通过，包括 549 次首次调用检查和 2745 次计时后检查，逐元素 `error=0`，输出保护区全部保持完整。比较保留原 FP32 累加误差界、NaN 输出预填、64 行前后保护区及原 `run_perftest` profiler：warmup=5、iters=51、自动参数轮换。公共 B shuffle、scale 解码和参考计算不计时，后端内部处理计时。

此次只测指定的 **25 个短 K + 8 个 K1536 目标**，测量对象为独立新变体、控制、合法注册 OPUS，以及 CKTile 27/28/29 和历史最快 CKTile 候选。18/33 只表示对本批 CKTile 对照的中位数优势，其中多数不足 3%。本次没有全后端扫描，也没有覆盖原 87 项的其余目标或完整 295 项，因此既有 **209 胜 / 86 负**的全量基线保持原记录，不据此改写。没有启动下一批测试，没有更改默认生产选型。

已有数据与源码入口：

- [最终 8-wave 的 33 项结果](../opus_resume_run_20260926/epilogue_r5/variant_comparison/selected_8wave_shapes.csv)、[仍落后本批 CKTile 的 15 项](../opus_resume_run_20260926/epilogue_r5/variant_comparison/remaining_cktile_shapes.csv)。
- [逐 shape、逐新变体比较 CSV](../opus_resume_run_20260926/epilogue_r5/variant_comparison/variants_by_shape.csv)、[按 K 汇总 CSV](../opus_resume_run_20260926/epilogue_r5/variant_comparison/variants_by_k.csv)、[汇总 JSON](../opus_resume_run_20260926/epilogue_r5/variant_comparison/summary.json)。这些文件包含 8-wave 和 4-wave 变体，控制组不计为新优化。
- [原始结果目录](../opus_resume_run_20260926/epilogue_r5/)：短 K 为 `gpu{0..7}_shortk_epilogue_r5_raw.csv`，K1536 为 `gpu{0..7}_k1536_epilogue_r5_raw.csv`；各前缀配套 `_correctness.csv`、`_summary.csv`、`_run.json`。例如 [GPU0 短 K 原始计时](../opus_resume_run_20260926/epilogue_r5/gpu0_shortk_epilogue_r5_raw.csv)、[GPU0 K1536 原始计时](../opus_resume_run_20260926/epilogue_r5/gpu0_k1536_epilogue_r5_raw.csv)。总 manifest 保存全部分片路径、GPU UUID、命令和哈希。
- 短 K 8-wave kernel：[pipeline_lds_8wave.cuh](shortk_epilogue_exp/pipeline_lds_8wave.cuh)；对照：[pipeline_control_8wave.cuh](shortk_epilogue_exp/pipeline_control_8wave.cuh)；[入口与 ID 分派](shortk_epilogue_exp/launch.hip)、[变体映射](shortk_epilogue_exp/variants.json)、[构建记录](shortk_epilogue_exp/build_manifest.json)。
- K1536 8-wave kernel：[pipeline_epilogue8.cuh](k1536_epilogue_exp/pipeline_epilogue8.cuh)；[入口与 ID 分派](k1536_epilogue_exp/launch.hip)、[变体映射](k1536_epilogue_exp/variants.json)、[构建记录](k1536_epilogue_exp/build_manifest.json)。
- [测量脚本](../opus_resume_run_20260926/experiment_benchmark.py)、[纯数据汇总脚本](../opus_resume_run_20260926/summarize_experiments.py)、[新机基础构建记录](../opus_resume_run_20260926/build.json)。所有新 kernel 仍位于独立实验目录。
