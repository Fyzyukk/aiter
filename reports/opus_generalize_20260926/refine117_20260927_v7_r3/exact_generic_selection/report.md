# 通用候选精确选择 / Exact generic portfolio

在本次报告声明的 **20** 个通用候选中，1% 条件下最少需要 **8** 个，另保留原始 5 个注册内核。
The exact minimum is **8 generic candidates + old5** within the declared measured pool.

保留 / Retained: long_runtime:20000, short_runtime:20010, short_runtime:20011, n224_runtime:20020, long_runtime_grid:20100, short_runtime_unified:20125, long_runtime_fused:20128, short_runtime_group4_cache2:20131.

每个形状均不超过同报告完整 generic+old5 包络的指定容差，并保留包络的全部外部中位数胜场。
Every shape remains within the stated envelope tolerance and preserves every envelope external median win.

| 容差 / Tolerance | Generic + old5 | 外部胜/负/平 W/L/T | Geomean gap | Worst gap | Feasible minimum subsets |
|---|---:|---:|---:|---:|---:|
| 0% | 12 + 5 | 291/4/0 | +0.0000% | +0.0000% | 1 |
| 1% | 8 + 5 | 291/4/0 | +0.0081% | +0.9566% | 2 |
| 2% | 6 + 5 | 291/4/0 | +0.0470% | +1.9823% | 4 |

同数量可行集合按逐形状几何平均延迟、最差偏差排序；正偏差表示更慢。穷举计数记录在 summary.json。
Equal-count feasible subsets are ranked by equal-shape geomean latency, worst gap, and stable labels; positive gaps mean slower.

这是已测候选的离线组合；生产调度尚未计时验证。每个形状保留其原始完整 worker、GPU/UUID、轮次和外部参考；不跨批次拼接候选计时，不使用固定 K 实验。
Measured portfolio selected from one audited refined295 report, not a benchmarked production dispatch. Each shape uses its complete selected worker's candidate and external medians; missing candidates remain unavailable. Original registered old5 are free, fixed-K experiments are excluded, and no historical timings are mixed.

Rounds by shape: `{'3': 295}`; external policies: `{'finalists': 295}`.

配置中的 `registered_opus_ids` 仅声明本报告的 old5 成员；现有实验配置加载器不消费该字段，因此它不能限制实际注册内核池。生产或受限调度均未在此实测。
The existing experiment loader does not consume `registered_opus_ids`; this declaration does not enforce a restricted registered pool. Neither production nor restricted dispatch is benchmarked here.

`selected_experiments.json`: 相对目录及 IDs / relative library directories and IDs.
`choices295.csv`: 每个形状的选择、原候选表行号、来源及 GPU/轮次 / per-shape choice and provenance.
`candidate_usage.csv`: 用量与移除影响 / usage and constraint failures after removing a retained generic.
`minimum_subsets.csv`: 各容差全部最小可行集合及排名 / every feasible minimum subset and ranking.
`summary.json`: 审计、精确搜索计数、指标及哈希 / audits, search counts, metrics, and hashes.
