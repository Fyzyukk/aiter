# gfx950：2026-10-30 本轮调优结果与优化记录

本轮已于 **2026-10-30 07:01:53 UTC** 完成 **421/421** 个目标 shape。最终 Opus 相对固定 CSV **236 个领先、185 个落后**，全部通过误差上限 0.05，最大记录误差 0.0009。本目录归档最终 CSV、逐项比较、测量记录以及后续 CPU 分析；归档过程没有启动 GPU 或修改 kernel/CO/JIT。

## 主要文件

| 文件 | 内容 |
|---|---|
| [full421_tuned.csv](full421_tuned.csv) | 本轮最终 421 项 Opus 结果：shape、kernelId、完整 kernelName、splitK、耗时、误差 |
| [full421_comparison.csv](full421_comparison.csv) | 每项与固定 gfx950 CSV 的对比、基线身份、是否领先及节省比例 |
| [full421_comparison.json](full421_comparison.json) | 完成数、236/185 统计、落后 N/K 分组与续跑设置 |
| [optimization_records.md](optimization_records.md) | 本轮候选与测量口径、最终交叉分类、14 项回退分析、171 项优化方向和待做事项 |
| [四步交接记录](../handoff_4steps_zh.md) | 步骤 1–4 的输入、候选补齐、续跑和最终完成记录 |
| [full421_progress.jsonl](full421_progress.jsonl) | 421 项累计完成记录；最终结果读取各行的 `best_opus` |
| [gpu_completion_receipt.json](gpu_completion_receipt.json) | 测量完成时间、进程退出记录和最终文件指纹 |
| [upload_manifest.json](upload_manifest.json) | 归档文件、原文件 SHA256、副本 SHA256 及归档方式 |

## 比较口径和计时轮次

基线为 [固定 csv_baseline.csv](../handoff_data/csv_baseline.csv)，SHA256 为 `d4dccd2826240b5a6467ddb1467095d1f96a36053061ed96a340cb123486d286`。只比较目标 421 keys 对应的 **gfx950 / cu_num=256 / w_scale_block=128x128** 行。`opus_us < baseline_us` 记为领先；本轮续跑只搜索 Opus，FlyDSL 直接引用固定 CSV。

最终结果保留已经完成的 **292 项**原轮次耗时；剩余 **129 项**在 06:03 UTC 后使用每 pipeline Top4 加兼容历史 seed、11 次计时、2 次预热快速完成。原完整搜索使用 Top8、101 次计时、5 次预热。`full421_comparison.json` 中 Top4/11/2 是最终续跑参数，不能将其解释成全部 421 项均采用短计时。逐项轮次在回退和持续落后分析 CSV 中保留。

累计 JSONL 还保留早期同轮 FlyDSL 的历史字段；`best_libtype` 可能为 FlyDSL，不能用它重新选择本轮最终结果。归档的终表全部为 Opus，并与每行 `best_opus` 的 ID、名称、splitK、耗时和误差一致。

## 回退与持续落后记录

| 历史同一 CSV 分类 | 当前领先 | 当前落后 |
|---|---:|---:|
| 历史领先 192 项 | 178 | 14 |
| 历史落后 229 项 | 58 | 171 |

当前 185 个落后项由 14 个新增回退和 171 个持续落后构成，两个集合没有重复。

- **14 项新增回退：** [明细 CSV](regressions14_audit.csv)、[统计 JSON](regressions14_audit.json)、[原因与幅度](cpu_regressions14_preparation/analysis_zh.md)、[精确候选](regressions14_pair_candidates.csv)、[复核准备包](cpu_regressions14_preparation/README.md)。13 项落后不到 3%，1 项落后 15.116%；复核脚本已准备，尚未执行 GPU 复核。
- **171 项持续落后：** [明细 CSV](persistent171_analysis.csv)、[分组 CSV](persistent171_groups.csv)、[分析](persistent171_analysis.md)、[完整 CPU 方案](cpu_persistent171_plan/README.md)。其中 92 项已比历史 Opus 改善，但仍没有胜过固定 CSV。
- **三条具体路线：** [4wave](cpu_persistent171_plan/4wave_plan.md)、[interleave](cpu_persistent171_plan/interleave_plan.md)、[窄 N](cpu_persistent171_plan/narrow_plan.md)。新排程仍是设计，尚未实现或构建；后续首筛共 11 个代表 shape、35 对配置，尚未测量。
- **候选选择与身份：** [search_seeds.csv](search_preparation/search_seeds.csv)、[candidate_coverage.csv](search_preparation/candidate_coverage.csv)、[identity_mappings.json](search_preparation/identity_mappings.json) 保存原准备阶段的 seed、覆盖和映射记录；它们不是逐候选实测耗时。

## 归档与复现边界

测量及 CPU 分析所用源码版本为 [488712ab](https://github.com/Fyzyukk/aiter/commit/488712ab1776ac819b69c26d0dc9ce8bd23c91b2)，远端分支为 `fyz/gfx950-search-identity-20261029`。原运行目录为 `/home/fyz/retune_dsv4_20261028/gpu_stage3_20261029T161551Z/stage4_full421`。

CSV、JSON、JSONL 和脚本按原字节复制；Markdown 中指向源码的本机链接改为仓库相对链接，CPU 准备包 README 增加归档说明。原文件与副本的指纹分别保存在 `upload_manifest.json`，本目录不改变原冻结输入。

CPU 准备包的 JSON、指纹和脚本保留原环境路径，依赖原环境冻结的 SO/CO、launcher 与计时源码；原 CPU 报告认证的是原准备包。它们是复核过程和设计的记录，不构成在新 checkout 中已验证的执行包。本次未运行归档脚本，未上传新性能结果或构建新 kernel。后续恢复测量前需重新核对产物、输入和路径；GPU 执行仍需后续明确要求。
