# gfx950 MXFP8 tune 分析与 CO 恢复记录

本目录保存 2026-10-28 的 tune 分析，以及 2026-10-29 build-co 筛选恢复后的状态，供另一台机器接续使用。

分支：`fyz/gfx950-co-spill-restore-20261029`，基于本地 `fyz/opus_blockscale_mxfp8_gemm` 的 `d4492067`。本次只向新分支 push。

## 当前状态

- gfx950 CO manifest 声明 304 个候选；262 个有可用镜像，42 个编译时寄存器分配失败。
- build-co 默认保留通过 pin 校验的 spill/scratch 候选，并记录资源警告。显式 `--fail-on-spill`、`--fail-on-scratch` 可恢复严格过滤。
- 11 个此前被资源筛选拒绝的 CO 已恢复，其中补 PAD 的 `sp16/32/64 × u2/u4` 六种组合全部可用。
- 当前候选路由为：显式 pin 配置走 CO，非 pin 配置走 JIT。`4wave_pin_agpr` 的 PAD/non-PAD 两条路径均有显式 pin，当前均走 CO。
- 模型配置 `aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv` 当前 1,236 行均为 FlyDSL；没有将分析中的 OPUS 胜出项写入默认配置。
- 几何筛选已改为默认 Top8，可用 `--opus_top_geometries 0` 保留全部合法几何，并可按 pipeline 和 JIT/CO 路由筛选。
- m_repeat 新增 102 个非 pin JIT 变体，JIT 候选由 240 增至 342；原有候选编号保持不变。已完成 CPU 覆盖和代表配置编译检查，没有新增测试文件，没有追加 GPU 正确性或性能复测。

## 文档和数据

| 文件 | 内容 |
| --- | --- |
| [tune_analysis_zh.md](tune_analysis_zh.md) | 421 个 shape 的领先幅度、落后原因和候选覆盖分析，含逐条胜出表 |
| [build_co_analysis_zh.md](build_co_analysis_zh.md) | gfx1250/gfx950 筛选差异、pin 验证含义、11 个恢复配置和验证范围 |
| [next_steps_zh.md](next_steps_zh.md) | 新机器拉取、使用已提交 CO、可选重建、后续 tune 和优化顺序 |
| [cpu_search_optimization_zh.md](cpu_search_optimization_zh.md)、[cpu_candidate_coverage.csv](cpu_candidate_coverage.csv) | Top4/Top8/全量的 CPU 覆盖比较、m_repeat JIT 新变体与验证范围 |
| [all_shapes_421.csv](all_shapes_421.csv) | 所有 shape 的耗时、候选覆盖和 kernel 记录 |
| [baseline_wins_192.csv](baseline_wins_192.csv) | 胜过保存 FlyDSL 基线的 192 项，每项的领先百分比和节省微秒数 |
| [fresh_wins_207.csv](fresh_wins_207.csv) | 胜过同轮 FlyDSL 的 207 项 |
| [old_opus_improvements_200.csv](old_opus_improvements_200.csv) | 比旧 OPUS 记录更快的 200 项 |
| [baseline_losses_229_with_evidence.csv](baseline_losses_229_with_evidence.csv) | 229 个基线落后项及已有证据、优化建议 |
| [candidate_coverage_421.csv](candidate_coverage_421.csv) | 原 tune 的合法/已测几何覆盖情况 |
| [best_measured_by_pipeline.csv](best_measured_by_pipeline.csv) | 各 pipeline 在每个 shape 上的实测最佳项 |
| [summary_by_nk.csv](summary_by_nk.csv)、[summary.json](summary.json) | 分组和总体统计 |
| [shapes_421.csv](shapes_421.csv)、[priority_missing_old_geometry_22.csv](priority_missing_old_geometry_22.csv) | 可用作后续 tune 输入的 shape 列表 |

原始统计中没有“201 个胜出”：保存基线口径为 192，同轮口径为 207，相对旧 OPUS 改善为 200。统计和候选覆盖数据均来自 CO 恢复前的快照；恢复 11 个 CO 后未重新计算性能排名。

原始 tune 工件位于构建机器的 `/home/fyz/retune_dsv4_20261028/tune_nonpin_20261028T162327Z_8gpu/`，本目录包含分析导出数据，不包含整个原始测量日志。恢复构建工件位于该机器的 `/home/fyz/retune_dsv4_20261028/restore_co_spill_policy_20261029T020553Z/`。
