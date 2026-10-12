# gfx950 blockscale 与 blockscale_mxscale 的 745 shape 比较

输入固定为 ROCm/aiter main 的提交 `4eecc6509f0713f5af1c8c642267f2e953a9af5f`（2026-10-11T11:44:30Z）。本报告比较 CSV 内的 `us`，没有重新运行 GPU benchmark，也不代表模型端到端加速。

匹配键为 `gfx, cu_num, M, N, K`。原 blockscale 有 745 个 gfx950 shape，均为 256 CU；mxscale 在 `w_scale_block=128x128` 下全部覆盖。mxscale 的 gfx950 共 1176 条记录，其中 854 条与这 745 个 shape 匹配：745 条 BMM 路径、109 条 shuffled-scale 2D GEMM 路径。

实际 raw-scale blockscale mxscale 路径按 `libtype=flydsl, kernelId=bmm` 选择记录。见 [选择逻辑](https://github.com/ROCm/aiter/blob/4eecc6509f0713f5af1c8c642267f2e953a9af5f/aiter/ops/gemm_op_a8w8.py#L977-L987)。这些 BMM 使用 B=1。

每个 shape 的加速比为 `blockscale_us / mxscale_bmm_us`；大于 1 表示 mxscale 更快。耗时下降百分比为 `100 * (1 - mxscale_bmm_us / blockscale_us)`。

| 统计口径 | 实际 BMM 路径 | 每个 shape 取 CSV 中最低 us，仅供参考 |
| --- | ---: | ---: |
| 几何平均加速比 | 1.199928× | 1.210697× |
| 算术平均加速比 | 1.224273× | 1.235106× |
| 中位加速比 | 1.177267× | 1.187595× |
| 每个 shape 各运行一次的总耗时比 | 1.216082× | 1.240221× |
| 总耗时下降 | 17.768702% | 19.369212% |
| 更快 / 相等 / 更慢 | 634 / 0 / 111 | 643 / 0 / 102 |

原 blockscale 的总耗时为 202455.4772 µs，实际 mxscale BMM 为 166481.767 µs。取 CSV 最快记录的参考结果为 163241.4474 µs。两个 mxscale 路径读取的 scale 布局不同，最低 us 的参考结果不意味着运行时会自动选择该记录，也不计 scale 布局转换开销。

`comparison_gfx950_745.csv` 当前保留 Opus 慢于 FlyDSL BMM 的 317 个 shape，列为 M、N、K 和 blockscale、FlyDSL BMM、Opus 三个耗时。文件名沿用最初的 745 shape 比较；本文及 `summary.json` 的整体统计仍基于完整 745 项，完整输入保存在三个原始 CSV 快照中。


## 加入 2026-10-08 最后完成的 Opus 745 shape 结果

Opus 数据来自 [最后完成的混合编译器全量调优](../opus_clang23_mixed_retune_20261008/README.md) 的 `tuned_opus.csv`，745 行均为纯 Opus。已逐行核对其为同轮原始 `profile.csv` 中最快的有效 Opus 候选，13,027 个 Opus 配置均有效、errRatio=0。这里使用全部 745 个 shape；仓库正式选型的 693 个 Opus 赢家及 52 个外部赢家是另一种选型口径。

原始 Opus 结果冻结在 [opus_tuned_20261008.csv](opus_tuned_20261008.csv)，来源与编译器收据在 [opus_source.json](opus_source.json)。9000/9001/9010/9011 使用 pin-AGPR Clang24，其余使用 Clang23。warmup=5、iters=51，torch profiler 计时。

这里的 Opus 加速比定义为 `FlyDSL_us / Opus_us`，大于 1 表示 Opus 更快，与原表中 `blockscale_us / mxscale_us` 的比较对象不同。

| 比较口径 | FlyDSL BMM，同 raw scale 契约 | 每 shape 最快 FlyDSL CSV 记录，供参考 |
| --- | ---: | ---: |
| Opus 几何平均加速比 | 1.035941×，速度提高 3.59% | 1.026726×，速度提高 2.67% |
| 中位加速比 | 1.030773× | 1.020662× |
| Opus 更快 / FlyDSL 更快 | 428 / 317 | 411 / 334 |
| FlyDSL 总耗时 | 166.481767 ms | 163.241447 ms |
| Opus 总耗时 | 146.680140 ms | 146.680140 ms |
| 每个 shape 各运行一次的总耗时加速比 | 1.134999× | 1.112908× |
| 总耗时下降 | 11.89% | 10.15% |
| 仅落后 shape 的几何平均耗时增加 | 12.17% | 11.92% |

相对原 blockscale，Opus 的几何平均加速比为 **1.243054×**；699 个 shape 更快、46 个更慢，总耗时从 202.455477 ms 降到 146.680140 ms，下降 **27.55%**。

按 M 分组（加速比仍为 FlyDSL/Opus）：

| M 范围 | Shape 数 | 对 BMM 的 GM 加速比 | 对 BMM：Opus快 / FlyDSL快 | 对最快 CSV 的 GM 加速比 | 对最快 CSV：Opus快 / FlyDSL快 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M<=128 | 128 | 0.895871× | 18 / 110 | 0.895871× | 18 / 110 |
| 128<M<=1024 | 323 | 0.999263× | 159 / 164 | 0.996365× | 155 / 168 |
| M>1024 | 294 | 1.148158× | 251 / 43 | 1.126042× | 238 / 56 |

小 M 是主要差距来源：M≤128 的 128 项中，Opus 只有 18 项快于 BMM；128<M≤1024 整体接近；M>1024 的 Opus 优势更明显。全部分组保存于 [opus_vs_flydsl_by_M.csv](opus_vs_flydsl_by_M.csv)。

对比 CSV 已按用户要求精简为六列：`M,N,K,blockscale_us,mxscale_bmm_us,opus_us`。三个耗时均为 µs；FlyDSL 使用 BMM 路径，Opus 使用 10 月 8 日最后完成的纯 Opus 调优结果。其余统计保存在 `summary.json`，原始结果保留在对应输入快照中。

此处比较来自不同时间、工具链和测试环境保存的 CSV；原全量 tune 未得到有效 FlyDSL 全量数据，同卡 Opus/FlyDSL 复测尚未完成。3.59%/2.67% 的整体差距只作为历史数据参考，不代表当前源码的新实测结果或模型端到端加速。


## 10 月 9 日新增候选对 334 个慢项的覆盖核对

逐 shape 对照 [10 月 9 日原始覆盖清单](../opus_flydsl_all_20261009/coverage334.csv) 和当前参数 catalog，原始 334 个慢项与本报告精确一致，Opus/FlyDSL 历史耗时逐项相同。第一批 62 个优化实现覆盖 334/334 个 shape，共 1,452 个 shape/candidate 组合。其后加入两个较早 short-K 原型并注册为 92 个默认配置，runtime split-K 将三个固定 split 别名移为兼容入口，目前为 89 个默认配置、105 个兼容 ID。当前 61 个 92000+ 新默认配置满足 shape 条件的并集仍覆盖全部 334 项。

| M 范围 | 历史慢项 | 有新默认配置覆盖 | 第一批 main / hybrid / small / large 的 shape 数 | register 融合可覆盖 | LDS 融合可覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M<=128 | 110 | 110 | 6 / 38 / 66 / 0 | 110 | 110 |
| 128<M<=512 | 92 | 92 | 21 / 56 / 15 / 0 | 92 | 92 |
| 512<M<=1024 | 76 | 76 | 56 / 7 / 13 / 0 | 0 | 76 |
| 1024<M<=2048 | 29 | 29 | 10 / 6 / 13 / 0 | 0 | 29 |
| M>2048 | 27 | 27 | 17 / 0 / 0 / 10 | 0 | 0 |

优化方向：register 92310/11/20/21/30/40 支持小 tile、local WaveK 与 runtime global split；LDS 92200–92223 保留 A 协作并将 B 放入提前 1/2/3 步的寄存器队列；fine LDS 92410/20/30 增加 tile 与 runtime split；tiled 92000–92012/92020/21 改变 BM64/96/128/160、固定 K/短 K scale panel 和调度；pin 92100–92120 为固定 K 参数；large-output 92500/01 针对 10 个 9030 大输出慢项，包含 B 直读、A 三 stage 和 C 分块输出。

后续参数化调优已准备 204 组去重 compile config（包含 115 个无旧 ID 的新参数 tuple），还有 22 组独立融合 split-K 配置：register 10、LDS 6、register local WaveK 6。register/WaveK 融合限制 M≤512，LDS 融合限制 M≤2048。参数合法及 shape 可覆盖仅表示可参与验证/调优，不能推断更快。

本次核对实际有效 receipt 后得到的最新进度，比 18:43 的旧交接文字更新：普通数值 smoke 9/9；融合数值 smoke 10/10，均无 invalid；当前 parent 与 FlyDSL BMM 的基线计时 14/745。新候选 tune、完整 split tune、fusion tune 和 745 replay 的有效结果均为 0。最新状态文件保存于 10 月 9 日 20:37 UTC，状态为等待空闲 GPU；此处没有新启动测试或后台进程，也不据旧 PID 声称 watcher 现在仍运行。

逐 shape 当前候选、参数和融合合法域见 [opus_losers334_candidate_coverage.csv](opus_losers334_candidate_coverage.csv)，按 M 汇总见 [opus_losers334_candidate_coverage_by_M.csv](opus_losers334_candidate_coverage_by_M.csv)，有效收据及来源汇总见 [opus_candidate_coverage_summary.json](opus_candidate_coverage_summary.json)。已有 `comparison_gfx950_745.csv` 的 `opus_us` 仍是 10 月 8 日结果，不含新增候选尚未测到的收益。
