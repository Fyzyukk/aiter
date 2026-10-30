# 第 3 步 CPU 分析和有限 CO 实验选择

以下辅助程序不导入 torch/aiter，不编译，不运行 GPU。GPU 0–3 的调度由主 agent 控制。

`analyze_stage3_existing.py` 默认读取 `step3_existing_round1/`。构建或测量未完成时，可读取 `candidates/*.json` 的当前 worker 计数及 `progress.jsonl` 的逐 shape 赢家。worker JSON 会覆盖写，不能据它累计所有候选，也不能从它得到每个 kid 的耗时。最终 `profile.csv` 产生后，程序按当前完整名字/traits 核对身份，输出每个 pipeline 的 CO/JIT 前三、全部合法 4wave parent 以及 B/overlap 的实际 `B_AHEAD`。

首轮完成后，准备六轮交替复测：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/prepare_existing_repeat.py
```

如首轮发生 shape watchdog/dummy，`profile.csv` 仍可能有 19 个 shape，而其中没有正确 OPUS 的 shape 不能算正确覆盖。主 agent 统一调度时可先调用 `prepare_stage3_recovery.py --run`：正确覆盖已经齐全时仅在 CPU 上去掉无效候选并生成 `step3_existing_round1/profile_confirmable.csv`；存在缺口时，仅用失败 shape 的原几何运行 Top0、OPUS、mp4、21 iterations、timeout7200，再合并正确记录，恢复优先并按 `(shape,kid)` 去重。必须 19 个预期 shape 均有当前完整名字/ID且合法、正确的 OPUS，才发布 confirmable。仅准备和审查输入时省略 `--run`，不会启动 GPU。之后准备复测需显式传 `--profile .../profile_confirmable.csv`，原测量文件保持不变。

输出为 `step3_existing_round1/cpu_analysis/repeat_plan.json`，另有四个预分片文件。主 agent 可用 `run_repeat_shards.py` 做负载分片和运行。计划包含总榜与每 pipeline 前三、每个 pipeline 的最快 CO/JIT、全部合法 4wave parent、普通 B `PublishM=-1` 与同完整基础 traits 的 overlap `PublishM=0`、最优 m-repeat 的精确 BoundedOutput 对照。严格 CO peer 使用完整 17 项 base traits、tile/wave、launch bounds、device flags、grid、pin、pad、wide、fixedK/maxK；JIT 对照补齐当前 C++ 的全部默认参数后核对。同一 shape 的候选和六轮须保持同一个 GPU。

完成复测后汇总，可指定实际分片输出路径：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/summarize_existing_repeat.py \
  '/path/to/repeat/shard*.jsonl' \
  --plan docs/gfx950_mxscale_20261029/local_20261029/step3_existing_round1/cpu_analysis/repeat_plan.json
```

默认产生 `step3_existing_round1/repeat_report/report.json`、summary、候选中位数表和按 shape/pipeline 排序的明细。每个候选要求六轮完整、六轮正确且 `rotate_args=0`，中位 OPUS 耗时低于固定 CSV 才计胜出。该程序没有远端历史赢家 cohort 的前提，不能直接用 `summarize_opus_repeat.py` 处理含 `remote_winner=null` 的第 3 步计划。

根据稳定中位数写有限 CO plan：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/select_stage3_co.py
python3 docs/gfx950_mxscale_20261029/local_20261029/prepare_co_experiment.py \
  --plan docs/gfx950_mxscale_20261029/local_20261029/selected_co_plan.json --check-only
```

选择器对四个重点 N/K，逐代表 M 选择完成六轮且正确的最快 panel/PAD/WideOutput parent，只追加 GroupM4/8。同 runtime-K parent 在不同 N/K/M 复用时合并其合法 shape，避免把大 M 的实际最优 panel 强制换成较小 M 的最优组合。窄 N768/1536/2048 只针对本轮仍落后项，按当前最大相对落差的 shape 选择已完成复测的严格 B/overlap CO pair，再分别改变 XCD、NB3、有信息量的 BD，以及预算允许的 NB2。最多产生 24 个新 CO；实际 `B_AHEAD` 相同的 BD、非法 NB、预算截断都写出原因。新方案仍待编译和 GPU 实证，这些 CPU 输出不表示新增性能收益。

辅助脚本的 ROOT 由自身所在位置计算。如在隔离 `experiment_repo` 使用这些程序，须把辅助脚本目录复制到该副本对应的 `docs/.../local_20261029`，使 registry_metadata 核对副本中的稀疏新 ID。

实验副本中的分析器自动读取 `manifest._experimental_extensions.points`，无条件把每个合法新 CO 和它的精确 seed parent 加入同 shape 的复测计划，避免 TopK 裁剪导致缺少同轮对照。`--include-experiment-parents` 可显式要求该扩展元数据存在；新 CO 尚未 build 成可用镜像时会报错，不能被静默省略。计划中的 `experimental_points` 保留配对来源，候选 roles 为 `experimental_co`、`exact_experiment_parent`。CPU 声明 registry fixture 已核对未测新 ID 100000/100001 和 parent 451/643 强制纳入，且通过运行器的完整名字、traits、geometry `--dry-run` 校验；该 fixture 没有创建 CO 镜像、没有编译或运行 GPU。

CPU 验证使用前两步的真实数据：首轮分析读取步骤 2 的 1097 个 OPUS 候选，重复计划通过运行器 `--dry-run` 的完整身份核对；复测汇总读取步骤 2 的 882 条记录，复现 46 个完成、42 胜 CSV、4 落后，147 个候选均完整六轮正确。验证输出放在 `stage3_analysis_cpu_validation/`，与第 3 步实际结果目录分开。
