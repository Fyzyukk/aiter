# 本轮 421 个 NT shape 的合并与报告

`finish_workflow.py` 只使用 CPU。它读取本轮完整候选 profile、shape 进度和六轮交替复测，以保存的 `handoff_data/csv_baseline.csv` 为固定基线，输出本轮最佳 OPUS 与剩余落后集合。

先 source 同目录的 `env.sh`，下面示例中的完整首轮目录假设为 `full421`；实际运行时替换为对应目录。下面的累计诊断可以纳入旧批次，最终完成报告使用后面的当前批次命令。

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "$OPUS_RUN_DIR/step1_profile.csv" "$OPUS_RUN_DIR/step2_profile.csv" \
    "$OPUS_RUN_DIR/step3_existing_round1/profile.csv" "$OPUS_RUN_DIR/full421/profile.csv" \
  --progress "$OPUS_RUN_DIR/step1_progress.jsonl" "$OPUS_RUN_DIR/step2_progress.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_round1/progress.jsonl" \
  --full-progress "$OPUS_RUN_DIR/full421/progress.jsonl" \
  --fallback-progress "$OPUS_RUN_DIR/full421_fallback/progress.jsonl" \
  --fallback-expected "$OPUS_RUN_DIR/top8_audit/winner_fallback_shapes.csv" \
  --repeat "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step2_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_round1/repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/full421/repeat_shard*.jsonl" \
  --output-dir "$OPUS_RUN_DIR/full421_report"
```

传入实际已经产生的文件；运行中的缺失文件会列在 `missing_inputs`，尚未写完的 JSONL 最后一行会暂时忽略。其他损坏行、重复的 shape/name/round、同名候选的当前 ID 冲突以及基线不一致会报错。globs 用引号保留，让脚本展开。

`--profile` 会读取所有正确 OPUS 候选，保证最快候选被复测推翻时，合法次优仍可参与排序。每个完整六轮批次独立，不能把不同步骤的 round 0–5 拼成一轮结果；同目录中去掉 `_shard0` / `_gpu0` 后的文件名标识该批次。命名新的复测批次时使用独立目录或独立批次前缀。

同一完整六字段 shape key 与完整当前 kernelName 的最新完整复测，覆盖其所有单轮观测。六轮均正确且使用自动输入轮换时，中位耗时参与 CSV 比较；任何一轮错误都会使该候选失效。未完成的批次不覆盖已有结果。`rotate_args != 0` 的热缓存复测不会进入 CSV 胜负或完成 shape 数。历史跨机器结果仅供其他审计工具选择候选，本脚本不会将它们计入本轮胜出。

最终生产统计加 `--current-registry`，以当前可用 kid、完整名字和生产支持 shape 过滤所有候选。隔离实验中未集成、名字不匹配或被生产 scope 排除的 CO 不会成为生产赢家；排除原因和来源写入 `summary.json` 的 `excluded_experiment_candidates` 及同名 CSV。正常错误 profile 的 `kernelName=None` 仍计入正确性失败观测，不会误列为实验候选。此选项默认关闭，原有第 1/2 步的统计口径保持可复现。

正式六轮结果加 `--require-repeat-winners`。正式最佳只从完整正确、自动输入轮换、同文件且同逻辑 GPU 0–3 的六轮中位数选出。所有 profile/progress 仍参与候选发现；完整复测覆盖同候选的单轮观测后，最快剩余候选记录在 `discovery_best`，未确认单轮不会进入正式胜 CSV 数。`discovery_unconfirmed.csv` 列出单轮快于已确认最佳或尚无已确认最佳的 shape，供后续确认。

输出：

- `summary.json` / `summary_zh.md`：完成、胜 CSV、剩余落后、待测数，N/K 与 M 类别统计、正确性失败观测数。
- `report.json`：每个 shape 的结果、最佳来源、复测批次状态与本轮定义。
- `shape_status.csv`：完整预期 shape 清单，包含待测项。
- `best_opus.csv`：存在正确最佳 OPUS 的 shape，包含当前 ID、名字、耗时、错误比例与证据标签。
- `remaining_losses.csv` / `pending_shapes.csv`：本轮实际落后与待测项。
- `confirmation_pending.csv`：最佳结果仍为单轮或仍有未完成确认批次的 shape。
- `discovery_unconfirmed.csv`：快于已确认最佳的未确认单轮候选所在 shape。
- 可选 `--candidate-details` 输出全部候选的 `candidate_status.csv`，完整 421 阶段该文件可能较大。

完整 Top8 完成数只来自 `--full-progress`，与前两步和定向测试的累计覆盖分开。指定 `--fallback-expected` 时，完整验证状态同时检查补测 shape 的覆盖。`--workflow-state "$OPUS_RUN_DIR/workflow_state.json"` 将报告进度写回流程状态；加 `--finalize` 会要求完整 Top8 和指定补测覆盖完、每个 shape 均有正确 OPUS 结果且所有传入文件存在，再将第 4 步标记 complete。同时使用 `--require-repeat-winners --finalize` 时，还要求全部预期 shape 均以完整正确复测中位数作为正式最佳，单轮最佳、待确认最佳和无正确 OPUS 均为零。

CPU 验证已复现前两步的 76 个唯一 shape、60 胜、16 落后。纳入两个完整候选 profile 后，不要求复测的诊断模式有 24 个 shape 的合法次优单轮耗时低于已复测候选的中位耗时；这些形状保留 `single_pass` 和待确认标签。要求复测的正式模式则为 76/76 个六轮中位最佳、60 胜/16 落后，零单轮最佳、零待确认、零无正确候选；上述 24 个较快单轮只保留在 discovery。额外临时 fixture 验证了较新的完整中位覆盖旧低值、失败完整复测使候选失效、批次不混合、热缓存复测不计完成、完整首轮覆盖独立统计及缺少部分当前复测时拒绝 finalize。

完整 421 的最终报告只传最终批次的复测文件。旧步骤的六轮用于准备 shortlist，不能填补最终批次的缺测。下面目录按本轮计划命名；最终 repeat 输出若另用目录，应替换对应路径。

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --full-progress "$OPUS_RUN_DIR/step4_full421_top8/progress.jsonl" \
  --fallback-progress "$OPUS_RUN_DIR/step4_winner_fallback_top0/progress.jsonl" \
  --fallback-expected "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv" \
  --repeat "$OPUS_RUN_DIR/final/repeats/gpu*.jsonl" \
  --current-registry --require-repeat-winners --finalize \
  --workflow-state "$OPUS_RUN_DIR/workflow_state.json" \
  --output-dir "$OPUS_RUN_DIR/full421_final_report"
```

## 完整 Top8 与已知赢家补测一次预编译

`prepare_full421_union.py` 默认只用 CPU 调用实际调参脚本的纯 Top8/指定 Top0 几何逻辑，读取动态 registry，写 `full421_union/union.json`、`receipt.json` 和逐 shape 的 `per_shape_receipt.json`。它同时纳入默认 mxscale 配置中按完整名字匹配的合法 seeds；完整调参脚本若会读取其他 tune-file seeds，使用 `--seed-csv` 明确传入那些 CSV。

```bash
# 第 3 步期间仅生成 CPU 集合，不运行编译。
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_full421_union.py"

# 第 3 步结束、选中的新 CO 已 build/集成后，重新生成动态集合并一次编译。
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_full421_union.py" --compile

# 若核心源码、manifest、输入和可用 registry 均未改变，也可复用已生成集合。
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_full421_union.py" \
  --from-json "$OPUS_RUN_DIR/full421_union/union.json" --compile
```

`--compile` 使用实际 `ensure_opus_kids_compiled(union, cu_num=256)`，只在该显式选项存在时导入编译所需模块，继续限制可见 GPU 为 0–3。`--from-json` 会重新核对源码、输入、所有可用配置和 union 配置的 SHA256；变化后应重新生成集合。新增 sparse CO 使用动态注册表读取，不假定总数或连续 ID。新 CO 镜像必须先 build，才能进入 available registry。

提交 `39cdf47d8` 的 CPU 结果为：421 个 Top8 共 838597 个 shape/candidate，去重 26300 个 kid（684 CO、25616 JIT）；37 个赢家几何补测共 4424 个 shape/candidate，去重 788 个 kid，全部已经包含在 Top8 的全局去重集合中。因此这一版补测并不需要额外 JIT kid；仍需实际测那些被各自 shape 的 Top8 裁剪的几何。一次预编译后进行完整首轮，再运行指定几何的 Top0 补测。

## 最终六轮 shortlist 准备

`prepare_final_repeat.py` 仅做 CPU 准备，并调用现有 runner 的 `--dry-run` 检查当前 ID、完整名字、traits、geometry 和完整 shape 覆盖。输入最终完整首轮及 fallback profiles，默认每个 shape 保留正确候选的全局 Top5、每 pipeline Top2、完整名字可映射的远端赢家；额外保留原有 32/46 及第 3 步的最佳完整六轮候选，以及 current report 中已验证最佳和已集成的稳定新 CO。

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_final_repeat.py" \
  --profile "$OPUS_RUN_DIR/full421/profile.csv" "$OPUS_RUN_DIR/full421_fallback/profile.csv" \
  --known-repeat "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step2_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_round1/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round1/repeats/gpu*.jsonl" \
  --current-report "$OPUS_RUN_DIR/current_report/report.json" \
    "$OPUS_RUN_DIR/co_integration/receipt.json" \
  --output-dir "$OPUS_RUN_DIR/final_repeat_plan"
```

传入实际存在的路径，并选择全新的 output-dir。输出 `repeat_plan.json` 为 version2，可交给 `run_repeat_shards.py` 在 GPU 0–3 进行六轮 101iters 交替确认。`receipt.json` 记录覆盖、候选数、选择角色、输入 SHA256 和实际 CPU dry-run 通过状态。完整 profiles 必须对每个预期 shape 至少提供一个正确 OPUS；未完成 421 首轮不能误生成完整验证计划。

`--known-repeat` 按独立批次验证 round0–5 均正确、自动输入轮换、同文件/同 GPU、固定 CSV 基线，然后选每个 shape 的最佳中位候选。未集成、名字不符或生产 shape 不合法的实验 ID 自动跳过。当前 report 的 `single_pass` 结果不冒充已验证最佳；它们只按最终 profiles 的 Top5/Top2 选择。不会隐式读入所有历史 profile，也不会扩大到全部 12 个 4wave parent。

## Focused 子集筛选与复测

`prepare_final_repeat.py --shapes` 支持完整 421 清单的子集，宽范围的 known-repeat 和 current-report 自动只取该子集。family integration receipt 的 `selected_stable_improvement/current_median_us` 字段也可作为已验证新 CO 来源。

`--known-csv` 读取 focused helper 的 `M,N,K,kid,kernelName,kind,role,geometry,traits` 清单，按当前完整身份和 shape 合法性核对，强制保留其中在最终 profile 正确的候选。缺测或 profile 失败的候选明确列在 receipt 的 `known_csv_omissions` 和 `known_csv_omissions.csv`，不会强制标为正确。`--known-kids` 另外保留指定生产配置在每个合法且 profile 正确的 shape 中的观测。

v2 focused 输入使用 `focused_expansion_known_complete/`；它保留 225 个 known/new shape-candidate（含 kid48096），56 个 shape、109 个几何、2716 个 shape/candidate。旧 `focused_expansion/` 输入仍用于本轮编译 receipt 的 hash，不应覆盖。

```bash
# 完成 focused 21iters 初筛后，仅用 CPU 准备完整六轮计划。
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_final_repeat.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
  --shapes "$OPUS_RUN_DIR/focused_expansion_known_complete/shapes.csv" \
  --known-csv "$OPUS_RUN_DIR/focused_expansion_known_complete/known_winners.csv" \
  --known-kids 100003 100024 100025 100030 100031 \
  --known-repeat "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step2_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round1/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round2/repeats/gpu*.jsonl" \
  --current-report "$OPUS_RUN_DIR/co_rounds_family_integration/receipt.json" \
  --output-dir "$OPUS_RUN_DIR/focused_repeat_plan"

# 由 root 调度 GPU 0–3，按现有默认 101iters 做六轮交替确认。
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/run_repeat_shards.py" \
  --plan "$OPUS_RUN_DIR/focused_repeat_plan/repeat_plan.json" \
  --out-dir "$OPUS_RUN_DIR/focused_expansion_known_complete/repeats" --rounds 6

# 仅统计 focused56，本轮正式胜负只使用当前 focused 六轮批次。
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --expected "$OPUS_RUN_DIR/focused_expansion_known_complete/shapes.csv" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
  --progress "$OPUS_RUN_DIR/focused_expansion_known_complete/progress.jsonl" \
  --repeat "$OPUS_RUN_DIR/focused_expansion_known_complete/repeats/gpu*.jsonl" \
  --current-registry --filter-to-expected --require-repeat-winners \
  --output-dir "$OPUS_RUN_DIR/focused_report"
```

focused 不使用 `--full-progress` 或 `--finalize`，避免把子集当作完整首轮。`completed_shapes` 表示已有本轮证据覆盖，完整六轮的新完成率应另外按 focused plan 的全部候选核对本次批次的 round0–5；旧批次不能补新批次的轮数。
