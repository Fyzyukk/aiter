# 本轮完整 421 的最终命令

先 source `env.sh`。完整首轮目录为 `step4_full421_top8/`，37 个已知赢家指定几何 Top0 补测目录为 `step4_winner_fallback_top0/`。以下命令在这两轮都结束后执行。所有旧六轮仅用于挑选候选，正式统计只用本次最终六轮。

## 1. CPU 确认首轮与 37 个补测覆盖

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
    "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --full-progress "$OPUS_RUN_DIR/step4_full421_top8/progress.jsonl" \
  --fallback-progress "$OPUS_RUN_DIR/step4_winner_fallback_top0/progress.jsonl" \
  --fallback-expected "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv" \
  --current-registry \
  --output-dir "$OPUS_RUN_DIR/step4_discovery_report"

"$OPUS_RUN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
s = json.loads((Path(os.environ['OPUS_RUN_DIR']) / 'step4_discovery_report/summary.json').read_text())
assert s['full_top8_completed_shapes'] == 421, s
assert s['fallback_completed_shapes'] == s['fallback_expected_shapes'] == 37, s
assert s['full_validation_complete'] and not s['missing_inputs'] and not s['no_correct_opus'], s
print('完整 Top8 421/421 与指定几何补测 37/37 覆盖通过；此处仍是单轮发现结果。')
PY
```

## 2. CPU 准备最终完整六轮 shortlist

profile 输入按测量时间顺序为 focused → full421 → fallback，因为同 kid 的较后 profile 会覆盖较前观测。全部三个 profiles 参与选全局 Top5 和每 pipeline Top2；保留完整名字匹配的远端赢家、当前可用的旧六轮最佳、已集成 22 个稳定 CO 的对应已验证 shape，以及 focused v2 的 225 条指定已知候选。

显式 9 个生产 ID 是 5 个已确认方向与其父配置的去重集合：`389, 100001, 100003, 100022, 100023, 100024, 100025, 100030, 100031`。它们只在当前 profile 正确且支持相应 shape 时强制保留。v2 known CSV 同时保留 kid48096。

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_final_repeat.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
    "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --known-csv "$OPUS_RUN_DIR/focused_expansion_known_complete/known_winners.csv" \
  --known-kids 389 100001 100003 100022 100023 100024 100025 100030 100031 \
  --known-repeat "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step2_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round1/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round2/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/focused_expansion_known_complete/repeats/gpu*.jsonl" \
  --current-report "$OPUS_RUN_DIR/co_rounds_family_integration/receipt.json" \
  --output-dir "$OPUS_RUN_DIR/final_repeat_plan"
```

`final_repeat_plan/` 必须是新目录。receipt 中检查 `repeat_shapes=421`、`cpu_dry_run_passed=true`、`known_csv_required_candidates=225`、`known_csv_candidates_selected=225` 和 `known_csv_omission_count=0`。显式 9 个 ID 的当前正确 profile shape 数在 `explicit_known_kid_correct_profile_shapes`；`known_repeat_audit` 明确列出未集成/名字不符/scope 不合法的旧实验，避免把那些 ID 带入生产候选。

## 3. 由主 agent 调度 GPU 0–3 做最终六轮

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/run_repeat_shards.py" \
  --plan "$OPUS_RUN_DIR/final_repeat_plan/repeat_plan.json" \
  --out-dir "$OPUS_RUN_DIR/final/repeats" --rounds 6
```

runner 使用默认 101iters、round 0–5 的交替顺序及自动输入轮换。每个 shape 的所有候选留在同一文件、同一个逻辑 GPU 0–3；`final/repeats/` 必须是新目录。应先核对该批次全部计划候选均写齐六轮与正确性结果，再生成正式报告。

下面的 CPU 审计只读本次 plan/raw，严格核对每个计划候选的六轮、名字/kind/roles、GPU/文件、位置、自动轮换及固定 CSV，另核对 known225 全部在 plan 且本次六轮均正确。候选错误和未完成数独立写在 receipt；错误候选不能成为正式最佳。

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/audit_final_repeat.py" \
  --plan "$OPUS_RUN_DIR/final_repeat_plan/repeat_plan.json" \
  --files "$OPUS_RUN_DIR/final/repeats/gpu*.jsonl" \
  --require-complete \
  --output-dir "$OPUS_RUN_DIR/final_batch_audit"
```

## 4. CPU 生成正式完成报告

```bash
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
    "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --full-progress "$OPUS_RUN_DIR/step4_full421_top8/progress.jsonl" \
  --fallback-progress "$OPUS_RUN_DIR/step4_winner_fallback_top0/progress.jsonl" \
  --fallback-expected "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv" \
  --repeat "$OPUS_RUN_DIR/final/repeats/gpu*.jsonl" \
  --current-registry --require-repeat-winners --finalize \
  --workflow-state "$OPUS_RUN_DIR/workflow_state.json" \
  --output-dir "$OPUS_RUN_DIR/full421_final_report"
```

该命令要求 421 个 shape 的正式最佳全部为本次完整正确六轮的中位数，零单轮最佳、零待确认、零无正确 OPUS，并核对完整 Top8 和 37 个补测覆盖。profiles/progress 只保留候选发现和覆盖证据；较快未确认单轮留在 `discovery_best` 与 `discovery_unconfirmed.csv`，不计正式胜 CSV。仅正确且耗时低于保存的固定 CSV 基线算胜出；报告同时写正式剩余落后和待测集合。
