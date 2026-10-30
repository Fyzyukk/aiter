# 2026-10-29 本机 CPU 审计

`audit_progress.py` 只依赖 Python 标准库，不导入 torch/aiter 或访问 GPU。统计口径是正确性满足调参阈值（默认 `errRatio <= 0.05`），且 `best_opus.us < handoff_data/csv_baseline.csv.us`。`best_libtype` 仅保留作参考。

单轮统计，输入 shape 清单可用于显示待完成数量：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/audit_progress.py progress \
  /path/to/run.progress.jsonl \
  --expected docs/gfx950_mxscale_20261029/handoff_data/m_repeat_32_shapes.csv \
  --output /path/to/summary.json --losses-csv /path/to/losses.csv
```

多轮按正确性通过的最低 OPUS 耗时合并，注意这个结果是候选发现统计，前几名仍须交替复测：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/audit_progress.py progress \
  /path/to/round1.jsonl /path/to/round2.jsonl --selection best \
  --output /path/to/merged.json
```

`latest` 为默认模式，重复 shape 使用参数所列文件顺序中的最后一条记录；`best` 只从满足正确性条件的记录中取最低 OPUS 耗时。正常运行中的不完整 JSON 行会被记录并忽略。基线缺失和没有正确 OPUS 候选的 shape 会单独报告。

完整 Top8 审计直接抽取当前调参脚本的纯函数，再使用当前可用 registry 和合法 shape 过滤。未添加 seed，以便识别纯 Top8 的裁剪情况：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/audit_progress.py top8
```

本次源提交 `39cdf47d8` 的结果：48212 个候选可用，其中 780 个 CO。完整 421 个 shape 合计 838597 个 Top8 OPUS 候选。远端记录的 421 个最佳几何全部合法并进入 Top8，包括 192 个曾胜 CSV 的 shape；前两步 32/46 个指定几何也全部保留。远端 32 个 m-repeat 旧完整名字包含旧接口参数，不自动等同当前完整配置，几何覆盖独立核对。

本地 full185 的 185 个历史最佳几何中 120 个进入 Top8，72 个历史 CSV 赢家中 39 个进入；priority22 的 22 个历史最佳几何中 4 个进入，6 个历史 CSV 赢家均未进入。去重后，37 个曾胜 CSV 的 shape 的 37 个合法历史几何需要定向 Top0 补测。

加入本轮证据的审计：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/audit_progress.py top8 \
  --current-evidence docs/gfx950_mxscale_20261029/local_20261029/step3_inputs/evidence.json \
  --current-repeat docs/gfx950_mxscale_20261029/local_20261029/step1_repeat_shard*.jsonl \
                   docs/gfx950_mxscale_20261029/local_20261029/step2_repeat_shard*.jsonl \
  --output-dir docs/gfx950_mxscale_20261029/local_20261029/top8_current_audit
```

后续19个shape首轮/六轮复测完成后，追加 `--current-progress /path/to/progress.jsonl` 或 `--current-repeat /path/to/repeat*.jsonl`，以实际最终配置检查裁剪。`--current-repeat` 只从自动轮转且全部六轮正确的候选中，选择每shape最快中位耗时。`current_sources` 单独列出本轮候选和CSV赢家覆盖，不将历史赢家统计当作本轮胜出。`current_winner_fallback_shapes.csv` / `current_winner_fallback_geometries.csv` 只列本轮赢家被裁剪项；完整历史赢家裁剪另见 `winner_fallback_*`。

本次前两步76个shape的最终六轮median配置全部合法、完整名字全部进入Top8，包括60个本轮CSV赢家；本轮赢家需补测几何数为0。远端32个m-repeat旧16参数的PinA/B/C全部为false，去掉三个槽后32条均匹配当前合法13参数traits。`normalized_m_repeat_*` 字段额外审计这种参数对应关系，未改动调参器自动seed-name解析；旧完整名字仍不会凭旧数值ID恢复。

`top8_audit/winner_fallback_shapes.csv` 与 `winner_fallback_geometries.csv` 是这 37 个历史赢家的补测输入。`fallback_shapes.csv` 与 `fallback_geometries.csv` 包含全部 76 个被裁剪的历史最佳几何（含历史落后项）。不要将这些局部几何文件传给完整 421 个 shape 首轮；完整首轮后用局部输入和 `--opus_top_geometries 0` 补测并合并。

`historical185_summary.json` 验证统计脚本可复现交接文档的 185 个完成、72 个胜 CSV、113 个落后。

## 第 3 步更新落后集合

完成前两步及六轮交替复测后，重新运行 CPU 脚本，生成当时实际证据对应的输入：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/update_losses.py
```

默认读取本目录的 `step1_progress.jsonl`、`step2_progress.jsonl`，及 `step1_repeat*.jsonl` / `step2_repeat*.jsonl`。也可明确指定文件以隔离不同轮次：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/update_losses.py \
  --progress /path/to/step1_progress.jsonl /path/to/step2_progress.jsonl \
  --repeat /path/to/step1_repeat_gpu0.jsonl /path/to/step1_repeat_gpu1.jsonl \
  --output-dir /path/to/current_step3_inputs
```

前两步统计只覆盖 76 个唯一 shape。每个 progress 来源保留最后一条 shape 记录；不同输入文件搜索的不同几何候选都作为本轮候选发现保留。完整六轮复测须每轮正确，使用中位耗时替换该完整候选名的单轮结果，避免保留已被复测推翻的较低耗时。尚未复测的候选保留 `single_pass` 标签；不足六轮的复测标 `confirmation_pending`，不覆盖其单轮结果。`rotate_args != 0` 的热缓存复测排除 CSV 比较。`evidence.json` 保存每条 progress 和复测的路径、行号与时间。

历史 full185、priority22 和 handoff both_losses67 只用于建立有来源的待测落后集合。只要本轮胜出，该 shape 会从后续落后输入移除；没有本轮观测的历史 loss 明确标记 `historical_loss_pending_current_measurement`。多个历史来源胜负不一致时额外标记 `historical_mixed_evidence`，不把它解释为本轮输赢，也不将任何历史赢家纳入本轮 wins。

输出目录 `step3_inputs/`：

- `summary.json` 与 `current76_status.csv`：76 个唯一 shape 的完成/胜CSV/观测落后/待完成，以及来源和复测状态。
- `observed_losses_shapes.csv` / `observed_losses_status.csv`：本轮实际观测落后，正确性无有效候选也算未胜。
- `historical_pending_losses_shapes.csv` / `historical_pending_losses_status.csv`：尚未本轮复测的历史 loss。
- `followup_shapes.csv` / `followup_geometries.csv`：上述两部分并集及当前合法的远端/本地历史最佳几何。
- `representative_shapes.csv` / `representative_geometries.csv`：定向优化代表输入。按 N/K 与 M 类别分组，默认每组至多两项；有本轮 loss 时选最严重相对慢项和最大 M，尚无本轮观测时选中位 M 和最大 M。
- `representative_status.csv` / `geometry_selection_details.csv`：选择理由、证据来源、每个几何的合法参数候选数。
- `unavailable_geometries.csv`：历史几何在当前 registry 不合法时单独记录，不传给调参器。

窄 N768/1536/2048 及 N6144 的代表几何对 B-VGPR 与 overlap 分别覆盖 BN64/128、2/4 wave；同一类中优先保留已知最佳几何，再按当前调参器的几何评分选择。N>=6144 另外保留 pin_agpr 与两个 m-repeat 几何。指定几何配合 Top0 会继续测这些几何下全部合法参数，包括 stage/publish/cache；没有修改筛选规则。真实优化仍需分别控制实际 `B_AHEAD`、pin、XCD 等参数，不能将几何筛选当成变量隔离。

已用 CPU 合成数据核对：六轮中位慢于 CSV 会推翻单轮胜出；六轮任何一轮错误会拒绝该候选；五轮不完整复测不会覆盖单轮结果，并保留 pending 标记；所有逐轮来源均保留。

`top8_audit/dedup_evidence.json` 解释 838597 的计数：当前 48212 个可用 ID 各有唯一完整名字，无别名重复；Top8 在选择 ID 时用集合消重，调参器每个 shape/ID 生成一次任务。同几何的不同参数都保留，跨 shape 重用一个 ID 属于独立 shape 测量。没有根据相近机器码、BD 裁剪或 runtime/fixed K 等未经证明的等价关系删候选。

## 第 3 步既有配置首轮 19 个 shape

`plan_stage3_existing.py` 读取上述 `evidence.json` 与带来源的历史 pending 表。默认仅stdout预览；`--write` 只有前两步76个shape完成、第一步32个shape六轮复测完成且无部分复测pending时才写最终输入。

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/plan_stage3_existing.py --write
```

本次实际完成后的输入已生成在 `step3_existing_round1/`：`shapes.csv`、`geometries.csv`、`selection.json`、`geometry_candidate_counts.csv` 和 `run.sh`。共19个shape、63965个既有OPUS候选、828行几何，`run.sh` 设置 `libtype opus`、Top0、mp4，继续使用GPU0–3；不重复FlyDSL，比较固定CSV基线。

窄N768/1536/2048，各选一个小M与一个大M，在已观测loss优先的基础上靠近6144/8192；本次选择2048/8192、4096/20480、2048/12288。四个4wave重点(N,K)=(6144,7168)、(7168,7168)、(7168,16384)、(7168,4096)，各加入M6144/8192/32768共12个方向对照，再加入1664×65536×1536的宽N未胜项。

每个shape保留B/overlap BM64/96/128、BN64/128、WN2/4，所有合法BK和完整参数，覆盖A stage2/3/4；保留m-repeat BM128/192/256及全部合法4wave CO。前18项均有12个合法4wave CO，宽N的M1664只能保留6个PAD CO。此轮范围比原30代表的16651个候选更完整，候选数约3.84倍。完成后用 `inspect_co_parents.py` 查看最优panel/PAD/WideOutput，再考虑 `prepare_co_experiment.py` 的有限GroupM或严格B配对实验。
