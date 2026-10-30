# 有限 CO 实验机制（尚未选定性能候选）

先用既有配置完成第 3 步首轮方向比较，再按实测最优 parent 填 selection plan。此处工具不会自动挑 parent、不会 build、不会使用 GPU，原仓库核心源码和 manifest 只读。

## 查询实测 parent

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/inspect_co_parents.py \
  --profile /path/to/step3_profile.csv --shape 8192,6144,7168 \
  --output docs/gfx950_mxscale_20261029/local_20261029/parent_review.json
```

结果保留正确性满足阈值的全部已测 4wave CO（panel16/32/64、PAD、WideOutput），并给出 B/overlap 前三个既有 CO parent、完整参数、配对 peer 及实际 `B_AHEAD`。JIT parent 不自动转换成 CO；既有 JIT stage2/3/4 仍在完整 profile 中供首轮比较。

## 明确填写单变量 plan

`co_selection_plan_template.json` 是空模板，直接运行会被拒绝。以下只展示格式；示例 seed/shapes 不表示本轮性能候选，实际应根据结束后的 profile 替换。

```json
{
  "experiments": [
    {
      "seed_kid": 0,
      "vary": {"GroupM": 4},
      "shapes": [[8192, 6144, 7168]],
      "reason": "替换为实测最优panel/PAD/WideOutput parent及当前loss选择依据"
    },
    {
      "seed_kid": 0,
      "vary": {"GroupM": 8},
      "shapes": [[8192, 6144, 7168]],
      "reason": "保持同一最优parent，仅比较GroupM8"
    }
  ],
  "paired_experiments": [
    {
      "b_seed_kid": 451,
      "overlap_seed_kid": 643,
      "vary": {"NB": 3},
      "shapes": [[1280, 7168, 384]],
      "reason": "替换为实测几何/pin/Contiguous固定后的A stage比较"
    }
  ]
}
```

`experiments` 每条产生一个新 CO，`paired_experiments` 每条只产生严格匹配的普通 B 和 overlap 两个新 CO。每个点必须只改变一个明确变量。4wave 只接受 GroupM4/8，保留 runtime K、Unroll、panel、PAD、WideOutput 和已有 tile/LDS 路线。B/overlap 只接受 NB、BD 或 XcdOrder；几何、pin、Contiguous、cache、输出方式和 PublishM 保持 parent 值。

严格 B/overlap 配对须完整17项基础traits、launch、device flags、pin、grid等完全一致，只有 pipeline 的预取发布位置区别。普通 B 的发布含义为 -1，overlap 保存原 PublishM。变更 XcdOrder 同时更新对应 host grid 的 linear/xcd 合约，这是同一变量必需的配套改变。

工具按当前 C++ traits 计算 `B_FRAGMENTS`、`ROTATE_B`、`B_SCHEDULE_AHEAD`、`B_AHEAD`、`B_SETS` 和 LDS，并记录变更前后值。BD改了但实际 `B_AHEAD` 相同，默认拒绝以免做无信息实验；确需记录这种对照，可在该点显式设 `allow_equal_b_ahead: true`。NB变化也可能改变实际 B_AHEAD，报告明确记录，不将它解释为纯 A-stage 收益。

## CPU 核对并创建隔离副本

先只核对选定 plan：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/prepare_co_experiment.py \
  --plan docs/gfx950_mxscale_20261029/local_20261029/selected_co_plan.json --check-only
```

审核选定 plan 后创建副本，仍不 build：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/prepare_co_experiment.py \
  --plan docs/gfx950_mxscale_20261029/local_20261029/selected_co_plan.json \
  --out-dir docs/gfx950_mxscale_20261029/local_20261029/co_round1
```

副本为 `co_round1/.cache/experiment_repo`。只在副本 `opus_gemm_common.py` 将 JIT base 从 `len(CO)` 显式冻结为780，manifest尾部追加100000起的稀疏ID。原0–779 CO条目和suffix原样保留，780–48211 JIT ID不移动。默认最多24个新CO，禁止完整Cartesian展开。源码使用独立字节复制，不硬链接可修改源码。

`identity_and_points.json` 和 `copied_source_identity.json` 核对所有旧48212个ID、完整名字、完整dataclass配置逐项相等，另生成整体SHA256摘要；新完整名字须唯一，包含改变的参数、NB/BD/实际BA/XCD/Contiguous/GroupM/pin等字段及完整配置hash。每个选定shape先去掉实验shape限制，再用原合法性谓词验证；有限shape限制不会绕过形状检查。

副本有独立 `AITER_JIT_DIR`、FlyDSL和Triton缓存，继续只可见GPU0–3。创建时写出但不执行：

- `build_new_co.sh`：调用现有 `build_co.py --arch gfx950 --kids 100000...`，只build新增项，明确指定 `new_co_build_info.json`，不覆盖780个旧CO报告。
- `run_tune.sh`：使用指定shape/geometry、Top0和`libtype opus`，生成完整候选profile及progress，供新旧参数与保存CSV比较。
- `experiment_shapes.csv` / `experiment_geometries.csv`：只包含plan明确选择的shape和新增CO几何；Top0仍测同几何合法旧参数作对照。

只有实际选定候选之后，root再执行build/tune；性能有效后才将少量有效项集成回原仓库。集成同样必须保留JIT base780并核对原48212身份，直接增加CO数量会使旧JIT IDs整体移动。

## 已完成的机制验证

`prepare_co_experiment.py --mechanism-check` 只用CPU fixture临时构造一个GroupM4及一组BD2的严格B/overlap配对，退出时删除临时manifest，不创建实验repo或任何性能计划。`co_mechanism_fixture_report.json` 已验证原48212身份完全不变，扩展后为48215个声明ID，新100000–100002完整名字唯一；B/overlap fixture的实际B_AHEAD均从4变为8。原配置摘要为 `37ea3a12fbdf5c797f9c62af0e9d0caf7f008ff58d39a0606107741c2a88d206`。没有编译或GPU执行。
