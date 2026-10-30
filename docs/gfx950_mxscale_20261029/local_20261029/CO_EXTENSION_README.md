# 在第一轮隔离仓库追加第二轮 CO

`extend_co_experiment.py` 复用 `co_round1/.cache/experiment_repo` 的804个CO，
要求其 common 已冻结 JIT base780。新ID从100024开始，seed可来自原780
或第一轮100000–100023。工具只进行CPU工作，不启动build或GPU任务。

实际第二轮计划为
`co_round1/independent_review/next_round_plan.json`。CPU审阅结果位于
`co_round1/independent_review/extension_check/receipt.json`：12个新CO、
6个shape、48个new/seed候选，IDs100024–100035，旧48212+24全部身份保留。
这里48指去重后的candidate×shape记录；实际new-parent shape配对为30。
计划中的 `required_existing_winners` 包含 `(4096,1536,7168)` 的kid48096。
当前生成器尚未自动处理该附加字段，需要在生成helper plan后按完整
registry metadata加入该候选、role=`known_full_geometry_winner`，再dry-run；
加入后候选记录数为49。

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/extend_co_experiment.py \
  --base-experiment-dir docs/gfx950_mxscale_20261029/local_20261029/co_round1 \
  --plan docs/gfx950_mxscale_20261029/local_20261029/co_round1/independent_review/next_round_plan.json \
  --out-dir docs/gfx950_mxscale_20261029/local_20261029/co_round2_review \
  --check-only
```

默认与 `--check-only` 只在新的本地目录写 `receipt.json`、
`proposed_manifest.json`、`base_manifest.json`、`selection_plan.json`、
定向shape/geometry CSV、`helper_repeat_plan.json`和launchers。
clone仍为804CO。未来需要实际追加时，在另一个新的输出目录使用
`--apply`；该模式重新完成相同CPU检查，备份隔离manifest后原子写入，
失败恢复备份。公共分支的common和manifest始终不由此工具修改。

计划继续使用 `experiments` 和 `paired_experiments`，每点必须有
`seed_kid`/pair seeds、一个 `vary` 轴、明确shape和reason，顶层必须有
已完成六轮的 `source_report`。源报告294个候选的原始记录已经按完整
身份、正确性、计时模式和六轮位置核对；新CO明确为GPU pending。

本轮4wave保持runtimeK、Unroll2，只选择单变量GroupM16或Panel32。
现有C++允许GroupM0/4/8/16、Panel16/32/64；GroupM32非法。
`transform`能描述C++已有的Unroll2/4，但本轮计划验证明确拒绝改变
Unroll2。Unroll是traits字段，不能误当LDS stages进行修改。

B/overlap仍仅允许NB、BD或XcdOrder单变量。严格pair保持17个基础
traits和tile/wave/launch/pin/grid/output契约。记录实际B_AHEAD/LDS变化，
BD被裁剪成相同B_AHEAD时默认拒绝。新配置若已有相同完整device/host
契约，工具拒绝再花一个CO只改变名字或supported_shapes。

第一轮旧seed的 `supported_shapes` 不能扩大，否则旧完整dataclass
身份发生变化。例如100020/21仅支持其N768两形，100022/23仅支持
其N1536两形；工具分别检查new（移除scope捷径后的真实支持谓词）
及exact seed对每个指定shape的合法性。无需引入“等价parent per shape”
等新映射框架。

实际追加后，使用输出目录的 `build_new_co.sh` 只build本轮IDs，记录
本轮独立 `new_co_build_info.json`，再CPU合并第一轮与本轮报告为
`merged_co_build_info.json`，保留两份报告的路径和SHA256。原780的报告
不重写；合并发现重复ID会拒绝覆盖历史。`run_tune.sh`使用指定几何、
Top0和mp4；`run_helper_repeat.sh`调用已有四GPU shard runner，强制将
每个new及其exact seed纳入六轮交替复测，不受TopK裁剪。

分析器需要从复用clone里的脚本运行，以读取追加后的registry；
生成的 `experiment_env.sh` 会把 `OPUS_RUN_REPO` 指向现有clone，并为
第二轮建立独立AITER_JIT/FlyDSL/Triton cache路径。六轮汇总使用
`--plan <round-dir>/helper_repeat_plan.json`。本轮活跃
`_experimental_extensions.points`只包含新点，第一轮metadata完整存入
`_experimental_extension_history`；804旧条目和所有旧suffix原样保留。

本工具有意限定为804基线的第二轮追加，不承担公共集成或自动展开
完整421 shape。后续稳定收益筛选需使用实际GPU结果，不能将这些CPU
提案或机制fixture计为CSV赢家。

可复现CPU机制核对：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/validate_co_extension_cpu.py
```

它只使用临时manifest模拟追加，保留
`co_extension_cpu_fixture_report.json`，不会实际扩展仓库。
实际追加后，该验证脚本会使用 `co_round2/base_manifest.json` 的804基线
快照进行CPU模拟，继续保留当前live manifest不变。
