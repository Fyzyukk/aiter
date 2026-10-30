# 14 项回退的 CPU 准备包

> 本目录是原运行环境的归档副本。JSON 指纹与脚本仍引用原运行目录及冻结 CO/JIT；本次未执行归档脚本。Markdown 链接已改为仓库相对路径，原 CPU 检查报告认证的是原文件，归档副本的校验值见 [upload_manifest.json](../upload_manifest.json)。

本目录准备了 14 个历史 CSV 赢家回退项的分析、精确配对复核脚本及结果分析器。CPU 准备阶段没有启动 GPU、导入 Torch/Aiter，或修改最终 421 项结果。

原因与完整幅度表见 [analysis_zh.md](analysis_zh.md)。13 项落后固定 CSV 不到 3%（其中 7 项不到 1%），1 项落后 15.116%。历史赢家没有漏进搜索；重点长 K 项的旧 CO 指令快照和 launcher 与当前一致。测量口径、真实加载身份及运行条件仍需后续窄复核确认。

## 已准备的范围

| 范围 | Shape | `(shape,kid)` 配对行 | 三批结果行 |
|---|---:|---:|---:|
| priority：两个重点项 | 2 | 4 | 12 |
| remaining：其余项 | 12 | 22 | 66 |
| all：一次覆盖全部 | 14 | 26 | 78 |

两段范围互不重复。全部配对行涉及 10 个不同的 CO ID，历史和当前赢家同 ID 的两个 shape 已合并。固定 gfx950/256/128x128 CSV 只作参考，不执行 FlyDSL，也不搜索额外候选。

`cpu_preflight.py` 验证形状、角色、完整名字与 ID、CO ready 状态、编译清单/receipt、对齐/固定 K/地址限制，以及冻结的 SO、CO、launcher、计时源码和 CSV 指纹。遇文件或版本变化直接报错，执行脚本也拒绝自动重编。

## 可以现在运行的 CPU 命令

在本目录执行；这些命令只使用 Python 标准库，不访问 GPU：

```bash
python3 run_recheck.py check --scope all
python3 run_recheck.py plan --scope priority
python3 run_recheck.py plan --scope remaining
python3 compare_longk_artifacts.py
```

不指定动作时，`run_recheck.py` 默认输出 priority 的 CPU 检查与计划。`frozen_inputs.json` 已在准备阶段生成；后续 check 不会修改冻结结果。`plan_all.json`、`plan_priority.json`、`plan_remaining.json` 是已准备的范围清单。

## 后续明确要求使用 GPU 时的手动命令

以下两条是执行入口，CPU 准备阶段没有运行。物理 GPU 只允许 4/5/6/7 中明确选定的一张；执行前检查空闲及实际 gfx950/256 CU/PCI 身份。示例选择 GPU 6：

```bash
/opt/venv/bin/python3 run_recheck.py execute --execute-gpu --gpu 6 --scope priority --output-dir recheck_priority
/opt/venv/bin/python3 run_recheck.py execute --execute-gpu --gpu 6 --scope remaining --output-dir recheck_remaining
```

也可用 `--scope all --output-dir recheck_all` 一次执行全部。两种选择不能再把相同结果拼在一起，否则分析器会拒绝重复记录。

每个 shape 使用 seed=0 数据和一次参考 GEMM。两个精确候选统一 101 次计时、5 次预热、3 批，顺序 AB/BA/AB，保留原 NaN 输出初始化和完整正确性检查。共享轮转份数按原公式探测后固定，写入 manifest；可显式指定 `--rotate-args 1..101`，但改变该值会改变测量口径。

每条结果立即写入 `results.jsonl` 并 flush/fsync；目录另存 `selected_pairs.csv`、`manifest.json` 和设备计时事件。脚本只加载已冻结 SO/CO，编译 fallback 被阻断，运行前后检查指纹。原来的全量结果不会被覆盖。

驱动 KFD PID 和容器 PID 分别记录。初始空闲、创建自身上下文后只有一个 KFD PID 时，可以识别自身驱动 PID并检查后续外部进程；若驱动列表未暴露自身 PID，后续仅记录占用快照，报告明确标记身份未解析，不能宣称已保证运行期间独占。该分支没有在本次 CPU 准备中访问 GPU。

中断后，可对同一目录和完全相同的参数增加 `--resume`。有效结果会跳过，配置/输入/角色/顺序/轮转不符、重复结果及已记录错误会报错，避免把错误当完成。恢复不允许更换 scope、GPU 或计时参数。

## 只在 CPU 上合并、分析后续结果

两段测完后：

```bash
python3 analyze_recheck_results.py --results recheck_priority/results.jsonl recheck_remaining/results.jsonl --pairs ../regressions14_pair_candidates.csv --output-dir combined_analysis --require-complete
```

一次 all 范围测完后：

```bash
python3 analyze_recheck_results.py --results recheck_all/results.jsonl --pairs recheck_all/selected_pairs.csv --output-dir all_analysis --require-complete
```

也可只分析 priority 的结果，传入该目录的 `selected_pairs.csv` 即可。

结果分为三批都领先 `consistent_win`、三批都落后 `consistent_loss`、胜负跨线 `mixed`、缺失/错误/正确性失败 `incomplete`。不完整数据不会算作回退，跨文件重复和额外配置会被拒绝。输出 `comparison.csv`、`summary.json`，包括逐批赢家与每个候选的中位数/波动。

这些标签只说明已记录的三批是否一致，不构成统计置信区间。设备事件留作进一步检查前后迭代、异常值与采样趋势；如果事件数量或名称不能可靠对应 101 次调用，脚本保留原事件并标明限制，不伪造逐迭代样本。

CPU 检查报告见 `cpu_analyzer_checks.json`、`cpu_runner_checks.json` 和 `cpu_preparation_checks.json`。所有用于检查分析器和恢复逻辑的模拟数据均标记 synthetic，位于已清理的临时目录；它们不是新的 GPU 性能结果。

完整 CPU 门禁与计划检查可重跑：`python3 -S check_cpu_preparation.py`。它会主动拒绝 Torch/Aiter/Csrc 导入，核对默认、all、priority、remaining CPU 计划和缺失执行参数的拒绝行为。本次 8 项独立检查、23 项分析器合成检查与 71 项执行脚本 CPU/仿真检查均已通过；GPU 执行分支尚未实测。
