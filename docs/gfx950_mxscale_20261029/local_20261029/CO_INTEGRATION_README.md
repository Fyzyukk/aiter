# 稳定 CO 改进的 CPU 集成检查

`integrate_co_winners.py` 读取隔离实验目录和已经完成的
`summarize_existing_repeat.py` 六轮报告。默认与 `--check-only` 都只在新的
`local_20261029` 子目录写提案和 receipt，不编译、不使用 GPU、不修改公共文件。

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/integrate_co_winners.py \
  --experiment-dir docs/gfx950_mxscale_20261029/local_20261029/co_experiment \
  --report docs/gfx950_mxscale_20261029/local_20261029/co_experiment/repeat_report/report.json \
  --out-dir docs/gfx950_mxscale_20261029/local_20261029/co_integration_review \
  --check-only
```

`--repeat-jsonl` 可指定原始路径或引号 glob；默认使用报告的 `summary.files`。
报告须 `rounds=6`、`pending_shapes=0`。每个实验点的指定 shape 都必须同时
包含 exact new CO 和 seed parent。需要在隔离实验副本中运行已有分析器，
以保证其 metadata 来自实验 registry。

选择规则为两者六轮 `0..5` 完整正确，`errRatio` 在 `[0,0.05]`，有限正
耗时，`rotate_args=0`，同 shape 在同一原始文件、同一个 GPU 0–3；
每轮候选 position 唯一完整，new/parent 在轮对 `(0,1)/(2,3)/(4,5)`
中反转先后顺序。new 中位数必须严格小于 parent，且至少 4 轮比同轮
parent 更快。逐项重算报告的中位数、最小/最大、标准差、正确轮数、错误率
与 CSV 胜负；不接受报告或原始身份字段缺失、重复或不一致。

CSV 胜负继续使用固定 `handoff_data/csv_baseline.csv`，按六个完整字段
匹配 421 shape。该 CSV 含其他架构/模型行，它们不会参与这一 cohort。
稳定改进仍可加入候选，即使尚未胜 CSV；只有 `wins_csv_median=true`
计为 CSV 胜。若只准备 CSV 赢家，可额外使用 `--require-csv-win`。

提案默认只启用实测稳定改善的 shape。可选 `--expand-scope nk-mclass`
扩展到同 N/K、同 M 类别的合法完整421 shape，或 `--expand-scope nk`
扩展到同 N/K 的合法完整421 shape。M 类别为 `<128`、`128–1023`、
`1024–8191`、`>=8192`。任何实测不稳定/退化 shape 都不会被扩展重新加入。
扩展项明确标记 `gpu_validation_pending=true`，不计 GPU 完成数或 CSV 赢家。

输出包括 `receipt.json`、`parent_comparisons.csv`、`new_kernel_scopes.csv`、
`proposed_co_kernels_gfx950.json` 和 `proposed_opus_gemm_common.py`。
receipt 同时列候选/shape pair 数与去重 shape 数，以避免候选重复计数。

未来显式执行 `--apply` 时，工具会再次核对输入、公共/实验源码和镜像
SHA256，备份公共 `opus_gemm_common.py` 与 `co_kernels_gfx950.json`，
仅复制选中的新 `.co`，冻结 JIT base 为780，再原子写入 manifest。
失败会恢复两公共文件并删除本次新复制的镜像。780旧条目、suffix与
原48212个 ID、完整名字、完整 dataclass 配置全部逐项保持一致；新ID
沿用原实验的100000+稀疏ID，未选中的ID不压缩重排。

本版本针对原780 CO基线的一次集成；已经存在 sparse 扩展或镜像路径冲突
会拒绝并要求审阅，避免自动重映射已经保存的调优ID。检查新镜像须有
匹配 `status=ready`、symbol、image、gfx950 ELF target 和无未定义设备
符号的 build report。gfx950 builder 当前不总是记录镜像 hash，工具会
计算并保存实际镜像 SHA256，有 builder hash 时还会比对它。

CPU 验证：

```bash
python3 docs/gfx950_mxscale_20261029/local_20261029/validate_co_integration_cpu.py
```

只保留 `co_integration_cpu_fixture_report.json`。合成 timings 和简化 ELF
header 明确为 fixture，均在临时目录中销毁，不是 GPU 性能证据。
集成、备份、失败回滚只在临时假公共仓库验证；真实公共文件始终不变。
