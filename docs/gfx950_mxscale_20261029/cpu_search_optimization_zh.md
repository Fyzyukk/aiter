# CPU 阶段：候选覆盖与 m_repeat JIT 搜索优化

这批修改根据原 tune 中的两项证据推进：Top4 漏掉合法几何；m_repeat 拿到全体 OPUS 最优的 32 个 shape 全部使用 JIT。当前没有新的 GPU 测量，以下数字表示候选覆盖和编译结果。

## 筛选机制

CPU 候选选择移到 `csrc/opus_gemm/opus_gemm_search.py`，GPU tuner 使用同一函数。该模块只依赖 Python 标准库和 registry，可以在没有 GPU、没有 PyTorch 的环境里导出待测候选。

- `--opus_top_geometries` 默认 8；0 保留全部合法、去重后的几何；4 可使用原 Top4 范围。
- `--opus_pipelines pingpong_lds` 集中搜索 m_repeat，也可选择其他 pipeline。
- `--opus_routes jit` 集中搜索非 pin JIT 配置；CO 路由也可单独选择。
- 历史 seed 以 kernel 名称匹配当前 ID，再保留通过 shape/pipeline/route 筛选的候选，避免因编号变化把 seed 指到其他 kernel。

421 个 shape 都检查了 Top4 选择与原筛选函数一致、Top8 包含 Top4、全量包含 Top8、全量等于合法去重集合，以及合法 seed 可绕过 TopN 截断。

| 当前候选集合上的筛选 | 候选/shape 组合数 | 几何/shape 组合数 | 22 个漏测旧几何已保留数 |
| --- | ---: | ---: | ---: |
| Top4 | 116,239 | 7,018 | 0 |
| Top8 | 236,702 | 11,886 | 4 |
| 全量 | 727,231 | 27,426 | 22 |

这张表使用恢复 CO 并追加新 JIT 变体后的同一 registry。它与原 tune 的 102,961 次测量不是同一个候选快照，不能据此计算性能收益。默认 Top8 扩大搜索范围；优先 22 个 shape 建议用全量模式，其他 shape 可分阶段测量。

逐 shape 数据见 [cpu_candidate_coverage.csv](cpu_candidate_coverage.csv)。

## m_repeat JIT 变体

在原有 JIT 表后追加 102 个配置，IDs 9454–9555，原有 9,412 个可用候选的 kid/name 配对全部保留。m_repeat JIT 候选由 240 增至 342，CO 镜像保持 34 个。

新增配置集中于原 tune 表现较好的两个几何：`128×128 / wave 2×2`、`256×256 / wave 4×2`。复用现有 traits 和 kernel，所有 A/B/C 显式 pin 都关闭。

- GroupM=4/8：让相邻 M tiles 复用 B 的分组调度进入 JIT 搜索。
- GroupN=16/32：比较宽 N 下的分组调度；仍保留原 GroupN=8。
- panel=256：只为通过 traits 合法性检查的 `128×128 / wave 2×2` 几何提供更大的 scale cache，最大支持 K=32768。三个 K=21504 的 shape 现在各有 30 个合法 m_repeat JIT 候选。

GroupM/GroupN 变体在调度条件未触发的 shape 上可能执行相同的映射。GPU 测量后应进一步合并等价配置，再缩小最终搜索范围。

## 编译验证

使用普通 ROCm 7.2.4 clang 在 CPU 上完成三个代表配置的 gfx950 设备编译，读取实际 ELF metadata。未运行 GPU kernel，也没有将这些临时编译镜像注册为 CO。

| kid | 代表配置 | VGPR/SGPR spill | private bytes | LDS bytes |
| --- | --- | --- | ---: | ---: |
| 9503 | 128×128，panel256，GroupM4，非 arrays | 0 / 0 | 0 | 101,376 |
| 9537 | 256×256，panel64，GroupM8，arrays | 0 / 0 | 0 | 151,808 |
| 9481 | 128×128，panel128，GroupN32，非 arrays | 0 / 0 | 0 | 84,480 |

全部新配置经过 Python 中的 traits 合法性检查；这里只编译了三个代表项，其余项由后续正常 JIT 构建验证。数值、PAD 边界和性能仍需 gfx950 GPU 检查，默认模型配置尚未改为采用这些新变体。

本机 CPU 工件目录为 `/home/fyz/retune_dsv4_20261028/cpu_search_optimization_20261029/`，包括候选导出、编号快照、覆盖检查和代表配置编译记录。使用命令见 [后续步骤](next_steps_zh.md)。
