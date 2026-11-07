# build-co 筛选分析与恢复结果

gfx950 原有筛选把 VGPR spill、SGPR spill 和非零 private segment 都作为无条件拒绝条件，而且先拒绝、后验证实际 pin。对性能搜索而言，这会过早丢弃编译成功的候选；尤其 SGPR spill 可能保存到 VGPR lane，不等于发生 scratch 内存访问。

gfx1250 默认记录并警告 VGPR/SGPR spill，仅显式 `--fail-on-spill` 时拒绝。gfx950 已采用相同的默认资源策略，并用单独的 `--fail-on-scratch` 控制 private segment。

## pin 验证是什么

pin 指源码显式要求某些片段放入指定 AGPR。gfx950 builder 读取实际 CO metadata 并反汇编，检查指定 A/B 的 `ds_read_b128` 是否到达预期 AGPR 范围；对显式或部分 pin 的 C，检查 MFMA 的 AGPR 范围和后续 C/D 更新是否保持在同一片段。

这验证编译器是否落实寄存器放置要求。它不是 GEMM 数值正确性测试，也不保证耗时更低。恢复时保留了这项检查和已有 LDS 上限检查。

当前 `4wave_pin_agpr` 的 PAD/non-PAD 都 pin A/B，并部分 pin C，都走 CO。是否补 PAD 与是否必须 pin 没有普遍的等价关系；本分支按显式 pin 配置决定 CO 路由，其他非 pin pipeline 配置走 JIT。

## 已恢复的候选

304 个候选的结果从 `251 ready / 53 rejected` 变为 `262 ready / 42 rejected`。11 个恢复项均通过实际 pin 校验，之前的 251 个镜像保持不变。

| kid | pipeline / 配置 | VGPR spill | SGPR spill | private segment bytes |
| --- | --- | ---: | ---: | ---: |
| 1 | 4wave，非 PAD，sp16/u4 | 818 | 0 | 1724 |
| 3 | 4wave，非 PAD，sp32/u4 | 4 | 0 | 20 |
| 5 | 4wave，非 PAD，sp64/u4 | 8 | 0 | 36 |
| 7 | 4wave，PAD，sp16/u4 | 789 | 3 | 1684 |
| 8 | 4wave，PAD，sp32/u2 | 0 | 3 | 0 |
| 9 | 4wave，PAD，sp32/u4 | 0 | 3 | 0 |
| 10 | 4wave，PAD，sp64/u2 | 0 | 4 | 0 |
| 11 | 4wave，PAD，sp64/u4 | 0 | 8 | 0 |
| 78 | m_repeat，sp64，pin A0/B1/C0 | 0 | 0 | 1040 |
| 79 | m_repeat，sp64，pin A1/B0/C0 | 0 | 0 | 1040 |
| 80 | m_repeat，sp64，pin A1/B1/C0 | 0 | 0 | 1040 |

旧 PAD 候选只有 kid 6（sp16/u2）可用。现在 kids 6–11 六个均有 CO。kids 8–11 只有少量 SGPR spill，旧规则将其与大量 VGPR spill/scratch 一起过滤，证据足以确认筛选过严；恢复并不意味着已经确认性能收益。

42 个剩余失败项的诊断都是 `no registers from class available to allocate`，发生在编译阶段，放宽编译后的资源筛选不能修复。

## 实现和记录

[`build_co.py`](../../csrc/opus_gemm/gen_co/build_co.py) 将资源策略拆出为 `_check_gfx950_spill_policy()`，在 pin 验证成功后应用。新编译和 `--skip-existing` 都遵循同一策略，资源数字和 `resource_warnings` 写入 [`gfx950/build_info.json`](../../csrc/opus_gemm/gen_co/gfx950/build_info.json)。

- 默认：记录 spill/private segment，打印警告，保留通过已有校验的候选。
- `--fail-on-spill`：拒绝任意非零 VGPR/SGPR spill。
- `--fail-on-scratch`：gfx950 拒绝非零 private segment。
- 两个开关同时开启：恢复原先的资源拒绝条件。

实际镜像、完整构建记录和 42 个编译失败诊断均随分支提交。`build_info.json` 的诊断链接已整理为相对其目录的路径；日志内构建命令保留原机器路径，用于复现当时环境。

本次完成了编译、实际 pin 校验、最终代码语法/默认与严格策略检查，以及 262 个 CO 的登记和文件一致性检查。按用户要求停止追加 GPU 测试，恢复的 11 项尚无新的运行时正确性和性能结果。

未增加 gfx950 的实际 kernarg size=96 校验、traits LDS 精确比较，也未扩展 `--skip-existing` 的 ELF 目标/链接校验；这些属于以后独立完善的构建检查，不能写成当前已具备的能力。
