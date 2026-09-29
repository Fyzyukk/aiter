# 通用小 M 候选：代码完成，GPU 未运行

2026-09-29。已恢复上一轮记录，并按用户“先做完，别跑 GPU”的要求完成此前约定三步的代码和 CPU 检查。每个小 M 候选仍固定一个 tile、使用运行时 K。

## 本轮改动

1. **9020 / 9022 的 M 尾块覆盖**：注册表和生成入口由 M 按 64 行对齐放宽到按 16 行对齐。A/C 原有有界访存和 16 字节 A scale 向量加载对应这一约束；设备 pipeline 未改。在已有 290 个 M≤512 shape 中，两者各新增覆盖 162 个，包含 M=144/160/176 的宽 N 情况。
2. **small-LDS 的写回开销**：共享 MFMA16 写回函数按相邻累加片段配对；E_N 为奇数时也能跨 M 配对。9043 的每个 wave 从两次半数 lane 写回变为一次全部 lane 写回。相同编译选项下，9043 静态 `ds_bpermute_b32` 从 4 条减至 2 条，`buffer_store_dwordx4` 从 2 条减至 1 条，设备函数从 7,744 字节减至 7,660 字节；VGPR/SGPR 仍为 48/62。这些是编译结果，不是耗时测量。短 K 原有的一次性矩阵预取流程保留。
3. **新增 9046**：64×128 small-LDS、4 wave、wave 分工 2×2，6 个预取槽位，长 K 每两个 K128 共用一次同步，wave 内打包 C 直写。支持 M=1–512、N 为 128 的倍数、K=128–16384 且为 128 的倍数，并保留张量字节数限制。最大动态 LDS 为 160,384 字节，小于 gfx950 的 160 KiB 上限。

相对本目录 `initial_source/`，生产改动限于四个文件：

- `csrc/opus_gemm/opus_gemm_common.py`
- `csrc/opus_gemm/codegen/gen_instances_gfx950.py`
- `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh`
- `csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh`

原有大 tile 设备代码、Python wrapper、共享 tuner、数值比较器均未改；初始记录中其余 28 个源文件的哈希保持一致。没有提交或推送。

## 已完成的检查

- 290 个 shape 的候选覆盖差异、9020/9022 的 M 对齐限制、9046 的 M/N/K 边界检查。
- 15 个公开入口的 CPU 代码生成；9040–9046 每个入口只有一次固定 tile 的 kernel launch，使用运行时 K。
- 对使用共享打包写回的 9040/9043/9044/9046，以普通 MFMA16 输出坐标为参考检查 14,848 个元素，以及完整的 tile 内 M 尾块和不同输出步长。
- 用现有 LLVM 和编译选项离线编译 9040–9046，开启机器指令验证；全部通过，全部没有 scratch、VGPR spill 或 SGPR spill。9046 使用 116 VGPR、76 SGPR、0 AGPR。仅生成并读取设备对象，没有加载 HIP/JIT 模块或启动 GPU。
- 动态 LDS 上限的编译期检查，以及四个改动文件的空白检查。

完整检查记录：[cpu_ready/validation.json](cpu_ready/validation.json)。本轮代码差异：[cpu_ready/changes.diff](cpu_ready/changes.diff)。离线编译命令、日志和设备指令均保存在 `cpu_ready/`。

## 实测状态

本轮 **没有运行 GPU 正确性检查、benchmark 或 tune**，也没有排队或安排后续 GPU 任务。9020/9022 新覆盖的 M 尾块、改进的 9043 写回，以及 9046 的运行时数值正确性和性能仍需 GPU 验证。

最后一轮正式的通用版实测仍是 `../opus_mle512_generic_tune_20260929/full290_r3/`：OPUS 胜出 234/290，几何平均加速 1.159373×，56 个落后项。该结果对应修改前版本，不包含本轮改动；旧的 273/290 成绩属于按 shape 分发版本，不能用作当前通用版成绩。
