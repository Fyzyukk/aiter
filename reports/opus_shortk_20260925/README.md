第一组 **25 个短 K 目标的独立候选已实现**，使用新 ID **9040/9041/9042**。CPU 布局、调度检查和 gfx950 离线编译已通过；GPU 数值正确性与性能仍待验证，当前没有任何目标被标记为性能问题已解决。全程未使用或探测 GPU。

9000 的 pipeline、traits、共享 helper、旧注册参数和生成的 launcher 保持原样。9030～9033 的已有工作区改动也保留。新候选没有加入默认编译集合，没有更改调优 CSV 或默认选型。当前工作基于 `10ab5064`。

| 新 ID | 固定 K | K128 步数 | 首批目标 M | 首批目标 N | 目标数 |
|---:|---:|---:|---|---:|---:|
| 9040 | 384 | 3 | 1024～1728，步长 64 | 7168 | 12 |
| 9041 | 768 | 6 | 1024～1728，步长 64 | 7168 | 12 |
| 9042 | 1024 | 8 | 1024 | 7168 | 1 |

三个实例共用一个新的固定 K 模板，采用 **192×256×128、8 waves / 512 threads**，沿用 9031 的矩阵布局与两份 LDS 矩阵缓冲。固定 K 后，在编译期生成完整的 3/6/8 步，不再保留运行时 K 循环、stage 切换或 scale refill 分支。一次只准备实际需要的 3/6/8 组 scale，代替通用 S64/S128 panel 的准备代码。

所有有效 K 数据块仍参与计算，K+2 预取只在后续数据块存在时生成。必需的 producer/consumer 等待、barrier 和操作数最后一次使用后的替换顺序保留。这里没有把去掉正确性同步作为优化手段，也没有引入 wave 分组 ping-pong。较少的控制与 scale 准备成本能否抵消展开后的寄存器压力，需要实测。

源码与接入位置：

- [独立 pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh)：固定调度、scale 准备及输出。
- [独立 traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh)：固定 K、布局参数及 LDS 大小。
- [注册与标量 shape 契约](../../csrc/opus_gemm/opus_gemm_common.py)：新 ID 和 `fixed_k` 元数据；旧实例默认 `fixed_k=None`。
- [gfx950 代码生成](../../csrc/opus_gemm/codegen/gen_instances_gfx950.py)：独立符号/文件名及 launch 前的精确 K 检查。
- [本组 tuner adapter](tune_adapter.py)：候选过滤调用同一标量契约；CPU 检查只提取函数 AST，不导入这个 GPU-aware adapter。

入口的一般形状条件是正 M/N、M%64=0、N%256=0、各 ID 的精确 K，并满足已有 gfx950、FP8 输入、BF16 输出、batch=1、连续矩阵、原生 E8M0 scale 布局、地址对齐及输出不重叠检查。A/B/C 单张量字节范围不超过 `2^31-1`。首轮验收范围限定为上表 25 项；注册支持更广的合法 M/N，不代表那些组合已有 GPU 正确性或性能证据。

**离线编译结果**记录在 [offline_compile.json](offline_compile.json)。使用固定版本的 clang，直接指定 `--offload-arch=gfx950`，启用 `-verify-machineinstrs`；编译三份 device ISA 和含新旧实例的 host TU，未加载或执行编译产物。

| ID | VGPR | SGPR | AGPR | LDS 字节 | scratch 字节 | VGPR/SGPR spill | 向后分支 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9040 | 212 | 34 | 0 | 118854 | 0 | 0 / 0 | 0 |
| 9041 | 256 | 40 | 0 | 119436 | 0 | 0 / 0 | 0 |
| 9042 | 246 | 40 | 0 | 119824 | 0 | 0 / 0 | 0 |

9041 的 256 个 VGPR 是后续检查的重点。没有 spill 不等于寄存器成本低，也不证明 occupancy 或速度优于旧候选。三个实例的矩阵 LDS 均为 118272 字节；按需 scale 使总 LDS 降至约 116～117 KiB，但仍不足以仅靠 LDS 容量让两个这样的工作组同时驻留。

新 kernel 本身使用编译器寄存器分配，但读取未改动的 4wave 布局 helper 会传递其 pin-attribute 编译器检查，因此离线构建仍使用 `/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin/clang++`。这不是对普通 ROCm clang 的兼容性声明。host TU 只有既有头文件/编译选项警告，没有编译错误。

**CPU 检查**见 [布局与调度报告](validation/README.md) 和 [集成检查](integration/REVIEW.md)：

- 执行生产矩阵布局 helper、提取的 scale 读写及固定步调度，覆盖两槽边界、每个 K/M/N repeat、A/B 读取、M64 尾行及输出坐标。
- 检查 3/6/8 次矩阵预取、每 wave 72/144/192 次 MFMA 调用、4/7/9 个 barrier、一次 scale-panel load；没有越界 group、过早 LDS 复用或操作数替换。
- 103/103 项集成检查通过：每个目标只得到对应的新 ID；其他 K 被过滤且被生成的 C++ guard 拒绝。295 项旧候选集合、历史 1275 条候选记录、旧生成文件与源码 SHA、230 个生产配置文件和默认编译集合保持一致；另有 4098 个标量边界用例通过。

CPU 检查确认坐标与逻辑调度一致；它不模拟硬件延迟，不执行 MFMA 数值计算，也不能证明全部 ISA hazard 或硬件 buffer OOB 行为。GPU 正确性和至少五轮同批性能比较仍是验收条件。

以下命令只执行 CPU 工作，先重新生成代码并编译，再核对对应产物：

```sh
python reports/opus_shortk_20260925/offline_compile.py
python reports/opus_shortk_20260925/validation/check_layout.py
python reports/opus_shortk_20260925/integration/review.py
```

[87 项逐组台账](plan/README.md) 按共同特征分成 **25+8+12+26+10+3+3**，成员不重叠且没有遗漏。当前仅首组完成候选实现与 CPU 检查；其余组保持待实施，所有组的 GPU 验收保持待验证。下一组是 **N=16384、K=1536、M=1088～1536（步长 64）的 8 项**，应单独评估 12 步流水，不能从这组 3/6/8 步外推结论。

基线仍是已完成的 295 项：OPUS 209 胜、86 负。原 87 项包含这 86 个负项及一个近平局项；原选型和计时均已冻结，新候选时间保持空值。即使以后首组全部通过，仍有 62 项待解决；全部完成前还需复核完整 295 项并保留已有胜项。
