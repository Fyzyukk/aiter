# 更新 9010/9011/9012 后的全量联合调优（2026-09-25）

当前状态：新源码适配和独立编译完成，185 项专项测试通过（99 GPU、86 CPU/接口）。本目录的单卡扫描按用户要求在 24 个 shape 后停止；新的全量测量和最终结果位于 `../opus_mxfp8_full_retune_4gpu_20260925/`。本目录的部分性能数据不参与新一轮结果。

## 实测范围

- 原模型目录全部 305 个 gfx950/256-CU、M≥1024 shape；295 个存在合法 OPUS 候选。另 10 个超过当前 tensor 地址范围，列于 `deferred_shapes.csv`。
- 全部已注册 OPUS：9000、9010、9011、9012、9020。9030–9033 是未注册实验，不属于此生产接口的候选。
- 完整候选共 27,690：CK 5,310、CKTile 9,735、ASM 11,370（包含合法 split-K 分区）、OPUS 1,275。枚举清单见 `expected_all_candidates.csv`。
- 每个 shape 的全部候选使用同一份 FP8/E8M0 输入，参考后端接收对应的精确 FP32 scale 解码。数值标准保持原 FP32 累加区间和零超界元素要求。
- 每个通过正确性检查的候选测三轮，每轮 warmup=5、iters=51，使用原 profiler、自动输入轮换并轮换候选顺序。计时包含后端内部转换，排除公共输入准备、B preshuffle、scale 解码和参考计算。
- GPU 固定物理卡 7、PCI F5:00.0，每个 shape/轮次之间检查外部 GPU 活动。源码和二进制在开始/结束核对哈希。

## 新源码适配

`user_supplied_initial/` 保存用户刚替换的三个 pipeline 及当时 traits 原文，`integration.patch` 记录本次适配。

- 将独立源码的 `traits.hpp`、`kernel_ops.hpp` 接到仓库的独立 traits 和共享 4-wave 辅助函数。
- 保留新的 A LDS XOR 布局、打包 scale、编译器寄存器分配，以及 9012 的 N768/K7168 CTA 映射。
- 新 pipeline 将每个 B scale 复制为 uint32；三个 traits 的 SFB 区从 64 字节扩为 256 字节，防止 LDS 区域越界。A packed scale 保持原总字节数。
- 本轮使用现有已注册参数：9010=S3/K+2、9011=S3/K+2、9012=S4/K+3；独立源码内可选 S2 分支未配置成额外候选。9012 对接现有直接 BF16 单次 launch，内部 SPLIT_K=1 对应 tuner 的 splitK=0。
- 为 9012 补齐 host stub、LDS 布局别名及 scale 打包常量。共享 9000/9020 源码、公共数值标准和编译配置保持原值。

`jit_03/` 是根据适配后当前源码新编译的 OPUS 模块；CK/CKTile/ASM 使用源码/二进制哈希匹配的既有模块，所有性能重新测量。
`jit_01` / `jit_02` 在准备参考清单时因历史清单的相对路径解析退出，没有开始编译或 GPU 计时；已修正路径处理后完成 `jit_03`。

## 记录

- `jit_03_build.json` / `jit_03_build.log`：编译、源码快照和二进制哈希。
- `jit_03_validation.json` / `jit_03_tests.xml`：185 项测试结果。
- `full_r3_run.json`：扫描状态、参数和来源；`full_r3.log`：进度日志。
- `full_r3_candidates.csv`：候选覆盖；`full_r3_correctness.csv`：所有数值检查；`full_r3_raw.csv`：原始逐轮时间。
- `full_r3_summary.csv`：候选三轮中位数；`full_r3_choices.csv`：逐 shape 选择。

完整扫描通过后运行 `python reports/opus_mxfp8_full_retune_20260925/summarize.py` 复核覆盖和中位数，并导出可回放的原生 E8M0 选择 CSV。旧 206/89 只用于比较胜负状态变化，旧时间不参与本轮选优。
