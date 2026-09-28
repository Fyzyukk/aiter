# OPUS MXFP8 tune 集成文件清单（提交前审计）

本次提交前的分支 `aiter-opus-mxfp8-bpreshuffle`，父提交 HEAD `90a09bec3040`，跟踪 `origin/aiter-opus-mxfp8-bpreshuffle`（`90a09bec3040`）。
本清单比较基线是 `git merge-base HEAD upstream/main` 得到的 `b3d0cf4e0939`。当前 `upstream/main` 已是 `049fae4d0ec5`，未将其较新的内容并入本次清单基线。

最终工作区相对 merge-base 有 **78 个非 reports 净变更文件**。HEAD 到工作区有 **35 个待提交路径**：7 个已跟踪修改、14 个已跟踪删除、14 个未跟踪新增。
完整审计清单共 **92 个路径**，其中 14 个文件仅曾在分支中新增、现已删除，因此保留其删除记录，但不计入最终净变更。

## 提交依赖与范围

- 14 个尚未 tracked 的生产头文件必须与 codegen/common 的变更及 14 个旧头文件删除一起提交，否则干净 checkout 的实例生成将引用缺失头文件。
- 旧名 opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh 仍被 retained 源码引用，不属于待删除旧头文件。
- shared generic tuner 保持无净改动：aiter.utility、CK blockscale generic tuner 和 opus_gemm_tune.py 的相关改动均已恢复；专用行为局限于 opus_gemm_mxscale_bpreshuffle_tune.py。
- 非 reports 的未 tracked 文件仅上述 14 个生产头文件；reports 下未 tracked 的历史源码、压缩包和运行产物不是生产构建依赖。文档所需轻量验证证据可由根 agent 单独选择提交。
- 保留包属于已提交的历史实验 kernel / 重建材料，其私有 ID 不加入当前生产 registry；不应将其描述为当前 8 个生产候选之外的生产候选。

本报告只进行源码和 Git 状态读取；只新写本清单与对应 JSON，没有修改生产文件，也没有执行 Git 写操作。

## 分类总数

| 类别 | 最终净变更文件数 |
|---|---:|
| 调度与注册 | 3 |
| Codegen / JIT | 2 |
| 专用 tuner | 1 |
| 生产 kernel / traits | 17 |
| 保留实验 kernel 与重建材料 | 50 |
| 模型 shape 配置 | 1 |
| 文档 | 3 |
| 许可证 | 1 |

## Shared generic tuner 状态

以下文件相对 merge-base 没有净改动，工作区也没有待提交修改；专用 native E8M0 / FP32 reference 行为保留在 MXFP8 adapter 中。

- [aiter/utility/base_tuner.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/aiter/utility/base_tuner.py)
- [aiter/utility/mp_tuner.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/aiter/utility/mp_tuner.py)
- [csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py)
- [csrc/opus_gemm/opus_gemm_tune.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/opus_gemm_tune.py)

## 完整文件清单

状态定义：`A/M/D` 分别表示新增/修改/删除；工作区 `??` 为尚未跟踪的新文件；`clean` 表示该文件已提交且工作区干净；最终净状态 `none` 表示其相对 merge-base 已无文件差异。

### 调度与注册

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [aiter/ops/opus/gemm_op_a8w8.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/aiter/ops/opus/gemm_op_a8w8.py) | `M` | `clean` | `M` | 将 packed E8M0 A scale 视图的元数据规范为物理 [K/128,M] ABI；不重排数据。 |
| [aiter/ops/opus/launch_plan.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/aiter/ops/opus/launch_plan.py) | `M` | `clean` | `M` | 将 a8w8_mxscale_gemm_bpreshuffle tag 接入已有 B-preshuffle family。 |
| [csrc/opus_gemm/opus_gemm_common.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/opus_gemm_common.py) | `M` | `M` | `M` | 注册 compact-E8M0 候选与形状/字节边界；工作区统一为 9000、9010、9020–9024、9030，并加入 large-output 合约。 |

### Codegen / JIT

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [csrc/opus_gemm/codegen/gen_instances_gfx950.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/codegen/gen_instances_gfx950.py) | `M` | `M` | `M` | 生成 gfx950 host/device 实例、traits/kernel 绑定及输入检查；工作区绑定独立几何头文件和 large-output 64 位 C 基址边界。 |
| [csrc/opus_gemm/gen_instances.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/gen_instances.py) | `M` | `clean` | `M` | 将新 tag 纳入输入 dtype、B-preshuffle exact-kid dispatch 与实例生成。 |

### 专用 tuner

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py) | `A` | `clean` | `A` | gfx950 专用联合调优 adapter；复用 CK/CKTile/ASM 后端和 OPUS subset JIT，分别使用 native E8M0 / FP32 scale 数据及参考。 |

### 生产 kernel / traits

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh) | `none` | `??` | `A` | 9021：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh) | `none` | `??` | `A` | 9022：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh) | `none` | `??` | `A` | 9010：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh) | `none` | `??` | `A` | 9023：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh) | `none` | `??` | `A` | 9024：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh) | `A` | `M` | `A` | 9000 原始 256x256 主体及生产候选共用矩阵 layout helpers；工作区已拆出 9010 并整理布局。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh) | `none` | `??` | `A` | 9020：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh) | `none` | `??` | `A` | 9030：独立运行时 K pipeline / kernel 主体；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_main_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_main_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh) | `A` | `clean` | `A` | 历史 192x256 traits；保留包的 long_epilogue_sync / long_runtime / long_runtime_grid / long_runtime_fused 仍引用，需保留。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh) | `none` | `??` | `A` | 9021：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh) | `none` | `??` | `A` | 9022：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh) | `none` | `??` | `A` | 9010：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh) | `none` | `??` | `A` | 9023：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh) | `none` | `??` | `A` | 9024：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh) | `none` | `??` | `A` | 9020：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh) | `none` | `??` | `A` | 9030：固定几何、流水线及 LDS 常量；当前生产 codegen 的必需头文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh) | `A` | `M` | `A` | 共享 kargs ABI 与 9000 固定 traits；工作区集中布局/流水线常量。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_main_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_main_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |
| [csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh) | `A` | `D` | `none` | 分支历史中的旧候选或旧 family 命名文件，工作区已删除；当前 codegen 已切换到独立几何文件。 |

### 保留实验 kernel 与重建材料

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/audit_device_kernels.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/audit_device_kernels.py) | `A` | `clean` | `A` | 只读解析 code object 并核对历史 device 入口与机器码证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/build_retained.py](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/build_retained.py) | `A` | `clean` | `A` | 从包内源码和命令模板重建保留共享库，无需旧 reports 实验源码。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/device_audit.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/device_audit.json) | `A` | `clean` | `A` | 保留的 11 个 device kernel 入口与机器码审查结果。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/experiments.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/experiments.json) | `A` | `clean` | `A` | 保留库与历史私有 ID 的调优配置索引。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/pipeline_generic.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/pipeline_generic.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/pipeline_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/pipeline_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/traits_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/traits_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/pipeline_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/pipeline_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/traits_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/traits_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/pipeline_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/pipeline_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/traits_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/traits_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/pipeline.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/pipeline.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/traits.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/traits.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/pipeline.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/pipeline.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/traits.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/traits.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/retained_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/retained_manifest.json) | `A` | `clean` | `A` | 历史选择结果与源码来源追溯清单。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/pipeline_short_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/pipeline_short_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/traits_short_runtime.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/traits_short_runtime.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/pipeline.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/pipeline.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/traits.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/traits.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/build_manifest.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/build_manifest.json) | `A` | `clean` | `A` | 保留库构建命令模板、源码/产物哈希和构建证据。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/launch.hip](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/launch.hip) | `A` | `clean` | `A` | 对应历史私有候选库的 standalone C ABI launcher；不加入生产全局 registry。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/pipeline.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/pipeline.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/traits.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/traits.cuh) | `A` | `clean` | `A` | 对应历史私有候选库的 kernel / traits 源码，供保留包重建。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/variants.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/variants.json) | `A` | `clean` | `A` | 该历史库的私有 variant / 几何配置。 |

### 模型 shape 配置

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv](/root/workspace/aiter-opus-mxfp8-bpreshuffle/aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv) | `A` | `clean` | `A` | 305 个 M >= 1024 shape 的未调优输入清单；不是新增生产 winner 表。 |

### 文档

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [HANDOFF_MXFP8.md](/root/workspace/aiter-opus-mxfp8-bpreshuffle/HANDOFF_MXFP8.md) | `A` | `M` | `A` | 集成、候选演进、源码约定、验证与迁移交接记录。 |
| [csrc/opus_gemm/README.md](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/README.md) | `M` | `M` | `M` | 生产候选、文件映射、运行时约束、调优命令与验证入口说明。 |
| [csrc/opus_gemm/mxfp8_bpreshuffle_retained/README.md](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/mxfp8_bpreshuffle_retained/README.md) | `A` | `M` | `A` | 历史保留 kernel 包、私有 ID 与当前生产 ID 的区别，以及重建说明。 |

### 许可证

| 实际路径 | 分支已提交差异 | 工作区待提交 | 最终净差异 | 职责 |
|---|---|---|---|---|
| [csrc/opus_gemm/licenses/gcnasm-LICENSE](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/licenses/gcnasm-LICENSE) | `A` | `clean` | `A` | 随导入的 gcnasm/OPUS kernel 保留 Apache-2.0 许可证。 |

机器可读清单：[integration_inventory.json](/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9021_9030_style_20260928/integration_inventory.json)。
