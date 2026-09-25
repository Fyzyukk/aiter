# 9010 / 9011 / 9012 当前已验收源码替换包

六个独立头文件及两个共享头文件均逐一匹配 `reports/opus_9010_9011_opt_20260924/harness/final_r5_run.json` 的源码哈希。9012 是已验收的 9000 风格流程及后续等价命名版本；来源见 `reports/opus_9012_flow_20260924/naming/verification.json`。本包基于现有 AITER 仓库，不是独立可编译的项目。

| Kid | Tile M×N×K | Pipeline 文件后缀 | Traits 文件后缀 |
|---|---|---|---|
| 9010 | 128×128×128 | `_128x128_gfx950.cuh` | `_128x128_gfx950.cuh` |
| 9011 | 64×128×128 | `_64x128_gfx950.cuh` | `_64x128_gfx950.cuh` |
| 9012 | 64×64×128 | `_64x64_gfx950.cuh` | `_64x64_gfx950.cuh` |

完整 pipeline 前缀为 `opus_gemm_pipeline_a8w8_mxscale_bpreshuffle`，traits 前缀为 `opus_gemm_traits_a8w8_mxscale_bpreshuffle`。文件都在 `kernels/csrc/opus_gemm/include/gfx950/`。

## 替换范围

1. 在已有同源 AITER 工作区中，将 `kernels/` 内文件按仓库相对路径复制。每个 kernel 应连同自己的 traits 一起替换。
2. 本包也带了共享 `_4wave_gfx950.cuh` pipeline 和无 tile 后缀的公共 traits；它们提供三个 kernel 复用的类型和函数，也被 9000 使用。目标版本若与本包一致，可保留原文件；若不同，先核对差异后同步相同版本。
3. `integration_reference/` 是配套 codegen/registry 的完整参考文件。若目标工作区仍使用旧的 `..._small_gfx950.cuh`，须同步 codegen 中的独立 traits 选择；当前位置在 `gen_mxscale_bpreshuffle_instance()`。若目标已包含其他开发改动，合并相关段落。
4. Registry 映射为 9010→128×128、9011→64×128、9012→64×64，位于 `opus_gemm_common.py` 的 `a8w8_mxscale_gemm_bpreshuffle_kernels_list`。目标库需已有 MXFP8 bpreshuffle 的 Python/C++ 接口；本包不包含一整套接口移植。
5. 替换后在目标机器重新生成实例并编译，确保没有加载旧的 JIT `.so`。沿用此前已验证的编译器（支持 `clang::amdgpu_pin_agpr` 的 `amdgpu-pin-op-dst` 分支）及编译配置。

本包已完成源码一致性与 ZIP 内容校验，本次打包未重新编译或运行 GPU 测试。`manifest.json` 给出每个文件的角色、仓库路径和 SHA256。

## 昨天 101 项的变化

- N=768、K=7168：5→3；M=6144、8192 后续已由 9011 胜出，剩余 M=10240、12288、14336。
- N=2048、K=7168：13→3；M=1088、1152、1216、1280、1344、1408、1472、1600、1664、1728 后续已胜出，剩余 M=4096、10240、12288。
- 其余 83 项保持待处理，因此 101−12=89。89 项中 5 项今天复测仍落后，84 项历史落后待当前版本复测。

`resolved_since_101.csv` 保存新增解决的 12 项正式比较，`remaining_89.csv` 保存最新待办及来源。它们是历史全量加后续逐项更新的进度记录，不是当前版本一次全量性能验收。
