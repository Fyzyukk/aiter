# MXFP8 B-preshuffle 优化交接（2026-09-25）

本分支：`Fyzyukk/aiter:aiter-opus-mxfp8-bpreshuffle`。实验源码提交为 `b7df6147`。本次交接把当前实验源码、最新完整基线、未完成实验记录及复跑工具提交到同一分支，供换服务器继续优化。本次没有启动 GPU 测试，也没有等待原机空闲。

## 先看结论和未完成事项

- **最新完整基线为 209 胜 / 86 负**：295 个支持的 M≥1024 模型 shape，另 10 项超出既有单张量有符号 32 位字节寻址范围。基线代码是 `10ab50645d1f25e11844b814b66002b27181dfbf`，已同步 upstream `b3d0cf4e`。此前 208/87、206/89 属于旧批次。
- **优化范围仍保留原 87 项**：上述 86 个负项，以及 `(4096,2048,7168)` 这个仅领先约 0.033% 的近平局项。不能因为最新基线少了一个负项就删除它。
- **9030–9033** 已接入注册和代码生成，GPU 边界检查通过；87 项五轮全后端扫描未完成，不能据此给出完整胜负统计。
- **9040–9042、9050–9051** 已实现，离线编译、CPU 检查和首次 GPU 正确性验证均通过；33 项五轮性能扫描未完成，尚未关闭任何优化目标。
- 9000/9010/9011/9012/9020 保持原实现；新 ID 均为可显式调用的实验候选，未加入默认编译集合，未更改生产调度 CSV。

最先补测 **25 个短 K + 8 个 K1536**；再按原 87 项分组继续，最终完整复核 295 项及原有胜项。要分别记录“超过旧 OPUS”和“超过最快有效外部后端”。

## 已恢复的证据

| 记录 | 当前可用结论 | 入口 |
|---|---|---|
| 当前分支全量重建、扫描、确认、回放 | 295 项；OPUS 209 胜、CKTile 86 胜；16 项追加五轮确认；295 项选择回放通过 | [完整结果](reports/opus_local_gap_current_20260925/RESULTS.md)、[最终比较](reports/opus_local_gap_current_20260925/final_comparison.csv)、[落后清单](reports/opus_local_gap_current_20260925/final_remaining_shapes.csv) |
| 当前全量候选数 | 27,690；OPUS 1,275 全有效；CK 5,160/5,310 有效；CKTile 9,535/9,735 有效；ASM 0/11,370 有效 | [全部候选记录](reports/opus_local_gap_current_20260925/profile.csv)、[拒绝记录](reports/opus_local_gap_current_20260925/rejected_candidates.csv) |
| 9030–9033 GPU 验证 | 156 项数值/保护区检查通过，12 项非法对齐被拒绝 | [validation.json](reports/opus_9030_targets87_20260925/validation.json) |
| fixed-K GPU 验证 | 96 项数值检查通过：41 项目标调用、50 项边界、5 项 raw E8M0；40 项非法 shape 被拒绝 | [validation.json](reports/opus_fixedk_gpu_20260925/validation.json) |
| fixed-K CPU/编译 | 短 K 集成检查 103/103；K1536 集成检查 127/127；布局/调度检查及 gfx950 编译通过 | [短 K](reports/opus_shortk_20260925/README.md)、[K1536 集成](reports/opus_k1536_20260925/integration/REVIEW.md)、[K1536 编译资源](reports/opus_k1536_20260925/offline/README.md) |

短 K / K1536 目录内较早文档的“GPU 待验证”描述对应当时的 CPU 阶段。以较新的 `opus_fixedk_gpu_20260925/validation.json` 和本交接页为准；性能验收仍未完成。本次交接重新执行 K1536 标准库集成检查，127/127 通过。

**中断记录的解释：** `opus_9030_targets87_20260925` 的四个 `gpu*_r5_run.json`，以及 `opus_fixedk_gpu_20260925` 的三个同名文件都残留 `status=running`。本次恢复时当前进程空间没有对应测试进程；这些是未正常收尾的记录，不能当作仍在后台执行或已完成。

- 903x 原始计时各卡分别为 103/100/99/100 行，都只触及分片的第一个 shape，未完成五轮。
- fixed-K 原始计时各卡分别为 327/315/361 行，每卡触及两个 shape；每卡仅首个 shape 有五轮汇总。不要把三个局部结果外推成 33 项结论。
- 原机器持续有外部占用，保留部分记录仅用于追溯。新服务器从全新 prefix 开始，同一个 shape 的所有候选在同卡同批重新测量。

## 源码入口和优化方向

注册：[opus_gemm_common.py](csrc/opus_gemm/opus_gemm_common.py)；生成：[gen_instances_gfx950.py](csrc/opus_gemm/codegen/gen_instances_gfx950.py)；调用：`aiter.ops.opus.opus_gemm(..., kid=..., layout="bpreshuffle", x_scale=..., w_scale=...)`。

pipeline/traits 位于 `csrc/opus_gemm/include/gfx950/`，公共文件名前缀为 `opus_gemm_{pipeline,traits}_a8w8_mxscale_bpreshuffle_`。

| ID | 几何/调度 | 文件后缀和说明 |
|---|---|---|
| 9000 | 256×256×128，4 wave | `4wave_gfx950.cuh`；冻结基准，保留 pipeline、traits 和影响它的共享 helper |
| 9010 / 9011 / 9012 | 128×128 / 64×128 / 64×64；K128 | 对应尺寸后缀；正式配置分别为 S3/K+2、S3/K+2、S4/K+3，直接 BF16 |
| 9020 | padded M | `padded_m_gfx950.cuh`，复用 9000 主体 |
| 9030 / 9031 | 192×256×128；4 / 8 wave；S64 | `192x256_gfx950.cuh` |
| 9032 / 9033 | 同几何；4 / 8 wave；S128 | 同上，scale panel 更大 |
| 9040 / 9041 / 9042 | 192×256×128，8 wave；固定 K384 / K768 / K1024 | `shortk_gfx950.cuh`；完整展开 3/6/8 步，只准备实际 scale |
| 9050 / 9051 | 同几何；固定 K1536 | `k1536_gfx950.cuh`；完整展开12步 / 两步循环 |

固定 K 入口要求精确 K，正 M/N、M%64=0、N%256=0，保留 FP8/BF16、原生 E8M0 布局、对齐和字节范围检查。固定 K 不能用于其他 K；过滤逻辑为 `a8w8_mxscale_bpreshuffle_supports_shape`，generated launcher 也有精确 K guard。

优先关注寄存器压力：9041 为 256 VGPR、无 spill；9050 为 256 VGPR、22 VGPR spills、76 private bytes；9051 为 206 VGPR、无 spill。编译资源只提供优化线索，不能替代 GPU 性能比较。不要为了减少等待而删除必要同步，也不要从短 K 结果外推长 K。

原 87 项分组为 **25 + 8 + 12 + 26 + 10 + 3 + 3**：短 K、K1536、N7168/K3072、N6144或7168/K7168、N7168/K16384、窄 N768、大 M/N2048。精确成员与旧基线冻结在 [分组台账](reports/opus_shortk_20260925/plan/targets87_ledger.csv) 和 [cohorts.csv](reports/opus_shortk_20260925/plan/cohorts.csv)。

## 新服务器准备

```bash
git clone --branch aiter-opus-mxfp8-bpreshuffle --single-branch \
  git@github.com:Fyzyukk/aiter.git
cd aiter
git submodule update --init --recursive
git rev-parse HEAD
git -C 3rdparty/composable_kernel rev-parse HEAD
```

CK submodule 应为 `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`。本交接已把当前修改提交进分支，不需要再应用旧 overlay 或旧 patch。

历史环境是 MI355X / gfx950 / 256 CU，Python 3.12、PyTorch `2.11.0+rocm7.14.0`、HIP `7.14.60850`。需要原生 `torch.float8_e8m0fnu`、可用的 ROCm/PyTorch 开发环境，以及 `rocm-smi`。新机环境不同，应保存版本并重建该环境的基线。已有匹配的 ROCm/PyTorch/Triton 环境中，可按仓库安装方式执行：

```bash
BUILD_TARGET=rocm AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  python -m pip install -e . --no-build-isolation
```

**OPUS 需要定制 clang。** 9000 使用 `clang::amdgpu_pin_agpr`；新候选引用相关布局 helper，也会遇到该编译器要求。普通 ROCm clang 不能直接代替。历史 OPUS 编译器为 `https://github.com/yuyzhang512/llvm-project.git` 的 `49c41889681640665400cb01c9fbb4c0a024cde4`（clang 24）；CK/CKTile/ASM 使用 ROCm clang 23。可迁移已构建工具链，或在新机按旧配置构建：

```bash
git clone https://github.com/yuyzhang512/llvm-project.git llvm-pin-src
git -C llvm-pin-src checkout 49c41889681640665400cb01c9fbb4c0a024cde4
cmake -G Ninja -S llvm-pin-src/llvm -B llvm-pin-build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_PROJECTS='clang;lld' -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build llvm-pin-build --target clang lld --parallel 12
```

保留 `aiter/jit/optCompilerConfig.json` 的编译语义。完整编译器版本记录见 [build.json](reports/opus_fixedk_gpu_20260925/build.json)。工具链、ROCm/PyTorch 和 `.so` 不随 Git 推送，必须在目标机器准备。

## 生成适配新机的入口

使用 [prepare_remote.py](reports/opus_remote_handoff_20260925/prepare_remote.py)，只读归档模板，在 `reports/` 下生成新目录。它不导入 torch/aiter、不查询 GPU、不编译、不测试，也不会覆盖已有目录。

先在新机核对同一张空闲卡的 **rocm-smi 物理编号、ROCr UUID 和 PCI bus**。ROCr 数字编号可能与物理编号不同。下面三个设备值必须替换；`85` 是 PCI `0000:85:00.0` 的十六进制 bus 字节。当前 harness 限定 PCI domain/device 为 0、gfx950/256CU。

```bash
python reports/opus_remote_handoff_20260925/prepare_remote.py \
  --output-dir reports/opus_remote_run \
  --gpu-index 0 --gpu-uuid GPU-0123456789abcdef --pci-bus 85 \
  --opus-clang-path /absolute/path/to/llvm-pin-build/bin \
  --stock-clang-path /opt/rocm/llvm/bin

# 仅列清单，不访问 GPU。
python reports/opus_remote_run/benchmark.py \
  --shapes reports/opus_remote_run/shapes33.csv --list-shapes
```

生成目录包含全新 `build.py`、`validate.py`、`benchmark.py`、最新 tuner adapter，以及 25/8/33/87/295 项 shape CSV。构建脚本从源码构建五个后端模块，OPUS 显式包含全部 14 个候选；不会复制旧机的 `.so`。计时脚本保留空闲检测、实际加载路径和源码/二进制哈希检查。

这些迁移入口已做 CPU 生成、语法、shape 清单检查；新机编译和 GPU 运行仍需执行。不要直接启动归档目录里的 `launch.py`，也不要直接跑归档 `build.py`：旧脚本有固定设备/路径，部分还需要旧 JIT 缓存。

## 新机重建、正确性和计时

在仓库根目录依次执行；每个命令成功后再继续。构建较重，编译时不要同时计时。

```bash
python -u reports/opus_remote_run/build.py
python -u reports/opus_remote_run/validate.py

# 先补完整 25+8 项，每个候选先检查数值，合格才计时。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 5 \
  --shapes reports/opus_remote_run/shapes33.csv \
  --jit-dir reports/opus_remote_run/jit --prefix fixedk_r5

# 后续覆盖原87项，包括全部合法旧/新OPUS和参考后端。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 5 \
  --shapes reports/opus_remote_run/shapes87.csv \
  --jit-dir reports/opus_remote_run/jit --prefix targets87_r5

# 候选优化完成后，完整回归，避免丢掉已有胜项。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 3 \
  --shapes reports/opus_remote_run/shapes295.csv \
  --jit-dir reports/opus_remote_run/jit --prefix all295_r3
```

`validate.py` 覆盖 fixed-K 的目标、边界、raw E8M0 和非法 K/对齐拒绝。benchmark 对每个合法候选独立检查数值并复查输出；修改 903x 或其他旧 kernel 时，还应适配其专项边界回归，不能只看性能目标。旧 903x 验证脚本在 [validate.py](reports/opus_9030_targets87_20260925/validate.py)，使用前需另行调整其固定 GPU 和输出目录。

若 GPU 空闲检查失败，保留此次输出，以新 prefix 重跑该 shape 的完整批次；不要删除检测或混合两次计时。prefix 相对路径落在新生成目录下，输出文件若已存在会拒绝覆盖。`validate.py` 和 `build.py` 的结果也应使用新目录保存，避免覆盖上一版控制组。

接近 ±3% 或轮次胜负不一致的 shape，从 `*_choices.csv` / `*_raw.csv` 另建包含 M/N/K 的 CSV，然后用 **同一次完整 sweep、同卡、同一套源码/二进制** 追加五轮：

```bash
python -u reports/opus_remote_run/benchmark.py \
  --final reports/opus_remote_run/all295_r3_run.json --rounds 5 \
  --shapes reports/opus_remote_run/close_shapes.csv \
  --jit-dir reports/opus_remote_run/jit --prefix close_r5
```

`--final` 保留全部合法 OPUS，以及有效外部对手中距最快者≤5%的候选。确认批次整体替换该 shape 的旧批次，不能挑两批最小值。完成性能选型后，还需把最终选择通过原生 E8M0 接口回放；旧 [replay.py](reports/opus_local_gap_current_20260925/replay.py) 可作实现参考，其历史路径和 GPU 映射需要适配。

## 必须保留的比较口径

- 原 FP32 逐元素误差界：`1e-4 + 5e-5 * sum(abs(A_i*B_i))`，区间端点舍入 BF16，`error==0`，保留 NaN 预填和输出保护区。数值失败不参与性能选优。
- 使用原 `run_perftest` profiler，warmup=5、iters=51、自动输入轮换，轮换候选顺序。公共 B shuffle、scale 解码和参考计算不计时；后端内部转换计时；CPU launch 开销不计时。
- 每个 shape 的全部对比在同一张实际空闲 GPU 上完成。新机器的耗时不与旧机器逐项拼接；旧值只作参考。
- 差距为 `(OPUS_us / reference_us - 1) * 100%`，正值表示 OPUS 更慢。上游 CSV 的历史 us 不直接替代同批重测值。计时条件差异分析见 [调查记录](reports/cktile_timing_diagnosis_20260925/README.md)。
- 9000 保持冻结；新方案用独立候选和独立 JIT 目录。新候选通过前不改默认选型；最终保留原 87 项成员、10 项寻址排除和全 295 项覆盖。

## 交回时保留什么

源码 commit、环境/编译命令、新控制组与候选的源码/二进制 SHA256、设备 UUID/PCI、完整 raw/correctness/summary/run 文件、最终 295 项比较与选择、87 项进度和仍未解决项。报告区分数值通过、超过旧 OPUS、超过最快外部对手三种结论。

本分支保存 2026-09-25 的文本报告、原始 CSV/JSON、验证源码及此前 9030 实验记录；更早的完整本地实验目录不在本次交接范围。文件清单和 SHA256 见 [manifest.json](reports/opus_remote_handoff_20260925/manifest.json)。`reports/.gitattributes` 保留证据原始字节，避免 CSV 换行归一化破坏历史哈希。

旧 JSON 内的绝对路径与二进制哈希描述原机，不要求新编译产物匹配旧哈希。未上传 JIT 缓存、编译二进制及完整工具链；新生成的运行入口以新哈希记录本次执行。
