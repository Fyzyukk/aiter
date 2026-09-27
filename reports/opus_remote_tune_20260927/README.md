# 新机器调优入口：合并版 MXFP8 B-preshuffle

此目录保留为补充实验工具。用户要求的原入口已恢复为
[`csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py`](../../csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py)，
当前输入是从原始 DSV4 基线提取的
[`dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv`](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_m_ge1024_untuned_gemm.csv)，
包含 gfx950/256 CU、`M >= 1024` 的全部 305 个唯一 shape。
当前推荐命令见 [HANDOFF_MXFP8.md](../../HANDOFF_MXFP8.md)。
该入口使用正式 ID 9060–9064；下面的 21000/21310/21311/21220/21221
仍仅指本目录独立库的历史 ID，295 项是本实验工具的子集。

本入口对应本分支当前的 main / small / narrow 正式头文件。目标是比较
**新 7 候选池、旧 16 候选池、CK/CKTile/ASM**，每个 shape 的全部候选在同一张
新机器 GPU 上重新检查数值并计时。旧机器 CSV 只作历史参考。

## 环境

- AMD MI355X / gfx950，256 CU；ROCm 和带原生 `torch.float8_e8m0fnu` 的 PyTorch。
- `rocm-smi` 可用，进程可读取本机 KFD 节点信息。
- OPUS 编译器需要支持 `clang::amdgpu_pin_agpr`：
  `https://github.com/yuyzhang512/llvm-project.git`，
  已验证提交 `49c41889681640665400cb01c9fbb4c0a024cde4`。
  参数指向编译器的 `bin` 目录。工具链和 JIT 二进制不随 Git 上传。
- 初始化仓库 submodule，并按下面命令安装当前源码。

## 从分支启动

```bash
git clone --branch aiter-opus-mxfp8-bpreshuffle --single-branch \
  git@github.com:Fyzyukk/aiter.git
cd aiter
git submodule update --init --recursive

BUILD_TARGET=rocm AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  python -m pip install -e . --no-build-isolation
```

如果已经拉过这个分支，在干净工作区执行 `git pull --ff-only`，然后更新 submodule。

在仓库根目录依次运行；将编译器路径替换为新机器的实际路径。
下面默认用物理 GPU 0；八卡时把 prepare 和 launch 两处的 `--gpus` 都改成
`0,1,2,3,4,5,6,7`，也可以只选 `4,5,6,7`。

```bash
# 发现所选物理 GPU 的 UUID/PCI，生成一个新的运行目录。
python reports/opus_remote_tune_20260927/prepare.py \
  --output-dir reports/opus_remote_run \
  --gpus 0 \
  --opus-clang-path /absolute/path/to/llvm-pin-build/bin

# 在新机器重建 JIT 和独立库；ASM 设备代码使用本分支 hsa/ 中的版本。
python -u reports/opus_remote_run/build.py

# 295 个 shape，三轮，完整枚举外部后端。
python -u reports/opus_remote_run/launch.py \
  --batch full295_r3 --gpus 0 --rounds 3 --external-mode full

# 汇总新7、旧16和外部后端的同卡比较。
python reports/opus_remote_run/analyze.py --batch full295_r3
```

`build.py` 从当前 checkout 重建五个 JIT 模块和十二个独立库；其中 ASM
重建的是 wrapper，设备 `.co` 使用仓库已跟踪的 `hsa/` 版本。

`prepare.py` 拒绝覆盖已有运行目录。重新优化源码后，使用新的运行目录，例如
`reports/opus_remote_run_v2`，重新 prepare / build / launch，以保存上一版的完整结果。
GPU 正在被其他任务使用时，launch 会停止；待空闲后用新的 batch 名启动。

只想先看代表 shape 时，构建后可运行：

```bash
python -u reports/opus_remote_run/launch.py \
  --shapes reports/opus_remote_run/pilot62.csv \
  --batch pilot62_r3 --gpus 0 --rounds 3 --external-mode full
python reports/opus_remote_run/analyze.py --batch pilot62_r3
```

62 项是较小的试测集合；它的结果不能代替完整 295 项结果。

## 候选和比较口径

新池严格包含以下 7 个候选，旧控制组的胜项不会算成新池胜项：

| ID | Tile M×N×K | 源码 |
|---|---|---|
| 9000 | 256×256×128 | 原 4wave pipeline |
| 9020 | 256×256×128，M 尾块 | 原 padded-M pipeline |
| 21000 | 192×256×128 | `include/gfx950/*_main_gfx950.cuh` |
| 21310 / 21311 | 128×128×128 / 160×128×128 | `include/gfx950/*_small_gfx950.cuh` |
| 21220 / 21221 | 64×128×128 / 64×64×128 | `include/gfx950/*_narrow_gfx950.cuh` |

源码路径均相对 `csrc/opus_gemm/`。新五个 ID 是独立库中的私有 ID，
通过生成的测量 adapter 调用；不要把它们作为生产 `opus_gemm(kid=...)` 的全局 ID。

旧池为原注册 `9000、9010、9011、9012、9020`，加保留包中的 11 个 ID：
`13163、20000、20010、20011、20020、20100、20124、20125、20126、20128、20131`。
保留源码位于 `csrc/opus_gemm/mxfp8_bpreshuffle_retained/`。新机也会重建并计时这些控制组。

CK、CKTile、ASM 在新卡完整枚举，数值失败的候选排除。所有后端使用同一份
FP8 / E8M0 数据；公共 shuffle、scale 解码和 FP32 参考计算在计时之外。
保留原逐元素误差界、NaN 输出保护区和原计时方式，不增加额外边界测试。

## Shape 清单

完整输入为 [shapes295.csv](shapes295.csv)，295 个唯一 `(M,N,K)`；
每项 M≥1024，沿用既有寻址限制，未将原本排除的 10 项加入。
完整 M 值在 [shape_groups.csv](shape_groups.csv)。

| N | K | Shape 数 | M 最小值～最大值 |
|---:|---:|---:|---:|
| 768 | 7168 | 25 | 1024～32768 |
| 2048 | 7168 | 25 | 1024～32768 |
| 6144 | 7168 | 32 | 1024～65536 |
| 7168 | 384 | 25 | 1024～32768 |
| 7168 | 768 | 32 | 1024～65536 |
| 7168 | 1024 | 6 | 1024～32768 |
| 7168 | 3072 | 32 | 1024～65536 |
| 7168 | 7168 | 32 | 1024～65536 |
| 7168 | 16384 | 32 | 1024～65536 |
| 16384 | 1536 | 31 | 1024～57344 |
| 65536 | 1536 | 23 | 1024～14336 |
| **合计** | | **295** | |

## 基线 CSV

历史逐 shape 基线是
[baseline/old16_vs_external_295.csv](baseline/old16_vs_external_295.csv)：

- `opus_kid / opus_us`：旧 16 池中最快 OPUS 及耗时。
- `reference_lib / reference / reference_us`：最快有效 CK/CKTile/ASM 及耗时。
- `delta_external_pct`：`(opus_us/reference_us - 1)×100%`；正值表示 OPUS 更慢。
- `old_opus / old_opus_us`：原 5 注册候选中的最快者，区别于旧 16 池。

它是上次 2026-09-27 完整 295 项、三轮扫描的原样副本，结果为
OPUS 291 胜、CKTile 4 胜。历史整体选型见
[baseline/overall_selection_295.csv](baseline/overall_selection_295.csv)；
来源路径和文件哈希见 [baseline/provenance.json](baseline/provenance.json)。
旧绝对路径、UUID 和哈希是历史来源记录，不是新机运行配置。

新机性能结论应使用新 batch 的 `analysis/` 结果。新 7 池对旧 16 池、对外部后端
分别比较；不要把旧 CSV 的微秒数拼进新批次或从多次批次里逐候选挑最小值。

以 `full295_r3` 为例，主要输出位于 `reports/opus_remote_run/full295_r3/analysis/`：

| 文件 | 含义 |
|---|---|
| `comparison.csv` | 每个shape的新7最快、旧16最快、外部最快，以及两种耗时差 |
| `new_pool_choices.csv` | 新7池内的逐shape最佳候选 |
| `old16_choices.csv` | 同批重测的旧16池内最佳候选，是新机回归对照 |
| `overall_selection.csv` | 新7池加外部后端的整体最终选型 |
| `candidate_usage.csv` | 各候选的有效数、失败数和选中次数 |
| `remaining_losses.csv` | 新7池仍慢于外部后端的shape |
| `regressions_vs_old16.csv` | 新7池相对旧16池退化的shape |
| `RESULTS.md` | 总体胜负、几何平均差异、数值检查和候选用量 |

`delta_old16_pct=(new_pool_us/old16_us-1)×100%`，
`delta_external_pct=(new_pool_us/external_us-1)×100%`；正值表示新池更慢。

## 当前完成状态

新三个 family 已编译，五个入口与样式整理前的计算指令和归一化资源描述一致。
最新 small/narrow 版本及最终七候选组合的完整 GPU 比较尚未完成，此入口用于完成
该比较。原来中断的队列、旧 JIT、旧 GPU 映射不会作为新机运行状态使用。
`reports/opus_native_style_20260927/build_verify.py` 是原机历史等价性验证工具；
新机重建使用本页生成的 `build.py`。

迁移包的隔离 checkout 检查和实际独立库编译结果见 [VALIDATION.md](VALIDATION.md)。
