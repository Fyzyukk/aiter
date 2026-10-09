# CK / ASM 同候选慢例归因（2026-10-08）

CK 的重点慢例已确认包含 CK 源码版本变化造成的退化；ASM 设备二进制未变，慢值受缓存与主机提交节奏影响，其中 splitK 输出清零的设备耗时有明显变化。不能用总体几何平均接近，推断所有 shape 已复现历史时间。

本报告为诊断，不修改 745-shape 正式选型、原始 profile 或编译器路由。

## 候选身份与历史来源

| M×N×K | 后端 | 基线 / 本次对应候选 ID | splitK | 实际 kernel | 基线 µs | 本次全量 tune 同候选 µs |
|---|---|---|---|---|---:|---:|
| 96×7168×7168 | CK | 17 / 17 | 0 / 0 | 256-thread、64×64×256、intrawave v1 | 23.2427 | 31.3888 |
| 112×7168×3072 | CK | 17 / 17 | 0 / 0 | 同上 | 12.1297 | 15.6641 |
| 240×768×7168 | ASM | 31 / 31 | 8 / 8 | 80×128 | 10.8125 | 13.4949 |
| 64×6144×7168 | ASM | 20 / 20 | 5 / 5 | 64×128 | 14.9877 | 18.8820 |

CK 完整名称：
`a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1`。
ASM 使用 `kernelName + splitK` 确定实际调用，数字 ID 是枚举序号；全 133 项中 91 个 ID 不变、42 个重新编号，但全部实际配置在新 tune 测过。CK 的 splitK 参数在此 wrapper 中计算后没有传入 kernel。

表中是重测基线候选的时间，区别于新 tune 最终重新选出的赢家：96×7168×7168 全后端选 OPUS9044，112×7168×3072 也选 OPUS9044；240×768×7168 最终选 ASM15/split8。因此“新选型 ID”和“基线候选在新环境下的时间”不能混用。

沿 CSV 历史追到微秒值首次写入处（而非后来 bw 单位转换提交）：

| 行 | 原始 us 提交 / PR | 当时 CK gitlink |
|---|---|---|
| CK96×7168×7168 | `559411e9253e3dcafb71fb38fe5989449b421327` / [#3238](https://github.com/ROCm/aiter/pull/3238) | `33b62ed0878369db891a85d743576605e62b3d1c` |
| CK112×7168×3072 | `cd5d265b9d59e700c3c9bafa4b7a69ddcd999a00` / [#3383](https://github.com/ROCm/aiter/pull/3383) | `83566edb0fded5e1c618c2c19110adbb74532762` |
| ASM240×768×7168 | `46e6c92b3eb33f64823aaa1ff39a14586b059ef5` / [#3339](https://github.com/ROCm/aiter/pull/3339) | `83566edb0fded5e1c618c2c19110adbb74532762` |

当前 CK 为 `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`。CK common wrapper 与上述历史提交字节相同；ASM wrapper 也相同，6 个 B-preshuffle `.co` 均与 #3339 提交中字节相同。完整行、源码及二进制 SHA 见 [candidate_provenance.json](candidate_provenance.json)。历史 runner 实际加载的 binary hash 和 compiler receipt 仍未知。

## CK：同 ID、同模板参数，CK 版本不同导致代码不同

把两个历史 CK revision 隔离导出，在同一 Clang23 / ROCm7 SDK / 原编译 flags 下，编译相同的当前 ID17 生成 TU 和 common wrapper。原地 CK checkout 未动。旧 CK 的 BF16 类型是 uint16_t，当前 CK 在 ROCm7 下用 builtin __bf16，所以旧设备 object 不能混入当前 host registry；诊断使用各自兼容的固定 ID17 host entry，另用原正式 host entry 作控制。

每个 shape 的所有版本在同一个 GPU、相同输入和地址池测量。128 warmup / 256 iterations，3 个随机正反序块，共 6 次测量，沿用原 profiler 的有效 GPU event 求和及异常值过滤，逐次复查输出。当前诊断 Clang23 的指令 hash 与正式 Clang23 ID17 一致；两个历史 CK revision 的指令 hash 也一致。

第一轮中位时间如下（µs，全部 Clang23）：

| M×N×K | 当前 CK，复用1地址 | 历史 CK，复用1地址 | 当前版本延迟变化 | 当前 CK，16地址 | 历史 CK，16地址 | 当前版本延迟变化 |
|---|---:|---:|---:|---:|---:|---:|
| 96×7168×7168 | 27.4506 | 23.7525 | +15.57% | 31.0561 | 28.3064 | +9.71% |
| 112×7168×3072 | 12.9785 | 11.0299 | +17.67% | 15.3051 | 14.2037 | +7.75% |
| 80×7168×7168 | 27.1078 | 23.1509 | +17.09% | 30.8492 | 28.2789 | +9.09% |
| 80×7168×3072 | 12.8606 | 10.8101 | +18.97% | 14.9733 | 14.0663 | +6.45% |
| 128×7168×7168 | 21.3292 | 17.8381 | +19.57% | 22.8953 | 22.7648 | +0.57% |
| 128×7168×3072 | 11.1718 | 9.7121 | +15.03% | 12.0329 | 11.9396 | +0.78% |

精确数据以 [ck_ab.csv](ck_ab.csv) 为准。历史 CK33b62 的 96×7168×7168 复用时间 23.7418µs，与基线 23.2427µs 相差 +2.15%。当前 CK 的地址复用收益及版本回退收益共同解释该重点慢例的大部分差距。112×7168×3072 的历史 CK 复用时间反而快于上游值，说明历史协议未知，不能把跨历史差值强行做精确加法分解。

补齐路径在轮换地址下更显著，但版本差异也影响无需补齐的 M=128 的复用表现，不能只归为 padding。源文件有 WarpTileConfig、KGroup、数据类型、kernarg 等变化；这里实证的是 CK revision 这一变量，尚未二分到某一条源码修改。

第二轮隔离 `-DCK_USE_LLVM_BUILTIN_BF16=0`，其它条件不变：

| M×N×K | 当前 CK23 | 历史 CK23 | 当前 CK23，builtin BF16关闭 |
|---|---:|---:|---:|
| 96×7168×7168，复用 | 27.4204 | 23.7003 | 27.6319 |
| 112×7168×3072，复用 | 12.9942 | 10.9994 | 13.2102 |

这两个 shape 没有因恢复旧 BF16 类型而恢复性能；不能将版本退化仅归因于 builtin BF16。详见 [ck_bf16_ab.csv](ck_bf16_ab.csv)。

另编当前 CK / Clang20 作为诊断：96×7168×7168 为 615.46µs，112×7168×3072 为 262.53µs。对应 hot-loop 特化有 256 VGPR、620B/thread scratch、277 VGPR spills；正式 Clang23 为 204 VGPR、scratch0、spill0。**当前正式慢几十微秒的 CK 不属于这个 Clang20 spill 问题**。原历史编译器未知，也不能反推历史基线用了 Clang20。指令及 metadata：[ck_id17_metadata.json](ck_id17_metadata.json)。

## ASM：设备代码相同，缓存和提交节奏改变完整操作耗时

splitK>1 的 ASM 在 GEMM 之前调用 hipMemsetAsync 清零输出，供后续原子累加。两部分必须计入完整调用。官方 profiler 算 device event 时间之和，不含 CPU 开销及 kernel 间 idle gap；但主机提交间隔仍能改变 GPU cache/运行状态和每个 device event 的时间。

两轮使用完全相同正式 ASM 库、同 GPU、同 shape 和 128/256 协议，分别拆分清零与 GEMM。240×768×7168（同一 GPU `0000:85:00.0`）复用1地址：

| 独立运行 | 完整调用 µs | 清零 µs | GEMM µs |
|---|---:|---:|---:|
| 第一轮中位 | 10.3663 | 2.5718 | 7.8081 |
| 第二轮中位 | 12.5858 | 4.3076 | 8.3004 |
| 历史基线 | 10.8125 | 未记录 | 未记录 |

各分项分别取中位数，因此中位分项之和与中位总时间可能略有差异。**相同 `.co` 已经能测到接近基线的完整操作；残留 +19% 不是稳定的机器码退化。** 两轮之间清零增加约1.74µs，占总差约2.22µs的大部分。尚不能证明上游省略了清零；历史 wrapper 与本次相同。

64×6144×7168 在第一轮复用为14.9886µs，64地址池为18.8951µs；基线14.9877µs。其地址轮换慢值主要增加在 GEMM 部分（约11.27→14.87µs），清零约3.72→4.03µs。详见 [asm_components.csv](asm_components.csv)。

进一步在 8 卡分别对相同两个 ASM shape 测试：官方 Python 调用；预先转换 tensor 描述符后直接调用同一个 C ABI；直接 C ABI 每次提交前增加20µs主机间隔。**所有调用保留原 wrapper 的输出清零及 GEMM，数据、splitK、二进制均相同，每个对照同卡同地址。** 每配置4个随机正反序块/8次测量。固定 CABI 消除了每次 tensor 参数转换的可变 CPU 开销。

240×768×7168 在 GPU `0000:05:00.0`：

| 调用方式 | 完整操作 µs | 清零 µs | GEMM µs | 相邻设备 event 平均 gap µs |
|---|---:|---:|---:|---:|
| 官方调用 | 9.389 | 1.771 | 7.627 | 4.934 |
| 固定参数、连续 CABI | 12.476 | 3.798 | 8.681 | 0.011 |
| 固定参数、增加20µs提交间隔 | 10.696 | 2.742 | 7.942 | 9.126 |

提交节奏不同可使相同 ASM device code 的事件计时变化。CPU 更快、gap更小没有使 device event 求和必然更小；不能把 GPU idle gap 直接加到本次报告的 us。8卡官方调用中位值约8.97–12.16µs，固定CABI连续提交更集中于12.42–12.69µs。因未控制或测得原始历史时钟/功率/driver状态，不能继续细分成“必然是频率”或某一HIP实现分支。详见 [asm_pacing.csv](asm_pacing.csv)。

## 验证范围和限制

- 1,600 条完成的有效计时全部复查正确性，最大 errRatio 0.0148112，小于0.05门槛。
- 8GPU同时运行，物理锁和namespace-safe KFD owner监控无外部进程污染。第一轮有1个shard被遗留诊断输出阻止启动，随后独立重跑完成；第二轮和提交节奏实验均8/8完成。
- 初始import/参数适配及HIP graph诊断失败记录另存。Graph profiler发生不完整事件及一次进程崩溃，其数据全部排除，结论只使用完整普通host调用trace。详细记录见 [summary.json](summary.json)。
- 所有有效原始测量汇总为 [valid_raw.csv](valid_raw.csv)。构建脚本、commands、object及binary hash均保留；历史CK仅隔离导出，当前子模块保持af9且clean。
- 已核对正式745行选择、79,504条raw profile、上游baseline文件SHA全部未变。
- CK revision效应目前只对ID17及8个shape验证，不能外推全部409项。ASM诊断说明实际慢值可由相同device code的测量/运行状态重现，不能确定上游原始测量使用了哪一种协议。
- 上游公开机器出处大多为MI355X，未发现MI350X证据；SKU差异不是已确认原因。同SKU环境差异仍需原始收据才能对齐。

正式745-shape tune仍保存于 `reports/opus_clang23_mixed_retune_20261008/`，配置仍为 `aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv`（OPUS693/ASM34/CKTile10/CK8）。要评估切回旧CK对全部选型的影响，需要把旧CK作为额外明确版本候选进行完整同轮对照，不能用这里8个shape预测并覆盖745行。
