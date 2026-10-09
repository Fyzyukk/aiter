# 9023/9024 当前赢家 CPU 审查与有限 ATT 计划

CPU 审查完成；ATT 由 root 执行，本审查没有创建或采用 kernel 候选。基线是 Oct8 正式 selected module，SHA256 `6f6a0117b122f06b4802833c020effb4b24ff9fea56f20c8c16b9ced9787173f`。四个 entry 的当前对象指令、metadata、归一化 descriptor 与 Oct8 audit 和保留 official ELF 一致，反汇编机器字也已重建并核对指令 SHA。只创建本报告与 [narrow_review.json](narrow_review.json)，未改 kernel/shared helper、未 build 或运行 GPU。9000 冻结，完成9021保持。

## 实际赢家与分支

9023 fixed 条件为 `M>=1024 && N<=1024 && K==7168 && ceil(M/64)*(N/128)<=256`。四个赢家 N7168、M544/576 全部是 runtime `<3,32,0>`，K7168 也没有走 fixed。grid_n56 使内部 swizzle 条件全部失败。9024 九个赢家满足 `1024<=M<=2048 && N<=1024 && K==7168`，全部 fixed `<64,7168,4>`、GroupM4 遍历。252/264 WG 之间没有源码分支切换，最后 grouped-M 分别有1/2个 tile。

| Entry | M,N,K | Grid | WG / waves | Tile efficiency | K128 / refills | MFMA / complete wave |
| --- | --- | --- | --- | --- | --- | --- |
| 9023_runtime | 544,7168,7168 | [56, 9, 1] | 504 / 2016 | 0.944444 | 56 / 1 | 448 |
| 9023_runtime | 544,7168,16384 | [56, 9, 1] | 504 / 2016 | 0.944444 | 128 / 3 | 1024 |
| 9023_runtime | 576,7168,7168 | [56, 9, 1] | 504 / 2016 | 1.000000 | 56 / 1 | 448 |
| 9023_runtime | 576,7168,16384 | [56, 9, 1] | 504 / 2016 | 1.000000 | 128 / 3 | 1024 |
| 9024_fixed | 1024,768,7168 | [12, 16, 1] | 192 / 768 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1088,768,7168 | [12, 17, 1] | 204 / 816 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1152,768,7168 | [12, 18, 1] | 216 / 864 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1216,768,7168 | [12, 19, 1] | 228 / 912 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1280,768,7168 | [12, 20, 1] | 240 / 960 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1344,768,7168 | [12, 21, 1] | 252 / 1008 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1408,768,7168 | [12, 22, 1] | 264 / 1056 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1472,768,7168 | [12, 23, 1] | 276 / 1104 | 1.000000 | 56 / 0 | 224 |
| 9024_fixed | 1536,768,7168 | [12, 24, 1] | 288 / 1152 | 1.000000 | 56 / 0 | 224 |

十三项均 split-K1、workspace0。runtime9023 为64×128×128、4wave、3stage、panel32，K7168 为32+24两 panel，K16384 为4个 full panel（refill32/64/96）。fixed9024 为64×64×128、4wave、4stage、panel64；56个 Ktile 小于64，没有 refill。

| Entry | Winners | VGPR / SGPR / scalar spill | LDS B | Instruction B | Static MFMA / write-lane / read-lane |
| --- | --- | --- | --- | --- | --- |
| 9023_runtime | 4 | 232 / 106 / 46 | 78112 | 25976 | 40 / 46 / 166 |
| 9023_fixed | 0 | 176 / 59 / 0 | 105536 | 8340 | 40 / 0 / 0 |
| 9024_runtime | 0 | 152 / 106 / 44 | 69664 | 21736 | 24 / 44 / 132 |
| 9024_fixed | 9 | 110 / 58 / 0 | 71744 | 7276 | 20 / 0 / 0 |

四个 entry 的 AGPR/private/VGPR spill/scratch load/store 都为0。runtime scalar spill 保存在 VGPR lane，不是 HBM spill。9024 runtime 的44 slots/132条 static readlane 不能解释九个 fixed 赢家；fixed lane ops 为0。两个实际 entry 的 LDS 单项在160KiB中可容纳两个 WG（含源码1280B round模型），完整驻留仍受 descriptor 与其他资源影响，这不是动态 occupancy 结论。

## Scale 与相对 PC

SFA producer 已把 low/high 16B vectors 通过八个 v_perm 组合 u16 pair，每 producer lane 两个 DS b128 publish，每 K128 consumer 一个 u16。SFB 每 K128 一个 byte，DS b8 publish、DS u8 consumer。它们与完成9021的 raw-byte layout 不同，保持现有 u16 layout 与共享 helper。

helper 原 guard：`valid_rows>=16 && offset%16==0` 用 vector16，否则每 byte 默认0x7f，只在 `byte<valid_rows` 时 load。所有实际 M/stride_sfa 为16B对齐。M544 最后 WG 有32个 valid rows，low 两个 chunk 全有效、high 两个 chunk 全无效，没有1–15 valid rows 的 partial vector。high 路径不发有效 byte VMEM，但仍可能执行 default、pack、readlane/EXEC 控制。M576 全满；需要实际 tail WG capture 才可量化尾路径。

SFA group=`thread_id/2`：panel32 full 使用前64threads（wave0）、partial24 使用前48threads；fixed56 使用前112threads（wave0 加 wave1 前48lanes）。SFB 对相应 panel 用前32/24/56threads。ATT 需区分 producer/consumer wave 与 EXEC 路径。

以下 PC 均为 kernel-relative，decoder 当前 exact symbol/code load address 需独立减去。保留 disassembly base runtime9023=0x1f00、fixed9024=0x7400 仅用于 CPU 核对，不能直接迁移 private absolute PC。

| Phase | 9023 runtime relative PC | 9024 fixed relative PC |
| --- | --- | --- |
| Startup SMEM wait | 0x20 | 0x18 |
| +0x40 scale kernarg issue → wait | 0x2e0 → 0x3a4 | 0x33c → 0x364 |
| SFA low/high vector16 | 0xba4 / 0x12fc | 0xbc8 / 0x13ac |
| SFA wait → u16 LDS publish | 0x1310 → 0x1354,0x1388 | 0x13c0 → 0x13fc,0x1424 |
| SFB issue → wait → publish | 0x13f0 → 0x1400 → 0x1404 | 0x1494 → 0x14a4 → 0x14a8 |
| Final panel wait → barrier | 0x141c → 0x1420 | 0x14d8 → 0x14dc |
| Initial SFA/SFB LDS read | 0x1434 / 0x1444 | 0x14f4 / 0x1554 |
| Initial operand wait / first static MFMA site | 0x14e8 / 0x1b28 | 0x155c / 0x162c |
| First steady wait → barrier | Full map in JSON | 0x16a4(vmcnt8,LGKM0) → 0x16b8 |
| First next SFA/SFB LDS read | Full map in JSON | 0x169c / 0x1700 |

runtime9023 三 tile 静态回跳 `0x1b10..0x4c9c/0x4ca4` 含24 MFMA/120 readlane，余数 `0x507c..0x6174/0x617c` 含8 MFMA/40 readlane。区间包含条件 refill/byte/default 路径；例如0x1c64/0x1c6c恢复 v83 lanes3/4 到 s46/s47 用于 EXEC mask，然后发有界 SFA byte 请求。full vector 路径可跳过该 readlane，不能把120乘迭代次数当动态 spill 成本。remainder 控制恢复0x4d38/0x4d48/0x4d50，output 恢复0x6340/0x6348/0x640c需单列。

refill publish barrier 为0x2a94/0x3b0c/0x4b88/0x6024。full path 中 SFB 在 SFA wait/pack 后请求，是否延长 publish barrier 待动态证据。fixed9024 无 refill，关注 startup SMEM、fourstage ring 的 vmcnt8/4/0 drain、u16/u8就绪、barrier与4 MFMA/K128排布。静态 byte/vector/DS 数量包含 conditional 与 duplicated 路径，不是动态请求或 HBM 流量。

## 已拒绝 unroll1

历史 runtime-only unroll1 使9023 VGPR232→121、spill46→21、lane46/166→21/21、指令25976→10928B；9024runtime VGPR152→87、spill44→22、lane44/132→22/22、指令21736→10776B。fixed 完全未变，SGPR106、LDS/private/AGPR不变。

| Parent / shape | Baseline → candidate μs | Event median ratio change | Paired median change | Faster rounds |
| --- | --- | --- | --- | --- |
| 9023 / 544,7168,7168 | 47.5369 → 47.7228 | -0.390% | -0.387% | 1/5 |
| 9023 / 544,7168,16384 | 99.3186 → 99.0299 | +0.291% | +0.941% | 4/5 |
| 9023 / 576,7168,7168 | 48.2577 → 48.2240 | +0.070% | -0.374% | 2/5 |
| 9023 / 576,7168,16384 | 99.0574 → 100.2205 | -1.161% | -1.157% | 1/5 |
| 9024 / 608,768,7168 | 20.9100 → 20.8872 | +0.109% | +0.109% | 4/5 |
| 9024 / 1536,768,16384 | 62.5260 → 63.3566 | -1.311% | -1.393% | 0/5 |

Oct7 PCI85单 clean process 窗口、共享51地址池、五轮AB/BA HIPgraph Event，数值/guard通过。52机制与718 runtime支持域未完成，没有 ATT/counter。实际9023 near-zero/mixed且有约1.16%退步；9024runtime maxK五轮全慢约1.31%。576×7168×7168两统计符号不同，不能挑正口径。保持全局拒绝，不为旧unroll1追加寻找正值；profiler活动时间不能替代完整调用 Event。

## 有限 ATT 代表

| Priority | Entry / shape | Purpose | MFMA per complete wave / four waves |
| --- | --- | --- | --- |
| 1 (primary) | 9023 / 544,7168,16384 | LongK three refills, tail whole invalid high vectors | 1024 / 4096 |
| 2 (primary) | 9023 / 576,7168,7168 | FullM aligned vector, one refill | 448 / 1792 |
| 3 (primary) | 9024 / 1344,768,7168 | 252WG fixed GroupM4 tail1Mtile | 224 / 896 |
| 4 (primary) | 9024 / 1408,768,7168 | 264WG same fixed branch tail2Mtiles | 224 / 896 |
| 5 (optional) | 9024 / 1024,768,7168 | 192WG lower grid, full GroupM4 | 224 / 896 |
| 6 (conditional_second_round) | 9023 / 544,7168,7168 | SameK tail/full match if unresolved | 448 / 1792 |
| 7 (conditional_second_round) | 9023 / 576,7168,16384 | Same longK tail/full match if unresolved | 1024 / 4096 |

首轮四 primary，192WG fixed可选。其他两个9023仅在同 K tail/full不确定性需消除时第二轮追加。252/264是同 fixed/group4分支与不同尾 group形状，不能归因于256源码边界。

Exact symbols：

```text
_Z41gemm_a8w8_mxfp8_scale_4wave_64x128_kernelI61opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950ILi3ELi32ELi0EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
_Z40gemm_a8w8_mxfp8_scale_4wave_64x64_kernelI60opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950ILi64ELi7168ELi4EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
```

SDK 实际可能过滤 demangled 函数名；JSON保留两种名称，最终核对 captured code exact symbol。核对 codeID/load address/指令 SHA、grid与每完整 wave MFMA。SE0/CU0有限波可能不含 M544最后WG或fixed尾group，absence不是尾路径验证。报告 complete/truncated、可取得的WGID，按 dynamic MFMA ordinal拆 ordinary tile、refill32/64/96、end drain，数实际readlane与EXEC/address/control依赖。

先拆 first MFMA前 SMEM、grid/group算术、matrix/scale issue、publish barrier、LDS与runtime setup。fixed9024单独核对普通fourstage schedule与drain。gfx9 event=`[time,type,stall,duration,code_line]`；duration已含stall，successful issue=`time+stall`，单位shader clocks，clip4clock量化尾差。wait queue membership不是单 request返回时间，LGKM可含SMEM/DS，VMEM可含matrix/scales，empty wait可能省略。不能从未校准GRBM转ns、绝对MFMA利用或occupancy。

若VMEM wait主导，按最新trans_github层次 dispatch CPC/SPI→SQ/TA→TCP/UTCL1→GL2/SoC→return→compute补 matched counter；LDS是local branch，SMEM startup另列。高VMEM wait、逻辑bytes不能证明HBM饱和。CU0 start/end/reuse只给局部dispatch线索，全256CU fill/SPI资源/grid tail需matched SQ/SPI或更广coverage。

候选仅在动态证据成立后准备：9023runtime/remainder uniform control或guarded default path隔离、local低高/SFB issue再u16 pack/publish顺序，或fixed9024 operand/barrier schedule。保留原guards/EXEC/stages/panel/output/grid与shared helper。root对所有实际affected winners和callable fallback/dispatch guards使用current official共享地址AB/BA Event与数值/guard验证；减少lane或ATT gap、profiler时间本身不能采用。

## 输入证据

- [当前inventory](inventory.json)、[Oct8 identity audit](../opus_resume_20261008/formal_selected/identity_audit.json)。
- [Compute文档](../opus_bound_analysis_20261007/COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md) §§9.2/9.2.1与§11；[Memory文档](../opus_bound_analysis_20261007/MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md)。
- [shape inventory](../opus_bound_analysis_20261007/shape_inventory.json)、[旧narrow Event](../opus_bound_analysis_20261007/results/narrow_runtime_window_analysis.json)、[旧narrow ISA](../opus_bound_analysis_20261007/narrow_unroll/isa_comparison.json)。
- [保留9023 ISA](../opus_bound_analysis_20261007/official_compute_metadata/kid9023.s)、[保留9024 ISA](../opus_bound_analysis_20261007/official_compute_metadata/kid9024.s)，本轮重建机器字并匹配当前指令SHA。
- 概念来源 `/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md`；完整symbol、resources、输入SHA在JSON。
