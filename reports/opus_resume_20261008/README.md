# 2026-10-08 MXFP8 B-preshuffle：恢复、定位与优化

状态：`applied_verified_not_committed`。已恢复昨晚 Oct7 selected，按最新总结完成分层 counter 与短/长 K ATT，定位出 9021 短 K 的 scale 请求串行和 LDS 发布屏障等待。新 issue/publish 实现通过 42 个实际赢家 Event、正式 device 身份和八项 API 检查，已应用唯一 9021 header；未提交或推送。

## 最新记录入口

- [Compute 分析第 11 节](../opus_bound_analysis_20261007/COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md#11-2026-10-08先定位限制再选择9021改动)：实际等待位置、长短 K 区别、候选 Event 与 ATT 机制。
- [Memory 分析第 11 节](../opus_bound_analysis_20261007/MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md#11-2026-10-08分层counter与att把限制收窄到具体接口)：L2/TA/TCP/UTCL1/EA/LDS 的证据与范围。
- [最新合并总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)：按工作量/时间 → GL2/SoC → 请求与翻译 → wave 等待 → 实际 PC 判断限制。
- [Oct7 最终应用记录](../opus_bound_analysis_20261007/formal_selected/integration_manifest.json)及[本次恢复清单](recovery_manifest.json)：昨晚基线的来源和文件 SHA。

重要结论集中回填上述已有 Compute/Memory 文档。本目录保存可复算的技术证据；准备、构建、测量等 manifest 的状态保留各自采集时点，由后续记录衔接。

## 恢复起点

HEAD 为 `b152ab834e68f8fde18adbd7b200c587770b9fb5`。恢复基线是 **Oct7 selected working tree**，包括 9021 prologue 与 9042/9053/9054 B-scale 复用；它不同于原 HEAD。Oct7 已验证后应用但未 commit/push，本次没有修改历史采用决定。

| 生产文件，位于 `csrc/opus_gemm/include/gfx950/` | Oct7 selected SHA256 |
| --- | --- |
| `opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh` | `f42695b106bbaaa9564cdd2cdaf128a3218c653dd78e213a26d4361b7d79e836` |
| `opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh` | `34586b3e28cbf8d3b9bf4325d631fd0eb91d56f7ba4a1c6d9f965a041ff7b962` |
| `opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh`，9022 保留原 HEAD | `72db4b2b35ca4c46c83dc913c386cdf3c313090892e8d646026d641b47ae2b8d` |

Oct7 收益继续作为历史结果：9021 的 42/42 受影响实际赢家 median 更快，geomean +3.155%；三个 register alias 的六个实际赢家均 5/5 更快。Oct8 全部新 A/B 使用已采用版本为 baseline，不把两次百分比相加，也不把 Oct7 私有候选原来的 `adopted=false` 字段改成现在时状态。

## 现在定位到了哪里

固定当前 9021 的 `480×7168×384`，以同 224 WG / 896 wave grid 的 `K16384` 为长 K 控制。六次独立 counter pass 与两个 baseline ATT 捕获均有 clean 物理 PCI/owner 证据，实际 device/FUNC/metadata 身份匹配 Oct7 selected。

- 累计 wave dependency wait 64.296%，issue wait 11.595%，LDS issue wait 1.804%；优先查依赖返回和同步，不能把它们当墙钟损失分解。
- 短 K 的一次 CU0 四波 ATT 中，首 MFMA 前平均 4523/9100 shader clocks，约 49.70%。producer 先 SFA load/wait/store，再 SFB load/wait/store，其他三波在发布 barrier 等 636–680 clocks。
- 长 K 启动占比约 3.51%；普通 interior tile 边界中位数 492 clocks，scale panel reload 边界 2014 clocks，二者应分别优化。
- raw GRBM CSV 把八个 XCC 值求和；修正为 JSON raw max 后仍与 timestamp 窗口不一致。因此没有绝对 MFMA 利用率/occupancy，也不套固定因子或把 counter cycles 转 ns。现有数据没有证明 HBM 带宽已饱和。

详见 [clock_audit.json](diagnostics/clock_audit.json)、[layer_analysis.json](diagnostics/layer_analysis.json)、[att_analysis.json](diagnostics/att_analysis.json)。每个 K 的 ATT 仅一次 SE0/CU0 四波捕获，局部时间线用于机制定位；无 profiler Event 决定性能。

## 有证据支持的优化

[scale_issue_publish/](scale_issue_publish/)只改 9021 的 prologue 和 SCALE_PANEL32 refill：先发 SFA/SFB 请求，再将两侧原始 scale 发布到 LDS。数学、原生 E8M0、标准 `(16,16)` B preshuffle、K128 scale、FP32→BF16、guard/tail、矩阵请求、ring 生命周期和全部原硬件同步保持。

[源码审查](scale_issue_publish/source_manifest.json)、[构建](scale_issue_publish/build_manifest.json)、[device 身份](scale_issue_publish/device_audit.json)和[ISA 次序](scale_issue_publish/isa_order_audit.json)已通过。完整且对齐的 SFA 路径在首次 publish 前发出两侧请求；M-tail 原逐 byte load/wait/pack 路径保持串行。VGPR164、LDS105504B、scratch/spill0 保持，SGPR98→99，ISA8588→8660B。

[十项 screen](results/scale_issue_publish_screen_analysis.json)覆盖五个实际赢家和五个机制/边界用例；[独立 confirmation](results/scale_issue_publish_confirmation_analysis.json)精确覆盖全部 42 个实际赢家及两个 boundary。两次均共享 51 地址池、5 轮 AB/BA、signed8 数值/guard/重复性检查。confirmation 的 704 次原地址调用与 22440 个 Event 池输出检查通过。

| 42 个实际赢家，Oct8 候选相对 Oct7 selected | 结果 |
| --- | ---: |
| median 更快 / 更慢 | 41 / 1 |
| 5/5 更快 / 5/5 更慢 | 30 / 0 |
| 几何平均加速 | 1.575621% |
| 各 shape median 时间之和加速，合成统计 | 0.997733% |

唯一负 median 为 `416×7168×16384`，耗时增加 0.034535%，该项 4/5 配对更快。两个非赢家边界的负向已复现：`1×128×128` 耗时增加 1.429% / 2.328%，`15×128×256` 增加 0.369% / 0.454%。按 Oct7 相同标准接受这两个有限代价，保留 global9021；没有声称全支持域零退步，没有加未经测量的新 K 阈值。

[候选 ATT](diagnostics/scale_issue_publish_att_analysis.json)确认实际重叠：producer 的 SFA→SFB issue 间隔 328→120 clocks，SFA issue 到发布 barrier release 968→620；其他三波 barrier 时长降低，consumer scale LDS wait 仍约 37+4。它支持请求/发布次序的解释，不能据单次 CU0 capture 计算全卡加速。

## 已关闭方向及保留范围

[SFA packed scatter](results/sfa_packed_screen_analysis.json)数值通过但十项 Event median 全慢，拒绝采用；代表 `480×7168×384` 慢 8.455%，VGPR164→178。DPP 未产生候选源码或测试结果，不继续这一方向。

9000、9020、共享 helper、所有 Oct7 source/plan/library/result 冻结。9022 仍保留原实现；旧 global9022 prologue、global register default、narrow、finewait、fixed_n32 等拒绝决定保持。可选支持域扫描仍为历史 partial，不补跑全 745。

## 正式集成与复现

`scale_issue_publish/prepare.py` 准备隔离源码，`build.py --dry-run` 查看 exact argv，`audit_device.py` 检查 ELF；`analyze_event.py --input … --claim … --output …` 复算完整 Event 审计。诊断分析脚本在 `diagnostics/`，均只读原始证据并生成各自技术汇总。

[正式 device 身份](formal_selected/identity_audit.json)核对26parent/56entry，仅1个9021 entry改变、其余55个保留。[八项API检查](formal_selected/gpu_smoke_analysis.json)在clean PCI95/HIP7完成128次数值/重复/guard调用、实际加载module SHA一致；API无性能计时，性能A/B仍为PCI65同卡结果。

[源码集成记录](formal_selected/integration_manifest.json)确认新9021 header SHA为`2ff8cf90368c6945d3394fe11075c2336e6f8f99888c7167d3eb550045009bfa`，逐字匹配已测private与正式isolated tree。生产相对HEAD仍是两个文件：9021累计两晚改动、small traits保留Oct7配置；9000/9020/9022及共享helper均不变。正式module在本目录独立`jit_formal_selected`，后续默认JIT重建须使用应用后的源码与记录工具链。

Oct7最终manifest文档SHA保留历史时点，更新后的两份总结及本README/HANDOFF由Oct8最终文档清单绑定。本次恢复、定位和第一项证据支持的优化已完成；长K普通tile边界的matrix async issue/ring bookkeeping留作后续独立实验。没有commit或push。
