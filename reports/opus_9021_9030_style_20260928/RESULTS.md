# 9021 / 9022 / 9023 / 9024 / 9030 整理记录

五个候选的 pipeline 与 traits 已整理。直接复用已有且适用的 helper；不同的布局保留各自实现，并使用 shape/dim/unfold、前置 gmem/layout、集中 offset 和单行注释。

| 候选 | 复用与保留的区别 |
|---|---|
| 9021 / 9022 | GA/SA/RA/GB/SB/RB 复用 9000；原始字节 SFA reader 复用 9020。保留 A/C pin、原 scale producer 分工、M 尾行处理及 3/2-stage ring。 |
| 9023 / 9024 | GA 经原 lane XOR 后复用 9000，SA/GB/SB/RB 同样复用；XOR RA 用本地 shape/dim。保留 SFA u16 打包、SFB u32 复制、直接 C 输出及 3/4-stage ring。 |
| 9030 | GA/SA/GB/SB/RB 复用 9000，RA/raw SFA reader 复用 9020。保留两个 128-byte SFB 面板、64-bit C 基址、tile-local C bounds 和原 U2 调度。 |

五个候选的 traits 独立列出几何、存储、派生常量和断言；地址计算均在 pipeline。A/B/C/SFA/SFB 基址包含 batch stride，tile row/col 放入对应基址。9021/9022 保留原 pin 容器；9023/9024/9030 使用默认 MMA vtype 和 clear(v_c)。

9023/9024 的 scale layout 表达组内位置，四个 scale offset 表达 K group/panel 推进。SFA 动态地址保留 (panel_begin + group) * stride_sfa 的计算顺序；不在主循环前缓存完整 group 字节地址。最终 source 与被检查的编译快照逐字一致。

## CPU 验证

| ID | AGPR 前→后 | VGPR 前→后 | SGPR 前→后 | LDS bytes | 指令 bytes 前→后 |
|---|---:|---:|---:|---:|---:|
| 9021 | 64→64 | 200→200 | 55→62 | 105504 | 6672→6952 |
| 9022 | 80→80 | 240→244 | 59→66 | 81184 | 8528→9104 |
| 9023 | 0→0 | 144→138 | 75→74 | 80384 | 5880→6044 |
| 9024 | 0→0 | 114→112 | 72→73 | 71936 | 5600→5724 |
| 9030 | 0→0 | 207→205 | 48→50 | 143104 | 7884→8424 |

五个最终 device TU 编译通过，ABI 与 LDS 大小保持，private segment 和 VGPR/SGPR spill 全为 0。9021/9022 的部分 paired LDS writes 被拆为 single writes；机器码不是逐字相同，不能声称所有指令计数完全一致。具体差异保存在 final_device_comparison.json。

最终 host-only 检查使用实际 helper，对照整理前的地址公式，覆盖各线程、RA repeat/chunk/stage、scale pass/pack、K panel 边界与 guard、tile 基址和 C 布局。结果见 [host validation](host_layout_final/validation.json)。此前第一版地址检查也通过，但 9023/9024 的编译 VGPR 增至 233/151；这些版本未作为最终源码交付。独立诊断产物保留，最终使用组内 layout 与动态 K offset 分工。

本轮未运行 GPU 数值测试、benchmark 或重新 tune。寄存器计数只说明编译资源，不代表实测性能。之前完整 305-shape sweep 发生在这些源码整理之前。

## 文件与证据

[分支整体 OPUS tune 集成清单](INTEGRATION_FILES.md)；[最终编译对照](final_device_comparison.json)；[源码范围与快照核对](source_check.json)；[8 个候选的 CPU codegen 依赖核对](integration_codegen.json)。

**9021**

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh)

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh)

**9022**

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh)

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh)

**9023**

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh)

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh)

**9024**

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh)

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh)

**9030**

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh)

[/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh)
