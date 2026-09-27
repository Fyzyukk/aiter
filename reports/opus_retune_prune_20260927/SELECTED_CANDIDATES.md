# 本轮保留的16个OPUS候选

本次完整295项、3轮重新调优，按每个shape整体最快实现实际使用的OPUS ID保留，最终为**原5个生产注册候选 + 11个runtime-K候选**。16个候选在整体选择中覆盖291/295项：原5覆盖201项，runtime-K候选覆盖90项。

“整体中选”是与全部有效后端比较后成为最终选择的shape数；“最快OPUS”只在OPUS内部比较。保留条件为整体中选次数大于0。完整数据见[selected_candidates.csv](selected_candidates.csv)、[selected_candidates.json](selected_candidates.json)，实测来源为[candidate_usage.csv](full295_r3/results/candidate_usage.csv)。

所有候选的A/B为`fp8_t`，输出为`bf16_t`，累加FP32，scale为原生E8M0；`K`均从运行时参数读取，无fixed-K实例。tile格式为M×N×K，wave为Wave64。

| ID | 库/注册 | Tile | Wave | runtime K | 整体中选 | 最快OPUS | 实现 |
|---:|---|---|---:|---|---:|---:|---|
| 9000 | 原生产注册 | 256x256x128 | 4 | ≥128，步长128 | 129 | 129 | 原始256×256分半流水，4 wave，scale panel64，直接BF16输出。 |
| 9010 | 原生产注册 | 128x128x128 | 4 | ≥128，步长128 | 9 | 9 | 3层矩阵LDS，K+2预取，scale panel64，直接BF16输出。 |
| 9011 | 原生产注册 | 64x128x128 | 4 | ≥128，步长128 | 24 | 24 | 3层矩阵LDS，K+2预取，scale panel64，直接BF16输出。 |
| 9012 | 原生产注册 | 64x64x128 | 4 | ≥128，步长128 | 7 | 7 | 4层矩阵LDS，K+3预取，scale panel64，直接BF16输出。 |
| 9020 | 原生产注册 | 256x256x128 | 4 | ≥128，步长128 | 32 | 32 | 复用9000主体，traits继承9000，仅PAD_M=true。 |
| 13163 | long_epilogue_sync | 192x256x128 | 8 | ≥128，步长128 | 1 | 1 | 192×256，8 wave，scale panel128；去重复pre-output同步；LDS BF16输出vec8/cache2。 |
| 20000 | long_runtime | 192x256x128 | 8 | 128–16384，步长128 | 9 | 9 | 192×256，8 wave，runtime U2，scale容量128组；N-first，非融合vec8/cache2输出。 |
| 20010 | short_runtime | 128x128x128 | 4 | 128–1536，步长128 | 7 | 7 | 128×128，4 wave，runtime U2，scale容量12组；LDS BF16输出vec8/cache2。 |
| 20011 | short_runtime | 160x128x128 | 4 | 128–1536，步长128 | 14 | 14 | 160×128，4 wave，runtime U2，scale容量12组；LDS BF16输出vec8/cache2。 |
| 20020 | n224_runtime | 192x224x128 | 8 | 128–1536，步长128 | 1 | 1 | 192×224，8 wave；最终MFMA与BF16输出融合，vec4/cache0，N-first。 |
| 20100 | long_runtime_grid | 192x256x128 | 8 | 128–16384，步长128 | 32 | 32 | 20000主体；N≤2048且M≥4096时group-M4，否则N-first。 |
| 20124 | n224_runtime_loop | 192x224x128 | 8 | 128–1536，步长128 | 1 | 2 | 192×224，8 wave；统一U2循环与drain，融合输出，vec4/cache0，N-first。 |
| 20125 | short_runtime_unified | 192x256x128 | 8 | 128–1536，步长128 | 2 | 2 | 192×256，8 wave；统一U2循环与drain，非融合vec8；M≥4096时group-M4/cache2，否则N-first/cache0。 |
| 20126 | short_runtime_unified | 192x256x128 | 8 | 128–1536，步长128 | 2 | 4 | 192×256，8 wave；统一U2循环与drain，融合vec4；M≥4096时group-M4/cache2，否则N-first/cache0。 |
| 20128 | long_runtime_fused | 192x256x128 | 8 | 128–16384，步长128 | 19 | 20 | 192×256，8 wave；统一U2循环，最终MFMA与BF16输出融合；N-first、vec8/cache2。 |
| 20131 | short_runtime_group4_cache2 | 192x256x128 | 8 | 128–1536，步长128 | 2 | 2 | 192×256，8 wave；统一U2循环与drain，融合vec4，始终group-M4/cache2。 |

原5及13163没有独立的固定K上限，仍须满足各入口的shape对齐与有符号32位字节寻址限制。其余K范围是对应入口明确接受的范围。

下表列出实际C++入口及Traits类型。普通global入口使用固定Traits；带尖括号的入口按所列模板参数实例化。`TileM`、`OutputVector`、`Waves/ScalePanel`是模板轴，K不是模板轴。

| ID | 实际kernel入口/模板实例 | Traits类型 | Traits继承或模板参数 | 当前源码 |
|---:|---|---|---|---|
| 9000 | `gemm_a8w8_mxfp8_scale_kernel<opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950>` | `opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950` | `Traits type only; no fixed K` | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh) / [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh) |
| 9010 | `gemm_a8w8_mxfp8_bpreshuffle_128x128_kernel<opus_gemm_mxscale_bpreshuffle_128x128_traits_gfx950>` | `opus_gemm_mxscale_bpreshuffle_128x128_traits_gfx950` | `Traits type only; no fixed K` | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh) / [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh) |
| 9011 | `gemm_a8w8_mxfp8_bpreshuffle_64x128_kernel<opus_gemm_mxscale_bpreshuffle_64x128_traits_gfx950>` | `opus_gemm_mxscale_bpreshuffle_64x128_traits_gfx950` | `Traits type only; no fixed K` | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh) / [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh) |
| 9012 | `gemm_a8w8_mxfp8_bpreshuffle_64x64_kernel<opus_gemm_mxscale_bpreshuffle_64x64_traits_gfx950>` | `opus_gemm_mxscale_bpreshuffle_64x64_traits_gfx950` | `Traits type only; no fixed K` | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh) / [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh) |
| 9020 | `gemm_a8w8_mxfp8_scale_kernel<opus_gemm_mxscale_bpreshuffle_padded_m_traits_gfx950>` | `opus_gemm_mxscale_bpreshuffle_padded_m_traits_gfx950` | `Traits type only; no fixed K` | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh) / [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh) |
| 13163 | `gemm_a8w8_mxfp8_bpreshuffle_generic_no_repeat_sync_kernel<opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128>>` | `opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128>` | `Traits<Waves, ScalePanel>; no fixed K` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_epilogue_sync/pipeline_generic.cuh) / [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh) |
| 20000 | `gemm_a8w8_mxfp8_bpreshuffle_long_runtime_kernel` | `opus_gemm_long_runtime_traits` | `opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128>` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/pipeline_runtime.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime/traits_runtime.cuh) |
| 20010 | `short_runtime_kernel<short_runtime_traits<128>>` | `short_runtime_traits<128>` | `TileM=128` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/pipeline_short_runtime.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/traits_short_runtime.cuh) |
| 20011 | `short_runtime_kernel<short_runtime_traits<160>>` | `short_runtime_traits<160>` | `TileM=160` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/pipeline_short_runtime.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime/traits_short_runtime.cuh) |
| 20020 | `runtime_n224_kernel` | `runtime_n224_traits` | `none; concrete global entry` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/pipeline.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime/traits.cuh) |
| 20100 | `gemm_a8w8_mxfp8_bpreshuffle_long_runtime_grid_kernel` | `opus_gemm_long_runtime_grid_traits` | `opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128>` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/pipeline_runtime.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_grid/traits_runtime.cuh) |
| 20124 | `runtime_n224_loop_kernel` | `runtime_n224_loop_traits` | `none; concrete global entry` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/pipeline.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/n224_runtime_loop/traits.cuh) |
| 20125 | `short_runtime_unified_kernel<8>` | `short_runtime_unified_traits` | `OutputVector=8; nonfused` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/pipeline.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/traits.cuh) |
| 20126 | `short_runtime_unified_kernel<4>` | `short_runtime_unified_traits` | `OutputVector=4; fused` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/pipeline.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_unified/traits.cuh) |
| 20128 | `gemm_a8w8_mxfp8_bpreshuffle_long_runtime_fused_kernel` | `opus_gemm_long_runtime_fused_traits` | `opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128>` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/pipeline_runtime.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/long_runtime_fused/traits_runtime.cuh) |
| 20131 | `short_runtime_group4_cache2_kernel` | `short_runtime_group4_cache2_traits` | `none; concrete global entry` | [pipeline](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/pipeline.cuh) / [traits](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/short_runtime_group4_cache2/traits.cuh) |

11个runtime-K候选保存在[csrc/opus_gemm/mxfp8_bpreshuffle_retained](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/README.md)，共9个独立库。每个库通过自己的`launch` C ABI使用私有候选ID；9000、9010、9011、9012、9020继续使用生产注册。私有ID没有加入生产全局registry，20000段仍属于gfx1250生产ID空间。

包内现有[device_audit.json](../../csrc/opus_gemm/mxfp8_bpreshuffle_retained/device_audit.json)记录`passed`：实际只有11个中选device入口，各入口的instruction bytes均与本次实测对应kernel一致。此结论针对GPU kernel机器码；构建后的完整共享库哈希可因launcher裁剪而变化。9000及其共享helpers保留，192×256 traits仍供13163/20000/20100/20128使用。
