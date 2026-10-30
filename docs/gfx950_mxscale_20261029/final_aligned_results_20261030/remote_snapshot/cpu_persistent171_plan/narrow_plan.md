# 窄 N：171 持续落败形状的 CPU 静态优化计划

先补现有精确配置的覆盖，再决定是否改排程。首轮限定四个真实持续落败形状、16 对 shape/kid，其中 **5 对补 Top8 覆盖、4 对当前锚点、7 对受控复核**。本次只生成计划：没有 GPU 访问、Torch 导入、kernel 执行、新 JIT 编译、源码/CO 修改或 FlyDSL 重测；冻结的 14 形状包保持原样。

171 的定义是历史最佳 Opus 与当前最佳 Opus 都没有严格快于同一份 gfx950 / 256 CU / 128x128 保存 CSV。N=768/1536/2048 共有32个（9/15/8），以下只选四个代表；寄存器、带宽、占用率和流水线瓶颈均为待验证假设。

| M,N,K | 保存 CSV µs | 历史最佳 Opus µs /差距 | 当前最佳 µs /差距 | 当前 kid /真实几何 |
|---|---:|---:|---:|---|
| 1152,768,7168 | 16.5620 | 18.3744 / +10.943% | 18.1938 / +9.853% | 7260 / prefetch_b_vgpr 64x64x256 w1x2 |
| 16384,768,7168 | 86.6241 | 101.5232 / +17.200% | 96.2508 / +11.113% | 20569 / pingpong_lds 256x256x128 w4x2 |
| 16384,1536,7168 | 159.6125 | 172.1219 / +7.837% | 161.8780 / +1.419% | 379 / prefetch_b_vgpr_overlap 128x256x128 w1x4 |
| 10240,2048,7168 | 151.2044 | 160.5505 / +6.181% | 154.4224 / +2.128% | 48094 / prefetch_b_vgpr 128x256x128 w1x4 |

历史最佳几何取 `best_opus_pipeline/tile/waves` 与完整 `opus_kernel`，没有使用 `old_opus_geometry`。CO379 名称的 `256x128x256x128` 首项是 block size；真实 BM/BN/BK 是 **128/256/128**。四个当前锚点均来自保留的 Top8 / 101 次 / 5 warmup 轮次。完整 baseline 名称、splitK、历史名称与当前 traits 都在 JSON，精确运行身份也写入 CSV。

保存 CSV 的真实基线身份如下，四项 splitK 均为1，数值只读保存记录：

- 1152×768×7168：`kernelId=bmm`，`flydsl_bmm_mxfp8_mfma_t48x64x256_w1x2_nb5_sk1_bd2_xcd`。
- 16384×768×7168：`kernelId=mxfp8`，`flydsl_mxfp8_8w_256x256x128_F8_F8_B16_3x4`。
- 16384×1536×7168：`kernelId=mxfp8`，`flydsl_mxfp8_8w_256x256x128_F8_F8_B16_2x4`。
- 10240×2048×7168：`kernelId=mxfp8`，`flydsl_mxfp8_8w_256x256x128_F8_F8_B16_3x4`。

## 覆盖结论与首轮已有配置

| shape M,N,K | 合法 / Top8+原种子 | 当前几何合法 / 已选 | 结论 |
|---|---:|---:|---|
| 1152,768,7168 | 5414 / 1983 | 48 / 48 | 当前几何全部入选；复核不能称作补覆盖 |
| 16384,768,7168 | 5420 / 1822 | 63 / 1 | 当前几何只留锚点种子，另62配置未进入Top8 |
| 16384,1536,7168 | 5420 / 1776 | 6 / 6 | 当前几何全部入选；复核不能称作补覆盖 |
| 10240,2048,7168 | 5420 / 1703 | 30 / 30 | 当前几何全部入选；复核不能称作补覆盖 |

Top8 先按每个 pipeline 的几何成本选几何，再加入精确 seed kid；seed 不展开整套几何。[选择源码](../../../../csrc/opus_gemm/opus_gemm_search.py#L325)。`selected` 表示静态重建的选择成员。现存结果只保留这些形状的最佳项，没有全部逐候选结果；因此不把入选成员的时延、正确性或测量稳定性当作已经证明。

| shape | kid | 对照 kid | 参数变化 | 作用 |
|---|---:|---:|---|---|
| 1152,768,7168 | 7260 | — | 当前锚点 | 锚点 |
| 1152,768,7168 | 7356 | 7260 | BN 64→128 | 补覆盖 |
| 1152,768,7168 | 7344 | 7356 | BD 2→1 | 补覆盖 |
| 1152,768,7168 | 41624 | 7260 | PublishM -1→2 | 受控复核 |
| 16384,768,7168 | 20569 | — | 当前锚点 | 锚点 |
| 16384,768,7168 | 20568 | 20569 | FragmentArrays True→False | 补覆盖 |
| 16384,768,7168 | 20563 | 20569 | GroupM 8→4 | 补覆盖 |
| 16384,768,7168 | 48179 | 20569 | BoundedOutput True→False | 补覆盖 |
| 16384,1536,7168 | 379 | — | 当前锚点 | 锚点 |
| 16384,1536,7168 | 187 | 379 | PublishM 0→-1 | 受控复核 |
| 16384,1536,7168 | 380 | 379 | PinC False→True | 受控复核 |
| 16384,1536,7168 | 378 | 380 | PinA True→False | 受控复核 |
| 10240,2048,7168 | 48094 | — | 当前锚点 | 锚点 |
| 10240,2048,7168 | 48095 | 48094 | Output 0→2 | 受控复核 |
| 10240,2048,7168 | 48088 | 48094 | NB 4→3 | 受控复核 |
| 10240,2048,7168 | 48082 | 48088 | BD 2→1 | 受控复核 |

7356/7344 补入 1152 的 BN128 几何（该几何48个合法项、Top8选0）；它将 workgroups216降为108，同时把 NR2增为4，属于明确的队列/几何权衡，静态证据不能预测收益。20568/20563/48179 补入 16384×768 已被单 seed 保留的 m-repeat 几何，只试 fragments、真实 GroupM 与输出 guard 三个独立因素。379/187 与 48094/48088 已选，安排它们是为了受控解释现有排程/输出差异，不能再声称“漏测当前几何”。

所有非锚点按规范化完整 traits 验证只有一个输入参数变化。378相对380只改 PinA，但 **B_AHEAD也因现有公式3→8**；该结果不能只归因于 pinning。缓存组合不做全笛卡尔积；保持已有锚点缓存。12项是已有 JIT registry 项（此处未编译），4项是现存 ready CO，不能把 JIT 注册等同于已有编译结果。

## 有效参数去重

[B-VGPR traits](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh#L71) 给出：`MR=BM/16, NR=BN/(WN*16), F=(BK/128)*NR, B_AHEAD=min(BD*F,B_SCHEDULE_AHEAD), B_SETS=B_AHEAD+1`。WG2或 `(BK/128)*MR>=16` 走 ROTATE_B：NB2上限1，NB3/4上限4；否则受 register room 约束，并在 BK128、NR>=4、PinA、FixedK>=1536 时上限3。这个预算是源码启发式，不是已验证的硬件占用率。

| 真实几何 /条件 | nominal NB,BD →有效 B_AHEAD | 应保留/去重 |
|---|---|---|
| 64×64×256 w1×2 WG1 | 2,1→4；3,1→4；3,2→8；4,2→8 | NB不同仍影响 A stage；不能仅按 B_AHEAD合并 |
| 同几何 WG2 | 2,1→1；3,1→4；3,2→4；4,2→4 | NB3的 ba1/ba2可按有效排程合并；plain48项→42（去6个cache/output副本） |
| 128×256×128 w1×4 WG1 PinA0 | 2,1→4；3,1→4；3,2→8；4,2→8 | 48094=8，B_SETS9，B ring72 dwords/lane |
| 同几何 PinA1、K7168 | NB4 BD1/2均→3 | 379=3，B_SETS4；379与688仍有XCD/contiguous差异，不能按队列合并完整配置 |

7260 的 F4、B_AHEAD8、B_SETS9；7356/7344 的 F8，分别 B_AHEAD16/8。`BD` 是请求的 panel ahead，最终队列按 fragment 数计，名称 `_ba2` 不能直接当成硬件深度。这里只按有效排程去重；不同符号或编译代码布局仍可能带来时延差异。

[m-repeat swizzle](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh#L373) 只有 `grid_n>=256 && M>BM` 分支引用 GroupN。窄N<=2048、合法BN64/128/256时 grid_n最多32，**GroupN4/8/16/32/64对执行的 tile 映射均无影响**；M>=4096走固定 `SWIZZLE_GROUP_M=4`，若显式 GroupM>0则再覆盖映射。因此20563选gm4，不再加gn16/gn32；窄N下gm0与gm4也可能同映射。不同编译符号/代码放置的时延仍可能不同，不把它当作B重用的因果证据。`GROUP_N=128` 是量化分组，和 `SWIZZLE_GROUP_N` 不是同一参数。

## 后置的两项源码排程提案

这些是未来新增配置与源码变化，本次没有实现、编译或运行，也没有分配新kid。先根据上面16对已有项决定是否推进。

1. **B-VGPR按fragment选择 A 发布点。** [当前循环](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh#L664) 在每个B fragment内复用同一B寄存器给全部MR，A发布条件固定 `fragment==0`；overlap只枚举该fragment的PublishM。追加默认0的PublishFragment参数并保留旧ID/名字，先仅比较PublishFragment0/1、固定PublishM。必须同步重推wa/newer_b/pending_a计数并静态检查启动、稳态、尾段的VMEM事件与A LDS slot所有权；不能只移动issue_a。这个方向检验A发布与B请求的相对顺序，不预设现有wait是瓶颈。

2. **m-repeat可选 A fragment ring2/3。** [traits](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh#L41) 当前A_RING=min(E_M,3)，[主体](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh#L642) 在m=E_M-A_RING释放当前LDS给K+2。256×256 WM4得到E_M4/E_N8，ring3用24 A dwords/lane，在m1发布、余24 MFMA；ring2用16 dwords，在m2发布、余16 MFMA。追加默认现行为的参数，保留B8 fragments驻留并跨全部4个M repeat复用，加载/覆盖只发生在旧fragment最后消费者之后；更小寄存器活跃集可能被更少重叠抵消。当前traits的PUBLISH_M没有在m-repeat主体引用，直接扫这个值不改变源定义的发布点。FixedK16384特殊延迟wait只适用于明确的E_M4/E_N8数组条件，不能直接推广到K7168。

## 保护与 CPU 检查

窄N已有领先保护形状1216×768×7168：保存CSV18.5621µs，当前kid41804=17.8714µs（-3.721%）。它只用于未来源码变化的保护，未添加到首轮16对。

40960×65536×1536列为CPU地址guard：当前kid48197合法，C=5368709120bytes（5GiB），A=62914560、B=100663296，单C tile跨度33554432bytes；[int64 C基址](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh#L418)和[registry span guard](../../../../csrc/opus_gemm/opus_gemm_common.py#L3253)需保持。它在171之外，不增加首轮大C性能测试。171输入A最大1GiB、B最大147MiB、C最大896MiB，4GB不是这171的已知根因。

静态验证通过：四形状均属于171；16对kid/name/shape合法且唯一；12个非锚点均单输入参数变化；5个补覆盖项确实未进入重建Top8；4个CO均ready并保存原资源元数据；Torch等GPU模块未加载；源码、CO清单/构建信息及冻结14包的文件sha256未变化。逐候选时延缺失明确记录为未知；旧日志中数字kid187对应其他完整名称，拒绝把这个ID碰撞当性能证据。

JSON字段 `representatives`、`variants`、`source_refs` 可供总计划整合；`narrow_existing_candidates.csv` 保存精确16对与比较关系。
