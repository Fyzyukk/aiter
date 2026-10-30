# 4wave pin_agpr：171持续落后项的有限 CPU 方案

首轮只设计一个改动：把下个 scale panel 的全局加载提前到前一轮 K128 的 C00 计算结束处。最多生成 Panel32、Panel64 两个实验实例，分别以现有 ID388、ID389 为精确基准；先看4个 shape、10个配置组合，一批筛选。只有正确且有效的新版本进入三批交替复核。当前只完成 CPU 设计和静态检查，未修改源码/CO、未导入 Torch、未执行 GPU、未编新 GPU JIT/CO。

四个主组共108项，其中当前4wave赢家72项，分别是 N7168/K16384：19/28，N7168/K7168：19/28，N7168/K4096：20/26，N6144/K7168：14/26。12个基础4wave CO都已注册且ready，不能把这些落后解释为基础几何漏补。

## 代表 shape 与固定 CSV 基线

|编号|M,N,K|当前ID / µs|固定 CSV ID / µs|当前落后|轮次|
|---|---|---|---|---|---|
|R1|1792,7168,16384|389 / 206.8056|mxfp8 / 179.0289|15.515%|Top4_11_2_fast|
|R2|6144,7168,16384|388 / 606.7227|mxfp8 / 534.8130|13.446%|Top4_11_2_fast|
|R3|32768,7168,7168|390 / 1305.3985|mxfp8 / 1207.2079|8.134%|Top8_101_5_retained|
|R4|20480,7168,4096|394 / 499.1650|mxfp8 / 461.3211|8.203%|Top8_101_5_retained|
|R5|1856,7168,7168|1 / 96.0749|mxfp8 / 88.5061|8.552%|Top8_101_5_retained|
|R6|32768,6144,7168|389 / 1125.8822|mxfp8 / 1044.7218|7.769%|Top8_101_5_retained|

完整当前名称与基线名称：

- R1 (1792,7168,16384)：当前 ID389，opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_k0_reset0_u2_gm0_wide_nt0；固定 gfx950 CSV IDmxfp8，flydsl_mxfp8_8w_256x256x128_F8_F8_B16_4x4。用途：K16384/P64: one refill; 196 workgroups below 256 CUs; separate schedule effects from launch geometry。
- R2 (6144,7168,16384)：当前 ID388，opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp32_k0_reset0_u2_gm0_wide_nt0；固定 gfx950 CSV IDmxfp8，flydsl_mxfp8_8w_256x256x128_F8_F8_B16_2x8。用途：K16384/P32: three refill handoffs; 672 workgroups; same pad_m=false/wide path as proposed P32 instance。
- R3 (32768,7168,7168)：当前 ID390，opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp16_u2；固定 gfx950 CSV IDmxfp8，flydsl_mxfp8_8w_256x256x128_F8_F8_B16_3x4。用途：K7168: current P16 does three refills; test P32 base/new pair with one refill, keeping current winner as reference。
- R4 (20480,7168,4096)：当前 ID394，opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_sp32_u2；固定 gfx950 CSV IDmxfp8，flydsl_mxfp8_8w_256x256x128_F8_F8_B16_4x8。用途：K4096/P32 has no refill; a negative control for added prefetch bookkeeping and branch overhead。
- R5 (1856,7168,7168)：当前 ID1，opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm1_sp32_k0_reset0_u2_gm0_wide_nt0；固定 gfx950 CSV IDmxfp8，flydsl_mxfp8_8w_256x256x128_F8_F8_B16_4x4。用途：Partial M tile (1856): keep existing padded-P32/P64 controls; new unpadded instances are illegal here。
- R6 (32768,6144,7168)：当前 ID389，opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_k0_reset0_u2_gm0_wide_nt0；固定 gfx950 CSV IDmxfp8，flydsl_mxfp8_8w_256x256x128_F8_F8_B16_1x4。用途：N6144/K7168 expansion case; current P64 has no refill, P32 pair would isolate one refill。

R1–R4进入首轮。R5只有partial-M现有候选对照，不能启动这两个pad_m=false新实例；R6作为后续N6144扩组。当前CSV混有101/5与11/2计时，表中落后幅度不能直接证明当前硬件瓶颈。

## 源码中的准确边界

- [traits](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L30) 固定4个Wave64、BM256/BN256/BK128、2×2 wave layout、E_M/E_N=4；LOOP_UNROLL=2。
- [matrix LDS与scale LDS](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L496) 为两槽A/B矩阵加SFA/SFB panel；矩阵槽也被输出epilogue复用。
- [load_sfa_panel](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L638) 先把Panel/16个16-byte向量放到raw数组；[publish_sfa_panel](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L658) 做4×4字节转置再写LDS。四个K32 lane group共享同一个K128 scale，不能改变其字节排列。
- [refill_scale_panel](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L704) 在next_tile的panel边界：lgkmcnt(0)→barrier→SFA/SFB global load→vmcnt(0)→publish→lgkmcnt(0)→barrier。两个barrier保护旧panel读取与新panel可见性。
- [mainloop](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L779) 每轮执行refill(k_tile+1)，矩阵future_tile=k_tile+2。C00前半在[793](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L793)有vmcnt(0)/lgkmcnt(0)，797有barrier，随后才重写旧矩阵槽。不要直接放宽这些wait。
- [C00结束](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L829) 是新全局scale issue点；790的next-SFA LDS读已经发出，844的next-SFB LDS读尚会使用旧LDS，因此这里只写raw寄存器。
- [tail](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L998) 仍有refill(tile+1)；[final K](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L1147)没有下一panel。early issue必须严格R<loops。

## 静态资源能证明什么

|Panel|LDS bytes|是否超过80KiB|仅靠LDS能否并驻2WG|
|---|---:|---|---|
|16|139392|是|不能|
|32|143616|是|不能|
|64|152064|是|不能|

这三个值由[panel traits](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L119)计算，并逐一用现有ELF notes核对。减少panel会减少scale LDS，但矩阵两槽本身135168 bytes，仍阻止仅靠LDS容量获得2WG/CU。

现代ID389：VGPR456、SGPR104、AGPR212、LDS152064、VGPR/SGPR spill均0、private segment0。pad=true的P32/P64（ID1/2及394/395）报告3/4个SGPR spill；P32 ISA同时出现3次v_writelane和3次v_readlane。private segment为0，所以不能把它写成HBM scratch瓶颈。更长raw生命周期是否增加spill必须留到隔离构建后的metadata/ISA检查。

ID389整段静态ISA有320条scaled MFMA、68条waitcnt、16条barrier；包含prologue、展开主循环和tail。它证明指令存在，不代表每次K循环次数、stall周期、实际占用率或cache命中率。JSON保存每个现有CO的hash、ELF资源和opcode统计。

## 首轮唯一新策略：scale global load提前一轮

状态：design_only；没有分配新kernel ID，没有可运行的新CO。两个实例保持pad_m=false、wide_output=true、GroupM0、Unroll2，只分别继承ID388/P32与ID389/P64。

1. 当前轮k_tile=R-2且R为panel边界、R<loops时，在C00结束处调用现有load_sfa_panel(R)，复用之前publish后已不再使用的raw数组。
2. 同时用原来的wave/lane guard读取SFB到单独pending word；保存pending panel begin/valid。此处不写任何scale LDS。
3. 下一轮refill(R)使用pending raw。保留原先两个LDS barrier、VM wait和publish顺序，保留边界fallback及tail路径。矩阵请求顺序、MFMA顺序、AGPR pin范围、输出路径都不变。
4. 全部有效边界R恰好在R-2轮提前发出一次；R==loops永不发出。已用纯Python验证Panel32/64、loops1–200的覆盖和tail边界。

这不是已证实的stall修复。提前scale请求可能与matrix请求竞争，还使P32的8个raw SFA dword、P64的16个raw dword与pending SFB/state活得更久，可能增加寄存器成本。首轮要回答的只有“同一配置把global issue提前，是否有稳定收益”。

首轮配置清单：

|Shape|保留当前winner|精确matched base|新设计|
|---|---|---|---|
|1792,7168,16384|389|389|scale_pf1_after_c00_P64|
|6144,7168,16384|388|388|scale_pf1_after_c00_P32|
|32768,7168,7168|390|388|scale_pf1_after_c00_P32|
|20480,7168,4096|394|388|scale_pf1_after_c00_P32|

去重后10个shape/config组合，首轮一批10条结果；101计时+5预热时，常规Opus调用1060次，另计shared rotation探测。只有正确且有效的新版本与其matched base进入三批交替复核，完整矩阵三批上限为30条结果、3180次常规调用。仍直接读取固定CSV，不重新运行FlyDSL，不展开171全量搜索。每个新实例只与其同shape matched base比较；当前winner作为运行条件与最终收益参考，不能混作单因素基准。

## 后置已有GroupM对照

[GroupM映射](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh#L403)只改变输出tile的遍历次序，不增加workgroup数量。R1只有196个工作组，所以GroupM不能消除其静态grid不足；这也不能单独解释15.515%差距。

磁盘有legacy gm0/gm4/gm8 P64/pad=false/wide/unroll2镜像，三者VGPR456、SGPR68、AGPR212、LDS152064，scaled MFMA/wait/barrier静态总数也相同。当前注册JSON的12个pin_agpr却全部GroupM0，legacy镜像没有当前注册ID和新receipt；不放入首轮精确清单。

后续只能比较匹配legacy gm0/4/8，先核96-byte kernarg ABI、image/symbol/loader与资源边界，并在隔离registry/launcher中加入。现代389的SGPR为104，不应与legacy gm4直接比较后声称GroupM单因素收益。镜像与符号的存在也不等于它们已经包含当前全部源码更新。

## 后置结构设计及检查

若单个schedule策略没有稳定收益，再单独讨论K128矩阵单槽加P32：理论LDS135168/2+8192+256=76032 bytes，LDS容量条件可容两份；现有VGPR/AGPR和pin限制仍可能只允许1WG，不能承诺占用率提高。单槽需重做publish与operand-lifetime证明，不能把stage^1简单改为0。BM128+当前两槽/P64约110080 bytes，仍大于80KiB，其可能价值是更多grid tile和更少累加器，属于另一个结构变量。

当前完成的CPU合法性检查：6个代表均来自真实171 CSV且与当前inventory完整ID/name/traits一致，基线严格gfx950/256/128x128；N可整除256、K可整除128，pad=false代表M整除256；input/C-tile窗口在signed31范围内；对所有代表枚举GroupM0/4/8/16映射均为双射。没有据此声称GPU正确。

未来新CO的门禁：原96-byte ABI、raw E8M0布局和K128广播不变；A/B边界与64-bit C base/tile-window不变；两个refill barrier和VM/LDS依赖不变；[pin验证](../../../../csrc/opus_gemm/gen_co/build_co.py#L313)、LDS上限与[spill/private区分](../../../../csrc/opus_gemm/gen_co/build_co.py#L353)逐项审查。之后才以NaN初始化完整BF16输出、原误差阈值0.05、同seed/轮转/计时设置做有限测量。未获稳定matched-base收益或出现负对照退化，就不扩组。

本方案只生成两个计划文件；当前源码、CO、第四步结果及14项冻结包保持原样。
