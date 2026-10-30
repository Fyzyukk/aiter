# Interleave：小 M、N7168、K16384 的有限优化方案

本文件是 CPU 设计记录。未修改 kernel、未导入 Torch、未执行 GPU 测量、未编译新 GPU JIT。三个新版本均尚未实现，不填写虚构的新 kernel ID；第一轮只实现并比较 I1、I3，I2 留作有条件的第三个版本。14 项回退的冻结包保持原样。

## 目标与已有证据

171 个历史和本轮均落后的 shape 中，interleave 当前最佳有 15 项，其中 8 项集中于 `N=7168、K=16384、M=1216–1664`，最佳几何都是 `192×256×256 / wave2×4`。这 8 项本轮慢 CSV 约 11–15%，历史也未胜同一固定 CSV，因此值得研究现有 kernel。这个结果没有证明 VMEM、LDS、寄存器或驻留 wave 中的哪一项是硬件瓶颈。

| 类型 | M,N,K | 本轮最佳 kid | 本轮 Opus μs | 固定 CSV μs | 状态 |
|---|---|---:|---:|---:|---|
| 代表，尾 tile 有效 64 行 | 1216,7168,16384 | 47680 | 175.7013 | 152.7889 | 慢 15.00% |
| 代表，M tile 完整 | 1536,7168,16384 | 47680 | 186.4453 | 164.8976 | 慢 13.07% |
| 代表，尾 tile 有效 128 行 | 1664,7168,16384 | 47669 | 193.2055 | 173.8363 | 慢 11.14% |
| 保护，当前最佳为其他 pipeline | 1088,7168,16384 | 692 | 123.3045 | 139.7252 | 领先 11.75% |
| 保护，当前最佳为其他 pipeline | 1152,7168,16384 | 691 | 122.3527 | 144.4897 | 领先 15.32% |

47680 的完整名称为 `opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_wave_interleave_192x256x256_w2x4_k0_gm8_xcd1_cA1cB17_nt0`；47669 为 `opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_wave_interleave_192x256x256_w2x4_k0_gm0_xcd0_cA17cB1_nt0`。保护项当前完整名称与固定基线名称见 JSON。当前 421 项没有 interleave 作为总体最佳且领先 CSV 的项，因此保护项选相邻 M 的既有 overlap 赢家，不能把它们表述为 interleave 赢家。

前三个代表来自 Top4、11 次计时的快速轮次。未来获准使用 GPU 后，必须重新计时原版作为同次对照，不把新版本直接与短计时旧值相减来归因。

现有 Top4 已对小 M 代表覆盖 interleave 的 BM128/192 × WaveN2/4 四种几何，共 120 个合法配置；GroupM0/4/8、XCD、cache 和合法 FixedK 都由既有注册展开。历史精确赢家 seed 也已覆盖。这里的新工作是源码变体，而不是再次登记这些相同参数。Top4 评分只按 M/N padding、WG/CU 和 tile 尺寸，不按 K、wave、LDS 或寄存器评分；这是一条筛选局限，不是对具体瓶颈的诊断。

## 源码中的计算和数据依赖

所有行号对应当前文件，指纹保存在 JSON。

- [traits 的 tile、wave 与 LDS 槽](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh#L10)：第 10–40 行定义 `T_M=2、T_N=4、BM192、BN256、K_STAGE128、HALF_M96、E_M3、E_N2`；第 54–71 行定义两个 stage、每 stage 两个 A/B half 槽及 scale 区。
- [pipeline](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh#L24)：24–60 行是 A global/LDS producer 布局，65–86 行是 consumer fragment 布局；393–410 行把 A global/consumer 坐标与 XOR swizzle 对齐。
- pipeline 426–432 行分配 LDS：四个 A 槽，随后四个 B 槽，再是 SFA/SFB；matrix 区最后复用于输出。434–445 行分配 `v_a[2][E_M]、v_b[2*E_N]、v_c[4*E_M*E_N]` 与 scale 寄存器。
- [global→LDS 和预取](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh#L458)：458–472 行 `load_a/load_b` 为 `buffer async_load<16>`；544–549 行 `issue_tile` 当前顺序为 A0、A1、B0、B1。
- [wait、barrier 和 operand 环](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh#L558)：每个 K128 consume 在 560–562 行先 `vmcnt(0)、lgkmcnt(0)、WG barrier`，563–564 行只在还有下一 tile 时预取另一个 stage；566–577 行读当前 A、scale、全部 B 到寄存器并等待 LDS；579–590 行执行 MFMA，同一 B fragment 复用于两个 M half。606–609 行在 stage0/1 间交替；K16384 有 128 个 K128 consume。
- SFA/SFB：478–535 行 prologue 加载、transpose 和发布 scale；537–542 行按 k 取 scale，596–600 行先发布 scale 后发首个 matrix tile。新版本不删改这些操作。
- [输出的 drain 与复用](../../../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh#L612)：614–616 行必须排空 matrix 读再复用 LDS；621–637 行 LDS staging 和连续 BF16 输出。381–384 行 C 的 int64 基址和局部 SRD 窗口必须保持。

## I1：只禁止加载 A half 内永不消费的填充行，首轮核心

唯一因素是 **A producer 的编译期 half-padding 请求策略**。固定全部 geometry、GroupM/XCD、cache、K、scale、A_SLOT/B_SLOT、数组、预取位置和 wait/barrier。仅在 `load_a` 的每个 step 处增加 wave 一致的条件：

```cpp
constexpr int rows_per_block = T::BLOCK_SIZE / (T::K_STAGE / T::VEC_A);
const int producer_row_begin = wave_id * a_rows_per_wave + step * rows_per_block;
if (producer_row_begin < T::HALF_M) {
    // 原 async_load<16>，所有地址、cache 与 LDS slot 不变。
}
```

这个条件适用于现有 HALF_M 为 64/96/128 且均能被每 wave 8 行整除的几何；实现时增加相应 static_assert。可用独立 compile-time variant tag 标识，原版为默认 mode0，当前 ID/符号不变，新版未来分配新名字和 ID。

BM192/WaveN4 时每 step 64 行，`A_STEPS=2`，每 half 原 producer 加载 128 行，但 consumer 只访问 0–95 行。step1 的 wave4–7 对应 96–127 行，这些行从不被该 half 的 A fragment 消费，可整 wave 跳过请求。保留 16KiB A_SLOT，不在同一版本压缩 LDS，以免混入第二因素。

这区别于实际 M 尾 tile 的越界行：例如 valid_rows=64 时，编译期有效 consumer 中的一些 A 行仍需要由原 SRD 产生并写入零。**不能按 valid_rows 把这些必读零的 LDS 写入省掉**，否则会读到旧 stage 数据。本候选只删 half 内 96–127 的从未消费部分，不删 SFA 中对应真实 consumer 的任何行。

静态名义计数：每个 K128 step 原 A producer 发 32 个 wave 级 buffer 请求，新版发 24 个，A 的名义 lane payload 从 32KiB 降为 24KiB，减少 25%。B 保持 32 个 wave 请求、32KiB 名义 lane payload，A+B 从 64KiB 降为 56KiB，减少 **12.5%**。这些是源码布局和全 lane 请求的计数，不是实际 HBM 字节、cache miss 或预计加速；尾部 SRD 掩码、合并和缓存复用会改变实际外存流量。LDS 仍为 **163840 字节**，不预测驻留 WG 数增加。

CPU 可检查的不变量：producer 的有效 row/vector 覆盖与全部 consumer 地址一致；A XOR swizzle 在每行 128 字节内是双射；被跳过的 writer 地址和所有 reader 地址不相交；跨 wave 的 A 共享仍由原 WG barrier 发布；SFA/SFB 的 row/pack/half 映射完全不变；首次加载和每次 stage 复用都先初始化每个会被消费的元素，实际尾 M 必读零也保留。

## I3：只把 B producer 请求提前，首轮第二个独立版本

唯一因素是 `issue_tile` 的 operand 请求顺序：A0、A1、B0、B1 → **B0、B1、A0、A1**。不启用 I1、不改变请求总数、地址、stage、预取距离、cache、寄存器或 scale，位置仍在原 562 行 barrier 之后。未来确认生成 ISA 确实采用预期顺序；若编译器恢复原顺序，该版本不能算完成预期实验。

目的只是验证 B 先发能否改善两种 operand 的就绪顺序。CPU 没有证据表明当前受 B 延迟限制，也不承诺 B 先发更快。A/B LDS 区域不相交，当前 stage 与 next stage 不相交；所有 writer 请求仍先于对应 consume 的原 wait/barrier，不能把 producer 整体移到 stage 复用 barrier 之前。

## I2：B fragment 按 N half 复用一个寄存器环，条件性的第三版本

仅改变 B 的 register-live 分块：保留两个 M half 的 A 寄存器、scale、global→LDS 和下一 stage 预取。把 `v_b[2*E_N]` 换为 `v_b[E_N]`，先读 N half0 的 E_N 个 B fragment，等 LDS 完成并对两个 M half 做 MFMA；再读 N half1 到同一数组，等 LDS 完成后计算。所有 C 索引与每个 C fragment 跨 K 的 MFMA 顺序保持原样，两个 half 的 scale selector 不变。

这是一个因素（B register staging），新增的就绪等待是该 staging 的必要依赖；不同时采用 I1/I3、不改变 A/B 的 global 请求数和 LDS 槽。BM192/WaveN4 时 B 数组元素数由 4 降为 2，但实际 VGPR/AGPR、spill 与 ISA 结果只能在未来获准编译后确认。因为原 LDS 仍占 160KiB，不能据数组缩小推断多驻留一个 WG。只有 baseline 的编译/ISA 审查或后续 profiling 支持寄存器/operand 排程方向时，才进入此第三版本。

## 有限验证矩阵与接受标准

首轮只用 3 个落后代表 × 同一 interleave anchor 的原版、I1、I3，共 **9 个精确 shape/version 对，一批筛选**。只有正确且显示收益的版本进入代表项重复配对和两个保护 shape；保护项同时保留各自既有 dispatch 赢家 692/691 的原版。完整设计上限为 3 个代表 + 2 个保护各测原版、I1、I3，再加两个固定 dispatch 赢家，共 **17 对**；若全部进入 3 批重复，上限为 51 个计时组，并非首轮必跑规模。不跑完整 Top4/Top8。I2 仅在前两版证据不足且寄存器/operand 排程方向有依据时追加，最多再加 5 对，不与 I1/I3 做笛卡尔积。保护 shape 的 interleave anchor 固定为 47680，不伪称它是此前保护项最佳。所有新版本仍需源码实现和单独验证，本计划还没有可执行的新 kernel。

未来 GPU 验证需另行满足当时授权：先正确性，再同次交替计时；计时原版/新版本都用相同输入、101 次计时与 5 次预热，保留每次计时，汇报中位数和波动。和固定 CSV 的胜负另报，与原版同次差值用于归因，不再 tune FlyDSL。误差上限沿用 0.05，另记录实际误差；保护项既有最佳也随次计时。仅当多数代表收益可重复、保护项无超出波动的明显回退、静态及正确性检查通过，才扩到上述 8 个 interleave 落后 shape；全 171 项验证属于后续阶段。

CPU 枚举结果见 `interleave_plan.json.static_validation`：六种合法 BM/WaveN 几何的全部 wave、lane、A step、两 M half、consumer fragment 地址均通过覆盖与不消费省略行的检查；scale coverage 保持；stage 0/1 模型针对 2、4、56、128 个 K128 step 检查了预取边界、初始化、复用和尾部 drain 顺序。模型是源码设计的数学枚举，不是设备执行或编译后的 ISA 证明。
