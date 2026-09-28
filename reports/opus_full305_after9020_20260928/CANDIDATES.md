# 当前 8 个 OPUS MXFP8 B-preshuffle 候选

本次完整输入为 **305 个 shape**；相对历史 295 项新增的是 **10 个 shape**。当前 OPUS registry 有 **8 个候选 ID**，不是新增 10 个 kernel。下表按 2026-09-28 当前生产源码整理，9020 为刚整合的 B-pointer 版本；最终每个 shape 的选择与速度以本次完整重调优结果为准。

映射依据：[registry 与 shape guard](../../csrc/opus_gemm/opus_gemm_common.py#L1733)、[exact-instance codegen](../../csrc/opus_gemm/codegen/gen_instances_gfx950.py#L2826)。表中 tile 单位为元素，wave 均为 Wave64；所有候选的 K 都是运行时参数，不是按具体 K 生成独立 kernel。

| ID | Tile M×N×K；waves | Runtime K | M / N 对齐与尾部 | Pipeline 入口 | Traits 实例 |
|---|---|---|---|---|---|
| 9000 | 256×256×128；4 | 正 128 倍数；registry/launcher 未设 16384 上限，仍受 extent 限制 | M%256=0，N%256=0；无 M 尾块 | [4wave pipeline][p0] | [4wave traits][t0] |
| 9010 | 256×256×128；4 | 同 9000 | M%64=0，N%256=0；末 M tile 内部零填充、输出有界 | [padded-M wrapper][p1]，计算主体为 [4wave][p0] | [padded-M traits][t1]，继承 4wave 并设 `PAD_M=true` |
| 9020 | 192×256×128；8 | 128…16384，步长 128 | M%64=0，N%256=0；支持 M 尾块 | [main pipeline][p2] | [main traits][t2]；pipeline 内再包一层本地 storage traits |
| 9021 | 128×128×128；4 | 128…16384，步长 128 | M%64=0，N%128=0；支持 M 尾块 | [small pipeline][ps] | [small traits][ts]`<128>` |
| 9022 | 160×128×128；4 | 128…16384，步长 128 | M%64=0，N%128=0；支持 M 尾块 | [small pipeline][ps] | [small traits][ts]`<160>` |
| 9023 | 64×128×128；4 | 128…16384，步长 128 | M%64=0，N%128=0；M tile 恰好整除 | [narrow pipeline][pn] | [narrow traits][tn]`<128>` |
| 9024 | 64×64×128；4 | 128…16384，步长 128 | M%64=0，**N 仍须 %128=0**，由 B-scale 的 N128 分组约束 | [narrow pipeline][pn] | [narrow traits][tn]`<64>` |
| 9030 | 192×256×128；8 | 128…16384，步长 128 | M%64=0，N%256=0；支持 M 尾块；仅接受大输出范围 | [large-output pipeline][pl] | [large-output traits][tl]，继承未改 storage 的 main traits |

8 个候选对应 **6 个 pipeline 入口头文件、5 个实际 device kernel 模板主体**：9010 的入口只是 wrapper，复用 9000 主体；9021/9022 共享 small 主体；9023/9024 共享 narrow 主体；9020 与 9030 各有独立主体。不能把候选 ID 数等同于独立 pipeline 文件数。

| ID | 源码中的关键优化 | 面向的瓶颈 / shape 特征 |
|---|---|---|
| 9000 | 256×256 大 tile、双 LDS stage；A/B 及部分累加器固定 AGPR；显式 MFMA/LDS/VMEM 调度；64 个 K128 scale-group panel 可 refill；M、N 都为 512 倍数时做 2×2 tile swizzle。LDS 152064 bytes。 | 大而规则的 M/N tile，通过复用输入、减少 workgroup 数及细排指令提高计算吞吐；scale panel refill 处理较长 K。 |
| 9010 | 保留 9000 的计算调度，以 bounded A buffer 零填充尾行、检查输出行；一次 GEMM 完成，无单独尾 kernel 或全局 padded tensor。LDS 152064 bytes。 | M 是 64 倍数但不是 256 倍数时，补足原 9000 的覆盖，避免额外尾部 launch / padding 搬运。 |
| 9020 | 192×256/8-wave、两 stage；A 每 wave 连续 8 行 compact 装载，K16 chunk 用 `((row>>1)&7)` XOR；K0 先于 scale panel 发射；每轮先算 8 个 MFMA 再发布下一 tile；两个 N128 B-scale 字节打包一 word，每组一次 LDS 读取；使用点重建 wave-uniform B resource；最终 MFMA 与 BF16 LDS staging 交错。N≤2048 且 M≥4096 时按 4 个 M tiles 分组。有效 LDS **143360 bytes**。 | 本轮重点针对 N=7168、K=384/768、M≈1472…1728 的短 K 加载/屏障与 scale 读取开销；A 布局改善装载局部性及 LDS bank 分布，B resource 重建避免描述符跨 scale 分歧路径。M 分组针对高 M、窄 N 的 B 复用。全部仍是一个通用 runtime-K 流程。 |
| 9021 | M128 使用 **3 个 matrix slots**；4-wave、A/C 固定 AGPR；32-group scale panel refill；runtime U1 ring，中点 partial VMEM wait，省去三 slot 情况下每轮末的 consumer barrier；BF16 经 LDS 合并为 16-byte 输出。LDS 105504 bytes。 | tile 更小时增加可调度 workgroup，减少大 tile 的尾部浪费；三 slot 主要隐藏单 workgroup 驻留时的访存/同步延迟。 |
| 9022 | 同一 small 主体改为 M160、**2 个 matrix slots**，保留 A/C 固定 AGPR、32-group panel 和末次使用后替换操作数。LDS **81184 bytes**，在 80 KiB/workgroup 预算内。 | 在较高 M 复用与两个 workgroup 的 LDS 驻留容量之间折中；覆盖与 M128 不同的 grid/tail 分布。源码容量允许双驻留，不代表所有资源条件下都保证双驻留。 |
| 9023 | 64×128 小 tile、**3 个 matrix slots / 预取距离 2**；A XOR 布局，两个 M-repeat 的 A scale 打包；64-group scale panel refill；按真实未完成预取数选择 VMEM wait；直接 MFMA fragment→BF16 global 输出。LDS 80384 bytes。 | 窄 N、workgroup 数偏少或大 tile 同步开销较重时，提高并行度并隐藏访存；小输出 tile 避免完整 C-shuffle 的 LDS 往返。grid_n≤16 且总 tile 数>256 时有几何分区遍历。 |
| 9024 | 同一 narrow 主体改为 64×64、**4 个 matrix slots / 预取距离 3**；其余 A/scale 布局与统一 advance 相同。LDS 71936 bytes；紧凑 grid 还可按成对 M tiles 交错遍历。 | 更窄 N 或更少输出列时增加 workgroup 数，借更深预取补偿每组计算量减少。成对遍历只在 `grid_n≤16`、`grid_n%4=0`、`16≤grid_m≤32` 时启用，条件只依赖几何。 |
| 9030 | C 基址改为 64-bit `row*stride_c+col`，buffer range 只覆盖当前输出 tile；输入与 tile-local offset 继续使用受限 32-bit 地址。保留此前 192×256 主体的双 stage、U2、128-group resident scales、最终 MFMA/BF16 staging 和 M 分组。LDS **143104 bytes**。 | 解决 BF16 C 总字节数超过 signed-int 上限的覆盖问题，是新增 10 个大 shape 所需的地址扩展；没有同步带入此次 9020 的 A compact/XOR、中点调度、B-scale pack 或 B-resource 重建。 |

共同调用约束：gfx950；FP8 A/B 与 BF16 C；连续存储；2D 或 batch=1 的 3D；A/B/C 地址 16-byte 对齐；逻辑 A-scale 为 `[M,K/128]`、稠密列主序，B-scale 为 `[N/128,K/128]`、行主序；scale 为 native E8M0 或保存其字节的 u8。9020–9030 的生成 launcher 还检查 A-scale 16-byte 对齐。B 必须已经按约定 preshuffle；这些候选不在 kernel 内完成 B 预处理。候选均为 direct GEMM，没有 OPUS split-K/中间 workspace 路径。 编译要求 clang 支持 `clang::amdgpu_pin_agpr`；所有这些 pipeline 都包含带该能力检查的 4wave 头文件。

地址范围按 [shape guard](../../csrc/opus_gemm/opus_gemm_common.py#L1782) 和 [生成 launcher](../../csrc/opus_gemm/codegen/gen_instances_gfx950.py#L2912) 区分：9000–9024 要求 `max(M*K, N*K, 2*M*N) ≤ 2^31−1`；9030 要求 A/B 各不超过该上限、`2*((192−1)*N+256) ≤ 2^31−1`，且 `2^31−1 < 2*M*N ≤ 2^63−1`。因此 9030 不会作为普通小输出 shape 的替代候选。

9020 的 [公开 main traits][t2] 仍保留 143104-byte 原布局，便于 9030 继续继承；9020 专属 [pipeline 内 storage traits][p2] 才把 `SFB_BYTES` 改为 512、`B_SCALE_PACKS` 改为 1、总 LDS 改为 143360。main 和 large-output 的公共几何来源是 [192×256 base traits][tb]。本表的优化目标来自源码结构与当前任务范围，未额外运行编译、GPU 或性能测试。

[p0]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh
[t0]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh
[p1]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh
[t1]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh
[p2]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_main_gfx950.cuh
[t2]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_main_gfx950.cuh
[ps]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_gfx950.cuh
[ts]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh
[pn]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh
[tn]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh
[pl]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_large_output_gfx950.cuh
[tl]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_large_output_gfx950.cuh
[tb]: ../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh
