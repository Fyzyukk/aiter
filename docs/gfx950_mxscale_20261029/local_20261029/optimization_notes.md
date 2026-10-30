# 第三步定向实验准备（CPU 源码审查，尚无新 GPU 结论）

基于分支 `fyz/10-29-aiter`、提交 `39cdf47d864c`。本文件只整理实验依据，不改变核心源码或候选清单。最终实验对象需以第一、二步的新实测剩余落后集合为准。

## 现有候选和历史目标

CPU 导入 `csrc/opus_gemm/opus_gemm_common.py` 核对到 780 个 CO：`pin_agpr` 12、`prefetch_b_vgpr` 384、`prefetch_b_vgpr_overlap` 384。普通 B 和 overlap 当前 CO 均使用 BK128、A stage4，只有两个完整 bundle：`BD2/XCD=true/Contiguous=true` 与 `BD1/XCD=false/Contiguous=false`，带已有 pin 组合；它们不能作为只改 BD 的消融实验。

历史 67 个两边落后的集合如下；这是选择初始样本的依据，不能用作本轮最终统计。

| N | K | 数量 | M |
| --- | --- | --- | --- |
| 768 | 7168 | 8 | 1088,1152,1280,1536,12288,14336,16384,32768 |
| 1536 | 7168 | 14 | 2048,6144,8192,10240,14336,16384,20480,24576,28672,32768,40960,49152,57344,65536 |
| 2048 | 7168 | 7 | 6144,8192,10240,12288,14336,16384,32768 |
| 6144 | 7168 | 26 | 1344,1408,1472,1536,1600,1664,1728,1792,1856,1920,1984,2048,6144,8192,10240,12288,14336,16384,20480,24576,28672,32768,40960,49152,57344,65536 |
| 7168 | 768 | 5 | 1920,1984,8192,12288,24576 |
| 7168 | 1024 | 1 | 8192 |
| 7168 | 2048 | 6 | 1216,1536,1600,1664,1792,1984 |

## B-VGPR / overlap：记录实际队列距离

依据 `opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh:71`：

```text
MR = BM / 16
NR = BN / (WN * 16)
B_FRAGMENTS = (BK / 128) * NR
REGISTER_BUDGET = WGPerCU == 2 ? 256 : 512
B_REGISTER_ROOM = REGISTER_BUDGET - MR*NR*4 - (BK/128)*MR*8 - 48
B_MAX_AHEAD = max(B_REGISTER_ROOM/8 - 1, 1)
ROTATE_B = WGPerCU == 2 || (BK/128)*MR >= 16
B_SCHEDULE_AHEAD = ROTATE_B ? (NB == 2 ? 1 : 4)
  : (BK == 128 && NR >= 4 && PinA && K >= 1536 ? 3 : B_MAX_AHEAD)
B_AHEAD = min(BD * B_FRAGMENTS, B_SCHEDULE_AHEAD)
```

整数除法与 C++ 保持一致。名字中的 `bd` / `ba` 是输入值，不是实际距离。

| 现有 K7168、PinA-only 几何（BM×BN×BK，waves） | BD1 kid | BD2 kid | B_FRAGMENTS | B_AHEAD(BD1/BD2) |
| --- | --- | --- | --- | --- |
| 48×128×128，1×2 | 418 | 133 | 4 | 3 / 3 |
| 64×128×128，1×2 | 526 | 25 | 4 | 3 / 3 |
| 64×128×128，1×4 | 442 | 157 | 2 | 2 / 4 |
| 96×128×128，1×4 | 556 | 55 | 2 | 2 / 4 |
| 96×256×128，1×4 | 472 | 79 | 4 | 3 / 3 |
| 128×128×128，1×4 | 580 | 103 | 2 | 2 / 4 |
| 128×256×128，1×4 | 496 | 187 | 4 | 3 / 3 |

建议先固定本轮每个 N/K、M 类别最好的几何和 pin 组合，随后只做有限消融：

1. 先使用已有 JIT 比较 BN64/128、WN2/4、NB2/3/4，保持缓存、输出算法、WGPerCU=1 一致。现有大范围 B/overlap JIT 为 BK256，pin 均关闭，XCD=true；不能将它与 BK128/pinned CO 的差别仅归因于某一参数。
2. 最好的两种几何各做固定 NB/BD/Contiguous 的 XCD false/true 对照，再固定 XCD 改 BD。若计算出的 `B_AHEAD` 相同，优先跳过重复队列距离；如要比较编译器行为需明确标为“相同实际距离”。固定 `Contiguous`，因为它把运行时 stride 替换为编译期 K。
3. 普通 B 的 `PublishM=-1` 在每个 panel 首个 fragment 的 MFMA 之前发出未来 A；overlap 在 `m==PublishM` 的 MFMA 之后发出未来 A。先比较 -1 与 0，已有 JIT 才扩到该几何其余合法 publish point。两者共享同一 `bulk_panel_body`，因此可做严格配对。
4. 对 BK256、BM128 路线，`ROTATE_B=true`：NB2 队列距离固定到 1；NB3/4通常固定到4，BD1/2容易相同。NB从2到3不只增加A stage，也改变B队列上限，统计中需注明。

初始代表性输入可选仍落后者中的 `(M,N,K)=(1088,768,7168)、(12288,768,7168)、(2048,1536,7168)、(32768,1536,7168)、(8192,2048,7168)、(32768,2048,7168)`，以覆盖小M和大M。先得到有效方向，再扩到该 N/K 剩余输入；不要直接展开全套笛卡尔积。

## 4wave：先用已有 12 个 CO，再做分组消融

当前全部为 256×256×128、wave2×2、runtime K、Unroll2、GroupM0、LDS输出。12个来自 panel16/32/64 × PAD/non-PAD × WideOutputfalse/true：

| PAD | WideOutput | panel16 | panel32 | panel64 |
| --- | --- | --- | --- | --- |
| true | true | kid0 | kid1 | kid2 |
| false | true | kid387 | kid388 | kid389 |
| false | false | kid390 | kid391 | kid392 |
| true | false | kid393 | kid394 | kid395 |

non-PAD 要求 M%256==0。WideOutput=false 的 `max_k=null`，true 路线注册 `max_k=16384`；当前四个重点K都在允许范围内。WideOutputtrue 使用64位矩阵基址和tile限定32位SRD，false 使用普通SRD；两条仍通过LDS输出，不应称作两种不同寄存器/直接输出算法。

| panel | LDS bytes | K7168(56×K128)面板次数 | K16384(128×K128)面板次数 |
| --- | --- | --- | --- |
| 16 | 139392 | 4 | 8 |
| 32 | 143616 | 2 | 4 |
| 64 | 152064 | 1 | 2 |

所有面板的 LDS 均超过160KiB的一半，减小panel不会使当前几何自动变成2 WG/CU。panel16/32可能少量减少LDS分配，却增加同步refill次数；需看实测耗时和wait，而不是仅由面板大小推断驻留改善。`refill_scale_panel` 在边界等待lgkmcnt、同步、载入SFA/SFB、等待vmcnt、发布LDS、再等待/同步，长K中可能暴露周期性停顿。

GroupM0 在 `((m|n)&511)==0` 时默认交换 block_m/block_n 最低位，形成2×2 tile swizzle；GroupM4/8会替代它。**不要将GroupM0统一描述为线性遍历。** 推荐按M是否512整除分组记录。

实验顺序：

1. 对仍落后的 `(N,K)=(6144,7168)、(7168,7168)、(7168,16384)、(7168,4096)`，先每类选1个小M/部分tile和1个大M（已有输入中存在者），测全部合法的现有12个CO。对同一 M%256==0 输入，PAD/non-PAD也可严格配对。
2. 固定各类中最好的panel/PAD/WideOutput和原有pin/编译参数，追加只改GroupM4/8的两个实验配置；先与GroupM0交替复测。只有4/8有稳定收益，才增加GroupM16。
3. 有收益再扩到该类所有落后输入，同时验证首个/最后一个分组、partial M tile正确性。
4. 如果GroupM无稳定收益，再分析实测MFMA活跃度、VMEM/LDS wait和驻留wave，优先考虑scale refill与下个A/B加载的重叠。等待计数与LDS槽生命周期耦合，不能只删除wait或barrier。

## 注册和记录的注意事项

`opus_gemm_common.py:2938` 当前用 `len(gfx950_co_kernels_declared)` 作为 JIT 起始编号，即780。**直接在CO JSON追加2个条目会导致所有现有JIT编号整体偏移**。第3步实验建议使用独立临时CO目录/配置并按完整名字匹配；如集成到清单，须先明确保留已有JIT编号的分配方案，新增CO避免与现有780–48211范围冲突。历史kid不能直接当本分支配置。

每次实验记录至少包括 `(M,N,K)、pipeline、kid和完整traits、BM/BN/BK/WN、NB、输入BD与实际B_AHEAD、PinA/PinC、XCD、Contiguous、PublishM、GroupM、panel、PAD、WideOutput、正确性、每次重复耗时、固定CSV基线`。依据正确性和OPUS耗时判断CSV胜负，同轮FlyDSL仅补充比较。上述源码审查没有产生新GPU胜出数。
