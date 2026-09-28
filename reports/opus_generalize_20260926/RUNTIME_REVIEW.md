# Runtime v1 CPU 静态审查（2026-09-26）

本审查只读取源码及已有 `experiments.so`，使用 CPU 解析 ELF/offload bundle、`llvm-readobj --notes`、`llvm-objdump -d --mcpu=gfx950` 和 kernel descriptor。**没有重新编译、没有运行 GPU、没有新增性能或数值实测。** 9000 和四个 v1 kernel 均未修改。后四张 GPU 被其他任务占用，v1 的性能判断等待目标 99 项实测；不以静态推测替代测量，也不据此盲改 kernel。

## 实际 kernel 数与支持范围

三个共享库的 gfx950 AMDGPU metadata 分别列出 **1、2、1 个 device kernel，共 4 个**。同时检查了 host launch 和 device 名称：没有按具体 K 值分发的固定 K 实例，也没有用一个 ID 包装多个固定 K kernel。short 的模板参数只有 tile M；静态 stage0/stage1 是同一 device kernel 内部的 U2 流水线。

| ID | 编译后的 device kernel | Tile / waves | K 范围 | M、N 约束 | 99 项中可接受项数 |
| --- | --- | --- | --- | --- | ---: |
| 20000 | `gemm_a8w8_mxfp8_bpreshuffle_long_runtime_kernel` | 192×256 / 8 | 128…16384，128 的倍数 | M%64=0，N%256=0 | 99 |
| 20010 | `short_runtime_kernel<short_runtime_traits<128>>` | 128×128 / 4 | 128…1536，128 的倍数 | M%64=0，N%128=0 | 43 |
| 20011 | `short_runtime_kernel<short_runtime_traits<160>>` | 160×128 / 4 | 128…1536，128 的倍数 | M%64=0，N%128=0 | 43 |
| 20020 | `runtime_n224_kernel` | 192×224 / 8 | 128…1536，128 的倍数 | M%64=0，N%896=0 | 33 |

范围计数依据 `shapes99.csv` 和 launcher 尺寸约束，不代表已通过数值验证或达到旧候选性能。目标 99 项在尺寸上没有覆盖空洞，全部可由 20000 接受。43 项短 K 是 K384/768/1024/1536 的 19/13/1/10 项；20020 可接受其中前 33 项，不能覆盖 N16384/65536 的 10 项 K1536。所有 launcher 还要求正尺寸、合法指针、对齐和 int32 安全的矩阵字节范围。

若最终保留冻结的 9000 作为独立候选，应明确报告为“4 个新增 runtime kernel + 9000”，不是总共 4 个。当前比较配置里保留旧 25 个候选用于测量，不等于已完成最终候选替换。

审查产物 SHA256：

| 库 | SHA256 |
| --- | --- |
| long_runtime/experiments.so | `d034a6c6897adee03379fbed289fd286f000fbd3cef0ceca50764efdd73069f3` |
| short_runtime/experiments.so | `90f4fdd7d7a8804a4b4f4ca54b4de7bcd31c3b338dab43d49a5a3e7135a4e05a` |
| n224_runtime/experiments.so | `ac8d26eb9362a69b5ee6bbfdb16fe40784f4579f7ed8071e7815c2e9f9308551` |

## 编译资源与 VGPR 口径修正

这批 gfx950 LLVM 产物中，short 的 metadata `.vgpr_count` **已经包含 AGPR 区间**，不能在 204/244 上再加 64/80。`accum_offset` 从 `.kd` 的 `COMPUTE_PGM_RSRC3` 解码，硬件分配数从 `COMPUTE_PGM_RSRC1` 的 granulated VGPR count 解码；字段位置参考本机 `/opt/rocm/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h`。普通 V 高水位通过反汇编操作数统计，最终分配应以 descriptor 为准。

| 候选 | metadata VGPR | AGPR | accum_offset | 指令普通 V 最高编号 | descriptor 分配 VGPR | SGPR | LDS bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20000 | 207 | 0 | 208 | — | 208 | 48 | 143104 |
| 20010 | 204 | 64 | 140 | v137 | 208 | 46 | 69132 |
| 20011 | 244 | 80 | 164 | v163 | 248 | 48 | 77964 |
| 20020 | 254 | 0 | 256 | — | 256 | 64 | 112164 |
| 旧 12340，K384 | 128 | 64 | 64 | v61 | 176 | 35 | 67971 |
| 旧 12341，K768 | 152 | 64 | 88 | v85 | 176 | 45 | 68358 |
| 旧 12342，K1024 | 152 | 64 | 88 | v85 | 176 | 46 | 68616 |
| 旧 12840，K384 | 180 | 80 | 100 | v99 | 184 | 36 | 76515 |
| 旧 12841，K768 | 180 | 80 | 100 | v97 | 184 | 47 | 76998 |
| 旧 12843，K1536 | 180 | 80 | 100 | v99 | 184 | 49 | 77964 |

旧 1234x 的 descriptor 已预留 176 个，超过 metadata 的 128/152；不能把 metadata 差额直接当成实际分配差额。short 的实际分配变化是 **176→208、184→248**。旧 fixed-long K3072/K7168/K16384 的 metadata VGPR 都是 208，20000 为 207；旧 n224 K384 为 252，20020 为 254。四个新 kernel 的 private segment、VGPR spill、SGPR spill 均为 0，反汇编未见 scratch 指令。

两种 short 的 LDS 均小于 80 KiB，两个 workgroup 总 LDS 仍在 160 KiB 内；三个则超出。按 gfx950 常规四 SIMD 的 4-wave workgroup 分配，两个 workgroup 所需普通/累加寄存器分配上界分别为 416、496，不超过 512。**静态资源约束没有显示 short 从 2 WGs/CU 降为 1 WG/CU。** 这不是 GPU 实测 occupancy；寄存器活跃区、指令调度或等待仍可能影响延迟。

已链接二进制不保留源码形式的 `.amdhsa_next_free_vgpr` 指令。本表使用实际 V 操作数高水位、`accum_offset`、metadata 和 descriptor 分配共同核对，不把推导值冒充原始 assembler directive。

## 性能最可能退化的位置

1. **short 的 runtime U2 增大普通 V 活跃区，并增加首组/短路径初始化。** C 仍固定为 AGPR64/80；反汇编每个 kernel 只有 64/80 条 `v_accvgpr_read_b32`，对应输出搬出，没有发现每轮 C→V→C 复制。普通 V 高水位从旧 v61/v85、v97/v99 增到 v137、v163，U2 两个静态 stage 使用不同 A/B 寄存器组，动态尾部合流延长了部分值的活跃期。新两个 kernel 各有静态 128/160 条 `v_accvgpr_write_b32`，出现在初始化/短路径，旧固定 K 为 0；旧首组 MFMA 可直接使用零累加器。静态条数不是每个 K 实际执行次数，也不能直接换算时间。

2. **20000 的边界条件留在每个 advance 内。** `long_runtime/pipeline_runtime.cuh:154` 的 `tile+2<loops` 和 U2 第二步条件，与旧 fixed-long 无边界检查的稳态、独立无预取末步不同。K3072/7168/16384 原来就使用 U2，动态 K 本身不是唯一差异。20000 还固定预留 128-group scales，K3072/7168 的 LDS 从旧 122928/129136 增为 143104，不过新旧都超过 80 KiB，未显示 workgroup occupancy 档位变化；实际 scale 全局读取仍按当前 K 截止。

3. **目前四个配置没有保留旧 grid/pitch/cache/fused 的全部性能特点。** 20000 固定 N-first、pitch264、cache2；旧 11971/12071 有 M-first/group-M4，10272 使用 pitch272。旧选择表中 9 项来自 long_grid；将它们交给同一个 N-first runtime kernel 是否持平，需要测量。大 M、K384/768 的旧 192×256 short 候选还有 group-M4、cache0 或 fused final MFMA/output。20020 的 192×224/cache0/fused 只能部分覆盖这些作用；20010/20011 则使用更小 tile，CTA 数更多。源码通用范围充分，并不意味着旧 25 个赢家的速度自然保留。

4. **20020 的短 K 完全展开改为 runtime U2。** 它保留 n224 的 B-scale 跨 128 列组处理、cache0、pitch232、末次 MFMA 与输出转换交错。寄存器计数基本持平，但 SGPR 从旧 43 增为 64；动态 scale 地址和循环开销可能对 K384 这种短工作更显著。现有静态信息不足以判断 20020 是否能替代旧 15940。

如果 v1 的目标测量证实退化，优先尝试以下局部变动，而不是新增固定 K 特化：首组单独处理并显式使用零累加器；把无条件预取的 U2 稳态与最多两步尾部拆开；限制下一组 scale/矩阵临时量的作用域，保留原有最后一次使用后替换顺序。这些仍可保持每个 tile 一个 runtime kernel。是否需要第二种通用 grid 或短 K 的 192×256 配置，应由 99 项的实际损失分布决定。当前 v1 先保持不变。

## 数值与同步的源码检查

未发现针对当前 99 项的明确 K drain、scale 越界或重复 final compute 问题。三个主体都先加载 K0，只在存在 K1 时预取 K1；advance 仅在下一组存在时执行，未来预取只针对真实 K+2；最后 K group 恰好计算一次。K128、偶数组和奇数组分支在 workgroup 内一致。scale 的最大容量与实际读取范围分离，未初始化的多余 SFA panel 不被读取。

删除重复 output 前 wait/barrier 的前提在源码中仍成立：K128 经过 prologue 操作数读取后的 barrier，其他 K 经过最终 advance 的完整 wait/barrier；之后 MFMA 只读取寄存器，再把矩阵 LDS 用于 BF16 输出。输出写 LDS 后的发布 barrier 保留。20020 对跨 N128 scale 组的地址计算沿用旧 n224 逻辑。以上是源码分析，不是数值实测结论。

## 下一步范围

等待允许使用的后四张 GPU（4–7）空闲后，只继续已经计划的 **99 个目标 shape 的 kernel 性能与数值检查**，同 GPU、同批比较四个 runtime v1 与旧实际赢家，保留 9000 冻结。首先检查旧 short n128/m160 与 long_grid 获胜项是否退化，再据结果决定是否调整稳态/尾部。不要额外扩展随机 shape、边界压力测试、全仓测试或新的 kernel 变体搜索。
