# 当前八个 OPUS 候选的优化方案

本文按当前 ID 说明目标、最终采用的步骤和保留配置。步骤按技术依赖组织，**不是完整实验时间线，也不表示每一步都有独立的性能收益测量**。9000 是保留的基线；当前文件拆分和内部章节整理属于源码重构。

本次内部章节重组已完成。[最终结构验证](attempt2/validation.json)确认 6 个独立计算主体的原始机器指令字节均与重构前一致，metadata、public、registry 和 host 内容一致；9000 原主体不动，9010 继续复用它。两轮 CPU 验证共完成 12 次编译，本轮没有新增 GPU 性能测量。

最近的测量依据是 [full305 原始 tuner 结果](../opus_full305_after9020_20260928/RESULTS.md)：一次完整扫描覆盖 305 个 shape、29,180 行 profile，全部 1,955 条 OPUS 候选/shape 记录通过 `errRatio=0`，305/305 均有有效最小值选择。扫描使用 GPU0–7、warmup=5、iters=51，包含全部后端和合法 splitK；它不是多轮独立复测。全部后端最终赢家为 OPUS 299、CK 2、CKTile 4、ASM 0。在有有效外部对照的 300 个 shape 中，OPUS 快 294、慢 6，几何平均耗时降低 12.3983%；另 5 个只有有效 OPUS 候选，不能计算外部加速比。这些结果描述该次已测实现的覆盖，不能据此给本次源码重排增加性能结论。

下表的“有效 / 全后端胜出 / OPUS 内最快”均来自该次 full305；有效数量仅表示本批次测得的覆盖，并非完整支持域。所有 tile 的 K 维均为 128，wave 均为 Wave64。scale 组数以一个 K128 组为单位。

| ID | Tile M×N；wave 数 | 矩阵 LDS 槽数 | Scale 策略 | LDS 字节 | 有效 / 全后端胜出 / OPUS 内最快 |
| --- | --- | --- | --- | ---: | ---: |
| 9000 | 256×256；4 | 2 | 64 组面板，循环补充 | 152064 | 175 / 130 / 130 |
| 9010 | 256×256；4 | 2 | 64 组面板，循环补充 | 152064 | 295 / 29 / 29 |
| 9020 | 192×256；8 | 2 | 128 组常驻，B scale 打包 | 143360 | 295 / 44 / 49 |
| 9021 | 128×128；4 | 3 | 32 组面板，循环补充 | 105504 | 295 / 12 / 12 |
| 9022 | 160×128；4 | 2 | 32 组面板，循环补充 | 81184 | 295 / 51 / 51 |
| 9023 | 64×128；4 | 3 | 64 组面板，循环补充 | 80384 | 295 / 16 / 16 |
| 9024 | 64×64；4 | 4 | 64 组面板，循环补充 | 71936 | 295 / 7 / 8 |
| 9030 | 192×256；8 | 2 | 128 组常驻，保留原 B scale 布局 | 143104 | 10 / 10 / 10 |

## 9000：整块 256×256 基线

目标是能充分利用大 tile 的规则形状，以输入复用和较少的 workgroup 降低开销。要求 `M%256=0`、`N%256=0`，不处理 M 尾部；K 是正的 128 倍数，注册约束未像 9020 等候选一样显式限制为 16384，但仍须满足 buffer 范围限制。

最终采用方案的技术顺序：

1. 使用四 wave 的 256×256 tile，提高每次 A/B 读取对应的计算量。
2. 使用两级矩阵 LDS 流水，让直接 VMEM 预取与当前 tile 的 MFMA 重叠。
3. 将 A/B 和选定累加器片段固定到 AGPR，并显式组织 MFMA、LDS 和 VMEM 调度，控制寄存器使用和依赖等待。
4. 用 64 组 scale 面板配合补充逻辑覆盖长 K；当 M、N 都是 512 的倍数时采用 2×2 tile swizzle，改善相邻 tile 的数据复用。

保留两级流水、64 组 scale 补充，以及 `AGPR → BF16 → LDS → global` 输出：先重排到 LDS，再以连续 8 个 BF16、即 16 字节写回。该候选是既有基线，不能将其步骤表述为本次新增优化。最近 full305 中有效 175 个 shape，130 个在全部后端中胜出。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh)。

## 9010：支持 M 尾部的 256×256

目标是让 `M%64=0`、`N%256=0` 的形状继续使用 9000 的大 tile，避免 M 不整除 256 时失去该实现。当前 9010 是 padded-M 256×256 候选。

最终采用方案的技术顺序：

1. 通过包装入口设置 `PAD_M=true`，直接复用 9000 的 device body 和计算布局。
2. 对 A 使用有界 buffer，尾部无效行读取得到零值，使完整 tile 计算仍然成立。
3. 在同一次 kernel 中约束输出有效行，省去显式 padded 全局张量和单独的尾部 kernel。

保留四 wave、两级流水、64 组 scale 补充、152064 字节 LDS，以及经 LDS 重排的 BF16 16 字节写回。最近 full305 中有效 295 个 shape，29 个在全部后端中胜出。

源码：[包装 pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh)。

## 9020：192×256 主流水的访存与发布时机调整

目标包括短 K=384/768、N=7168、M 约 1472–1728 的低效区，以及专项选取的 16 个慢 shape；192×256 tile 在输入复用和 workgroup 数量之间折中。支持 `M%64=0`、`N%256=0`、K 为 128–16384 内的 128 倍数。

最终采用方案的技术顺序：

1. **紧凑 A producer 与 XOR 布局。** 每个 wave 负责 8 个相邻行，K16 chunk 使用 `((row >> 1) & 7)` XOR，改善 A 读取局部性和 LDS bank 分布；每级为 3 条 A、4 条 B VMEM。
2. **打包 B scale。** 每个 K128 组把两个 N128 分组的 B scale 字节装入一个 u32，以一次 LDS word 读取配合 MFMA 的字节选择，减少 scale 读取和操作数准备。
3. **中点发布。** 先计算前 8 条 MFMA，再等待并发布下一个 tile；确认旧读者结束后，把未来 tile 发到释放的槽位，以当前计算覆盖等待并保持槽位生命周期明确。
4. **K0 与 scale 面板重叠。** Prologue 改为 `issue K0 → scale panel → issue K1`，并用 `vmcnt(7)` 允许 K1 在第一次计算时仍在途，缩短起步阶段的串行等待。
5. **每次 B 预取重建一致的资源描述符。** 将 64 位 B 地址拆成低/高 32 位，分别 `readfirstlane` 后重建并调用 `make_gmem`，避免不透明描述符穿过分歧 scale 加载路径所引入的 descriptor waterfall。

保留八 wave、两级流水、U2 runtime loop 和奇偶 drain；128 组 scale 在支持的 K 范围内一次加载常驻，无需补充。保留最后一组 MFMA 与 BF16 LDS staging 交错、再连续 16 字节有界写回；当 `N≤2048` 且 `M≥4096` 时，按 4 个 M tile 分组遍历以复用 B。当前 traits 直接包含打包后的 B scale 存储：`SFB_BYTES=512`、`B_SCALE_PACKS=1`、LDS=143360。最终方案没有采用强制 AGPR 累加器实验、完整输出 tile 的单独快路径或 16-MFMA 中点方案。

最近 full305 中有效 295 个 shape，44 个在全部后端中胜出，49 个是最快 OPUS 候选。此前 slow16 在这次完整扫描中是 **11 快 / 5 慢**。[专项三轮报告](../opus_9020_resume_20260928/RESULTS.md)中的三轮中位数 16/16 快于外部对照、相对同批旧 9020 有 15/16 更快且几何平均耗时降低 2.7454%，只适用于该专项批次，不能替代最近完整扫描的结论。`uintptr_t` 改用现有 `u64_t` 是 RTC 类型兼容处理，不作为新增性能优化。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh)。

## 9021：128×128 的三槽预取

目标是用较小 tile 增加 workgroup 并行度、减少尾部浪费，同时在单 CTA 驻留条件下尽量隐藏访存延迟。支持 `M%64=0`、`N%128=0`、K 为 128–16384 内的 128 倍数。

最终采用方案的技术顺序：

1. 使用四 wave 的 128×128 tile，使网格比大 tile 更细。
2. 使用三个矩阵槽，提前预取 K+2，增加可与当前计算重叠的访存距离。
3. A/C 固定到 AGPR，操作数最后一次使用之后才替换；以单一 U1 runtime ring 和中点 partial wait 管理流水。三槽所有权允许省去每轮结尾的 consumer barrier。
4. 将 scale 限制在 32 组面板内循环补充，控制 LDS 用量。

保留三槽、32 组 scale 补充和 105504 字节 LDS。进入输出复用 LDS 之前仍保留最终 consumer barrier；BF16 经 LDS shuffle 后进行连续 16 字节全局写回。当前是独立的固定 traits/device body。最近 full305 中有效 295 个 shape，12 个在全部后端中胜出。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh)。

## 9022：160×128 的复用与 LDS 预算折中

目标是在适合的形状上比 M128 提高 M 方向复用，同时兼顾网格数量、尾部几何和 LDS 驻留预算。支持域与 9021 相同。

最终采用方案的技术顺序：

1. 将 M tile 增至 160，增加一次 B 读取对应的 M 方向计算量。
2. 使用两个矩阵槽，把 LDS 控制到 81184 字节、低于 80 KiB，满足两 workgroup 的 LDS 预算；实际是否双驻留仍受其他资源约束。
3. A/C 固定到 AGPR，采用单一 U1 loop、中点 partial wait 和操作数最后使用后的替换；双槽立即复用，因此保留每轮结尾的 consumer barrier。
4. 采用 32 组 scale 面板补充，保持较小存储占用。

保留四 wave、两槽、32 组 scale 补充，以及 BF16 经 LDS shuffle 后的连续 16 字节全局写回。当前是独立的固定 traits/device body。最近 full305 中有效 295 个 shape，51 个在全部后端中胜出。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh)。

## 9023：64×128 的细网格与直接输出

目标是窄 N、workgroup 数量不足，以及同步和输出重排占比较高的形状。支持 `M%64=0`、`N%128=0`、K 为 128–16384 内的 128 倍数；M64 tile 无 M 尾部。

最终采用方案的技术顺序：

1. 使用四 wave 的 M64 tile，增加网格并行度。
2. 使用三个矩阵槽、预取距离 2；等待计数跟随实际在途的未来组数，统一正常推进和 drain 路径。
3. A producer/consumer 使用 XOR 布局；两个 M-repeat 的 A scale 字节合并为 u16 LDS 表示，B scale 复制到 u32，减少 scale 处理开销。
4. 将 MFMA 输出片段直接转换为 BF16 并写到 global，省去 C-shuffle 的 LDS 读写和相关同步。
5. 当 `grid_n≤16` 且总 tile 数大于 256 时，依据几何和整除条件采用 16 或 4 个分区遍历，改善跨 tile 的数据复用。

保留三槽、64 组 scale 面板补充、80384 字节 LDS。输出是直接 `store<T::VEC_C>`，其中 `VEC_C=4`，即每次 4 个 BF16、**8 字节**全局写回。当前是独立的固定 traits/device body。最近 full305 中有效 295 个 shape，16 个在全部后端中胜出。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh)。

## 9024：64×64 的窄输出 tile

目标是进一步扩大窄问题的 workgroup 数量，补偿单个 tile 计算量下降带来的访存延迟暴露。虽然 tileN=64，B scale 仍按 N128 分组，因此 **N 仍须是 128 的倍数**；M、K 约束与 9023 相同。

最终采用方案的技术顺序：

1. 将 N tile 缩至 64，在窄输出形状上增加 workgroup。
2. 使用四个矩阵槽、预取距离 3；按真实在途组数设置等待，覆盖尾部 drain，补偿较短的每 tile 计算时间。
3. 保留 A XOR 布局和打包 A scale，以 64 组面板循环补充。
4. 使用 BF16 fragment 直接全局输出，省去 C-shuffle。
5. 在 `grid_n≤16`、`grid_n%4=0`、`16≤grid_m≤32` 时采用紧凑网格的成对 M 遍历；其他形状使用一般分区规则。

保留四 wave、四槽、64 组 scale 补充、71936 字节 LDS，以及每次 4 个 BF16、8 字节的直接全局写回。当前是独立的固定 traits/device body。最近 full305 中有效 295 个 shape，7 个在全部后端中胜出、8 个是最快 OPUS 候选；`(M,N,K)=(1536,768,7168)` 当前最快 OPUS 是 9024，但该次扫描仍比 CK 慢 0.390%。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh)。

## 9030：超过 2 GiB 输出范围的地址扩展

目标是支持 C 总字节数超过 `INT32_MAX` 的大输出；A/B buffer 范围及单个 C tile 的字节跨度仍须满足 `INT32_MAX` 限制。这首先是地址正确性和支持域扩展。

最终采用方案的技术顺序：

1. 用 `int64_t(row)*stride_c+col` 计算 C tile 基地址，避免全局输出偏移的 32 位溢出。
2. 将 C buffer extent 限制为当前有效 tile，而非整个剩余输出范围，使单 tile 访问继续使用有界资源描述符。
3. 保留满足范围限制的 A/B buffer 和 tile 内 32 位偏移，维持原有数据加载和计算路径。

保留较早的八 wave 192×256 两级/U2 流水、128 组 scale 常驻、最终 MFMA 与 BF16 LDS staging 交错、16 字节连续写回，以及高 M/窄 N 时的四 M-tile 分组。其 `SFB_BYTES=256`、`B_SCALE_PACKS=2`、LDS=143104；traits 从通用 192×256 基础配置派生，**没有继承最新 9020 的紧凑 A/XOR、8-MFMA 中点发布、打包 B scale、K0 先于 scale 面板或每次 B 预取重建描述符**。

最近 full305 对 9030 的性能证据仅覆盖新增的 **10 个大输出 shape，全部 K=1536**，10 个均选择 9030。其中 5 个有有效外部对照，耗时降低 18.2500%–19.6603%，几何平均降低 18.9883%；另 5 个无有效外部对照，不能报告外部加速比。[初始大输出报告](../opus_large_output_20260928/RESULTS.md)也只测了这 10 个，不能扩展为整个地址边界支持域的测量结论。

源码：[pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh)、[traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh)。
