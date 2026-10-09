# 当前 Opus MXFP8 候选的 Compute bound 分析与优化方法

**2026-10-08 已验证并应用：**短 K 的 9021 已定位到 startup 中的 scale 请求串行与发布 barrier；依据分层 counter、实际 ISA 和 ATT，将两侧 scale 请求提前到首次 LDS 发布前。相对 Oct7 selected 的 42 个实际赢家几何平均加速 1.576%，41 项 median 更快；两个极小 M 非赢家路径有明确代价。正式 26parent/56entry 身份及八项 API、128 次数值调用通过，已应用一个 9021 header，未提交或推送，见[第 11 节](#11-2026-10-08先定位限制再选择9021改动)。长 K 普通 tile 边界仍为后续问题。下方保留 Oct7 的历史采用范围与清单身份。

更新：2026-10-07。对象：当前 gfx950、MI355X 上的 9000 系列 MXFP8 B-preshuffle GEMM。源码审查起点提交 `b152ab83`。本文将旧 compute 优化记录与最新带宽笔记结合，说明如何确定限制、选择实验、核对机器码并决定是否保留。

**本轮已完成并应用：保留9021 prologue及9042/9053/9054三个B-scale复用alias，共两文件、四改变entry；9022全局prologue、narrow、fine和fixed_n32不采用。9021的42个实际赢家median全正，geomean加速3.155%、各shape单次时间之和加速2.313%；register六个实际赢家均5/5轮更快。最终26parent/56entry身份核对及12项正式API、176次数值/重复/guard检查通过，原checkout源码与已测selected逐文件SHA和diff一致，未提交或推送。** 201个prologue赢家、机制/异常复核及保留理由见9.1，正式模块与应用证据见9.7。2026-09-30调优数据属于已有测量；counter绝对时间窗尚未核准，支持域额外拓展保留partial，不据此声称全域收益或已唯一定位全部硬件瓶颈。

配套 [Memory bound 文档](MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md)负责 TA/TCP/UTCL1/GL2/SoC 和实际流量的进一步排查。两个文档使用同一目标：减少完成有用 GEMM 工作的时间。

## 1. 从哪一步开始

审查起点baseline的745个已选shape，其实际分支、traits、工作组数量、LDS/寄存器、逻辑工作量和历史时间已逐项核对，机器证据见 [shape_inventory.csv](shape_inventory.csv) 与 [完整清单](shape_inventory.json)。全部745行匹配本轮起点正式baseline代码对象；清单覆盖各shape历史胜出的parent，不代表本轮已测全部候选。本次应用保持host选择、grid及工作量，四个改变entry的最终身份由 [selected核验](formal_selected/identity_audit.json)单独连接到已测candidate；不把baseline inventory当作应用后的全745重测，也不由静态资源直接分类瓶颈。

遇到吞吐偏低时，先按下面的顺序收集证据，再选择改动。

| 顺序 | 要得到的结论 | 最小证据 |
|---|---|---|
| 1 | 优化哪个数学问题、哪个 shape 和实际 device 分支 | M/N/K、数据布局、parent ID、traits、源码/二进制哈希 |
| 2 | 理论工作量、逻辑数据量、实际时间的量级 | FLOPs、有效输出、tile grid、完整 kernel 耗时 |
| 3 | 是否由计算管线限制 | 实际 MFMA/矩阵管线活动、依赖和发射等待，物理访存/回压 |
| 4 | 下一条有用 MFMA 为什么被推迟 | ISA 和需要时的 ATT，关联 DS/VMEM/wait/barrier/VALU |
| 5 | 哪个修改有机会减少损失 | 一项实验、一项机制假设；资源和同步预算 |
| 6 | 源码意图是否正确落到机器码 | 实际 dispatch、native MFMA、scale selector、wait、spill/metadata |
| 7 | 是否正确、是否更快 | 机制相关正确性用例；同卡、同地址 AB/BA 完整计时 |
| 8 | 是否保留 | 独立窗口复核、受影响 shape 覆盖、无有效收益就关闭 |

算术强度高描述工作特征，不能直接证明当前 kernel 已受 MFMA 峰值限制。高 VMEM 等待也不能直接证明 HBM 带宽饱和。实际限制可能来自矩阵指令发射、依赖、局部 LDS/scale 供数、前端派发、寄存器约束或下游回压。

按最新笔记第四章，先列出四类待检验的限制。分类针对具体 shape、实际分支和测量条件；一个 kernel 可以同时有多项损失，改动后需要重新判断。

| 限制候选 | 需要的支持证据 | 优先检验的改动 |
|---|---|---|
| Compute bound | 数据已就绪，相关 CU 计算管线或指令发射持续受限；执行工作量与对应管线的可达到速率相符 | 降低无用执行量、缩短计算依赖、调整矩阵/VALU/标量的分工和发射 |
| Bandwidth bound | 某个明确接口的实际 bytes/time 接近同条件可达到带宽，并有对应队列/下游服务压力 | 减少该接口流量、提高复用或改善请求分布 |
| Latency bound | 有用工作等待返回/依赖，带宽尚未到服务上限，有效独立请求或并行工作不足 | 增加安全预取与独立工作、减少依赖和往返次数 |
| Dispatch bound | 可运行 wave 的补充或分配跟不上，结合 grid、资源、SPI/前端证据解释 CU 空闲 | 改工作粒度、grid 或资源预算，同时计算复用与归约代价 |

低 occupancy 单独不能区分 latency 与 dispatch；大量驻留但都等依赖的 wave，也不能说明请求供给充足。

## 2. 固定数学、布局和数值契约

逻辑运算为 `C = A_dequant × B_dequantᵀ`，B 的逻辑 shape 为 `[N,K]`。令 `q=floor(k/128)`：

```text
A_dequant[m,k] = decode_fp8(A[m,k]) × decode_e8m0(SFA[q,m])
B_dequant[n,k] = decode_fp8(B[n,k]) × decode_e8m0(SFB[floor(n/128),q])
C[m,n] = BF16(Σ_k A_dequant[m,k] × B_dequant[n,k])
```

| 对象 | 当前要求 | 审查重点 |
|---|---|---|
| A | 连续 FP8 `[M,K]` | 沿 K 合并；无效 M 行有界读与零填充 |
| B | 标准 `shuffle_weight(..., layout=(16,16))` 预排 FP8 | 物理 K 步进不能照搬普通 row-major |
| SFA | 原生 E8M0，连续 `[K/128,M]` | 原始字节、M-repeat 位置、尾行、面板换代 |
| SFB | 原生 E8M0，连续 `[N/128,K/128]` | N128 共享；MFMA 的 byte selector 与打包格式一致 |
| 累加 / C | FP32 / 连续 BF16 `[M,N]` | 最后转换，输出覆盖唯一且完整 |
| K 分组 | 外部 scale 粒度 K128 | 硬件 K32 子组使用重复的 K128 scale，不能更改数学契约 |
| K 划分 | 一般沿原 K 顺序；部分候选 wave-K/global split-K | 改归约顺序后需独立参考验证，不能要求与另一归约方案逐位相同 |

9000/9010/9020–9024/9030 的 kernel 参数和地址保留 batch stride；当前二维调优入口与本文私有实验 batch=1。904x/905x/906x 的 compact GEMM 实现使用二维地址，不应未经实现核对声称支持 batch。Global split-K 的 z 维表示 K 分区。

支持域须以 `opus_gemm_common.py:a8w8_mxscale_bpreshuffle_supports_shape` 和生成 launcher 的双重检查为准。9021 和 narrow 支持任意正 M，9020/9022 要求 M%16=0，9010/9030 保留 M%64=0；9000 要求完整 M/N256 tile。普通候选 A/B/C 或 partial 的有界字节数有 signed-int 限制；9030 专门将 C global base 扩展为64位，A/B 和 tile-local offsets仍有范围限制。当前9030只接受 `2×M×N > INT32_MAX` 的大输出域，不能把它当作普通大小输出上的192×256候选。

## 3. 把工作量、复用与时间预算算清楚

### 3.1 有用 FLOPs 和吞吐

```text
F_useful = 2 × M × N × K
P_useful [FLOP/s] = F_useful / t_seconds
P_useful [TFLOP/s] = 2 × M × N × K / (t_us × 10^6)
```

尾部 padding 会让实际执行 FLOPs 大于有用 FLOPs：

```text
G_M = ceil(M/B_M), G_N = ceil(N/B_N)
WG_count = G_M × G_N × global_split_K
η_tile = (M×N)/(G_M×B_M×G_N×B_N)
```

这项效率只描述边界浪费；它没有计算资源争用、cache 命中或动态派发。对 wave-K，要用实际 K128 分配计算每 wave 的动态 MFMA；空 K 分区仍需满足输出初始化/归约契约。

单个完整 K128 tile、没有 wave-K/global split-K 时：

```text
F_step = 2 × B_M × B_N × 128
A/B bytes_step = (B_M+B_N) × 128
MFMA_per_wave_step = B_M×B_N / (waves×16×16)
FP32_C_per_lane = B_M×B_N / (waves×64)
AI_tile_AB = 2×B_M×B_N/(B_M+B_N) [FLOP/B]
```

单条 `v_mfma_scale_f32_16x16x128_f8f6f4` 对一个 wave 更新16×16输出，即65536 FLOPs；每 lane 4个FP32 C、A/B fragment各32B。当前使用 `mfma_adaptor_swap_ab`，审查操作数角色应按布局和 wrapper 核对。

| 主要完整 tile | waves | MFMA/wave/K128 | FP32 C/lane | A/B B/K128 | tile内 A/B 强度 FLOP/B |
|---|---:|---:|---:|---:|---:|
| 256×256 | 4 | 64 | 256 | 65536 | 256.00 |
| 192×256 | 8 | 24 | 96 | 57344 | 219.43 |
| 128×128 | 4 | 16 | 64 | 32768 | 128.00 |
| 160×128 | 4 | 20 | 80 | 36864 | 142.22 |
| 64×128 | 4 | 8 | 32 | 24576 | 85.33 |
| 64×64 | 4 | 4 | 16 | 16384 | 64.00 |

更大 tile提高复用，同时增大 C/operand存活量并减少 grid。更深队列提高在途工作，同时增加寄存器或 LDS。它们都不是单调收益。

### 3.2 区分逻辑字节、tile重复请求和物理流量

```text
D_min = M×K + N×K + 2×M×N
        + M×(K/128) + (N/128)×(K/128)
```

这是每份输入理想只取一次、每份C只写一次的逻辑最低量，不含cache-line放大和workspace。

完整 tile工作组请求预算：

```text
D_WG_AB ≈ G_M×G_N×(B_M+B_N)×K
```

它包含不同 tile复用同一A/B的重复请求；有界尾部会减少实际请求。这个预算、tuner输出的`bw`、最低字节量都不等于HBM实测。

Global split-K=S 时，额外 FP32 workspace 约 `4×S×M×N` B；main写partial、reduce读partial、最终写BF16。在计算整体收益时同时计入两次launch和额外全局流量，不能只报告main吞吐。

Roofline起点可写为：

```text
F_exec = 当前分支实际执行的矩阵 FLOPs（含 padding）
η_work = F_useful/F_exec
t >= max(F_exec/P_compute_ceiling,
         max_interface(D_physical_interface/BW_interface_ceiling))
P_useful <= min(η_work×P_compute_ceiling,
                min_interface(AI_useful_interface×BW_interface_ceiling))
AI_useful_interface = F_useful/D_physical_interface
```

`η_work`与执行量先由实际分支/MFMA动态次数核对；只有没有额外矩阵执行时才可直接用`F_useful`作计算时间分子。MFMA FLOPs 不包含所有 VALU、地址、同步和转换成本，Roofline 给出理想下界，这些成本仍要单独定位。对每个接口记录读/写方向与实例范围；读写共享容量时才将两种 bytes 相加对同一个上限，独立方向分别比较。

`ceiling`取对应设备、同条件的可靠规格上界或独立带宽/计算校准，不能用待诊断 kernel 自己的`F/t`或`D/t`回填成上限。相同输入精度、dense/sparse语义、目标GPU、实际时钟和计算指令口径必须匹配。旧8192³的约3.30P是FP8 scaled GEMM，不是BF16输入算力。不要将论文的MI350 57.6B/cycle/CU、FIFO长度或MI300吞吐直接当作本机gfx950已验证规格。

### 3.3 结合 Little 定律和 wave 周转模型

最新带宽笔记从 `N=λT` 出发，要求统计同一种对象。对wave供给和访存请求分别统计；wave驻留多不代表在途请求多。

为避免与 GEMM 列数 N 混淆，将请求在途量写为`N_req`。同一个接口、实例范围和稳态窗口下：

```text
N_req = λ_req×T_req
BW_interface = λ_req×R_avg
N_req_required = BW_target×T_req/R_avg
```

`T_req`是请求从该接口进入到完成的平均停留时间；它与wave寿命`T_s`不同。没有队列积分或平均请求延迟时，可以用这个关系组织实验，不能把wave数量代替`N_req`完成数值反推。短 kernel 的填充、drain 和变化中的队列需要另外检查，不能默认满足稳态条件。

第三章简化模型：

```text
BW_CU ≈ W_o × n × R / (T_d+T_s)
```

其中nR为每wave请求字节，T_s为wave寿命，T_d为供给间隙。原文对W_o的定义在静态目标驻留量和实际平均活跃wave之间切换；最新笔记已指出可能重复扣除空闲。若要实际代入，在同质wave的稳态周转近似下选以下一种一致口径：

```text
W_slot = 模型中参与周转的 wave slot 数量/CU
W_active_avg = 完整统计窗口内的平均活跃 wave 数量/CU
W_active_avg ≈ W_slot×T_s/(T_s+T_d)
BW_CU ≈ W_slot×nR/(T_s+T_d) ≈ W_active_avg×nR/T_s
```

`W_slot`仍不能简单取硬件最大值，必须受kernel资源、grid和分配约束。两种写法都是简化供给预算；工作量/寿命异质、有限grid、请求在wave内分布不均时，最终以相应接口真实bytes/墙钟时间为准。该式帮助组织假设，不把论文通路常量强行代入当前GPU。

Compute分析关心T_s中有用计算、数据依赖、issue等待、barrier和收尾分别占多少。提高tile复用或并发可能同时增加T_s却提高总吞吐，所以不能以最短wave寿命作为单独目标。

### 3.4 前端供给和固定开销：解释小grid与短K

带宽笔记3.1在原文前端模型下给出：

```text
C_p = #XCD × C
C_s = (C/#SPI) × WG × 4
原文 #SPI=4 时：C_s/C_p = WG/#XCD

T_p = #TotalCUs × W_o / WG
ρ_wave = T_s/T_p
```

这里C为chunk内工作组数，WG为每工作组wave数；C_p描述CPC再次拿到chunk的周期，C_s描述SPI用完它的周期。T_p沿用原文派发假设，表示同一slot获得下一wave的整圈轮转周期，不是wave结束后的纯空闲T_d；ρ_wave也不是实测occupancy。C_s<C_p、T_s<T_p分别提示chunk供给或slot轮转跟不上。当前gfx950要先核对前端与周期参数，不能只把4wave/8wave代入旧文常数便宣布某个kernel已经dispatch bound。

当前调优中更直接的起点是实际grid和静态可驻留WG：

```text
round_budget ≈ ceil(WG_count / (#CUs × resident_WG_per_CU))
```

这是均匀分配的调度预算，实际XCD/CU分布与尾部要用计数器核对。wave-K改变工作组内部并行而不增加grid；global split-K增加grid，但也增加归约、流量和launch。更小tile增加grid，同时减少复用并增大重复请求。

对同一M/N与相同device分支，可用多个K的完整计时检查 `t(K)≈t_fixed+α×K` 是否成立，帮助区分prologue/output固定成本和稳态计算成本。跨K专化、scale面板换代或cache状态变化时，不能把所有点拟合成一条线。Amdahl预算也应按时间：若拟改部分占比f、该部分预计加速s，则总加速上限为 `1/((1-f)+f/s)`；静态指令数比例不能替代f。

## 4. 当前26个默认候选：按实际分支分析

### 4.1 原始大 tile / 中 tile / narrow / 大输出

当前公开registry在 `csrc/opus_gemm/opus_gemm_common.py:1863`。下面是当前源码事实；LDS是分配预算，驻留仍需实际metadata与设备查询。

| ID / 分支 | tile / waves | 矩阵slots / scale panel | LDS B | 当前供数/输出策略 |
|---|---|---|---:|---|
| 9000 | 256×256 / 4 | 2 / 64 | 152064 | A/B AGPR pin，混合C，packed scale，LDS重排16B输出 |
| 9010 | 256×256 / 4 | 2 / 64 | 152064 | 独立padded-M主体，保留9000节奏，有界A/C |
| 9020默认 | 192×256 / 8 | 2 / 128 | 143360 | 默认MMA vtype，midpoint发布，packed B scale，末轮MFMA/BF16交错 |
| 9020短/固定K | 192×256 / 8 | 2 / 8,32,64 | 119840/124544/130816 | 同pipeline，不同scale面板和固定K展开 |
| 9020高M窄N | 128×128 / 8 | 2 / 64 | 76032 | 改细grid，减少调度round |
| 9021 | 128×128 / 4 | 3 / 32 | 105504 | 默认MMA vtype，逐fragment换代，midpoint partial wait，LDS重排16B输出 |
| 9022 | 160×128 / 4 | 2 / 32 | 81184 | 默认MMA vtype，保留每轮consumer barrier，LDS预算可容两WG |
| 9023默认 | 64×128 / 4 | 3 / 32 | 78112 | XOR A，packed u16 A scale，B byte，8B直写 |
| 9023特定K7168 | 64×128 / 4 | 4 / 64 | 105536 | 更深预取；LDS单项只能一WG |
| 9024默认 | 64×64 / 4 | 4 / 32 | 69664 | XOR A，packed u16 A scale，B byte，8B直写 |
| 9024特定K7168 | 64×64 / 4 | 4 / 64 | 71744 | 窄grid使用固定K与group4遍历 |
| 9030 | 192×256 / 8 | 2 / 128 | 143360 | C64位base，packed B scale，保留旧end-wait U2与末轮输出交错 |

**版本纠偏：** 旧README/HANDOFF及20260928总结仍出现“9021/9022 A/C pin”“9030 byte SFB/143104B”“9023/9024 64scale/80384/71936B”。这些是历史状态。当前9021/9022没有`amdgpu_pin_agpr`；9030的SFB为512B、LDS143360B；narrow默认scale32。AGPR的最终自然分配仍应读新二进制，不能由源码类型推断。

9020/9023/9024实际分支在 `codegen/gen_instances_gfx950.py:3025`：

- 9020：高M窄N条件先选择8wave128×128；其他`m>=1024`的K384/768使用scale8，K1536/3072用scale32，K7168用scale64；其他使用runtime-K默认面板。
- 9023：`M>=1024,N<=1024,K=7168`且64×128 grid≤256时，用4slots/scale64。
- 9024：`1024<=M<=2048,N<=1024,K=7168`时，用scale64/fixedK/group4。

CSV的parent ID和public kernelName不足以确定实际几何、ISA、LDS或K展开。

### 4.2 Register queue：9040–9042、9051–9054

共用 `...small_register_gfx950.cuh`：B预排fragment直接喂MFMA，矩阵不经过LDS；prefetch队列保存A/B/scale，多个wave沿K分区时用LDS汇总FP32再转换一次BF16。

| 默认ID | 基本tile | wave-K | queue | 主要计算侧取舍 |
|---|---|---:|---:|---|
| 9040 | 16×32，1wave | 1 | 6 | 小grid、直接供数，较深VGPR队列 |
| 9041 | 16×16，8wave | 8 | 2；条件下3 | K分区并行与FP32工作组归约 |
| 9042 | 32×32，4wave | 4 | 3；K7168有N48固定分支 | 增加N利用，mask N16 tail，分布式归约 |
| 9051 | 16×32，4wave | 4 | 3 | 四K分区、较深队列 |
| 9052 | 16×32，8wave | 8 | 2；K7168为3 | 更细K并行、一次最终BF16 |
| 9053 | 32×32，8wave | 8 | 2；K7168为4wave/queue4 | 形状决定并行和VGPR存活量 |
| 9054 | 32×64，4wave | 4 | 2 | 更大输出、K分区复用 |

重点不是单独看wave数量。检查每wave有效K128数量、空partition、queue饱和、VGPR分配粒度、归约后哪个wave写C。默认runtime循环限制展开以控制寄存器；fixed-K路径完全展开可能改变ISA尺寸和资源。

`OUTPUT=4`分布式归约让每个最终fragment由一个K-wave负责，全部partial留在LDS，最后仅一次BF16转换；普通wave-K路径由wk0读取其他wave的FP32结果。两种路径归约顺序与输出范围须独立验证。

本轮已验证9042实际K7168分支的`fixed_n32`单项实验：9071 M16/N48→N32、9073 M32/N48→N32，保持4个K-wave、prefetch4/3、Output4和原cache策略，显式关闭B-scale复用。对应VGPR168→152、185→143，归约LDS12288→8192B、24576→16384B；目标M192/M384、N768的grid由192→288。它减少N-tail和每WG存储，同时增加A重复请求估计约50%；两目标共享51地址池、5轮交替Event确认均变慢，时间分别增加45.20%和46.54%，**拒绝采用，保留现有N48分支**。尚无counter证据，不能把退步归因于某个具体接口；详情见配套Memory文档6.5、8.2及 [fixed_n32/variants.json](fixed_n32/variants.json)。

独立B-scale复用实验已收敛并应用为仅9042/9053/9054三个alias显式true，默认false。它有效改变baseline745集合的全部6个实际赢家，6/6已有干净共享池5轮Event，每项5/5更快；原实验、scoped库及最终selected入口的指令、完整metadata和归一化descriptor相同，因此复用已有计时。5个新机制case、6个已观察异常Event及最终12项正式API核验均完成，限定方案与9021合并为两文件改动。可选646项支持域扫测停止于210项；两个非赢家M8/N16384/K1536路径时间增加0.81%/0.64%，保留该有限代价。完整受影响赢家交集与remaining=[]见[当前赢家覆盖](register_reuse_scoped/current_winner_coverage.json)，机制/异常复核见[register收尾](results/register_minimal_closing_analysis.json)、[应用核验](formal_selected/integration_manifest.json)及Memory8.6。

### 4.3 LDS queue与cluster：9043–9047、9049、9055

共用 `...small_lds_gfx950.cuh`。实际动态LDS：

```text
L = ceil((K/128)/global_split_K)
active_slots = min(L, NUM_STAGES)
LDS_launch = active_slots×(A_STAGE+B_STAGE)
             + (REGISTER_SCALES ? 0 : (B_M+B_GROUPS)×L)
```

各K分区的loops可能少于L，launcher为最大分区预算。`.group_segment_fixed_size=0`不表示没有LDS，必须同时记录launch动态字节。

| 默认ID | 基本tile / waves | slots / cluster | 计算侧重点 |
|---|---|---|---|
| 9043 | 32×64 / 4 | 8 / 2 | 两K128共用一次同步，wave内打包C |
| 9044 | 64×64 / 4 | 4 / 1；条件下8 / 2 | 深队列改善一round grid，LDS增加 |
| 9045 | 96×64 / 4 | 4 / 1 | M复用，read-only drain，C经LDS重排 |
| 9046 | 64×128 / 8 | 6 / 2 | 八wave分工，cluster2，动态LDS |
| 9047 | 32×64 / 4 | 4 / 1 | register scales，减少scale LDS成本 |
| 9049 | 32×128 / 4 | 4 / 2 | XOR LDS、register scales、预取先于读 |
| 9055 | 32×64 / 4 | 12 / 4 | 更长queue，四K128共用一次同步 |

cluster越大，同步频率越低，但需要更长预取距离和更多storage，可能降低驻留。检查每组最后消费者是否已经读完、partial tail待完成VMEM数量、read-only drain是否真正不再覆盖旧槽。

### 4.4 Fine-M / Global split-K：9060–9063

共用small-LDS主体，traits在 `...fine_gfx950.cuh`。M48/M80/M112需要最后一次A copy只有部分wave参与；默认N128、K128 tile。

| 默认ID | 基本tile / waves | 固定global split-K | 作用 |
|---|---|---:|---|
| 9060 | 80×128 / 4 | 1 | 避免M padding造成grid round变化 |
| 9061 | 96×128 / 8 | 1 | 更多wave分担同tile的C与供数 |
| 9062 | M80或M96×128 / 4或8 | 2 | 增加grid，两个FP32 partition |
| 9063 | M48/M80/M96/M112/M128×128 | 4 | 细M tile与四K分区匹配不同grid |

调优CSV的`splitK=0`可能仍表示ID内部固定split-K=2/4，不能因此报告“无split-K”。S>1时生成入口先写FP32 workspace再launch reduction；完整耗时必须包括二者。分区采用floor/remainder均衡K128，并覆盖空partition的零partial。

### 4.5 Parent dispatch的具体条件

当前26默认ID合并13个兼容ID，条件在 `opus_gemm_common.py:1942`。每条以first-match顺序选择，未命中回到runtime fallback：

| Parent ID | 条件 / 实际配置 |
|---|---|
| 9041 | K>=8192，或K>=4096且M16/N16 grid<256 → 原9050 queue3；否则queue2 |
| 9042 | K7168且M16/N48 grid<=256 → 原9071 M16/N48/4wave/queue4；其余K7168 → 原9073 M32/N48/4wave/queue3；其他K → M32/N32 fallback |
| 9044 | K>=1536且M64/N64 grid<=256 → 原9056 slots8/cluster2；否则slots4 |
| 9052 | K7168 → 原9070 queue3、distributed FP32 reduction；否则runtime fallback |
| 9053 | K7168 → 原9072 M32/N32/4wave/queue4、distributed reduction；否则8wave fallback |
| 9062 | ceil(M/96)<ceil(M/80) → M96；否则M80；两者都是S=2 |
| 9063 | K>=8192时：M<=48→M48；80<M<=96→M96/4wave；96<M<=112→M112；112<M<=128→M128；其他条件且M80 grid×4>256→M96/8wave；否则M80 |

9060另有K3072/7168固定编译分支；9062/9063包含K16384分支；9063的M48配置包含K7168专化，但parent当前长K条件是否可选中应按实际launcher核对。Compatibility ID可显式调用，但不参加默认26候选互选。不得把“同parent ID”理解为“同device binary”。

### 4.6 审查起点正式baseline资源：8个parent、16个device variants

从审查起点正式baseline JIT对象的gfx950 HIP bundle读取metadata和kernel字节，未启动GPU。完整symbol、generated TU/traits、metadata、指令SHA和descriptor在 [official_compute_metadata.json](official_compute_metadata.json)。下表VGPR为总向量分配口径，AGPR已包含其中；所有行private bytes/VGPR spill为0。应用后的四改变entry与baseline资源/身份关系见9.7的selected核验，不将本表当作应用后的另一次测量。

| ID / 实际traits | VGPR total | AGPR | SGPR | SGPR spill slots | LDS B | kernel指令B |
|---|---:|---:|---:|---:|---:|---:|
| 9000 | 477 | 221 | 70 | 0 | 152064 | 20744 |
| 9010 | 497 | 241 | 80 | 0 | 152064 | 23840 |
| 9020 192×256 scale128/runtime | 203 | 0 | 55 | 0 | 143360 | 8784 |
| 9020 128×128 scale64/runtime | 96 | 0 | 51 | 0 | 76032 | 4740 |
| 9020 192×256 scale8/K384 | 216 | 0 | 54 | 0 | 119840 | 6284 |
| 9020 192×256 scale8/K768 | 204 | 0 | 54 | 0 | 119840 | 6880 |
| 9020 192×256 scale32/K1536 | 204 | 0 | 54 | 0 | 124544 | 6912 |
| 9020 192×256 scale32/K3072 | 204 | 0 | 54 | 0 | 124544 | 6912 |
| 9020 192×256 scale64/K7168 | 204 | 0 | 53 | 0 | 130816 | 7300 |
| 9021 runtime | 164 | 0 | 96 | 0 | 105504 | 8316 |
| 9022 runtime | 194 | 0 | 66 | 0 | 81184 | 6888 |
| 9023 slots3/scale32/runtime | 232 | 0 | 106 | 46 | 78112 | 25976 |
| 9023 slots4/scale64/K7168 | 176 | 0 | 59 | 0 | 105536 | 8340 |
| 9024 scale32/runtime | 152 | 0 | 106 | 44 | 69664 | 21736 |
| 9024 scale64/K7168/group4 | 110 | 0 | 58 | 0 | 71744 | 7276 |
| 9030 | 203 | 0 | 50 | 0 | 143360 | 7796 |

当前9000/9010的477/221、497/241已经存在于20260928集成记录；本次重建未证明新的寄存器回退。旧464/212对应冻结standalone源码和binary，而旧目录当前`tmpl_generic.hpp`后来已修改，不能替代冻结身份。两版10处显式pin表达式相同，但SFB发布/byte selector、`readfirstlane`表达、OPUS include及JIT flags不同：旧SFB复制byte并按`n_repeat`选择，当前仅保留低byte并固定B selector0；两者面板仍512B。当前编译器实际`--version`也报告49c4188968…，不能根据目录名断定切换compiler。额外自然AGPR分配的具体因果尚未被隔离，详见JSON的`historical_resource_comparison`。

Descriptor按8个寄存器分配：旧464→464、当前477→480、497→504；AGPR起点分别252、256、256。三者在512总向量寄存器/SIMD预算下都只能驻留1 wave/SIMD，152064B LDS在160KiB预算下也都只容1 WG/CU，未跨驻留档位。这是静态上限，不能替代动态active-wave或性能测量。

**runtime9023/9024的SGPR spill通过`v_writelane_b32`保存到VGPR lane、`v_readlane_b32`取回；ISA没有scratch load/store。** 这增加lane move与依赖/资源成本，但不能报告成HBM spill流量，也不能因private=0便报告全部零spill。

资源表描述CPU重建产物，当前驻留与吞吐仍须设备查询和动态计数；不能用它冒充2026-09-30旧binary的metadata。

## 5. 怎样证明 Compute bound，并定位下一条MFMA的延迟

第四章4.1的四类perfmon各自承担一步：**Clocking**核对时间窗口和模块周期，**Workload Characterization**核对动态指令/请求/执行量，**IP Architecture**观察接口带宽、命中和回压，**IP Microarchitecture**拆队列、地址转换、资源与发射停顿。先确认前三类口径，微架构事件才能用于定位根因；事件名称和高计数本身不会证明某项修改处在完整任务的关键路径。

### 5.1 先验证实际接口未先饱和

按最新笔记第四章：先看完整吞吐和GL2↔SoC实际流量，再核对GL2/TCP/TA/UTCL1回压。如果物理流量接近平台当前可达到的接口上限，或下游stall传回CU，应使用Memory文档继续定位。

计数器名称、单位、聚合范围以本机gfx950实际支持为准。纸面`TCC_EA_*_BW`名称可能代表字节，仍需除以时间。论文`SQ_WAVE_LIFE`和`SQ_WAIT_INST_VMEM`不能直接作为当前rocprofv3的可采集名称。

本机 `/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml` 的gfx950定义已做CPU核对。当前已采集9000、9052实际9070、9042实际9071/9073四组干净counter，各51个选定dispatch；下表给出定义口径，是否可据此归因还需实际时间窗核验：

| 信息 | 当前raw/derived counter与计算 | 解释条件 |
|---|---|---|
| 平均wave寿命T_s | `4×ΣSQ_WAVE_CYCLES / ΣSQ_WAVES` | wavecycles单位quad-cycles，×4换cycle；两者返回per-SE聚合，范围与时间窗匹配 |
| 等待 / issue / active比例估计 | `ΣSQ_WAIT_ANY / ΣSQ_WAVE_CYCLES`；`ΣSQ_WAIT_INST_ANY / ΣSQ_WAVE_CYCLES`；`ΣSQ_ACTIVE_INST_ANY / ΣSQ_WAVE_CYCLES` | 同为quad单位；当前204个样本三项和都为100%，仍不能将多个wave重叠等待直接从墙钟时间相减；ANY不能替代VMEM细分等待 |
| 矩阵管线忙碌 | `MfmaUtil = 100×ΣSQ_VALU_MFMA_BUSY_CYCLES /(max(GRBM_GUI_ACTIVE)×SIMD_NUM)` | `SIMD_NUM=simd_count`，不能用SE/CU数替代；本轮GRBM时间窗未核准，此公式值不能证明管线利用率或饱和 |
| 平均驻留waves/CU | `4×ΣSQ_WAVE_CYCLES /(max(GRBM_GUI_ACTIVE)×CU_NUM)` | 同范围、GUI有效周期窗；`CU_NUM=simd_count/simd_per_cu`；不要自动将工具occupancy的固定32-wave分母当成本机驻留上限 |
| 动态F8计算量交叉核对 | `512×ΣSQ_INSTS_VALU_MFMA_MOPS_F8` | 定义为F8 add/mul ops除512；先与当前scaled MFMA动态次数对照，再用于FLOP/s；含padding执行量，与`2MNK`有用FLOPs分开 |
| LDS等待与冲突线索 | `SQ_WAIT_INST_LDS`、`SQ_LDS_BANK_CONFLICT` | 与操作数读取、scale读、输出重排和归约路径对应；单个高计数不能独立证明根因 |

`SQ_INSTS_MFMA`还可核对动态指令数量。不要只看VALU指标判断GEMM计算瓶颈，应覆盖MFMA/矩阵管线及其发射等待。定义文件中存在事件只证明工具有该定义，实际采集还需核对目标agent、组合限制和结果完整性。

四组共204个dispatch的实际正式module SHA、物理PCI、grid、waves和F8 issued工作量全部一致。每个样本的`SQ_VALU_MFMA_BUSY_CYCLES`恰为动态scaled MFMA数的32倍，现有busy计数主要与指令工作量一致，单独提供的独立饱和信息有限。按SDK定义使用`max(GRBM_GUI_ACTIVE)`，却由`GRBM_COUNT / profiled_duration_ns`得到3.34–6.14GHz，超过该agent记录的2.4GHz最大engine clock；原始GRBM实例也不在当前scalar CSV中。因此先保留工作量、waves及同范围SQ比例，暂不使用这四组的9.34%/1.23%/2.28%/3.68%“MFMA利用率”或occupancy值分类bound，也不自行按固定因子修正。核验见 [counter_runtime_consistency_audit.json](profiles/counter_runtime_consistency_audit.json)。9021 prologue 后续已有单个真实赢家的独立 A/B counter，范围与解释见 5.5；没有由这一目标覆盖全部 prologue 路径。

### 5.2 将wave时间分为可定位的部分

| 观测 | 可检验的假设 | 下一步 |
|---|---|---|
| 矩阵pipeline活动高、数据就绪，MFMA发射竞争高 | 计算/issue throughput限制 | 比较tile、wave分工和动态MFMA；验证对应精度的峰值 |
| MFMA之间DS/wait长、HBM未满 | LDS/scale/operand供数限制 | 按fragment最后使用安排load，核对bank/packing |
| VMEM dependency长、物理带宽低 | 请求不足或latency限制 | 队列深度、有效并发、grid、回压来源 |
| barrier等待高 | wave路径不均或槽位发布太晚 | 逐wave生产/消费时间；保持读者退休规则 |
| 静态能多WG，实际active waves偏低 | grid不足、派发或尾部不均 | grid round、wave-K/split-K代价、SPI资源stall |
| 初始化/输出占比高 | 固定开销限制 | 短K剖分、完整Amdahl预算 |

高等待率只是线索，不能精确归因到尚未实测的某一级。

### 5.3 ISA/ATT审查步骤

1. 固定同一个展开副本中的两条相邻有用MFMA PC，列出中间实际执行的SALU/VALU/DS/VMEM/wait/barrier。
2. 核对操作数就绪、SrcC/Dst依赖、scale selector、AGPR/VGPR搬运与排队。
3. 覆盖同一K128的所有M/N repeats，避免只改善一个窗口却推迟下一窗口。
4. 汇总两个或整个queue周期的时间；ATT局部shader clocks不能代替整核ms。
5. 最后用无profiler完整计时决定采用。Counter/ATT扰动单列，不混入最终性能。

动态指令与静态指令区分：单个wave动态MFMA取决于K128循环次数和实际分支。9000每K12864条、K8192每wave4096条；整份反汇编静态MFMA包含多个drain与loop副本，不能互相作分母。

### 5.4 本轮真实代表：先确认执行量，再解释供数

已在同一正式module、MI355X/gfx950 PCI`0000:85:00.0`采集9000的8192³和9051的`1×65536×1536`，每个独立PMC pass取51个选定dispatch。当前7组干净：9000的`sq_ea/utcl1_credits/ta_lds`及9051全部四组；9000首次`l2_tagmap`中断记录排除；该可选重测已停止，保留缺失范围。因此四组主PMC都已经有runtime接受证据，未覆盖的目标/可选组仍分别记录。完整数字、来源和复算脚本在 [characterization_counter_summary.json](char_profiles/characterization_counter_summary.json)。

| sq_ea观测 | 9000，8192³ | 9051，1×65536×1536 |
| --- | ---: | ---: |
| 工作组 / wave | 1024 / 4096 | 2048 / 8192 |
| 动态scaled MFMA | 16,777,216 | 49,152 |
| 有用 / 执行F8 FLOPs | 1,099,511,627,776 / 同左 | 201,326,592 / 3,221,225,472 |
| profile时间中位数 | 375.603 μs | 16.921 μs |
| 有用吞吐 | 2.927 PFLOP/s | 11.898 TFLOP/s |
| DRAM总bytes / profile带宽 | 941.990 MB / 2.508 TB/s | 100.887 MB / 5.962 TB/s |
| wave ANY wait / issue / active | 26.03% / 45.15% / 28.81% | 30.09% / 61.26% / 8.65% |

9000完整tile的执行量精确等于`2MNK`，其大tile复用把外部字节摊到更多有用计算；9051 BM16/M1的执行量为逻辑的16倍，先有明确padding浪费，再看访存请求供给。两者F8counter、wave和grid均逐dispatch匹配。profile吞吐只用于本轮诊断；与旧3.30P记录设备/窗口/工具不同，不直接认定性能退步。

独立`ta_lds` pass中，9000有4,194,304条direct-LDS buffer-read事件与18,747,392单位LDS bank-conflict计数；9051 direct-LDS和bank-conflict均为0。独立`utcl1_credits` pass的in-flight-max/gate约1.20%与6.18%；`sq_ea`的LFIFO/gate为0%与24.03%。这些支持9000继续查矩阵/scale/LDS与issue节奏，9051继续查缓存miss、LFIFO和翻译容量。它们来自不同subprocess，不能拼成同一次wave延迟分解。prologue自身的单目标A/B counter另见5.5；正式9000与9021是不同实现，不能借9000结果证明prologue优化因果。

9000长K的GRBM隐含周期约2.402GHz接近agent2.4GHz，9051约3.57–3.64GHz仍不一致；绝对MFMA利用率/occupancy不据此分类，也不做固定因子校正。9051的unwindowed SQ地址FIFO计数除SQ_BUSY得到443.64%，scope不等价，不能读作时间百分比。GL2/TA/UTCL1其它观测及限制详见配套Memory文档5.8。代表实测确认工作量和不同请求路径，当前具体compute/带宽/延迟瓶颈仍须相应时钟、回压、动态并发和受控修改支持。

### 5.5 Prologue A/B：工作量不变时，wave 计数与队列节奏怎样变化

9021 的真实赢家 `480×7168×384` 已完成 baseline/candidate 各自的 `sq_ea`、`ta_lds`，四个独立 pass 均干净。与 register 代表合计为 8 个 pass、408 个选定 dispatch。该批使用 MI355X/gfx950、物理 PCI `0000:15:00.0`、HIP visible3；每一对 A/B 在同卡串行执行，使用同 seed 和各自自动轮换的51组地址。前面的共享池 Event 来自 PCI `0000:85:00.0`，因此本节是另一张卡上的机制线索，**不把 profile 耗时变化计作采用收益，也不与 Event 数字拼成同一实验**。完整来源、claim、计划/runner/库 SHA 与 CPU 复算见 [A/B counter 汇总](ab_profiles/ab_mechanism_counter_summary.json)、[复核脚本](analyze_ab_mechanism_profiles.py) 和 [8项队列](ab_mechanism_profile_queue.json)。

两个 `sq_ea` pass 的全部102个选定 dispatch 都精确为224 WG、896 waves、43,008条动态 scaled MFMA，执行 F8 FLOPs 为2,818,572,288。BM128将M480覆盖到M512，执行量为有用 `2MNK=2,642,411,520` 的16/15；这个 padding 在A/B之间相同。两个 `ta_lds` pass 的direct-LDS与总buffer read事件也逐dispatch相同，先确认改动没有减少矩阵工作或删掉请求。

| 观测 / 独立来源 | baseline → candidate 中位数 | 可支持的解释 |
| --- | ---: | --- |
| 平均 wave life / sq_ea | 14,420.7 → 13,769.2 cycles，−4.52% | 同范围 `4Σwavecycles/Σwaves` 估计下降；不是墙钟时间减少4.52% |
| 累计 ANY wait / sq_ea | 2,089,158 → 1,981,077 quad-cycles，−5.17% | wave等待计数较少，ANY没有给出K0、scale、VMEM或barrier的单独分解 |
| 累计 issue wait / sq_ea | 392,181 → 355,376 quad-cycles，−9.38% | 与供数/发射节奏变化相容，尚未定位具体指令PC |
| DRAM读 / 写 / sq_ea | 4.3064 → 4.3151 MB；写均6.8813 MB | 物理字节没有下降；总字节变化约+0.078%，不解释成省HBM流量 |
| direct-LDS / 全buffer read / ta_lds | 均21,504 / 21,952事件 | 请求工作量不变，优化假设仍是排序和重叠 |
| LDS bank-conflict / LDS issue wait / ta_lds | conflict均172,032；wait 58,560 → 55,725 | conflict计数未变；不声称收益来自解决LDS bank冲突 |
| SQ地址FIFO full原始计数 / ta_lds | 52,006 → 12,035 | 地址队列节奏有变化线索；未严格windowing，不能换算节省时间或宣布根因 |
| TCP TCR原始stall / sq_ea | 84,509 → 117,504 | 下游回压没有统一下降；不能把所有stall概括为改善 |

这些观测支持继续检验“相同请求量下减少起步和issue等待”的源码/ISA假设，没有直接证明K0与scale重叠了多少时间。`sq_ea`和`ta_lds`来自不同subprocess，表中并排的是各自中位数，不能拼接成一个wave的延迟分解。寄存器复用目标的buffer请求数变化另见Memory文档5.9。

本批8个pass的 `GRBM_COUNT/profile duration` 中位数为4.57–5.06GHz，仍超过agent的2.4GHz最大engine clock。保留匹配范围的wave计数和原始事件，不计算绝对MFMA利用率/occupancy、不按固定因子修正，也不把wave-cycle差额换算成微秒。每个目标/label/group只有一次采集，51次dispatch是同一pass内样本；现有counter趋势不等于独立重复确认或完整采用域结论。前面的干净共享池Event继续作为其已测shape的性能证据；本轮必要机制、当前赢家完整Event与已观察异常复核分别记录，额外支持域保留partial，正式candidate15项入口验证已通过。

## 6. 当前源码的关键优化入口和同步风险

| 入口 | 当前机制 | 改动风险 / 判断 |
|---|---|---|
| 9000 `...4wave_gfx950.cuh:18`、`:435` | A/B、选定C固定AGPR；区间随最后使用换代 | pin改变全局RA，必须检查alias/旧fragment存活/copy/spill |
| 9000 `:668`–`:676` | C00[0..4]→wait→C00[5]→barrier | sched builtin不等于硬件barrier；删除可前移或拆分wait |
| 9000 `:1062` | AGPR/BF16/LDS/16B global输出 | 减少LDS中转可能损害全局写合并 |
| 9010 `...padded_m...:174` | inactive SFA每pass先清零 | 去掉定义可能扩大live range并重新spill |
| 9021 `...128x128...:269`、9022 `...160x128...:142` | byte SFA读后shift/or，每轮准备下一scale | producer预转置有潜力，成本须包含面板加载/transpose |
| 9021 `:304` | 已应用K0→scale→K1→partial wait；loops1仍完整等待 | 起步顺序与wait已验证，主循环发布/退休保留 |
| 9022 `:177` | 保留scale→K0→K1→完整等待 | 本轮全局prologue实验不采用 |
| 9021 `:327`、9022 `:194` | midpoint partial wait与逐fragment最后使用后替换 | 不能读未发布K+1，也不能覆盖别的wave还在读的旧slot |
| 9022 `:230` | 双槽末尾consumer barrier | 与三槽9021不同，不能直接删 |
| 9020 `...8wave_192x256...:350` | 前8MFMA→wait/publish/retire→K+2 | 保留next-slot发布和旧slot退休顺序 |
| 9030 `...large_output...:208` | 开头K+2，末尾wait/barrier | midpoint移植值得验证，C64位地址和bounds单独保留 |
| small register `:53`、`:95` | operand/scale queue、runtime限制展开 | 增queue增加VGPR，wave-K归约依赖/空wave不能破坏 |
| small LDS `:133`、`:227` | group partial wait、cluster同步、read-only drain | tail在途数量与steady-state不同，必须跟真实requests |
| small LDS `:250`、`:289` | FP32 partial写与最终reduce | 单测main不能代表global split-K完整收益 |

`sched_barrier`和`sched_group_barrier`约束编译器机器指令调度，本身不是运行时工作组汇合。`s_waitcnt`等待指定类型操作完成；`s_barrier`汇合wave。三种机制分开审查。Mask意义以所用LLVM的AMDGPUUsage定义为准，不能仅复制数字。

静态驻留预算：

```text
WG_static_limit = min(WG_LDS, WG_VGPR/AGPR, WG_SGPR,
                      WG_wave_slots, WG_other_limits)
```

须包含分配粒度、4wave/8wave分工和实际dynamic LDS。`__launch_bounds__(...,1)`不表示最多一WG，VGPR计数减少但未跨资源阈值也不自动提高并发。在当前gfx950代码对象中，`.vgpr_count`包含AGPR对应的总向量分配，不能再加`.agpr_count`计算一次总量；同时核对kernel descriptor、AGPR起点与实际occupancy查询。正常`v_accvgpr_read/write`不等同于scratch spill。

## 7. 已有测量：当前覆盖与历史经验

### 7.1 最新2026-09-30覆盖

来源 [745-shape报告](../opus_current745_tables_20260930/REPORT.md)。612个M<=2048 shape来自consolidate本轮、133个更大M来自full745_register上轮；每个shape使用各自同轮外部最优。当前八个原始/中tile候选pipeline与traits SHA同时匹配这两个run manifest。

- 当前26候选的最快Opus在725/745 shape快于同轮外部最优，几何平均加速1.1568×。
- 9000 129个Opus内胜出、全部快于外部；9010 38/38；9021 42/42；9022 159/159。
- 9020 76个Opus内胜出、73个快于外部；三个落后项约0.19%、1.12%、0.29%，单次sweep不足以证明稳定缺口。
- 已有306项测试通过、2项Graph未运行；612-shape调优11500条Opus测量全部有效。

这些证明已测实现的覆盖和时间，不证明MFMA、TA、L1或HBM是哪项实际瓶颈。

| 当前代表shape M×N×K | ID | 已测Opus us | 同轮外部us | 来源 |
|---|---:|---:|---:|---|
| 1792×7168×384 | 9000 | 13.0965 | 18.1456 | consolidate |
| 512×6144×7168 | 9021 | 38.4765 | 44.7491 | consolidate |
| 608×7168×7168 | 9022 | 48.8994 | 66.5876 | consolidate |
| 1536×768×7168 | 9024 | 23.9472 | 26.2289 | consolidate |
| 65536×7168×16384 | 9000 | 5326.1284 | 6992.4292 | full745_register |
| 65536×65536×1536 | 9030 | 8074.9334 | 13531.9413 | full745_register |

例如65536×7168×16384有用吞吐约2.8901 PFLOP/s，与旧8192³约3.30P问题规模/设备/环境不同，不能直接算成当前退步。

### 7.2 旧compute记录的成功与失败

旧参考 [Compute-bound GEMM从设计到优化](/root/workspace/gcnasm_new/gcnasm-mxfp8-final-pipeline-source-20260910/opus_gemm/mxfp8_gemm_16x16x128_blockscale_bpreshuffle_4wave/COMPUTE_BOUND_GEMM_DESIGN_AND_OPTIMIZATION_20260922.md)记录了同一历史严格消融：

| 历史阶段 | PFLOP/s | 相对上一阶段 |
|---|---:|---:|
| 串行、重复scale读取 | 1.189794 | — |
| packed scale与op_sel | 2.013928 | +69.2670% |
| 软件流水线，关闭pin | 3.187261 | +58.2609% |
| 该流水线上AGPR pin | 3.278888 | +2.8748% |

主要收益来自结构设计和数据复用。历史pin版本的C布局与当前混合布局不同，不能把阶段收益当作今天再加pin的潜力。

旧成熟4wave版本局部实验：

| 实验 | 同轮/配对结果 | 本轮启示 |
|---|---|---|
| 删除主循环两类sched barrier | 3.300942→2.951505P | “源码里看似无指令”可能维护关键wait位置 |
| 首轮零SrcC | 中位数−0.475447% | 少初始化仍可能新增安装/保存边界成本 |
| C00提前BF16转换 | 自然调度未建立收益；穿插方案退步 | 提前处理需真正推进关键输出 |
| 最后矩阵prefetch后移 | 未建立稳定收益 | 局部长尾改善可能不改变完整周期 |
| 未来地址递增 | 未建立稳定收益 | 少乘法可能改变RA/scale调度 |
| 全部或逐象限8B直写 | −12.404528% / −9.453355% | LDS重排和16B写合并有实际作用 |
| 两LDS输出槽 | 未建立收益 | 地址范围缩小未减少总LDS申请 |
| 旧8wave结构实验 | −4.445172%，0/5胜出 | 增wave并非单调收益 |

这些结果关闭对应实验分支；不能外推到所有新shape、small tile或全部output设计。

## 8. 本轮私有实验：9021/9022 Prologue

证据在 [compute_prologue/isa_metadata_comparison.json](compute_prologue/isa_metadata_comparison.json)、[源码差异](compute_prologue/changes.diff)、[编译参数](compute_prologue/build_manifest.json)。库为`baseline/experiments.so`和`candidate/experiments.so`。

只修改两个私有pipeline的prologue：

```text
baseline:  scale producer → K0 → K1 → vmcnt(0)
candidate: K0 → sched_barrier → scale producer → sched_barrier
           → K1 → sched_barrier → vmcnt(8 for 9021 / 9 for 9022)
loops=1:   candidate仍使用vmcnt(0)
```

保留所有原lgkm等待、硬件barrier、consumer retirement、main loop、scale refill和输出。ISA确认8/9条K0 LDSDMA先于scale、8/9条K1位于其后，接目标partial wait。Scale消费自带vmcnt(0)会先等K0，故预期仅是部分K0/scale重叠和K1继续在途，不声称已经消除全部起步等待。

| ID | baseline→candidate VGPR | AGPR | SGPR | LDS B | private/spills | 静态ISA B |
|---|---:|---:|---:|---:|---|---:|
| 9021 | 164→164 | 0→0 | 96→98 | 105504不变 | 全部0 | 8316→8588 |
| 9022 | 194→194 | 0→0 | 66→63 | 81184不变 | 全部0 | 6888→6944 |

MFMA、矩阵load、DS读和硬件barrier静态数量相同；9021 wait68→69、9022 wait32→35。其他SALU/VALU和控制流随新调度变化；ISA变大和资源不变都不足以决定性能。

编译使用 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin`、ROCm `/opt/rocm`、gfx950、retained manifest冻结参数和机器指令验证，未添加本机拒绝的`-amdgpu-coerce-illegal-types=1`。首次host pass因legacy HIP FP8/BF16 header与OCML声明不兼容失败；在两个相同private launcher的完整HIP runtime include之后定义`__HIPCC_RTC__`，仅使OPUS traits host pass采用minimal declarations，再编译通过。失败日志保留。

本轮随后对正式JIT生成TU完成核对：**9021/9022私有baseline与起点正式对应kernel的指令字节逐字节相同、完整metadata相同。** descriptor只有随TU布局变化的`kernel_code_entry_byte_offset`不同；清除此相对地址字段后其余descriptor相同，证据在 [official_compute_metadata.json](official_compute_metadata.json)。因此私有AB的device body有明确的正式基线身份；最终已应用范围的改变entry匹配对应private candidate，不变entry匹配baseline，详见9.7。

### 8.1 已完成的小shape GPU启动验证

记录 [results/compute_pilot.json](results/compute_pilot.json)：MI355X、PCI `0000:05:00.0`，9021的 `(17,256,384)` 与9022的 `(176,256,384)`；signed输入，baseline/candidate各3次重复，全输出符合当前累加误差契约、可重复、两侧输出guard通过。轮换计时返回的最后输出buffer也通过检查。

该pilot只有1轮AB、11次调用。记录中的时间来自`run_perftest`的torch.profiler GPU事件汇总，不能当作稳定收益或无profiler完整时间。随后GPU再次被外部任务使用，下一任务前自动停止。本pilot证明这两个短K/M尾块路径可运行；不覆盖长K、scale面板换代、全部轮换buffer或生产launcher集成，**该pilot阶段尚未采用prologue优化**；后续最终决定见9.1/9.7。

私有C ABI：

```cpp
extern "C" int launch(int variant, const void* a, const void* b,
                      const void* sfa, const void* sfb, void* c,
                      int m, int n, int k, void* stream);
```

variant9021/9022；返回HIP error code、不做同步。9021任意正M，9022 M%16=0，N%128=0，K128..16384步128，矩阵byte bounds≤INT32_MAX，A/B/C/SFA基址16B对齐。标准FP8/原生E8M0布局，batch=1且所有batch strides在args里完整设置。

### 8.2 六个prologue目标的完整Event初步结果

在MI355X、PCI `0000:85:00.0`、HIP visible5的空闲窗口完成六份正式JSON。目标0/2的第一次运行途中出现同卡外部任务，已中止并移存为`attempt_*.json`；这些记录的`timings_usable_for_final_decision=false`，全部排除。下表只使用重新取得空闲卡后的完整结果，各项claim end均为returncode0、`contamination=false`；进程监测无同卡外部任务。空闲卡被多次重新取得，未固定时钟/功率/温度，还不能视为受控的独立窗口复现。

| 被测ID / M×N×K | 被测WG数 | 当前745形状身份 | Event baseline→candidate us | 吞吐变化 | 更快轮数 |
|---|---:|---|---:|---:|---:|
| 9021 / 512×6144×7168 | 192 | 当前赢家9021 | 39.7548→38.9462 | +2.076% | 5/5 |
| 9021 / 128×7168×768 | 56 | 当前赢家9047，测试9021候选 | 8.3797→7.8761 | +6.393% | 5/5 |
| 9021 / 608×7168×7168 | 280 | 当前赢家9022，测试9021候选 | 72.5047→70.8702 | +2.306% | 5/5 |
| 9022 / 1408×7168×16384 | 504 | 当前赢家9022 | 166.0057→166.2488 | −0.146% | 2/5 |
| 9022 / 304×7168×3072 | 112 | 不在745集合，M尾块机制代表 | 22.5578→21.9939 | +2.564% | 5/5 |
| 9022 / 512×7168×768 | 224 | 当前赢家9021，测试9022候选 | 10.0197→9.4455 | +6.078% | 5/5 |

每项使用signed输入、计划seed17–22，baseline/candidate各8次原buffer检查，以及同一51地址池、5轮AB/BA、每graph51次完整调用。所有原buffer与Event每轮全部51输出的数值、重复和guard检查通过。吞吐变化按`median baseline / median candidate − 1`计算，不能将不同parent的历史赢家时间直接拿来当本轮配对基线。

五项正值的逐轮AB比值均为正；当前真实9021赢家512×6144×7168的逐轮提升约1.780%–2.123%，两侧范围/中位数波动0.118%/0.254%，值得独立确认。9021的56WG短K用例baseline波动3.622%、candidate0.279%，尚需排除顺序与起步状态影响；它也没有直接改变当前该shape的默认赢家。9022最大K真实赢家逐轮变化约−1.247%到+0.637%，两侧波动1.883%/1.754%，完整Event结果接近零；单轮profiler曾报告+8.179%，不能据此认定长K已经获得提升。

**这是首批阶段结果。** 六项中的两个是当前相同parent真实赢家，另外三个是在745中调用其他候选，M304是额外机制shape。后续32项机制、201个实际赢家及有限异常复核已经完成，最终分parent结论见9.1；额外支持域保持partial。单目标A/B counter见5.5；尚无ATT或K0/scale动态区间测量，起步重叠仍为源码、ISA及有限计数趋势支持的机制假设。首批逐轮结果、日志核对、历史形状身份和污染排除见 [compute_prologue_window_analysis.json](results/compute_prologue_window_analysis.json)。

### 8.3 第二批：32项机制检查与真实赢家确认

2026-10-07 19:05:15–19:06:15 UTC，在物理PCI `0000:85:00.0`、HIP visible5的空闲窗口完成本节干净结果。每个命令分别申请空闲卡，claim end均为returncode0、`contamination=false`；日志中的全部trace/Event数值与JSON逐项一致。未锁定时钟、功率或温度，因此继续用同地址AB/BA配对及完整覆盖判断，而不把这段结果称为固定时钟下的全域性能证明。

32项 [机制计划](plans/compute_prologue_mechanism_correctness.json) 全部通过baseline/candidate各8次signed输入检查，共512次原buffer输出数值、guard和重复一致检查。覆盖loops1的vmcnt0、loops2的K1在途、奇偶环复用、K4096/4224及8192/8320面板边界、K16384多次换代，以及9021的M1/15/16/17/127/128/129与9022的M16/144/160/176/304/320/336边界。每项另有1轮11次profiler筛选；该部分没有Event确认，时间只用于发现明显回退，不能作为最终收益。

下表全部是当前745集合内、同parent的实际默认赢家。Event每项5轮AB/BA，每graph51次完整调用，两侧使用同一51地址池。

| ID / M×N×K | 当前WG数 | Event baseline→candidate us | 吞吐变化 | candidate更快轮数 | baseline/candidate范围÷中位数 |
|---|---:|---:|---:|---:|---:|
| 9021 / 512×6144×7168，单独复测 | 192 | 39.6442→38.8238 | +2.113% | 5/5 | 0.506% / 0.814% |
| 9021 / 480×7168×384 | 224 | 7.1185→6.6549 | +6.965% | 5/5 | 0.342% / 0.625% |
| 9021 / 320×7168×768 | 168 | 8.5663→8.0918 | +5.864% | 5/5 | 0.449% / 0.281% |
| 9021 / 192×16384×1536 | 256 | 12.8825→12.5530 | +2.624% | 5/5 | 0.615% / 0.968% |
| 9022 / 544×7168×384 | 224 | 7.7365→7.2298 | +7.008% | 5/5 | 1.064% / 0.564% |
| 9022 / 896×7168×384 | 336 | 8.9538→8.7663 | +2.138% | 5/5 | 0.753% / 0.787% |
| 9022 / 896×7168×768 | 336 | 12.9279→12.6942 | +1.841% | 5/5 | 1.329% / 0.717% |

七项共3570个Event轮换输出检查，加上各侧原buffer8次检查，数值、guard与重复一致全部通过。这里没有workspace，JSON中的`workspace_repeatable=false`表示没有workspace可比较，不能误判为partial不一致。吞吐变化继续采用`100 × (median baseline us / median candidate us − 1)`；每轮配对方向也独立核对。

512×6144×7168先前提升2.076%，本次为2.113%，说明该真实9021赢家的改善在单独复测中仍存在。短K真实赢家多项为正，但不能仅凭这些数字将kernel分类为dispatch bound、memory bound或MFMA compute bound。9022的1408×7168×16384首批接近零；480×65536×1536、1536WG代表随后在当前赢家批次取得完整干净结果，按parent+shape计入一次覆盖。旧中断attempt时间继续排除；完整201项结论见9.1。

单轮profiler和Event方向可明显不同：9021的480×7168×384单轮trace约−15.216%，共享Event却为+6.965%。两者地址池和计时协议不同，筛选结果只触发复核，不能单独用来采用或拒绝。CPU核验、10条Event日志、每轮AB比值、claim对应epoch及中断排除见 [compute_prologue_confirmation_analysis.json](results/compute_prologue_confirmation_analysis.json)；可用 [分析脚本](analyze_compute_prologue_confirmation.py) 复现。

## 9. 下一轮优先实验与采用条件

### 9.1 全部201个实际赢家后的prologue保留决定

机制正确性32项、201个实际赢家与7项已观察fallback异常的Event复核全部完成。优先代表的最后9022/1536WG已在当前赢家集合中完成并去重；初始三文件candidate15项正式API通过。数值/入口验证和完整任务性能分别记录，额外1426/646支持域拓展没有扩大为硬条件。

[严格覆盖](results/compute_prologue_winner_event_coverage.json) 现为201/201个唯一parent+shape、203个clean窗口；两个独立重复分别为9021既有真实赢家与9022预先固定异常。所有记录核对当前pair SHA/runner、自身clean claim epoch、物理PCI、signed8及共享池每graph51次×5AB/BA、每轮全部output数值/重复/guard。按每shape最新clean窗口汇总，所有既有窗口仍保留；几何平均每shape等权，sumtime是各shape单次median之和，表示合成集合，不能当作一次实测batch。

| Parent | 实际赢家 | median更快/更慢 | 5/5更快/更慢/混合 | Geomean加速 | 各shape时间之和加速 | 单shape加速范围 | 本轮范围决定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 9021 | 42/42 | 42 / 0 | 38 / 0 / 4 | +3.155% | +2.313% | +0.923%～+9.208% | 已应用128×128 prologue |
| 9022 | 159/159 | 131 / 28 | 101 / 11 / 47 | +1.390% | +0.470% | −1.078%～+7.008% | 不采用160×128全局prologue |

完整K分组如下。正负为`baseline/candidate−1`，负数表示candidate更慢；不与candidate时间增加百分比混用。

| K | 9021项数 | 9021 geomean | 9022项数 | 9022 geomean | 9022 5/5更慢项 |
|---:|---:|---:|---:|---:|---:|
| 384 | 3 | +7.475% | 22 | +3.414% | 0 |
| 768 | 7 | +5.894% | 22 | +2.564% | 0 |
| 1024 | 1 | +5.714% | 0 | — | 0 |
| 1536 | 5 | +2.036% | 30 | +1.307% | 0 |
| 3072 | 4 | +2.214% | 22 | +1.735% | 0 |
| 7168 | 18 | +2.076% | 43 | +0.526% | 2 |
| 16384 | 4 | +1.859% | 20 | −0.468% | 9 |

9021所有K组为正，没有负median或5/5更慢shape；单独真实赢家复测、32机制及有限fallback异常核验也支持该起步改动。9022虽然总体小幅正向，K16384组的20项geomean为−0.468%、sumtime为−0.478%，其中9项5/5更慢；加上K7168两项，共11项5/5小幅变慢，单shape最差−1.078%。这是跨多个shape的使用范围取舍，不因单个负值自动拒绝，也不声称每个小回退已有唯一硬件根因。因此本轮保留9022旧prologue；其短K仍有供数空间，未来若研究分K选择是新variant/dispatch设计，本轮不追加。

唯一较大混合信号9022 `1216×7168×384`事先固定一次独立复测。旧窗口median−13.942%，5轮方向为正负混合、baseline/candidate range为22.207%/35.312%；新窗口9.5836→9.4315μs、+1.613%，5/5更快、range0.908%/0.632%，PCI`0000:65:00.0`/HIP2、claim clean。它没有复现为稳定大回退；新窗口按预先声明的latest-clean规则进入汇总，旧窗口完整保留，停止追加复测。原来的大异常不是拒绝9022的理由；拒绝依据是完整长K使用范围的收益/小回退取舍。

本轮最终保留并应用9021 prologue和9042/9053/9054三个B-scale复用alias，范围缩为两文件、四改变entry；9022、narrow、fine与fixed_n32维持旧实现。最终正式module/API身份及源码应用核对见9.7。

CPU落地范围见 [compute_adoption_scope.json](compute_adoption_scope.json)：两个prologue位置会改变当前745-shape默认赢家中的42+159=201项；在这745个shape上，9021/9022分别可调用735/691项。赢家数、候选支持数和已完成GPU数分别记录。计划生成逐项调用当前scalar支持函数、计算实际generated host条件，并核对既有范围清单，未导入torch或初始化HIP。

| 范围 | 9021 | 9022 | 每个shape的验证协议 | 时间用途 |
|---|---:|---:|---|---|
| 已准备的候选支持域拓展 | 735项，46个分片 | 691项，44个分片 | baseline/candidate各8次signed数值/重复/guard；1轮profiler、11次调用 | 可选额外筛选；未要求为采用把1426组合全部完成 |
| 当前默认赢家 | 42项 | 159项 | 各8次原buffer检查；共享自动轮换池、每graph51次完整调用、5轮AB/BA Event与全部pool输出检查 | 完整调用收益及回退判断 |

支持域计划每片最多16个shape，共90个命令、1426个parent-shape组合。同一个shape可能分别支持两个parent，所以1426不是新增shape数。赢家计划也按16项分片保存，初始执行队列为201个单shape命令，后来将剩余项合并为最多8项批次。具体计划、哈希及协议见 [完整范围清单](compute_prologue_full_scope_manifest.json)，执行队列为 [支持域筛选](compute_prologue_full_support_queue.json) 和 [201项赢家Event](compute_prologue_all_winners_event_queue.json)，CPU可用 [计划生成脚本](prepare_compute_prologue_full_scope.py) 复现。可选支持域拓展已停止于528/1426个clean parent-shape组合，见 [支持域CPU核验汇总](results/compute_prologue_support_analysis.json)；只计clean完整片，未测或污染部分保留partial说明，不能推定全域通过。

这两处prologue实验只改变单WG起步顺序/partial wait，main loop、地址、grid和布局未改；32项机制覆盖关键同步路径，额外支持检查及全部实际赢家Event含8rep数值。完成剩余所有fallback组合不是采用硬条件。执行期间优先正式API和实际赢家性能；必要验证完成后停止可选拓展，已有clean支持证据保留，范围审查见 [最小充分验证记录](formal_candidate/MINIMUM_REMAINING_GATES.md)。

另有 [短K优先Event队列](compute_prologue_all_winners_event_short_k_queue.json) 与 [排序清单](compute_prologue_event_priority_manifest.json)：按K升序安排全部201项，K384/768优先，因为先前真实代表的prologue收益较明显。排序只影响执行先后，保留全部42/159赢家和原始命令内容，不改变采用条件。CPU用 [排序脚本](prepare_compute_prologue_event_priority.py) 复现，原队列保留。

[实际赢家Event覆盖汇总](results/compute_prologue_winner_event_coverage.json) 按parent+shape去重，严格核对结果自身的clean epoch、实际GPU、相同库/计划SHA、signed8检查与共享池51calls×5轮Event。初次去重已有8个唯一真实赢家、9个clean记录，分别为9021/9022各4个；对应剩余38+155=193项写入 [剩余赢家队列](compute_prologue_winners_remaining_event_queue.json)，旧计划保留。后续累计数以动态coverage JSON为准。用 [CPU覆盖脚本](analyze_compute_winner_event_coverage.py) 的`--summary-only`在队列运行时更新汇总，禁止重写活动队列；重复窗口不算新增shape，未完成时明确标partial。另有 [逐parent/K性能决策汇总](results/compute_prologue_performance_decision.json) 和 [CPU复算脚本](analyze_compute_prologue_decision.py)，按每shape最新clean窗口保存geomean、单次各shape时间之和、5轮原始配对、窗口内波动与所有5/5变慢信号。0.5/1/2/5%敏感度只描述信号，不自动采用或拒绝；全部既有重复窗口保留。

为减少同卡空闲窗口中的Python/HIP重复起步，后续把剩余164个实际赢家整理成21个同parent批次，每批最多8项，见[批次队列](compute_prologue_remaining_batches_closing164_queue.json)及[CPU核验](compute_prologue_remaining_batches_closing164_manifest.json)。目标、库、signed8、每graph51次和5轮AB/BA均保持；每项仍单独构建地址池、计时和检查，批次合并只减少进程重启。不同批次不改变已完成37项证据，污染批次整体排除后重试。`480×65536×1536`已在当前赢家集合中，按相同shape去重，避免独立代表重复测试。

支持域筛选信号另用 [Event复核生成器](prepare_compute_support_event_followups.py) 整理 [动态复核清单](compute_support_event_followups_manifest.json) 与 [复核计划](plans/compute_support_event_followups.json)。先列出clean完整片中candidate时间增加至少5%的trace信号，该数值只用于安排复核，不是拒绝阈值。当前已有8个信号；一个M480/K384真实赢家已有干净Event为正，引用既有证据，其余7个9021 fallback候选安排共享池Event。这些shape当前赢家属于其他parent，但全局修改9021仍会改变其支持路径，所以核对已经观察到的信号。与201赢家队列及已完成Event按parent+shape去重，保留每项原cleanfile/hash、plan与claim；有限7项完成后，没有新的具体风险就停止追加，不扩成全部fallback Event。支持域仍是partial，不能声称全部fallback都更快。

上述7项fallback Event现已全部完成并通过严格claim、shared-pool逐buffer数值/guard、10项Event日志与单位审查，见性能决策JSON的`finite_fallback_followups`。N均7168：M144/K384、M256/K384、M96/K384、M64/K768、M16/K768分别为+3.903%、+9.261%、+0.822%、+4.971%、+2.983%，均5/5轮更快；M16/K384为+0.055%、M1/K384为−0.056%，逐轮方向混合、接近零。这里的正负为baseline/candidate−1，不将近零值称为稳定收益或回退。最初≥5%的trace变慢信号没有在这一轮共享池Event中复现为明显回退，有限异常核验完成。

9021/9022分别根据实测结果决定。已观察到的明显trace回退用同卡共享池Event确认；单轮trace不能成为最终拒绝依据。当前赢家需要完整Event，已有同parent+shape的clean记录可据相同机器身份计入，不因整体.so布局或独立队列重命名重测。综合K长度、M尾块、grid、逐轮配对方向和复测波动判断有效收益与回退，不设置脱离数据的统一百分比门槛，也不持续加轮直到出现正数。若只有明确区域稳定改善，就保存旧fallback，并仅在覆盖与邻接边界都验证过的范围选新variant。

确认保留范围后，最终selected重建与正式API核验完成，实际device body、metadata/descriptor、host launch、96B kargs ABI和实际加载module SHA均通过；未改变kernel使用CPU二进制身份核对。5.5的单目标counter提供有限供数线索，K0/scale具体重叠仍保留机制假设；数值与受控Event用于证明功能和完整时间。

### 9.2 新发现：runtime narrow展开与scalar存活

正式9023/9024 runtime循环分别展开3/4次，在动态scale面板与地址分支下产生VGPR lane保存的scalar spill。已准备 [narrow_unroll](narrow_unroll/CPU_READINESS.md) 私有单项实验：仅runtime main loop展开改为1，所有slots、scale、wait、barrier、grid/cache和输出保留；固定K7168仍使用原展开策略。

| Runtime baseline→candidate | VGPR | SGPR spill slots | 静态writelane/readlane | ISA B |
|---|---:|---:|---:|---:|
| 9023 | 232→121 | 46→21 | 46/166→21/21 | 25976→10928 |
| 9024 | 152→87 | 44→22 | 44/132→22/22 | 21736→10776 |

两者SGPR仍106、LDS不变、AGPR/private/VGPR spill仍0。私有baseline四variants全部匹配正式指令、metadata与归一化descriptor；两个fixed-K variants在baseline/candidate完全相同。源码差异与机器码证据分别见 [changes.diff](narrow_unroll/changes.diff)、[isa_comparison.json](narrow_unroll/isa_comparison.json)。

原9023主循环3个K128窗口共24MFMA/120静态readlane，候选单K128窗口8MFMA/16readlane；原9024四K128共16MFMA/88readlane，候选单K128为4MFMA/22readlane。还有余数循环、面板生产和外层回跳，按PC列出的静态区间数量不能直接当成动态总次数。限制展开会增加loop控制频率、改变预取/矩阵发射重叠；是否改善issue、供数、latency或dispatch需GPU证据。

已准备52个机制正确性目标及原12个计时目标：M1536/N768，K384/768/1536/3072/16384，两ID及不变K7168对照。CPU落地范围核对发现，这组M/N不包含当前9023的4个runtime赢家；新增 [plans/narrow_current_winners.json](plans/narrow_current_winners.json)优先测M544/576、N7168、K7168/16384，以及两个9024 runtime回归对照，并另备fixed dispatch边界。

当前745历史赢家中，runtime pragma改变仅涉及9023的上述4项；9024的9个赢家全部走fixed-K，机器码保持不变。在该745-shape集合里，两个parent各有718个可调用runtime路径和17个不变fixed路径，因此9024没有受影响赢家并不表示其runtime fallback可以免测。详见 [compute_adoption_scope.json](compute_adoption_scope.json)。

#### 9.2.1 本轮真实赢家与runtime对照结果

2026-10-07 18:56:23–18:57:10 UTC，在一个空闲MI355X窗口执行六项，物理PCI `0000:85:00.0`、HIP visible5。每项signed输入seed17，baseline/candidate各8次原buffer运行；再用同一51地址轮换池、5轮AB/BA、每graph51次完整调用做Event确认。所有原buffer数值/重复/guard以及每轮全部51个输出buffer的数值/重复/guard均通过；没有workspace。六项claim end均为returncode0、`contamination=false`，进程监测未发现同卡外部任务。其余7卡当时忙，未固定本卡时钟、功率或温度。

| ID / M×N×K | Event baseline→candidate us | Event吞吐变化 | 逐轮配对变化中位数 | candidate更快轮数 | 单轮profiler变化 |
|---|---:|---:|---:|---:|---:|
| 9023 / 544×7168×7168 | 47.5369→47.7228 | −0.390% | −0.387% | 1/5 | +0.480% |
| 9023 / 544×7168×16384 | 99.3186→99.0299 | +0.291% | +0.941% | 4/5 | +5.637% |
| 9023 / 576×7168×7168 | 48.2577→48.2240 | +0.070% | −0.374% | 2/5 | +1.118% |
| 9023 / 576×7168×16384 | 99.0574→100.2205 | −1.161% | −1.157% | 1/5 | +4.837% |
| 9024 / 608×768×7168 runtime | 20.9100→20.8872 | +0.109% | +0.109% | 4/5 | +1.606% |
| 9024 / 1536×768×16384 runtime | 62.5260→63.3566 | −1.311% | −1.393% | 0/5 | −0.760% |

Event吞吐变化定义为`100 × (median baseline us / median candidate us − 1)`，与runner存储summary一致；“逐轮配对变化中位数”先计算各轮AB比值，再取中位数，是另一种统计。9023的576×7168×7168在两个统计里符号相反，说明接近零的结果会受聚合口径影响，不能选择其中一个当作稳定提升。

四个9023赢家的单侧5轮范围/中位数波动约0.985%–2.272%，配对变化均出现正负混合。两个long-K shape的profiler活动时间曾显示+5.637%/+4.837%，共享池Event完整调用仅+0.291%/−1.161%。Profiler与Event采用不同地址池、执行与计时协议；现有结果不能将差异单独归因为launch、cache或某个硬件单元。9024最大K对照的candidate五轮全部慢约1.22%–1.50%，baseline/candidate单侧波动仅0.361%/0.495%，是本窗口明确的退步信号。

**本轮决策：9023/9024全局runtime pragma均不采用。** CPU减少VGPR、scalar lane spill和代码长度，没有变成这些实际shape的可靠整体收益。保留生产基线，停止为这个候选追加寻找正值的narrow计时；prologue独立验证随后已经完成，最终决定见9.1。此六项没有完成52个机制正确性目标、718个runtime候选支持域或独立窗口复核，也没有采集counter/ATT，不能给出MFMA issue、TA、L1或HBM的根因判断。

六份原始JSON、日志和claim记录，以及每轮Event、波动和两种统计的CPU核对见 [narrow_runtime_window_analysis.json](results/narrow_runtime_window_analysis.json)。

### 9.3 A scale预转置：先9021，后9022

当前9021/9022的M-repeat byte LDS读加shift/or可尝试移动到producer面板打包。9021四repeat可紧凑存为128B/K128，保持SFA总4096B；每轮消费减少为packed读。

9022有第五repeat。若简单每32坐标用两个u32，SFA5120→8192B、LDS81184→84256B，越过80KiB的两WG LDS预算。优先考虑“四byte packed面板 + 独立第五byte”，同时检查LDS bank、producer transpose代价和VGPRlive range。只有counter/ISA显示scale供数确有损失时推进。

### 9.4 9030 midpoint迁移

保持64位C基址、tile-local bounds和输出，先仅移植9020式midpoint publish/retire；独立评估K0/scale/K1起步，不与grid/scale面板/epilogue同时变化。先覆盖当前10个大输出K1536，再扩展短K、奇偶、K16384和2GiB地址边界；完成计时必须有足够空闲显存。

### 9.5 停止与保留标准

- 正确性、边界或同步未通过：修复或关闭，不以速度覆盖错误。
- 新spill、dynamic LDS超范围、关键资源阈值恶化：解释具体机制后再决定是否继续。
- 局部ATT改善而完整周期/无profiler计时无收益：关闭该实验。
- 配对收益接近噪声：预先约定复核规模和独立窗口，不不断增加轮数直到出现正数。
- 保留版本需受影响实际shape的有效改善、已发现回退的处理、生产TU一致性和必要机制数值验证；未测支持路径保留范围限制。

本轮性能范围决定为**已应用9021 prologue及三个register alias，拒绝9022全局prologue与其余已测候选**；最终源码应用和正式身份见9.7。该决定只覆盖本轮有限机制和实际使用集合，后续仍可提出独立假设。

### 9.6 最小生产集成：按已经验证的范围保留

本轮narrow runtime展开修改已拒绝，保留旧runtime和fixed-K实现。prologue与三个register alias分别维护采用范围：以受影响实际赢家、必要机制、已观察异常Event和正式API核验作为采用证据，可选支持域拓展按部分完成量报告。三个register alias的6/6实际赢家及有限收尾已通过；prologue独立状态见9.1。通过的最小源码差异已按精确SHA应用并记录；一个机制通过不能代替另一机制的验证。

若某项收益依M/N/K或grid变化，使用足够的shape及邻接边界建立可解释的选择条件，保留已有fallback和fixed-K优先顺序。不能仅依据一次profile的获胜shape设置阈值。新traits或device variant会改变C++symbol及可能的编译调度，需要重新核对实际generated TU、ABI、metadata、descriptor和正式入口。

每项集成保持已验证的数学和布局契约，核对改变entry与私有candidate一致、未改变entry与正式baseline一致，再确认正式API加载的binary身份和数值。不要将slot深度、scale面板、grid或epilogue同时叠加到已经验证的差异；它们需要各自的机制假设和证据。

### 9.7 最终两文件范围的正式身份与入口核对

最初在detached worktree `/root/workspace/opus-bound-candidate-20261007` 预构建了9021/9022两处prologue及9042/9053/9054三个显式B-scale复用alias，模板默认false。该三文件、五改变entry的产物与15项API证据在`formal_candidate/`保留为历史snapshot。根据完整201赢家结论，隔离worktree随后移除9022改动，只保留9021及三个alias；最终两文件差异已应用到原checkout。

初始三文件构建的26个parent共56个device entry（含5个split-K reducer）编译通过。改变的5个entry指令字节、完整metadata和归一化descriptor全部匹配已测private candidate；其余51个entry匹配正式baseline，26个generated TU和host launcher impl均字节相同。全构建额外检查206个CUDA对象、202个device bundle，未改变kernel身份通过；完整对象SHA因隔离源码路径及host布局均不同，不能把这一结果写成对象文件字节相同。初始.so SHA为`51d4b3d7dd033cc99d391d87e89c25e8afc501749bc919a786af492eb717d84a`，原baseline SHA未变。

差异、完整身份及复现方法见 [formal_candidate/CPU_READINESS.md](formal_candidate/CPU_READINESS.md)、[源码diff](formal_candidate/source_changes.diff)、[正式CPU身份核验](formal_candidate/identity_audit.json)。[15项正式smoke计划](plans/formal_candidate_smoke.json) 的 [CPU分支审查](formal_candidate/smoke_plan_cpu_audit.json) 核对当前scalar支持条件、generated host分支、私有launch范围与正式device身份，覆盖5个改变entry及4个不变entry对照；K7168对照未错误附上runtime private launcher。Rank2与当前公开GEMM-only family一致，公开opus_bmm拒绝此family，不能将底层rank3检查称为公开支持。此次没有改指针/stride/bounds或对齐，显式16B offset亦无需额外强制测试。

初始三文件正式GPU smoke为15/15通过，见 [逐项严格结果](formal_candidate/gpu_smoke_analysis.json) 和 [CPU复算脚本](analyze_formal_candidate_smoke.py)。UTC `2026-10-07 20:06:01.681–20:07:42.529`，MI355X/gfx950、PCI`0000:05:00.0`、HIP visible1；各自command epoch clean且GPU与物理claim匹配。Official120次与private96次、共216次signed8rep数值/重复/guard全部通过，实际加载路径/SHA匹配初始module，私有device身份也再次逐项匹配。这批check-only没有性能结论，不能将五entry历史产物冒充最终四entry模块。

最终两文件、四改变entry范围已重建并通过 [selected CPU身份核验](formal_selected/identity_audit.json)，脚本 [audit_formal_selected.py](audit_formal_selected.py)。26parent/56entry中，4个保留entry匹配private candidate指令字节、完整metadata和归一化descriptor；52个不变entry匹配正式baseline，含恢复旧prologue的9022。26个generated TU/host launcher字节保持、producer96B/reducer20B ABI保持；全构建206对象/202device bundles核对无非预期变动，whole-object SHA仍因隔离路径/布局不同，不称对象文件字节相同。最终新module SHA为`45efa9ee9b8c82f4f9b169c547a8f68e22c68ea1ac5badd1409d0514145bbd14`，源码/构建见 [selected diff](formal_selected/source_changes.diff)、[selected manifest](formal_selected/build_manifest.json)。

[最终12项API计划](plans/formal_selected_smoke.json) 及 [CPU分支/私有身份核验](formal_selected/smoke_plan_cpu_audit.json) 已通过：9021短/长K与M-tail3项、三register entry的6个真实赢家、恢复9022对照1项、fixed9071/9072对照2项，每项check-only8rep。恢复9022搭配private baseline，固定K对照没有错误的runtime私有launcher。最终 [GPU结果严格审查](formal_selected/gpu_smoke_analysis.json)、[CPU复算脚本](analyze_formal_selected_smoke.py)确认12/12通过：official96+prologue24+register48+恢复9022 baseline8，共176次数值/重复/output及workspace guard通过，覆盖4改变entry与3不变control entry。UTC`20:37:42.702–20:39:00.017`、MI355X/gfx950、PCI`0000:65:00.0`/HIP2；每项clean claim和唯一host owner marker通过，实际加载module路径/SHA逐项匹配上述最终新模块。这批check-only没有性能结论，性能依据仍为各自已测完整Event。

UTC`2026-10-07 20:41:26.667`，最终 [两文件diff](formal_selected/source_changes.diff)已应用到原checkout，见 [integration manifest](formal_selected/integration_manifest.json) 和 [应用前证据snapshot](formal_selected/pre_apply_evidence_snapshot.json)。状态`applied_verified_not_committed`：两文件SHA与已测selected源码完全相同，diff SHA为`155b09cce6c68d57aeb9daa9915bb88cd38c1aa68607f3f8bbf31de4a3c4966b`，9022源码SHA保持原HEAD，`git diff --check`通过，未commit或push。源码应用没有改变已经验证的body，因此复用selected正式API/机器身份，无需再跑GPU。经验证模块位于记录的独立JIT目录；以后默认JIT构建应使用已应用源码和所记录compiler/flags，不能把旧默认缓存称为已更新。此次采用验证结束，不追加性能扩测或新的分Kdispatch。

## 10. 复现实验和证据索引

### 10.1 正式调优入口

使用实际支持pin的LLVM与原生E8M0 PyTorch；实际device限制以GPU身份检查为准。本轮完整26候选的正式JIT CPU重建已经通过，记录见 [current_build.json](current_build.json)。本机LLVM24拒绝JIT探测误接受的`-amdgpu-coerce-illegal-types=1`；ROCm7.0的旧HIP头还依赖LLVM24 resource目录里缺少的OCML声明。现有 [build_current.py](build_current.py)只在本进程过滤该flag、指定 `/opt/rocm/lib/llvm/lib/clang/20` resource目录，保持kernel与JIT生产源码不变。

从仓库根目录复现该CPU构建：

```bash
python3 reports/opus_bound_analysis_20261007/build_current.py
```

直接用默认JIT新编译可能再次命中上述不兼容配置。GPU测量应复用已经记录SHA的构建，或在同一进程安装相同构建配置；每项记录实际生成TU、flags和binary身份。以下展示tuner参数选择，`<idle_hip_index>`需替换为实际空闲卡；先隔离少量shape再全量回归：

```bash
HIP_VISIBLE_DEVICES=<idle_hip_index> \
OPUS_HIP_CLANG_PATH=/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin \
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  -i aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv \
  -o /tmp/opus_bound_tuned.csv -o2 /tmp/opus_bound_profile.csv \
  --libtype all --splitK --shape_grouped --mp 1 --warmup 5 --iters 51 --all
```

编译新candidate前固定flags，不把当前不兼容LLVM的额外flag加入基线。缓存目录隔离并记录SHA；首次JIT、input生成、reference和profiler不混入稳态事件计时。

### 10.2 每次测量应记录

```text
source commit + source SHA + actual generated branch + binary SHA
GPU model/gfx + physical PCI + visible HIP index + ROCm/compiler/flags
M/N/K + layouts/dtypes/scales + workspace bytes + dynamic LDS
VGPR/AGPR/SGPR + private/spills + resident limit query
warmup/iters + raw paired timings + correctness scope
counter definitions/time/aggregation + ATT local scope if used
retained/rejected decision + affected support domain
```

### 10.3 本轮runner的计时口径与复核方法

空闲卡监测使用物理PCI及进程列表。后续发现容器内child PID与KFD/SMI报告的host PID可不同；现在用同一Python进程先open`/dev/kfd`、确认唯一新增host PID并经nonce握手，再以`runpy`执行原runner，argv和runner SHA保持相同。每个新epoch保留`owner_identity`，CPU审查要求唯一`new_host_pids=[host_pid]`、marker在自身start/end内、inner PID与monitor相同；实际日志已验证host3550573对应child1295956，其自身2,439,905,280B allocation被正确识别，clean完成。早期无marker的clean证据保持原口径；旧中断或污染轮次仍排除，不因这次监测修正追认时间。实现/CPU握手检查见 [owner_identity_cpu_validation.json](owner_identity_cpu_validation.json)。

现有 [experiment_runner.py](experiment_runner.py) 和 [official_smoke.py](official_smoke.py)复用仓库`run_perftest`：自动deepcopy输入/输出/workspace轮换，在torch.profiler中采集每次设备kernel时间；去掉首迭代，超过30迭代时进一步用IQR过滤后平均。单轮JSON保存这个聚合us，而非全部原始迭代。它适合与当前tuner一致地筛选候选；完整split-K汇总main与reduce的设备事件，但不包含两次launch之间的空隙和host/API成本。

每个AB侧单独调用`run_perftest`都会建立自己的轮换地址池，未保证两侧使用同一物理地址。它还先运行一次memory-profiling probe再deepcopy，轮换output/workspace副本可能已经含正确结果；现有正确性多次在NaN初始化的原地址运行，计时后只检查最后返回轮换buffer。因此原地址数值检查有效，但不能据此声称全部轮换buffer都检查过。

拟保留候选的复核机制已在runner实现为`--event-confirm`：统一预建输入/输出/workspace轮换池，两侧使用同一地址；所有输出/partial填NaN，正确性阶段逐buffer检查数值、guard和同label重复结果。无torch profiler/counter时用HIP graph批量重放、GPU Event记录整批完整调用，5轮AB/BA交替；split-K批次包含producer和reducer。独立`event_confirmation`字段保留整批毫秒，并报告`us_per_call = 1000×event_total_ms / iters`，包含设备调度空隙，与tuner的kernel活动时间分开。CPU假runtime测试通过共享地址、完整调用、逐buffer检查、单位、无capture内分配和清理编排，见 [event_orchestration_cpu_validation.json](event_orchestration_cpu_validation.json)。真实GPU已完成fixed_n32两项、register六个实际赢家、fine_wait九项和narrow六项完整Event；prologue最终覆盖201个唯一赢家、203个clean窗口，包含既有首两批及两次独立重复。各自范围、污染排除和决定分别见本节索引及Memory文档。早期fine_n64两次中断记录不作性能判断。已经完成的Event只覆盖相应shape；当前赢家、必要机制和已经观察到的异常分别按最小充分范围完成，1426/646支持域拓展保留partial，不作为新增硬条件。

不直接将`run_perftest(use_cuda_event=True)`当作相同轮换的替代：该分支逐次在原始args执行与同步，绕过已经建立的rotate_args。正式baseline API已经在真实GPU调用及上述四组counter采集时核对实际加载module的`__file__`、SHA及实际traits；初始三文件candidate的15项API smoke作为历史证据逐项匹配其独立模块；最终两文件selected的12项API、176次数值/重复/guard检查均通过，实际加载最终SHA与clean owner/claim一致，源码应用身份见9.7。历史统计、private baseline、初始candidate及最终selected各自保持身份记录。

### 10.4 主要索引

| 材料 | 用途 |
|---|---|
| [最新带宽笔记](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md) | Little定律、CU吞吐、分层回压、wave等待与派发口径 |
| [旧compute参考](/root/workspace/gcnasm_new/gcnasm-mxfp8-final-pipeline-source-20260910/opus_gemm/mxfp8_gemm_16x16x128_blockscale_bpreshuffle_4wave/COMPUTE_BOUND_GEMM_DESIGN_AND_OPTIMIZATION_20260922.md) | 数学契约、预算、pin、流水、ATT和严格历史消融 |
| [当前registry](../../csrc/opus_gemm/opus_gemm_common.py) | 默认/兼容IDs、parent dispatch、shape支持 |
| [当前codegen](../../csrc/opus_gemm/codegen/gen_instances_gfx950.py) | 实际traits选择、launcher与global split-K reduction |
| [745-shape报告](../opus_current745_tables_20260930/REPORT.md) | 当前已有数值/时间覆盖，两个来源分开记录 |
| [最新run manifest](../opus_consolidate_20260930/run.json) | 环境、命令、源码SHA与JIT身份 |
| [上一full745 manifest](../opus_full745_register_20260930/run.json) | 133大shape对应的当前实现SHA |
| [本轮private source manifest](compute_prologue/source_manifest.json) | 冻结审查起点baseline headers；最终应用身份另见integration manifest |
| [本轮ISA/resource对比](compute_prologue/isa_metadata_comparison.json) | prologue顺序、完整静态计数、private资源 |
| [本轮build manifest](compute_prologue/build_manifest.json) | CPU编译工具链与exact flags |
| [本轮正式JIT构建](current_build.json)、[构建脚本](build_current.py) | 26候选CPU重建通过、toolchain兼容配置、binary SHA |
| [正式compute metadata](official_compute_metadata.json)、[提取脚本](audit_current_compute_metadata.py) | 16实际variant资源、kernel字节SHA、私有baseline与正式身份 |
| [runtime narrow实验](narrow_unroll/CPU_READINESS.md)、[ISA核对](narrow_unroll/isa_comparison.json)、[CPU落地范围](compute_adoption_scope.json) | unroll1静态资源效果、fixed-K未变；实际四赢家与两个对照已完整Event，本轮拒绝 |
| [narrow真实赢家GPU结果与逐轮分析](results/narrow_runtime_window_analysis.json) | 四赢家/两个runtime对照数值通过，5轮共享Event近零或退步；本轮不采用全局pragma |
| [本机counter定义审查](counter_evidence.json)、[实际计数一致性审查](profiles/counter_runtime_consistency_audit.json) | gfx950单位/定义；4×51实际dispatch工作量通过，GRBM时间窗未核准，暂不使用绝对MFMA利用率/occupancy分类bound |
| [prologue小shape启动验证](results/compute_pilot.json) | 两条short-K路径数值/重复/guard通过；单轮时间不能证明收益 |
| [prologue六目标GPU结果与逐轮分析](results/compute_prologue_window_analysis.json) | 历史首批完整Event，污染attempt排除；最终判断见201项汇总 |
| [prologue第二批分析](results/compute_prologue_confirmation_analysis.json)、[CPU分析脚本](analyze_compute_prologue_confirmation.py) | 32项机制通过、真实9021独立复测+2.113%；最后1536WG代表在赢家批次去重完成 |
| [prologue完整范围计划](compute_prologue_full_scope_manifest.json)、[支持域核验](results/compute_prologue_support_analysis.json)、[汇总脚本](analyze_compute_prologue_support.py) | 735/691可调用支持域只计clean已完成片，拓展保持partial；不推定全域通过 |
| [完整201赢家覆盖](results/compute_prologue_winner_event_coverage.json)、[性能与K分组](results/compute_prologue_performance_decision.json) | 201唯一shape/203clean窗口；9021已应用，9022全局不采用，有限异常复测保留两窗口 |
| [最终两文件源码](formal_selected/source_changes.diff)、[CPU身份](formal_selected/identity_audit.json)、[12项API结果](formal_selected/gpu_smoke_analysis.json)、[应用核验](formal_selected/integration_manifest.json) | 4改变entry匹配private、52不变匹配baseline；12/176calls通过，精确两文件差异已应用、未commit/push |
| [短K优先Event清单](compute_prologue_event_priority_manifest.json)、[排序脚本](prepare_compute_prologue_event_priority.py) | 全201项K升序安排，原队列保留，命令和覆盖不变 |
| [Event复核CPU编排验证](event_orchestration_cpu_validation.json)、[fixed_n32完整GPU确认](results/fixed_n32_first.json) | CPU编排通过；真实GPU两目标逐buffer检查与5轮Event均完成，N32退步已拒绝；不替代compute实验的待测范围 |

当前结论：9000的大tile供数、scale复用、寄存器分工和合并输出已有较强历史证据；本轮完成独立候选的机制/完整时间验证后，已应用9021起步供数顺序及三个register alias，拒绝9022全局prologue、narrow、fine与fixed_n32。最终两文件正式身份/入口与原checkout应用均有独立记录，未commit或push。中tile/small families仍有queue、grid和长短K取舍，本轮验证结束。

这落实了最新笔记第五章的总结：硬件上限给出时间预算，软件可见模型把布局、并发、请求和等待联系起来，公开counter用于检验原因，完整任务时间用于决定改动。各层效率没有可以直接相乘的固定关系；缓存改变请求路径，回压改变供给，资源与tile改变执行量和wave寿命。每次优化后重新判断当前限制，保留实际通过的改动与范围。

## 11. 2026-10-08：先定位限制，再选择9021改动

### 11.1 按最新合并总结固定路径与证据身份

本次按[第一至第五章最新合并总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)先区分工作派发、访存请求、数据返回与计算消费，再用 counter 找到受影响接口，最后用 ATT 对齐实际等待指令。最新总结中的 MI300/2024 年硬件模型用于提出问题；端口宽度、时钟、资源上限和 counter 定义仍以当前 gfx950/MI355X 的已核对证据为准。高算术强度、低 occupancy 或某一等待比率不能单独完成 bound 分类。

固定诊断时Oct7 selected的9021、`M480/N7168`，比较 `K384` 与 `K16384`：两者均为224 WG、896 waves，不改变 tile/grid。诊断加载的是[隔离 baseline](../opus_resume_20261008/sfa_packed/baseline/experiments.so)，库 SHA 为 `15b080345c90be001ef9881a44c7e5b590032af4324e5e5b4157b226d3e7b49d`。[二进制核验](../opus_resume_20261008/sfa_packed/device_audit.json)确认其9021指令、完整metadata及归一化descriptor与Oct7最终selected一致，9022仍等于原baseline；ATT同样核对完整1498条静态指令及device SHA。诊断结果属于Oct7 selected，不能直接归给后续已验证的Oct8实现；候选机制另见11.6。

[分层汇总](../opus_resume_20261008/diagnostics/layer_analysis.json)和[CPU脚本](../opus_resume_20261008/diagnostics/analyze_layers.py)逐项检查应用、plan、runner、library、实际加载code-object切片SHA、物理PCI`0000:65:00.0`、owner PID、clean claim、counter别名表达式及trace dispatch身份。四组短K独立pass各有57次匹配dispatch，选末尾51次，共204次；228次原始CSV和callback均保留。另两个短/长K raw-clock pass由[clock审查](../opus_resume_20261008/diagnostics/clock_audit.json)独立记录。不同pass不拼接为同一次dispatch，也不把51次调用当作51次独立实验。此次profiling应用未对比数学reference或检查输出guard，正确性证据另行保留。

### 11.2 Counter先收窄等待类别，GRBM仍不支持绝对利用率

K384的LDS/SQ pass中，`SQ_WAIT_ANY/SQ_WAVE_CYCLES`中位数为64.296%，全部issue wait为11.595%，LDS-specific issue wait为1.804% wave life、15.692% issue wait；LDS active为3.431%。这些分子分母同属累计wave quad-cycles。它们支持优先查依赖等待来源，但不把64.296%解释为墙钟损失，也不从1.804%推断LDS数据返回没有成本。LDS issue wait只测发射等待；`ANY`尚未拆分VMEM返回、LDS返回、VALU依赖及barrier。LDS指令为76608、85.5/wave，bank-conflict raw为172032；事件数、指令数与墙钟周期分别保留。L2/UTCL1/EA证据与限制详见Memory文档第11节。

raw GRBM本次发现了具体的导出聚合问题：counter catalogue标明`DIMENSION_XCC=8`，JSON每dispatch保留八个raw值，而CSV把它们求和后写成一行。该CSV scalar不能当作`max(GRBM_COUNT)`。[clock审查](../opus_resume_20261008/diagnostics/clock_audit.json)保留八值及其顺序，不根据顺序猜XCC坐标；取JSON raw maximum排除这个求和问题后，K384的`maximum/duration`中位数仍为4.929GHz，51/51超过agent记录的2.4GHz；K16384为2.476GHz，46/51仍超过。对应独立profile duration中位数为8080ns与81600ns，不能转换成无profiler性能结论。

因此**仍不报告绝对MFMA利用率或动态occupancy，不套固定因子修正，不把counter cycles直接转ns**。Oct7的`OPUS_GFX950_*_MAX`使用显式`reduce(max)`，属于不同证据；本次raw-CSV问题不追溯改写旧结果。raw maximum与kernel timestamp是否覆盖同一时钟窗口仍待独立核准。动态MFMA/F8计数可以继续核对工作量，其恰好满足指令工作量的busy计数不能单独证明计算管线已饱和。

### 11.3 短K ATT：startup、scale producer及发布barrier

[ATT汇总](../opus_resume_20261008/diagnostics/att_analysis.json)每个K只有一次capture，仅SE0/CU0上四个完整wave各有一条拼接指令时间线。K384每wave48条动态MFMA，K16384每wave2048条，均与数学/tiling一致。以下均为shader clocks；成功MFMA issue时间按decoder的`time+stall`取值，`duration`已包含stall，不能重复相加。timeline端点最多超出wave end一个4-clock量化单位，汇总已裁剪。waitcnt dependency记录只建立队列成员关系，非单个访存返回时间，也可能省略空等待的动态出现。

| 四wave平均阶段 / shader clocks | K384 | K16384 |
| --- | ---: | ---: |
| wave总时长 | 9100 | 160241 |
| wave开始至首条成功MFMA | 4523，49.702% | 5625，3.510% |
| 首条至最后一条MFMA | 2807，30.848% | 153302，95.670% |
| 最后一条MFMA至wave结束 | 1770，19.451% | 1314，0.820% |

短K的首条MFMA前开销明显，不能把它概括为SFA consumer读/打包。4523 clocks包含：startup kernargs725、K0矩阵issue及scale kernargs1077、scale producer/K1 prefetch/publish barrier1249、scale与K0矩阵LDS读取585、runtime tail setup/accumulator clear/K2 prefetch887。两个SMEM kernarg wait PC`0x1f58/0x2150`的四wave中位数为624/548 clocks，同样属于启动链。

关键producer顺序已直接定位：短K只有SIMD0上的wave执行SFA vector16 VMEM`0x29e4`，随后`vmcnt(0)`在`0x29f8`等待并在`0x29fc`写LDS；SFB VMEM`0x2a64`此后才发出，再在`0x2a74`等待、`0x2a78`写LDS。第一处VMEM wait的依赖包含**八条K0 matrix async load与SFA**，196 clocks不能全部归给SFA。其余wave更早到达scale publication barrier`0x2b5c`：attempt跨度672 clocks，release跨度仅4 clocks，说明该位置存在producer完成时间与wave到达不均。它支持检验两种scale请求能否更早并行发出，仍不证明具体重排一定缩短整核时间。

最后Ktile还将BF16转换、C LDS写入及显式nop穿插到余下MFMA之间；最后MFMA到wave end的1770-clock平均段包含输出与endpgm。这些间隙不能列入稳态MFMA吞吐或scale stall预算。

### 11.4 长K ATT：普通边界、panel reload与tile内间隙分别判断

K16384的startup占比降至3.510%。在同一capture中，普通interior tile的第15条MFMA到下一tile第0条MFMA成功issue间隙有488个相关窗口，中位数492 clocks；其中包含下一tile矩阵LDS读取、runtime ring bookkeeping、八处async VMEM issue及nop。另有12个scale-panel-reload窗口，发生于tile30/62/94结束后的边界，中位数2014 clocks，额外包含SFA/SFB issue、VMEM wait、LDS publish及barrier。这些窗口属于同四个wave的重复行为，不是488/12次独立性能实验。

reload wait PC`0x3824`的中位数744 clocks同时涉及matrix与SFA队列；`0x3864`发布barrier中位数584 clocks。不能把普通边界的VMEM issue成本、reload等待和SFA字节读取混为同一个“HBM瓶颈”。tile内第3→4条MFMA间隙另有500个窗口，中位数184 clocks，包含下一tile scale LDS等待/打包、ring bookkeeping与`0x2f18`barrier；当前MFMA操作数已经在寄存器，局部issue仍被前置调度推迟。下一项稳态实验需要同时核对ring槽位覆盖和读者退休条件，不能凭这一个间隙删除barrier。

### 11.5 隔离实验设计：拒绝packed SFA，验证issue/publish

[SFA packed Event筛选](../opus_resume_20261008/results/sfa_packed_screen_analysis.json)已在同卡clean窗口完成十项5轮AB/BA、同51地址池的HIP graph Event，十项数值/guard/重复检查通过，但耗时中位数全部增加，故拒绝采用。代表`480×7168×384`为6.6142→7.1734μs，增加8.455%，5/5轮更慢。静态consumer读减少伴随producer byte LDS scatter、额外地址ALU与VGPR164→178；不能把“少三条consumer LDS读”直接等同于完成时间改善，也没有仅凭这轮Event证明某一bank-conflict机制是唯一原因。暂停的DPP producer准备不代表已编译或已测候选。

依据上述producer/barrier定位，新的[raw-scale issue/publish候选](../opus_resume_20261008/scale_issue_publish/source_changes.diff)保留原始SFA/SFB布局、vector16/byte读取、尾行填充、panel大小和原同步handoff，只在prologue及panel reload把“SFA load/wait/store后SFB load/wait/store”拆成“先issue两侧，再publish两侧”。issue与publish使用相同guard，未发出的lane不发布；数学、grid与K顺序不变，隔离实验时未改生产源码；后续精确应用见11.7。

[构建记录](../opus_resume_20261008/scale_issue_publish/build_manifest.json)和[device审查](../opus_resume_20261008/scale_issue_publish/device_audit.json)确认baseline仍匹配Oct7 selected，candidate只改变9021，9022指令/metadata/归一化descriptor不变。9021 VGPR164保持、SGPR98→99，LDS105504B、scratch/spill0；指令8588→8660B，静态MFMA、矩阵VMEM、scale VMEM、DS读写数量保持。[实际次序审查](../opus_resume_20261008/scale_issue_publish/isa_order_audit.json)确认prologue和refill的完整SFA路径中，A/B请求之间没有VMEM wait；M-tail仍为原16个有界byte load/wait/pack，不能声称该路径获得同样的重叠。源码准备manifest记录的是其采集时“未构建”状态，不改写该历史字段，后续状态由独立结果衔接。

### 11.6 完整Event复核与候选ATT：请求/发布次序是有效优化点

先完成[十项筛选](../opus_resume_20261008/results/scale_issue_publish_screen_analysis.json)，再独立完成[42个实际赢家及两个边界复核](../opus_resume_20261008/results/scale_issue_publish_confirmation_analysis.json)。两次均在PCI `0000:65:00.0` 的clean claim中使用同一51地址池、5轮AB/BA HIP graph Event；正式计时覆盖完整private调用，未加载profiler。每个输出按signed8独立FP32参考、guard和重复性检查，confirmation的704次原地址调用与22440个Event池输出检查全部通过。十项筛选覆盖5/42赢家；confirmation精确覆盖旧三份计划的42/42，不能把二者混为两次全域复核。

| Oct8 confirmation实际赢家集合 | 结果 |
| --- | ---: |
| median更快 / 更慢 | 41 / 1 |
| 5/5轮更快 / 5/5轮更慢 | 30 / 0 |
| 其余方向混合 | 12 |
| 按shape等权的几何平均加速 | 1.575621% |
| 各shape median之和，baseline→candidate | 1289.4490→1276.7109μs |
| 上述合成时间和的加速 | 0.997733% |

唯一负median为`416×7168×16384`，耗时增加0.034535%；该项4/5配对更快，配对加速中位数1.0605%，因此保留全部原始轮次，不把它写成稳定回退或全42项正收益。各K组的geomean均正：K384为3.655%、K768为3.543%、K1024为2.750%、K1536为1.248%、K3072为0.902%、K7168为0.869%、K16384为0.613%。集合统计不是测得的应用batch时间，也不代表未测全部支持域。

两个非赢家边界的代价已复现：`1×128×128`筛选/复核耗时增加1.429%/2.328%，均0/5轮更快；`15×128×256`为0.369%/0.454%，均1/5轮更快。其原SFA tail读取仍串行，不能把完整向量路径的机制收益外推到这里；现有两个点也不足以推出所有`K<384`均应走旧实现。按Oct7允许有限非赢家代价的同一标准，保留global9021候选，随后正式TU/API通过并应用；不新增未经验证的K阈值，不为把负值变正重复测试。

[候选ATT对照](../opus_resume_20261008/diagnostics/scale_issue_publish_att_analysis.json)核对每版本一次SE0/CU0捕获的完整device/FUNC/PC及四波各48 MFMA。producer的SFA→SFB成功issue间隔328→120 clocks，SFA issue到两侧LDS发布结束736→388，到publication barrier release968→620；两个VMEM wait事件总长588→188，但依赖含K0矩阵请求，不能当两条scale的纯返回延迟。另三wave的barrier时长从668/636/680降至268/220/208；consumer scale LDS wait仍为平均37+4 clocks。变化明确落在issue/publish重叠，未靠减少scale bytes或consumer读数。

四wave平均首MFMA前4523→4244 clocks，scale/K1/barrier段1249→913；相邻matrix/scale LDS段585→640、runtime/clear/K2段887→929，保留这些抵消变化。总wave跨度9100→8753只是一份有限局部capture对照，性能收益以无profilerEvent为准。独立数值/guard检查在已测目标验证了正确性，ATT instruction事件本身不能推导active-lane流量。源码和EXEC审查补充确认SFA的`opus::array`默认清零，SFB读取与发布使用相同mask；非producer的SFB publish可在EXEC=0时导出指令事件，不等于有效lane写入。正式TU身份与API通过后已应用，见11.7。

### 11.7 正式入口、源码应用与当前记录

[正式身份审计](../opus_resume_20261008/formal_selected/identity_audit.json)以Oct7最终selected为基线，26parent/56entry中仅9021的一个入口改变并精确匹配已测private候选，其余55entry保留；313个生成TU/host implementation、206个build object与linked module的202个gfx950 bundle、263个device entry已核对，未出现意外改变。两个worktree的路径差异仅用于构建，不当作指令差异；仅descriptor entry offset被归一化。

[八项正式API审计](../opus_resume_20261008/formal_selected/gpu_smoke_analysis.json)包含K128、M15/M17、K4096/K4224换panel、长K及未改9022对照，在clean PCI `0000:95:00.0` / HIP7完成64次official和64次private数值调用。实际加载module SHA、signed8参考、重复性及guard全通过，无计时。性能Event仍为PCI65的同卡实验，API正确性可以使用另一张干净卡。新official模块在Oct8独立JIT目录，后续默认JIT重建应使用当前应用源码和记录工具链。

精确已测9021源码已应用，见[Oct8集成记录](../opus_resume_20261008/formal_selected/integration_manifest.json)。header SHA为`2ff8cf90368c6945d3394fe11075c2336e6f8f99888c7167d3eb550045009bfa`，逐字匹配private候选及正式isolated tree；生产相对HEAD仍为两个文件，small traits保留Oct7状态，9022/9000/9020/共享helper及其余构建输入均不变。复用已完成验证，无需为逐字相同源码应用再跑GPU。

Oct7 source/build/result/manifests及其文档SHA保留历史时点，Oct8新增结论改变文档全文SHA，由Oct8最终文档清单衔接。当前状态`applied_verified_not_committed`，没有commit或push。


## 12. 2026-10-08剩余23个parent的有限优化与正式门禁

### 12.1 恢复基线、实际配置与证据范围

本节承接第11节的已采用9021，按[剩余inventory](../opus_remaining_20261008/inventory.json)收口23个public parent、44个有历史winner的实际device配置及536个历史winner。9020本轮已获用户授权；9000、共享helper、已采用9021保持冻结。基线仍是Oct8正式selected `module_deepgemm_opus.so`，SHA256 `6f6a0117b122f06b4802833c020effb4b24ff9fea56f20c8c16b9ced9787173f`，对应HEAD `b152ab834e68f8fde18adbd7b200c587770b9fb5`及先前small runtime B-scale alias。这里的actual配置以实际producer symbol/完整traits为准，同一parent的fixedK与runtime配置分别记录。

| 剩余范围 | public parent数 | 自身实际配置数 | 历史winner数 | 本轮最终决定 |
| --- | ---: | ---: | ---: | --- |
| 9020 | 1 | 7 | 76 | 仅保留已有fixed384配置候选，其余6配置保留当前版本 |
| 9022 | 1 | 1 | 159 | 拒绝global issue/publish候选，保留当前版本 |
| 9023 | 1 | 1 | 4 | 保留已有runtime配置候选 |
| 9024 | 1 | 1 | 9 | 保留已有fixed7168配置候选 |
| 9030 | 1 | 1 | 10 | 拒绝global midpoint候选，保留当前版本 |
| small/fine，详见12.7的18-parent表 | 18 | 33 | 278 | 四项候选拒绝，本轮没有新增采用 |
| 合计 | 23 | 44 | 536 | 本轮仅选定3个新device entry |

[当前44配置数值门](../opus_remaining_20261008/diagnostics/all_config_gate_results.json)已通过：每个配置各取一个自身actual winner，使用原Oct8正式selected、signed输入、2次repeat/reference/output及workspace guard，并记录实际加载的module SHA；该检查没有Event性能结果。small的33配置另有[逐配置审计](../opus_remaining_20261008/small_config_gate_audit.json)。**全部44配置均获得自身数值覆盖，不等于全部536历史winner获得本轮性能覆盖**，也不等于新正式selected的API门禁已结束。新候选完整winner性能覆盖分别为scale166、narrow13、9030的10以及small中9051的17；其他screen及局部ATT按自身有限范围单列，不能用共享pipeline的结论代替未测配置。

### 12.2 先定位startup及wave到达差，再判断可优化空间

[本轮ATT汇总](../opus_remaining_20261008/diagnostics/merged_att_analysis.json)核准当前module、CO、FUNC、full metadata、normalized descriptor、PC映射、完整wave及code/stat聚合，并使用clean同mm owner身份。gfx9成功issue仍按`time+stall`取值，duration已包含stall；shader clocks及wave份额不能换成整核墙钟损失，wait dependency只标记队列成员，EXEC未导出。原9030 capture的owner之外额外PID没有证明来源，不能事后猜为rocminfo而洗成clean；该记录保留为拒绝证据，采用后来独立clean补采。尚未核准的counter时钟窗口也不支持绝对MFMA利用率或动态occupancy。

| 当前exact body / 代表 | 完整wave数 | 首MFMA中位数 / clocks | wave时长中位数 / clocks | 局部机制含义 |
| --- | ---: | ---: | ---: | --- |
| 9020 fixed384 `[1536,7168,384]` | 8 | 3492 | 13876 | raw-scale启动链占较大比例，可检验更早issue |
| 9020 runtime `[1536,7168,16384]` | 8 | 9162 | 278388 | startup约3.3%，不能外推短K收益 |
| 9022 `[800,7168,384]` | 4 | 5016 | 10578 | 同一body短K的producer与发布等待明显 |
| 9022 `[800,7168,16384]` | 4 | 7022 | 213372 | 长K需区分普通边界与scale refill |
| 9030 `[65536,16384,1536]` | 688 | 8034 | 36834 | 86个八wave发布组，startup约21.8% |

9020 fixed384的producer先SFA wait496 clocks，再SFB wait376，其余wave的publication barrier可到1268；它支持raw-scale请求序列化诊断，但这些wait可能包含矩阵队列，不能按scale独立返回延迟计账。9020长K的普通interior窗口中位数2072 clocks，barrier wave份额20.13%，不表示可删除20.13%的整核时间。

9022短K只有producer wave执行SFA/SFB，相关wait为196/444 clocks，矩阵wait436，producer barrier8；其他wave barrier408–460。原SFA第二pass的合法guard在K384跳过，不能把静态存在的第二pass当作动态工作。长K每wave2560条MFMA，普通与refill混合窗口中位数1556，scale queue wait wave份额约1.59%；refill在tile30/62/94边界，需保留panel32的访问及发布同步。静态VGPR/SGPR/LDS计数只用于风险审查，不作为瓶颈证明。

### 12.3 Raw-scale issue/publish：安全版本与166项固定覆盖

[安全source/ISA复核](../opus_remaining_20261008/scale_issue/scale_source_review.json)确认9020把raw SFB提前到原SFA之前发出，`vmcnt(0)`与pack位于同一volatile asm block，等待之前不消费raw VGPR；9022先issue原SFA各pass及SFB，再按原guard发布。原始读取字节、SFA/SFB布局、scale-first矩阵prologue、panel32/refill、尾行guard、barrier及输出语义保留。两次拒绝revision继续留档：第一版编译器插入早wait而没有得到目标issue顺序；第二版把standalone wait与C++ pack分开后，编译器把pack提前到wait之前，数值失败。该unsafe版本不代表后来安全版本仍失败，也不能仅凭源码顺序宣称异步重排成立。

最终安全候选library SHA256 `10aabbbeef67cae83c9c18a0d0c307a5b5353dc836111cf546d949951c5ee7ea`、CO `706f335722d4d42ecfdf909ceb9e6fe16b9f706b44c971ebf98dc153fcff7702`。[166项独立审计](../opus_remaining_20261008/scale_issue/coverage_independent_review.json)覆盖已有9020 fixed384的全部7个winner及9022全部159个winner，均完成signed8/reference/guard/repeatability和5轮AB/BA、51次完整调用的共同自动地址池Event。Baseline使用未经改变的OfficialRunner包装，精确加载原Oct8正式module；此前私有9020 baseline有20字节差异，已排除出该覆盖依据。原结果没有逐项序列化`actual_module`字段；原OfficialRunner首次调用的已加载SHA核对通过，这个记录限制仍保留，不能补写成原始JSON已有该字段。

### 12.4 9020只保留已有fixed384，9022全局拒绝

[最终scope决定](../opus_remaining_20261008/scale_issue/retention_decision.json)保留9020**已有fixed384**，不新增shape或K阈值。7/7 shape Event中位数及paired中位数正，6个shape为5/5轮更快，另一个4/5，共34/35个positive pair；等shape几何平均speedup为+2.571542%，shape耗时中位数之和的speedup为+2.800398%。当前没有actual winner稳定5/5退化。该entry VGPR216、SGPR54、LDS119840、scratch及spill均保持，ISA6284→6312字节；这些资源没有变化本身不证明性能收益。

必须保留的负样本是`[12288,7168,384]`的**round2**：baseline `67.43829390581917 μs`，candidate `80.80135607251934 μs`，speedup `0.8346183428566867`，即**该轮耗时增加19.8152435%**。其余4轮改善约2.55%–3.00%，该shape总体中位数仍正；但该shape AB几何平均为−4.1740986%，全部7个shape的round2 pooled几何平均为−0.493013%。全部7个shape的AB/BA分别+1.6153806%/+2.6637477%。该轮处于clean epoch，原因未知，没有丢弃、归零、认定contamination或重跑寻找正结果。

机械分析器要求每一pooled round皆正，因此曾建议拒绝9020 fixed384；**原机械JSON保持不变，最终root scope决定明确以逐shape中位数/paired稳定性取代该机械建议，并接受上述孤立大代价风险**。这属于有限证据下的显式决定，不是改写异常。9020其余6个已有body只有各自一个screen代表，结果mixed；fixed3072代表speedup退化−2.70047%且0/5轮更快。因此global9020候选拒绝，不能把fixed384的完整7项收益扩大为9020全部76个winner收益。

9022全159项等shape几何平均虽为+0.820755%，shape中位耗时和speedup仅+0.429064%，127个ratio中位数正、32个负，79个5/5更快，却有6个actual winner **5/5更慢**。VGPR194、LDS81184不变，SGPR66→64、scratch及VGPR/SGPR spill为0，不抵消稳定退化。

| 9022 K cohort | winner数 | 等shape GM speedup | 稳定5/5更慢数 |
| --- | ---: | ---: | ---: |
| 384 | 22 | +1.786601% | 0 |
| 768 | 22 | +1.763129% | 0 |
| 1536 | 30 | +0.309205% | 1 |
| 3072 | 22 | +0.415938% | 1 |
| 7168 | 43 | +0.752891% | 1 |
| 16384 | 20 | +0.095636% | 3 |

K16384还存在10/20个ratio中位数负、12/20个paired中位数负及pooled round2负。六个稳定loser的shape及ratio speedup为`[512,16384,1536]` −1.153093%、`[704,7168,3072]` −0.556446%、`[896,7168,16384]` −0.406775%、`[1024,7168,16384]` −0.830989%、`[1280,7168,16384]` −0.818175%、`[4096,2048,7168]` −0.157735%；第一项paired中位数为−0.239276%，ratio与paired统计分开保留。已有9021/narrow接受有限代价时没有actual winner稳定5/5退化，本次不能用整体GM覆盖global9022的稳定loser。最终拒绝整个原global entry候选，保留当前9022，不发明短K阈值，不事后挑positive subset，不追加重测。

### 12.5 Narrow：保留9023 runtime与9024 fixed7168，并列资源代价

[独立narrow决定](../opus_remaining_20261008/narrow_candidate/retention_recommendation.json)与[所有静态副本CFG审计](../opus_remaining_20261008/narrow_candidate/isa_order_audit.json)支持保留原9023 runtime entry及原9024 fixed7168 entry。候选在完整vector路径先issue原SFA low/high及SFB，再publish；原guarded helper、显式wait/barrier、尾helper及epilogue保留。9023五份full-vector静态copy及9024一份均核对实际ISA顺序；partial byte helper仍可能序列化，不声称全支持域都重叠。

9023全部4个actual winner，含K7168/K16384、M544尾行及M576整行，加M16非winner边界；9024 fixed全部9个actual winner，另带unchanged runtime control，共15个target完成signed8/reference/repeat/guards及clean共同51地址池、5轮AB/BA完整调用Event。9023四winner GM speedup为+1.117277%，两个K组的聚合分别+1.916205%/+0.324612%，全四shape的五个pooled round均正，但longK分shape及其pooled round仍mixed，不把“两个K组总体正”写成“longK每轮正”。接受`[544,7168,16384]`中位耗时+0.542160%、2/5轮更快，以及M16边界+0.416545%、1/5轮更快；没有actual winner 0/5更快。

9024 fixed九winner GM为+1.274511%，九项中位数正且45/45 pair更快。未修改的9024 runtime control变化+0.486590%属于过程/噪声证据，不归因源码。9023 fixed及9024 runtime均保持原device身份，二者是合法support control，不属于本节44个历史winner配置。

**9023资源风险随决定保留**：VGPR232→251（+8.19%），SGPR lane spill46→48，ISA25976→26328（+1.36%）；LDS78112、private、VGPR spill及AGPR不变。额外live寄存器收窄后续编译器余量，本次没有证明动态occupancy变化。9024 fixed VGPR110及zero spill/LDS71744保持，SGPR58→60，ISA7276→7408（+1.81%）。有限call收益与这些风险分别记录，不把静态资源当作饱和证据。

### 12.6 9030 midpoint：同步证明成立，完整10-winner覆盖拒绝

[八wave cohort分析](../opus_remaining_20261008/large_midpoint/cohort_analysis.json)在clean补采中核准688个完整wave、86个发布组，每wave288条MFMA。774个interior group/tile、6192个相关wave/tile中，arrival spread中位数908 clocks，last MFMA spread900；latest arrival→release仅4，latest barrier8，而all-wave barrier duration中位数356。barrier的wave份额约22.15%，矩阵VMEM issue等待4.14%、LDS issue等待1.89%，与wave到达差一起判断，不能宣称删除barrier就收回22.15%整核时间。组合VMEM/LDS wait中位数76、latest84，latest matrix issue764且stall688；它们不是独立访存返回时间。

异常group8 tile6仍保留：first/last MFMA spread5388/5396 clocks，早四wave组合wait约4830、晚四waveasync issue约5500。没有排除样本。候选只把普通advance的`vmcnt(0)/lgkmcnt(0)+barrier`移到当前前8条MFMA之后，再issue K+2；同步证明要求先publish K+1、退休所有K读者后覆盖K+2，并保留最后transition的C-alias retirement barrier及loops1原prologue barrier。原scale-first prologue、loop/layout/guard、C64资源和output保持，[source/ISA审计](../opus_remaining_20261008/large_midpoint/isa_source_review.json)通过。VGPR203/LDS143360不变，SGPR50→53。

原M16385计划违反M%64，由baseline API先拒绝；保留在`rejected_prepare_revision_misaligned_M/`，随后合法`[16448,65536,K128..640]`五种terminal guard通过。候选library SHA256 `2bf61c8d18bb444fe9c71a9ff8de6cf5d42c1a16cfcb21e0797a8286e9f5efb9`，CO `5cee4b4df6b4083de16cca1839e9b1d3aa2f1c1a006480236c570310b07f1309`。大输出采用256-row分块FP32 reference，8个shared地址、5轮AB/BA、51次完整调用，记录actual module、完成状态及helper SHA。

首个true-winner screen曾有+0.261% ratio/+0.226% paired中位数、5/5更快；预声明的全部10个K1536 actual winner覆盖却不复现：[独立覆盖审计](../opus_remaining_20261008/large_midpoint/coverage_independent_review.json)数值/guard/clean identity通过，但10/10 ratio中位数负、9/10 paired中位数负、0/10同时在AB与BA正，仅4/50 pair更快。等shape GM ratio为−0.220256%、paired为−0.166456%。两个合法非winner control继续保留：K384 ratio−0.550244%、paired−0.587315%、0/5；K16384 ratio−0.933701%、paired−0.069249%、2/5，AB约−2.55%/BA约+0.28%。[全局决定](../opus_remaining_20261008/large_midpoint/global_decision.json)拒绝该midpoint，保留当前9030，没有新增K gate、posthoc subset或重新测试到正。

### 12.7 Small/fine：按自身33配置收口，四候选全部拒绝

以下18-parent表和对应范围从本轮新[small_append](../opus_remaining_20261008/small_final_document_append.md)引用。其逐symbol记录区分自身数值门、局部CU ATT、本轮candidate Event及历史exact-config Event。机制诊断仍有限：9040 startup纯`vmcnt`仅4 clocks；9047的PC `0x2540` 是`vmcnt(0)+lgkmcnt(0)`混合wait，平均约169 clocks（128–196），不能概括为VMEM等待近零。runtime9051本轮全部17个winner中另有3个shape稳定5/5更慢，故早期正screen不支持采用或事后切shape。

small与fine分支本轮按33个有历史winner的实际producer symbol配置收口，映射18个public parent、30个actual ID和278个历史winner。候选性能实测严格为4个配置、21个actual-winner形状：fixed9062 initial matrix-before-scale覆盖M144/M160两个winner且完整调用包含matching split2 reducer；9046与9055 steady ring issue-before-read各测一个自身winner；runtime9051 scale-first覆盖其全部17个winner。四个exact配置的候选均拒绝并保留原selected：fixed9062两项speedup0.996596/0.998854，9046/9055为0.980340/0.991871；9051完整17项为8项median正、8项退、1项平，仅3项全部5轮更快。所有自身signed8repeat、reference、重复及guard和clean sharedpool完整Event审查通过。两个9051早期正代表仅是进入固定覆盖的screen依据，全17 mixed结果作统一拒绝，不按shape新增阈值或重跑寻找正结果。本轮small没有新增source/device entry采用，9042/9053/9054此前已采用runtime B-scale alias保留，9071草稿在build前停止。

证据范围分别保存：7个exact配置各有一个自身有限形状的局部CU ATT，33个配置各自一个actualwinner的signed2repeat/reference/output/workspace guard数值门；它们分别证明局部机制和正确性，均不扩大为全部winner性能覆盖。9055原CU0 code=null/零wave记录排除，CU1补采独立核过身份；9051的32个ATT wave跨多个WG，聚合同PC barrier跨度不用于WG内到达差。该small正式入口集合为35个producer与5个独立reducer，其中33个producer与历史winner相交、两个兼容/runtime producer无历史winner；5个reducer保留机器身份，未声称每个reducer有独立性能实测。各配置的keep依据自身selected source/type、正式FUNC/fullmetadata/descriptor、数值门及存在时的自身历史exact-config Event；共享ring/register/fine机制说明、静态资源或其他配置reject都不算该配置性能。完整配置记录见 [small审查](../opus_remaining_20261008/small_family_review.json)，范围核验见 [small最终审计](../opus_remaining_20261008/small_final_closure_audit.json)。

| public parent | actual配置（同ID按fixedK区分） | 历史winner数 | 自身本轮候选Event形状 | 自身ATT配置 | 自身数值门配置 | 最终状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 9040 | 9040 | 11 | 0 | 1 | 1 | keep自身当前配置 |
| 9041 | 9041, 9050 | 17 | 0 | 0 | 2 | keep自身当前配置 |
| 9042 | 9042, 9071, 9073 | 15 | 0 | 1 | 3 | 保留已采用runtime alias；其余配置keep；9071保留N48 |
| 9043 | 9043 | 2 | 0 | 0 | 1 | keep自身当前配置 |
| 9044 | 9044, 9056 | 26 | 0 | 0 | 2 | keep自身当前配置 |
| 9045 | 9045 | 17 | 0 | 0 | 1 | keep自身当前配置 |
| 9046 | 9046 | 32 | 1 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9047 | 9047 | 27 | 0 | 1 | 1 | keep自身当前配置 |
| 9049 | 9049 | 6 | 0 | 0 | 1 | keep自身当前配置 |
| 9051 | 9051 | 17 | 17 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9052 | 9052, 9070 | 24 | 0 | 0 | 2 | keep自身当前配置 |
| 9053 | 9053, 9072 | 9 | 0 | 0 | 2 | 保留已采用runtime alias；其余配置keep |
| 9054 | 9054 | 3 | 0 | 0 | 1 | 保留已采用runtime alias；其余配置keep |
| 9055 | 9055 | 17 | 1 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9060 | 9060/K0, 9060/K3072, 9060/K7168 | 9 | 0 | 0 | 3 | keep自身当前配置 |
| 9061 | 9061/K0 | 19 | 0 | 0 | 1 | keep自身当前配置 |
| 9062 | 9062/K16384, 9064/K0 | 6 | 2 | 1 | 2 | 保留；拒绝自身有限候选 |
| 9063 | 9063/K0, 9063/K16384, 9065/K0, 9066/K0, 9067/K0, 9068/K0, 9069/K0 | 21 | 0 | 0 | 7 | keep自身当前配置 |

表中0项candidate Event表示该parent本轮保留自身配置，没有冒称该parent全部winner已测或已获得新性能增益；历史exact-config Event另存逐symbol记录。


### 12.8 三个selected entry正式验证并应用

本轮[最终selection](../opus_remaining_20261008/formal_selected/final_selection.json)只有**9020 existing fixed384、9023 existing runtime、9024 existing fixed7168**三个新entry；先前9021及small alias继续保留。原有guard/dispatch阈值沿用，没有把候选推广到support-only或其他配置。[正式CPU identity审计](../opus_remaining_20261008/formal_selected/identity_audit.json)已通过：26个public parent、56个device variant中3 changed/53 unchanged；三个changed的FUNC/full metadata/normalized descriptor精确匹配已测private candidate，所有其余entry精确匹配Oct8，9000/9010/9021冻结。313个generated file、206个build object（202个device object）、202个linked gfx950 bundle及263个linked device entry核对，0 failure。新正式module SHA256 `4be55119596805e3d2967a00c3cbf70129f280094d14d43fbbb2461f91aee1bd`；此identity通过本身不产生新性能结论。

[44项正式API审查](../opus_remaining_20261008/formal_selected/api_gate/gpu_api_analysis.json)已全部通过，在clean PCI `0000:65:00.0`完成352次official与120次private signed8/reference/repeat/guard调用，实际加载module SHA匹配正式identity；44个唯一owner及94个monitor由[独立API复核](../opus_remaining_20261008/formal_selected/api_gate/independent_gpu_api_review.json)确认。44个API target与12.1的44个历史winner实际配置是两个不同集合，不能混称全配置门禁或536项性能覆盖。检查覆盖26个public parent、三个selected entry及必要panel32/33、M17和grouped尾边界，无性能计时。

三个精确源码文件已应用，状态`applied_verified_not_committed`，见[本轮集成记录](../opus_remaining_20261008/formal_selected/integration_manifest.json)。[应用核验](../opus_remaining_20261008/formal_selected/independent_application_review.corrected.docs_authorized.json)确认生产全部2691个构建输入与正式selected worktree逐字相等，9000/共享helper/已完成9021/原small alias保留；旧4972份冻结证据SHA保持，原Compute/Memory第12节前字节保持。首次应用审计的Path与字符串排序差异、随后把授权文档追加误纳入冻结检查的失败记录均保留，由修正报告衔接。没有commit或push。

API运行时command fingerprint包含生产源码diff；应用后直接重算会变化。独立审查用preapply的2691输入与旧Oct8 worktree身份重建运行时diff，44/44 fingerprint一致，未把后续授权应用当作GPU污染。

[独立CPU dispatch复核](../opus_remaining_20261008/formal_selected/api_gate/independent_dispatch_review.json)确认44个合法且唯一target/26个parent、逐项branch与CPU计划一致，恰好三个changed symbol；changed target15个，9020=3、9023=8、9024=4，9023 fixed与9024 runtime control unchanged。target32 `[9020,16,256,384]`的purpose称fixed384 minimum tail，但M<1024实际落unchanged runtime branch6，故只算合法control；已有changed fixed384 `[1472,7168,384]`覆盖M%192=128尾行。该purpose问题非阻断，原plan/audit保持，不把它充作changed-entry证据。

本节保存机械建议被最终scope决定取代、两次scale坏revision、9030非法M计划、ownership歧义及所有负样本。当前性能决定基于各配置自身的有限clean Event，仍不声称全支持域获益、全部536 winner已测、绝对MFMA饱和或动态occupancy已知。


## 13. Oct8 9000/9010 独立候选与全部 bound 分类

本次用户要求把9000/9010分别优化并拆成少量可选候选。最终仅新增9001与9011，原9000/9010保持原机器entry，由现有tuner按实测选择。9001是SFA raw先清零（VGPR477→469），全129自身历史winner GM−0.0487%，禁止整体替换；两原5/5正样本的新seed仍median微正但仅3/5轮，保留为证据较弱的可选候选。9011是padded-M unroll4（VGPR497→492），全38自身winner GM+0.2808%、34正median、0个5/5 loser；4个negative median保留，原9010可回选。9000 unroll4全129 GM−0.3003%、46个5/5 loser，淘汰，候选不继续增殖。

最终源码状态与证据见[本轮记录](../opus_9000_9010_bound_20261008/README.md)及[正式集成manifest](../opus_9000_9010_bound_20261008/split_formal_scale_reset/integration_manifest.json)。28个公开parent/58entry，完整204个linked gfx950 bundle/265entry；所有原263 linked entry的FUNC/full metadata/normalized descriptor精确保持，新增两个精确匹配各自已测private候选（新traits仅改变metadata .name/.symbol）。14个official target/112次数值调用通过signed8/reference/repeat/guard、实际module SHA和严格GPU owner。正式module SHA为`e18fe59bf6b06ddc53349df5eaa5a3f5d31d3b00b75627705c26477d09db00eb`；六源码文件已应用，原9020/9021/9023/9024与small aliases保持。没有commit/push，未改保存tuning CSV。新ID需下次tune或显式API选择。

### 13.1 四种 bound 按完整系列、实际配置和代表形状归纳

依据[最新带宽文档4.2](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md:789)，完整分类见[逐parent/逐entry表](../opus_9000_9010_bound_20261008/BOUND_CLASSIFICATION.md)、[JSON](../opus_9000_9010_bound_20261008/bound_classification.json)和[CSV](../opus_9000_9010_bound_20261008/bound_classification.csv)。原26parent/56entry全部列出，包括47个历史winner producer配置、四support producer和五reducer；新增后28parent/58entry。745是历史winner归属，不能写成本次全745性能重测。

| 范围 | 类型方向 | 证据限制 |
| --- | --- | --- |
| 9000/9010长K | compute/issue与LDS/VMEM供数混合 | 自身ATT和clean SQ/EA；MFMA局部占时高，不等于校准全卡计算峰值 |
| 9000/9010短K | memory latency贡献＋启动/输出固定成本 | 不能强行把所有固定工作归为dispatch；新9001/9011未重采post-change ATT |
| 9020 | fixed384 scale latency；128x128 runtime VMEM依赖；其余compute/issue与供数混合 | 七实际body各用自己代表；fixed384采用后的主瓶颈未重采 |
| 9021/9022 | 短K scale请求/publish latency；长K matrix issue、LDS、panel refill混合 | 9021优化因果有前后ATT/Event；不称全部latency已消除 |
| 9023/9024 | memory latency贡献与compute/issue混合 | runtime/fixed7168采用前baseline机制；support/control body未独立分类 |
| 9030 | compute、供数、输出服务混合 | clean4.267GB/2.923TB/s不是HBM饱和证据；barrier cohort skew不等于可删墙钟 |
| 9040/9047及其他短K | memory latency贡献＋启动设置 | 约76%/66% startup和SMEM等待是线索，不证明dispatch |
| 9041/9042/9052/9053/9054 register | VMEM request issue与依赖，偏memory latency | runtime与各fixed/internal配置分开；无外部带宽饱和证明 |
| 9043–9046/9049/9055与9060–9063 producer | local LDS依赖、issue、同步；部分compute混合 | local供数层级明确；不能仅据LDS issue定为LDS bandwidth饱和 |
| 9051 | memory request/latency压力，可能bandwidth成分 | 旧clean5.962TB/s、LFIFO24.03%、UTCL1在途6.18%；无法唯一归为HBM bandwidth |

9040+的33个winner producer现在均有自身ATT：旧7加本次补26。新26项逐一通过CO/FUNC/full metadata/normalized descriptor、完整MFMA wave计数和严格owner检查；9054 K768为8/8/16/16、WG总48，12仅算术平均。自身代表不等于全部支持域，不以别的配置失败代替自身测量。所有memory latency结论明确VMEM/SMEM/LDS层级，wait queue成员不提供单请求返回时间。

当前没有可靠证据确认某个整个系列为纯HBM bandwidth bound或dispatch bound。Dispatch还需动态可运行wave、SPI/资源分配stall与完成时间影响；224WG/256CU、静态VGPR/LDS和长startup均不足。GRBM窗口未校准，不推断绝对utilization、动态occupancy或shader clocks→ns。新增9001/9011只有自身性能/数值与机器身份，最终主瓶颈标未确认；四support与五reducer也保留明确未确认状态。

### 13.2 有效决策与排除记录

9000 scale-reset全129由clean25项51地址池与clean串行104项8地址池组成，各同批基线/候选五轮AB/BA、51调用/graph Event；只汇总各shape ratio，非生产负载加权。9011全38为51地址池。首次大输出性能和首次中间9010替换API出现非owner PID，已排除并串行重跑；全历史owner_audit aggregate为failed仅因保留这两次已排除attempt，最终选用epoch均strict_clean=true。旧排除raw、负样本、资源失败和中间9010替换manifest保持，当前方案以独立9001/9011集成manifest为准。

原第13节前全部字节保持；本次文档追加与最终哈希见[final_manifest](../opus_9000_9010_bound_20261008/final_manifest.json)。全部有限实验已收口，不将全系列分类等同于全系列已获得优化收益。


## 14. Oct8 全28候选、745shape重新tune（8卡）

全量结果见[retune报告](../opus_retune28_20261008/README.md)，状态completed。原9000/9010保留，9001/9011与其他全部公开候选纳入，总28parent。745shape完成、13,027/13,027合法OPUS配置errRatio=0，全后端79,504条raw。开始150shape单卡，用户要求8卡后其余595shape按shape分组并行；每个shape的全部候选同卡计时，全部KFD握手、物理锁和严格非owner PID审计通过。

同轮原26最优vs完整28最优：9001选59、9011选19，全集合GM+0.036518%，单次时间和−0.039968%；78项单轮改善，最大延迟降幅1.371630%。这是新增候选的筛选增量，原26也使用当前已采用的其他优化；min统计有选型噪声，不等于稳定收益。

78项五轮shared-eight-address profiler复测，seed29 signed native E8M0，257次数值校验和1,285条计时。9001原59中29项中位数继续胜，9011原19中12项继续胜；仅2/3项分别5/5快，9011有2项5/5慢。重选后的新ID数量9001为30、9011为12（包含1项9011转9001）；证据来源分别保留，不把不同地址池/seed复测时间混入筛选GM。raw与复测config都在新目录，旧生产默认CSV保持。

OPUS28对同轮CK/CKTile/ASM最快者为720快/25慢、GM1.255110×；全后端选择OPUS720/ASM24/CK1。与Sep30旧OPUS表649快/96慢、GM1.041807×仅描述跨轮变化，不能归因于新两项。大输出>32M elements地址池上限8、reference/comparison256行分块；最大8GiB输出全部完成，旧自动轮换时间不能用于因果A/B。外部无效CK498/CKTile874记录保留，不参与选型。

bound仍按第13节的实际配置/阶段证据；本轮没有新的ATT/counter，不证明任何整个系列为纯HBM bandwidth或dispatch bound，9001/9011优化后的主瓶颈仍未确认。[完整58配置分类附本轮数量](../opus_retune28_20261008/bound_classification_with_retune.csv)明确parent计数不能跨重复device行求和。正式OPUS SO SHA仍为`e18fe59bf6b06ddc53349df5eaa5a3f5d31d3b00b75627705c26477d09db00eb`，最终审核和新增文件哈希见retune报告。原第14节前全部字节保持，不改变旧manifest所描述的历史快照。没有commit/push。
