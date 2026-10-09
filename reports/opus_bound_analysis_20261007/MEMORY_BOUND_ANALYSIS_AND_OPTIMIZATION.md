# Opus 9000 系列：Memory Bound 分析与优化方式

**2026-10-08 已验证并应用：**分层counter及实际ATT把短K的限制收窄到startup中的scale请求串行与发布barrier；新issue/publish实现相对Oct7 selected的42个实际赢家geomean加速1.576%，两个极小M非赢家路径有已复现代价。正式device身份及八项API、128次数值调用通过，已应用9021单文件，未提交或推送，见[第11节](#11-2026-10-08分层counter与att把限制收窄到具体接口)。GRBM窗口仍不支持绝对利用率，旧记录保留历史身份。

日期：2026-10-07  
审查起点：`b152ab834e68f8fde18adbd7b200c587770b9fb5`  
目标：当前 `gfx950`、原生紧凑 E8M0 scale、MXFP8 B-preshuffle GEMM 候选。

本文按[最新带宽章节总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)的第三章模型与第四章诊断流程，解释当前候选的真实工作分配、访存量、等待和优化验证方法。计算管线与大 tile 的展开见[Compute Bound 文档](/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_bound_analysis_20261007/COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md)。

## 1. 当前结论与证据范围

本轮已将 **9021 prologue起步重叠，以及9042/9053/9054显式B-scale复用** 应用到当前checkout，共两个源码文件、四个改变device入口，未提交或推送。应用文件与已测selected源码SHA精确一致，见[当前集成记录](formal_selected/integration_manifest.json)。

- 9021：全部42个受影响当前赢家完成共享地址池Event，几何平均吞吐增加3.155%，逐shape中位数求和的合成时间口径，吞吐增加2.313%，详见Compute文档9.1。
- register三alias：全部6/6受影响当前赢家均5/5轮更快，完成时间减少0.44%–2.59%；两个非赢家路径保留0.81%/0.64%的小幅时间代价，见8.6。
- 最终selected：4改变入口匹配已测候选、52不变入口匹配baseline；12项正式API检查、176次数值调用通过，见[selected严格结果](formal_selected/gpu_smoke_analysis.json)。初始含9022的15项检查属于历史candidate。

9022全局prologue已拒绝：159个当前赢家的几何平均吞吐虽增加1.390%，K16384的20项下降0.468%，其中9项5/5轮更慢。当前9022保留原实现。以下表格记录其它实验的实测取舍。

| 方向 | 已确认的静态事实 | 本轮实测与保留决定 |
| --- | --- | --- |
| `register_reuse` | 对齐的 N16/N32/N64 tile 可以复用 B scale；实际 ISA 的 byte load 与 VGPR 数减少 | **拒绝全局默认值修改**；9042/9053/9054显式开启已采用，6/6受影响当前赢家、5机制、6异常Event及最终selected 12项正式API完成 |
| `fine_wait` | Fine-M 各 wave 的 direct-LDS 请求数不完全相同；按 wave 调整 wait 可保留更多未来请求在途 | 九个干净Event目标中五项5/5回退，其余接近0且方向混合，**不采用本轮全局wait修改** |
| `fine_n64` | M80 split2 的 BN128 改 BN64，LDS 和 VGPR 减少，工作组数增加 | 后续两个目标clean Event均5/5退步，**拒绝BN64替换，保留BN128**；旧污染轮次仍排除 |
| `fixed_n32` | K7168的N48输出tail改N32，VGPR/LDS减少；baseline机器码与正式分支完全相同 | 两个目标的无干扰shared-pool Event确认均退步，**拒绝采用** |
| `narrow_unroll` | runtime K减少展开后VGPR与scalar spill减少，fixed-K保持相同 | 六项clean Event未形成可靠整体收益，**9023/9024全局runtime展开修改不采用**，详见Compute文档 |

编译和机器码证据不能等同于性能收益。**本轮 GPU 正确性、A/B 时间、counter 与最终保留范围统一记录在第 8 节。** 已确认拒绝fixed_n32、fine_n64、register_reuse的全局默认值方案及fine_wait的全局修改；register三alias限定方案已验证并应用到当前源码。可选646项支持域扫测停止于210个干净case，不声称全域通过；两个非赢家路径有0.81%/0.64%时间增加，见8.6。旧profile仅用于挑选优先验证的shape。

这里分析外部带宽限制、缓存/接口吞吐限制和访存延迟。请求供给不足还要区分访存并发不足与独立的 dispatch bound，不能统一归为 memory bandwidth bound。低算术强度说明访存可能重要；是否已经 bandwidth bound，需要真实流量、延迟、回压和并发共同支持。

## 2. 先确定实际运行的 kernel

已逐项核对审查起点745个历史胜出shape的实际派发及资源，见 [shape_inventory.csv](shape_inventory.csv) 与 [完整机器清单](shape_inventory.json)。清单保存baseline正式代码对象、scale/workspace逻辑字节预算和重复A/B请求估计；本文的“当前赢家”指该固定shape/dispatch集合。采用后的四个入口身份由[selected审查](formal_selected/identity_audit.json)与[集成记录](formal_selected/integration_manifest.json)确认，旧资源清单不会因此自动更新。预算用于选择测量方向；物理流量和动态瓶颈仍须counter验证。

### 2.1 当前候选组织

默认 tuning 枚举 26 个公开候选：

```text
9000、9010、9020–9024、9030、9040–9047、9049、9051–9055、9060–9063
```

较早配置保留显式调用，部分成为公开候选的内部 dispatch。公开 `kernelId`、CSV 中的 `kernelName` 和实际 device 模板参数可能不同。分析时必须记录最终 `BM、BN、waves、K-wave、stages、cluster、split、fixed-K、output、cache policy`。

| family | 公开候选及主要 tile | 访存分析关注点 |
| --- | --- | --- |
| 原始/兼容 | 9000/9010，256×256，4 waves；9010 支持 M padding | 大 tile 数据复用、M 尾部有效工作比例、流水填充与 drain |
| merged | 9020：192×256/8 waves；9021：128×128/4 waves；9022：160×128/4 waves | 实际 fixed-K 分支、矩阵 LDS/scale 路径、tile 复用、并发与尾部 |
| narrow | 9023：64×128/4 waves；9024：64×64/4 waves | N 窄时的 grid 数、A 重复请求与较小 tile 的并发取舍 |
| 大输出兼容 | 9030：192×256/8 waves | 仅覆盖 `INT32_MAX < 2MN <= max_tensor_bytes` 的 C 输出域；C 的 64 位基址/tile resource，A/B 仍保持有符号 32 位字节契约 |
| small register | 9040–9042、9051–9054 | K-wave 合作、VGPR prefetch、重复 scale load、局部 FP32 归约 |
| small LDS / register scale | 9043–9047、9049、9055 | direct-LDS、LDS ring 深度、scale 存放、wait/barrier、尾部 |
| fine-M / global split | 9060–9063 及内部 9064–9069 | M padding、split-K grid、workspace/reducer 成本、每 wave 请求差异 |
| fixed-K register tail | 内部 9070–9073 | K7168 特化、N48 尾部、B scale 是否能跨 fragment 复用、输出归约 |

9020 在特定大 M、窄 N 情况会切到 128×128/8 waves；K384/768/1536/3072/7168 也有编译特化。9023/9024 的 K7168 窄 N 情况有额外特化。必须核对当前 codegen 条件，不能按 family 的默认 tile 推断一次具体执行。

依据：[canonical registry](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/opus_gemm_common.py:1874)、[实际 launcher 生成](/root/workspace/aiter-opus-mxfp8-bpreshuffle/csrc/opus_gemm/codegen/gen_instances_gfx950.py:2977)。历史合并映射见 `reports/opus_consolidate_20260930/candidate_mapping.csv`。

### 2.2 Small 候选的真实配置

`P` 为 register prefetch 深度；`S/C` 为 LDS stages/cluster。`K waves` 表示一个工作组内部沿 K 分工，其余 waves 为 M/N 分工。下表记录baseline配置；本轮仅为9042/9053/9054的runtime alias显式开启B-scale复用，其余列与原派发条件保持原配置。

| ID | BM×BN | M/N waves | K waves 或 S/C | 关键区别 |
| --- | --- | --- | --- | --- |
| 9040 | 16×32 | 1/1 | K1，P6 | Output3，B cache3 |
| 9041 | 16×16 | 1/1 | K8，P2 | Output0，长 K/小 grid 可切 9050 |
| 9042 | 32×32 | 1/1 | K4，P3 | Output3，**B cache0**；K7168 切 9071/9073 |
| 9043 | 32×64 | 1/4 | S8/C2 | LDS scales，early scales |
| 9044 | 64×64 | 2/2 | S4/C1 | early scales、prefetch-before-read、read-only drain |
| 9045 | 96×64 | 2/2 | S4/C1 | Output1，LDS 输出重排、read-only drain |
| 9046 | 64×128 | 4/2 | S6/C2 | 8 waves，LDS scales |
| 9047 | 32×64 | 2/2 | S4/C1 | register scales |
| 9048 | 32×64 | 2/2 | S4/C1 | register scales + XOR LDS；历史显式候选 |
| 9049 | 32×128 | 2/2 | S4/C2 | register scales + XOR + prefetch-before-read |
| 9050 | 16×16 | 1/1 | K8，P3 | Output3，B cache3；内部 9041 分支 |
| 9051 | 16×32 | 1/1 | K4，P3 | Output3，B cache3 |
| 9052 | 16×32 | 1/1 | K8，P2 | Output3，B cache3；K7168 切 9070 |
| 9053 | 32×32 | 1/1 | K8，P2 | Output3，B cache3；K7168 切 9072 |
| 9054 | 32×64 | 1/1 | K4，P2 | Output3，B cache3 |
| 9055 | 32×64 | 1/4 | S12/C4 | 深 LDS ring，early scales |
| 9056 | 64×64 | 2/2 | S8/C2 | 内部 9044 深 ring 分支 |

当前 9048 没有动态进入 9047 的规则，移出默认 tuning 后仍可显式选择。不要把历史 parent 映射理解成自动 dispatch。

| fixed-K ID | BM×BN | K waves/P | B scale 复用 | 输出 |
| --- | --- | --- | --- | --- |
| 9070 | 16×32 | K8/P3 | 是 | Output4 |
| 9071 | 16×48 | K4/P4 | 否 | Output4，N16 尾部 mask |
| 9072 | 32×32 | K4/P4 | 是 | Output4 |
| 9073 | 32×48 | K4/P3 | 否，B cache0 | Output4，N16 尾部 mask |

9070–9073 只接受 K7168。Output4 把 FP32 局部归约分给 K waves，最后转 BF16；它们没有全局 split workspace。

### 2.3 Fine-M 与 dispatch 条件

Fine 默认 BN128；均为 LDS scales、early scales、prefetch-before-read、read-only drain，通常 S4/C1。

| ID | BM / waves | global split | reducer 与特化 |
| --- | --- | --- | --- |
| 9060 | 80/4 | 1 | K3072、K7168 固定 K |
| 9061 | 96/8 | 1 | runtime K |
| 9062 | 80/4 | 2 | K16384 固定 K，Vec4/Block128/cache0 |
| 9063 | 80/4 | 4 | runtime Vec16/Block128/cache2；K16384 Vec4/Block128/cache0 |
| 9064 | 96/8 | 2 | runtime Vec16/Block128/cache2 |
| 9065 | 96/8 | 4 | runtime Vec16/Block128/cache2 |
| 9066 | 96/4 | 4 | Vec16/Block64 |
| 9067 | 128/4 | 4 | Vec4/Block128 |
| 9068 | 112/4 | 4 | Vec4/Block128 |
| 9069 | 48/4 | 4 | K7168 显式特化为 **S6/C2**，Vec4/Block128 |

定义 `ceil_div(x,y)=ceil(x/y)`。当前规则顺序如下，阈值 256 属于当前代码设计，不能当作所有设备的通用常数：

| 公开 ID | 条件 | 实际配置 |
| --- | --- | --- |
| 9041 | `K>=8192`，或 `K>=4096 && ceil_div(M,16)*(N/16)<256` | 9050 |
| 9042 | `K==7168 && ceil_div(M,16)*ceil_div(N,48)<=256` | 9071 |
| 9042 | 其余 K7168 | 9073；其余 K 保留 9042 |
| 9044 | `K>=1536 && ceil_div(M,64)*(N/64)<=256` | 9056 |
| 9052 / 9053 | K7168 | 9070 / 9072 |
| 9062 | `ceil_div(M,96)<ceil_div(M,80)` | 9064；否则 M80 |
| 9063 | `K>=8192 && M<=48` | 9069 |
| 9063 | `K>=8192 && 80<M<=96` | 9066 |
| 9063 | `K>=8192 && 96<M<=112` | 9068 |
| 9063 | `K>=8192 && 112<M<=128` | 9067 |
| 9063 | 不在 `K>=8192 && M<=128` 范围，且 `ceil_div(M,80)*(N/128)*4>256` | 9065 |
| 9063 | 其余 | M80 split4 |

例如 `192×768×7168` 的 9042 实际为 9071；`384×768×7168` 为 9073；K7168 下记录 9053 时实际为 9072。用旧名字比较 ISA 会误判优化是否已经存在。

Small register 与 register-scale 常规契约为 M≤512；9043–9046、9055、fine 系列为 M≤2048。N、K 均须满足当前原生 scale 的 128 对齐，K≤16384；支持任意正 M 尾部。其余 family 的 M 对齐、输出字节范围等必须按 registry 分别检查。

## 3. 用第三章模型量化问题

### 3.1 `N = λT`：请求并发是否足够

为避免与 GEMM 的列数 N 混淆，这里把 Little 定律的在途对象数记为 `N_req`：

```text
N_req = λ_req × T_req
BW = λ_req × R_avg
需要的在途请求数 ≈ 目标带宽 × 平均请求延迟 / 平均请求大小
```

`N_req`、`λ_req`、`T_req` 必须统计同一种请求、同一个接口、同一个实例范围及时间窗。wave 驻留数、VMEM 指令数和 GL2 外部事务数不可以互相替代。

这里的`T_req`是请求在该接口内的平均停留时间，区别于wave寿命`T_s`。这些等式用于稳态平均值；短kernel的填充/drain或持续变化的队列需要单独处理。本轮没有匹配接口的队列积分与平均请求延迟，因此尚不能定量反推实际`N_req`；可以检验预取/并行改动是否减少完成时间，再用可用counter解释供给变化。

增加 wave、prefetch stages 或 global split，可能提供更多独立请求；如果它们只带来更高 LDS/VGPR 用量或队列争用，就不一定增加有效在途请求。L1/L2 命中也会使 CU 请求与外部事务的数量关系变化。

### 3.2 CPC/SPI 与 wave slot 补充

论文 3.1 的 chunk 模型为：

```text
C_p = C × #XCD
C_s = (C / #SPI) × WG × 4
```

其中 `WG` 表示一个工作组的 **wave 数**；`C` 是 chunk 内的工作组数，示例取 8 个工作组。原文使用 #SPI=4、SPI 派发 4 cycles/wave，相应推导得到 `C_s/C_p = WG/#XCD`。该比值检查 chunk 的消耗与再次供给，不能直接作为当前设备利用率。

原文 wave slot 模型进一步给出：

```text
T_p = #TotalCUs × W_o / WG
ρ_wave = T_s / T_p
T_s < T_p → 按该轮转模型，wave 可能先结束，等待下一次补充
```

`T_p` 沿用原文派发周期假设，表示同一 slot 获得下一 wave 的整圈轮转周期，不是 wave 结束后的纯空闲间隙 `T_d`；`ρ_wave` 是帮助阅读的模型比值，不是实测 occupancy。静态允许的 `W_o`、实际平均活跃 wave 数和在途访存数表达不同信息。当前代码用 grid 阈值选择深 prefetch 或不同 tile，正是在处理供给与 tile 周转问题；是否有效，需要当前设备的动态并发与运行时间验证。

不能通过主动拖长 wave 来“改善比值”。应该改善有用工作供给、减少空闲和等待，并检查完整任务时间。

### 3.3 CU 侧供给与等待

论文 3.2 的简化带宽模型为：

```text
BW_CU ≈ W_o × (每 wave 请求字节数) / (T_d + T_s)
```

原文在静态目标驻留量和实际平均活跃wave之间复用了`W_o`，最新笔记已提示空闲可能被重复扣除。若要代入数据，在同质wave的稳态周转近似下统一成以下口径：

```text
D_wave = 每 wave 在所分析接口发出的请求字节总量
W_slot = 受资源、grid和分配约束的参与周转slot数量/CU
W_active_avg ≈ W_slot×T_s/(T_s+T_d)
BW_CU ≈ W_slot×D_wave/(T_s+T_d) ≈ W_active_avg×D_wave/T_s
```

如果已经从完整窗口测得`W_active_avg`，使用第二种写法，派发空闲已经反映在平均量中。`W_slot`也不能直接填硬件最大wave数。有限grid、wave工作量/寿命不同、请求在wave内分布不均时，两式只是供给预算，实际带宽仍按该接口真实bytes/墙钟时间确认。

论文还以 CU↔L1 的 64 B/cycle 截断。这个硬件常数属于论文使用的模型，当前 gfx950 应先核对对应接口；不能把论文的 MI350 平均 57.6 B/cycle/CU 当作当前 MI355X 的实测上限。

优化动作对应到模型：

| 动作 | 想改变的量 | 必须同时检查的代价 |
| --- | --- | --- |
| 减少 VGPR/LDS | 静态可驻留量、实际 `W_active_avg` | grid 是否足够、寄存器分配粒度、spill、其他资源上限 |
| 增加 prefetch | 有效在途请求、覆盖返回延迟 | VGPR/LDS 增长、缓存压力、短 K 填充/drain |
| 增加 K waves / split | 并行工作、减少单 wave 长依赖 | 局部归约或全局 workspace、额外 grid 与 reducer |
| 更宽矩阵 load | 每条 VMEM 指令有效字节数 | 地址模式、边界 mask、TA/TCP 回压 |
| 复用 scale | 减少无用请求与地址处理 | 新增 live range、寄存器/标量指令变化 |
| 放宽安全 wait | 减少过早同步导致的 `T_s` 等待 | ISA 顺序、ring 安全、分支/SGPR 开销 |

如果 `W_o` 已按完整时间窗测得，它已含有派发空闲影响，不能再机械叠加同一 `T_d` 惩罚。实际吞吐最终仍用字节数/墙钟时间计算。

## 4. Logical BW、请求量与物理流量

### 4.1 当前 tuner 的指标

[base_tuner.calculate](/root/workspace/aiter-opus-mxfp8-bpreshuffle/aiter/utility/base_tuner.py:1603)对 FP8 输入和 BF16 输出计算：

```text
F = 2 × M × N × K
B_logical = M×K + N×K + 2×M×N
AI_logical = F / B_logical
BW_logical = B_logical / t
```

它把 A/B 各读一次、C 写一次作为工作量指标，**不包含 scale、重复 tile 请求、split workspace、cache line 粒度、写分配或物理 HBM 事务**。

历史 raw `profile.csv` 的 `BW` 由这个函数得到，单位为 **GB/s**；部分汇总文档另外转成 TB/s。比较前逐个确认文件单位。Logical BW 较高或较低均不能单独证明 HBM 是否已满。

### 4.2 Scale 与 split-K 的最低数据量

紧凑原生 scale 的逻辑大小为：

```text
B_scale = M×(K/128) + (N/128)×(K/128)             [Bytes]
```

一个 scale byte 被多条 wave 指令重复读取时，TA/VMEM 负担可以增加，而 HBM bytes 可能因缓存命中基本不变。这是 `register_reuse` 的主要判断点。

global split 为 `s` 时，FP32 workspace 的完整写入与 reducer 读取至少为：

```text
B_workspace = 4sMN 写入 + 4sMN 读取 = 8sMN        [Bytes]
B_final_C = 2MN                                  [Bytes]
t_total = t_producer + t_gap + t_reducer          [同一 stream 的设备完成时间]
```

`t_gap` 是两次 launch 之间的设备调度间隙；只相加 profiler 的两段 kernel 活动时间会漏掉它。本文 graph/Event 确认覆盖设备上的完整调用，不包含 caller 的输入准备或 Python/host API 开销。

K 分区互不重叠时，不能简单把所有 A/B 输入量乘 s；输出 tile 间重复加载、scale 复用损失和实际 cache 命中仍须考虑。

对于 `M144/160、N7168、K16384、split2`：

| M | workspace 写+读 | 相对 `B_logical` 的额外字节比例 |
| --- | --- | --- |
| 144 | 15.75 MiB | 13.55% |
| 160 | 17.50 MiB | 15.00% |

workspace 并不一定全到 HBM，可能在 L2 内被消费；此表表达必须执行的逻辑请求字节，不表达实测 HBM 增量。

### 4.3 每层分别建立 AI 与流量

忽略尾部和工作组内部重复加载时，tile 级输入请求可用下面的估算理解复用：

```text
A 请求 ≈ ceil(N/BN) × M × K
B 请求 ≈ ceil(M/BM) × N × K
```

它说明缩小 BN 常使 A 请求增加，缩小 BM 常使 B 请求增加。具体模板还受 wave 映射、padding/mask、K-wave 和 scale load 组织影响，精确值应从地址与动态指令计数恢复。

分别计算：

```text
AI_HBM = F / B_HBM
AI_GL2_SoC = F / B_GL2_SoC
AI_TCP_TCC = F / B_TCP_TCC
```

GL2↔SoC 包含 HBM、system memory、XGMI；确认目标之后才能估计 HBM。CU 请求、TCP↔TCC 流量、GL2 外部流量是不同接口，不能用同一个峰值做 Roofline。

M/N 尾部也影响有效计算比例。辅助量为 `M/[BM×ceil(M/BM)]`、`N/[BN×ceil(N/BN)]`；例如 N48 tile 的最后 N16 mask 与 M 尾部需要纳入比较，而不是只看完整 tile 的 MFMA 数。

Roofline还必须保持有用与执行FLOPs的口径一致：

```text
F_useful = 2MNK
F_exec = 实际执行的矩阵 FLOPs（从分支/动态 MFMA 或 F8 counter 核对）
η_work = F_useful/F_exec
t >= max(F_exec/P_compute_ceiling,
         max_interface(B_physical_interface/BW_interface_ceiling))
P_useful <= min(η_work×P_compute_ceiling,
                min_interface(AI_useful_interface×BW_interface_ceiling))
```

上面的每层AI采用`F_useful`时，对计算上限也乘`η_work`；若画执行AI/执行吞吐，则两者统一用`F_exec`。当前9051/M1的BM16 padding使这一区别达到16倍，不能拿执行F8吞吐直接报告有用GEMM吞吐。计算上限用相同精度、dense/sparse语义、指令形状和设备；接口上限用可靠规格或独立同条件校准，不用待诊断kernel自己的bytes/time回填。独立读/写端口分别比较，共享出口才按对应总容量相加。地址、同步、转换、归约等额外成本仍会使实际时间大于这个理想下界。

## 5. 按第四章顺序定位瓶颈

### 5.0 四类perfmon对应四类待检验限制

按4.1的分类，**Clocking**先确定周期/时间分母，**Workload Characterization**确认发出的指令、请求、字节和执行量，**IP Architecture**观察模块带宽、命中和接口回压，**IP Microarchitecture**定位set/队列/翻译/issue/资源停顿。后者提供原因线索，前面三类保证这些线索与实际工作和完整时间对应。

| 限制候选 | 最少要同时得到的证据 | 先检验什么 |
| --- | --- | --- |
| Compute bound | 数据已经就绪，相关矩阵/VALU/标量管线或指令发射受限，执行量与可达到速率匹配 | 降低无用执行、缩短计算依赖、调整发射；转Compute文档 |
| Bandwidth bound | 明确接口的物理bytes/time接近同条件可达到带宽，且有对应服务压力/回压 | 减少该接口请求、增加复用、改善channel/地址分布 |
| Latency bound | 有用工作等待返回或依赖，服务带宽未满，有效独立请求/工作不足 | 安全预取、更多独立工作、减少往返与依赖 |
| Dispatch bound | 前端补给或资源分配使可运行wave不足；grid/资源与SPI/前端证据支持 | 改工作粒度、grid或资源，核算额外请求和归约 |

低带宽与低occupancy单独不能完成分类。Figure 9用于沿接口追查原因；确认某层没有明显stall后还要检查请求是否被持续提供、计数器是否覆盖该路径，不能只由一个低stall数宣布该层没有影响。结论针对具体shape/分支，优化后可能转移到另一层。

### 5.1 第一步：固定测量条件，建立 Roofline 基线

记录 GPU 型号、gfx、可用 CU、频率/功耗、HIP/ROCm/编译器、源码 SHA、binary SHA、实际 dispatch、shape、输入/scale 格式、缓存/输入轮换方式。无其他 GPU 工作时串行执行，A/B 交替顺序，保留每轮结果并比较中位数及波动。

首先看 `2MNK/t`、对应精度的矩阵吞吐、分层真实字节/t、外部平均延迟。低外部带宽加高延迟，可能是请求供给/延迟限制；接近相同条件下可达到带宽、并有下游回压，才更支持 bandwidth bound。

counter采集、trace筛选和最终event确认分开。当前仓库`run_perftest`依赖torch.profiler trace，不是无profiler计时；其值可用于筛选，拟保留修改还需要共享地址池、graph/HIP Event的完整调用确认，详见9.2。

### 5.2 第二步：GL2、channel 与 SoC

| 要回答的问题 | 原文指标/事件示例 | 需要的解释 |
| --- | --- | --- |
| 实际外部 bytes 和请求大小 | `TCC_EA_RDREQ_BW`、`TCC_EA_WRREQ_BW` 及各尺寸请求数 | 原文 `BW` 派生量为累计 Bytes，须除时间 |
| GL2 是否承受明显压力 | `TCC_TAG_STALL/TCC_BUSY` | 拆内部 set/LFIFO 与外部 queue/credit 等停顿 |
| GL2 命中情况 | `TCC_HIT/(TCC_HIT+TCC_MISS)` | 高 hit 不保证内部接口没满；低 hit 不保证 HBM 已满 |
| GL2 是否反压 TCP | `TCP_TCR_TCP_STALL_CYCLES/TCP_BUSY` | 结合 GL2 与外部等待确认来源 |
| channel 是否均衡 | 各 TCC instance 的 bytes、busy、stall | `_sum` 可以隐藏热点；必须保留 instance 分布 |
| 外部平均请求延迟 | 在途 level 的逐周期积分/请求数 | 单次 level 采样不是积分量 |

论文的读请求恢复为 32/64/128 B，写请求为 32/64 B：

```text
B_read = 32r32 + 64r64 + 128r128
B_write = 32w32 + 64w64
```

原文表中 `r128=TCC_BUBBLE` 等事件含义不能直接移植。**在 gfx950 上先用实际工具 definitions 核对名称、单位、事件意义和实例范围，再构建公式。** 本机已经完成 CPU 定义核对，四组主 PMC 也已有干净 GPU runtime 接受证据，具体映射、目标覆盖及结果见 5.7/5.8；两个可选组仍未由此证明可运行。counter 不可用时写明缺失，不用旧架构 FIFO 深度或估算结果代替本轮测量。

### 5.3 第三步：TA、TCP、UTCL1

当前矩阵数据路径已有大量 `buffer_load_dwordx4`，LDS 模板使用 16B direct-LDS 请求。因此宽化矩阵 load 并非可以直接从 `dword` 再升到 `x4` 的默认机会；byte scale、地址计算、mask 和队列压力仍应分别分析。

| 层级 | 原文指标/事件示例 | 下一步 |
| --- | --- | --- |
| SQ→TA 命令、地址、写数据 | `SQ_VMEM_TA_CMD/ADDR/DATA_FIFO_FULL / SQ_BUSY` | 区分 VMEM 发射争用与缓存接收回压 |
| TA 地址效率 | `TA_BUF_ADDR_COALESCE_RATE` | 联合 load 宽度与有效 bytes 判断 |
| TCP→TA 回压 | `TCP_TCP_TA_ADDR_STALL_CYCLES/TCP_BUSY` | 继续拆 TCP 内部/UTCL1/GL2 |
| TCP lookup/分配 | set-full、alloc stall；各 tagram miss 请求 | 检查映射/stride；不同 miss rate 也会造成分布偏斜 |
| TCP LFIFO/RFIFO | 对应 stall cycles/TCP_BUSY | 联合 GL2 回压与返回延迟 |
| UTCL1 | hit/miss、`INFLIGHT_MAX/FULL`、`LFIFO_NOT_RES` 等 | 区分翻译吞吐、在途容量、下一级翻译/page walk 等待 |
| TCP↔TCC bytes | read/write bytes 汇总 | 按时间/相关 cycles/CU 范围转换，与外部 bytes 比较 |

TA 原文指标使用 0–1 比例 `q` 时：

```text
q = 4 × TA_BUFFER_WAVEFRONTS / TA_BUFFER_TOTAL_CYCLES
平均地址传递 cycles/wave = 4/q
q=1 → 4 cycles；q=0.25 → 16 cycles              [原文模型]
```

若工具输出百分数，须先除 100。25% 不是“只有四分之一线程合并”。原文模型下 dwordx4 即使不能做同种 TA 地址合并，也能靠每指令更多 bytes 达到更高字节吞吐。不能以合并率一个数字判定修改方向。

TCP set 请求偏斜需要与每个 set 的命中/缺失和停顿共同确认；tagram 请求事件可能只统计 miss，低流量或分母为零时也不能计算可靠比值。UTCL1 hit rate 很高仍可能受翻译吞吐限制。

### 5.4 第四步：wave 寿命、依赖和 issue

| 观测 | 能直接说明的事实 | 确认限制还需要什么 |
| --- | --- | --- |
| 平均 wave life 较长 | wave 周转变慢或单 wave 工作更多 | 比较有效工作、依赖等待、issue、barrier 与 dispatch |
| VMEM issue wait 高 | wave 等 VMEM 发射 | TA/TCP 回压、真实 bytes、请求延迟 |
| 依赖 wait 高 | 数据/同步等依赖尚未满足 | 区分 vmcnt、lgkmcnt、barrier、矩阵链依赖 |
| LDS issue wait 高 | wave 等 LDS 管线 | LDS 请求量、冲突、有效吞吐与同步 |
| VALU issue wait 高 | wave 等 VALU | VALU 利用；GEMM 另外看 MFMA/矩阵管线 |

论文 `SQ_WAVE_LIFE=4×SQ_WAVE_CYCLES` 是累计 wave cycles。平均寿命还要除同范围 wave 数；论文 `SQ_WAIT_INST_VMEM/SQ_WAVE_CYCLES` 的分母是累计 wave 时间，区别于接口 FIFO 停顿的 `SQ_BUSY`。**本机 gfx950 SDK 没有 `SQ_WAVE_LIFE`、`SQ_WAIT_INST_VMEM` 这两个可采集定义，不能把上述论文名称直接填进 rocprofv3；当前映射见 5.7。**

多个 wave 的等待可同时发生，累计 wave 等待不是可直接从墙钟时间减去的数。更大的 tile 可以同时提高 wave life 和完成吞吐；不以 wave life 单项最小作为优化目标。

### 5.5 第五步：资源、grid 与真实 occupancy

静态驻留上限取硬件 wave slots、V/AGPR 分配、LDS、SGPR/barrier、工作组约束的最小值。要使用当前 GPU 的资源容量和分配粒度；不能套论文的 64 KiB LDS、32 waves/CU 或将 AGPR/VGPR 一律相加。当前 gfx950 metadata 的 `.vgpr_count` 已包含 AGPR 对应的总向量分配，不能再加一次 `.agpr_count`；同时核对 descriptor、AGPR 起点和实际驻留查询。

Small LDS 的实际动态 LDS 由 launcher 根据当前 K 计算：

```text
loops = ceil((K/128)/split)
stages_used = min(loops, S)
LDS = stages_used×(A_STAGE+B_STAGE)
      + (register_scales ? 0 : (BM+B_GROUPS)×loops)
```

metadata 的 fixed LDS=0 不意味着不占 LDS。必须加 launcher 的动态分配；register 模板的 K-wave FP32 归约则可能占 fixed LDS。

实际 grid 为 `ceil(M/BM)×ceil(N/BN)×split`，还需考虑 wave 数和每工作组可驻留资源。grid 刚够一轮、少于一轮、跨过一轮及尾部不均衡分别测量。

若工具提供相同范围的 wave 积分 `I_wave`：

```text
平均 waves/CU = I_wave/(对应 cycles×对应 CU 数)
平均 wave life = I_wave/对应 wave 总数
```

静态允许更多 wave，并不保证 grid、dispatch 或独立访存持续填满；已经驻留但等待依赖的 wave 也不等于持续提供请求。

当前正式 JIT 的 narrow runtime 有明确的 SGPR spill：9023 为 VGPR232/SGPR106/SGPR spill46，9024 为 VGPR152/SGPR106/SGPR spill44。它们 `.private_segment_fixed_size=0`，ISA 使用 `v_writelane_b32/v_readlane_b32` 把 scalar 值保存到 VGPR lane，未见 scratch memory load/store；不能把该现象解释为新增 HBM scratch 流量。它会消耗向量寄存器、引入 lane 操作和依赖，可能影响 occupancy/issue/供数，仍要测量才能判定性能影响。相应 fixed-K7168 分支无此 spill。证据见 `official_compute_metadata.json`、`official_compute_metadata/kid9023/9024_metadata.json` 与反汇编，详细分析见 Compute 文档的 narrow 章节。

### 5.6 Counter 使用的统一规则

1. 先列出当前 gfx950 工具支持的事件与派生表达式，保留工具版本与定义。
2. 确认单位：Bytes、requests、cycles、quad-cycles、百分数或已平均值。
3. 确认实例：CU/SIMD/SE/XCD/TCC channel、`_sum` 还是单 instance。
4. 分子分母用同一窗口、同一范围；sum over CU 的 cycles 不能再重复除 CU 数。
5. `TB/s=Bytes/t_seconds/10^12`；B/cycle 使用对应模块 cycles，不能混用不同 clocks。
6. 缺失事件、replay 条件或不支持的派生量在结果里明确记录。原文的 10%/80% 是诊断示例，不是当前 kernel 的通用门槛。

### 5.7 本机 gfx950 已核对的 counter 映射

定义来源为 `/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml`；hash、原始定义、采集组预算、派生表达式与限制已保存在 `counter_evidence.json`。定义证据与实测状态分别记录：本轮已完成四个短shape的`sq_ea`，并完成8192³的9000和`1×65536×1536`的9051代表采集。当前四组主PMC均已有干净运行接受证据；具体目标覆盖和结果见5.8，两个可选组未由此得到接受证明。

| 信息 | 当前 gfx950 事件与公式 | 解释条件 |
| --- | --- | --- |
| DRAM read bytes | `32×ΣTCC_EA0_RDREQ_DRAM_32B` | 64B request计2、128B计4；不是“只有32B请求”的次数 |
| DRAM write bytes | `32×ΣTCC_EA0_WRREQ_WRITE_DRAM_32B` | 64B request计2，统计DRAM写入 |
| DRAM TB/s | `(read_bytes+write_bytes)/dispatch_ns/1000` | SDK timestamp为ns；profiler耗时用于诊断，收益仍用无profiler A/B |
| EA 总读 bytes | `32ΣRDREQ_32B+64ΣRDREQ_64B+128ΣRDREQ_128B` | 本机三个独立尺寸事件，不用论文的`TCC_BUBBLE`间接恢复；与DRAM子集分别分析 |
| 平均 wave life | `4×ΣSQ_WAVE_CYCLES/ΣSQ_WAVES` | SQ_WAVE_CYCLES是per-SE聚合quad-cycle，两者同窗同范围 |
| 等待、issue、active 比例估计 | `ΣSQ_WAIT_ANY/ΣSQ_WAVE_CYCLES`、`ΣSQ_WAIT_INST_ANY/ΣSQ_WAVE_CYCLES`、`ΣSQ_ACTIVE_INST_ANY/ΣSQ_WAVE_CYCLES` | 均为quad单位；当前选中样本内部三项合计100%，但不能从墙钟直接相减，也不能替代已缺失的VMEM细分等待 |
| MFMA busy | `100ΣSQ_VALU_MFMA_BUSY_CYCLES/(max(GRBM_GUI_ACTIVE)×SIMD_NUM)` | MFMA busy为per-SIMD cycle；本轮短shape的GRBM窗口未核准，公式值不用于绝对利用率/饱和分类 |
| 平均 waves/CU估计 | `4ΣSQ_WAVE_CYCLES/(max(GRBM_GUI_ACTIVE)×CU_NUM)` | CU_NUM用匹配实例的simd_count/simd_per_cu；工具OccupancyPercent固定32分母只是该派生模型 |
| TA paper地址率 | `4ΣTA_BUFFER_WAVEFRONTS/ΣTA_BUFFER_TOTAL_CYCLES` | 混合/direct-LDS流量语义须核对；另采`TA_BUFFER_READ_LDS_WAVEFRONTS`辨别direct-LDS |
| TCP回压比例估计 | `Σstall_cycles/ΣTCP_GATE_EN1` | 本机没有TCP_BUSY；GATE_EN1是unwindowed接口开钟周期，该比值与论文busy归一化不同 |
| Tagram请求份额 | `ΣTCP_TAGRAMi_REQ/Σ四个TAGRAM_REQ` | 定义为该TCP发往全部TCC的L2请求mapping；不是所有L1 hit lookup/set分布 |
| UTCL1线索 | `TCP_UTCL1_SERIALIZATION_STALL`、`THRASHING_STALL`、`STALL_INFLIGHT_MAX`及translation hit/miss | thrashing因probe overlap/MECO仅rough估计；不以单个counter确认TLB根因 |
| 外部credit线索 | DRAM/GMI read/write credit stall | read credit stall可能在不需要read时也计数；不能直接等同有效读请求的等待比例 |

`TCC_TAG_STALL` 的probe可以在多处阻塞并重叠，`TCC_BUSY`、TCP gate和部分TA/SQ事件不支持严格windowing。这些比例为诊断估计，GPU隔离和定义限制必须记录。以相同counter范围比较A/B，不能要求所有比值都严格在0–100%。

已准备 `pmc/sq_ea.txt`、`pmc/l2_tagmap.txt`、`pmc/utcl1_credits.txt`、`pmc/ta_lds.txt` 四组主采集及两个可选组；每组按本机SQ/TA/TCP/TCC等预算核对，但预算通过不保证runtime可调度。`parse_counters.py`保留各source/process/agent/queue/dispatch身份、原始instance rows，不跨PMC subprocess拼接伪样本。split-K分别分析producer/reducer，并核对顺序和队列后再配对完整调用的bytes/time。

### 5.8 两个真实代表：大tile复用与低M外部流量

本轮代表为正式9000的`8192×8192×8192`、正式9051的`1×65536×1536`。两者在MI355X/gfx950、PCI`0000:85:00.0`上调用审查起点正式module，加载SHA为`093bc9cd…`，自动轮换51组地址。每个PMC subprocess保留最后51个dispatch。已核对7组干净结果：9000的`sq_ea/utcl1_credits/ta_lds`及9051的全部四组；9000的首次`l2_tagmap`因外部工作中断，整份排除；可选重测已停止，本轮无该目标的干净结果。完整记录与可复算脚本见 [characterization_counter_summary.json](char_profiles/characterization_counter_summary.json) 和 [analyze_characterization_profiles.py](analyze_characterization_profiles.py)。

下表每个数取**自己的独立pass中位数**；横向排列用于诊断趋势，不表示这些counter来自同一次dispatch。所有时间和bytes/time均带profiler扰动，不能直接作为优化收益或峰值达成证明。

| 观测 / 来源pass | 9000，8192³ | 9051，1×65536×1536 | 可支持的解释 |
| --- | ---: | ---: | --- |
| 时间 / sq_ea | 375.603 μs | 16.921 μs | 两个当前正式实现的profile观测 |
| 有用吞吐 / sq_ea | 2.927 PFLOP/s | 11.898 TFLOP/s | 按`2MNK`；9051执行含M16 padding，不能把190.369 TFLOP/s执行量当有用吞吐 |
| DRAM读 / 写 / sq_ea | 807.594 / 134.382 MB | 100.756 / 0.131 MB | 物理DRAM事务；9000不同输出tile重复请求、cache-line等使输入物理bytes大于理想A/B一次量 |
| DRAM总带宽 / sq_ea | 2.508 TB/s | 5.962 TB/s | 9051外部流量更重要；仍需回压/延迟证据确认具体限制 |
| SQ wait / issue / active / sq_ea | 26.03% / 45.15% / 28.81% | 30.09% / 61.26% / 8.65% | 只提供ANY类别的wave时间线索，不能当VMEM、VALU、LDS根因细分 |
| TCP LFIFO / gate / sq_ea | 0% | 24.03% | 9051出现内存延迟FIFO容量压力线索；分母是未严格windowing接口开钟周期 |
| TCP→TA / gate / sq_ea | 1.70% | 5.86% | 接收回压线索；不能单项指定TA/TLB根因 |
| UTCL1 in-flight-max / gate / utcl1_credits | 1.20% | 6.18% | 9051翻译在途容量压力更值得查；serialization均为0，不能由此证明翻译没有成本 |
| direct-LDS buffer-read指令 / ta_lds | 4,194,304 | 0 | 确认9000矩阵走direct-LDS，9051矩阵直接到寄存器；单位为wave请求事件，不是bytes |
| LDS bank-conflict原始计数 / ta_lds | 18,747,392 | 0 | 与路径相符的线索；不是18,747,392个墙钟周期，也没有单独证明LDS是总瓶颈 |
| GL2 hit / tag-stall / l2_tagmap | 本轮无干净结果（可选重测停止） | 3.97% / 3.19% | 9051GL2大多miss；tag-stall/Busy是有probe/window限制的诊断比率 |
| TA原文地址比值 / l2_tagmap | 本轮无干净结果（可选重测停止） | 0.2995，约13.36 cycles/buffer wave | 联合矩阵x4和byte scale解释；不同路径不能直接以一个TA比值排优化优先级 |

9051每dispatch的wave数8192、动态scaled MFMA49,152，F8counter换算为3,221,225,472 FLOPs，恰好是逻辑`2MNK=201,326,592`的16倍：BM16在M1时执行大量无效行计算。9000的4096 waves、16,777,216条动态MFMA与逻辑1,099,511,627,776 FLOPs完全匹配。这个核对先确定执行量，再讨论计算或访存效率。

本轮还有两个需要保留的口径限制。第一，9000长K的`GRBM_COUNT/duration`中位数约2.402GHz，接近agent记录的2.4GHz，但9051各pass约3.57–3.64GHz，短shape也出现同样偏高现象。没有raw GRBM实例/窗口校准证据，不据此乘除固定因子或比较绝对occupancy/MFMA利用率。第二，9051的`SQ_VMEM_TA_ADDR_FIFO_FULL/SQ_BUSY_CYCLES`为443.64%；事件未windowing，scope不等价，该数**不是时间百分比**。保留raw计数，不把它解释成443% stall或最主要根因。

下一步按观测选择问题：9000查矩阵/scale/LDS的issue与地址队列节奏；9051查LFIFO、翻译在途容量、TA处理和缓存miss路径能否持续供给。现有GL2 scalar hit和四tagram聚合份额没有channel/set实例分布，不能确认热点；也没有平均DRAM请求延迟或队列积分，不能完成`N_req=λT`的实测反推。代表结果支持按路径分别优化，不把全部9000系列统一判为compute bound或HBM bandwidth bound。

### 5.9 Register A/B：验证少发指令，再区分请求数与物理字节

限定 `register_reuse_scoped` 的9042真实赢家 `32×7168×3072` 已完成两label各自的 `sq_ea`、`ta_lds`，四个pass均干净；与9021 prologue代表合计8个pass、408个选定dispatch。该批在 MI355X/gfx950、PCI `0000:15:00.0`、HIP visible3串行运行，每对A/B在同卡窗口，使用相同seed和各自51组自动轮换地址。第8.6节早前共享池Event来自PCI `0000:85:00.0`：本节提供另一张卡上的请求/等待线索，**不把profiler耗时变化作为优化收益，也不替代共享地址Event或支持域回归**。逐pass来源、clean claim、计划/runner/库SHA和复算见 [A/B counter汇总](ab_profiles/ab_mechanism_counter_summary.json)、[CPU复核脚本](analyze_ab_mechanism_profiles.py) 与 [执行队列](ab_mechanism_profile_queue.json)。

两个 `sq_ea` pass 的全部102个选定dispatch都为224 WG、896 waves、21,504条动态scaled MFMA，F8执行量1,409,286,144精确等于逻辑 `2MNK`，A/B完全相同。因此以下buffer请求下降没有伴随计算量下降。

| 观测 / 独立来源 | baseline → candidate 中位数 | 可支持的解释 |
| --- | ---: | --- |
| TA全buffer read事件 / ta_lds | 64,512 → 59,136，减少5,376（8.33%） | 每个选定dispatch都精确相同；动态请求事件确有减少 |
| direct-LDS / LDS bank-conflict / ta_lds | 两侧均0 / 0 | 矩阵到寄存器路径不变；不借direct-LDS/LDS冲突解释收益 |
| 平均wave life / sq_ea | 15,782.7 → 14,778.0 cycles，−6.37% | 匹配wave范围的估计下降；不是6.37%墙钟收益 |
| 累计issue wait / sq_ea | 1,556,867 → 1,356,490 quad-cycles，−12.87% | 发射等待计数下降，与更少请求/指令相容，尚未定位单个管线根因 |
| 累计ANY wait / sq_ea | 1,490,667 → 1,470,266 quad-cycles，−1.37% | raw略降；其wave时间占比反而42.34% → 44.85%，不以占比升高认定更慢 |
| DRAM读 / 写 / sq_ea | 22.8675 → 22.8654 MB；写均0.4588 MB | 读仅减少2,048B（0.009%）；减少请求事件没有变成同幅度HBM字节节省 |
| SQ地址FIFO full原始计数 / ta_lds | 1,145,432 → 1,028,860 | 原始计数下降线索；事件未windowing，不能换算节省周期/时间 |
| TCP read-tagconflict原始stall / ta_lds | 两侧均485,632 | raw未变；gate比率9.88% → 10.22%受分母变化影响，不能据此确认热点或回退 |
| TCP→TA / gate / sq_ea | 27.14% → 28.39% | 接口诊断比率没有统一下降；不把TA或L1定义为已证明根因 |

请求事件变化可与源码精确对照：9042为BM32/BN32、4个K-wave；每wave分到6个K128 tile，每tile有2个A片段和2个B片段。原路径每tile为8条矩阵vector load、2条A-scale byte load和2条重复B-scale byte load，合计12条buffer read；限定复用只把同一N128 scale组内第二条B-scale请求删掉，变为11条。因此 `896 waves × 6 tiles × (12−11) = 5,376`，两侧分别为64,512与59,136，和全部102个 `ta_lds` 选定dispatch精确一致。这确认动态请求数符合源码机制；该TA事件仍不能单独识别每条请求的byte宽度，具体被删请求类别由源代码和ISA提供。

当前证据支持“在相同计算量下少发B-scale请求，issue/wave计数下降”的有限机制判断。DRAM字节几乎不变，不能把已有Event改善写成HBM流量降低8.33%或HBM带宽瓶颈已解除。仍没有采集该A/B的L2/UTCL1组合、在途队列积分或平均请求延迟，不能用`N_req=λT`定量归因到某一级。9021 prologue的请求数量不变而wave计数下降，详见Compute文档5.5；两种改动各自解释。

本批8个pass的GRBM隐含周期中位数4.57–5.06GHz，仍超过agent的2.4GHz；不使用绝对MFMA利用率/occupancy，不做固定因子修正。SQ地址FIFO/SQ_BUSY的scope也没有等价证据，复核摘要已将其百分比排除于可解释metric之外，保留raw。各label/group只有一个独立pass，51个dispatch描述这一次采集；不能据其推断跨卡、全部646个runtime支持域或正式TU的收益。现有结果不改变第8.6节的有限采用条件。

## 6. 本轮优化实验与取舍

### 6.1 `register_reuse`：对齐 tile 内复用 B scale

**最初全局实验：**隔离 traits 的默认 `ReuseBScale=false` 改为 `128 % BlockN == 0`，其余 pipeline 和编译参数相同。该全局版本最终拒绝；当前采用的是三个alias显式true，见本节末和8.6。

证明条件：`col=block_x×BN`，各 N fragment 的起点 `nr` 位于 `[0,BN-16]`。BN 整除 128 且 tile 按 BN 对齐时：

```text
(col+nr)/128 = col/128
```

因此同一 K128 下 B scale 可复用。BN48 不能使用这个推导，例如 col96、nr32 会跨入下一个 128 分组，保留不复用。

私有 runtime variant 覆盖 9040、9041、9042、9050、9051、9052、9053、9054，直接实例化 runtime traits，**不经过生产父候选 dispatch**。9041 的条件分支9050也作为独立入口验证。K7168下真实公开9052/9053→9070/9072早已开启复用；固定分支身份与正式API control单独核验，私有runtime的收益不能归给这些特化。

| runtime ID | ISA byte load 数 baseline→candidate | VGPR | SGPR | fixed LDS |
| --- | --- | --- | --- | --- |
| 9040 | 51→34 | 190→182 | 59→60 | 0 |
| 9041 | 10→10 | 49→49 | 52→52 | 7168 B |
| 9042 | 32→24 | 145→142 | 58→58 | 12288 B |
| 9050 | 16→16 | 66→66 | 54→54 | 7168 B |
| 9051 | 24→16 | 104→101 | 55→56 | 6144 B |
| 9052 | 15→10 | 75→72 | 55→54 | 14336 B |
| 9053 | 20→15 | 109→105 | 58→55 | 28672 B |
| 9054 | 30→15 | 158→157 | 56→55 | 24576 B |

本实验各配置 AGPR、scratch、VGPR/SGPR spill 均为 0；矩阵 `buffer_load_dwordx4`、MFMA、输出 store、LDS reduction/bpermute 的静态数量不变。表中 byte load 是包含 prologue/loop/drain 的**静态 ISA 数量**，不能直接作为总动态读次数或 HBM bytes。BN16 的9041/9050每wave只有一个N fragment，开启复用后指令bytes/hash/resources全部相同，属于此次默认开关影响范围的回归对照。

表中为补齐8入口后的当前bundle。加入同一TU的两个template实例使9042/9051部分ISA调度与寄存器分配相比最初6入口发生变化，其余四个旧入口保持指令字节和资源相同；旧产物与比较保存在`previous_*_6_entries.json`和`device_audit.json`。现有六个源trait/case未改，但不能声称当前bundle所有旧entry机器码逐字节未变。A/B必须使用当前同一构建的两侧binary hash。

最初据此优先检验短/中K的9040/9051/9054及runtime9042/9052/9053，目标是确认scale请求、地址处理、wave/issue开销是否减少。后续有限A/B counter确认9042少发scale请求、DRAM bytes几乎不变，见5.9；其它路径的保留与拒绝见8.4/8.6。

证据：`register_reuse/build_manifest.json`、`register_reuse/device_audit.json`、两侧 `isa.txt`/`metadata.txt`。每个 `.so` 已确认有 8 个真实 gfx950 device 入口与 HIP fatbin 注册；重建脚本`register_reuse/build.py`只运行CPU编译/审查且关闭GPU可见性。

最初全局默认参数实验的影响范围也经过单独审查：它改变上述8个未显式传`ReuseBScale`的register alias。codegen对register_tail显式传该布尔参数，9070/9072已经为true、9071/9073为false，均不受默认变化；LDS/fine和9000–9030主体不使用这项默认参数。正式parent bundle与真实分支已独立重编核对，避免把同TU其他实例引起的调度变化误认成相同device。最终采用不修改默认值，只显式开启三个runtime alias。

按`shape_inventory.json`的实际派发，当前745个历史赢家中58行会改变模板标志：17行N16保持指令/资源相同，41行有效减少重复scale加载；另外38行固定register_tail保持相同。已有GPU首窗口中，9040在`16×7168×768`上5/5轮退步，3.5639→4.5389μs，时间增加27.36%。**因此不能把上述一行默认值修改全局落地。** 后续只给9042/9053/9054显式true的scoped方案已完成全部6/6受影响当前赢家、5个新机制、6个异常Event与正式API核验，完整device身份相同；限定方案已验证并采用，源码身份见[集成记录](formal_selected/integration_manifest.json)。可选支持域扩展已停止，9051/9052本窗口无稳定收益。CPU影响范围、最小修改和回归契约见`landing_conditions.json`；逐轮结果见第8.4/8.6节。

### 6.2 `fine_wait`：按每个 wave 的实际请求数等待

Fine-M A copy 每组 8 行，`BM/8` 不整除 wave 数时，前几个 wave 每 K128 多发一条 direct-LDS load。当前统一 `VMEM_TILE` 使用 floor，使这些 wave 等待更保守。

```text
A_base = floor((BM/8)/waves)
extra = wave_id < ((BM/8) % waves)
B_loads = BN×128/(BLOCK_SIZE×16)
V_wave = A_base + extra + B_loads
```

| 配置 | 每 wave 的矩阵请求 | steady wait 的低/高阈值 |
| --- | --- | --- |
| M80，4 waves，S4/C1 | 6/7 | 12/14 |
| M96，8 waves，S4/C1 | 3/4 | 6/8 |
| M112，4 waves，S4/C1 | 7/8 | 14/16 |
| M48，4 waves，S4/C1 | 5/6 | 10/12 |
| 9069 固定 K7168，S6/C2 | 5/6 | cluster wait 10/12 |

候选限定在 `FINE_M_LOADS && !REGISTER_SCALES && EARLY_SCALE_LOADS && remainder!=0`。按 wave 选择常量立即数，只替换三个 steady wait 位置；prologue 的 short K、空 split、未来 tile 不完整的尾部、read-only drain 仍保持 `vmcnt(0)`，LGKM wait、barrier、scale 顺序与 ring 保护保留。

这是减少保守等待的实验，当前源代码并未因此被认定为错误。安全性要从请求类别、VMEM 顺序、每条 direct-LDS 请求和 ring 覆盖关系证明，不能只把 wait 数字调大。

CPU复核的等待证明分四步：

1. 受影响路径`REGISTER_SCALES=false`，每轮prefetch只发矩阵direct-LDS，不在稳态追加scale请求。A每wave为`floor((BM/8)/waves)+extra`条，B均匀为`BN×128/(BLOCK_SIZE×16)`条；当前ISA确认`async_load<16>`落成单条16B direct-LDS。M尾部通过buffer边界处理，不改变该wave发出的指令数量；仅`group<BM/8`是整wave条件。
2. early SFA/SFB在初始矩阵prefetch之前。SFA可能按对齐/有效行退化成多个byte load，但这些都属于更早的VMEM-read前缀，不计入要留下的未来tile数。scale LDS store依赖返回值，`sched_barrier(0)`、`lgkmcnt(0)`与barrier保留。
3. 当前工具链源码`AMDGPUHWEvents.cpp:77`把LDS DMA和普通buffer read都分类为`VMEM_READ_ACCESS`；`SIInsertWaitcnts.cpp:2283`使用同类指令按顺序返回的规则，`:1036`明确direct-LDS结果使用前需vmcnt。矩阵读取与scale同类，S4/C1等待留下2个未来tile的本wave请求；S6/C2留下2个未来tile。各wave只排除自己的未来请求；barrier仍等待所有wave发布当前tile。该结论限于当前指令类别与编译产物，不能推广为任意混合VMEM都FIFO。
4. ring的新prefetch写的是距离`S-C`的未来槽，旧槽在覆盖前已经完成LDS读取，并由原`lgkmcnt(0)`和barrier保护。短K/空split、未来group不足、read-only drain仍使用0等待。按源代码S4/C1、S6/C2、S4/C2与每split0–128个K128 tile执行CPU枚举，检查当前tile完成、scale前缀完成、覆盖前旧tile已读，共2322个schedule条件通过；它是源级不变量核查，不代替所有GPU时序/边界回归。

静态结果：两侧各 15 producer + 5 reducer；受影响 producer SGPR +2；VGPR、AGPR、direct-LDS、barrier 数不变，无 scratch/spill。M96/4 waves、M128/4 waves 与所有 reducer 的指令字节哈希不变。选择分支与额外 SGPR 可能抵消等待收益。

Public 9060–9069 私有 launcher 复刻生产 dispatch 和 fixed-K；29060/29061/29062/29063/29069 是强制 runtime 的诊断 ID，不能写入 tuned CSV。按实际 producer + reducer 调用判断收益，并覆盖 short K、非整分区尾部、M tail、固定 K 及 runtime 分支。

证据：`fine_wait/static_validation.json`、`fine_wait/static_summary.json`、两侧 `device_audit.json`/`resource_isa.json`、`fine_wait/variants.json`。

GPU已完成九个干净Event目标：五项每轮都退步，包含真实当前赢家9062的M144/M160固定K16384 split2；其它四项接近0或正负混合，未建立可用收益。**不采用本轮helper加三个wait位置的全局修改，保留原保守wait。** 有限正确性全部通过，不说明当前保守代码有错误。未采counter，不能把分支或增加SGPR指定为退步根因；完整逐轮、producer+reducer与派发范围见第8.5节和`fine_timing_summary.json`。

### 6.3 `fine_n64`：用更小 BN 增加供给

目标为 `M144/160、N7168、K16384`。两侧都使用 M80、4 waves、S4/C1、split2、固定 K16384、同一保守 wait policy、Vec4/Block128/cache0；**唯一几何变化为 BN128→BN64**，不叠加 fine_wait。

| 静态项目 | BN128 baseline | BN64 candidate |
| --- | --- | --- |
| 目标 grid 工作组 | 224 | 448 |
| 动态 LDS/WG | 115008 B | 81216 B |
| VGPR / SGPR | 86/70 | 56/62 |
| vmcnt 立即数 | 0、12 | 0、8 |
| producer 指令字节 | 11052 | 9656 |
| producer 静态 MFMA | 80 | 40 |
| producer 静态 direct-LDS load | 49 | 35 |

两侧均无 AGPR、scratch/spill；reducer 指令字节完全相同，436 B、VGPR10/SGPR18。

若当前设备确认 LDS 容量 160 KiB，两个 BN64 工作组的 LDS 总量为 158.625 KiB，满足 LDS 容量条件；这只说明 LDS 不排除两组驻留，仍需核对寄存器粒度、wave/workgroup/barrier 上限和实测 occupancy。

风险：BN 减半使 N 方向 tile 数翻倍，A 与部分 scale 请求增加，TA/TCP/TCC 压力可能上升；producer 每工作组 MFMA 数减少不等于整个任务计算量减少。额外 wave/工作组与尾部成本也可能使它变慢。保留与否必须看完整 producer+reducer 时间和接口 bytes/等待变化。

证据：`fine_n64/build_manifest.json`、`fine_n64/device_audit.json`、两侧 `isa.txt`/`metadata.txt`、`fine_n64/variants.json`。两侧 `.so` 各有 producer + reducer 两个真实 device 入口。

前两次GPU运行均遇外部作业并中断。第一次完整raw未保留；现存第二轮`results/n64_interrupted_2.json`整份标污染，所有时间继续排除于最终性能判断。后续`results/n64_clean_target_0.json`与`_1.json`完成两个目标的干净确认：M144 44.5211→50.7714μs、M160 45.0945→51.1784μs，各5/5轮退步，**拒绝本轮BN64替换，保留BN128**。静态资源降低并未改善完成时间，未采counter，不能指定重复A流量为已证明的根因。详情见第8.3节。

### 6.4 Narrow runtime 展开控制：与 Compute 分析衔接

基于正式metadata发现的scalar spill，`narrow_unroll`隔离实验把9023/9024 runtime K-loop的展开因子改为1，固定K两分支完全保留。静态VGPR分别为232→121、152→87；scalar spill为46→21、44→22；ISA lane write/read分别为46/166→21/21、44/132→22/22，scratch memory仍为0。

这个实验可能改善寄存器驻留、scalar/lane issue和MFMA供数，也可能因减少展开降低指令并行性。六项干净Event已经完成，但未形成可靠整体收益，**9023/9024全局runtime展开修改不采用**，保留生产基线。没有counter/ATT，不能把时间变化指定为MFMA issue或访存根因；详情见Compute文档第9.2.1节、`results/narrow_runtime_window_analysis.json`和`narrow_unroll/isa_comparison.json`。固定K未改变。

CPU落地范围见 [compute_adoption_scope.json](compute_adoption_scope.json)：当前745历史赢家中，narrow仅改变9023的4个runtime赢家（M544/576、N7168、K7168/16384）；9024的9个赢家均fixed-K不变。该745-shape集合里，每个parent仍各有718个可调用runtime路径，不能从赢家数推定全部fallback都安全。[plans/narrow_current_winners.json](plans/narrow_current_winners.json)的六项已完成干净Event：四个9023实际赢家及两个9024 runtime对照，各侧signed8rep、每目标10×51共享池输出检查均通过。四个9023结果近零或正负混合；9024最大K对照62.5260→63.3566μs，5/5轮退步。本轮全局unroll1不采用，不扩大被拒绝版本的支持域回归。作为交叉排查，9021/9022 prologue的201个当前赢家随后全部完成完整Event。9021的42项median全正，geomean+3.155%、sumtime+2.313%，已采用；9022总体geomean+1.390%、sumtime+0.470%，但K16384组20项geomean−0.468%、9项5/5变慢，本轮不采用其全局改动。短K异常一次预先固定复测+1.613%，未复现稳定大回退；范围取舍依据完整长K集合。最终采用仅9021及三个register alias，见[集成记录](formal_selected/integration_manifest.json)。735/691个shape仍是可调用支持域数量，额外数值拓展保留partial，不能当作全域已测。详见Compute9.1和`results/compute_prologue_performance_decision.json`。

### 6.5 `fixed_n32`：固定 K7168 的 N48 与 N32 取舍

目标为 `192×768×7168` 的9071与 `384×768×7168` 的9073实际分支。两侧明确固定 `ReuseBScale=false`，以免将B-scale复用与tile变化叠在一起；固定K7168、4 K-waves、Output4及cache policy保留。baseline为N48+N-tail，candidate为N32+no-tail。

| ID / 目标M | prefetch / Bcache | VGPR baseline→candidate | SGPR | fixed LDS | grid |
| --- | --- | --- | --- | --- | --- |
| 9071 / M192 | 4 / 3 | 168→152 | 49→42 | 12288→8192 B | 192→288 |
| 9073 / M384 | 3 / 0 | 185→143 | 70→61 | 24576→16384 B | 192→288 |

两侧AGPR、scratch、VGPR/SGPR spill均为0。baseline两个入口的指令字节hash和完整metadata与本轮正式9042生成的两个fixed分支完全相同。producer指令bytes分别3956→3024、4200→3408；静态每tile的MFMA数45→30、48→32，来自tile变小，不表示任务计算量减少。

BN32带来更多工作组、较低资源和无需N-tail的地址/输出检查，同时目标N768的N-tile数16→24，A与相应scale tile请求近似增加50%；B聚合输入约相同。288工作组跨过256阈值，比192更大的grid可能增加尾轮，也可能改善并发。最终测量决定，不能从VGPR减少直接认定加速。

证据与私有同register ABI见`fixed_n32/variants.json`、`build_manifest.json`、`device_audit.json`及两侧反汇编。`timing_plan.json`覆盖两个目标，`correctness_plan.json`准备了60个固定K/M-tail/N组合。GPU完成两个目标的signed8rep、51地址shared pool及5轮交替Event确认，两目标N32都明显更慢，**关闭这个实验并拒绝采用，不再扩展其回归**。已测范围与时间见8.2。

## 7. 历史数据给出的验证优先级

下表来自 2026-09-30 当前 745-shape 汇总，**不是本轮时间，也不代表当前瓶颈已确认**。外部 backend 是当时同一轮选出的候选，跨轮/跨编译器不能直接比较。

| M×N×K | 当时最快 Opus / 实际分支 | Opus μs | 同轮外部 μs | 本轮优先检查 |
| --- | --- | --- | --- | --- |
| 192×768×7168 | 9042 / 9071 | 10.5548 | ASM 8.9962 | N48 尾部、output/归约、grid 与 wave 等待 |
| 384×768×7168 | 9042 / 9073 | 13.9680 | ASM 12.2429 | M32/N48、B cache0、请求与归约 |
| 144×7168×16384 | 9062 / M80 split2 | 44.7673 | ASM 43.5469 | fine_wait 与 BN64 分别验证、完整 reducer 成本 |
| 160×7168×16384 | 9062 / M80 split2 | 45.0196 | ASM 44.0384 | grid 224、LDS 并发、实际外部延迟 |
| 80×2048×7168 | 9053 / 9072 | 12.0102 | ASM 11.8421 | fixed-K 已复用 scale，查剩余 issue/归约 |
| 144×7168×768 | 9047 | 6.5711 | CK 6.0746 | register scales、短 K、LDS/wait/输出 |
| 128×7168×1024 | 9047 | 7.2733 | CK 7.0479 | 同上，grid 与短 K 流水摊销 |
| 1×768×7168 | 9041 / 9050 | 5.2881 | ASM 9.2939 | 小 grid、K-wave 供给、补充/尾部 |
| 1×65536×1536 | 9051 | 16.8963 | — | 大 N 低 M 的外部流量与 TA/TCP 吞吐 |

最后一项逻辑 AI≈2 FLOP/B、logical BW≈5.966 TB/s，适合测外部/接口供给。前两项逻辑 AI≈294.6/477.9 FLOP/B、logical BW≈0.680/0.633 TB/s；无法凭这两个带宽数认定为 HBM bandwidth bound，应同时走 compute 文档的矩阵 issue 分析。

来源：`reports/opus_current745_tables_20260930/shape_comparison_745.csv`；其它优先 scale-reuse shape 包括 `32×7168×384`、`16×7168×1024`、`64×7168×768/1024`。历史 resource 表若 traits 参数与当前源代码不一致，不用来计算当前 occupancy。

## 8. 本轮 GPU 验证与最终保留决定

GPU空闲窗口完成各实验的必要正确性和完整Event确认。最终selected仅保留9021 prologue及三个register alias，已通过独立正式构建、device身份审查和12项API检查，并应用到当前源码，见[集成记录](formal_selected/integration_manifest.json)。9022全局prologue、fixed_n32、fine_n64、register全局默认值、fine_wait及narrow_unroll均拒绝。早前含9022的五入口/15项API检查保留为历史证据。n64污染轮次始终排除；646项可选支持域计划只计实际完成的210项。

- `results/register_pilot.json`：9040/9042/9051/9052/9053/9054各测`17×256×128`，两侧signed输入3次重复，errRatio0、输出guard与重复一致检查通过。9041/9050、长K、生产fixed分支尚未由这份pilot覆盖。
- `results/fine_pilot.json`：9060`81×256×768`、9061`97×256×1280`、9062`144×256×16384`、9063`112×256×8320`、显式9069`48×256×7168`，两侧signed3次重复、output/workspace guards、workspace finite与重复一致检查通过。非均匀K分区和部分M tail有有限覆盖，不能称全部GPU边界或时序已证明。
- 上述旧pilot每项仅1轮trace筛选，A/B地址池不完全相同；自动轮换曾先执行probe再deepcopy，旧pilot只检查最后返回轮换buffer，不能声称每个轮换buffer都完成NaN/guard验证。新的register首窗口已使用共享池graph/Event和逐buffer检查；实际赢家、必要机制、异常Event及最终正式API在8.6分别记录，旧pilot不替代这些核验。

| 实验 | 本轮正确性 | 完整调用 A/B 时间 | Counter/资源解释 | 最终保留范围 |
| --- | --- | --- | --- | --- |
| register_reuse | 全部6/6受影响当前赢家、5个新机制、6个异常Event及最终selected 12项正式API完成；完整device身份精确匹配；可选支持域210项通过 | 9040 0.7852×拒绝；三alias六实际赢家1.0045–1.0266×，都5/5更快；两个非赢家路径+0.81%/+0.64%时间 | 9042单目标A/B动态buffer请求精确减少5,376、DRAM bytes近似不变；有限counter见5.9，不指定总瓶颈根因 | **拒绝全局默认修改及9040复用**；三alias限定方案已验证并采用 |
| fine_wait | 九目标signed8rep、4590pool输出及3060split partial检查通过；仅有限shape范围 | 五项5/5退步，另外四项接近0/方向混合；split项包括producer+reducer | 源级等待证明/SGPR代价；无counter，不指定根因 | **本轮全局修改拒绝**，保留原wait，无需扩大被拒绝版本回归 |
| fine_n64 | 后续clean两目标signed8rep、1020输出+1020partial pool检查全部通过 | M144 44.5211→50.7714μs；M160 45.0945→51.1784μs，0.8769/0.8811× | 无counter，新增A请求/资源取舍仅假设 | **已验证拒绝BN64替换**，保留BN128；旧污染轮次不参与结论 |
| fixed_n32 | 两目标signed8rep、51pool逐buffer检查、5轮AB通过 | 9071 10.2291→14.8527μs；9073 13.7766→20.1876μs | 无counter，不能把退步指定为某个硬件层 | **已验证拒绝** |
| narrow_unroll | 六目标signed8rep、3060共享池输出/guard/重复检查通过；fixed-K保持相同 | 四9023赢家近零/方向混合；9024最大K62.5260→63.3566μs，5/5退步 | VGPR/scalar lane spill静态减少；无counter/ATT，未指定根因 | **9023/9024全局runtime展开修改拒绝**，保留基线 |

保留条件：

- 使用当前 scale/累加契约通过正确性，signed 输入、多次重复、M/N tail、K/split 边界与输出 guard 均合格。
- A/B 使用同一 toolchain、同一设备与条件，交替顺序测量，收益超过本轮波动，相关 fallback 没有材料性退步。
- 记录实际 dispatch 和 binary SHA；源码 traits、ISA 和 launcher LDS 相互一致，无新增不可接受的 spill。
- 用真实接口 bytes、wave/issue/依赖等待、动态并发解释收益或退步；counter 缺失时只报告已支持的结论。
- split-K 的最终指标包括 reducer；私有 runtime ID 的结果不直接写进生产 tuned table。

### 8.1 正式分支需要的回归覆盖

下表是最初按baseline dispatch制定的边界检查菜单，说明不同方案需注意的实际device traits。最终采用范围已收敛为9021及三个register alias，必要赢家、机制、异常Event和正式API验证见8.6；未采用的fine/tile/unroll版本不再扩大回归。表内未测支持域仍按未测记录，不构成已经完成的覆盖。

| 范围 | 正式ID与M×N×K示例 | 必须覆盖的分支/边界 |
| --- | --- | --- |
| 9040/9051/9054 | 9040`17×256×128`与`32×7168×384`；9051`16×7168×1024`；9054`64×7168×768/1024` | 最短K、M tail、prefetch填充/尾部、scale组跨tile |
| 9041→9050 | 9041`1×768×7168`、`96×768×4096`、`1×768×8192`、`17×256×128` | grid<256的深prefetch分支、grid>=256 runtime、K>=8192分支、空K-wave |
| 9042→9071/9073 | 9042`17×256×128`、`192×768×7168`、`384×768×7168` | runtime改变、N48两条固定分支、后者不应受Reuse默认值影响 |
| 9052/9053 fixed与runtime | 两ID`17×256×128`及K16384；9052`16×7168×7168`、9053`80×2048×7168` | 空K-wave/runtime、显式true固定9070/9072回归 |
| 9060/9061 | 9060`81×256×512/640/768`、K3072/K7168；9061`97×256×512/640/1280` | loops=S、S+1、M tail、固定与runtime、SFA向量/byte分支 |
| 9062→9064 | 9062`144×256×16384`、`81×256×16384`、`144×256×1408`、`1×256×128` | M80固定、M96 runtime、11个K128分两段6/5、空split |
| 9063各M分支 | K8320分别M48/80/96/112/128、N256；M192,N7168；M160,N128 | 9069/父M80/9066/9068/9067/9065/父M80；后两例grid条件两侧 |
| 9063固定K | M80,N256,K16384；M48,N256,K8192 | M80固定K reducer Vec4/cache0与M48 runtime；短partition/scale bytes |
| 显式历史9069 | `48×256×7168`、`49×256×3200`、`1×256×128` | S6/C2固定、runtime S4/C1、空split；S6/C2固定因K7168不能被父9063的K>=8192条件选中 |
| 未受影响主体 | 9047/9049短K，9044深ring条件，9000–9030支持域样例 | REG_SCALE、普通M、非early路径、无remainder几何保持不变 |

对fine额外选`M=81/97/113`或其它非16对齐M，促使SFA vector helper使用byte fallback；选择K128 tile总数非split整除，覆盖每split不同loops。仅调整wave wait不改变load/store，但仍需验证这些路径与初始化/尾部计数相容。

### 8.2 Fixed N32 的无干扰完整确认

结果文件为`results/fixed_n32_first.json`，状态`passed`；当前源码SHA与本文审查起点一致，设备MI355X/gfx950、SMI6/HIP6、PCI`0000:e5:00.0`，调用者监控未检测外部干扰。每目标两侧signed8次重复通过；同一51地址池逐buffer NaN/guard/结果验证，graph replay以HIP-backed torch Events计时，5轮交替AB，中位数为：

| 目标 / actual fixed ID | N48 baseline μs | N32 candidate μs | baseline/candidate | 时间增加 |
| --- | --- | --- | --- | --- |
| 192×768×7168 / 9071 | 10.2291 | 14.8527 | 0.6887× | 45.20% |
| 384×768×7168 / 9073 | 13.7766 | 20.1876 | 0.6824× | 46.54% |

两个目标对应完成吞吐分别降低约31.13%和31.76%。这支持**保留现有N48分支、拒绝本轮N32替换**；不需要扩大被拒绝候选的回归。静态VGPR/LDS降低并没有带来完成时间改善，说明仅以资源数量/更多grid选择tile不充分。没有counter或动态指令证据，不能进一步声称A流量、TA、L2或dispatch中的某一层已经被证明是退步根因。

### 8.3 N64 的污染记录与后续干净拒绝

两次N64运行均被外部GPU作业重新占用而中断。第一次结果因相同output的rerun被覆盖，未保留完整raw；本文不再给该次时间数值建立证据链接，也不根据它作结论。现存可检查的第二轮是`results/n64_interrupted_2.json`与同名`.log`，设备MI355X/gfx950、HIP7、PCI`0000:95:00.0`。

第二轮JSON顶层状态为`interrupted_external_gpu_work`；contamination记录外部PID`2384048`、检测时间和`gpu_claim_log.jsonl`，明确`timings_usable_for_final_decision=false`。第一shape的局部`event_confirmation.status=passed`只说明该shape已完成检查，**不解除整份文件的污染标记，所有timing均排除于最终保留和因果结论**。

| 第二轮目标 | JSON已保存的Event进度 | 有限时间观测 | 正确性证据范围 |
| --- | --- | --- | --- |
| 144×7168×16384 | 5轮交替AB，共10条measurement | 中位数44.2498→50.3604μs，0.8787×；污染轮次，不作最终性能判断 | 每条measurement的51个共享池buffer均errRatio0、输出/workspace guard与重复一致检查通过 |
| 160×7168×16384 | 3轮AB，共6条measurement；原计划5轮未完成 | 不报告完整5轮中位数 | 已保存的6×51个pool检查均通过，尚未覆盖完整计划 |

两目标两侧直接signed8rep、errRatio0、output/workspace guards和重复一致检查均通过。日志虽出现下一条M160 Event输出，该条未完整保存到JSON，不能计作已完成的pool验证。以上仅保留旧轮次的有限正确性和污染观测，不参与后续干净结论，也不能指定TA/L1/L2/dispatch根因或推广至全体fine系列。

调用者已在idle wrapper加入旧输出自动备份和污染标记，后续rerun保留原始结果；这不改变本节两次中断的证据范围。

后续干净结果为`results/n64_clean_target_0.json`与`_1.json`，文件/Event均`passed`，MI355X/gfx950、HIP5、PCI`0000:85:00.0`；各claim结束returncode0、contamination=false，无外部作业开始记录。两目标两侧signed8rep、每目标10×51pool输出及FP32 partial检查全部通过，合计1020输出与1020partial检查。两侧使用同一51地址池，5轮交替AB/BA，Event覆盖完整M80 split2 producer+reducer：

| 目标 | BN128 baseline μs | BN64 candidate μs | speedup | 时间增加 | candidate快的轮数 |
| --- | --- | --- | --- | --- | --- |
| 144×7168×16384 | 44.5211 | 50.7714 | 0.8769× | 14.04% | 0/5 |
| 160×7168×16384 | 45.0945 | 51.1784 | 0.8811× | 13.49% | 0/5 |

两项目标的同轮speedup范围分别为0.8712–0.8824×、0.8768–0.8876×，五轮均退步，支持**拒绝BN64替换、保留原BN128**，无需扩测已拒绝候选。较低LDS/VGPR和更多grid仅改善了静态资源条件，没有改善任务完成吞吐；未采counter，不能声称新增A请求、TA/L1/L2或dispatch已经证明造成退步。完整核查见`n64_clean_summary.json`，旧污染文件保持原标记。

### 8.4 Register scale reuse 的第一个干净窗口

结果为`results/register_timing_0_kid9040.json`至`register_timing_5_kid9054.json`，各文件与局部Event均`passed`。设备MI355X/gfx950、HIP5、PCI`0000:85:00.0`；六项对应claim结束均returncode0、contamination=false，无`external_work_started`。每目标两侧signed8rep、独立FP32区间契约、输出guard与重复一致检查通过；5轮交替AB/BA，每轮每侧使用同一51地址池并逐buffer检查，共3060个pool输出检查通过。

本表采用完整私有调用的graph/HIP-backed Event中位数；`speedup`是baseline中位数除candidate中位数。“同轮范围”计算每轮baseline/candidate比值，避免用两侧独立中位数掩盖逐轮方向。单侧跨度为`(max-min)/median`，只是描述本窗口波动，不是置信区间。

| private runtime ID / M×N×K | baseline μs | candidate μs | speedup | candidate快的轮数 | 同轮speedup范围 | 单侧跨度B/C |
| --- | --- | --- | --- | --- | --- | --- |
| 9040 / 16×7168×768 | 3.5639 | 4.5389 | 0.7852× | 0/5 | 0.7769–0.7881× | 0.70% / 1.66% |
| 9042 / 64×7168×3072 | 11.6370 | 11.4221 | 1.0188× | 5/5 | 1.0144–1.0264× | 0.94% / 0.87% |
| 9051 / 1×65536×1536 | 16.6825 | 16.6433 | 1.0024× | 2/5 | 0.9960–1.0133× | 1.38% / 0.85% |
| 9052 / 1×7168×7168 | 9.8636 | 9.8370 | 1.0027× | 3/5 | 0.9830–1.0146× | 1.27% / 1.90% |
| 9053 / 32×7168×3072 | 7.9067 | 7.8016 | 1.0135× | 5/5 | 1.0087–1.0263× | 1.46% / 0.82% |
| 9054 / 32×7168×1536 | 6.6259 | 6.3757 | 1.0392× | 5/5 | 1.0278–1.0432× | 1.24% / 0.65% |

9040的时间增加27.36%，五轮及两种调用顺序都退步，幅度明显大于本窗口波动，支持拒绝本轮9040复用及全局默认值方案；无需扩大这版被拒绝9040候选的回归。9042/9053/9054在两种顺序下都有正信号，完成时间分别减少1.85%/1.33%/3.78%；首窗口当时只能支持进一步核验。后续实际赢家、机制、异常Event和最终正式API收尾见8.6，不把这三个旧候选shape直接当作完整赢家覆盖。9051的同轮speedup中位数为0.9988×；9052在AB顺序中位数0.9964×、BA为1.0125×，两者不能据独立中位数的小幅变化认定稳定优化。

必须核对这些私有实验与正式派发的关系：

| 本次shape | 同一公开ID正式派发 | current745中该shape的历史赢家 | 可支持的范围 |
| --- | --- | --- | --- |
| 16×7168×768 / private9040 | 9040 runtime | 9051 | 拒绝9040的复用；没有比较或推翻当前9051赢家 |
| 64×7168×3072 / private9042 | 9042 runtime | 9055 | 仅9042 A/B信号；没有验证比当前9055更快 |
| 1×65536×1536 / private9051 | 9051 runtime | 9051 | 覆盖真实当前赢家的一项，无稳定收益 |
| 1×7168×7168 / private9052 | **9070 fixedK7168，Reuse已true** | 9052→9070 | 这次测试强制runtime9052，不代表正式fixed9070获益 |
| 32×7168×3072 / private9053 | 9053 runtime | 9042 | 仅9053 A/B信号；没有比较当前9042赢家 |
| 32×7168×1536 / private9054 | 9054 runtime | 该shape不在745清单 | 仅候选shape信号；实际赢家验证另见8.6 |

当前测试两侧`.so` SHA与六份结果、`device_audit.json`及已修正manifest的final SHA完全一致：baseline`9a6bb761…`、candidate`e975181b…`。原manifest的编译后SHA与审查后SHA曾不同，原因是旧`llvm-objcopy --dump-section`调用重写host ELF；调用者已保存原manifest，将两阶段记录为`pre_audit_binary_sha256`与最终`binary_sha256`，当前被测`.so`未改变。

首窗口导出的原则是保留9040的false，以独立alias显式opt-in限定经过验证的配置，再核对正式host/device符号与机器码。后续据此只采用三个runtime alias，见8.6。该首窗口未采TA/TCP/TCC/UTCL1/wave counter，不能由较少byte load、VGPR和时间变化指定根因；9042后续的有限counter另见5.9。细节、逐轮数值、顺序统计和身份核对在`register_timing_summary.json`，CPU重现脚本为`analyze_register_timing.py`。

`plans/register_current_affected.json`已按inventory列出41个有效受影响的**当前赢家shape**，两侧private baseline指令逐项与当前赢家官方指令相同：9040共11、9042共1、9051共17、9052共7、9053共2、9054共3。这个计划描述原全局默认值实验的范围，不能整份无差别重跑已拒绝9040。限定9042/9053/9054后有效受影响的当前赢家恰好6项：9042`32×7168×3072`，9053`32×7168×1024/16384`，9054`48×7168×768`及`64×7168×768/1024`。其干净Event、必要机制和正式API均已完成，状态见8.6。

### 8.5 Fine wait 的九个干净Event目标

结果为`results/fine_timing_0_kid9060.json`至`fine_timing_8_kid9069.json`，文件/Event均`passed`，设备MI355X/gfx950、HIP5、PCI`0000:85:00.0`。九项claim均结束returncode0、contamination=false，无外部作业开始记录。两侧signed8rep通过；每目标5轮交替AB/BA、51共享池buffer，合计4590个输出检查均errRatio0、guard与重复一致通过。六个split2/4目标另有3060个FP32 partial检查，workspace有限值、guard与重复一致全部通过。

下表是完整调用的Event中位数。split2/4覆盖producer、reducer以及两次launch之间的设备调度间隙；不能把它当作producer单独时间。两侧库的磁盘、manifest、device audit与各结果SHA完全一致：baseline`04b30617…`、candidate`068fd358…`，实际producer baseline指令与官方当前分支相同。

| 公开/显式ID与实际producer | M×N×K | split / fixed K / S/C | baseline μs | candidate μs | speedup | candidate快的轮数 |
| --- | --- | --- | --- | --- | --- | --- |
| 9060→9060 M80/4wave | 80×7168×7168 | 1 / 7168 / 4/1 | 32.2292 | 32.7116 | 0.9853× | 0/5 |
| 9060→9060 M80/4wave | 160×7168×3072 | 1 / 3072 / 4/1 | 16.6879 | 16.9303 | 0.9857× | 0/5 |
| 9061→9061 M96/8wave | 192×7168×7168 | 1 / runtime / 4/1 | 33.0684 | 33.7116 | 0.9809× | 0/5 |
| 9062→9062 M80/4wave | 144×7168×16384 | 2 / 16384 / 4/1 | 44.3007 | 45.0967 | 0.9823× | 0/5 |
| 9062→9062 M80/4wave | 160×7168×16384 | 2 / 16384 / 4/1 | 45.1713 | 45.7932 | 0.9864× | 0/5 |
| 9063→9068 M112/4wave | 112×7168×16384 | 4 / runtime / 4/1 | 38.3422 | 38.4261 | 0.9978× | 1/5 |
| 9063→9069 M48/4wave | 48×7168×8320 | 4 / runtime / 4/1 | 19.8385 | 19.8511 | 0.9994× | 1/5 |
| 9063→9063 M80/4wave | 80×7168×16384 | 4 / 16384 / 4/1 | 34.4339 | 34.4167 | 1.0005× | 3/5 |
| 显式9069 M48/4wave | 48×7168×7168 | 4 / 7168 / **6/2** | 17.6252 | 17.5272 | 1.0056× | 3/5 |

前五项时间分别增加1.50%/1.45%/1.94%/1.80%/1.38%，五轮均退步；同轮speedup都低于1。第六/七项中位数接近0变化，只有1/5轮更快；M80 split4为0.05%时间改善，显式9069为0.56%，但都只有3/5轮更快。后者同轮speedup范围0.9917–1.0066×，单侧candidate跨度1.60%；不能将独立中位数1.0056×写成稳定优化。五轮来自同一窗口，统计仅描述这次运行，没有置信度或显著性推断。

只有四项目标是该shape在current745中的实际赢家：9062的M144/M160、9063→9068的M112、9063的M80。前三个测试shape的当前赢家分别为9044→9056、9045、9044；`48×7168×8320`不在745清单，`48×7168×7168`的当前赢家为9055。显式9069的K7168/S6C2不能被父9063的`K>=8192 && M<=48`条件选中，本轮明确通过显式ID调用；不能误写成父9063在该K的正式派发。

**本轮全局fine_wait修改不采用。** 保留原保守wait，已拒绝版本不再扩展边界回归。这次有限正确性检查覆盖了K8320的非均匀4分区（17/16/16/16），并未覆盖所有空split/M-byte路径；这些边界仍是将来不同方案若要落地必须验证的条件。counter尚未采集，时间与SGPR变化不证明分支开销、TA/L1/L2或dispatch是根因。逐轮、边界、派发及身份证据在`fine_timing_summary.json`，CPU重现脚本为`analyze_fine_timing.py`。

### 8.6 三个 register alias 的实际赢家确认与限定落地

`results/register_positive_winner_0.json`至`_5.json`已在干净窗口完成。按当前745实际dispatch的official baseline symbol与三个改变入口精确交集，effective changed runtime赢家恰好六项：9042一项、9053两项、9054三项；本表覆盖全部受影响的当前赢家，不是只挑六个代表。固定分支及其它alias未改变。逐项去重审查见[当前赢家完整覆盖](register_reuse_scoped/current_winner_coverage.json)，没有剩余赢家Event target。baseline指令与官方当前分支完全相同；两侧signed8rep、每目标10×51共享池输出检查、guard与重复一致全通过，合计3060输出检查。各claim结束returncode0、contamination=false，结果与磁盘、manifest、device audit库SHA一致。五轮交替AB/BA，每项candidate都5/5更快：

| actual runtime ID | M×N×K | baseline μs | candidate μs | speedup | 完成时间减少 | 同轮speedup范围 |
| --- | --- | --- | --- | --- | --- | --- |
| 9053 | 32×7168×1024 | 4.4314 | 4.3506 | 1.0186× | 1.82% | 1.0096–1.0333× |
| 9042 | 32×7168×3072 | 8.5413 | 8.3201 | 1.0266× | 2.59% | 1.0262–1.0325× |
| 9053 | 32×7168×16384 | 27.4285 | 27.3069 | 1.0045× | 0.44% | 1.0042–1.0055× |
| 9054 | 48×7168×768 | 4.5639 | 4.5075 | 1.0125× | 1.24% | 1.0111–1.0264× |
| 9054 | 64×7168×768 | 4.6322 | 4.5890 | 1.0094× | 0.93% | 1.0068–1.0266× |
| 9054 | 64×7168×1024 | 5.4291 | 5.3726 | 1.0105× | 1.04% | 1.0020–1.0137× |

六个受影响当前赢家均通过，有限机制和异常Event收尾也已完成，支持保留三个alias的限定复用；它不证明全部支持域都适合。长K9053约0.445%的吞吐改善虽本窗口方向一致，幅度较小，不作为广泛收益。其余原alias9040/9041/9050/9051/9052保持false；9070/9072原有显式true、9071/9073显式false继续保留。原全局default和fine_wait方案仍拒绝。三个runtime alias已应用，见[最终集成记录](formal_selected/integration_manifest.json)。

三个register alias与9021 prologue合并，已应用两个源码文件、四个有效device入口；9022保留原prologue。register全部6/6受影响当前赢家Event已覆盖，最终selected不增加register派发范围。新的[正式selected构建](formal_selected/build_manifest.json)与[device身份审查](formal_selected/identity_audit.json)确认：四个改变入口匹配已测private candidate，其余52入口保持baseline；206对象/202device bundle检查通过。selected module SHA为`45efa9ee9b8c82f4f9b169c547a8f68e22c68ea1ac5badd1409d0514145bbd14`。[selected正式API严格汇总](formal_selected/gpu_smoke_analysis.json)确认12项全部通过，覆盖4改变entry与3不变control，official96+private80=176次signed数值调用、重复与guard通过；UTC20:37:42.702–20:39:00.017，PCI`0000:65:00.0`/HIP2，实际加载SHA与owner/claim身份均匹配。此批check-only没有性能结论。当前[集成记录](formal_selected/integration_manifest.json)确认应用源码SHA与被测selected一致，未提交或推送。

CPU隔离编译的私有`register_reuse_scoped`已只在以下三个alias显式补齐`FixedK=0,PadN=false,ReuseBScale=true`，默认值保持false：9042的`small_register_32x32`、9053的`small_register_wavek_32x32`、9054的`small_register_wavek_32x64`。两侧8-entry编译和device审查确认：baseline八项指令hash/资源都等于原baseline；scoped candidate的三个开启项等于已测原candidate，其余五项等于原baseline，共16项全部一致。证据在`register_reuse_scoped/build_manifest.json`与`device_audit.json`；9042后续独立A/B counter见5.9。当前生产文件已采用相同的三个显式true；原被测私有`.so`保持不变，当前源码身份由集成记录独立确认。

正式parent的不同TU已按机器码、完整metadata和归一化descriptor核对，不能仅凭C++类型相同认为编译器调度一定相同。按scalar支持契约，这三个parent在745个shape域各可支持290项：9042与9053各178 runtime会改变、112 fixedK7168保持原显式策略；9054的290项全部runtime改变，合计646个有变化的parent-shape case。这个数量描述候选支持域；本轮针对当前实际使用的六个赢家优化，并以queue/M-tail/N128 scale机制代表、已发现的性能异常和正式入口作为必要核验，不要求把646项可选拓展全部跑完。

`register_scoped_support_regression.json`及22份`plans/register_scoped_*_runtime_changed_*.json`准备了646个runtime case，绑定scoped两侧私有库。**可选支持域拓展现已停止，采用范围仍限定三个runtime alias；不声称646项全域通过。** 停止时CPU核验为7个干净完整片、210项、3,360次signed数值调用通过：9042为114项，9053为96项，9054为0项；污染attempt和partial不计覆盖，未测部分不称通过。动态覆盖以[register支持域核验汇总](results/register_scoped_support_analysis.json)为准，CPU可用[汇总脚本](analyze_register_scoped_support.py)更新：只计完整signed8、日志逐行匹配、单个干净claim epoch和source/plan/runner/library身份一致的primary结果。1轮trace仅用于筛选，不能当作最终收益或退步。

原六个实际赢家Event与scoped/正式candidate的三个入口已完成两侧指令、完整metadata和归一化descriptor逐项匹配；原结果的库SHA、runner、plan、单个干净claim epoch、逐行日志及每目标510个共享池输出也重新核验。因此复用原六个Event，无须仅因host `.so` SHA不同重复计时。可重现审查在[Event与device身份复用](register_reuse_scoped/event_identity_reuse.json)及[CPU准备脚本](prepare_register_minimal_closing.py)。这份审查另复用9053 `32×7168×3072`的原Event，以及9054 `32×7168×1536`的full+1机制Event；不能把其时间外推到不同卡或未测shape。

部分支持域出现7条candidate时间增加≥5%的单trace筛选。9053 `32×7168×3072`已有同device身份的干净5轮Event（7.9067→7.8016μs，5/5更快），不重复。其余六项已完成共享池5轮交替Event：两侧signed8rep、每目标510个pool输出检查通过，合计3,060项；source/plan/runner/library SHA、单个干净claim epoch与日志逐行复算均通过。它们是调用该parent的候选对照，均不是该shape当前tuned赢家。新有限队列为[必要收尾队列](register_minimal_closing_queue.json)，包含5个机制case和6个Event target；没有重跑原六个赢家，也没有继续整个646项扫测。严格CPU汇总见[限定register收尾结果](results/register_minimal_closing_analysis.json)、[复算脚本](analyze_register_minimal_closing.py)。

| candidate parent / 候选对照shape | baseline μs | candidate μs | speedup | 时间变化 | candidate更快轮数 |
| --- | --- | --- | --- | --- | --- |
| 9042 / 8×16384×1536 | 6.6761 | 6.7303 | 0.9920× | +0.81% | 0/5 |
| 9042 / 32×7168×1024 | 4.4447 | 4.3891 | 1.0127× | −1.25% | 4/5 |
| 9042 / 64×7168×1024 | 5.7326 | 5.6432 | 1.0158× | −1.56% | 5/5 |
| 9042 / 176×7168×384 | 5.8353 | 5.7695 | 1.0114× | −1.13% | 5/5 |
| 9053 / 8×7168×384 | 3.4157 | 3.3600 | 1.0166× | −1.63% | 3/5 |
| 9053 / 8×16384×1536 | 6.6024 | 6.6448 | 0.9936× | +0.64% | 0/5 |

六项均未复现≥5%的Event时间损失，因此不因这轮筛选回退三个alias。两项M8/N16384/K1536虽都5/5更慢，实际时间增加分别0.81%/0.64%，必须保留这两个非赢家路径的小幅代价，不能写成“零回归”。9053 M8/K384两轮更慢、三轮更快，独立中位数正向不能称稳定收益。有限Event只解决这些shape的异常筛选，不证明其它fallback没有退步，也不指定TA/L1/L2或MFMA根因。

五个新机制case为9042 `33×384×1536/1664`、9053 `33×384×2048/2176`、9054 `33×384×1024`，两侧signed8rep、output guard、重复一致及完整干净epoch均通过，合计80次数值调用。覆盖prefetch满队列、不同K-wave的非整分配、M33尾部及N384跨三个N128 scale组；trace时间不作收益判断。9054的full+1由原`32×7168×1536`干净Event覆盖，K768实际赢家给出wave间loops2/1；9042/9053已测K384支持域给出有tile/空wave。9054空wave另有旧K128 pilot的signed3原buffer证据，保留其有限检查说明；不声称全部自动轮换buffer通过。具体wave tile分配、full/tail槽位和GPU身份随收尾汇总保存。

224个固定case用指令/资源hash和实际dispatch核对；最终selected API检查已覆盖9071固定边界与9072 M-tail control，早前历史candidate另覆盖9073。不重复整域GPU测试。已有有效结果可按shape、traits、binary、runner匹配后扣除；固定control不能用强制runtime私有调用替代。逐轮、身份与支持域清单见`register_positive_winner_summary.json`、`register_scoped_support_regression.json`，CPU重现脚本`analyze_memory_confirmations.py`。9042单目标限定库A/B counter已完成，有限请求/等待解释见5.9；不将这一目标外推到其它alias或完整支持域。

最初隔离worktree`/root/workspace/opus-bound-candidate-20261007`完成三文件正式candidate CPU构建：prologue两处与三个register alias显式true，默认复用仍false。26个parent/56个device入口中，5改变入口的指令、完整metadata、归一化descriptor匹配已测private candidate；51不变入口保持baseline。见[历史candidate差异](formal_candidate/source_changes.diff)、[构建manifest](formal_candidate/build_manifest.json)、[历史device身份审查](formal_candidate/identity_audit.json)。其15项正式rank2/native E8M0 API smoke全部通过，见[历史GPU结果](formal_candidate/gpu_smoke_analysis.json)：5改变entry与4不变control，official120+private96=216次signed数值、重复与guard检查通过；UTC20:06:01.681–20:07:42.529，PCI`0000:05:00.0`/HIP1，module SHA`51d4b3d7dd033cc99d391d87e89c25e8afc501749bc919a786af492eb717d84a`一致。该五entry版本包括最终拒绝的9022，仅保留为历史证据。

最终采用版本收敛为9021和三个register alias，见[selected源码diff](formal_selected/source_changes.diff)、[CPU正式身份](formal_selected/identity_audit.json)、[构建manifest](formal_selected/build_manifest.json)：26parent/56entry，4改变匹配已测private、52不变匹配baseline，9022恢复旧body；206对象/202device bundles核对通过。最终module SHA`45efa9ee9b8c82f4f9b169c547a8f68e22c68ea1ac5badd1409d0514145bbd14`。独立的[12项正式API计划](plans/formal_selected_smoke.json)和[严格GPU汇总](formal_selected/gpu_smoke_analysis.json)确认：4改变entry与3不变control、176次signed数值调用全通过，实际模块、private device、owner/claim和输出guard一致；UTC20:37:42.702–20:39:00.017，PCI`0000:65:00.0`/HIP2。两批API均check-only，不提供性能结论。

**原checkout已采用最终两个源码文件，尚未提交或推送。** [集成记录](formal_selected/integration_manifest.json)确认应用后SHA和diff与已测selected精确一致、9022源码SHA仍等于原HEAD、`git diff --check`通过。应用前的[source/evidence快照](formal_selected/pre_apply_evidence_snapshot.json)及历史JSON中的`adopted=false`记录的是各自采集时状态，保持冻结；当前采用状态看集成记录。已核准的selected module位于隔离JIT目录，之后默认JIT应使用当前应用源码和记录的compiler/flags，不把旧cache模块称作已更新。必要Event已通过device身份复用和正式API接上应用文件，源码应用无需追加GPU运行。

支持域扫描曾由独立空闲wrapper记录在`register_scoped_claim_log.jsonl`；汇总脚本可重复传`--claim-log`，每项claim/start/end只来自同一个日志窗口，两个wrapper日志不混用事件。历史结果审查保存当时的source/plan/runner/library SHA；源码应用后的集成身份独立记录，不重跑以当前working tree fingerprint覆盖旧证据。

## 9. 可复用的执行方式

### 9.1 编译与产物

本轮隔离实验使用 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++`、gfx950、`/opt/rocm`。完整 flags、源文件 hash、binary hash 记录在各实验 `build_manifest.json`。

保留了与当前候选相关的 MFMA VGPR form、early-inline、禁止 device function call、关闭 post-misched 等选项。baseline/candidate 必须使用同样 flags；编译器兼容处理见 `current_build.json`，它不等于修改生产 kernel 算法。

### 9.2 私有 A/B runner

`experiment_runner.py` 使用ctypes launch，caller预分配数据/output/workspace，采用当前原生scale与reference累加/误差检查。执行分成三个阶段：

1. **正确性与trace筛选：**仓库`run_perftest`自动轮换，通过torch.profiler收集kernel GPU活动时间，去首iter并IQR处理，再给每轮aggregate。A/B各自deepcopy地址池；split值通常是producer/reducer活动时间和，不含两次launch之间的空隙。保存每轮值及中位数，不把它称为无profiler完成时间。
2. **拟保留修改的最终event确认：**新增`--event-confirm`使用相同shared pool地址、CUDA graph/HIP Event，不开启torch.profiler；每个pool output/workspace先NaN初始化并检查guard，逐buffer验证，计入完整split调用及中间空隙。该阶段完成后才能提出稳定性能结论；当前状态以实际结果为准。
3. **counter诊断：**rocprofv3独立采集，使用定义已核对的PMC组，工具/replay耗时与上述性能结果分别报告。

两种计时均保存GPU/toolchain/source/binary、方法、pool与round信息。父候选和私有runtime是否对应同一个device分支需要明确记录。

plan 的基本格式如下，路径和 shape 由当前实验实际选择填写：

```json
{
  "workspace": false,
  "libraries": {
    "baseline": "reports/opus_bound_analysis_20261007/register_reuse/baseline/experiments.so",
    "candidate": "reports/opus_bound_analysis_20261007/register_reuse/candidate/experiments.so"
  },
  "targets": [
    {"kid": 9051, "shape": [16, 7168, 1024], "seed": 17, "signed": true}
  ]
}
```

```bash
/opt/venv/bin/python3 reports/opus_bound_analysis_20261007/experiment_runner.py \
  --plan <本轮计划.json> --output <本轮结果.json> \
  --rounds 5 --iters 51 --repetitions 8
```

fine_wait/fine_n64 使用 `workspace:true` 和目标的 `split`；fixed-K/shape/stride/16B 对齐契约见各 `variants.json`。GPU 选择与物理设备锁由调用者管理，多个实验串行。SMI GPU 编号与 HIP visible ordinal 不一定相同，必须按 PCI ID 核对。

空闲监测发现SMI报告宿主PID、`/proc`报告容器PID，旧调用者直接比较可能把自身较大的Event轮换池误判成外部作业；先前被中断的attempt仍排除，不据此改成干净结果。当前直接Python调用新增[身份握手](owned_python_launch.py)：目标导入HIP前打开KFD并等待父进程，父进程检查容器child PID、nonce、实际KFD fd和唯一新增sysfs宿主PID，再确认仍存活并ACK；目标在同PID/同mm中通过runpy运行，监测仅排除已验证的宿主PID。6个CPU分支用例和只打开KFD的实际探测通过，未分配GPU显存、未提交计算，见[身份验证记录](owner_identity_cpu_validation.json)。后续日志逐命令保存`owner_identity`及launcher SHA；0个或多个新增PID视为未解析并终止自身child。该方式适用于直接Python runner，rocprof子进程不纳入此握手。

profiling单独运行`--profile`，以rocprofv3的实际gfx950支持列表选择事件并过滤目标producer/reducer。先检查counter definitions，再把第5节的公式映射到该工具；不把原文名称自动认定为当前可采集counter。`--event-confirm`为最终性能确认入口，参数与pool规模按当前runner实际实现和结果manifest记录；未完成该阶段前只报告筛选值和有限正确性。

计数CSV解析示例：

```bash
/opt/venv/bin/python3 reports/opus_bound_analysis_20261007/parse_counters.py \
  <counter_collection.csv> --output <解析结果.json> --last <实际样本数> \
  --cu-count <已核对CU数> --simd-count <已核对SIMD数>
```

split-K只在文件包含相同调用条件的一个producer和一个reducer时加`--pair-split-k`。脚本拒绝不一致身份、重复raw/derived或不完整样本；不同PMC pass保留独立结果，解析出的profiled TB/s/MFMA吞吐与无profiler优化收益分别报告。

## 10. 优化判断的重点

每次选择一个具体 shape 和实际 kernel，完成下面这条分析链：

```text
实际 dispatch / 有效工作
  → 分层 bytes、吞吐和延迟
  → GL2/SoC 与 channel
  → TA/TCP/UTCL1 的回压
  → wave 依赖与 issue 等待
  → 资源上限、grid、dispatch 和有效在途请求
  → 一个明确改动
  → 正确性与完整调用 A/B
```

对当前 Opus，可操作的问题是：scale 请求是否重复、wait 是否过早限制在途请求、tile 的资源/复用取舍是否让实际供给不足，以及runtime展开是否引入scalar-lane开销。它们对应第三章的请求量、`T_s` 与有效并发，最终用第四章的counter和完整任务时间决定保留。

**目标是缩短正确完成 GEMM 的时间，并解释限制发生在哪个接口。** HBM 峰值、logical BW、occupancy、wave life、TA 合并率均为证据的一部分，不能单独代替这个判断。

这也是第五章总结在当前Opus上的落实：先用分层上限和软件请求模型提出假设，再由匹配定义的counter和完整时间检验，最后保留验证过的改动。各层效率不构成可以直接相乘的固定系数；缓存命中会改变路径和事务数量，回压会改变供给与停留时间。少发scale请求可以改善issue而几乎不改变HBM bytes；更长wave也可能因更多复用完成更多有用工作。因此每次改动后重新核对`F_useful/F_exec`、分层bytes、并发与等待，并以正确完成任务的配对时间决定采用。

## 11. 2026-10-08：分层counter与ATT把限制收窄到具体接口

### 11.1 按最新总结沿派发、请求、返回与消费定位

[第一至第五章最新合并总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)把问题分为：CPC/SPI向CU提供工作，SQ/VMEM经TA/TCP/UTCL1发出访问，GL2/SoC服务访问，数据返回后在VGPR/LDS被计算消费。LDS属于CU局部支路；TA地址处理与返回重组也不是同一个接口。此次先固定实际selected实现，再由各层counter筛选线索，最后用ATT定位等待PC，不依据单个峰值或occupancy先判“全部memory bound”。原文MI300/2024年的参数不直接移植为MI355X实测上限。

目标为当前9021的`480×7168×384`，补充同M/N、同224WG/896wave grid的`K16384`作为长窗口。库、代码对象及kernel指令与Oct7 selected精确一致，完整[device身份](../opus_resume_20261008/sfa_packed/device_audit.json)和[恢复记录](../opus_resume_20261008/recovery_manifest.json)保留；采集诊断时生产为Oct7 selected，Oct8应用见11.7。四组短K独立pass在PCI`0000:65:00.0`完成，[layer分析](../opus_resume_20261008/diagnostics/layer_analysis.json)逐项验证应用/plan/runner/library SHA、code-object URI切片SHA、agent/trace、owner PID及clean claim。每组57次匹配dispatch取末尾51次，共204次选定记录；228次raw记录均保留。单独的raw-clock与ATT各有自己的过程、时间窗和来源，不能混合成同一次dispatch或墙钟瓶颈分解。

### 11.2 四组短Kcounter：可解释的比率与接口线索

下表取每个**独立pass内部逐dispatch比率的中位数**；raw分子/分母中位数仅用于交代量级，不以“中位数之比”替代已有计算。四组profile duration中位数分别为7.840/7.800/7.880/7.960μs，带工具扰动；它们不构成Event性能结果。此次profiling应用未对比数学reference或检查输出guard，正确性证据另行保留。公式、原始行、callback与定义见[layer_analysis.json](../opus_resume_20261008/diagnostics/layer_analysis.json)及[可复算脚本](../opus_resume_20261008/diagnostics/analyze_layers.py)。

| 来源 / 观测 | K384结果 | 支持范围与限制 |
| --- | ---: | --- |
| L2 `HIT/(HIT+MISS)` | 70.245%；hit206507、miss87488 | 请求事件份额；UC reads算miss，不能解释为70.245%的HBM字节被缓存 |
| TCP四tagram请求份额 | 28.294/21.707/28.292/21.707% | 聚合份额有差异；无channel/set/TCP实例分布，不能确认热点 |
| TCC tag stall / busy | 0.02237 | raw上下文比率；Busy非windowable、probe可在多处阻塞，不是2.237%墙钟stall |
| TA buffer cycles / wavefront | 15.908 | 混合matrix/scale请求的计数比；不是整核时间或独立服务延迟 |
| UTCL1 serialization / gate | 0 | 本次事件为0；不证明地址转换完全没有成本 |
| UTCL1 inflight-max / gate | 0.02525；raw102846 | 同TCP上下文比率；gate非windowed，不是2.525%的GEMM时间 |
| UTCL1 thrashing / gate | 0.00009687；raw399 | probe/重叠计数有限，未支持thrashing主导 |
| TA downstream stall / busy | 0.16320；raw107767 | 两者非windowed，支持下游回压线索，不能指定某一级或16.320%耗时 |
| DRAM读credit / GMI读credit / GMI写credit | 0 / 0 / 0 | 本pass没有这些credit stall事件，不能据此排除外部服务延迟 |
| DRAM写credit stall | raw90419 | 没有安全的同范围墙钟分母；保持原始事件 |
| EA read level / read request | 1077.484 nominal counter cycles/request | SDK请求驻留积分/计数公式；非独立校准ns，也不是HBM-only latency |
| TCP→TCC latency / read request | 689.437 nominal counter cycles/request | latency非windowed，定义包含返回atomic；当前目标ISA无atomic，仍保留端点/时钟限制 |
| UTCL1 LFIFO-not-resident / client inflight integral | raw42192 / 18557402 | 翻译下级等待与请求驻留线索；无安全的occupancy或elapsed-time比例 |
| SQ ANY / all issue / LDS issue | 64.296% / 11.595% / 1.804% wave life | 同wave quad-cycle范围；ANY未拆分VMEM/LDS/VALU/barrier，LDS issue不含LDS返回依赖等待 |
| LDS active / instructions / bank-conflict | 3.431%；76608，85.5/wave；raw172032 | 累计wave时间、指令事件和冲突事件分别解释；不能直接推导LDS墙钟成本 |

这些结果未证明L2饱和、特定channel热点或HBM带宽用满。翻译在途容量、TA下游回压、TCP/EA请求驻留均值得关联到等待指令，但各自单pass不能按比率大小排“唯一根因”。更明确的筛选结论是：累计wave的依赖等待类别比instruction issue wait更大，需要继续区分返回与同步；仅减少consumer LDS指令数量没有足够依据。

### 11.3 Raw GRBM实例：排除CSV求和错误后仍保留时钟窗口限制

[clock_audit.json](../opus_resume_20261008/diagnostics/clock_audit.json)审查六个独立pass。Raw GRBM counter catalogue标明八个XCC实例，JSON callback每dispatch记录八值，CSV则把八值之和写成一个scalar；raw CSV scalar不能作为`max(GRBM_COUNT)`。已按顺序保存全部raw值，callback未携带实例坐标，不能为某一个值指定XCC编号。

短K raw最大值中位数39844，对应profile duration8080ns；逐dispatch`maximum/duration`中位数4.929GHz，51/51超过记录的2.4GHz上限。长K raw最大值中位数202587，duration81600ns，该比率中位数2.476GHz，46/51仍超过。求和错误被独立证实，但raw maximum与kernel timestamp的时钟/窗口对应仍未核准，故不输出绝对MFMA利用率、动态occupancy、固定因子修正或counter-cycle转ns。

Oct7显式`OPUS_GFX950_GRBM_*_MAX=reduce(...,max)`的scalar是不同采集方式，旧记录继续冻结。本次不会把新raw CSV求和结论套到所有旧counter。各层非windowedgate/busy/latency有自己的范围限制，同样不能用GRBM推算时间把它们修成墙钟百分比。

### 11.4 ATT将短K线索定位到producer与barrier，长K区分两类边界

[ATT分析](../opus_resume_20261008/diagnostics/att_analysis.json)每K一次capture，仅SE0/CU0上四个完整wave有指令时间线；成功MFMA issue按`time+stall`取值，所有区间为shader clocks。waitcnt依赖记录说明队列内有哪些指令，不给出单条访存返回时间，partial LGKM wait也可能在等待matrix时仍携带scale读取。因此下面的局部定位不会自动成为全卡性能百分比。

K384四wave平均时长9100 clocks，首条MFMA前4523 clocks，占49.702%；首末MFMA之间30.848%，最后MFMA到wave结束19.451%。startup包括kernarg SMEM等待、K0 matrix issue、scale生产/K1预取/发布barrier、K0 matrix及scale LDS读取、runtime setup/clear/K2预取。其开销并非SFA consumer打包一项。

短K scale producer上，SFA vector16 VMEM`0x29e4`→`vmcnt(0)`在`0x29f8`→LDS write`0x29fc`之后，才发SFB VMEM`0x2a64`→wait`0x2a74`→LDS write`0x2a78`；其他三个wave先到`0x2b5c`barrier，attempt跨度672 clocks而release跨度4 clocks。首个VMEM等待包括八条K0 matrix async load和SFA，不能把它的196-clock区间当作SFA单请求latency。该trace支持测试两侧scale请求更早发出的机制，随后由11.6完整Event确认任务时间与候选ATT机制。

K16384平均startup占比降至3.510%。普通interior `MFMA15→下一tile MFMA0`边界中位数492 clocks，包含下一matrix LDS读取、runtime ring bookkeeping和八处async VMEM issue；tile30/62/94后的scale-panel reload边界中位数2014 clocks，额外包含matrix/SFA VMEM wait、SFB请求、LDS发布及barrier。它们应分别测量。tile内`MFMA3→4`中位数184 clocks也包含next-scale LDS读/打包与barrier，局部延迟不等于外部带宽饱和。最后tile的BF16转换、C LDS输出、nop与endpgm段另行核算，不能混入稳态供数预算。

### 11.5 从诊断到单机制实验的当前状态

此前[SFA packed候选](../opus_resume_20261008/results/sfa_packed_screen_analysis.json)十项clean、同51地址池、5轮AB/BA HIP graph Event均通过数值/guard/重复，但十项耗时中位数全增加，已拒绝。代表`480×7168×384`为6.6142→7.1734μs，增加8.455%；即便静态consumer LDS读减少，producer byte scatter与地址处理成本也可能抵消它，VGPR164→178本身也不是原因证明。后续暂停的DPP准备不代表已测试方案。

本次据ATT位置提出[scale issue/publish实验](../opus_resume_20261008/scale_issue_publish/source_changes.diff)：保留原始scale布局和请求量，在prologue及panel reload先issue SFA/SFB，再publish到LDS，保持原同步handoff、M尾填充与issue/publish guard一致。[device_audit.json](../opus_resume_20261008/scale_issue_publish/device_audit.json)确认candidate只改变9021、9022精确不变；9021 VGPR164保持、SGPR98→99、LDS105504B与scratch/spill0保持，静态matrix/scale VMEM、DS读写和MFMA数量不变。[ISA次序核对](../opus_resume_20261008/scale_issue_publish/isa_order_audit.json)确认完整SFA路径两侧请求先于首次publish；M-tail逐byte等待保持原样。当前构建状态由[build_manifest.json](../opus_resume_20261008/scale_issue_publish/build_manifest.json)提供，不覆盖准备时source manifest中的历史状态。

这一实验检验scale请求发出/等待/发布的次序与wave不均衡，不能写成HBM bytes减少或绝对compute利用率提高。短K、K128空未来tile、M尾以及K4096/4224的panel边界已进入独立正确性与共享地址完整Event；长K普通边界保持独立问题。

### 11.6 Event和候选ATT共同验证请求/发布限制

[十项筛选](../opus_resume_20261008/results/scale_issue_publish_screen_analysis.json)后，[独立confirmation](../opus_resume_20261008/results/scale_issue_publish_confirmation_analysis.json)覆盖精确42/42实际赢家和两个小域边界。相对Oct7 selected，42项geomean加速1.575621%，各shape median之和加速0.997733%；41项median更快、30项5/5轮更快、0项5/5轮更慢。唯一负median `416×7168×16384`耗时增加0.034535%，但4/5配对更快，保留原值。K16384组geomean加速0.613%，普通steady边界未因此被认定为全部解决。两次均为clean物理PCI、共享51地址池、5轮AB/BA无profilerHIP graph Event；confirmation的704次原地址数值调用与22440个Event池输出检查全部通过。

非赢家`1×128×128`耗时增加1.429%/2.328%，两窗口均0/5轮更快；`15×128×256`增加0.369%/0.454%，两窗口均1/5轮更快。这是明确保留的有限代价，不隐藏，也不外推成所有小M/短K的规律。按Oct7相同标准保留global9021，正式TU/API通过后已应用；完整取舍统计及K分组见Compute第11.6节。

[candidate ATT对照](../opus_resume_20261008/diagnostics/scale_issue_publish_att_analysis.json)中，producer的SFA→SFB成功issue间隔328→120 clocks；SFA issue到两侧LDS发布结束736→388，到publication barrier release968→620。其VMEM wait事件总长588→188 clocks，依赖含K0 matrix/SFA/SFB，不能当作独立scale latency。其他三wave的barrier时长668/636/680→268/220/208，release跨度均4 clocks；consumer scale LDS wait保持平均37+4 clocks。四wave平均scale/K1/barrier段1249→913，而后续matrix/scale LDS段585→640，保留相邻阶段的抵消。

这把有效优化点定位到scale请求重叠与LDS发布时机，并未证明LDS bank或HBM带宽是主限。ATT仅每版本一次SE0/CU0完整捕获，不据总wave跨度9100→8753计算全卡性能提升；Event决定收益，数值/guard检查决定正确性。实际候选PC已改变，不能复用baseline PC语义；ATT事件也不揭示active-lane事务数。

### 11.7 正式集成检查

[正式device审计](../opus_resume_20261008/formal_selected/identity_audit.json)确认相对Oct7 selected仅9021一个entry改变、其他55个保留；全部generated TU/host调用和linked device身份通过。[正式API审计](../opus_resume_20261008/formal_selected/gpu_smoke_analysis.json)在clean PCI95/HIP7完成八目标、128次数值/重复/guard调用，实际加载module匹配新official SHA。该检查无性能计时，Event仍按PCI65同卡结果决定保留。

已应用精确已测单文件源码，见[Oct8集成记录](../opus_resume_20261008/formal_selected/integration_manifest.json)；生产9021 SHA为`2ff8cf90368c6945d3394fe11075c2336e6f8f99888c7167d3eb550045009bfa`，small traits继续保留Oct7已采用状态。9000/9020/9022、共享helper和全部Oct7实验证据冻结，旧文档SHA描述Oct7时点；Oct8收尾记录绑定更新后的文档。下一处未完成的机制问题是长K普通tile边界的matrix async issue与ring bookkeeping，需独立实验，不因本次scale改动便称全部限制已消除。当前`applied_verified_not_committed`，没有commit或push。

## 12. Oct8 剩余系列：按供数层次定位，再由完整调用取舍

本轮继续处理9020、9022、9023、9024、9030和全部公开9040+：共23个parent、44个与历史赢家相交的实际producer配置、536个历史winner形状。以已验收9021的Oct8正式模块为基线，9000、共享helper、9021和此前已采用的三个runtime B-scale alias保持。公开编号不等于单一device body；9040+的18个parent包含33个实际配置，另有两个无历史winner的producer和五个reducer。精确对应见[完整清单](../opus_remaining_20261008/inventory.json)、[逐配置处理状态](../opus_remaining_20261008/family_progress.json)。44个配置各自一个真实winner的正式API signed2/reference/repeat/guard检查全部通过；这项数值覆盖不等于536项新性能覆盖。

依据[最新合并总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)，分别检查VMEM请求、返回队列、VGPR/LDS消费与同步到达。没有把短K启动、长K普通tile、scale-panel refill、最终输出混成一个带宽瓶颈。ATT以当前CO/FUNC/PC重定位；成功issue=time+stall，duration已经含stall，wait依赖只标识队列成员。所有clocks均为shader clocks，不换算ns。GRBM窗口仍未校准，不报告绝对MFMA utilization或动态occupancy，也不套固定倍率。

### 12.1 证据有效范围和物理流量

[SQ/EA分析](../opus_remaining_20261008/diagnostics/sq_ea_analysis.json)保留20个有限pass，严格owner检查接受10个、排除10个；即使wrapper写contamination:false，发现非owner也排除。没有为补齐表格全量重采。9030必要的ATT/counter单独串行重采；9055原CU0 decoder code:null捕获保留并排除，CU1补采成功。9024 target12有非owner，不能用其与target11构造252/264WG对照。第一次ATT filter误用mangled symbol导致无波形的记录保留，修正为SDK要求的demangled函数名后才分析。

9030的clean SQ/EA在同一有限采集口径中给出median read **2,118,863,744B**、write **2,148,535,936B**、total **4,267,483,264B**；写量约2.149GB/2.001GiB。它确认大输出服务必须进入预算，但单独请求量不证明HBM已饱和，profiled时间不代替无profilerEvent。9062的split2 producer和reducer分开保留counter，完整候选计时包含两者；不能拿producer局部等待解释整个API时间。

### 12.2 Scale issue/publish：相同字节量，不同请求次序

9020 fixed384的scale producer处于起步关键路径，其他wave在publication barrier等待8–1268 clocks；long runtime代表startup只占约3.3%。9022短K有SFA load→wait/store→SFB load→wait/store串行，长K的panel32 refill也串行。9023 runtime的动态普通tile/对齐refill不执行静态大批readlane，实际每wave仅六次恢复用于remainder/output，不能把静态spill数量直接当稳态HBM损失；9024九个真实winner均fixed7168，无spill，startup约15%。对应[merged ATT](../opus_remaining_20261008/diagnostics/merged_att_analysis.json)和[narrow ATT](../opus_remaining_20261008/narrow_att_analysis.json)。

候选保留E8M0、标准(16,16) B preshuffle、K128 scale、FP32累加/BF16输出、u16 panel布局、尾界和0x7f填充。9020仅将SFB先issue，再执行原SFA load/publish，最后SFB publish。最初raw byte尝试被编译器提前消费，数值失败后修正为同一volatile asm中的vmcnt(0)+pack；失败revision和所有SHA保留，失败版本不进入性能选型。最终scoped源码只作用已有fixed384实例，其余六body在正式编译后精确不变。没有减少必要barrier或改变ring出版/退休。

完整[166项Event审查](../opus_remaining_20261008/scale_issue/coverage_independent_review.json)含9020 fixed384全部七winner及9022全部159winner。每项同一共享51地址池、五轮AB/BA HIP graph Event、signed8/reference/repeat/guards，通过strict物理GPU claim。9020七项median全正，geomean **+2.571542%**，34/35配对更快，保留已有fixed384；`12288×7168×384`第2轮speedup0.834618、调用时间增加19.815%完整保留，该shape AB geomean −4.174%、全七shape该轮geomean −0.493%。没有删值、认定它已证明是噪声或重测到正。机械analyzer的严格all-round占位推荐保留，由[最终取舍](../opus_remaining_20261008/scale_issue/retention_decision.json)说明人工判据。

9022整体geomean **+0.820755%**，127项median正、32项负，但六个真实winner五轮一致变慢；K16384的20项中12个paired median负。拒绝全局候选，保留原scale-first矩阵prologue，不用短K收益隐藏稳定winner回归，也不新增未经测试的K阈值。9020其他分支screen混合，其中fixed3072 speedup0.972995、0/5更快，所以同样保留原实现。

[narrow完整13winner](../opus_remaining_20261008/narrow_candidate/retention_recommendation.json)支持9023 runtime geomean **+1.117277%**、9024 fixed7168 **+1.274511%**；9024九项均5/5更快。9023保留`544×7168×16384`median时间+0.5422%与M16非winner+0.4165%的有限代价。VGPR232→251、scalar lane spill46→48代表资源余量减少，不能把零scratch写成无资源风险；收益机制由baseline ATT和candidate ISA支持，没有candidate ATT时间收益声明。

### 12.3 Ring、register和大输出：局部等待不等于可删耗时

9030 interior barrier累计share约22%，但cohort arrival skew median908 clocks、最后MFMA到达跨度900、last arrival→release仅4。它反映供数/计算到达不齐，不是可直接删除的22%墙钟。midpoint候选保留所有publication/retirement及最后alias-C保护；完整十winner使用预算后的同八地址池、51调用graph、五轮AB/BA和256-row FP32区间参考，不能混称自动51地址池。最终十项median全负，geomean **−0.220256%**，仅4/50配对更快；[拒绝global9030](../opus_remaining_20261008/large_midpoint/global_decision.json)。早期代表+0.261%和两项support控制回归均保留。

9046 cluster2的MFMA边界gap median648 clocks、普通边界316，VMEM wait约20，主要线索在LDS/issue/barrier而非单纯HBM等待。9062首barrier到达差2308，慢wave的SFA尾路径使初始matrix晚发；提前矩阵候选两个winner均退。9040/9047短K startup约76%/66%，SMEM waits约1048/1019；9040起步VMEM wait很短，9047另有约169 clocks的混合VMEM/LGKM wait，不能全归VMEM。当前捕获仍不支持更深queue或扩大reuse。9051 startup约87.8%，matrix/byte issue和VMEM等待可以结合旧exact-config LFIFO/translation线索，但这些不能给单请求HBM latency。其新scale-first候选两个代表正后扩大一次完整17winner，结果八正、八负、一平，仅三项5/5更快，也有三项5/5负；[统一拒绝9051](../opus_remaining_20261008/register_issue_order/coverage_analysis.json)。

small/fine本轮候选实测范围是四配置/21个真实winner：9062两项、9046/9055各一项、9051全部17项；四候选都拒绝，分别见[small最终范围审计](../opus_remaining_20261008/small_final_closure_audit.json)。七配置各有自身有限ATT，33配置各有自身数值门；其余keep依据自身type/source/机器身份、数值和存在时的历史exact-config证据。共享机制解释或其他配置的失败不是其性能实测。9071保留N48，旧N32失败仍有效，新草稿在build前停止。18parent逐配置表和五reducer范围见[small总结](../opus_remaining_20261008/small_family_review.md)。

### 12.4 正式集成状态

正式CPU身份审查已通过：26parent/56entry仅9020 fixed384、9023 runtime、9024 fixed7168三个entry改变且精确匹配已测候选；其余53entry、313生成文件、206build objects、202linked gfx950 bundles/263device entries符合Oct8当前基线。9000、9021、共享helper和全部small entry保持，见[identity审计](../opus_remaining_20261008/formal_selected/identity_audit.json)。[正式44项API](../opus_remaining_20261008/formal_selected/api_gate/gpu_api_analysis.json)在clean PCI65全部通过，352次official加120次private signed8/reference/repeat/guard和实际module SHA通过。三个精确源码文件已应用，2691个构建输入逐字匹配正式selected树，状态`applied_verified_not_committed`，见[本轮集成记录](../opus_remaining_20261008/formal_selected/integration_manifest.json)。正式44项API与原44个历史winner配置是不同集合；small在最终API中为18个parent control，原33配置数值门结合40entry机器身份保持，不冒称重新测完33配置或278winner性能。应用审计的排序问题及授权文档追加检查失败记录保留并已纠正；其余4972冻结证据SHA不变，原文档第12前字节保留。

旧Oct7/Oct8构建、原始结果和SHA snapshot保留；本节新增改变原文档全文SHA，由remaining最终manifest绑定。收益是同一形状等权集合的有限实验，不是生产应用batch收益或全部支持域保证。本轮未执行全745重tune。


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
