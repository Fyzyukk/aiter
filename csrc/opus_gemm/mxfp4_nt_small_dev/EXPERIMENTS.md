# small-tile pipeline — 实验记录（2026-11-13）

## 来源与结构
- B_DIRECT 部分直接取自 work_smallm 的 smallm_kernel（已验证：M ≤ 64 时 60 个目标形状里 56 个快过 asm，见 work_smallm/EXPERIMENTS.md），
  只做了改名（namespace opus_mxfp4_small、kargs opus_gemm_mxfp4_small_kargs_gfx950）和把 split-K 归约 / C 写回抽成公共函数。
- B_LDS 是新写的：A、B 都经 LDS-DMA（buffer_load … lds）进 LDS，D 级 LDS 环，每 K tile 一个 barrier，scale 直接读进寄存器。
  - A：8 行一块（一条 DMA = 64 lane × 16 B），块距 1088 B，16 B 块按 (r>>1)&3 XOR（沿用 smallm_lds，ds_read_b128 无冲突）。
  - B：shuffle_weight 的 (16 列块, K 半段) = 连续 1 KB 原样拷入，读 lane*16（天然无冲突）。
  - 一个 K tile 的 A / B DMA 在 4 个 wave 间轮流分配；wave 布局 1×4（BM ≤ 64）或 2×2（128×128）。
  - 计数等待：迭代 kt 只等自己的拷贝，比它新的恰好是 D−2 组（每组 G = A_DMA + B_DMA + SA + SB 条）。
- 两种路径共用：scale 用 e8m0_shuffle 的原始 dword + 编译期 opsel；跨 block split-K（最后到达的 block 归约，sc1 读写，无 L2 flush）；
  M / N 尾块靠 buffer range（越界读 0、写丢弃）+ 16 列块的 C 列屏蔽；K 超出部分走 0 长度描述符，循环里无分支。

## B_LDS 的两个问题（trace 定位，HIP 6）
1. **编译器在读当前 stage 前等刚发出的下一组拷贝**：先发 kt+D−1 的 DMA 再 ds_read 时，编译器分不清 LDS-DMA 写的是另一个 stage，
   在 ds_read 前插 `s_waitcnt vmcnt(2)`（每 K tile ~1100 cycle）。→ 改成先 ds_read 本 K tile 的全部操作数，再发下一组拷贝。
   32×128 D2：256×4096×4096 9.6 → 8.9 µs。
2. **一个大 LDS 数组导致"全部可能别名"**：D ≥ 3 时编译器仍在 ds_read 前等所有在途 DMA（vmcnt(2)，400–600 cycle / K tile），
   等于把环压回 D = 2。→ 每个 stage 用独立的 `__shared__` 对象（lds_stage<I, BYTES>()），waitcnt 分析能区分 stage，
   等待变成 vmcnt(14–21)。32×128 D4：256×4096×4096 → 7.7–7.9 µs。

冒烟计时（HIP 6，CU_OCCUPANCY 前后为空；graph 100 次 × 5 取中位数；asm 同法，见 perf_all.py）：

| 形状 | B_LDS 最好 | asm |
|---|---|---|
| 128×4096×4096 | 7.45（32×128 D4） | 5.2 |
| 256×4096×4096 | 7.86（32×128 D4） | 5.8–5.95 |
| 512×4096×4096 | 11.2（64×128 D4） | 7.2 |
| 256×4096×8192 | 12.3（32×128 D4） | 8.6 |

→ M = 128–512 仍慢于 asm 约 1.3–1.55 倍（2×2 的 128×128 是 1.5–1.9 倍、B_DIRECT 是 1.2–1.6 倍）。
按用户决定，正式性能测试改到无占用的机器上用 prof/perf_all.py 做；这里没有继续调。可能的方向（未验证）：每 CU 多个 block
（32×128 D3 用 62 KB LDS，只能放 2 个；减小 stage 或 launch_bounds(256, 2)）、A 的 DMA 和 B 的 DMA 分到不同 wave、
2×2 / 1×4 以外的 wave 分工、split-K 与小 tile 组合。

## 资源（标准 clang，全部 VGPR spill = 0）
| 配置 | VGPR / AGPR | SGPR spill | LDS |
|---|---|---|---|
| direct 16×32 KS4 D2–4 | 80–120 / 8 | 0 | 6 KB |
| direct 16×64 KS2/4 D2–4 | 80–196 / 8–16 | 0 | 4–12 KB |
| direct 32×32 KS4 D2–4 | 92–160 / 16 | 0 | 12 KB |
| direct 32×64 KS2/4 D2–4 | 92–244 / 16–32 | 0 | 8–24 KB |
| lds 32×128 D2 / D3 / D4 | 152 / 16 | 0 / 0 / 1 | 41 / 62 / 83 KB |
| lds 64×128 D2 / D3 / D4 | 204–232 / 32 | 0 / 0 / 25 | 50 / 75 / 100 KB |
| lds 128×128 D2 / D3 / D4 | 292–296 / 64 | 0 / 10 / 34 | 68 / 101 / 135 KB |
| lds 32×256 D2 / D3 | 228–232 / 32 | 0 / 19 | 74 / 111 KB |
| lds 64×256 D2 | 296 / 64 | 2 | 83 KB |
（SGPR spill 进 VGPR lane，不走内存。host 只实例化上表这些配置。）

## 正确性
prof/check.sh：每个配置 264 次 = 目标 (N,K) 6 组 × M ∈ {1,64,128,512} × split {1,2}，
加不对齐 M ∈ {3,50,100,300} × N ∈ {128,400,1040} × K ∈ {256,768,2304} × split {1,3,8} × batch {1, 2 + MXFP4_SFA_K_PATTERN}。
scale 补齐区填 0xFF（e8m0 NaN），补齐值若进入 C 即报错。结果见下（HIP 6 / HIP 2，被占用的卡上跑，不影响正确性）。

结果（prof/check_results.txt）：19 个配置全部 264/264 errors=0——
direct 16×32 KS4 D3/D4、16×64 KS4 D3、16×64 KS2 D3、32×32 KS4 D2/D3、32×64 KS4 D3/D4、32×64 KS2 D3；
lds 32×128 D2/D3/D4、64×128 D2/D3/D4、64×256 D2、128×128 D2/D3、32×256 D3（split 取 1/2/3/8）。
补充：lds 128×128 D4、32×256 D2 也 264/264（共 21 个配置全部通过）。

## 不 shuffle（row-major）版本：Triton gemm_afp4wfp4 的布局（2026-11-13，只和 Triton 比）
输入：A [M, K/2]、B [N, K/2] 行主序 fp4x2，scaleA [M, K/32]、scaleB [N, K/32] 行主序 e8m0，不补齐、不 shuffle；N % 8，K % 256。
同一个文件，traits 加第 6 个模板参数 LAYOUT（默认 SHUFFLED = 原行为）。**shuffle 路径的接口和生成代码不变**：row-major 用单独的
body_direct_rm / body_lds_rm（kernel 入口按 LAYOUT 选），shuffle 的 30 个实例与 00bc51f 逐条指令相同（已 diff）。
host：`-l 1` 选 row-major（scale 后面接 256 行 0xFF NaN 守护区，kernel 不能读到）；`-D dir` 导出输入和 C 供外部比较。

做法：
- B_DIRECT：B 操作数和 A 一样按行读（每行 4 个 lane × 16 B）。scale 每个 16 行块读本行的 8 B（本 K tile 的 8 个 group），
  一条 v_perm 把 lane 组 g 的 kk0 / kk1 字节（第 g、4+g 字节）放到字节 0 / 1，MFMA opsel = kk。
  偶数预取深度时（PAIR），每个 wave 的 K tile 改成连续一段（KS 不再交错），两个 K tile 的 scale（每行连续 16 B）用一条 16 B load 取，
  scale 指令减半（单个 row-major scale load 覆盖 16 行 = 16 条 cache line，是主要开销）。
- B_LDS：B 进 LDS 用和 A 相同的 8 行块 / 1088 B 块距 / 16 B XOR swizzle 映像（m7/m8 做法），按 A 的方式读。
  scale 也经 LDS-DMA：每 K tile 一组 4 B/lane 的 DMA（一条 = 32 行 × 8 B，整块共享），每个 stage 末尾放 [row][8 B] 映像，wave 用 ds_read_b64 取本行。
  DMA 条数不能被 4 整除时多出来的 wave 重复拷最后一份（同数据同地址），因此 row-major 还支持 16×64、32×64、16×128、64×64 的 LDS tile。
  坑：重复拷贝的 clamp 让编译器认不出块号 / 描述符是 wave 一致的，每条 DMA 变成 readfirstlane 瀑布循环（≈3 倍慢）→ 块号 readfirstlane，
  scale 描述符用 readfirstlane 过的基址 / 大小现造（不在两个描述符之间 select）。

正确性：prof/check.sh 加 LAYOUT 字段（row-major 的不对齐 N 用 {136, 400, 1032}），264 用例 / 配置；另用 prof/vs_triton.py 在同一份输入上
和 Triton gemm_afp4wfp4 逐元素比较：3 个配置 × 5 个形状（含 300×400×768、128×4104×2304）全部 rel_diff = 0（逐位相同）。

开发计时（HIP 4，graph，单位 µs；Triton 数取自 HIP 6 同形状）：
| 形状 | row-major 最好 | Triton |
|---|---|---|
| 16×4096×4096 | 5.22（lds 16×64 d4） | 4.97 |
| 64×4096×4096 | 5.66（lds 16×64 d4） | 5.30 |
| 128×4096×4096 | 6.30（lds 32×64 d4） | 8.48 |
| 256×4096×4096 | 9.68（lds 32×128 d4） | 12.13 |
正式对比见下面的 HIP 6 全量表（perf_rowmajor.py）。

### 结果（HIP 6，perf_rowmajor.py，Triton config=None，graph 计时；prof/results_rowmajor_HIP6/）
| M | 4096×4096 | 2048×7168 | 3072×6144 | 4096×8192 |
|---|---|---|---|---|
| 1 | 4.93 / 6.07（0.81） | 5.98 / 9.66（0.62） | 5.83 / 7.09（0.82） | 6.40 / 7.85（0.82） |
| 16 | 5.25 / 4.95（1.06） | 6.05 / 7.50（0.81） | 6.03 / 6.72（0.90） | 6.52 / 8.35（0.78） |
| 64 | 5.72 / 5.26（1.09） | 7.06 / 7.22（0.98） | 7.08 / 6.88（1.03） | 8.43 / 12.09（0.70） |
| 128 | 6.31 / 8.45（0.75） | 7.38 / 13.03（0.57） | 7.87 / 11.54（0.68） | 9.60 / 18.49（0.52） |
| 256 | 8.85 / 12.15（0.73） | 8.95 / 19.22（0.47） | 10.40 / 16.99（0.61） | 13.22 / 23.07（0.57） |
| 512 | 12.48 / 18.08（0.69） | 12.45 / 28.12（0.44） | 15.60 / 26.23（0.59） | 20.45 / 43.41（0.47） |
（µs，小 tile 最好 / Triton，括号内比值。M=512 的 4096×4096、3072×6144 不 shuffle 2×2 的 128×128 更快：11.08、14.18。）
M ≤ 128：16 个形状里 13 个赢 Triton；输的 3 个是 16×4096×4096（1.06）、64×4096×4096（1.09）、64×3072×6144（1.03）。
补测这 3 个形状上的 D2 / split 2 / 16×128 / direct KS2 等，都没有超过表中配置。
正确性：row-major 25 个配置（direct 16×32、16×64、32×32、32×64 各 KS/D；lds 16×64、32×64、64×64、16×128、32×128、64×128、128×128、32×256、64×256）× 264 全部通过
（prof/check_results_rowmajor.txt）；与 Triton 逐位相同（vs_triton.py）。

### 实际有用的配置（用于接成 kid）
- row-major（选择表里出现过的）：lds 16×64 D3 / D4、lds 32×64 D4、lds 64×64 D3、lds 64×128 D3 / D4、direct 16×32 KS4 D2；split 1 / 2 / 4。
  M ≥ 512 时可能被不 shuffle 2×2 的 128×128 np 超过（由分发按 tile 数选）。
- shuffled（work_smallm/select_table_HIP4.csv，M ≤ 128 部分）：direct 16×32 KS4 D3 / D4、32×32 KS4 D2、32×64 KS4 D2 / KS2 D3、16×64 KS4 D3；
  M = 128–512 用 lds 32×128 D3 / D4、64×128 D4（这几个只做过冒烟计时，正式的 shuffled 选择表要用 perf_all.py 在空卡上补）。split 均为 1。

## shuffle 版 M = 128–512：瓶颈分析与 A_LDS 模式（smi 3 = HIP 0，2026-11-13）
**trace（256×4096×4096，shuffled B_LDS 32×128 D4，7.86 µs；asm f4gemm 32×128 5.95 µs，单 CU ATT，每 wave cycle）**
| | 总 | main loop（16 个 K tile） | loop 外 |
|---|---|---|---|
| 我们 B_LDS 32×128 D4 | 18.3k | 11.1k（≈690 / K tile） | 7.2k |
| asm 32×128 | 12.9k | 7.3k（≈455 / K tile） | 5.6k |
| 我们 A_LDS 32×128 D4（新） | 17.2k | 9.4k（≈590 / K tile） | 7.7k |
- loop 外：入口 `s_waitcnt lgkmcnt(0)` 等 kernarg 的 s_load ≈1.2–1.4k cycle（kernarg 段在主机内存）；
  `--amdgpu-kernarg-preload-count` 无效（kernel 参数是一个按值传的 struct，不能预载）——要拿到需要把 kernel 签名改成标量参数（接入层的事，记录为需求）。
  其余是 prologue 的冷 load（200–700 cycle / 条）和结尾等 C store（~0.9k）。
- asm 结构（反汇编）：1 block / CU、160 KB LDS、512 VGPR；**A 经 LDS-DMA、B 直接进 VGPR**；每 K tile `vmcnt(12)` + barrier 后，8 条 MFMA 之间
  穿插：下一个 K tile 的 A ds_read（A 提前一个 tile 读）、1 条 A DMA、1 条 scale DMA、4 条 B load、1 条 scale load。每 K tile VMEM 指令数和我们一样（7 条）。
  我们 B_LDS 的 load 集中在 MFMA 之前成串发出，每条在 TA 上卡 100–220 cycle。
- 试过无效：split-K（S = 2/4 让每 CU 2 个 block，反而更慢，归约开销更大）；更浅的环让 2 个 block 共存（只有 M = 512 有利）。

**A_LDS（BMODE 2，新增，shuffled 专用）**：A 经 LDS-DMA 进 D 级 LDS 环（每 stage 一个 __shared__ 对象），B 和 scale 直接进 D 级寄存器环；
asm 式流水：iteration kt 等 group kt+1 落地 + barrier，读 A(kt+1)（提前一个 tile），MFMA(kt) 与 kt+D 的拷贝一条 MFMA 一条 VMEM 穿插。
实例：32×64 D4/6/8/10、32×128 D3/4/6/8、64×128 D3/4/6、32×256 D3/4（受 (D−1)·G ≤ 63 限制）。

M = 128–512 同卡对比（µs，prof/mid_m_shuffled_vs_asm_smi3.txt）：
| M×N×K | A_LDS 最好 | B_LDS 最好 | asm |
|---|---|---|---|
| 128×4096×4096 | 5.88（32×64 D6） | 7.66 | 5.20 |
| 256×4096×4096 | 6.98（32×128 D4） | 7.86 | 5.95 |
| 512×4096×4096 | 9.83（64×128 D4） | 11.17 | 7.48 |
| 128×2048×7168 | 7.26（32×64 D6） | 10.36 | 6.85 |
| 256×2048×7168 | 7.85（32×64 D6） | 10.67 | 7.21 |
| 512×2048×7168 | 9.64（32×128 D4） | 11.48 | 15.61（asm default kernel） |
| 128×3072×6144 | 6.65（32×64 D6） | 9.46 | 6.46 |
| 256×3072×6144 | 8.40（32×128 D3） | 9.98 | 6.88 |
| 512×3072×6144 | 11.93（64×128 D4） | 13.93 | 8.56 |
| 128×4096×8192 | 8.53（32×64 D6） | 11.68 | 7.72 |
| 256×4096×8192 | 10.46（32×128 D4） | 12.36 | 8.86 |
| 512×4096×8192 | 15.14（64×128 D4） | 17.67 | 13.52 |
A_LDS 比 B_LDS 快 12–30%，和 asm 的差距从 1.3–1.55× 缩到 M = 128：1.03–1.13×、M = 256：1.09–1.22×、M = 512：1.12–1.39×（512×2048×7168 赢 asm 的 default kernel）。
剩余差距：loop 内 barrier 等待（每 tile 70–150 cycle，asm ~67）和 loop 外（kernarg、冷启动、C store）。

**不 shuffle 的 3 个形状**：把同样的 asm 式流水加到 row-major B_LDS（宏 SMALL_RM_PIPE2）：16×64 略慢（5.25→5.40）、32×64 略快（5.77→5.53），
3 个形状都没超过 Triton → 默认关闭（已接入的 row-major kid 代码不变）。

**一致性**：已接入的 30400–30408 / 30500–30520 对应的 71 个实例与 9c7522d 逐条指令相同（新增的只有 13 个 A_LDS 实例）。

**接入层需求（只记录，未改）**
1. 只读查询入口，例如 `opus_gemm_a4w4_select_kid(M, N, K, batch, noshuffle) -> int`（返回调优表命中或启发式选的 kid），供 op 测试断言 M ≤ 64 选到小 tile kid。
2. 启发式在 M ≤ 64 时要求小 tile 的 B_M ≥ M：M = 64 的 shuffle 侧只能选到 64×128 B_LDS，但 16×32 / 32×32 / 16×64 direct 更快
   （B_M < M 的 tile 合法，M 方向多几个 block 即可）。
3. M = 128–512 的 shuffle 侧可加 A_LDS kid：32×64 D6、32×128 D3/D4、64×128 D4（BMODE 2）。
4. kernarg 预载需要标量参数形式的 kernel 签名（每次省 ~0.5 µs，对 5–10 µs 的小形状占比显著）。
