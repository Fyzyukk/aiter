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

**不 shuffle 的 3 个形状**：把同样的 asm 式流水加到 row-major B_LDS（宏 SMALL_RM_PIPE2，代码已在整理时删除，见 52a6ca608 及之前的提交）：16×64 略慢（5.25→5.40）、32×64 略快（5.77→5.53），
3 个形状都没超过 Triton → 默认关闭（已接入的 row-major kid 代码不变）。

**一致性**：已接入的 30400–30408 / 30500–30520 对应的 71 个实例与 9c7522d 逐条指令相同（新增的只有 13 个 A_LDS 实例）。

**接入层需求（只记录，未改）**
1. 只读查询入口，例如 `opus_gemm_a4w4_select_kid(M, N, K, batch, noshuffle) -> int`（返回调优表命中或启发式选的 kid），供 op 测试断言 M ≤ 64 选到小 tile kid。
2. 启发式在 M ≤ 64 时要求小 tile 的 B_M ≥ M：M = 64 的 shuffle 侧只能选到 64×128 B_LDS，但 16×32 / 32×32 / 16×64 direct 更快
   （B_M < M 的 tile 合法，M 方向多几个 block 即可）。
3. M = 128–512 的 shuffle 侧可加 A_LDS kid：32×64 D6、32×128 D3/D4、64×128 D4（BMODE 2）。
4. kernarg 预载需要标量参数形式的 kernel 签名（每次省 ~0.5 µs，对 5–10 µs 的小形状占比显著）。

## row-major 小 tile 改写成 opus 风格（2026-11-14，kid 30500–30520，名字不变）

照 shuffle 小 tile（b41c923f6）的做法重写 `opus_gemm_pipeline_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh` 和 traits：
- traits：aiter 命名（B_M/B_N/B_K、T_M = 1、T_N = 4/KS、T_K = KS、W_*、E_M/E_N/E_K、VEC_*），LDS stage / DMA 条数 / 每步 VMEM
  条数等全部 static constexpr + static_assert；kargs 去掉 ws / cnt，改成 split_k + ws_handle（与 shuffle 同结构，各自定义）。
- pipeline：namespace `opus_mxfp4_small_noshuffle`，`direct::`（B_DIRECT：make_layout_ga/gb/gsfa/gsfb）和 `lds::`
  （B_LDS：make_layout_ga/gb（LDS-DMA）、sa/sb、gs_chunk、gsf（4 字节 scale DMA）、ra/rb、rsfa/rsfb），共用 make_layout_gc；
  opus make_gmem / load / store / _async_load、opus mfma_f32_16x16x128_fp4_fp4（scale 重载，opsel = kk）、opus::static_for、
  sync_threads。算法不变（D 深寄存器环 / LDS 环、counted vmcnt、零长度 buffer 越过 K 末尾、M/N 尾部在 kernel 内）。
- 保留 raw：readfirstlane、sched_barrier、v_perm（scale 打包）、fence、`asm volatile("s_waitcnt vmcnt(..)" ::: "memory")`、
  16B→32B 的 __builtin_shufflevector；每个 ring stage 一个 `__shared__` 对象 + 由它派生的 address_space(3) 指针（同 shuffle 的原因）。
- 删掉：B_LDS 2×2 的 128×128（没有 kid 用）；B_LDS 里为“codegen 不变”保留的无用 per-wave scale 描述符；in-kernel split-K（ws/cnt 原子）。
- split-K：S > 1 写 fp32 partial [S][batch][Mp][Np]，launcher 再跑 splitk_reduce_kernel（row-major N % 8，需要 HAS_OOB = true 的尾部）。
- 寄存器：B_LDS 的 VGPR 大幅下降（64×128 D3 198 → 91，16×64 D4 54 → 37），旧版 64×128 D4 / 32×128 D4 的 SGPR spill（33 / 26）没有了。
- harness 修正：比较函数在 -ffast-math 下检测不到 NaN（`!(d <= tol)` 被折叠），改成按位判断 Inf/NaN。此前的 check 结果对“C 没写到（0xFF 毒值）”
  不敏感。

性能（HIP 6，harness hipGraph 中位数，新旧交替、3 轮取最好；7 个 config × S 1/2/4 × M {1, 8, 16, 64, 128, 256, 512} × 4 组 (N, K)；
/tmp/small_ns_tools/full/raw.csv）见下表。“负责的形状” = 旧版该 kid 在 21 个 kid 里最快或在最快的 3% 以内，加上 select_kid 的默认（30518：M ≤ 8，30503：8 < M ≤ 64）。

| kid | config | S | 全部 28 形状 新/旧 几何平均 | 负责形状 新/旧 最差 |
|---|---|---|---|---|
| 30500 | lds 16×64 D3 | 1 | 0.965 | 0.969（64×3072×6144） |
| 30503 | lds 16×64 D4 | 1 | 0.978 | 0.995（16×4096×8192） |
| 30506 | lds 32×64 D4 | 1 | 0.954 | 0.950（128×4096×4096） |
| 30509 | lds 64×64 D3 | 1 | 0.972 | 0.996（512×4096×4096） |
| 30512 | lds 64×128 D3 | 1 | 0.994 | 1.000（512×4096×8192） |
| 30515 | lds 64×128 D4 | 1 | 0.993 | 0.994（512×4096×8192） |
| 30518 | direct 16×32 KS4 D2 | 1 | 0.990 | 0.996（1×2048×7168） |
| 30501 / 30504 / 30507 / 30510 / 30513 / 30516 / 30519 | 同上 | 2 | 1.222 / 1.204 / 1.195 / 1.166 / 1.020 / 0.983 / 1.137 | 30501 1.318（64×2048×7168），30507 1.273（64×4096×8192） |
| 30502 / 30505 / 30508 / 30511 / 30514 / 30517 / 30520 | 同上 | 4 | 1.246 / 1.236 / 1.198 / 1.147 / 0.965 / 0.930 / 1.177 | 30502 1.621（8×3072×6144：6.00 → 9.73 µs），30505 1.381（16×4096×8192） |

S = 1：全部不慢（几何平均 0.95–0.99，负责形状最差 1.000）。S > 1：splitk_reduce 方案每次多 ≈ 2.5–3.7 µs（第二个 kernel +
workspace 往返 + host-coherent ws_handle 读），旧版 in-kernel 归约选中的 10 个形状（M ≤ 16 且 K ≥ 6144，加 64×4096×8192）上新版没有
S > 1 kid 能赢 S = 1，按“每个形状最快的 kid”算比旧版慢 3–23%（16×4096×8192：6.54 → 8.02 µs，1×4096×8192：6.35 → 7.52 µs）。
**未达到 2% 门槛，未提交**（S > 1 的去留待定）。

## row-major 小 tile 合并成一条与 shuffle 小 tile 相同的路径（2026-11-14，取代上一节的 direct / lds 两条路径）

结构与 opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh 完全一致（1×4 wave 共享 A；A 用 LDS-DMA 进 STAGES 深的 LDS 环，每 stage 一个
`__shared__`，XOR swizzle，A 提前一步 ds_read；B 和 scale 进 STAGES 深的寄存器环；KS = 1/2/4 在 block 内沿 K 切 wave，LDS 归约一次；
counted vmcnt（asm + memory clobber，单位 VMEM_PER_STEP）、每步一个 barrier、sched_group<MASK, CNT>() 交错 MFMA / VMEM；K 末尾之后走零长度
buffer），函数 / 命名相同（make_layout_ga/sa/gas/ra/gb/gsfa/gsfb/gc、issue / read_a、mma_scale_one）。只有两处不同：
- B：row-major，lane 读第 16 e + lane % 16 行的 [64 kk + 16 (lane / 16), +16) 字节（每条指令 16 行 × 64 B）；
- scale：每 16 行块一条 b64（lane 读第 lane % 16 行这个 K tile 的 8 字节），一条 v_perm 把 K 组 lane / 16 和 4 + lane / 16 放到字节 0 / 1，
  MFMA opsel = kk。这是最便宜的寄存器方案：opsel 必须是编译期常数而需要的字节随 lane 变，所以至少要一条 VALU；配对两个 K tile 的 16 字节加载
  会让每步 VMEM 条数不再是常数（counted vmcnt 不再简单），没有采用。
删除：BMODE 枚举 / 模板参数、direct:: / lds:: 两个 body、SF_PAIR、2 级 scale DMA 等；traits 变成 <BM, BN, D, KS>（与 shuffle 相同，另允许 B_N = 16）。

选 config（prof/sweep_rowmajor.py → prof/sweep_rowmajor_HIP6.csv，HIP 6）：41 个 (BM, BN, KS, D) × S 1/2/4（S > 1 只测 M ≤ 64）×
M {1, 8, 16, 64, 128, 256, 512, 1024} × 4 组 (N, K)。S = 2 / 4 从来没赢过 S = 1 → 只建 S = 1 kid（split-K 代码保留）。贪心覆盖（每个形状
落在最好的 2–3% 内，M ≤ 512）选出 7 个：30500 16×16 KS4 D6、30501 16×16 KS4 D8、30502 16×32 KS2 D4、30503 32×32 KS4 D4、30504 32×64 KS2 D4、
30505 64×64 D4、30506 64×128 D4（M = 1024 只有 64×256 D3 更好，那里是 2×2 的范围，没收）。select_kid：M ≤ 16 → K ≥ 7168 ? 30501 : 30500，
M ≤ 64 → 30503。

**结论：M ≤ 16 好，M ≥ 64 明显变慢。** 与旧版（HEAD，21 个 B_LDS / B_DIRECT kid 的最好）相比：M 1–16 新 / 旧 0.87–1.17（多数更快；
K = 8192 时旧的 in-kernel split-K 更快），M = 64 1.02–1.42，M = 128–512 1.18–2.26。原因是 row-major 的逐 lane 加载受地址处理限制：
实验（M = 64 × 4096 × 4096，16×64 KS1 D4，同一份 kernel 只改地址，结果不对只测时间）：9.97 µs；scale 改成连续地址 7.49；B 改成连续地址 7.55；
两者都改 5.17（旧 B_LDS 5.66）。B 每条指令 16 行 × 64 B（LDS-DMA 是 8 行 × 128 B，行数减半），scale 每条指令 16 行各 8 字节，而且 A 的 scale
在 4 个 wave 里重复读。要在 M ≥ 64 拿回旧版的速度，需要像旧 B_LDS 那样让 B（和 scale）也走 LDS-DMA 环 —— 这超出了“B 直接进寄存器”的设计，留给决定。

与 Triton（gemm_afp4wfp4，config=None）和 no-shuffle 2×2 kid 的对比（prof/perf_ns_vs_triton.py，通过 aiter，全部先对参考值检查，
CUDA graph 100 次中位数，3 轮取最好，HIP 6）：见 /tmp/small_ns_tools/final_table.md（下表节选，µs）。

| M×N×K | 2×2 最好 | 新 small 最好 | Triton | 旧 small 最好 |
|---|---|---|---|---|
| 1×4096×4096 | 10.00 | 5.00 | 6.08 | 4.91 |
| 16×2048×7168 | 15.16 | 5.85 | 7.48 | 6.09 |
| 16×4096×8192 | 16.91 | 7.66 | 8.35 | 6.54 |
| 64×4096×4096 | 10.08 | 7.72 | 5.29 | 5.66 |
| 128×4096×4096 | 10.29 | 10.84 | 8.42 | 6.26 |
| 256×4096×8192 | 17.54 | 29.60 | 23.08 | 13.11 |
| 512×3072×6144 | 13.91 | 24.26 | 26.12 | 15.55 |
| 1024×4096×4096 | 13.46 | 43.31 | 42.16 | |
| 8192×4096×8192 | 119.10 | 613.55 | 181.47 | |

另外：shuffle 小 tile 的 7 个 kid（30410–30416）用修好的比较函数（NaN / Inf 按位判断）重跑 check.sh：7 × 264 全部通过。

## row-major 小 tile：B 和 scale 也走 LDS 环（2026-11-14，取代上一节的“B 直接进寄存器”版本）

上一节的单一路径在 M ≥ 64 受地址处理限制，按决定改成：A、B、scale 全部用 LDS-DMA 进 STAGES 深的 LDS 环（每 stage 一个 `__shared__`），
每个 wave 提前一步 ds_read 自己的操作数，其余与 shuffle 小 tile 相同（1×4 wave、KS 沿 K 切 wave、counted vmcnt、sched_group 交错、
零长度 buffer、KS 归约、splitk_reduce 收尾）。
- B：与 A 相同的 8 行 × 128 B 块（XOR swizzle，1088 B 步长），整个 block 的 B_N 行由 4 个 wave 轮流 DMA；每个 wave ds_read 自己的 E_N 个 16 列块；
- scale：每个 K tile 一张 [row][8 B] 图（A 行 max(B_M, 32)、B 行 max(B_N, 32)），block 级 4 字节 LDS-DMA（一条 = 32 行 × 8 B）；读回时每 16 行块
  一条 ds_read_b64 + 一条 v_perm（opsel = kk）；
- stage 布局 [KS 个 A tile][KS 个 B tile][KS 个 A scale tile][KS 个 B scale tile]；每个 wave 的 DMA 块号（K tile、LDS 偏移、soffset、scale 的源 /
  范围 / lane 偏移）在循环外算好（SGPR），循环内只加 K tile；
- traits <BM, BN, D, KS>（B_M 16/32/64，B_N 16–256），KS 2/4 保留（选中的 kid 用到 KS 2）。

选 config（prof/sweep_rowmajor.py，44 个 (BM, BN, KS, D) × S 1/2/4（S > 1 只测 M ≤ 64），M 1–1024 × 4 组 (N, K)，HIP 6，
prof/sweep_rowmajor_HIP6.csv）：S = 2 / 4 一次也没赢；5 个 config 让 M ≤ 512 的每个形状都在最好的 3% 以内：30500 16×32 KS2 D4（M ≤ 16，
以及 64×2048×7168）、30501 32×32 KS2 D4（M ≈ 64）、30502 32×64 D4、30503 64×64 D4、30504 64×128 D4（M 128–512）。select_kid：M ≤ 16 → 30500，
M ≤ 64 → 30501。

结果（prof/perf_ns_vs_triton.py，通过 aiter，全部先对参考值检查，CUDA graph 100 次中位数，3 轮取最好，HIP 6；旧 small = HEAD 的 21 个 kid
在 harness 里的最好）：新 small / 旧 small：M 1–16 0.72–0.88，M 64 0.71–0.84，M 128–512 0.76–0.96，M ≥ 1024 0.91–1.00；新 small 比上一节
的“B 直接进寄存器”版本快 15–64%。与 Triton：M ≤ 2048 全部更快（M ≤ 512：0.26–0.89×）；M ≥ 1024 的形状 2×2 kid 最快（最好 / Triton 0.26–0.70）。

| M×N×K | 2×2 最好 | 新 small 最好 | Triton | 旧 small 最好 |
|---|---|---|---|---|
| 1×4096×4096 | 10.00 | 3.67 | 6.13 | 4.91 |
| 16×4096×8192 | 16.92 | 5.75 | 10.15 | 6.54 |
| 64×4096×4096 | 10.08 | 4.68 | 5.27 | 5.66 |
| 128×4096×4096 | 10.31 | 5.74 | 8.41 | 6.26 |
| 256×4096×8192 | 17.53 | 10.55 | 23.10 | 13.11 |
| 512×3072×6144 | 14.00 | 13.72 | 31.98 | 15.55 |
| 1024×4096×4096 | 13.47 | 20.85 | 42.17 | 22.94 |
| 8192×4096×8192 | 119.19 | 287.39 | 181.55 | 297.70 |
