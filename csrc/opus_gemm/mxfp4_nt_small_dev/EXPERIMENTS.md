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
（128×128 D4、32×256 D2 的补充检查在本次提交时仍在运行。）
