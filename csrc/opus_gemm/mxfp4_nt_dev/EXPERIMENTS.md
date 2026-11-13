# work_tiles —— 2×2 流水线多 tile 参数化 + M/N 边界处理（线 1，2026-11-13）

基础：work_m10 当前最好（SF_DMA SF_DMA2 SF_SHIFT PF_SPREAD ZC_PEEL）。先用 /tmp/tiles_tools/unifdef.py 把这些宏固化、删掉其他实验分支
（固化后的模板与 /tmp/exp_m10/zc 生成的代码逐字节相同，已核对）。构建 `./build_var.sh name ""` → /tmp/exp_tiles/name；
一个可执行文件里有 4 个 tile 的 kernel，`-t 256x256|256x128|128x256|128x128` 选择。正确性脚本 `/tmp/exp_tiles/check.sh exe gpu tile`。

## 改掉的写死之处
| 位置 | 原来 | 现在 |
|---|---|---|
| traits | `struct ..._traits`，B_M = B_N = 256，断言 256 线程一线程一行、LDS == 160 KB | `traits_t<BM, BN>`（256/128），新增 SF_WAVES_A/B、C_COPIES、C_SWZ；LDS ≤ 160 KB |
| make_layout_ga / sa / gb / sb | 每 wave 每半 tile 4 条 DMA（`ih` 维 2、`issue` 维 4、断言 "4 issues per half"） | `a/b_buffer_load_insts`（128 行时 2），`ih` 维 = insts/2 |
| scale panel DMA | 4 个 wave 各拷 2 个 32 行块（只对 256 行成立）；sfa/sfb 共用读偏移 | `wave % SF_WAVES`：块少时多余的 wave 重复拷同一份（同数据写同地址），每个 wave 的指令数不变；sfb 单独的读偏移（panel 大小可不同） |
| scale 读取 / SF_SHIFT | 每半 4 个 dword，2 次 64 位移位 | 每半 E_M / E_N 个 dword，E/2 次移位 |
| 调度模板 | 按每 quadrant 32 条 MFMA 写死 | 256×256 保留原手工模板；其他 tile 用 `sched_spread<Q, DS, VMEM, VALU, STORE>` 按指令数均匀铺开 |
| 手写 vmcnt | B_b `2*a`；prologue `4*a + 16` / `4*a` | B_b `a + b`；prologue `2(a + b) + C_COPIES` / `2(a + b)`（只计比必须完成的那条更新的 VMEM） |
| C 暂存 swizzle | 按 528 B 行距写死 | 仅 B_N = 256 时启用（`C_SWZ`），其他 tile 不 swizzle |
| 旧的 scale layout（gsfa / ssfa / rsfa…） | 构造了但不用，带 `B_M == 256` 断言 | 删除 |

## 边界处理（不补 pad）
- A / B / C 的 buffer range = 本 tile 在矩阵内的行数 × 行距（下一 tile 的预取同理）：越界行读回 0、写被丢弃。
- C 列：一个线程在所有 copy 里的列相同，按 `col + c < N` 每线程算一次"越界加量"（2^30 个元素 → 字节偏移 2^31，超出 range），加到偏移上；
  推迟写的下半 C 记下自己 tile 的 rows / cols。（先试过每条 store 用 `c < N ? off : OOB` 选择：256×256 出现 103 个 VGPR spill，改成每线程一个加量后为 0。）
- K 放宽到 256 的倍数（最后一个 scale panel 只用一半）；N 为 16 的倍数（shuffle_weight）。
- host 的 scale 按 aiter e8m0_shuffle 把行补到 256，补齐值默认 0xFF（e8m0 的 NaN），确认补齐值只影响不写出的行列；`MXFP4_SF_PAD` 可改。

## 正确性（HIP 2）
矩阵：原 6 形状 + M ∈ {1,16,100,300,1000} × N ∈ {128,400,1280,4112}（K 轮换 256/768/7168，batch 轮换 1/2），各配 默认 / MXFP4_SFA_K_PATTERN=1 / PERSIST_GRID=37 / PERSIST_GRID=1，共 104 个。
| tile | 结果 | VGPR / AGPR | VGPR spill | SGPR spill | LDS |
|---|---|---|---|---|---|
| 256×256 | 104/104 | 512 / 256 | 0 | 21（zc 16） | 160 KB |
| 256×128 | 104/104 | 448 / 224 | 0 | 23 | 120 KB |
| 128×256 | 104/104 | 448 / 224 | 0 | 20 | 120 KB |
| 128×128 | 104/104 | 416 / 208 | 0 | 18 | 80 KB |

## 与 zc（m10 最好）的代码差异（256×256）
指令 3210 → 3286（+76）：v_add_u32 +45（C 列越界加量、按真实 M/N 的 buffer range）、s_mul_i32 +7、v_writelane/readlane 各 +5（SGPR spill 16 → 21，进 VGPR lane）、
v_cndmask +3、v_mad_u64 +3。C 写回 / store_pend_c 的指令序列保持原样（只多一个每线程的越界加量），没有做"地址简化"。

## 计时
prof/timing.sh（等 HIP 2 空闲后测：8192³ zc vs 256×256 交替 6 轮；1024×4096×8192、2048×4096×8192 上 4 个 tile + aiter asm）。
截至提交时 HIP 2 一直被他人进程占用（use 82–100%），**尚无有效计时**。卡空闲后运行 `prof/timing.sh`。

## 接入 aiter 第 1 步：traits / kargs 改成 aiter 形式（2026-11-13）
- traits：`opus_gemm_a4w4_mxfp4_nt_traits_gfx950<BLOCK_SIZE, seq<B_M,B_N,B_K>, tuple<fp4,fp4,bf16,fp32,e8m0>, seq<16,16,4>, seq<1,1,32>>`
  （断言 BLOCK_SIZE 256、B_K 256、VEC、GROUP 只支持当前值），简写 `opus_mxfp4_nt_traits<B_M, B_N>`；内部推导不变。
- kargs `opus_gemm_mxfp4_scale_kargs_gfx950`；kernel `gemm_a4w4_mxfp4_nt_kernel`，体内 `#if defined(__gfx950__)`；`ceil_div_scale` → `opus_mxfp4_ceil_div`（避免与其他头冲突）。
- traits 只 include opus（hip_minimal 用于 device / RTC），不 include opus_gemm_utils.cuh：后者在非 RTC host pass 拉入完整 HIP runtime，pin 工具链编不过；launcher 只需要 kargs。
- 生成的代码与改名前（/tmp/exp_tiles/t2）逐条指令相同（9169 行反汇编一致，只有符号名不同）；4 个 tile × {8192³, 300×400×768×b2, 1000×4112×256} × SFA_K_PATTERN 全部 errors=0。

## 接入 aiter 第 2 步：gfx950 预编译 .co（2026-11-13）
- pipeline 拆成 `__device__ gemm_a4w4_mxfp4_nt_body<Traits>(kargs)` + 模板 `__global__` 包装（dev 构建生成代码不变，已核对）。
- 新增 `gen_co/`（只带 gfx950 / mxfp4 需要的部分，未移植 blockscale 分支的 gfx1250 / mxfp8 条目和其 codegen 依赖）：
  `co_kernels_gfx950.json`（4 个 tile）、`build_co_gfx950.py`（stub → pin clang `--cuda-device-only` → unbundle → 校验符号 → build_info.json）、README。
- 4 个 .co 已提交（pin 工具链 yuyzhang512 amdgpu-pin-op-dst，opus 用 blockscale 分支 csrc/include，flags `-O3 -std=c++17 -ffast-math`）：

| image | VGPR / AGPR | VGPR spill | SGPR spill | LDS | scratch |
|---|---|---|---|---|---|
| ..._co_256x256x256 | 512 / 256 | 0 | 21 | 163840 | 0 |
| ..._co_256x128x256 | 448 / 224 | 0 | 23 | 122880 | 0 |
| ..._co_128x256x256 | 448 / 224 | 0 | 20 | 122880 | 0 |
| ..._co_128x128x256 | 416 / 208 | 0 | 18 | 81920 | 0 |

- 与 dev 构建（模板 __global__）相比：寄存器 / spill / LDS 相同；256×256 的指令差异只有尾部 s_nop 填充、45 处 v_and → v_mul_i32_i24（地址计算）和少量 SALU 重排（extern "C" 入口 + 不同 TU 的寄存器分配差异）。
  先试过 blockscale 的 gfx950 默认 flags（`-fgpu-flush-denormals-to-zero -fno-offload-uniform-block ...`），代码差异更大，改回与 dev 相同的 `-ffast-math`。
- 验证：dev host 新增 `MXFP4_CO_DIR=<gen_co/gfx950>`，用 hipModuleLoad 加载 .co 并复用原校验；4 个 tile × {8192³, 300×400×768×b2, 1000×4112×256} × SFA_K_PATTERN 全部 errors=0。
- 计时仍等卡（HIP 2 被占）。

## 接入 aiter 第 3–5 步：codegen / 分发 / Python / 测试（2026-11-13）
- opus_gemm_common.py：`a4w4_mxfp4_nt_kernels_list`，kid 30000–30003（256×256、256×128、128×256、128×128），
  `OpusGemmInstance(256, B_M, B_N, 256, 2, 2, 16, 16, 128, 16, 16, 4, 1, 1, 32, "a4w4_mxfp4_nt", ["bf16_t"], arch_prefix="gfx950", name_tag="a4w4_mxfp4_nt_co")`。
- gen_instances_gfx950.py：4 张表加 tag；`gen_a4w4_mxfp4_nt_instance` 只生成 host launcher（fp4 字节 stride、e8m0_shuffle scale stride = 32 × cols、
  persistent grid = min(tile 数, CU 数) 写入 grid_blocks、bf16 输出、`opus_gfx950_co::mxfp4_co_kernel(symbol)` 运行时加载 .co），只记 host 实例化、无 device TU。
- gen_instances.py：gfx950 构建总把 mxfp4 kid 加进编译集；生成 `opus_gemm_a4w4_mxfp4_lookup.h`（kid 表 + 可选 `--a4w4_mxfp4_tune_file` 的 (M,N,K)→kid 表）。
  离线跑 `GPU_ARCHS=gfx950 gen_instances.py` 已验证：4 个 impl、host TU 里 4 个实例化、manifest 4 个声明。
- include/gfx950/opus_co_launch_mxfp4_gfx950.cuh：读 `OPUS_GEN_CO_DIR`（环境变量优先，JIT 加 `-DOPUS_GEN_CO_DIR=<csrc>/opus_gemm/gen_co`）下 gfx950/<symbol>.co，
  AiterAsmKernelFast 注册，按 sizeof(kargs) 传参（kargs 104 B，不是 a8w8 co 的 96 B）。
- opus_gemm.cu：`opus_gemm()` 新增 fp4x2 分支 → `opus_dispatch_a4w4_mxfp4_gfx950`（调优表 → 启发式：min 轮数 × tile 面积 / 效率，效率暂定 1.0 / 0.9 / 0.78，待计时校准）；
  新入口 `opus_gemm_a4w4_tune(XQ, WQ, Y, x_scale, w_scale, kernelId)`；opus_gemm.h / rocm_ops.hpp / pybind 注册。
- Python：`aiter/ops/opus/gemm_op_a4w4.py`（`gemm_a4w4_opus`、`opus_gemm_a4w4_tune`），`aiter.ops.opus` 导出。
- 测试：`op_tests/test_opus_a4w4_gemm.py`（torch 反量化参考 + aiter gemm_a4w4 asm 对比；`--perf` 计时）。

## 接入 aiter 第 6 步：JIT 实际构建 + op 测试（2026-11-13，HIP 2）
- 用本 worktree 的 aiter（`PYTHONPATH=<worktree> GPU_ARCHS=gfx950`）JIT 构建 module_deepgemm_opus 成功（标准 ROCm hipcc；.co 不参与 JIT 编译，运行时从 `csrc/opus_gemm/gen_co/gfx950` 加载）。
  环境注意：本分支 aiter 要求 triton ≥ 3.6，本机是 3.4，需 `AITER_USE_SYSTEM_TRITON=1` 才能 import（与本改动无关）。
- `op_tests/test_opus_a4w4_gemm.py` 全部通过：7 个 shape（256×256×512、1024×2048×1024×b2、300×400×768×b2、1×128×256、100×1280×7168、1000×4112×256、16×4096×4096）
  × {dispatch, kid 30000–30003}，相对 torch 反量化参考 rel_err 1.2e-3–1.7e-3（bf16 输出舍入量级）；aiter gemm_a4w4 asm 在 256×256×512 上同样 1.67e-3。
- 性能（`--perf`）尚未测：HIP 2 一直被他人占用（use 100%）。

## 输入检查（2026-11-13）
- opus_gemm.cu `opus_a4w4_mxfp4_check`（opus_gemm() 的 fp4x2 分支和 opus_gemm_a4w4_tune() 都调用）新增 AITER_CHECK：
  WQ / Y 形状与 XQ 一致；N % 16 == 0；K % 256 == 0；x_scale / w_scale 为 1 字节、2D [batch*rows, cols] 或 3D [batch, rows, cols]，
  列数 == ceil(K/32 / 8) * 8，每 batch 行数 ≥ ceil(rows/32) * 32 且为 32 的倍数（e8m0_shuffle 布局）。
- gemm_op_a4w4.py 的 `gemm_a4w4_opus` / `opus_gemm_a4w4_tune` 先在 Python 侧做同样的检查（ValueError）。
- op 测试加 2 个应报错用例（N=100、K=320），分别验证 Python 检查和绕过 Python 直接调 C++ 时的 AITER_CHECK；
  强制重建 module_deepgemm_opus 后：4 次都被拒绝（C++ 报在 opus_gemm.cu:152 / 153），原有 36 个正确性用例仍全部 OK。
- 计时脚本已停掉（按要求，性能 / 调优暂不接入）。

## 192 tile：192×256、256×192（E = 3，kid 30004 / 30005，2026-11-13）
- traits：B_M / B_N 允许 192；DMA 条数不再要求偶数，新增 `A_ODD` / `B_ODD`。
- A/B LDS-DMA：每 wave 每半区 3 条，chunk 对（16 行块的两半 / B 的 kk 两半）会跨两个 wave，所以 E = 3 时改用 `dma_odd_a/b`：
  wave w 拷 chunk w×3 + i，chunk → 行的对应关系与读 layout 相同（A：行 (c/2)·16 + 2·(lane/8) + c%2，B：16 行块 c/2、K 半 c%2），
  LDS 偏移用 MUBUF 立即数（i × 1088），奇偶由 wave 号在运行时决定（wave 内一致）。偶数条的 tile 仍走原 `async_load_imm`。
- scale：panel 6 个 32 行块 → wave 0–2 各拷 2 块、wave 3 重复 wave 0（原 wave % SF_WAVES 机制）；SF_SHIFT 每半区 3 个 dword，最后一个用 32 位移位。
- C：B_N = 192 时一行 24 个 16 B，256 线程不能整行覆盖，改用"copy i → 行 (i/3)·32 + t/8、列 (i%3)·64 + (t%8)·8"（12 份），每份单独算列越界；
  不做 C 暂存 swizzle（行距 400 B）。bank 冲突没测成：rocprofv3 --pmc 对 dev 可执行文件崩溃（core dump），未深究。
- 调度用 sched_spread（每 quadrant 24 条 MFMA），SV 改为 ceil(E/2)（E 为偶数时不变），vmcnt 公式自动适配。
- 资源：两个新 tile 都是 VGPR 480 / AGPR 240，VGPR spill 0，SGPR spill 23 / 25，LDS 140 KB，scratch 0。
- 原 4 个 tile：dev 构建与改动前逐条指令相同；.co 代码段与改动前相同。
- `build_co_gfx950.py` 加 `-cuid=<name>`：之前每次重编 .co 只因 `__hip_cuid_<随机>` 符号不同而变（代码相同），现在可复现（连编两次 md5 相同）。4 个旧 .co 因此重新提交一次（只有该符号不同）。
- 正确性：dev 104 用例矩阵 192×256、256×192 各 104/104；新 .co 经 MXFP4_CO_DIR 加载 3 个 shape × SFA_K_PATTERN 全部 errors=0；
  aiter op 测试（重建 module_deepgemm_opus）50 个正确性用例全 OK（含 kid 30004 / 30005 共 14 个），4 个报错用例都被拒绝。
- 启发式效率表加 192×256 / 256×192 一档（0.95，暂定）。性能未测（按要求等小 shape 流水线后统一测）。

## 非 persistent / split-K / 无边界 变体（2026-11-13，HIP 6）
traits 增加 `MODE_`（0 persistent，1 每 block 一个 tile，2 split-K）和 `BOUNDARY_`（false：M、N 整除时去掉边界处理）；kargs 末尾加 `split_k`、`ptr_ws`（104 → 120 B）。
- MODE 1：grid = tile 数，tile 循环只跑一次；编译期去掉下一 tile 预取（A/B/scale 的 DMA）、prologue 的下半 C 补写和循环后的 pend；下半 C 在 epilogue 直接写出。
- MODE 2：block = (tile, K 段)，s = block % S（同一 tile 的段相邻）。各段 K = K/S（256 的倍数；奇数个 K tile 时最后一个 scale panel 只用一半，已覆盖）。
  A / B / scale 的基址按段起点平移（A：字节，B：shuffle_weight 下每 K tile 2 KB，scale：e8m0_shuffle 下每 K tile 256 B）。
  归约：各段把 fp32 部分和按 C 片段布局（每 lane 16 B）写到 workspace，agent 范围 release fence + 原子计数；最后到达的 block（计数 = S−1）
  把计数清零、acquire fence、读其余 S−1 份加到自己的累加器，再走正常 bf16 epilogue（含边界）。单次 launch，无需归约 kernel。
  workspace = 固定 4 KB 计数区（≤ 1024 tile，只在分配时清零一次）+ tiles × S × B_M × B_N fp32。256×256 不做 split-K（VGPR 已满，spill 86）。
- 生成代码：6 个 persistent tile 与改动前逐条指令相同；MODE 1/2 均无 VGPR spill（split-K SGPR spill 38–45，进 VGPR lane）。
- .co：18 个（persistent 6、np 6、sk 5、nob 1），kid 30000–30005 / 30010–30015 / 30021–30025 / 30030。
- 接入：kid 表按 variant 生成；launcher 按 variant 决定 grid（persistent: min(tiles, CU)；np: tiles；sk: tiles × S，S = `opus_mxfp4_pick_split`：
  {8,4,2} 中满足 tiles×S ≤ CU 且 K % (S×256) == 0 的最大值，`OPUS_MXFP4_SPLITK` 可覆盖）；workspace 每 stream 一份、按需增长（不可在 graph capture 中首次分配）。
  分发启发式改成按 (tile, variant) 估时（rounds × 面积 × K_seg × 速率 / 效率 + split-K 固定开销 4 µs），常数待计时校准。
- 正确性：aiter op 测试 169 个用例全 OK（含 split-K S ∈ {1,2,4,8} 75 个、np、nob）；dev 104 矩阵：mode 1 的 256×256 / 256×128 / 128×256 已 104/104，其余在跑。

### split-K 修正与开销（HIP 6）
- 第一版用 agent 范围 release / acquire fence：gfx950 上它们会 write back / invalidate 整个 L2（含其他 block 的 A/B），split-K 慢到 S 倍（128×4096×4096：S=8 128 µs）。
  改成部分和用 sc0 sc1（系统一致、绕过非一致 L2）读写 + vmcnt(0) + relaxed device 原子计数，去掉 fence：S=8 24.7 µs。只改 sc1 也对，aux 0 / nt 结果错（跨 XCD 不一致）。
- 剩余开销来自"最后到达的 block"单 CU 读 S−1 份 64 KB 部分和（单 CU 只有约 25 GB/s）。另做了"只写部分和 + 独立 reduce kernel"（kargs.split_k < 0，
  `gemm_a4w4_mxfp4_nt_sk_reduce`，dev 用 MXFP4_SK_SEPARATE=1）：reduce kernel 一开始因按 lane 构造 buffer 基址变成 readfirstlane 瀑布循环（43 µs），改成每 tile 基址后单独 3 µs；
  但"部分和 kernel + reduce kernel"合计在 S≥4 时远大于两者之和（例如 S=4：7.5 + 3.2 → 32 µs），原因未查清；S=2 与 in-kernel 相当。
- 量化（dev harness，128×128，非图）：

| shape | 非 split（np） | S=2 in-kernel / 独立 reduce | S=4 | S=8 |
|---|---|---|---|---|
| 128×4096×8192 | 15.1 | 15.2 / 15.3 | 17.7 / 14.8 | 24.0 / 17.7 |
| 256×4096×8192 | 15.3 | 17.7 / 19.1 | 19.0 / 17.1 | 41.1 / 23.9 |
| 512×2048×7168 | 13.8 | 16.3 / 17.5 | 19.3 / 19.2 | — |
| 128×2048×7168 | 13.3 | 14.1 / 13.3 | 15.7 / 12.7 | — |

  S=2 的固定开销约 5–6 µs（比小 shape 线的 3–4 µs 大），2×2 的 128×128 tile 上 split-K 基本没有收益（最好 −5%）。小 M 落后 asm 的根本原因是 tile 太大（asm 用 32×128 / 64×128，tile 数多）。
- 分发启发式重新标定（固定 3.5 µs + 轮数 × 面积 × K × 速率 / 效率；tile 数 ≤ CU 用 np；split-K 加 6 µs），集成只用 in-kernel 归约。
- 正确性：dev 矩阵 mode 1 六个 tile 104/104；split-K 128×128 S ∈ {1,2,4,8}、其余 4 个 tile S ∈ {2,8}、256×192 S=4 全部通过（不计 tile 数 > 1024 的预期拒绝）；
  nob 256×256 在整除形状 24/24；aiter op 测试 169 个用例全 OK。

## 最终性能表（HIP 6，CUDA graph × 100 次取 5 次中位数，每个形状前后确认无其他进程 CU_OCCUPANCY；perf/perf_vs_asm_HIP6_final.txt）

| M N K | asm | opus 分发 | 分发 / asm | 我们最好（tile variant） | 最好 / asm |
|---|---|---|---|---|---|
| 128 4096 4096 | 5.20 | 8.75 | 1.68 | 128x128 np 8.76 | 1.68 |
| 128 2048 7168 | 6.87 | 12.83 | 1.87 | 128x128 np 12.80 | 1.86 |
| 128 3072 6144 | 6.44 | 11.50 | 1.79 | 128x128 np 11.50 | 1.79 |
| 128 4096 8192 | 7.68 | 14.35 | 1.87 | 128x128 np 14.34 | 1.87 |
| 256 4096 4096 | 5.91 | 8.89 | 1.50 | 128x128 np 8.86 | 1.50 |
| 256 2048 7168 | 7.16 | 13.05 | 1.82 | 128x128 np 12.86 | 1.80 |
| 256 3072 6144 | 6.88 | 11.52 | 1.67 | 128x128 np 11.50 | 1.67 |
| 256 4096 8192 | 8.70 | 14.49 | 1.67 | 128x128 np 14.46 | 1.66 |
| 512 4096 4096 | 7.32 | 9.46 | 1.29 | 128x128 np 9.20 | 1.26 |
| 512 2048 7168 | 15.64 | 12.93 | 0.83 | 128x128 np 12.95 | 0.83 |
| 512 3072 6144 | 8.56 | 11.88 | 1.39 | 128x128 np 11.81 | 1.38 |
| 512 4096 8192 | 13.05 | 20.92 | 1.60 | 128x128 sk2 18.67 | 1.43 |
| 1024 4096 4096 | 12.07 | 11.92 | 0.99 | 128x128 np 11.55 | 0.96 |
| 1024 2048 7168 | 17.32 | 19.01 | 1.10 | 128x128 sk2 17.47 | 1.01 |
| 1024 3072 6144 | 13.99 | 16.92 | 1.21 | 128x128 np 16.67 | 1.19 |
| 1024 4096 8192 | 19.54 | 23.25 | 1.19 | 128x128 np 23.31 | 1.19 |
| 2048 4096 4096 | 19.52 | 20.49 | 1.05 | 256x128 np 19.64 | 1.01 |
| 2048 2048 7168 | 20.41 | 22.12 | 1.08 | 128x128 np 21.51 | 1.05 |
| 2048 3072 6144 | 24.36 | 24.13 | 0.99 | 256x128 np 23.49 | 0.96 |
| 2048 4096 8192 | 32.45 | 35.22 | 1.09 | 256x128 np 35.09 | 1.08 |

- M = 128–256：我们最好的是 128×128 非 persistent，比 asm 慢 1.5–1.9 倍（asm 用 32×128 tile、tile 数多 4 倍）；split-K 在这段没有收益。
- M = 512：N=2048/K=7168 快 17%（asm 该形状未调优，走 default），其余慢 1.2–1.6 倍；512×4096×8192 split-K S=2 最好（18.7 vs np 20.9），但仍慢 43%。
- M = 1024：4096×4096 持平，其余慢 1–1.2 倍。M = 2048：持平到慢 9%。
- 分发（重新标定后）与最好 kid 的差距 0–8%，不再选到慢的 split-K（旧常数在 256 行左右会选 256×128 sk8，84 µs）。
- 8192³（7 轮交替，perf/nob.txt）：persistent 222.5、无边界 nob 221.5、原 zc 220.9 µs（中位数）——nob 找回约 1 µs（0.45%），剩余约 0.6 µs。
- 结论：M ≤ 64 归小 shape 线；M = 128–1024 的差距主要在 tile 粒度（2×2 最小 128×128），要追上 asm 需要 64×128 / 32×128 一类的 tile（2×2 下每 wave 只剩 16–32 行，
  更适合单独的小 tile 流水线），split-K 在本流水线上固定开销 5–6 µs、只在 S=2 偶有小收益。

