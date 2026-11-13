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
