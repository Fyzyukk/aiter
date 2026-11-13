# MXFP4 NT 2×2 kernel：tile 参数如何推导，以及如何接入 aiter opus_gemm（2026-11-13）

## 1. 现在的 traits：只有 B_M、B_N 两个模板参数

`gemm_a4w4_mxfp4_4wave_traits_t<B_M, B_N>`（`include/gfx950/opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh`），B_M、B_N ∈ {256, 128}。

**固定不变的**

| 参数 | 值 | 原因 |
|---|---|---|
| BLOCK_SIZE / NUM_WAVES | 256 / 4 | 4-wave，每 SIMD 一个 wave |
| T_M × T_N × T_K | 2 × 2 × 1 | 2×2 wave 布局 |
| W_M × W_N × W_K | 16 × 16 × 128 | `v_mfma_scale_f32_16x16x128_f8f6f4`（fp4） |
| B_K | 256（128 字节/行） | 一个 K tile = 2 条 MFMA 的 K（E_K = 2，断言） |
| VEC_A / VEC_B | 16 字节 | 一条 b128 = 一个 fp4 MFMA 操作数；LDS-DMA 每 lane 最多 16 B |
| GROUP_K | 32 | MXFP4 1x32 UE8M0 |
| smem_padding | 64 字节 | chunk 跨度 1088 B，配 16 B swizzle 让 ds_read_b128 无 bank 冲突 |
| SCALE_PANEL_BUFFERS | 3 | 主循环旋转后 scale 要提前两个 tile 拷 |

**由 B_M / B_N 推导的**

| 参数 | 公式 | 256×256 | 128×128 |
|---|---|---|---|
| HALF_B_M / HALF_B_N | B_M/2、B_N/2 | 128 / 128 | 64 / 64 |
| E_M / E_N（每 wave 每半区 16 行块数） | HALF / (16 × 2) | 4 / 4 | 2 / 2 |
| 每 quadrant MFMA 数 | E_M × E_N × 2 | 32 | 8 |
| a/b_buffer_load_insts（每 wave 每半区 LDS-DMA 条数） | HALF × 128 / (256 × 16) | 4 / 4 | 2 / 2 |
| smem_m_rep / smem_n_rep（每半区 chunk 数） | HALF / 8 | 16 | 8 |
| SF_WAVES_A / SF_WAVES_B（拷 scale panel 的 wave 数） | B_M/64、B_N/64 | 4 / 4 | 2 / 2（另 2 个 wave 重复拷） |
| SFA/SFB_PANEL_BYTES | B × 16 | 4096 | 2048 |
| C_COPIES（每线程上半 C 的 16 B 拷贝数） | HALF_B_M × B_N / (256 × 8) | 16 | 4 |
| C_SWZ（C 暂存行 swizzle） | B_N == 256 | 是 | 否 |
| LDS_BYTES | A、B 两 stage + 3 × scale panel | 160 KB | 80 KB |

pipeline 里另外按这些推导：调度（256×256 用原手工模板，其他 tile 用 `sched_spread<Q, DS, VMEM, VALU, STORE>`）、
手写 vmcnt（B_b = `a + b`；prologue = `2(a + b) + C_COPIES` / `2(a + b)`）、SF_SHIFT 每半区 E 个 dword。

## 2. aiter opus_gemm 现在怎么组织（fork main ca68b4f）

| 层 | 文件 | 做法 |
|---|---|---|
| 实例定义 | `opus_gemm_common.py` | `OpusGemmInstance(BLOCK_SIZE, B_M, B_N, B_K, T_M, T_N, W_M, W_N, W_K, VEC_A, VEC_B, VEC_C, GROUP_M, GROUP_N, GROUP_K, kernel_tag, ...)`，每个 pipeline 一张 kid 表（如 `a8w8_scale_kernels_list = {1: ...}`），合并成 `kernels_list` |
| 代码生成 | `codegen/gen_instances_gfx950.py` | 按 `kernel_tag` 查 4 张表：pipeline 头文件、traits 名、kargs 名、`__global__` 函数名；每个实例生成一个 launcher，名字如 `opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128`；`register_emit("gfx950", tag, fn)` |
| traits 形参 | `opus_gemm_traits_a8w8_scale_gfx950.cuh` | `template<BLOCK_SIZE, opus::seq<B_M,B_N,B_K>, opus::tuple<D_A,D_B,D_C,D_ACC,D_SF>, opus::seq<VEC_A,VEC_B,VEC_C>, opus::seq<GROUP_M,GROUP_N,GROUP_K>>`，内部推导 E_M、buffer_load_insts 等（和我们同一套思路） |
| 分发 | `opus_gemm.cu` | `opus_gemm()` 按 XQ dtype 分支：bf16 → `opus_dispatch_a16w16`（按 (M,N,K) 查调优 csv 烤进的表，查不到走启发式）；fp8 → 写死的单个 launcher |
| Python | `aiter/ops/opus/gemm_op_*.py` | `@compile_ops("module_deepgemm_opus", fc_name=...)`；JIT 模块在 `aiter/jit/optCompilerConfig.json`，`blob_gen_cmd` 调 gen_instances.py 并传入调优 csv |
| 编译 | aiter JIT | 标准 ROCm hipcc |

## 3. 接入需要做什么

1. **traits / kargs 改成 aiter 形式**：`opus_gemm_a4w4_mxfp4_nt_traits_gfx950<BLOCK_SIZE, seq<B_M,B_N,B_K>, tuple<fp4x2,fp4x2,bf16,fp32,e8m0>, seq<16,16,4>, seq<1,1,32>>`，
   内部推导保持不变；kargs 改名 `opus_gemm_mxfp4_scale_kargs_gfx950`（比 a8w8 scale kargs 多 `grid_blocks`）。kernel 体包 `#if defined(__gfx950__)` 多架构保护。
2. **实例表**：`kernel_tag = "a4w4_mxfp4_nt"`，4 个实例
   `OpusGemmInstance(256, {256|128}, {256|128}, 256, 2, 2, 16, 16, 128, 16, 16, 4, 1, 1, 32, "a4w4_mxfp4_nt", ["bf16_t"])`，选一个不冲突的 kid 段。
3. **launcher 生成**：在 gen_instances_gfx950.py 的 4 张表里加 tag；emit 函数要处理和 a8w8_scale 不同的地方——
   fp4 的 stride 按字节（stride_a = K/2）、scale stride 按 e8m0_shuffle（一个 32 行块的字节数 = 32 × K/32 补到 8）、
   persistent grid = min(tile 数, CU 数) 并写入 `kargs.grid_blocks`、输出 bf16。
4. **分发**：`opus_gemm()` 增加 fp4x2 + scale 分支 → `opus_dispatch_a4w4_mxfp4_gfx950(M, N, K, batch)`：先查调优表，查不到用启发式
   （按 tile 数和 CU 数选使轮数 × 单 tile 时间最小的 tile）；另加按 kid 直接调用的 tune 入口。
5. **Python**：`aiter/ops/opus/gemm_op_a4w4.py`（主入口 + tune 入口），输入布局与 `gemm_a4w4(bpreshuffle=True)` 相同；调优 csv 和 tune 脚本。
6. **测试**：`op_tests/test_opus_a4w4_gemm.py`，与 `gemm_a4w4` asm 及 torch 参考对比。

## 4. 最大的阻碍：编译工具链

kernel 依赖 `[[clang::amdgpu_pin_agpr/vgpr]]`，只有 pin 工具链（clang 24 + pin 补丁）能编；aiter JIT 用标准 hipcc，编不了。
另外 fork main 的 `csrc/include/opus/opus.hpp` 也不能用 pin 工具链编译（builtin 参数类型不匹配），现在借用 blockscale 分支的 opus。

可选方案：
- **预编译 .co（推荐）**：blockscale 分支（`aiter-blockscale-mxfp8-nn-nt-tt`）已有这套：`csrc/opus_gemm/gen_co/build_co.py` +
  `co_kernels_gfx950.json` 用 pin 工具链离线编出 `.co`，launcher 在运行时 `hipModuleLoad`；之前对比用的 PR6696 pin_agpr kernel 就是这样接入的。
  但 fork main 和上游 main 都还没有 gen_co，需要先把这套基础设施合进来（或等它上游合入）。
- 去掉 pin 用 hipcc 编：性能会大幅下降（pin 是关键优化），不建议。
- 等上游 LLVM 合入 pin 支持。

## 5. 建议的步骤

1. 先做第 1 步（traits/kargs 改成 aiter 形式），dev 目录的独立测试继续可用，确认生成的代码不变。
2. 决定编译路线：如果走 .co，把 blockscale 分支的 gen_co 基础设施移植到本分支，加 mxfp4 的 co 条目。
3. 再做 codegen / 分发 / Python / 测试（第 2–6 步）。
