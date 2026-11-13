# MXFP4 × MXFP4 NT GEMM，不 shuffle 输入（opus，gfx950，2×2 4-wave）—— 开发 / 测试目录

kernel 在 `../include/gfx950/`：
- `opus_gemm_pipeline_a4w4_mxfp4_nt_noshuffle_gfx950.cuh`：`opus_mxfp4_ns::gemm_a4w4_mxfp4_nt_noshuffle_kernel<Traits>`
  （以及给 .co 用的 `__device__` 体 `gemm_a4w4_mxfp4_nt_noshuffle_body`）。整个文件在 `namespace opus_mxfp4_ns` 里，
  宏在文件末尾 `#undef`，可以和 shuffle 版 pipeline 放进同一个 TU。
- `opus_gemm_traits_a4w4_mxfp4_nt_noshuffle_gfx950.cuh`：`opus_gemm_a4w4_mxfp4_nt_noshuffle_traits_gfx950<...>`（aiter 形式），
  简写 `opus_mxfp4_nt_ns_traits<B_M, B_N, MODE, BOUNDARY>`；kargs 复用 shuffle 版的 `opus_gemm_mxfp4_scale_kargs_gfx950`。

输入布局（全部行主序，不做 shuffle）：A `[M, K/2]`、B `[N, K/2]` fp4x2；scaleA `[M, K/32]`、scaleB `[N, K/32]` e8m0；C `[M, N]` bf16。
kargs：`stride_a = stride_b = K/2`，`stride_sfa = stride_sfb = K/32`（行距，字节），`stride_*_batch` 为每个 batch 的元素 / 字节数。

| 项 | 支持 |
|---|---|
| tile（B_M×B_N×B_K） | 256×256、256×128、128×256、128×128、192×256、256×192，B_K = 256 |
| 模式 | 0 = persistent（grid = min(tile 数, CU 数)）；1 = 每 block 一个 tile；另有 256×256 关闭边界处理的变体（`BOUNDARY=false`，需 M、N 整除） |
| shape | 任意 M ≥ 1；N 为 8 的倍数（C 按 16 B 列块写）；K 为 256 的倍数 |
| split-K | 暂不支持（等 shuffle 版的实现稳定后移植） |

pipeline 由 `gen_noshuffle_pipeline.py` 从 shuffle 版 pipeline 生成（`python3 gen_noshuffle_pipeline.py <shuffle .cuh> <noshuffle .cuh>`）：
换回 m8 的 B 布局（行主序 + 16 B XOR swizzle，与 A 对称）和 scale 路径（每 lane 读一行 16 B → 4×4 字节转置 → LDS），
其余（参数化、边界、persistent / 非 persistent、ZC_PEEL、C 写回）与 shuffle 版相同。shuffle 版改动后重新生成即可。

## 构建与运行
```bash
./build_var.sh base ""                       # -> /tmp/opus_mxfp4_ns/base/build/gemm_a4w4_mxfp4_4wave.exe
E=/tmp/opus_mxfp4_ns/base/build/gemm_a4w4_mxfp4_4wave.exe
$E -m 8192 -n 8192 -k 8192 -b 1 -t 256x256 -v 1 -w 0 -i 0        # 正确性
$E -m 8192 -n 8192 -k 8192 -b 1 -t 256x256 -v 0 -w 1000 -i 200   # 计时
prof/check.sh $E 2 256x256                   # 104 用例正确性矩阵（HIP 2）
prof/timing.sh 2 2                           # 等卡空闲后与 candidate_m8 交替计时 + 各 tile 中等 shape
```
`-t` 选 tile，`-p 0|1` 选模式（默认按 tile 数自动），`-o 0` 关闭边界处理（256×256、模式 0）。
环境变量：`MXFP4_SFA_K_PATTERN=1`、`PERSIST_GRID=N`、`MXFP4_SF_PAD`（scale 后面守护行的填充值，默认 0xFF = NaN，
用来确认 kernel 不会读到矩阵之外的 scale）、`MXFP4_CO_DIR=<dir>`（加载预编译 .co）。

## 预编译 .co
已并入共享的 `../gen_co/`：`co_kernels_gfx950.json` 里 pipeline `a4w4_mxfp4_nt_noshuffle` 的 13 条（json kid 100–105 persistent、
110–115 每 block 一个 tile、130 为 256×256 无边界），.co 在 `../gen_co/gfx950/`，用 `../gen_co/build_co_gfx950.py` 重编。

## aiter 接口
- 不 shuffle 版单独调用，输入布局与 Triton `gemm_afp4wfp4` 相同：
  `aiter.ops.opus.gemm_a4w4_opus_noshuffle(XQ, WQ, x_scale, w_scale, Y=None, kernelId=None)`
  （XQ `[M, K/2]`、WQ `[N, K/2]` 为 fp4x2 或 uint8；x_scale `[M, K/32]`、w_scale `[N, K/32]` 为 e8m0 或 uint8；
  kernelId 取 `A4W4_MXFP4_NT_NOSHUFFLE_KIDS` 之一，默认 None = opus 分发）。支持 batch（3D 输入）。
- 检查：N % 8、K % 256、scale 行主序 `[rows, K/32]`（每 batch ≥ M / N 行，2D `[batch*rows, K/32]` 或 3D）。
- C++：`opus_gemm_a4w4(XQ, WQ, Y, x_scale, w_scale, bpreshuffle=False)`（内部实现）、`opus_gemm_a4w4_tune`（kid 301xx）。
- shuffle 版不在这里：走 `aiter.gemm_a4w4(..., bpreshuffle=True)` 的 opus 后端（见 ../mxfp4_nt_dev/README.md）。

## 性能测试（可移植）
`perf/perf_noshuffle.sh`：`GPU=<HIP 编号> SMI=<smi 编号，可空> AITER=<checkout> [SHAPES="M,N,K ..."] [KIDS=0] [M8_EXE=.. DEV_EXE=..]`；
CUDA graph 测 `gemm_a4w4_opus_noshuffle`（分发 + 每个 kid）、Triton `gemm_afp4wfp4`（同一份输入）、asm（`gemm_a4w4` 默认，同数据的 shuffle 副本）；
可选 dev harness 与 candidate_m8 交替测 8192³。
