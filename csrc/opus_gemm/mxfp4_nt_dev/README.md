# MXFP4 × MXFP4 NT GEMM（opus，gfx950，2×2 4-wave 流水线）—— 开发 / 测试目录

kernel 本体在 `../include/gfx950/`：
- `opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh`：kernel 模板 `gemm_a4w4_mxfp4_4wave_kernel<Traits>`
- `opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh`：`gemm_a4w4_mxfp4_4wave_traits_t<B_M, B_N>`（tile 256×256 / 256×128 / 128×256 / 128×128，B_K = 256）和 kargs

输入布局与 aiter `gemm_a4w4(bpreshuffle=True)` 一致：A `[M, K/2]` 行主序 fp4x2，B `shuffle_weight(16,16)`，
scaleA / scaleB `e8m0_shuffle`（1x32 UE8M0），C `[M, N]` bf16。persistent grid（min(tile 数, CU 数)）。

边界在 kernel 内处理（不补 pad）：M / N 尾块靠 buffer range（读回 0、写丢弃）+ C 按 16 B 列块屏蔽；K 为 256 的倍数，N 为 16 的倍数。
设计说明见仓库 355_1_trans 的 `mxfp4_gemm_4wave_20261112/MULTITILE_DESIGN.md`；实验记录见 `EXPERIMENTS.md`。

## 构建与运行
需要 pin 工具链（clang 24 + `amdgpu_pin_agpr/vgpr` 补丁，默认 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build`）。
opus 暂用 aiter blockscale 分支的 `csrc/include`（本分支的 opus 还不能用这个工具链编译），可用 `OPUS_INCLUDE_DIR=` 覆盖。

```bash
./build_var.sh base ""                                  # -> /tmp/opus_mxfp4/base/build/gemm_a4w4_mxfp4_4wave.exe
E=/tmp/opus_mxfp4/base/build/gemm_a4w4_mxfp4_4wave.exe
$E -m 8192 -n 8192 -k 8192 -b 1 -t 256x256 -v 1 -w 1 -i 1      # 正确性
$E -m 8192 -n 8192 -k 8192 -b 1 -t 256x256 -v 0 -w 1000 -i 200 # 计时
```
`-t` 选 tile；环境变量 `MXFP4_SFA_K_PATTERN=1`（按 K group 变化的 scale）、`PERSIST_GRID=N`（指定 block 数）、`MXFP4_SF_PAD`（scale 补齐值）。
`prof/`：ATT trace（att.sh）与分析脚本（seg.py / quad.py / coexec.py）。

## 性能（8192³，GPU2 = HIP 2）
256×256：208.8 µs（约 5.27P），同时段 aiter asm f4gemm BpreShuffle_256x256 205.8 µs。
