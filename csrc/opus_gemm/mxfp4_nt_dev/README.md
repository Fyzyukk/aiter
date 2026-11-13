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

## 注意：JIT 模块不会自动重编
aiter 的 JIT 不会因为 opus 源码（opus_gemm.cu、gen_instances.py、codegen、include、rocm_ops.hpp、pybind）更新而重编
`aiter/jit/module_deepgemm_opus.so`；旧 .so 可能缺新入口（例如 `opus_gemm_a4w4`，报 "not found"）。更新代码后先
`rm -f aiter/jit/module_deepgemm_opus.so`（或设 `AITER_REBUILD=1`）。perf 脚本（perf/perf_gemm_a4w4.sh、
../mxfp4_nt_noshuffle_dev/perf/perf_noshuffle.sh）开头会检查 .so 是否比源码旧或缺 `opus_gemm_a4w4`，`REBUILD=1` 时自动删除。

## 调优与调用（aiter a4w4 调优流程，opus 为第三组候选）
`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` 的候选 = CK + asm + opus（kid 300xx：persistent / np / split-K / nob），
opus 胜出时 tuned csv 写 `kernelName=opus_a4w4_kid<kid>`，`aiter.gemm_a4w4(bpreshuffle=True)` 读到后调用该 kid。
```bash
# 1) 自己准备 untuned csv（列 M,N,K，一行一个形状），例如 /path/to/a4w4_untuned.csv
# 2) 调优（新机器先 rm -f aiter/jit/module_deepgemm_opus.so，保证 opus JIT 模块是最新的）
cd csrc/ck_gemm_a4w4_blockscale
HIP_VISIBLE_DEVICES=0 PYTHONPATH=<aiter> python3 gemm_a4w4_blockscale_tune.py \
    -i /path/to/a4w4_untuned.csv -o /path/to/a4w4_tuned.csv --mp 8
# 3) 使用：指向 tuned csv（或把其中的行合入 aiter/configs/a4w4_blockscale_tuned_gemm.csv）
AITER_CONFIG_GEMM_A4W4=/path/to/a4w4_tuned.csv python3 -c "from aiter.ops.gemm_op_a4w4 import gemm_a4w4; ..."
#    y = gemm_a4w4(xq, shuffle_weight(w, layout=(16, 16)), e8m0_shuffle(xs), e8m0_shuffle(ws), bpreshuffle=True)
```
