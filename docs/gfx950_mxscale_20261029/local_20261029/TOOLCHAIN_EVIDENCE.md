# gfx950 本机编译器核对

分支源提交：`39cdf47d864c463b08f0cf9be146595db755578d`。此处只有 CPU 编译探测，没有运行 GPU。

当前 `csrc/opus_gemm/gen_co/gfx950/build_info.json` 记录了 780 个 ready CO，远端 `llvm_bin` 为 `/root/workspace/opus-pin-toolchain-49c41889-restore-20261028/build/bin`。该绝对路径本机不存在，780 个已保存镜像本机均可用，运行这些 CO 不需要重建。

JIT 配置 `aiter/jit/optCompilerConfig.json` 的 `module_deepgemm_opus` 要求 `-mllvm --amdgpu-mfma-vgpr-form`，并对 `*a8w8_bs_bpreshuf*.device.cu` 增加 `-mllvm --disable-machine-licm`。gfx950 这条配置没有启用 `amdgpu-expert-scheduling-mode`（该项条件限定 gfx1250）。`aiter/jit/core.py::hip_flag_checker` 会将编译器拒绝的 flags 警告后删除，因此 stock hipcc 的 mfma flag 被删除会改变 JIT 编译条件。

对一个空 HIP kernel，使用 `/opt/rocm/bin/hipcc --offload-arch=gfx950 -x hip --genco -O3` 分别探测三个 flags，完整记录见 `toolchain_probe.json`。

| 编译器 | version 标识 | mfma-vgpr-form | disable-machine-licm | expert-scheduling-mode |
| --- | --- | --- | --- | --- |
| `/opt/rocm/llvm/bin` | AMD clang 20，ROCm 7.0.0 | 拒绝 | 接受 | 拒绝 |
| `/root/workspace/llvm-src/build/bin` | clang 23，`46fcb339fb61119b337f973c7ca9e710a319fdd0` | 接受 | 接受 | 接受 |
| 旧 pin wrapper | clang 24，`49c41889681640665400cb01c9fbb4c0a024cde4` | 接受 | 接受 | 接受 |

旧 pin wrapper 完整目录：

```text
/root/workspace/aiter-fyz-opus-blockscale-mxfp8-gemm/reports/opus_retune_20261020_183111/pin_agpr_llvm/bin
```

其 `clang` 与 `clang++` 是脚本，最终调用 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang[++]`，并固定 `-resource-dir=/opt/rocm/lib/llvm/lib/clang/20`。版本标识与交接 CO 的 49c41889 一致。wrapper 明确拒绝旧 `-amdgpu-coerce-illegal-types=1`，此项被 JIT flag checker 删除属于 wrapper 的既有行为。

当前分支没有读取 `OPUS_HIP_CLANG_PATH`，单独设置该变量不会改变编译器。`module_deepgemm_opus` 没有 `hip_clang_path` 配置字段，调参器的 `_ensure_kids_compiled` 也直接调用 `build_module`，使用 hipcc 的原生环境变量可靠：

```bash
export HIP_CLANG_PATH=/root/workspace/aiter-fyz-opus-blockscale-mxfp8-gemm/reports/opus_retune_20261020_183111/pin_agpr_llvm/bin
export AITER_REBUILD=1
```

切换编译器后使用新进程并要求重建。`hip_flag_checker` 的进程内缓存只按 flag 字符串记忆，不包含编译器路径。`_ensure_kids_compiled` 会在父进程完成要求的重建并核对候选覆盖，再将子进程环境的 `AITER_REBUILD` 归零，避免多 GPU 子进程相互删改构建产物。

根 agent 已停止尚未进行 GPU 测量的 stock 编译预备阶段，使用旧 pin wrapper 重启第 1 步。最终公平比较还应核对实际构建日志中 mfma flag 被保留；空源探测证明工具链接受 flag，不代替真实 kernel 的正确性和性能验证。
