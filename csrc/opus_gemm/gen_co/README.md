# gen_co —— gfx950 预编译 code object（.co）

a4w4_mxfp4_nt（MXFP4 × MXFP4 NT，2×2 4-wave，persistent）用 `[[clang::amdgpu_pin_agpr/vgpr]]` 固定 A/B/C 片段，
只有 pin 工具链（clang 24 + amdgpu pin 补丁，如 yuyzhang512/llvm-project amdgpu-pin-op-dst）能编；aiter JIT 的标准 hipcc 编不了。
所以 device 代码在这里离线编成裸 ELF code object，运行时由 `include/gfx950/opus_co_launch_mxfp4_gfx950.cuh` 加载
（目录取 `OPUS_GEN_CO_DIR`，编译期宏或环境变量）；host launcher / dispatch 走普通 JIT。

- `co_kernels_gfx950.json`：每个 kernel 的 tile、traits、body 函数、pipeline 头、device flags。
- 命名：`<name>.co` == `extern "C"` 符号 == host launcher 名，`name = opus_gemm_gfx950_<pipeline>_co_<B_M>x<B_N>x<B_K>`。
- `build_co_gfx950.py --llvm-bin <pin toolchain>/bin --opus-include <能用该工具链编的 opus 的 csrc/include>`：
  生成 stub（extern "C" 入口调用 pipeline 的 `__device__` body），`--cuda-device-only` 编译，必要时 unbundle 成裸 ELF，
  校验符号名，记录编译器版本、flags、寄存器 / spill / LDS / sha256 到 `gfx950/build_info.json`。`--keep-asm` 把反汇编放到 `gfx950/asm/`（不提交）。
- 只包含 gfx950 的 mxfp4 kernel；与 blockscale 分支的 gen_co（gfx1250 / mxfp8）互不依赖。
