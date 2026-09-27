# MXFP8 合并 kernel：统一为 9000/9020 的源码结构

新机器构建和调优使用
[可移植入口](../opus_remote_tune_20260927/README.md)。
本目录的 `build_verify.py` 保留原机改名前后的等价性证据，需要原机参考二进制；
它不是新机器的构建入口。

三个新 family 已使用正式的 `opus_gemm_pipeline_*_gfx950.cuh` /
`opus_gemm_traits_*_gfx950.cuh` 命名，放在 `csrc/opus_gemm/include/gfx950/`。
三个 kernel 入口均为 `template<class Traits>`，复用原 96 字节 kargs、
host stub、gfx950 条件编译和 Traits launch bounds。

| Family | 私有 ID | Tile M×N×K | Traits |
|---|---|---|---|
| main | 21000 | 192×256×128 | `opus_gemm_mxscale_bpreshuffle_main_traits_gfx950` |
| small | 21310 / 21311 | 128×128×128 / 160×128×128 | `opus_gemm_mxscale_bpreshuffle_small_traits_gfx950<128/160>` |
| narrow | 21220 / 21221 | 64×128×128 / 64×64×128 | `opus_gemm_mxscale_bpreshuffle_narrow_traits_gfx950<128/64>` |

对应的正式源文件：

| Family | Pipeline | Traits |
|---|---|---|
| main | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_main_gfx950.cuh) | [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_main_gfx950.cuh) |
| small | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_gfx950.cuh) | [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh) |
| narrow | [pipeline](../../csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh) | [traits](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh) |

例如 small 的两个实例通过同一模板生成：

```cpp
using Traits = opus_gemm_mxscale_bpreshuffle_small_traits_gfx950<128>;
gemm_a8w8_mxfp8_bpreshuffle_small_kernel<Traits>
    <<<grid, Traits::BLOCK_SIZE, 0, stream>>>(args);
```

`main/launch.hip`、`small/launch.hip`、`narrow/launch.hip` 仅保留测量用的
C ABI 入口、原 shape/指针对齐检查和启动配置，计算主体全部来自正式头文件。
[experiments.json](experiments.json) 引用这三个 adapter；五个私有 ID 沿用原编号，
没有新增全局注册。最终合并池仍计划为原 9000/9020 加最多五个新 geometry。

## 验证结果

执行：

```bash
python reports/opus_native_style_20260927/build_verify.py
```

三库共五个 device kernel 均编译通过；按 ID 比较改名前后 symbol，五个入口的
指令字节数和 SHA256 全部一致，kernel descriptor 除重定位代码地址外全部一致。
整库和 symbol 名称因模板化/改名而改变，不用它们替代指令比较。
94 个既有文件的哈希保持不变，包括 9000/9020、现有公共头、
`csrc/include/opus` helpers、注册、codegen，以及来源版本的源码/二进制。
本轮只做 CPU 编译和 ELF 检查，未运行 GPU。

证据：[build_manifest.json](build_manifest.json)，以及各 family 的
`device_audit.json`。来源映射见 [sources.json](sources.json)；
来源依次为 `main_v1`、`small_v4`、`tiny_v3`。

## 接续状态

本轮完成源码结构整理，未把它当作全量调优完成：
第三批原 `pilot62_v34_r3` 已中断，八份 summary 无数据行；
合并版完整 295 项比较、最终选型、旧候选删除和 JIT 替换仍待完成。
当前已完成验证并保留的 16 候选基线不变。

后续应从新的 batch 路径重新开始测量。新的 `experiments.json` 可用于测量
这五个正式头实例；若仍需比较 v2/v3/v4 多版本，应在新的测量配置中明确加入
对照。旧 `continue_measurements.py` 和安装工具保留原版本目录/符号/哈希映射，
不会自动切换到本次头文件；最终安装前需要根据全量实测选择更新这些映射。
