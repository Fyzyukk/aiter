# MXFP8 B-preshuffle retained kernels

本包保留2026-09-27完整295项、3轮重新调优中实际被整体选择使用的11个runtime-K OPUS kernel，共9个独立共享库。当时的另5个候选为9000、9010、9011、9012、9020。完整16候选的历史模板、Traits、几何和选择次数见[SELECTED_CANDIDATES.md](../../../reports/opus_retune_prune_20260927/SELECTED_CANDIDATES.md)。

2026-09-28已删除旧9010/9011/9012，并重编号当前七个注册项；本目录配置中的注册ID仍表示历史版本，当前列表见[上级README](../README.md)。

| Library | 私有ID | Device kernel数量 |
|---|---|---:|
| [long_epilogue_sync](long_epilogue_sync/launch.hip) | 13163 | 1 |
| [long_runtime](long_runtime/launch.hip) | 20000 | 1 |
| [short_runtime](short_runtime/launch.hip) | 20010, 20011 | 2 |
| [n224_runtime](n224_runtime/launch.hip) | 20020 | 1 |
| [long_runtime_grid](long_runtime_grid/launch.hip) | 20100 | 1 |
| [n224_runtime_loop](n224_runtime_loop/launch.hip) | 20124 | 1 |
| [short_runtime_unified](short_runtime_unified/launch.hip) | 20125, 20126 | 2 |
| [long_runtime_fused](long_runtime_fused/launch.hip) | 20128 | 1 |
| [short_runtime_group4_cache2](short_runtime_group4_cache2/launch.hip) | 20131 | 1 |

每个`<library>/experiments.so`独立导出以下C ABI，`variant`仅在该库内解释：

```cpp
extern "C" int launch(
    int variant, const void* a, const void* b,
    const void* sfa, const void* sfb, void* c,
    int m, int n, int k, void* stream);
```

`a/b`指向FP8输入，B已采用本项目的B-preshuffle布局；`sfa/sfb`指向E8M0 scale，`c`指向BF16输出，`stream`为HIP stream。各launcher保留相应的shape、对齐和字节范围检查，返回HIP错误码。私有ID未加入生产全局registry；尤其20000段与gfx1250生产ID空间重合，调用时应通过本包对应共享库和[experiments.json](experiments.json)定位。配置中的`registered_opus_ids`声明另外5个生产候选。

从仓库根目录重新构建全部保留库：

```bash
python csrc/opus_gemm/mxfp8_bpreshuffle_retained/build_retained.py
```

构建使用包内保存的编译命令模板及源码哈希，不再读取旧reports实验源码。默认采用已固定的LLVM工具链；迁移工具链位置时可加`--llvm /path/to/pinned-llvm/bin`，仓库位置可通过`--root /path/to/aiter`指定。脚本只做CPU侧编译与device入口/机器码比对，输出各库`build_manifest.json`以及包级[build_manifest.json](build_manifest.json)、[device_audit.json](device_audit.json)。

所有11个kernel的K均为运行时参数，没有K384、K768、K1024或K1536专属实例。`short_runtime`包含`TileM=128/160`两个实例，`short_runtime_unified`包含`OutputVector=8/4`两个实例，其余库各一个入口。

当前构建与机器码比对记录均为`passed`：[device_audit.json](device_audit.json)记录恰好11个中选device入口，每个入口的instruction bytes与调优时实测kernel一致；共享库整体哈希不要求与裁剪前相同。[retained_manifest.json](retained_manifest.json)保存迁移前实测入口与源码的来源；其中`prepared_not_compiled`描述准备阶段，当前编译状态以`build_manifest.json`和`device_audit.json`为准。

共享依赖位于`../include/gfx950/`与`../../include/opus/`。9000的4wave pipeline、公共helpers和基础traits保持原样；192×256 traits还被13163、20000、20100、20128使用。
