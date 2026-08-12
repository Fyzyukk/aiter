# OPUS GEMM 任务二详细文件修改与功能总结

更新时间：`2026-08-12 UTC`

状态：Task2 B0--B7 已实施并完成可用硬件范围内的验收。本文件按**当前最终源码**整理文件级
修改，不按中间实验状态整理。

## 1. 文档口径

当前工作树同时包含 Task1 和 Task2 的累计未提交修改，直接读取 `git diff` 会把两项任务混在
一起。本文件使用以下口径归属 Task2：

1. 以 `docs/task2_step_b0_detail.md` 到 `docs/task2_step_b7_process.md` 的逐步记录为施工证据；
2. 再与当前源码中的函数、宏、generated table 和测试节点交叉核对；
3. 同一个文件若同时含 Task1 和 Task2 修改，只描述其中属于 Task2 的部分；
4. B6 期间短暂出现过的 prepared/prevalidated workspace 实验已经回退，当前最终实现是
   checked-only launcher，不把该实验写成 Task2 最终功能；
5. `gfx942`、`gfx1250` 的 codegen、对象编译和 ABI 已检查，但当前节点没有对应实机，不能把这些
   结果写成 GPU 数值或性能通过。

## 2. Task2 完成的核心功能

Task2 将原来混合了 runtime policy、generic dispatch 和旧 tune 名称的 OPUS GEMM 接口，整理为
四条稳定的 family-specific canonical launch：

```text
opus_gemm_a16w16_launch
opus_gemm_a8w8_launch
opus_gemm_a8w8_blockscale_launch
opus_gemm_a8w8_blockscale_bpreshuffle_launch
```

最终调用链为：

```text
Python public API
  -> explicit / tuned row / Python heuristic or per-arch default
  -> resolved arch + logical family + actual kid + Y dtype + split_k
  -> one canonical private raw binding
  -> family-specific C++ entry
  -> current-arch/current-dtype typed exact-kid table
  -> generated physical-contract launcher
  -> kernel
```

完成的主要能力包括：

- A16 explicit、tuned、heuristic 统一复用 Python selector，并在 workspace 准备前解析
  `actual_kid`；
- C++ 不再读取 runtime shape 表、不再运行 heuristic，也不再提供 generic `opus_gemm()` mega
  entry；
- A8 拆为 no-scale、plain-WQ blockscale、blockscale-bpreshuffle 三种独立物理合同和 typed
  dispatch 表；
- gfx950 no-scale kid 2、gfx950 blockscale kid 1、gfx942 bpreshuffle kid 11000 由 canonical
  registry 驱动，不在 C++ 中硬编码 concrete kernel symbol；
- gfx950/gfx1250 的 bpreshuffle 公共 ABI 和空表槽位稳定存在，当前明确报告 no registered
  kernel，不借用 gfx942 kid 11000；
- build-time CSV、sidecar、Python heuristic defaults 和 mandatory A8 kids 继续组成 subset
  compile set，但 CSV shape 不再成为 C++ runtime policy；
- 仓内生产调用方和 tuner 全部迁移到 canonical launch；旧 Python tune 名只保留为带 warning
  的 adapter，旧 C++/pybind tune 符号已删除；
- generated launcher 继续执行最终 dtype、shape、stride、layout、tile、prefetch 和 workspace
  合同检查；
- A8 public adapter 增加安全的 device/registry metadata 热缓存，降低 no-scale Python 开销，同时
  保留每次调用的 Tensor/type/same-device 检查和 C++ 最终复核。

当前 A8 capability 为：

| Logical family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `a8w8` | 空 | kid 2，FP32 Y | 空 |
| `a8w8_blockscale` | 空 | kid 1，FP32 Y，plain WQ | 空 |
| `a8w8_blockscale_bpreshuffle` | kid 11000，BF16 Y | 可编译空表 | 可编译空表 |

### 2.1 文件总览

下面是 Task2 实际修改、新增或删除的代码/测试文件快速索引；混合归属文件只计算本文后续标出的
Task2 部分。

| 分组 | 文件 |
|---|---|
| C++/pybind | `csrc/opus_gemm/include/opus_gemm.h`、`csrc/opus_gemm/opus_gemm.cu`、`csrc/include/rocm_ops.hpp`、`csrc/pybind/opus_gemm_pybind.cu` |
| Python OPUS | `aiter/ops/opus/gemm_op_a16w16.py`、`aiter/ops/opus/gemm_op_a8w8.py`、`aiter/ops/opus/__init__.py`、`aiter/ops/opus/_selector_a16w16.py` |
| 生产调用方/tuner | `aiter/tuned_gemm.py`、`csrc/opus_gemm/opus_gemm_tune.py`、`csrc/gemm_a16w16/gemm_a16w16_tune.py`、`aiter/ops/deepgemm.py`、`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`、`aiter/ops/gemm_op_a8w8.py` |
| Registry/codegen | `csrc/opus_gemm/opus_gemm_common.py`、`csrc/opus_gemm/gen_instances.py`、三个 `codegen/gen_instances_gfx*.py`、三个 `include/gfx*/opus_gemm_arch_gfx*.cuh` |
| 删除的 C++ policy | 三个 `include/gfx*/opus_gemm_heuristic_dispatch_gfx*.cuh` |
| 混合归属注释 | `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` |
| 测试/benchmark | `op_tests/test_opus_interfaces.py`、`test_opus_dispatch.py`、`test_opus_workspace.py`、`test_opus_graph.py`、`test_opus_gfx950_exhaustive.py`、`bench_opus_task1_task2_interfaces.py`、`bench_opus_gfx950_workspace_ab.py` |
| 用户文档 | `aiter/ops/opus/README.md`、`csrc/opus_gemm/README.md` |
| Task2 过程文档 | `docs/opus_gemm_two_tasks_final_plan.md`、`docs/task2_step_b0_*` 到 `docs/task2_step_b6_*`、`docs/task2_step_b7_process.md`、`docs/task2_checkpoint.md` |

## 3. C++ 公共接口和 pybind

### 3.1 `csrc/opus_gemm/include/opus_gemm.h`

修改位置：OPUS GEMM 公共函数声明区。

修改内容：

- 新增并最终只保留四条 canonical C++ launch 声明；
- A16 参数统一为 `bias, workspace, kid, split_k`；
- 两个 blockscale API 的 `x_scale`、`w_scale` 改为必选 `aiter_tensor_t&`，不再是 optional；
- 删除旧 `opus_gemm_a16w16_tune` 和旧 gfx942 A8 tune 声明；
- 删除无 family 后缀的 generic `opus_gemm()` 声明。

完成的功能：公共 ABI 从“generic/tune 混合入口”收敛为四种物理合同明确的 family launch。

### 3.2 `csrc/opus_gemm/opus_gemm.cu`

修改位置：arch router、A16 launch helper、A8 family dispatch 和四条 public C++ entry。

修改内容：

- A16 通过 `opus_gemm_a16w16_launch_impl()` 进入 workspace/non-workspace exact-kid dispatch；
- 新增三条 A8 family router：
  - `opus_a8w8_kid_dispatch()`；
  - `opus_a8w8_blockscale_kid_dispatch()`；
  - `opus_a8w8_blockscale_bpreshuffle_kid_dispatch<CDataType>()`；
- router 先读取当前 HIP device arch，再选择当前 arch 和 Y dtype 的 generated table；
- 公共检查覆盖 GPU/device 一致性、输入 dtype、scale device 和 family 支持的 Y dtype；
- 区分三类错误：module 未编入 arch、family/dtype 表为空、非空表中 unknown kid；
- 删除 generic `opus_gemm()`、hardcoded scale/no-scale function-pointer helper 和 concrete symbol
  switch；
- 删除旧 A16/A8 C++ tune 实现；
- 删除 C++ runtime shape lookup、heuristic 和 framework fallback 入口。

完成的功能：C++ 成为纯粹的 current-device + family + dtype + exact-kid 安全路由层，不再与
Python 重复做 runtime 选核策略。

### 3.3 `csrc/include/rocm_ops.hpp`

修改位置：OPUS pybind 宏区。

修改内容：

- 新增四个 `...LAUNCH_PYBIND` 宏并固定参数名和顺序；
- 删除 `OPUS_GEMM_PYBIND` generic 宏；
- 删除 A16 和 gfx942 A8 旧 tune pybind 宏；
- 保留其他模块的同名/相邻宏，例如 deepgemm CK，不做误删。

完成的功能：pybind schema 与 C++/Python canonical ABI 对齐。

### 3.4 `csrc/pybind/opus_gemm_pybind.cu`

修改位置：`PYBIND11_MODULE` 注册体。

当前业务注册项仅为：

```text
OPUS_GEMM_A16W16_LAUNCH_PYBIND
OPUS_GEMM_A8W8_LAUNCH_PYBIND
OPUS_GEMM_A8W8_BLOCKSCALE_LAUNCH_PYBIND
OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_LAUNCH_PYBIND
```

删除 generic 和旧 tune registration。完成的功能是三架构模块暴露相同的四条业务 raw 属性，
即使某个 family 当前是空 capability，符号本身也不会消失。

## 4. Python OPUS 接口和 runtime policy

### 4.1 `aiter/ops/opus/gemm_op_a16w16.py`

Task2 修改位置：canonical raw、layout 检查、explicit launch、compat wrapper 和高层
`gemm_a16w16_opus()` 的 raw 目标。

修改内容：

- 新增 `_opus_gemm_a16w16_launch_raw()` 及 fake registration；
- 新增 public `opus_gemm_a16w16_launch()`；
- `_explicit_a16w16_launch()` 复用 `select_launch_config()`，先得到 `actual_kid`，再准备 exact-kid
  workspace；
- explicit、CSV tuned、heuristic 最终都调用 `_opus_gemm_a16w16_launch_raw()`；
- 保留 2D/3D normalize、padded leading stride、bias、framework fallback 和 Task1 workspace
  shape/dtype/capacity；
- 旧 `opus_gemm_a16w16_tune()` 改为纯 Python deprecated adapter：解析旧位置参数和
  `kernelId/splitK`，只发一次 warning，然后调用 canonical launch；
- 删除旧 private tune raw/fake registration；
- 删除 `_opus_gemm_bf16_dispatch` generic raw/fake 和未使用的 `group_layout`；
- 统一错误和注释中的 `launch/kid/split_k` 术语。

完成的功能：A16 public 行为不变，但所有成功 OPUS 路径只经过一个 canonical raw ABI。

### 4.2 `aiter/ops/opus/gemm_op_a8w8.py`

修改位置：device helpers、registry capability helpers、三条 raw binding、三条 public wrapper、
bpreshuffle kid resolution 和旧 wrapper。

修改内容：

- 新增三条 private raw：
  - `_opus_gemm_a8w8_launch_raw()`；
  - `_opus_gemm_a8w8_blockscale_launch_raw()`；
  - `_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw()`；
- 新增对应 public canonical wrapper，均返回调用方传入的同一个 `Y`；
- no-scale/blockscale 的默认 kid 分别为 gfx950 2/1，但仍按
  `(arch, family, kid, Y.dtype)` 做 strict registry 校验；
- bpreshuffle 的 `kid=None` 按 explicit kid → 当前 shape 的 OPUS tuned row → per-arch default
  解析，raw 永远只接收 resolved integer kid；
- gfx942 旧 `...bpreshuffle_tune()` 只负责旧 `Y=None/kernelId` 兼容、warning 和 canonical 转发；
- `_check_same_device()` 每次 public 调用仍校验所有 Tensor 类型和同 device，合法热路径不再创建
  临时 list/set；
- `_device_arch()` 按规范化后的显式 `torch.device` 缓存 arch，不把无 index 的 `cuda` 错误缓存为
  固定物理卡；
- `_require_registered_kid_cached()` 只缓存成功的 immutable capability 查询；异常不会进入
  `lru_cache`，因此 empty/unknown capability 不会冻结为 stale negative；
- 不缓存 Tensor、data pointer、stream 或 raw launcher，C++/generated validator 仍是最后防线；
- 文档明确 bpreshuffle 是 WQ 内容语义，metadata 检查不能证明 weight 真正经过 shuffle。

完成的功能：建立三种 A8 独立 public/ABI 合同，并在不删除安全检查的前提下降低 no-scale
Python adapter 热路径开销。

### 4.3 `aiter/ops/opus/__init__.py`

修改位置：supported-arch import、unsupported stub 和 `__all__`。

修改内容：

- `gfx942/gfx950/gfx1250` 均导出四条 canonical launch；
- 同时导出两个有迁移期限的旧 Python compatibility 名；
- 三个 OPUS arch 之外使用调用时失败的 stub，而不是 import-time 中断；
- 空 family 不在 import 层隐藏符号，而由 runtime capability 错误表达；
- 修正 supported arch 提示，包含 gfx1250。

完成的功能：三架构 Python 暴露面稳定一致，unsupported OPUS 不会截断 `aiter` 顶层后续导入。

### 4.4 `aiter/ops/opus/_selector_a16w16.py`

这是 Task1/Task2 混合归属文件。Task2 只做以下收口：

- 错误、注释和字段说明由旧 `tune dispatcher` 术语改为 canonical `launch/split_k`；
- 明确 Python selector 是唯一 runtime policy 层；
- 保持 explicit → tuned → heuristic → framework fallback 顺序不变。

该文件中的 gfx1250 fused selector、workspace dtype 和 bias 细节主要属于 Task1，不应整体归入
Task2。

## 5. 仓内调用方和 tuner 迁移

| 文件 | Task2 修改位置 | 修改内容与完成的功能 |
|---|---|---|
| `aiter/tuned_gemm.py` | OPUS lazy import、availability gate、实际调用 | 由旧 tune wrapper 改调 `opus_gemm_a16w16_launch(kid=..., split_k=...)`；CSV 的 `splitK` 字段格式保持不变 |
| `csrc/opus_gemm/opus_gemm_tune.py` | import、`run_opus_gemm()`、`run_opus_gemm_bench()` | tuner 直接调用 canonical A16 launch；调优流程中的 tune 语义和 CSV 字段继续保留 |
| `csrc/gemm_a16w16/gemm_a16w16_tune.py` | OPUS import、availability、`run_opus_gemm_bf16()` | 汇总 tuner 改用 canonical A16 launch 和 `kid/split_k` 参数 |
| `aiter/ops/deepgemm.py` | legacy OPUS shim | shim 自己发一条 warning 后直接调用 canonical launch，避免经过旧 wrapper 再发第二条 warning |
| `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` | OPUS import、registry 枚举、launch | 改调 canonical bpreshuffle launch；按 arch/family/dtype 查询 canonical registry，不再把代码写死为 gfx942-only |
| `aiter/ops/gemm_op_a8w8.py` | tuned high-level router 的 `libtype == "opus"` 分支 | 迁移到 canonical bpreshuffle launch；CK/CKTile/ASM/FlyDSL/Triton/Gluon 分支保持原路径，空 OPUS 表不会抢占非 OPUS backend |

### 5.1 `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh`

该文件主体的 kernel/traits 修改属于 Task1。Task2 只更新架构保护注释中的公开名字和 host router
名字：旧 `opus_gemm_a16w16_tune`/tune-dispatch 改为 canonical launch/kid-dispatch。不要把该文件
其余 mono/FP32 物理修复归入 Task2。

## 6. Registry、subset compile 和 codegen

### 6.1 `csrc/opus_gemm/opus_gemm_common.py`

Task2 修改位置：family registry 查询、heuristic compile defaults 和 A8 mandatory set。

修改内容：

- 新增 `OPUS_KERNEL_TAGS_BY_ARCH_FAMILY`，显式定义 arch → logical family → canonical tag 集合；
- 扩展 `get_kernel_instance(arch, family, kid, output_dtype=None)`，按
  `(arch, family, kid, Y.dtype)` 窄查询；
- 空 tag 集合表示稳定但当前无 kernel 的 capability slot，缺少 family key 表示未知 family；
- 新增 `OPUS_MANDATORY_A8_KIDS`：gfx950 `{1,2}`、gfx942 `{11000}`、gfx1250 空；
- 保留 Python `HEURISTIC_DEFAULT_KIDS` 供 runtime selector 和 build invariant 使用；
- 删除只服务 C++ runtime shape lookup 的 `default_kernels_dict`，不删除 canonical
  `kernels_list`。

完成的功能：registry 成为 A16/A8 metadata single source of truth，同时严格阻止跨 arch、跨 family
或错 dtype 借用同一裸 kid。

### 6.2 `csrc/opus_gemm/gen_instances.py`

Task2 修改位置：host ABI 常量、A16/A8 dispatch header 生成、subset compile 和旧 artifact 清理。

修改内容：

- A8 blockscale host extra 从两个 optional Tensor 改为两个必选引用；
- 生成 `opus_gemm_a16w16_kid_dispatch.h` 和 `opus_gemm_a8w8_kid_dispatch.h`；
- A8 生成 no-scale、plain blockscale、以及按 `(arch,Y.dtype)` 分表的 bpreshuffle typed table；
- 每个表生成 `_SIZE`，空 family 使用 size 0 + `std::array<Entry,0>`，不引用不存在的 launcher；
- 表按数值 kid 排序并使用 family-specific function pointer，不做 reinterpret cast；
- 删除 `gen_lookup_dict()`、`get_tune_dict()`、临时合并 shape CSV 和
  `opus_gemm_lookup.h` runtime shape 烘焙链；
- 删除旧 `*_tune_lookup.h` 生成；复用旧 blob 时仅删除三个 stale header；
- subset compile 保持：

```text
S = (CSV OPUS kids | compiled-kids sidecar | HEURISTIC_DEFAULT_KIDS)
    & valid kids
    & target arch/filter
    | per-arch mandatory A8 kids
```

- 强制检查 Python heuristic 可返回的 kid 必须在 force-compiled set 中；
- 日志只报告 compile-set 来源，不再报告 baked runtime shape entries。

完成的功能：删除第二套 C++ runtime policy，同时保证 tuned/heuristic/mandatory kid 仍一定可被
编译和 exact dispatch。

### 6.3 `csrc/opus_gemm/codegen/gen_instances_gfx950.py`

Task2 修改位置：A16/A8 host ABI 命名、gfx950 kid 1/2 generated launcher 校验。

修改内容：

- tune/lookup 术语改为 launch/kid-dispatch；
- gfx950 blockscale 的 scale 参数改为必选引用；
- no-scale kid 2 检查 3D shape、batch/M/N/K、FP8 inputs、FP32 Y、K-contiguous/contiguous、
  K-loop minimum/even 和 K-even；
- blockscale kid 1 额外检查 FP32 contiguous scales、同 device、128 grouping、scale shape 和
  prefetch 所需 K-tile 最小/偶数约束；
- 保留既有 kernel/kargs 算法、internal tag mapping 和 Task1 workspace 合同；
- B4 fresh build 时修正 family-only codegen 阻塞，使新 family module 可实际链接和 launch。

完成的功能：把 gfx950 A8 的物理约束放在 exact generated launcher 边界，避免错误输入进入 device
prefetch pipeline。

### 6.4 `csrc/opus_gemm/codegen/gen_instances_gfx942.py`

Task2 修改位置：A16 host ABI 命名和 kid 11000 bpreshuffle launcher。

修改内容：

- bpreshuffle 参数顺序统一为 `XQ,WQ,x_scale,w_scale,Y`；
- scale 改为必选引用；
- generated launcher 检查 2D/3D、batch=1、FP8/BF16/FP32 scale dtype、shape、contiguous、
  exact 128 tile、scale storage 和同 device；
- 使用 family-specific host declaration/manifest/explicit instantiation；
- 删除 generic/tune 过渡说明，保留 kid 11000 现有 kernel 和物理布局。

完成的功能：gfx942 bpreshuffle 的真实物理合同与 canonical ABI 对齐，且内容语义仍要求调用方
提供真正 shuffle 后的 WQ。

### 6.5 `csrc/opus_gemm/codegen/gen_instances_gfx1250.py`

这是 Task1/Task2 混合归属文件。Task2 的范围是：

- 对齐 launch/kid-dispatch emitter 接口和命名；
- 接受中央 generator 提供的 bpreshuffle empty capability 槽位；
- 不增加、不伪造任何 gfx1250 OPUS A8 kernel；
- 证明未来只需新增 registry/tag/emitter 数据即可把 size 0 表变为 size 1，公共 ABI 无需修改。

本文件中的 gfx1250 A16 two-stage/fused workspace、typed workspace 和 reduce 改造属于 Task1，
不应作为 Task2 新功能重复计算。

## 7. 三架构 strict typed dispatch header

### 7.1 修改文件

- `csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh`
- `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh`
- `csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh`

共同修改：

- include 新的 A16/A8 kid-dispatch generated header；
- 分别定义 A16 non-workspace、A16 workspace 和三种 A8 family 的 typed function pointer/entry；
- 使用有序 `std::array<Entry,N>` 和 `find_kid()` 做 exact lookup；
- A16 保留 workspace membership 与 checked workspace dispatch；
- bpreshuffle 为每个 arch 的 BF16/FP32 table 生成 size-aware 路由；
- 空表报告 `no registered kernel`，非空表 miss 报告 `unknown kid`；
- 删除 `OpusA16W16Shape`、runtime shape entry、`find_shape_kid()` 和
  `opus_select_a16w16_kid_gfx*()`；
- 删除 heuristic include/内嵌 heuristic 和只用于选核的 4 GiB policy probe；
- 保留 generated launcher 内的物理 4 GiB 安全检查。

当前最终 workspace row 是 checked-only `{kid, func}`；不存在 prepared/prevalidated function
pointer 或 runtime matcher。

### 7.2 删除文件

- `csrc/opus_gemm/include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh`
- `csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh`
- `csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh`

删除原因：runtime heuristic 已统一由 Python selector 拥有，C++ 不再维护第二份策略。

## 8. 测试和 benchmark 文件

### 8.1 `op_tests/test_opus_interfaces.py`（Task2 新增，后续持续扩展）

覆盖内容：

- Python/C++/pybind 四条 canonical signature 和参数顺序；
- 三个支持 arch 的一致 export、unsupported arch 调用时 stub 和顶层 import 连续性；
- 旧 Python adapter 的参数兼容、单 warning 和 canonical delegation；
- old generic/private tune C++/pybind/raw symbol 已删除；
- A8 raw fake registration 和 `torch.compile(fullgraph=True)`；
- A8 `(arch,family,kid,dtype)` capability matrix、mandatory set 和 empty slot；
- subset compile 公式、arch filter、sidecar、heuristic defaults 和 invalid/off-arch 排除；
- generated table 类型、size、kid-set digest、两次生成字节稳定；
- synthetic gfx950/gfx1250 bpreshuffle table `0 -> 1`，证明公共 ABI 不需要修改；
- gfx950 kid 1/2 数值、shape/dtype/device/single-scale/unknown-kid 负例；
- gfx942 kid 11000 数值和物理合同条件测试；
- CK/CKTile/ASM/FlyDSL/Triton/Gluon 路由不被空 OPUS family 抢占；
- A8 arch cache 按显式 device 隔离、registry cache 只缓存成功查询。

### 8.2 `op_tests/test_opus_dispatch.py`

Task2 修改内容：

- 固定 explicit > tuned > heuristic > fallback 顺序；
- 固定 tuned kid/split_k 原子回退和 gfx942 actual-kid redirect；
- framework fallback 必须真实执行且不得碰 raw binding；
- production fake raw 目标迁移到 canonical A16 raw；
- 增加 C++ runtime shape policy 删除合同；
- 保留 4 GiB physical safety 证据；
- 增加 heuristic kid 必须属于 force-compiled set 的 invariant；
- 删除 generic OPUS probe。

### 8.3 `op_tests/test_opus_workspace.py`

Task2 修改内容：

- 所有 A16 production fake/raw helper 改用 canonical launch 名；
- workspace 表断言改为新 `*_KID_DISPATCH_*` 文件和 macro；
- 继续固定 Task1 actual-kid workspace shape/dtype/capacity/ownership；
- 增加 A8 三个 family 不获得 workspace capability 的合同；
- 保持 raw wrong dtype/size/device、non-workspace `workspace=None` 等严格负例。

该文件中大量 gfx1250 fused/workspace 测试属于 Task1；Task2 只是迁移接口并把这些不变量继续纳入
回归。

### 8.4 `op_tests/test_opus_graph.py`

Task2 修改内容：

- A16 graph capture/replay 和双 stream 改用 canonical raw/public 名；
- 新增 gfx950 A8 no-scale kid 2 和 blockscale kid 1 的真实 graph capture/replay；
- 新增两个 A8 family 的双 stream 数值测试；
- 继续验证调用级 workspace 不共享、无 Python Tensor cache。

### 8.5 `op_tests/test_opus_gfx950_exhaustive.py`

Task2 修改内容：把 direct/public 调用迁移到 canonical A16 launch 和 `kid/split_k` 参数，保留
canonical registry 枚举和 Task1 全 kid oracle。最终同一 fresh module 上验证 gfx950 A16
`140/140`，其中 external-workspace `48/48`。

### 8.6 `op_tests/bench_opus_task1_task2_interfaces.py`（Task2 新增）

完成的功能：提供可复现的 Task1/Task2 接口性能 ABBA 工具。它固定相同输入、actual kid、split-K
和模块，分别记录 public/high-level、compile_ops raw、direct pybind/C++ 和 graph replay，覆盖 A16
及 gfx950 A8 family，定位 Python、C++ 和 kernel 层开销。

### 8.7 `op_tests/bench_opus_gfx950_workspace_ab.py`

Task2 只把 current endpoint 改为 canonical A16 raw；baseline 分支仍有意保留旧
`_opus_gemm_a16w16_tune_raw`，用于加载历史 Task1 checkout，不是当前生产调用方。

### 8.8 执行但不归属 Task2 修改的测试

`op_tests/test_opus_a16w16_gemm.py` 和 `op_tests/test_gemm_codegen.py` 被纳入 Task2 验收，但其当前
dirty/既有实现不因此自动归属 Task2。特别是 mono FP32 修复属于 Task1。

## 9. 用户文档和过程证据

### 9.1 最终用户/开发者文档

| 文件 | Task2 更新内容 |
|---|---|
| `aiter/ops/opus/README.md` | 四条 Python API、capability matrix、Python runtime policy、actual kid、A8 metadata cache、compat wrapper、Task1 workspace 合同 |
| `csrc/opus_gemm/README.md` | 四条 C++ ABI、typed table、subset compile、strict errors、generated physical validation、empty table 和 checked-only workspace |

### 9.2 Task2 计划、步骤和验收记录

| 文件 | 用途 |
|---|---|
| `docs/opus_gemm_two_tasks_final_plan.md` | Task1 冻结边界、Task2 B0--B7 施工顺序、风险和 Definition of Done |
| `docs/task2_step_b0_process.md` / `docs/task2_step_b0_detail.md` | Task1 端点、旧接口、registry/kid-set 和数值 golden |
| `docs/task2_step_b1_process.md` / `docs/task2_step_b1_detail.md` | additive canonical A16 launch |
| `docs/task2_step_b2_process.md` / `docs/task2_step_b2_detail.md` | A16 调用方迁移和旧 C++ tune 删除 |
| `docs/task2_step_b3_process.md` / `docs/task2_step_b3_detail.md` | 三条 A8 family、capability matrix、typed codegen 和路由 |
| `docs/task2_step_b4_process.md` / `docs/task2_step_b4_detail.md` | generic mega entry 删除和 fresh build |
| `docs/task2_step_b5_process.md` / `docs/task2_step_b5_detail.md` | C++ runtime shape lookup/heuristic 删除和 compile-set 保留 |
| `docs/task2_step_b6_process.md` / `docs/task2_step_b6_detail.md` | family 校验、调用方/README 收尾；顶部勘误说明 prepared 实验已回退 |
| `docs/task2_step_b7_process.md` | static、codegen、CPU、GPU、ABI 和性能的完整执行记录 |
| `docs/task2_checkpoint.md` | 当前权威短状态、checked-only 勘误、最终回归和 no-scale 优化结果 |

`docs/opus_gemm_two_tasks_final_plan.html` 是计划的渲染快照；后续状态应以 Markdown 计划、B7 记录
和 checkpoint 为准。

## 10. 当前 dirty 工作树中不应整体归入 Task2 的文件

以下路径出现在当前累计 `git diff` 中，但主要属于 Task1：

- `aiter/ops/opus/_workspace.py`、`aiter/ops/opus/_workspace_a16w16.py` 的删除；
- `csrc/opus_gemm/codegen/common.py` 的 typed workspace 基础设施；
- gfx1250 two-stage/fused pipeline、traits、reduce 和新增 fused header；
- gfx950 mono-tile/4G-safe pipeline 和 FP32 store 修复；
- `op_tests/test_opus_a16w16_gemm.py` 的 mono/数值改动；
- `docs/task1_*`、`docs/opus_gemm_splitk_workspace_torch_current_flow_changes.*`；
- Task1 workspace benchmark 本身。

其中 `_selector_a16w16.py`、`gen_instances_gfx1250.py`、gfx950 traits 和部分 workspace 测试属于
混合文件，本文件前文已经只列出 Task2 的窄修改面。

`docs/opus_gemm_next_performance_optimization_plan.md` 是后续独立性能方案，不属于已经实施的 Task2
功能，也没有在 B7 中被修改。

## 11. 验收结果与仍开放边界

| 验收项 | 当前结果 |
|---|---|
| 静态检查 | `git diff --check`、目标 `py_compile` 和旧符号扫描通过 |
| 三架构 codegen | 双 fresh、字节稳定、kid-set golden 和 arch header 通过 |
| generated 编译 | `3516/3516` TU syntax 和 `3516/3516` 对象编译通过 |
| CPU/当前综合回归 | `239 passed, 22 skipped`；skip 均为缺少 gfx942/gfx1250 硬件项 |
| gfx950 A8 | kid 1/2 数值、负例、graph、双 stream 通过 |
| gfx950 A16 | canonical `140/140`，external-workspace `48/48` 通过 |
| ABI | gfx942/gfx950/gfx1250 独立 module 均只暴露四条 canonical 业务 raw；旧 C++/pybind 符号缺失 |
| 性能 | A16、blockscale、raw/direct/graph 无 kernel 回退；no-scale public 从 `+20.591%` 缩小到 `+4.835%` |

仍开放的边界：

- 当前节点没有 gfx942/gfx1250 实机，因此对应数值、graph、并发和性能仍是“未执行”；
- 目标节点的完整执行顺序、现有自动化与必须补齐的 GPU test/performance 门见
  `docs/gfx942_gfx1250_validation_runbook.md`；
- gfx950 no-scale public 相对 Task1 仍有 `+0.644 us / +4.835%` 的 adapter 差异，来自保留的动态
  Tensor 安全检查和热 metadata cache 查询，不能写成完全消失；
- 当前没有恢复 prepared launcher/workspace cache，也不应为追求该差异恢复此前无端到端收益的
  A16 prepared 实验。

## 12. 最终结论

Task2 已完成的本质不是新增一批 kernel，而是完成 OPUS GEMM 控制面的重构：

- Python 唯一拥有 runtime 选核 policy；
- C++/generated code 只执行 family-scoped、dtype-scoped、exact-kid launch；
- 四条 canonical ABI 在 Python、pybind、C++ 和三架构 generated table 上一致；
- A8 三种物理合同彼此隔离，空 capability 也有稳定 ABI 和明确错误；
- build-time compile reachability、Task1 workspace、graph 和 kernel 物理安全合同均被保留；
- 所有生产调用方已迁移，旧 C++ ABI 已清理，旧 Python 名只作为短期兼容 adapter。
