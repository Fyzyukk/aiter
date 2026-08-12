# OPUS GEMM 下一步性能优化计划

记录时间：`2026-08-11 UTC`；实施与最终验收：`2026-08-12 UTC`

状态：**Phase 1已经实施、验收并采用。** 正式a16w16 public/high-level路径已经从pybind raw
切换到C ABI/ctypes raw；原pybind raw继续保留为兼容和A/B端点。Phase 2A/2B没有启动。

## 0. 最终结果摘要

本轮没有修改device kernel、workspace布局、kid集合或generated traits。C ABI仍调用原有
`opus_gemm_a16w16_launch` checked launcher，不复制dispatch或validator。最终gfx950结果为：

| 同边界比较 | pybind A | ctypes B | 变化 | 平均每项变化 |
|---|---:|---:|---:|---:|
| raw eager，96项配对总和 | `1644.093507 us` | `1372.621492 us` | `-16.511957%` | `-2.827833 us` |
| raw graph replay，96项配对总和 | `1109.366086 us` | `1107.945324 us` | `-0.128070%` | `-0.014800 us` |
| public eager，96项配对总和 | `2487.414976 us` | `2205.882312 us` | `-11.318283%` | `-2.932632 us` |
| public graph replay，96项配对总和 | `1109.186787 us` | `1111.668675 us` | `+0.223757%` | `+0.025853 us` |

raw eager有`94/96`项更快；public eager有`95/96`项更快。graph测试只计相同captured kernel的
replay，不会在每次replay重新执行Python或C ABI，因此上述`-0.128%/+0.224%`属于轮次噪声，
没有可归因的graph回退。Phase 1超过“至少追回约`1 us`”的采用阈值，因此不进入Phase 2。

采用前的双设备补充测试发现：只设置thread-local stream而不切换HIP current device时，
“当前device 0、Tensor在device 1”的隔离进程会发生memory fault。最终实现已改为成对切换并
恢复HIP device和thread-local stream，并在共同checked launcher中先验证XQ/WQ/Y、optional
bias和workspace的device合同。修复后的device 1数值、current-device恢复和mixed-device
launch前拒绝均通过。

## 1. 当前基线

已经回退的实验内容：

- gfx950 `_prevalidated` launcher；
- `<kernel>_impl<Validate, D_C>`双wrapper；
- thread-local prepared workspace合同；
- 合并`has_workspace(kid)`和`workspace_dispatch(kid)`的单次查询。

当前行为：

- generated workspace row为`{kid, func}`；
- 三架构workspace entry都只有一个function pointer；
- runtime先调用`has_workspace(kid)`，命中后再调用`workspace_dispatch(kid)`；
- gfx950 workspace launcher每次执行完整checked validator。

以下内容继续保留，不属于待回退范围：Torch-owned workspace、gfx950 mono-tile FP32修复、
graph/stream/lifetime合同、gfx1250 fused family以及任务二的接口/dispatch重构。

回退后fresh gfx950 focused suite结果为：

```text
218 passed, 23 skipped, 0 failed
```

## 2. 已知性能结论

此前第2/3项实验的分层结果：

| 测量边界 | 修改前 | prepared实验 | 变化 |
|---|---:|---:|---:|
| 隔离pybind/C++，kid 200 | `5.514345 us` | `5.145200 us` | `-6.694%` |
| 正常Torch raw eager，96项配对总和 | `1600.927 us` | `1601.621 us` | `+0.043%` |
| graph replay，96项配对总和 | `1111.840 us` | `1111.310 us` | `-0.048%` |

结论：跳过C++检查和合并一次查表虽然能在隔离C++边界节省约`0.369 us`，但正常Torch
端到端没有可测收益。下一轮不应恢复第2/3项，而应优化Torch/custom-op/pybind边界。

### 2026-08-11只读转换微基准

代表输入为kid 200常用的四个device Tensor：XQ、WQ、Y和workspace。每种方式warmup后运行
`10000`次、重复`7`轮，报告每次转换四个Tensor的median：

| 转换方式 | median | 相对当前pybind |
|---|---:|---:|
| `torch_to_aiter_pybind` | `6.107 us` | 基线 |
| `torch_to_aiter`（ctypes struct） | `3.347 us` | 约省`2.760 us` |
| 只读取四个`data_ptr()` | `0.283 us` | 仅作下界参考 |

这个微基准只测转换，不等价于完整launch；但ctypes的潜在节省已经大于当前待追回的约
`1.5--2 us` eager host差距，因此应先做低侵入的C ABI/ctypes端到端原型。

## 3. 推荐实施顺序

### Phase 1：并行C ABI/ctypes实验入口（优先）

当前路径：

```text
Torch Tensor
  -> torch.ops
  -> Python读取每个Tensor的metadata
  -> 创建pybind aiter_tensor_t
  -> C++ checked launcher
```

目标路径：

```text
Torch Tensor
  -> torch.ops
  -> ctypes aiter_tensor_t
  -> C ABI（显式传当前HIP stream）
  -> 原有C++ checked launcher
```

设计要求：

1. 保持公开Python API和workspace所有权不变。
2. 先增加一个并行实验symbol，不立即删除现有pybind入口，便于同源码A/B。
3. C ABI接收XQ/WQ/Y、optional bias、optional workspace对应的`aiter_tensor_t*`，以及
   `kid`、`split_k`和当前`hipStream_t`。
4. C ABI入口设置当前thread-local HIP stream，然后调用现有
   `opus_gemm_a16w16_launch`；不复制kernel dispatch或workspace validator。
5. 使用现有ctypes异常桥，把C++异常转换为状态码和thread-local错误字符串，不能让异常穿过
   C ABI。
6. Torch custom-op schema继续保留所有Tensor参数。不能把公开/custom-op边界改成只有整数
   pointer，否则Torch无法正确追踪Tensor生命周期、读写、alias和graph依赖。
7. 不增加全局Tensor cache，不保存Tensor对象或历史data pointer。
8. pybind和ctypes共用同一个模块时，必须先验证JIT首次构建、`.so`加载和现有a8接口不会因
   `torch_exclude`/Python-module构建模式发生冲突；必要时使用独立的薄C ABI TU或明确的混合
   build配置。

Phase 1的目标是验证“去掉pybind对象构造”能否在正常Torch raw端到端稳定追回差距，而不是
提前引入prepared状态。

### Phase 2A：native Torch C++ Tensor边界（ctypes不足时首选）

如果Phase 1仍不能满足目标，增加一个薄的native Torch C++ operator：

```text
torch.ops
  -> C++ at::Tensor参数
  -> C++栈上构造轻量aiter_tensor_t/raw launch view
  -> 现有checked launcher
```

该TU可以单独包含Torch/ATen header，kernel、generated host TU和公共launcher继续保持
torch-free。优点是Tensor metadata转换全部在C++完成，同时Torch dispatcher天然持有Tensor
引用和alias信息。需要单独验证编译时间、首次JIT大小、torch.compile fake/meta注册以及当前
HIP stream获取方式。

### Phase 2B：prepared descriptor/pointer ABI（最后才做）

只有Phase 1/2A profile仍证明metadata或validator是主要剩余开销时，才引入prepared
descriptor。

prepare阶段只保存不可变合同：

```text
kid / checked或专用function pointer
batch / M / N / K / effective split-K
workspace dtype / required numel / alignment
必要的tile和stride标量
```

每次launch仍接收当前XQ/WQ/Y/bias/workspace Tensor，并读取当前data pointer和stream。
descriptor不得保存Tensor、storage ownership或历史XQ/WQ/Y/workspace data pointer。compact
guard失败时必须回到checked路径或明确报错，不能静默继续。

新的ABI内部可以重新使用“预验证launcher”的概念，但不依赖、也不应直接恢复已回退的
thread-local prepared cache。

## 4. 明确不采用的方法

- 不直接恢复性能实验第2/3项；
- 不通过继续删除`AITER_CHECK`解决Torch边界开销；
- 不把workspace改回C++内部allocator；
- 不建立全局Torch Tensor或pybind对象cache；
- 不以更换workspace物理dtype作为host优化；
- 不在Torch custom-op schema中只传整数pointer；
- 不以牺牲CUDA/HIP graph、当前stream或多stream并发语义换取微小收益。

## 5. Phase 1实际修改边界

- `aiter/jit/core.py`
  - ctypes loader增加`force_torch_exclude`控制，允许C symbol和pybind共用同一个混合Python
    `.so`；默认行为仍是torch-free；
  - `compile_ops()`增加`ctypes_force_torch_exclude`；
  - 恢复ctypes dispatcher的原始Python signature。
- `aiter/ops/opus/gemm_op_a16w16.py`
  - 增加私有`_opus_gemm_a16w16_launch_ctypes_raw`；
  - public explicit、deprecated adapter和shape-driven high-level最终都使用该raw；
  - 原`_opus_gemm_a16w16_launch_raw` pybind入口继续存在，只用于兼容、测试和A/B。
- `csrc/opus_gemm/include/opus_gemm.h`
  - 声明`opus_gemm_a16w16_launch_cabi(...)`；optional Tensor用空指针，kid/split_k为
    `int64_t`，stream显式传入。
- `csrc/opus_gemm/opus_gemm.cu`
  - 接入现有TLS异常桥，C++异常不会越过C ABI；
  - C ABI检查整数范围后转调原canonical checked launcher；
  - `OpusCabiDeviceStreamGuard`切换/恢复HIP device和thread-local stream；
  - canonical launcher补齐输入、bias和workspace同设备检查。
- `op_tests/test_opus_ctypes.py`
  - 覆盖ABI形状、fake/`torch.compile`、BF16/FP32 parity、五种workspace错误、非默认
    stream、graph、双stream、跨current-device成功与mixed-device安全拒绝。
- `op_tests/test_opus_interfaces.py`、`test_opus_dispatch.py`、`test_opus_workspace.py`、
  `test_opus_graph.py`
  - production mock/hook改为跟随最终ctypes后端，既有合同继续回归。
- `op_tests/bench_opus_gfx950_workspace_ab.py`
  - 增加ctypes、public-pybind和最终public端点，可分别做raw与public同边界ABBA。
- `op_tests/bench_opus_task1_task2_interfaces.py`
  - 当前端点跟随ctypes；冻结Task1模块仍可注入旧pybind raw做历史对照。

没有修改`aiter/jit/optCompilerConfig.json`，也没有修改任何gfx950/gfx942/gfx1250 device
kernel、workspace物理布局、registry、codegen kid集合或generated traits。

## 6. 最终验证

### 6.1 fresh构建与ABI

最终full-142-kid gfx950目录：

```text
/tmp/aiter-opus-ctypes-final.MINoRH
```

compiled-kids sidecar SHA-256：

```text
b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
```

fresh JIT完成生成、编译、链接和public数值启动。最终`.so`导出：

```text
aiter_ctypes_abi_version
aiter_get_last_error
aiter_clear_last_error
opus_gemm_a16w16_launch_cabi
```

### 6.2 正确性、graph、stream和lifetime

- ctypes定向组：`15 passed`；
- focused组（dispatch/workspace/graph/a16w16/interfaces/ctypes）：
  `254 passed, 22 skipped`；
- gfx950 canonical全量public sweep：`140 passed`，覆盖92个non-workspace和48个workspace
  kid，各自BF16/FP32输出，以及workspace复用/自动分配合同；
- 22个skip仍是gfx942/gfx1250硬件条件项，不是通过结论；
- gfx942/gfx950/gfx1250 fresh默认32-kid subset生成在
  `/tmp/aiter-opus-ctypes-multiarch.kRKLyS`；三个`all_instances_host_<arch>.cu`分别通过目标
  `hipcc -fsyntax-only`，包含三架构router和最终device/stream guard的`opus_gemm.cu`也通过；
- `py_compile`与定向`git diff --check`通过。

gfx942/gfx1250仍只有fresh codegen和host syntax结论；实机数值、graph、并发和性能继续按
`docs/gfx942_gfx1250_validation_runbook.md`执行，不能写成已通过。

### 6.3 最终性能方法

最终日志目录：

```text
/tmp/aiter-opus-ctypes-final-perf.nFPg1D
```

设备为gfx950 `AMD Instinct MI355X`。raw和public各自执行
`A1 -> B1 -> B2 -> A2`，覆盖48个workspace kid乘BF16/FP32共96项；每项
`20 warmup + 9 rounds x 100 launches`。每轮前GPU占用记录为`0--1%`，每个case先做数值
断言。A/B值均为同一case两轮median的平均后再求和。

raw边界：

| 项目 | pybind A | ctypes B | 变化 |
|---|---:|---:|---:|
| eager全部 | `1644.093507 us` | `1372.621492 us` | `-16.511957%` |
| eager BF16 | `821.327029 us` | `686.094224 us` | `-16.465159%` |
| eager FP32 | `822.766478 us` | `686.527268 us` | `-16.558673%` |
| graph全部 | `1109.366086 us` | `1107.945324 us` | `-0.128070%` |

public边界使用完全相同的selector/workspace helper，只把末端raw在pybind和ctypes之间切换：

| 项目 | public-pybind A | final public ctypes B | 变化 |
|---|---:|---:|---:|
| eager全部 | `2487.414976 us` | `2205.882312 us` | `-11.318283%` |
| eager BF16 | `1242.759873 us` | `1106.718597 us` | `-10.946706%` |
| eager FP32 | `1244.655104 us` | `1099.163715 us` | `-11.689294%` |
| graph全部 | `1109.186787 us` | `1111.668675 us` | `+0.223757%` |

最终日志SHA-256：

```text
d08e6439b2316523602e511ad89af8f7383329d5075f26e4f8a6f89fd72d5649  perf_raw_A1.log
0f6bc2bfdde0bc780c95687dce96b375b37d487ae51513062eb13463ba1fba19  perf_raw_A2.log
b957bcd61def8dca8a1dee1a853c887680a6bfacb3f6f07b65f699bcbabd47e8  perf_raw_B1.log
c19ddb43e1be11c298facaca566825472ecab771c5127b49f9346e684d9e2af1  perf_raw_B2.log
600002f661efe5e83d9d662aa00ed1c8ebba7925c95de1e073d2c9f67adf74b7  perf_public_A1.log
b66a5837948c92a72b3b5646edd349a340dfc40a99b4d6be3a5619ed04660f79  perf_public_A2.log
fe9fc20da48a3f8d8f75bb46825c888797257779599c30c7d5dc46befb0141b9  perf_public_B1.log
051c930f7aa43f111db20dc642d8d18243fc95e594a32fc097b18978e117b0b7  perf_public_B2.log
```

## 7. 最终采用状态

1. Phase 1的eager收益稳定且超过阈值，正式public/high-level路径采用ctypes。
2. pybind raw不删除，保留兼容、故障隔离和可重复A/B能力。
3. Phase 2A native Torch C++ Tensor边界不启动；当前没有继续扩大实现面的收益理由。
4. Phase 2B prepared descriptor/pointer ABI不启动；此前prepared/prevalidated实验仍保持回退。
5. 不缓存Tensor、data pointer、stream、workspace或launcher，不删除generated安全检查。
6. 若未来继续优化，应先profile public路径剩余的Python selector/workspace规划成本，建立新的
   同边界ABBA；不得把本轮ctypes收益当作恢复prepared状态的依据。

## 8. 一句话结论

Phase 1已经安全落地：最终public eager在gfx950全96项同边界比较中改善`11.318%`、平均每项
追回`2.933 us`，graph无可归因回退，因此保留checked validator并停止在Phase 1。
