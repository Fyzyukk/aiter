# 9020 B gmem 构造风格整理（2026-09-28）

生产 9020 的 B 全局视图改为与 9000 相同的直接 typed-pointer 写法：

```cpp
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
```

删除 `b_address`、`b_lo`、`b_hi` 及源码中的两次指针 `readfirstlane`。此前这段处理是为避免编译器生成 descriptor waterfall。当前版本直接构造指针的编译结果没有此问题；ISA 中仍由编译器生成必要的 `v_readfirstlane_b32`（总数仍为 3），无需在源码中手工拆分地址。

`D_B` 是单字节 FP8；`col` 由 workgroup ID 和统一的 kernel 参数计算，在所有 lane 间相同。生产入口的 9020 registry 限制 `max_tensor_bytes=2**31-1`，codegen 在维度窄化前检查 `N*K <= INT32_MAX`，并设 `stride_b=K`，因此支持范围内 `col*stride_b` 的 int 乘法不溢出，最终地址等价。

A/C 的显式资源长度保留，用于 M 尾块；SFA/SFB 视图及 global views 之后的完整 pipeline 原文不变。9000 pipeline/traits 与 9020 traits 哈希不变。

仅做一次 CPU 编译，复用上一轮已核实源码与依赖哈希的基线二进制；没有运行 GPU、数值测试、benchmark 或 tune。

| 项目 | 改前 | 改后 |
| --- | ---: | ---: |
| VGPR | 204 | 204 |
| SGPR | 52 | 51 |
| LDS bytes | 143360 | 143360 |
| Private / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| 指令字节 | 7808 | 7812 |
| v_readfirstlane_b32 | 3 | 3 |
| v_cmp_eq / s_cbranch_execnz | 0 / 0 | 0 / 0 |

内存指令计数与 kernel ABI 一致，机器码不逐字节相同；此检查不构成性能相同或更快的实测结论。

- [源码差异](change.diff)
- [源码及边界检查记录](validation.json)
- [编译及 ISA 比较](device_comparison.json)
- [构建记录](direct/build_manifest.json)
- [改后 ISA](direct/device_isa.txt)
