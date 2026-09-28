# 9020 五个 gmem 基址补齐 batch 偏移（2026-09-28）

按用户要求，仅在生产 9020 pipeline 中声明 `const int batch_id = block_id_z();`，并给 A、B、C、SFA、SFB 的 typed-pointer 基址加上对应的 `batch_id * kargs.stride_*_batch`。保留全部 row/col 偏移、A/C resource bounds、layout、scale packing、预取和输出顺序。9000、其他候选、traits、registry 和入口代码未改。

| 基址 | 新增偏移 | 单位 |
| --- | --- | --- |
| `g_a` | `batch_id * kargs.stride_a_batch` | FP8 元素，1 byte |
| `g_b` | `batch_id * kargs.stride_b_batch` | FP8 元素，1 byte |
| `g_c` | `batch_id * kargs.stride_c_batch` | BF16 元素，2 bytes |
| `g_sfa` | `batch_id * kargs.stride_sfa_batch` | E8M0 byte |
| `g_sfb` | `batch_id * kargs.stride_sfb_batch` | E8M0 byte |

当前入口限制 batch=1，grid.z=1，因此本次五个新增项在实际启动中均为 0。补齐表达式不会使入口自动支持多 batch。A/C 的资源长度相对于已经移动的基址描述当前 batch 内剩余范围，不需额外加入 batch 偏移；global SFB 偏移按 byte 计算，与 LDS 内 packed uint 的单位区分。

复用上一轮直接 B 指针版本的已核实基线，仅进行一次 CPU 编译并核对 ISA，没有运行 GPU、数值测试、benchmark 或 tune。

| 项目 | 改前 | 改后 |
| --- | ---: | ---: |
| VGPR | 204 | 204 |
| SGPR | 51 | 54 |
| LDS bytes | 143360 | 143360 |
| Private / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| 指令字节 | 7812 | 7896 |

内存指令计数和 kernel ABI 一致，未生成 descriptor waterfall。由于编译器不能在这个独立 kernel 中将 block_id_z 恒定折叠成 0，机器码保留 batch 地址计算；源码地址在当前启动方式下等价，不代表机器码或实测性能完全相同。

- [源码差异](change.diff)
- [源码及 ISA 检查记录](validation.json)
- [资源及指令比较](device_comparison.json)
- [构建记录](after/build_manifest.json)
- [改后 ISA](after/device_isa.txt)
