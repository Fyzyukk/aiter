# 9020 使用默认 MMA 寄存器类型（2026-09-28）

生产 9020 改为直接声明 `typename decltype(mma)::vtype_a v_a`、`vtype_b v_b`、`vtype_c v_c`，用 `clear(v_c)` 清零。删除 `BaseMMA/AFragment/BFragment/AccFragment/AChunk/BChunk/CTile` 别名、自建寄存器数组、`c00` 名称和 `a_chunks/b_chunks` 裸指针。

9020 原实现没有 pin AGPR；旧 array 声明只描述分片容器。当前 E_M=3、E_N=8、E_K=1，LDS 双缓冲配合逐片用完即替换，只需一份完整的 A/B/C 寄存器 tile，无需新增示例中的 A[2]、C[2][2] 或 v_mma[2]。

当前 Opus 默认 `OPUS_TILE_CONTAINER=0`。逻辑 A 为 96 个 FP8、B 为 256 个 FP8、C 为 96 个 FP32。按 Opus 自身 tiled-MMA 的方式，以编译期 `slice/set_slice` 表达分片：

| 对象 | 旧 fragment 索引 | 新元素区间 |
| --- | --- | --- |
| A | `v_a[mi]`，每片 32 FP8 | `[mi*32, (mi+1)*32)` |
| B | `v_b[ni]`，每片 32 FP8 | `[ni*32, (ni+1)*32)` |
| C | `c00[mi*8+ni]`，每片 4 FP32 | `[(mi*8+ni)*4, (mi*8+ni+1)*4)` |

两次 16 字节 LDS load 仍按原来的 fragment/chunk 顺序写入 A/B。每次单片计算仍调用 `decltype(mma)::MMA`（保留 swap_ab adaptor），Scale A/B selector 与原代码相同；最终逐片 BF16 store 从对应的四个 FP32 元素取值。gmem、layout、traits、入口、scale 搬运、advance/prologue/main-loop 和 output writeback 均保持；只调整寄存器容器和对应读写表达式。

一次 CPU 编译通过，复用已核实的上一轮基线。没有运行 GPU、数值测试、benchmark 或 tune。

| 项目 | 改前 | 改后 |
| --- | ---: | ---: |
| AGPR | 0 | 0 |
| VGPR | 204 | 203 |
| SGPR | 54 | 54 |
| LDS bytes | 143360 | 143360 |
| Private / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| 指令字节 | 7896 | 7896 |
| 静态 MFMA 指令数 | 72 | 72 |

内存、MFMA、wait/barrier 指令计数及 kernel ABI 一致，无 descriptor waterfall；机器码不逐字节相同。寄存器计数减少不代表已实测性能更快。9020 仍通过包含 9000 header 复用 layout helpers，本次没有改变这条包含关系或移除共享 header 的工具链要求。

- [源码差异](change.diff)
- [源码及映射检查](validation.json)
- [编译资源及指令比较](device_comparison.json)
- [构建记录](after/build_manifest.json)
