# 9020 下半段 pipeline 风格整理（2026-09-28）

仅修改 9020 pipeline，从 C 输出布局到最终写回按 9000 的命名和组织方式整理，保留 9020 本身的八 wave 几何、整块 scale 一次装入、SFA 后打包、SFB 两 byte 打包、逐片 MFMA/LDS 交错及 M 尾处理。

## C 输出

- `p_coord` 改为 `p_coord_c`，`u_sc/sc_offsets` 改为 `u_gc/gc_offsets`。
- `u_gc` 使用原来的 `C_LDS_ROW_STRIDE_ELEMS`，实际描述 BF16 在 LDS 的暂存排布，沿用 9000 的名称。
- `gc_offsets` 是按 MFMA fragment 列出的 LDS 地址表，写入端直接使用。
- `c_offset(row_c,col_c)` 集中表达 `row_c*C_LDS_ROW_STRIDE_ELEMS+col_c`，供最终合作写回时读取 LDS。9020 layout 已覆盖完整单 tile，未添加 9000 的 half-tile 偏移或重复的 M/N 偏移。
- `stage_output_fragment` 封装 FP32 slice、BF16 cast 和 LDS store，仍在每次最终 MFMA 后立即调用。`copy_output_bf16(copy_index)` 使用相同遍历顺序及原 M 尾判断。

## Scale 与其余下半段

四个 scale offset 与 A/B offset 集中声明，并用于实际读写：

| offset | 公式 | 单位/用途 |
| --- | --- | --- |
| `gsfa_offset(panel_k_begin)` | `panel_k_begin*stride_sfa` | global SFA byte |
| `gsfb_offset(panel_k_begin)` | `panel_k_begin` | global SFB byte |
| `ssfa_offset(tile_k)` | `tile_k*B_M` | LDS SFA byte |
| `ssfb_offset(tile_k)` | `tile_k` | LDS SFB packed uint word |

原 `load_scale_panel` 按原执行顺序拆为 `load_sfa_panel(0)`、`load_sfb_panel(0)`；二者只调用一次，没有新增 refill、mask 或额外同步。SFA 每 pass 仍是 global load 后存原始 byte，read_scales 再按三个 M repeat 打包。SFB 仍读取两个 N128 scale byte，合为一个 uint。

cached layout 通过标准 `layout_to_offsets` 提取标量 offset，保留原 load/store overload；A/B 读片统一为 `load_a_fragment/load_b_fragment`。K 坐标统一 `tile_k`，M/N 片段统一 `m_repeat/n_repeat`，删除无用 `group=tile`。主循环步长直接引用既有 `T::LOOP_UNROLL=2`。

源码审查确认：`GROUP_K=B_K=128`，新 scale group 数与旧 loops 相同；所有 global/LDS 地址在现有调用下等价。显式 wait/barrier/sched_barrier 的顺序与参数逐项一致。9000、其他候选、traits、Opus、registry 和入口代码均未改。

## 验证

一次 CPU 编译通过，复用已核实的上一轮基线；没有运行 GPU、数值测试、benchmark 或 tune。

| 项目 | 改前 | 改后 |
| --- | ---: | ---: |
| AGPR | 0 | 0 |
| VGPR | 203 | 203 |
| SGPR | 54 | 54 |
| LDS bytes | 143360 | 143360 |
| Private / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| 指令字节 | 7896 | 7896 |
| 静态 MFMA 指令数 | 72 | 72 |

完整 kernel metadata（包括 ABI）一致，内存/MFMA/wait/barrier 指令计数一致，无 descriptor waterfall。机器码不逐字节相同，不据此宣称实测性能相同。

- [源码差异](change.diff)
- [源码及同步检查](validation.json)
- [编译资源及指令比较](device_comparison.json)
- [构建记录](after/build_manifest.json)
