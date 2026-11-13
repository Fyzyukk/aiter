# 不 shuffle 版（m8 路线）参数化 + 边界处理 —— 实验记录（2026-11-13）

分支 `fyz/opus_mxfp4_nt_gemm_noshuffle`（基于 shuffle 分支 983eaaf）。起点：candidate_m8（GPU2 约 217.9 µs，这条线目前最好）。

## 做法
不从 m8 模板重新参数化，而是以已经参数化、带边界处理的 shuffle 版 pipeline 为底，换回 m8 与 m10 的两处差异
（m8 → m10 的模板差异只有 145 行：m9 改 scale 路径、m10 改 B 布局）。用脚本 `gen_noshuffle_pipeline.py` 生成，
每处替换都断言匹配次数：

| 部分 | shuffle 版 | 不 shuffle 版（m8） |
|---|---|---|
| B global / LDS 布局 | shuffle_weight 的 1 KB 块原样搬 | 行主序，与 A 对称：chunk = (wave_m·2 + wave_n)·insts + i，16 B XOR swizzle；`make_layout_gb/sb/rb`、`dma_odd_b`（E = 3）、`gb_offset`、`load_b` 的 kk 偏移 |
| scale 全局读 | e8m0_shuffle panel，LDS-DMA | 写 wave（A：h = wave_n、w = wave_m）lane 4r + e 读第 `h·HALF_B + e·32 + w·16 + r` 行的 16 B（2 个 K tile），进 VGPR |
| scale 写 LDS | DMA 原样 | 4×4 字节 DPP 转置后写两份 8 B（m8 的 put_sf，slot XOR swizzle） |
| scale 读 | ds_read2st64 + 64 位移位 + opsel 2kk | 每半区一条 ds_read_b64，字节 e 即 repeat e，opsel = m / n，无 VALU |
| A1 / B1 预取位置 | PF_SPREAD（C11 / C00） | m8：C01（PF_SPREAD 在 m8 上实测更慢） |
| C11 调度 | DMA 槽 | m8 的 sched_c11（转置 VALU + e1：panel ds_write 后接下一 panel 的 buffer_load） |
| prologue | SF_DMA2 计数等待 | m8：panel 0 / panel 1 sfb 进 VGPR，第一个 barrier 后转置写入；`vmcnt(2(a+b)+1)` |
| 最后一个 tile 预取下一 tile 的 panel 0 | DMA | VGPR load，偏移用不透明 lane id 现算（否则 256×256 有 2 个 VGPR spill） |

scale 路径和 tile 尺寸基本无关：每个写 wave 固定负责 64 个 lane（4 个 32 行块 × 16 行），转置后字节 e = repeat e；
E < 4 时多出来的字节不会被 MFMA 选中。LDS 中 scale 镜像固定 4 KB / panel（2 个 K tile × 4 wave × 64 slot × 8 B），3 个 buffer × 2 = 24 KB。
保留不变：参数化（E_M / E_N、DMA 条数、奇数 DMA、C_COPIES / C_SWZ、调度 sched_spread、vmcnt 公式）、边界处理、
persistent / 非 persistent、ZC_PEEL、C 写回指令序列。split-K 暂不支持（traits 断言 MODE ∈ {0, 1}）。

## 资源（pin 工具链，dev 构建与 .co 一致）

| tile | VGPR / AGPR | VGPR spill | SGPR spill | LDS |
|---|---|---|---|---|
| 256×256（persistent / np / nob） | 512 / 256 | 0 | 10–12 | 160 KB |
| 256×128、128×256 | 448 / 224 | 0 | 8–11 | 126 KB |
| 128×128 | 416 / 208 | 0 | 6–7 | 92 KB |
| 192×256、256×192 | 480 / 240 | 0 | 11–17 | 143 KB |

## 与 m8 的代码对比（256×256 persistent，指令条数）

| | m8 | 本版（无边界） | 本版（边界） |
|---|---|---|---|
| 总数 | 2986 | 3316 | 3333 |
| MFMA | 640 | 896 | 896 |
| ds_read | 202 | 270 | 270 |
| buffer_load | 106 | 140 | 140 |
| v_mov / accvgpr | 578 | 400 | 400 |
| VALU（其余） | 774 | 847 | 864 |

差异可解释：ZC_PEEL 单独实例化第一趟 main loop（+256 MFMA、+68 ds_read、+34 buffer_load、+4 barrier），
同时省掉每 tile 192 条累加器清零（v_mov / accvgpr −178）；边界处理多 17 条 VALU。

## 正确性
- 104 用例矩阵（6 个原形状 + M ∈ {1,16,100,300,1000} × N ∈ {128,400,1280,4112}，K 轮换 256/768/7168，batch 1/2；
  默认 / MXFP4_SFA_K_PATTERN=1 / PERSIST_GRID=37 / PERSIST_GRID=1），scale 每个 batch 后加 256 行 0xFF（NaN）守护：结果见下。
- 13 个 .co 通过 `MXFP4_CO_DIR` 加载：6 tile × {8192³ persistent、300×400×768×b2 每 block 一个 tile、1000×4112×7168 persistent}
  + 256×256 无边界 8192³（均带 MXFP4_SFA_K_PATTERN=1）：19/19 VALID。

### 结果（HIP 2，dev 构建）
| tile | 104 用例矩阵 |
|---|---|
| 256×256 | 104/104 |
| 256×128 | 104/104 |
| 128×256 | 104/104 |
| 128×128 | 104/104 |
| 192×256 | 104/104 |
| 256×192 | 104/104 |

## 计时
未完成：HIP 2（smi GPU[2]）在整个工作期间被其他进程占用（use 100%、CU_OCCUPANCY > 0）。`prof/timing.sh 2 2` 会等卡连续 3 次空闲后，
与 candidate_m8 交替计时 8192³（含 256×256 无边界变体），再测 6 种 tile 的中等 shape；结果写到 /tmp/opus_mxfp4_ns/timing.txt。

## 合并进 shuffle 分支时要改的共享文件
- `opus_gemm_common.py`：新 kernel_tag `a4w4_mxfp4_nt_noshuffle`，kid 30100–30105（persistent）、30110–30115（np）、30130（256×256 nob）。
- `codegen/gen_instances_gfx950.py`：launcher 的 stride 改行主序（stride_sfa / stride_sfb = K/32，batch stride = rows × K/32），检查 N % 8、K % 256。
- `gen_co/co_kernels_gfx950.json`：并入 `mxfp4_nt_noshuffle_dev/gen_co/co_kernels_gfx950_noshuffle.json` 的 13 条；.co 移到 `gen_co/gfx950/`。
- `opus_gemm.cu` / pybind / `opus_gemm.h`：fp4x2 分支按 bpreshuffle（或 scale 布局）选 shuffle / noshuffle 的分发；输入检查分两套。
- `aiter/ops/opus/gemm_op_a4w4.py`：`bpreshuffle` 参数（与 aiter `gemm_a4w4` 一致），不 shuffle 时检查 scale 为 `[M, K/32]` / `[N, K/32]`。
- `op_tests/test_opus_a4w4_gemm.py`：不 shuffle 的用例（输入直接用 quant 输出，不做 shuffle）。
- 若 shuffle 版 pipeline 后续改动（例如加 split-K），重新运行 `gen_noshuffle_pipeline.py`；替换点失配时脚本会断言报错。
