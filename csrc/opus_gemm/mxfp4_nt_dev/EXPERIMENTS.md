# work_tiles —— 2×2 流水线多 tile 参数化 + M/N 边界处理（线 1，2026-11-13）

基础：work_m10 当前最好（SF_DMA SF_DMA2 SF_SHIFT PF_SPREAD ZC_PEEL）。先用 /tmp/tiles_tools/unifdef.py 把这些宏固化、删掉其他实验分支
（固化后的模板与 /tmp/exp_m10/zc 生成的代码逐字节相同，已核对）。构建 `./build_var.sh name ""` → /tmp/exp_tiles/name；
一个可执行文件里有 4 个 tile 的 kernel，`-t 256x256|256x128|128x256|128x128` 选择。正确性脚本 `/tmp/exp_tiles/check.sh exe gpu tile`。

## 改掉的写死之处
| 位置 | 原来 | 现在 |
|---|---|---|
| traits | `struct ..._traits`，B_M = B_N = 256，断言 256 线程一线程一行、LDS == 160 KB | `traits_t<BM, BN>`（256/128），新增 SF_WAVES_A/B、C_COPIES、C_SWZ；LDS ≤ 160 KB |
| make_layout_ga / sa / gb / sb | 每 wave 每半 tile 4 条 DMA（`ih` 维 2、`issue` 维 4、断言 "4 issues per half"） | `a/b_buffer_load_insts`（128 行时 2），`ih` 维 = insts/2 |
| scale panel DMA | 4 个 wave 各拷 2 个 32 行块（只对 256 行成立）；sfa/sfb 共用读偏移 | `wave % SF_WAVES`：块少时多余的 wave 重复拷同一份（同数据写同地址），每个 wave 的指令数不变；sfb 单独的读偏移（panel 大小可不同） |
| scale 读取 / SF_SHIFT | 每半 4 个 dword，2 次 64 位移位 | 每半 E_M / E_N 个 dword，E/2 次移位 |
| 调度模板 | 按每 quadrant 32 条 MFMA 写死 | 256×256 保留原手工模板；其他 tile 用 `sched_spread<Q, DS, VMEM, VALU, STORE>` 按指令数均匀铺开 |
| 手写 vmcnt | B_b `2*a`；prologue `4*a + 16` / `4*a` | B_b `a + b`；prologue `2(a + b) + C_COPIES` / `2(a + b)`（只计比必须完成的那条更新的 VMEM） |
| C 暂存 swizzle | 按 528 B 行距写死 | 仅 B_N = 256 时启用（`C_SWZ`），其他 tile 不 swizzle |
| 旧的 scale layout（gsfa / ssfa / rsfa…） | 构造了但不用，带 `B_M == 256` 断言 | 删除 |

## 边界处理（不补 pad）
- A / B / C 的 buffer range = 本 tile 在矩阵内的行数 × 行距（下一 tile 的预取同理）：越界行读回 0、写被丢弃。
- C 列：一个线程在所有 copy 里的列相同，按 `col + c < N` 每线程算一次"越界加量"（2^30 个元素 → 字节偏移 2^31，超出 range），加到偏移上；
  推迟写的下半 C 记下自己 tile 的 rows / cols。（先试过每条 store 用 `c < N ? off : OOB` 选择：256×256 出现 103 个 VGPR spill，改成每线程一个加量后为 0。）
- K 放宽到 256 的倍数（最后一个 scale panel 只用一半）；N 为 16 的倍数（shuffle_weight）。
- host 的 scale 按 aiter e8m0_shuffle 把行补到 256，补齐值默认 0xFF（e8m0 的 NaN），确认补齐值只影响不写出的行列；`MXFP4_SF_PAD` 可改。

## 正确性（HIP 2）
矩阵：原 6 形状 + M ∈ {1,16,100,300,1000} × N ∈ {128,400,1280,4112}（K 轮换 256/768/7168，batch 轮换 1/2），各配 默认 / MXFP4_SFA_K_PATTERN=1 / PERSIST_GRID=37 / PERSIST_GRID=1，共 104 个。
| tile | 结果 | VGPR / AGPR | VGPR spill | SGPR spill | LDS |
|---|---|---|---|---|---|
| 256×256 | 104/104 | 512 / 256 | 0 | 21（zc 16） | 160 KB |
| 256×128 | 104/104 | 448 / 224 | 0 | 23 | 120 KB |
| 128×256 | 见下 | 448 / 224 | 0 | 20 | 120 KB |
| 128×128 | 104/104 | 416 / 208 | 0 | 18 | 80 KB |
