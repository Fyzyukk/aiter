# 9020 按 9000 风格整理接口和常量

范围为 9020 pipeline、对应 traits 和交接约定；9000 源码及 traits 未修改。保留现有后打包 Scale A、packed Scale B、M 尾保护、矩阵/scale 实际加载方式和流水线执行顺序。

## 完成的整理

- SFA：row 加入 g_sfa 基址；gsfa 去掉 row 参数，签名与 9000 一致。
- SFB：col 对应的 N128 scale 起点加入 g_sfb 基址；gsfb/ssfb 使用 lane_id、wave_id；gsfb/ssfb/rsfb 均显式写出 block_shape、block_dim、unfold_x_stride、unfold_p_coord。
- RA：原独立八-wave映射保留；固定 rows_per_wave 及每 fragment chunk 数从 traits 引用。
- 常量：swizzle 的分组与阈值、矩阵预取 VMEM 数、BF16 输出向量宽度与 LDS 行距、输出次数及对应断言集中到 traits。SWIZZLE_GROUP_M 与量化 GROUP_M 分开命名；VEC_OUTPUT 与 MFMA 的 VEC_C 分开命名。
- 现有 VEC_A/B 用于矩阵 load、async_load 和 layout_to_offsets；lambda 内只保留随 repeat/stage 实例化而变化的局部坐标。
- C 的 p_coord/u_sc/sc_offsets 在 MMA 初始化后声明，最终 MFMA、LDS 写入和输出时序保持。

## CPU 验证

用相同工具链和冻结 flags 分别编译变更前后的当前 9020。两次编译都成功，没有运行 GPU。

| 项目 | 修改前 | 修改后 |
|---|---:|---:|
| VGPR | 204 | 204 |
| SGPR | 53 | 52 |
| LDS bytes | 143360 | 143360 |
| private bytes | 0 | 0 |
| VGPR / SGPR spill | 0 / 0 | 0 / 0 |
| kernel 指令字节 | 7812 | 7808 |

ABI 一致。内存指令计数一致，包括 28 条 A/B direct-to-LDS、3 条 SFA 16B global load、2 条 SFB byte global load，以及各 LDS read/write 指令。

另做一次 host-only 编译及一次 CPU 运行，直接使用真实 Opus layout 和当前 traits。核对所有 512 个线程的 RA、SFA global/store/read、SFB global/store/read，共 14848 项最终地址；SFA/SFB 的 global 地址把迁入基址的 row/col 偏移计入比较。地址、数量和顺序全部一致。

本轮机器码不是逐字节相同，未用 CPU 编译或地址检查推断 GPU 数值/性能结果，也没有再次 tune。

## 记录

- `change.diff`：两个生产文件的完整修改。
- `source_check.json`：9000保护、接口/常量整理、M尾保留及内存指令计数。
- `device_validation.json`：完整 AMDGPU metadata、ABI 与资源对比。
- `host_layout/validation.json`：CPU 地址核对命令与结果。
- `before/`、`after/`：各自冻结源码、launcher、构建记录、device code 和反汇编。
- `snapshot/`、`before_hashes.json`：本轮修改前的来源快照。
