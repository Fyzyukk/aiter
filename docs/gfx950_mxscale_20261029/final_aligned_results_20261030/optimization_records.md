最后一轮优化记录（stage18）

远端来源：[optimization_records.md](remote_snapshot/optimization_records.md)、[README.md](remote_snapshot/README.md)，固定提交 `54bef351`。远端的 14 项回退及 171 项 CPU 计划用于选择方向；先与本地最终结果对齐，只处理仍落后的 84 项。

本轮构建 12 个新版本、10 个同配置对照；加入远端及回退分析中与本地 84 项相关的 98 条 shape/candidate 记录，去重后形成初筛计划。4wave 方向覆盖提前 scale 加载；interleave 比较未消费 A 行裁剪与独立 B-first；narrow 方向补齐精确既有候选，并按本地当前 winner 测试 m-repeat A_RING=2。

提前 scale 加载：在 chunked 主循环 C00 之后发出下一 panel 的 raw 读取，在同一 iteration C11 之后保留原等待和 barrier，再发布。没有改变 MFMA/矩阵数据流。panel32/64、loops1–200 共 400 个调度模型检查通过。提前加载增加寄存器占用，panel64 个别版本存在 SGPR spill，实际资源已逐个从 linked CO 的 ELF notes 读取。

interleave：I1 只跳过 BM192 中没有 consumer 的 producer 行 96–127，仍保留真正 tail-M 的零写入；I3 独立按 B0/B1/A0/A1 发出请求。6 个 producer/consumer 覆盖模型通过。两个 B-first linked CO 的 24 条 matrix async-load 指令全部核对，3 组从 AAAABBBB 改为 BBBBAAAA。I1 的静态指令数未减少，其 wave guard 减少实际执行请求。所有 interleave 版本 VGPR182、无 spill、LDS160 KiB。

m-repeat：只用 derived traits 将 A_RING 从 3 改为 2，保留原流水线和同步；E_M3/4、ring2/3 的 slot ownership 模型通过。编译后 LDS 与各自原版相同，因为总 workspace 被其他区域/epilogue 约束；不能据 ring2 推断驻留率提升。全部 m-repeat 对照和新版本无 spill。

窄 N 远端代表 `(1152,768,7168)`、`(16384,768,7168)`、`(16384,1536,7168)` 仍落后，加入其精确现有候选；`(10240,2048,7168)` 本地已经领先，保持已有行。远端小 M 长 K 代表在本地当前最快属于 m-repeat，因此 interleave 改动同时与同配置原版和本地当前最优比较。

测量使用物理 GPU0/3/6/5，PCI 验证；逻辑0/1/2/3映射分别为0000:05:00.0、0000:75:00.0、0000:e5:00.0、0000:95:00.0。原初筛到确认边界的 GPU5 空闲 preflight 读取78%而停止；四个初筛 worker 均退出0，之后重新核对空闲，仅恢复确认，不重跑初筛。原始日志和恢复回执均保留。

六轮汇总后发现 GPU5 上六个 shape 的当前 winner/control 计时最大/最小比超过1.5，其中 `(8192,1536,7168)` 同一旧 kernel 在90和210 us间跳变。按照同一规则选出六项，保持候选不变，转到物理 GPU0/3/6 补做完整六轮；全部354条正确，该旧 kernel 恢复至约90.26 us，六轮比约1.012。每个受影响 shape 的整组结果全部替换，未按最快批次挑选；原始4794条及首次公开回执保存在 stage18/initial_verification_snapshot。

本轮实际 9096 条性能测量全部正确（初筛3948 + 原确认4794 + 换卡核对354）；最终统计接受初筛3948和确认4794，共8742条，354条受波动影响的原确认记录仅归档。799个 shape/candidate 组合六轮完整。公开最优的稳定候选收益 20 项；新 CO 成为 2 项最优，新增领先 1 项。详见逐项 CSV，最终收益不从远端计时推导。

| 新版本 pipeline | 六轮 matched 比较对数 | 达到≥0.5%、≥4/6快 | 成为公开最优 |
|---|---:|---:|---:|
| pin_agpr | 108 | 1 | 0 |
| pingpong_lds | 59 | 1 | 0 |
| wave_interleave | 149 | 89 | 2 |

表中 matched 收益包括未成为本地最优的实验候选，不能等同新增领先或整合收益。全部试验保留，最终只使用公共可运行候选。

第18轮为本次最后一轮；剩余落后项仅归档，不继续扩新候选或启动第19轮。

实际整合 kernel 的两项收益：

| M / N / K | 对本地当前最优耗时下降 % | 对同配置原版耗时下降 % | 更快轮数 | 最终仍落后 % |
|---|---:|---:|---:|---:|
| 1344 / 7168 / 16384 | 1.6871 | 4.1255 | 6/6 | 4.1674 |
| 1280 / 7168 / 16384 | 1.1638 | 3.1986 | 6/6 | 5.9047 |
