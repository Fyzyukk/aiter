# 本轮 gfx950 调优与后续 CPU 优化记录

本轮最终结果为 **421/421 完成，236 个领先固定 CSV、185 个落后**。实际 kernel 结果在 [full421_tuned.csv](full421_tuned.csv)，逐项基线对比在 [full421_comparison.csv](full421_comparison.csv)。本记录将已经执行的调优与尚待实现、验证的优化方案分别注明。

## 已完成的候选与调优工作

源码与候选身份对应 `fyz/gfx950-search-identity-20261029` 的 `488712ab`。已注册并纳入统一搜索的路线包括 4wave pin_agpr、m-repeat、interleave、B-VGPR 和 overlap；`pingpong_lds` 就是 m-repeat JIT，不能另算一条路线。4wave 使用 CO，m-repeat/interleave 使用 JIT，B-VGPR/overlap 有 JIT/CO。

此前补齐的 m-repeat 搜索配置、旧赢家输出 guard 配置及步骤 2/3 的 CO/JIT 候选均由本轮统一候选库提供。历史 seed 使用完整名称与当前配置匹配，不能直接把历史数字 ID 当作当前 ID。[四步交接文档](../handoff_4steps_zh.md) 保留详细参数与执行计划；[候选覆盖表](search_preparation/candidate_coverage.csv) 和 [身份映射](search_preparation/identity_mappings.json) 保存准备阶段的检查记录。合法并入选表示搜索覆盖，不代表全部合法参数都被测量，也不证明每个入选候选的时延稳定。

完整阶段按每个 pipeline Top8 保留几何，并展开该几何的合法配置及兼容 seed。2026-10-30 06:03 UTC 按减少测试的要求保留 292 个已经完成的结果，对剩余 129 项改用 Top4、11 次计时、2 次预热。最终于 07:01:53 UTC 完成，控制器和四个 GPU worker 均退出；最大误差为 0.0009，上限为 0.05。

只使用固定的 gfx950/256 CU/128x128 CSV 对应行。历史按同一基线为 192 胜、229 负；当前为 236 胜、185 负。历史与当前交叉为 178 双赢、14 新增回退、58 转赢、171 持续落后。早期进度里的 fresh FlyDSL 字段保留为历史信息，终表始终取 `best_opus`。

## 14 个新增回退：CPU 分析已完成，GPU 复核未执行

完整幅度和候选证据见 [regressions14_audit.csv](regressions14_audit.csv) 与 [analysis_zh.md](cpu_regressions14_preparation/analysis_zh.md)。13 项仅慢固定 CSV 0.040%–2.989%，其中 7 项不到 1%；唯一明显项 `(3072,7168,21504)` 慢 15.116%。12 项来自快速计时，两项来自保留的完整计时。

14 个历史赢家均进入对应候选集合。明显项历史赢家 CO693 为 410.7753 μs，本轮同名候选为 480.3212 μs；本轮最佳 CO501 为 479.1171 μs，固定 CSV 为 416.2038 μs。已安装历史快照与当前 CO693/501 的 `.text/.rodata/.note`、launcher 和资源元数据一致；完整 ELF 差异不能直接解释为指令变化。历史缺少逐次加载 receipt，因此仍无法严格认证历史计时实际加载的 image。

短计时与完整计时在样本数、异常值过滤和输入轮转上不完全等价，不能把所有小幅回退都断定为噪声，也没有 profiling 证据可以断定 HBM 或 occupancy 根因。

[CPU 复核准备包](cpu_regressions14_preparation/README.md) 已准备 26 个精确 `(shape,kid)` 配对行：两个重点 shape 为 4 对，余下 12 个为 22 对。统一 101 次计时、5 次预热、三批交替顺序，继续只引用固定 CSV。CPU 默认/门禁、分析器合成及 runner 仿真检查已通过；本次以及归档过程都没有执行 GPU 复核。

## 171 个持续落后：CPU 分组和方案已完成，新 kernel 未实现

[persistent171_analysis.csv](persistent171_analysis.csv) 保存全部形状、当前及历史耗时和当前赢家。92 项已比历史 Opus 更快，79 项变慢；同组相对 CSV 的中位差距从历史 5.57% 降到当前 4.88%。135 项来自原 101 次计时，36 项来自快速轮。当前小于 1% 的有 13 项，1%–3% 有 33 项，3%–10% 有 110 项，至少 10% 有 15 项。

四个重点 `(N,K)` 组为 `(7168,16384)`、`(7168,7168)`、`(7168,4096)` 和 `(6144,7168)`，共 108 项，占 63.2%。当前赢家路线为 4wave 102、m-repeat 34、interleave 15、overlap 11、B-VGPR 9。

| 路线 | 具体 CPU 方案 | 后续首筛规模 |
|---|---|---:|
| [4wave](cpu_persistent171_plan/4wave_plan.md) | 下一 scale panel 的 global load 提前到前一 K128 iteration 的 C00 结束处，保留原 wait、两次 barrier、LDS 发布与输出路径；一个策略、panel32/64 两个设计实例 | 4 个 shape，10 对配置 |
| [interleave](cpu_persistent171_plan/interleave_plan.md) | BM192/WaveN4 时跳过 M half 内没有 consumer 的 A 填充行，保留实际尾 M 必读零；另做独立 B 先发版本 | 3 个 shape，9 对配置 |
| [窄 N](cpu_persistent171_plan/narrow_plan.md) | 先固定四个当前赢家，补少量 fragments、GroupM、输出 guard、发布和预取对照；源码发布点/A ring2/3 后置 | 4 个 shape，16 对配置 |

合计 **11 个代表、35 对配置、一批筛选**，只对正确且有效的版本再做重复配对。新版本仍是 `design_only`，未修改源码、未构建新 CO/JIT、未产生新耗时。16 对窄 N 项是已有注册配置，其中 4 个是 ready CO、12 个是 JIT 注册项；CPU 检查不等同于全部已有可加载 JIT。

明确静态发现与限制：

- 4wave panel16/32/64 的 LDS 分别为 139392/143616/152064 字节，均超过 80 KiB；单靠 panel 变化不能实现两个 workgroup 同驻一个 CU。旧 GroupM 镜像与现代 CO 的编译控制不同，不能把跨版本差异直接归因于 GroupM。K4096/P32 无中途 refill 仍落后，是提前 scale 策略的负对照。
- interleave BM192 每个 half 消费 96 行 A，现加载布局覆盖 128 行。静态布局枚举确认可以省去未消费部分，A 请求减少 25%、A+B 名义请求字节减少 12.5%，LDS 不变。这不是实测 HBM 流量或预期加速比例。
- 窄 N m-repeat 的一个赢家几何有 63 个合法配置，Top8 只通过精确 seed 保留了一个，适合有限补覆盖。窄 N 的 GroupN 不改变实际 tile 映射；B_AHEAD 应按裁剪后的实际值分析，PinA 等参数可能同时改变队列上限。
- 171 项 A 最大 1 GiB、B 最大 147 MiB、BF16 C 最大 896 MiB，超过 4 GiB 的 tensor 不是这些形状落后的解释。修改共用代码仍须保留 C 的 int64 基址和局部窗口；全 421 中有最大 8 GiB 的 C。

[CPU 校验报告](cpu_persistent171_plan/cpu_validation.json) 确认 171 个当前赢家身份与 shape 合法性、16 对窄 N 配置、12 个单输入参数对照及 47 个冻结文件指纹。布局与阶段模型是 CPU 静态证据，GPU 正确性、编译后资源/ISA、真实瓶颈和性能收益仍待验证。

## 后续工作状态

下一步是按上述有限方案在独立源码和产物目录实现新版本，保留本轮已有 CO/JIT 与 14 项冻结包。后续明确开始 GPU 验证时，统一同 shape 输入、轮转、101 次计时和5次预热，先筛选、再复测有效版本、再扩所属 N/K 组和领先保护项；完整 421 验证放在最终阶段。当前没有启动这些工作中的 GPU 执行。

本目录归档方式与原环境依赖见 [README.md](README.md)。原 JSON 指纹认证原文件；Markdown 链接的归档修改和副本校验值记录在 [upload_manifest.json](upload_manifest.json)。
