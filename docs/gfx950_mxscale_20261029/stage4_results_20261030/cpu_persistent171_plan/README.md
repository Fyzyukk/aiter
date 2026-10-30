# 171 个持续落后项：CPU 优化方案

> 本目录是原运行环境的归档副本。JSON 指纹与脚本仍引用原运行目录及冻结 CO/JIT；本次未执行归档脚本。Markdown 链接已改为仓库相对路径，原 CPU 检查报告认证的是原文件，归档副本的校验值见 [upload_manifest.json](../upload_manifest.json)。

这是针对现有 Opus kernel 的方案包。只读当前源码、已构建 CO、调优结果与固定 CSV；本次不运行 GPU、不编译新 CO/JIT、不改 kernel 源码。候选中的新排程仍需实现和验证，不能当作已有性能结果。

## 范围与优先级

比较口径固定为目标 421 shape 中的 `gfx950 / cu_num=256 / w_scale_block=128x128` CSV 行。历史与当前都落后的 171 项中，92 项已比历史 Opus 改善；当前仍落后 CSV 的中位幅度为 4.88%。135 项来自 101 次计时，36 项来自 11 次快测。优先研究两轮都落后至少 3% 的 117 项；小于 1% 的 13 项先保留精确复核。

| 优先路线 | 为什么投入 | 首要操作 |
|---|---|---|
| 大 M 的 4wave | 当前 102 个持续落后赢家走该路线；四个重点 N/K 组中有 72 个 | 将下一 scale panel 的 global load 提前到前一 K128 iteration，保持原发布、等待与 barrier；首轮一个策略、至多两个实例 |
| 小 M、N7168、K16384 的 interleave | M1216–1664 的 8 项长期存在较大差距 | 检查 BM192 的 A 加载覆盖，优先去除没有 consumer 的加载；再比较独立的发请求顺序 |
| 窄 N 的 B-VGPR/overlap/m-repeat | N768/1536/2048、K7168 共 32 项持续落后 | 4 个代表 shape、每个最多 4 个精确已有配置；先排除参数覆盖再改排程 |

4wave 的 12 个基础 CO、interleave 的 BM128/192 × WaveN2/4 已覆盖重点代表项，不能用重复加入这些几何解释差距。m-repeat 在 Top4 外只通过精确 seed 加入时，同几何其他参数未必展开，应先做有限补齐。`pingpong_lds` 就是 m-repeat，路线数不能重复统计。

窄 N 的首轮 16 对包括 4 个当前锚点、5 个未进入重建 Top8 的补覆盖项、7 个受控复核项。其中 `(16384,768,7168)` 的赢家 m-repeat 几何有 63 个合法配置，Top8 只保留精确 seed 一个，先补 fragments、GroupM、输出 guard 三项对照。这里的“已有配置”指注册表中的精确 traits；4 个候选是 ready CO，12 个是 JIT 注册项，本次没有验证全部 JIT 已有可加载产物。

interleave 首轮只选 3 个落后代表 × 原版、跳过无 consumer 的 A 填充行、B 先发三个版本，共 9 对一批筛选；两个新版本分开实现和比较。静态覆盖确认 BM192/WaveN4 的 A 请求减少 25%、A+B 名义请求字节减少 12.5%，LDS 大小保持不变。4wave 三种 panel 的 LDS 均超过 80 KiB，单靠 panel 变化不能实现两个 WG/CU 并驻；旧 GroupM CO 与当前 CO 不属于相同编译控制，不能直接把差异归因于 GroupM。

详细源码位置、具体候选、静态资源和正确性约束分别在 [4wave_plan.md](4wave_plan.md)、[interleave_plan.md](interleave_plan.md)、[narrow_plan.md](narrow_plan.md)。[current_winner_inventory.csv](current_winner_inventory.csv) 保存 171 个当前赢家的精确 ID、完整名称、traits、几何和矩阵尺寸；已经用 CPU 注册表逐项验证名称和 shape 合法性。

## 实施顺序

1. 保留本轮全部已有结果、CO、JIT 和 14 项复核包。未来实现新版本时，在独立源码/产物目录构建，使用新名称和单独的候选清单，避免覆盖旧测量所依赖的文件。
2. 窄 N 先使用 [narrow_existing_candidates.csv](narrow_existing_candidates.csv) 的少量精确配置。4wave/interleave 首轮先复测代表 shape 的当前赢家，再分别添加一个新版本，确保能判断改动的效果。
3. 每次只改变一个因素。排程、几何、cache、GroupM、LDS 大小、预取距离分开比较；无效 GroupN 和裁剪后相同的 B_AHEAD 不当作独立优化轴。组合版本仅在独立版本通过后考虑。
4. 有效版本先扩展所属 N/K 和 M 区间，并检查既有领先保护项。确认收益和正确性后才更新选择规则；完整 421 复测放在最终验证阶段。

本目录没有 GPU 执行入口。后续明确开始 GPU 验证时，只测 Opus 精确候选，继续引用固定 CSV；统一 101 次计时、5 次预热、相同数据和轮转数。先做一批筛选，仅对进入决赛的当前/新版本做交替顺序的重复配对。不重复 FlyDSL 调优，不再次展开全候选搜索。既有 14 项包的两个重点异常项仍应在原产物保持冻结时单独复核，它们与本方案的代表 shape 数不是同一个统计。

## CPU 能确认和待验证的部分

CPU 能确认候选覆盖、traits 合法性、地址范围、布局覆盖、重复加载、依赖顺序，以及已有 CO 的静态 LDS/寄存器/scratch 信息。GPU 的真实瓶颈、cache 行为、驻留 wave、编译后排程效果和耗时改善需要实测。静态请求减少比例不等于 HBM 流量或性能改善比例。

171 项的 A 最大 1 GiB、B 最大 147 MiB、BF16 C 最大 896 MiB；当前赢家的 C tile 寻址跨度最大 25,035,264 字节。超过 4 GiB 的张量不是这些持续落后项的解释。仍须保留输出 tile 的 64 位基址、32 位局部偏移和末尾 M guard。全 421 中存在最大 8 GiB 的 C；改变共用寻址代码时，要额外检查这些已领先 shape 的地址范围，避免把局部优化推广成全矩阵 32 位寻址。

后续是否保留版本，以正确性、同口径配对耗时、固定 CSV 差距和所属组的保护结果共同决定。没有稳定收益时保留原版本；快测的小幅差距和三批一致本身不能充当统计置信区间。

## 复查本方案

纯 CPU 校验命令：

```bash
python3 -S /home/fyz/retune_dsv4_20261028/gpu_stage3_20261029T161551Z/stage4_full421/cpu_persistent171_plan/verify_plan.py
```

校验只读取冻结输入和候选身份，不调用 GPU 或编译器。输入、源码或方案变化会报出指纹不匹配；修改后需要重新审计，不能把旧检查结论沿用到新方案。结果保存在 [cpu_validation.json](cpu_validation.json)。
