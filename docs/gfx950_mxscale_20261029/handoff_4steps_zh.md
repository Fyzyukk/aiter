> 2026-10-30 最新最终结果：**338领先 / 83落后**，已完成远端54bef351对齐及最后第18轮优化。远端236领先全部保留，原本地337领先逐字段保持，新增1项复测领先；新interleave kernel提升2项但仍落后。领先汇总、完整421结果、优化归因与换卡异常核对见 [最终合并结果](final_aligned_results_20261030/README.md)。优化已结束，下文保留原四步交接记录。

# gfx950 MXFP8 优化交接：四步执行

## 1. 定向补测 32 个 m-repeat shape

当前 m-repeat、interleave 走 JIT；4wave_pin_agpr 走 CO；B-VGPR 和 overlap 同时有 JIT/CO。CO 共 780 个，编号 0–779，全部 build ready：4wave 12、B-VGPR 384、overlap 384。m-repeat 已删除 CO 和 pin 接口，使用 13 参数 traits。

`opus_gemm_common.py` 已追加 102 个 m-repeat 搜索配置（48096–48197），又补齐远端旧赢家的 14 个 `BoundedOutput=false` 配置（48198–48211）。这 14 个固定 `FixedK=0、GroupM=0、GroupN=8、XcdOrder=false、PublishM=0`，覆盖远端 32 个 m-repeat 最优 shape。当前 m-repeat JIT 共 12,230 个；新增项完成 CPU 覆盖核对，GPU 正确性和性能待测。

输入使用 [m_repeat_32_shapes.csv](handoff_data/m_repeat_32_shapes.csv)。对照包括新增 false 版本、原有对应 true 版本、同几何其他合法配置，以及保存的 CSV 基线。这 32 个中原来 16 个胜 CSV、16 个落后，两部分都测。

旧赢家几何已写入 [m_repeat_32_geometries.csv](handoff_data/m_repeat_32_geometries.csv)，并核对当前每个 shape 都有合法候选。使用指定几何和 `--opus_top_geometries 0`，测这些几何下的全部合法参数，保证目标进入测量列表：

```bash
AITER_REBUILD=1 python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype both --mp 8 --shape_grouped \
  --opus_top_geometries 0 \
  --opus_geometry_file docs/gfx950_mxscale_20261029/handoff_data/m_repeat_32_geometries.csv \
  -i docs/gfx950_mxscale_20261029/handoff_data/m_repeat_32_shapes.csv \
  -o /tmp/gfx950_m_repeat_32_tuned.csv
```

`BoundedOutput=false` 增加逐行输出 guard，true 省略该 guard；两者都使用 tile 限定 SRD。恢复完整参数路线是否改善性能，需要本次测量确认。

## 2. 复测 46 个旧赢家回退 shape

本地全量 tune 停在 185 个 shape：72 个胜保存 CSV、113 个落后。其中 46 个在远端曾胜过同一份 CSV，旧赢家 38 个走 CO、8 个走 JIT。另有 67 个两边都落后。远端 192 个是 CSV 胜出；207 个是同轮 FlyDSL 胜出，后续统计统一使用 CSV 口径。

输入使用 [regressions_46_shapes.csv](handoff_data/regressions_46_shapes.csv)，几何使用 [regressions_46_geometries.csv](handoff_data/regressions_46_geometries.csv)。沿用第一步命令，替换这两个输入路径和输出文件，仍使用 `--opus_top_geometries 0`。32 与 46 两批有 2 个重复 shape，合计 76 个不同 shape。

优先复测已恢复的 CO。例如 `96×256 / wave1×4、NB4、BD1、XCD=false、Contiguous=false、PinA-only`：普通 B 的 K768/K384 对应当前 kid 454/451，overlap 的 K384 对应 643。相关旧配置已经在当前清单中，先判断新 build 后能追回多少。4wave 的 panel16/32/64、Unroll2、PAD/non-PAD 也已被当前 12 个 CO 覆盖。

历史记录只用于选择参数和比较参考耗时。旧 kid 不能直接用于当前分支；m-repeat 旧 16 参数须去掉 PinA/B/C 三个槽后匹配当前 13 参数。此次指定几何不依赖旧名字自动恢复 seed。

build-co 的 spill/scratch 默认保留策略已经生效，当前 780 个都 ready，这一步使用现有镜像即可。测完两批后重新统计仍落后项，以实测结果决定后续优化对象。

## 3. 优化补测后仍落后的 shape

旧快照中，两边都落后的 67 个输入在 [both_losses_67_shapes.csv](handoff_data/both_losses_67_shapes.csv)。其中 K7168 共 55 个：N6144 有 26 个，N1536 有 14 个，N768 有 8 个，N2048 有 7 个；剩余是 N7168/K2048 的 6 个、K768 的 5 个、K1024 的 1 个。先用步骤 1、2 的新结果更新这个集合。

**B-VGPR / overlap：** 窄 N768/1536/2048 优先比较 BN64/128、2/4 wave、A stage2/3/4 和实际 B 预取距离。先固定几何、pin 和其他参数，分别比较预取与 XCD。`B_AHEAD = min(BD × B_FRAGMENTS, B_SCHEDULE_AHEAD)`，BD1/2 可能裁剪成相同距离；应记录实际 B_AHEAD，不能把两套 bundle 的差距全部归因于 BD。当前 CO 两套 bundle 保持 `(BD2,XCD=true,Contiguous=true)` 和 `(BD1,XCD=false,Contiguous=false)`，不直接扩大整套笛卡尔积。普通 B 的 PublishM=-1、overlap=0，两条调度分别比较。N6144 同时比较 B、overlap、m-repeat 和 4wave，以该 shape 的实测排序确定方向。

**4wave / 大 M 长 K：** 先比较现有 12 个 CO 的 panel16/32/64、PAD 和 WideOutputfalse/true。固定最好的组合，保持 runtime K、Unroll2，再有限比较 GroupM0/4/8；有收益再考虑 16。panel 改变 scale refill 频率及 LDS 占用，WideOutput改变寻址路径，当前两套输出算法都仍是 LDS。观察 MFMA 活跃度、VMEM/LDS wait 和实际驻留 wave，再决定调整加载、等待与计算的顺序。重点参考 `(N,K)=(6144,7168)、(7168,7168)、(7168,16384)、(7168,4096)`。

m-repeat 新增 GroupM4/8 可比较 M tile 遍历；GroupN16/32 需要 `N/BN>=256`，适合 N65536 等宽 N，N7168 不触发。panel256 主要补 K21504 的覆盖。interleave 放在已有旧赢家复测和上述重点组之后，再依据结果决定投入。

## 4. 验证完整 421 个 shape

定向优化确认有效后，使用 [shapes_421.csv](shapes_421.csv) 覆盖全部 shape。首轮用 Top8：每个 shape、每个 pipeline 保留评分前 8 种几何，保留这些几何下的合法 JIT/CO 参数。当前默认是 0，启用 Top8 必须显式传 8。

```bash
AITER_REBUILD=1 python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype both --mp 8 --shape_grouped \
  --opus_top_geometries 8 \
  -i docs/gfx950_mxscale_20261029/shapes_421.csv \
  -o /tmp/gfx950_full421_tuned.csv
```

421 阶段不传前两步的局部几何文件。先核对已知赢家是否进入 Top8 列表；被截掉的赢家另用指定几何和 0 定向补测，再合并结果。若要测全部合法候选，将完整命令的 8 改成 0。

基线固定使用 [csv_baseline.csv](handoff_data/csv_baseline.csv)，逐 shape 参考值也在 [远端完整记录](all_shapes_421.csv) 的 `baseline_flydsl_us` 中。正确性通过后，`best_opus.us < CSV基线.us` 就算胜出；同轮 FlyDSL 最佳作为补充比较，不能仅凭最终 `best_libtype` 统计 CSV 胜负。前几名交替重复测量，确认排序和耗时波动。每轮报告完成 shape 数、胜 CSV 数、剩余落后数及主要 N/K、M 类别。

历史核对数据：[本地 full185](handoff_data/local_full185_progress.jsonl)、[本地 priority22](handoff_data/local_priority22_progress.jsonl)。远端 192 个 CSV 胜出与本地已有结果的历史胜出并集为 230 个；该数字只表示跨机器历史覆盖，当前候选的最终胜出数以本轮测量为准。
