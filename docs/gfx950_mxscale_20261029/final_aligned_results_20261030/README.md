最终合并结果：**338 项领先 / 83 项落后**。全部领先项在 [leading_tuned.csv](leading_tuned.csv)，完整 421 项在 [full421_tuned.csv](full421_tuned.csv)。

按完整六字段 gfx、cu_num、M、N、K、w_scale_block 对齐。固定基线的原字节 SHA-256 相同，421 个 shape 集合完全一致。远端 kernel 均按完整名称映射并核对 shape 支持。

| 对齐类别 | 项数 |
|---|---:|
| 双方领先 | 236 |
| 本地独有领先 | 101 |
| 远端独有领先 | 0 |
| 双方落后、进入最后优化 | 84 |

远端 `54bef35161139556602a39d0d2dcf367102193fc` 的 236 个领先全部包含在本地 337 个领先中。本地有 101 个远端未领先的 shape。最终仍完整保留这 337 行，再加入最后一轮新增的 1 个领先。

最后一轮只测原 84 个共同落后项：2 轮初筛、6 轮独立确认，每次确认计时 101 次；9096 条性能记录全部正确、pending 0。原 337 个领先项沿用其已验证结果，逐字段不变。新 CO 已经过公共注册表、JIT 产物、正确性验证。

新增领先归因：新 kernel 且原 control 仍落后 0 项；既有候选且原 control 仍落后 0 项；原 control 本次复测也越线 1 项。成对比较和逐项证据见 [last_round84_comparison.csv](last_round84_comparison.csv) 与 [newly_leading_shapes.csv](newly_leading_shapes.csv)。

整合 1 个新 CO：`[101312]`。稳定整合要求最终最优、六轮全正确、至少 4/6 轮快于本地当前 control、成对中位 speedup 至少 0.5%，并对同配置原版满足同样门槛。comparison-only 镜像不整合。

[optimization_records.md](optimization_records.md) 记录远端方案的借鉴、实际测量和未采用方向；[last_round_matched_kernel_trials.csv](last_round_matched_kernel_trials.csv) 保留每个已确认的新版本与同配置原版的比较。

剩余落后中位 3.3152%，最大 8.8026%。完整清单见 [remaining_losses.csv](remaining_losses.csv)。本轮结束后停止优化。

| 剩余最优 pipeline | 项数 | 落后中位 % | 最大 % |
|---|---:|---:|---:|
| pin_agpr | 53 | 3.3697 | 8.3302 |
| pingpong_lds | 20 | 1.6341 | 8.6511 |
| prefetch_b_vgpr | 4 | 3.2162 | 4.6758 |
| prefetch_b_vgpr_overlap | 3 | 5.2749 | 7.1015 |
| wave_interleave | 3 | 5.9047 | 8.8026 |

文件入口：

- [full421_comparison.csv](full421_comparison.csv)：最终对固定 gfx950 CSV 的比较，含初始领先分类。
- [all421_alignment.csv](all421_alignment.csv)：优化前的双方结果对齐；远端耗时仅用于参考。
- [shape_evidence421.csv](shape_evidence421.csv)：每项最终证据的轮次和来源。
- [new_integrated_co.csv](new_integrated_co.csv)：本轮整合 CO 的身份、shape 范围和镜像 hash。
- [summary.json](summary.json)、[last_round_completion_receipt.json](last_round_completion_receipt.json)：完整核验回执。
- [run_env.sh](run_env.sh)：当前公开运行环境，使用本地 pin-AGPR LLVM wrapper。
- [remote_snapshot/README.md](remote_snapshot/README.md)：固定远端提交的原始索引，可进入三条 kernel 路线文档。

只 tune Opus MXFP8 NT gfx950/256 CU；FlyDSL 使用固定 CSV。公共旧 ID/name/config 保持，JIT 起点 780。A 端 2 GiB guard 和 C 端 64-bit base + tile buffer window 保留。远端混合使用 Top8/101 次与 Top4/11 次计时，不能把跨环境耗时差直接归因于 kernel。
