# 本次选中 OPUS 与上游 FlyDSL mxscale 表比较

按两份已保存的时间比较，**本次选中的693个OPUS shape整体略占优势，但小M更偏向FlyDSL，不能全部选OPUS。** 当前还没有完成同卡OPUS/FlyDSL复测；下面都是历史CSV数据的描述性比较。

## 文件和输入契约

本次 OPUS 选择来自：
`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv`。
文件总共745个shape，其中693行选择OPUS，其余52行选择CK/CKTile/ASM。本报告仅比较这693行OPUS赢家，保持本次kernelId及混合编译器选择不变。

上游文件：
[dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/6264f8f5cc88472878356347a9d5f287760bc8e6/aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv)。
读取时main commit为 `6264f8f5cc88472878356347a9d5f287760bc8e6`，原始文件已冻结为
[upstream_mxscale_main.csv](upstream_mxscale_main.csv)，URL/commit/SHA在 [upstream_receipt.json](upstream_receipt.json)。
上游1357行中1176行为gfx950，181行为gfx1250；**gfx1250全部排除**。693个本次OPUS shape全部有gfx950 FlyDSL记录。

三种实现数学契约相同：FP8 A/B，逻辑E8M0 A-scale每1×128、B-scale每128×128，BF16输出，B使用16×16预重排，kernel内部广播scale供1×32 scaled-MFMA使用。

- OPUS读取二维E8M0 scale，A-scale逻辑 `(M,K/128)` 但物理字节按column-major排列。
- FlyDSL `kernelId=bmm` 读取同一二维raw scale契约，以batch=1执行。693个shape均有记录。
- FlyDSL `flydsl_mxpsh_*` 读取提前重排成一维的compact scale；逻辑分块仍为1×128/128×128。100个选中OPUS shape有此记录。两类FlyDSL记录不是可以随意混用的重复行；上游按scale布局分别路由。

这里同时报告两条路径，以及允许调用方预先准备相应scale布局时，每shape选择两条路径中更快者。预重排A-scale的每调用成本不在CSV GEMM时间中；B-scale通常可在权重准备时只做一次。

## 历史记录的逐shape结果

加速比定义 `FlyDSL_us / OPUS_us`，大于1表示OPUS更快。每shape只计一次，不把100个双路径shape重复计入总数。

| 对照范围 | shape数 | OPUS更快 | FlyDSL更快 | OPUS几何平均加速 |
|---|---:|---:|---:|---:|
| 同raw scale契约的FlyDSL BMM | 693 | 416 | 277 | 1.0509× |
| FlyDSL mxpsh重合子集 | 100 | 70 | 30 | 1.0688× |
| 每shape取可用FlyDSL路径的最快记录 | 693 | 399 | 294 | 1.0408× |

对每shape最快FlyDSL：OPUS快超过1.05倍的311项，FlyDSL快超过1.05倍的191项，另191项差距在1/1.05–1.05倍之间。因此整体4.1%的优势不足以作为当前同卡实测结论。

按M区间看：

| M区间 | shape数 | OPUS更快 | FlyDSL更快 | OPUS几何平均加速 | 数据倾向 |
|---|---:|---:|---:|---:|---|
| M≤128 | 114 | 18 | 96 | 0.9143× | FlyDSL约快1.0937× |
| 128<M≤1024 | 292 | 149 | 143 | 1.0140× | 非常接近 |
| M>1024 | 287 | 232 | 55 | 1.1254× | OPUS优势更明显 |

这些是区间统计，不能用固定M阈值代替逐shape选择。N768/K7168组总体更偏FlyDSL（OPUS0.8982×）；N16384/K1536组更偏OPUS（OPUS1.1102×）。完整组表见 [historical_by_nk.csv](historical_by_nk.csv)。

具体shape（µs）：

| M×N×K | 当前OPUS候选 | 当前OPUS保存时间 | 上游最快FlyDSL | 保存时间 | 历史较快者 |
|---|---:|---:|---|---:|---|
| 16×768×7168 | 9041 | 7.3852 | BMM/split4 | 4.9910 | FlyDSL，1.48× |
| 96×7168×7168 | 9044 | 19.2018 | BMM/split2 | 17.8750 | FlyDSL，1.074× |
| 112×768×7168 | 9052 | 9.7860 | BMM/split4 | 7.0170 | FlyDSL，1.395× |
| 64×6144×7168 | 9055 | 17.3209 | BMM/split2 | 12.7160 | FlyDSL，1.362× |
| 1024×16384×1536 | 9000 | 26.9723 | BMM/split1 | 42.1210 | OPUS，1.562× |
| 4096×2048×7168 | 9022 | 73.5048 | mxpsh/split1 | 65.6712 | FlyDSL，1.119× |
| 8192×2048×7168 | 9000 | 92.9164 | mxpsh/split1 | 105.1812 | OPUS，1.132× |

详表：[historical_best_per_shape.csv](historical_best_per_shape.csv)。
所有693 BMM + 100 mxpsh候选分别保存于 [historical_all_candidates.csv](historical_all_candidates.csv)，统计在 [historical_summary.json](historical_summary.json)。

## 当前同卡复测准备及状态

旧全量tune未包含有效FlyDSL比较：环境当时安装FlyDSL0.2.2，当前代码要求至少0.2.4，候选导入被关闭。因此此前“OPUS693”表示胜过同轮CK/CKTile/ASM，不表示胜过FlyDSL。

本轮在独立 `python_packages/` 中安装当前仓库requirements指定的 `flydsl==0.3.4.1`，没有替换原环境包，安装源/下载SHA记录在 [pip_install_report.json](pip_install_report.json)。CPU检查已确认新版本以及BMM/mxpsh模块可导入；使用原来的AITER_AOT_IMPORT=1避免无关Triton版本导入问题。

已准备 [compare_same_gpu.py](compare_same_gpu.py)：

- 固定本次693个OPUS赢家，对应上游793个gfx950 FlyDSL配置全部列入；没有加入gfx1250候选。
- 同shape同GPU、相同signed FP8数值和完全相同的逻辑E8M0 scale，用各自正确布局。scale重排、reference、首次编译在计时外。
- OPUS保持正式混合编译产物，并预分配其splitK workspace；FlyDSL使用正式host API，包含该候选所需的splitK归约及GPU辅助操作。
- 独立profiler5warmup/51iters，复用1地址和共享8地址池，各3个正反序块/6次测量。8个GPU进程；物理锁、KFD owner监控。
- 共同FP32 reference + 相同外部误差门槛；OPUS另外保留严格累加区间门槛。每个kernel先正确性检查，通过后才计时。
- 693个shape分成8个shard；大reference沿用此前chunk实现。

**截至本报告保存时GPU复测没有启动、没有任何FlyDSL GPU结果。** 8卡都被外部host PID1193783持有，GPU0约33GiB显存，其余卡也有上下文；利用率为0不等于没有owner。占用快照在 [foreign_gpu_processes.json](foreign_gpu_processes.json)，等待记录在 [claim_smoke.jsonl](claim_smoke.jsonl)。等待队列已停止，本报告没有把等待写成已完成测量。

8卡释放后，先运行8shape smoke：

```bash
/opt/venv/bin/python3 reports/opus_flydsl_comparison_20261008/run_eight_gpu.py \
  --queue reports/opus_flydsl_comparison_20261008/queue_smoke.json \
  --log reports/opus_flydsl_comparison_20261008/claim_smoke_resume.jsonl
```

检查smoke正确性/合法候选后，运行完整693shape配置对照：

```bash
/opt/venv/bin/python3 reports/opus_flydsl_comparison_20261008/run_eight_gpu.py \
  --queue reports/opus_flydsl_comparison_20261008/queue_eight.json \
  --log reports/opus_flydsl_comparison_20261008/claim_eight.jsonl
```

脚本已做CPU语法及模块导入检查，GPU契约与全量结果尚待smoke和实际执行验证。历史工具链、缓存/预热及运行状态不同可影响这4.1%的整体差距。当前应该保留逐shapeOPUS/FlyDSL竞争，不应据历史CSV直接覆盖已完成的745行正式结果。
