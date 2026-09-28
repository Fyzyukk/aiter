# 305-shape MXFP8 B-preshuffle retune

全部 **305 个 shape 已扫描，300 项获得有效最优，5 项无有效候选**。
四批执行结束，调优器统计耗时 1076.5402 秒（约 17.94 分钟，含扫描期间的 JIT 构建）。
profile 29170 行；tuned 300 行；进程因未解决的 5 项返回退出码 1。
300 条已保存选择均为本批最小有效耗时且 `errRatio=0`，没有 worker 崩溃或超时证据。

最优后端计数：`{"opus": 278, "ck": 7, "cktile": 15}`。
OPUS 最优 ID 计数：`{"9064": 10, "9063": 11, "9061": 14, "9062": 50, "9060": 31, "9000": 132, "9020": 30}`。
候选状态计数：`{"valid": 16730, "numerical_rejection": 11760, "unsupported_or_runtime_failure": 680}`。

OPUS 的 1945 条候选记录全部通过数值检查，全部七个 ID 均有中选。
新增五项合计中选 116 个 shape；原 9000/9020 合计中选 162 个。
在双方都有有效候选的 295 项上，OPUS 比最快有效外部候选快 278 项、慢 17 项，
等权几何平均耗时降低 **11.87%**。加入新五项后，相对仅 9000/9020，
候选池在 133 项上更快，其余 162 项相同，几何平均耗时降低 **15.88%**。

| 比较（左方 vs 右方） | 可比 shape | 左快 / 平 / 左慢 | 等权几何平均 左/右 | 几何平均 speedup 右/左 | 左方耗时降低 % |
|---|---:|---:|---:|---:|---:|
| opus_vs_external | 295 | 278 / 0 / 17 | 0.881320 | 1.134661 | 11.867965 |
| new5_vs_original2 | 295 | 133 / 0 / 162 | 0.907188 | 1.102308 | 9.281234 |
| all7_vs_original2 | 295 | 133 / 162 / 0 | 0.841234 | 1.188729 | 15.876562 |

仅双方都有本批有效候选的 shape 进入对应比较；没有有效候选的数量和全部逐 shape 数值见 summary.json、comparison.csv。

| 后端 | 候选记录 | 有候选 shape | 有有效候选 shape |
|---|---:|---:|---:|
| ck | 5490 | 305 | 300 |
| cktile | 10065 | 305 | 295 |
| asm | 11670 | 305 | 0 |
| opus | 1945 | 295 | 295 |

检查失败项：tuned_covers_input_exactly, process_exit_zero, no_unresolved_shapes_log_evidence。
受保护文件 SHA256：71/71 一致；具体路径与前后哈希见 summary.json。

未获得有效最优项的 shape（M/N/K）：

- `32768/65536/1536`
- `40960/65536/1536`
- `49152/65536/1536`
- `57344/65536/1536`
- `65536/65536/1536`

这五项的 BF16 输出分别为 4/5/6/7/8 GiB，每项的候选结果一致：

- OPUS：无候选；现有 `max_tensor_bytes=2^31-1` 支持限制检查 `2*M*N`，这些输出被明确过滤。
- CKTile：33 项均为 `us=-1`，日志明确报告 `Arguments not supported`；源码的大张量路径不支持本次 `ABQuantGrouped` 模式。
- CK：18 项均有正耗时，但 `errRatio=1`，全部被数值检查拒绝。
- ASM：30 项均有正耗时，`errRatio` 约 0.4691–0.5071，全部被数值检查拒绝。

ASM 不只在这五项失败：本轮全部 11670 条 ASM 记录均被当前严格数值检查拒绝，
包括 splitK=1。只读核对显示专用 tuner 保留 generic ASM 的输入生成、参数顺序和 runner，
参考使用对应原始权重与 FP32 scale，未发现适配层明显布局或参考复用错误。
仅凭记录不能确定 CK/ASM 的底层数值差异原因，也没有据此放宽阈值。
CK 的 `errRatio=1` 也可能来自非有限输出等提前拒绝，不能直接解释为所有元素均超差。
对应支持条件见 [OPUS 支持判断](../../csrc/opus_gemm/opus_gemm_common.py:1783)
和 [CKTile 大张量条件](../../3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp:1241)。

原始 305 项输入、生产配置、kernel 与共享 tuner 均未修改。
本轮 GPU 工作已退出，未追加复测、提交或推送。

日志 completion：2 条。
- `[aiter] processed 4 batches of 4, Processing Status ====> 100.0% tuned in opus_mxscale_bpreshuffle`
- `[aiter] Tuning Finished. tune 305 shapes, total tuning time is 1076.5402 seconds`

日志 fatal：0 条。

日志 worker_failure：0 条。

日志 incomplete_selection：6 条。
- `[aiter] No kernel can be used for ('gfx950', 256, 65536, 65536, 1536)`
- `[aiter] [Tuning not Finished] some shapes are not tuned or all failed, please check the result file or tune with --profile_file to get more details`

日志 missing_asm_list：0 条。

日志 distribution：4 条。
- `Distributing 100 task groups across 8 GPUs`
- `Distributing 5 task groups across 8 GPUs`

OPUS 无候选记录的 shape（gfx/CU/M/N/K）：

- `gfx950/256/16384/65536/1536`
- `gfx950/256/20480/65536/1536`
- `gfx950/256/24576/65536/1536`
- `gfx950/256/28672/65536/1536`
- `gfx950/256/32768/65536/1536`
- `gfx950/256/40960/65536/1536`
- `gfx950/256/49152/65536/1536`
- `gfx950/256/57344/65536/1536`
- `gfx950/256/65536/16384/1536`
- `gfx950/256/65536/65536/1536`

opus_vs_external 相对最慢的 5 项（左/右越大越慢；完整前 10 项及绝对耗时最慢项见 summary.json）：

- M/N/K=1472/7168/768：17.2066 / 16.5248 us，左/右=1.041259。
- M/N/K=1536/7168/768：17.3910 / 16.7192 us，左/右=1.040181。
- M/N/K=1664/7168/768：17.8552 / 17.2904 us，左/右=1.032666。
- M/N/K=1600/7168/768：17.4392 / 16.9528 us，左/右=1.028691。
- M/N/K=1728/7168/384：13.6548 / 13.3217 us，左/右=1.025004。

new5_vs_original2 相对最慢的 5 项（左/右越大越慢；完整前 10 项及绝对耗时最慢项见 summary.json）：

- M/N/K=8192/2048/7168：146.4816 / 100.2763 us，左/右=1.460780。
- M/N/K=1024/16384/1536：39.9226 / 29.2919 us，左/右=1.362923。
- M/N/K=16384/768/7168：131.7807 / 102.2445 us，左/右=1.288878。
- M/N/K=2048/7168/16384：252.6722 / 196.6431 us，左/右=1.284928。
- M/N/K=2048/7168/7168：119.3142 / 94.5789 us，左/右=1.261531。

all7_vs_original2 相对最慢的 5 项（左/右越大越慢；完整前 10 项及绝对耗时最慢项见 summary.json）：

- M/N/K=1024/16384/1536：29.2919 / 29.2919 us，左/右=1.000000。
- M/N/K=1024/65536/1536：111.5959 / 111.5959 us，左/右=1.000000。
- M/N/K=1216/65536/1536：146.9947 / 146.9947 us，左/右=1.000000。
- M/N/K=1280/65536/1536：141.7897 / 141.7897 us，左/右=1.000000。
- M/N/K=1408/65536/1536：172.4619 / 172.4619 us，左/右=1.000000。

解释限制：

- 本轮只有一次完整扫描，每候选 warmup 5、计时 51 次；不是多轮独立调优，不能宣称跨轮稳定获胜。
- OPUS 使用原生 E8M0 scale；CK/CKTile/ASM 使用独立随机 FP32 scale，各自计算参考。跨后端耗时比较不代表输入张量数学等同。
- 只比较本轮有效候选，不使用历史微秒数；几何平均按可比 shape 等权计算，耗时比为左/右，加速比为右/左。
- OPUS 对 10 个大 shape 没有候选记录，其中 5 个由 CK 得到有效选择，另外 5 个没有任何有效选择。
- CSV 中 `us<=0` 同时容纳不支持和运行错误；上面五项 CKTile 的“不支持”结论另有日志佐证。正耗时但非零误差属于数值拒绝。
- 后端和 shape 覆盖检查不等同于独立重新枚举全部注册项；本次候选枚举沿用原专用 tuner。
