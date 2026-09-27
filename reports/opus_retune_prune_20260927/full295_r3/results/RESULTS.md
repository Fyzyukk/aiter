# 295 项全后端重新调优

本次完整独立三轮，8 张物理卡，耗时 22.20 分钟。
OPUS 胜 291/295；外部胜 4/295。
同 shape 等权几何平均耗时相对有效外部 -12.9453%，相对原 5 -4.2688%。
根据整体最快选择，25 个 OPUS 候选保留 16 个，其中 runtime-K 11 个。
无历史耗时拼接，无预设候选数，无 1% 子集损失阈值。CK/CKTile/ASM 本次完整枚举；每个候选按原 FP32 误差界和输出保护区检查，失败候选不参与选型。

| ID | 库 | 合法 shape | 有效 shape | 最快 OPUS | 整体选中 | 保留 |
|---:|---|---:|---:|---:|---:|---|
| 9000 | registered_opus | 175 | 175 | 129 | 129 | 是 |
| 9010 | registered_opus | 215 | 215 | 9 | 9 | 是 |
| 9011 | registered_opus | 295 | 295 | 24 | 24 | 是 |
| 9012 | registered_opus | 295 | 295 | 7 | 7 | 是 |
| 9020 | registered_opus | 295 | 295 | 32 | 32 | 是 |
| 9661 | generic_lds | 295 | 295 | 0 | 0 | 否 |
| 9663 | generic_lds | 295 | 295 | 0 | 0 | 否 |
| 13163 | long_epilogue_sync | 295 | 295 | 1 | 1 | 是 |
| 20000 | long_runtime | 295 | 295 | 9 | 9 | 是 |
| 20010 | short_runtime | 117 | 117 | 7 | 7 | 是 |
| 20011 | short_runtime | 117 | 117 | 14 | 14 | 是 |
| 20020 | n224_runtime | 63 | 63 | 1 | 1 | 是 |
| 20100 | long_runtime_grid | 295 | 295 | 32 | 32 | 是 |
| 20120 | short_fused_runtime | 117 | 117 | 0 | 0 | 否 |
| 20121 | short_fused_runtime_loop | 117 | 117 | 0 | 0 | 否 |
| 20122 | short_runtime_refine | 117 | 117 | 0 | 0 | 否 |
| 20123 | short_runtime_refine | 117 | 117 | 0 | 0 | 否 |
| 20124 | n224_runtime_loop | 63 | 63 | 2 | 1 | 是 |
| 20125 | short_runtime_unified | 117 | 117 | 2 | 2 | 是 |
| 20126 | short_runtime_unified | 117 | 117 | 4 | 2 | 是 |
| 20127 | short_runtime_group4 | 117 | 117 | 0 | 0 | 否 |
| 20128 | long_runtime_fused | 295 | 295 | 20 | 19 | 是 |
| 20129 | short_runtime_unified_small | 117 | 117 | 0 | 0 | 否 |
| 20130 | short_runtime_unified_small | 117 | 117 | 0 | 0 | 否 |
| 20131 | short_runtime_group4_cache2 | 117 | 117 | 2 | 2 | 是 |

原始 GPU 数值、计时与来源记录位于上级批次目录。summary.json 保存审计文件 SHA256 和后端有效/拒绝统计。
