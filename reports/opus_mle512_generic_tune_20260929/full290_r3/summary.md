本次通用版：M=1–512 的全部 290 个 shape，三轮实测中位数。
OPUS 胜出 234/290；落后 56 个，其中超过 1% 的有 48 个。
相对本轮最快有效 CK/CKTile/ASM 候选的几何平均加速为 1.159373×。

| Kernel | Pipeline | Tile | OPUS 内选中 | 超过外部 | 选中 shape 的几何平均加速 |
|---|---|---|---:|---:|---:|
| 9000 | existing |  | 2 | 2 | 1.3343× |
| 9010 | existing |  | 1 | 1 | 1.2397× |
| 9020 | existing |  | 2 | 1 | 0.9982× |
| 9021 | existing |  | 69 | 48 | 1.0916× |
| 9022 | existing |  | 7 | 7 | 1.1039× |
| 9023 | existing |  | 20 | 16 | 1.1198× |
| 9024 | existing |  | 7 | 5 | 1.0864× |
| 9030 | existing |  | 0 | 0 | — |
| 9040 | small_register | 16x32 | 28 | 19 | 1.0489× |
| 9041 | small_register | 16x16 | 33 | 33 | 1.6405× |
| 9042 | small_register | 32x32 | 35 | 30 | 1.1360× |
| 9043 | small_lds | 32x64 | 29 | 21 | 1.0830× |
| 9044 | small_lds | 64x64 | 31 | 28 | 1.1525× |
| 9045 | small_lds | 96x64 | 26 | 23 | 1.1514× |

2,884 项 OPUS 配置和额外 60 项运行时边界检查全部通过。外部池共 25,842 项。
外部正确性：{"opus": {"total": 2884, "passed": 2884, "rejected": 0}, "ck": {"total": 5220, "passed": 5052, "rejected": 168}, "cktile": {"total": 9570, "passed": 9345, "rejected": 225}, "asm": {"total": 11052, "passed": 0, "rejected": 11052}}

OPUS 使用原生随机 E8M0 scale；外部后端沿用原 tuner 的随机 FP32 scale。各自对照独立 FP32 reference，比较器保持不变。
9040–9045 每个 ID 固定一个 tile 和 pipeline，使用运行时 K。本轮全部重新编译、重新测试。
旧版历史成绩为 273/290、1.219300×；当时的 9040/9041 包含按 shape 分发的多个实现，与当前通用候选不同。

逐 shape 结果：comparison.csv；全部落后项：slow_shapes.csv；OPUS 调优表：tuned_opus.csv；跨后端调优表：tuned_all_backends.csv。
