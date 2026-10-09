# 2026-10-09：恢复 OPUS / FlyDSL 同卡比较，限定前四张物理 GPU

用户已明确要求停止测试。全部测试及等待进程停止，不自动重启；已有记录保留。

恢复来源为 [10 月 8 日比较记录](../opus_flydsl_comparison_20261008/README.md)，
最新旧设备检查保存于 2026-10-08 11:25:55 UTC。
745 项正式选型已完成：OPUS 693、ASM 34、CKTile 10、CK 8。
本目录接续尚未执行的 693 shape / 793 FlyDSL 配置同卡比较，先验证原来的 8 个代表 shape。

**最终状态（2026-10-09 02:56:04 UTC）：前四卡重新被外部任务占用，续跑停止。**
最新 [启动检查](preflight_latest.jsonl) 返回 75：计算峰值为 74/63/79/76%，
外部 PID 分别为 1134540/1135339/1134927/1134252，显存约 91.4–91.7 GiB。
没有留下我们的测试或后台等待进程；693 项全量比较未启动。

首次四个 worker 在 AOT 导入阶段失败，已在新比较脚本设置缺失的 `aiter.dtypes` alias。
第二次尝试完成了八个 shape 的初次数值检查，所有记录中的误差比均为 0，
但监控发现四卡分别有新外部 owner，四份相邻 receipt 全部标记 `valid_for_analysis: false`。
这些诊断检查及计时不能用于正式性能结论；smoke 仍需在持续独占的 GPU 上重跑。
尝试记录见 [claim_smoke.jsonl](claim_smoke.jsonl) 和 [claim_smoke_v2.jsonl](claim_smoke_v2.jsonl)。

正式 CSV 和 OPUS `.so` 的 SHA256 与旧 plan 完全一致，收据见 [recovery.json](recovery.json)。
原比较脚本、8 卡队列和全部历史结果保留。新目录保存四卡分片、修正后的比较脚本及独立输出。

## 本次 GPU 检查

[gpu_check.json](gpu_check.json) 保存 2026-10-09 02:50:49 UTC 的 AMD-SMI 检查：
每秒一次、连续 5 次活动采样，并查询物理卡进程归属。
这次早期快照中前四张物理卡均有其他进程及实际计算活动，不能满足独占计时要求。
随后 02:54:39 UTC 的 [只读启动检查](preflight_only.jsonl) 显示前四卡已释放：
连续 3 次采样均为 0%，进程列表为空、各使用 0.285 GiB。
检查返回 0，随后尝试四卡 smoke；结果和重新占用状态如上。

| 物理 SMI 编号 | HIP 编号 | PCI BDF | 5 次采样计算峰值 | 使用显存 GiB | 外部 host PID |
|---:|---:|---|---:|---:|---|
| 0 | 1 | 0000:05:00.0 | 70% | 102.17 | 1074167 |
| 1 | 3 | 0000:15:00.0 | 65% | 101.54 | 1076459 |
| 2 | 2 | 0000:65:00.0 | 68% | 102.03 | 1074527 |
| 3 | 0 | 0000:75:00.0 | 61% | 117.62 | 1073831、1077881 |

物理 4–7 同期利用率为 99–100%，续跑范围仅为前四卡。

## 四卡续跑

`run_four_gpu.py` 只允许物理 SMI 0、1、2、3，记录实际 HIP 编号及 BDF。
保持旧脚本的空进程列表、计算峰值不超过 2%、显存不超过 8 GiB、可用显存至少 16 GiB、
物理锁和 KFD owner 校验。默认只检查一轮，不满足条件时返回 75；不会留下等待进程。
每个 shape 的 OPUS 与全部对应 FlyDSL 配置在同一张卡、相同输入池中完成。

四个 full 分片共 693 项，四个 smoke 分片共 8 项。
新比较脚本修正 FlyDSL BMM 的 A-scale 描述符：从原 column-major 字节建立连续的零拷贝 view，
在计时前准备，不改变 scale 值、底层地址或 layout 契约；同时拒绝非有限输出。
这些适配完成 CPU 检查后，仍需实际 smoke 验证。

GPU 释放后在仓库根目录运行：

```bash
/opt/venv/bin/python3 reports/opus_flydsl_resume_20261009/run_four_gpu.py \
  --queue reports/opus_flydsl_resume_20261009/queue_smoke_v3.json \
  --log reports/opus_flydsl_resume_20261009/claim_smoke_v3.jsonl
```

检查四个 smoke 输出都完成、所有预期候选正确性通过后，再运行全量：

```bash
/opt/venv/bin/python3 reports/opus_flydsl_resume_20261009/run_four_gpu.py \
  --queue reports/opus_flydsl_resume_20261009/queue_four.json \
  --log reports/opus_flydsl_resume_20261009/claim_four.jsonl
```

启动器拒绝覆盖已有结果；重跑需使用新的结果和日志路径。新全量结果完成前，
历史 CSV 中的 399 胜 / 294 负、GM 1.040823× 仍只是历史比较，尚无 FlyDSL 同卡实测结论。
