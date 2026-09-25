"""Render final audited measurements and remaining-shape groups in Chinese."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
read = lambda name: list(csv.DictReader((HERE / name).open()))
stats = json.loads((HERE / "final_summary.json").read_text())
replay = json.loads((HERE / "replay.json").read_text())
assert stats["status"] == replay["status"] == "passed" and replay["shapes"] == 295
full = stats["full_sweep"]
rows = read("final_comparison.csv")
losses = [r for r in rows if r["opus_wins"] == "False"]
transition = stats["transitions_vs_previous_ledger"]


def m_list(values):
    values = sorted(values)
    if len(values) >= 3 and all(b-a == 64 for a, b in zip(values, values[1:])):
        return f"{values[0]}～{values[-1]}，步长 64"
    return "、".join(map(str, values))


lines = [
    "# 更新 9010/9011/9012 后的全量实测结果", "",
    f"物理 GPU 4–7 已完成全量复测：295 个受支持的 M≥1024 shape，OPUS 领先 **{stats['opus_wins']}** 个，仍落后 **{stats['remaining']}** 个。另 10 个原模型 shape 超出现有寻址范围，未进行性能验收。", "",
    f"全量扫描覆盖 27,690 个候选，每个数值合格候选测三轮；三轮原始统计为 {full['opus_wins']} 胜、{full['remaining']} 负。随后对差距 ≤3% 或三轮胜负不一致的 {stats['confirmation_shapes']} 个 shape，在各自原 GPU 上追加五轮确认，包含全部合法 OPUS 及全量扫描中距最快有效对手 ≤5% 的参考候选。", "",
    "本文最终统计用五轮确认批次整体替换对应 shape，其余保留三轮批次；不在两个批次间挑选最小时间。所有候选先按原 FP32 误差区间检查正确性，再计时；每轮 warmup=5、iters=51，使用原 profiler 和自动输入轮换。每个 shape 的所有对比都在同一张卡上完成。", "",
    f"剩余 {stats['remaining']} 项中，{stats['losses_within_3pct']} 项的差距 ≤3%，{stats['losses_over_3pct']} 项的差距 >3%。这里的胜负按实测中位数严格比较，≤3% 项仍可能受测量波动影响。", "",
    "## 剩余清单", "", "| N | K | 剩余数量 | 对应 M |", "|---:|---:|---:|---|",
]
groups = defaultdict(list)
for r in losses:
    groups[int(r["N"]), int(r["K"])].append(int(r["M"]))
for (n, k), ms in sorted(groups.items()):
    lines.append(f"| {n} | {k} | {len(ms)} | {m_list(ms)} |")
lines += ["", "逐项 kernel、时间和差距见 [final_remaining_shapes.csv](final_remaining_shapes.csv)。", "",
          "## 候选正确性与最终选型", "",
          "| 后端 | 完整候选数 | 三轮有效候选数 | 排除候选数 | 最终胜出 shape 数 |",
          "|---|---:|---:|---:|---:|"]
for lib in ("opus", "ck", "cktile", "asm"):
    lines.append(f"| {lib} | {full['candidates_by_backend'].get(lib,0)} | {full['valid_candidates_by_backend'].get(lib,0)} | {full['rejected_candidates_by_backend'].get(lib,0)} | {stats['selected_backends'].get(lib,0)} |")
lines += ["", "ASM 包含全部合法 split-K 候选；其不合格结果保存在原始数值检查和 [rejected_candidates.csv](rejected_candidates.csv)，未用失败结果参与性能选优。", "",
          "| OPUS kid | OPUS 内部最优 shape 数 | 战胜全部有效参考的 shape 数 |",
          "|---:|---:|---:|"]
for kid in (9000, 9010, 9011, 9012, 9020):
    key = f"opus_{kid}"
    lines.append(f"| {kid} | {stats['best_opus_kids'].get(key,0)} | {stats['winning_opus_kids'].get(key,0)} |")
lines += ["", "## 与历史台账对照", "",
          f"历史台账为 206 胜 / 89 负。对照本次全量结果：原负转胜 {transition.get('new_win',0)} 项，原胜转负 {transition.get('new_loss',0)} 项，持续领先 {transition.get('still_wins',0)} 项，持续落后 {transition.get('still_loses',0)} 项。旧时间没有参与本轮选优。", ""]
for label, status in (("原负转胜", "new_win"), ("原胜转负", "new_loss")):
    local = defaultdict(list)
    for r in rows:
        if r["status_vs_previous_ledger"] == status:
            local[int(r["N"]), int(r["K"])].append(int(r["M"]))
    for (n, k), ms in sorted(local.items()):
        lines.append(f"- {label}：N={n}，K={k}，M={m_list(ms)}。")
lines += ["", "## 交付与验证", "",
          "- [final_tuned.csv](final_tuned.csv)：295 行最终联合选型，原生 E8M0 tuner 的 13 列格式。",
          "- [final_opus_best.csv](final_opus_best.csv)：每个 shape 的最佳 OPUS。",
          "- [final_comparison.csv](final_comparison.csv)：最终比较、GPU 来源、批次和轮次一致性；确认批次只列实际复测的参考候选。",
          "- [comparison.csv](comparison.csv) / [profile.csv](profile.csv)：完整三轮扫描的逐 shape 对比和所有候选记录。",
          "- [final_close_shapes.csv](final_close_shapes.csv)：最终差距 ≤3% 或轮次胜负不一致的 shape。",
          "- [deferred_shapes.csv](deferred_shapes.csv)：10 个寻址范围排除项。",
          "- [full_r3_run.json](full_r3_run.json) / [final_summary.json](final_summary.json)：完整审计、四卡原始记录索引和最终统计。",
          "- [replay.json](replay.json)：295 行最终选择全部通过原生 E8M0 接口回放，误差比例为 0；回放时间不替换 tune 时间。", "",
          "新源码适配已通过 185 项专项测试（99 GPU、86 CPU/接口）；四卡扫描、五轮确认和回放均核对冻结源码与二进制哈希。实际注册配置为 9010=S3/K+2、9011=S3/K+2、9012=S4/K+3、直接 BF16。构建及源码适配详情见 [构建记录](../opus_mxfp8_full_retune_20260925/README.md)。", ""]
(HERE / "RESULTS.md").write_text("\n".join(lines))
readme = HERE / "README.md"
text = readme.read_text()
text = text.replace("状态：正在运行。以 `launcher.json`、四个 `gpu*_r3_run.json` 和完成后的 `summary.json` 为准。",
                    f"状态：已完成。全量扫描、{stats['confirmation_shapes']} 项五轮确认及 295 项选择回放全部通过。最终 OPUS {stats['opus_wins']} 胜 / {stats['remaining']} 负，详见 [RESULTS.md](RESULTS.md)。")
readme.write_text(text)
print("Wrote", HERE / "RESULTS.md")
