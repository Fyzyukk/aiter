"""Render the local gap and all optimization targets from audited measurements."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
read = lambda name: list(csv.DictReader((HERE / name).open()))
stats = json.loads((HERE / "final_summary.json").read_text())
replay = json.loads((HERE / "replay.json").read_text())
build = json.loads((HERE / "build.json").read_text())
assert stats["status"] == replay["status"] == build["status"] == "passed"
rows = read("final_comparison.csv")
assert len(rows) == replay["shapes"] == 295
losses = [r for r in rows if r["opus_wins"] == "False"]
wins = [r for r in rows if r["opus_wins"] == "True"]
margin = lambda r: float(r["opus_slower_pct"])
strong_wins = [r for r in rows if margin(r) < -3]
close = [r for r in rows if abs(margin(r)) <= 3]
strong_losses = [r for r in rows if margin(r) > 3]
loss_gaps = [margin(r) for r in losses]
transition = stats["transitions_vs_previous_ledger"]
full = stats["full_sweep"]
summary = dict(
    status="passed", head=build["head"], shapes=len(rows), excluded=10,
    strict_wins=len(wins), strict_losses=len(losses),
    wins_over_3pct=len(strong_wins), within_3pct=len(close), losses_over_3pct=len(strong_losses),
    all_shapes_geomean_speedup=stats["geomean_reference_over_opus"],
    median_loss_pct=statistics.median(loss_gaps) if loss_gaps else None,
    max_loss_pct=max(loss_gaps) if loss_gaps else None,
    losses_by_best_opus=dict(Counter(r["opus"] for r in losses)),
    selected_backends=stats["selected_backends"],
    transitions_from_pre_sync_208_87=transition,
    round_inconsistent=sum(r["round_status_consistent"] != "True" for r in rows),
    formula="100 * (OPUS_us / best_valid_reference_us - 1)",
)
(HERE / "gap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
def write(name, local):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(local)
write("remaining_by_gap.csv", sorted(losses, key=margin, reverse=True))
detailed=[]
for row in rows:
    o,b=float(row["opus_us"]),float(row["reference_us"])
    detailed.append(dict(row,delta_us=o-b,opus_time_reduction_to_tie_pct=max(0,1-b/o)*100))
for name,selected in (("local_gap_details.csv",detailed),("optimization_targets.csv",[r for r in detailed if r["opus_wins"]=="False"])):
    with (HERE/name).open("w",newline="") as f:
        w=csv.DictWriter(f,list(detailed[0]));w.writeheader();w.writerows(selected)
write("transitions_from_previous.csv", [r for r in rows if r["status_vs_previous_ledger"] in ("new_win", "new_loss")])
def m_list(values):
    values = sorted(values)
    if len(values) >= 3 and all(b-a == 64 for a,b in zip(values,values[1:])):
        return f"{values[0]}～{values[-1]}，步长 64"
    return "、".join(map(str,values))

lines = ["# 当前分支的本机 GEMM 差距", "",
         f"当前 HEAD `{build['head'][:8]}`，在物理 GPU 4–6 上以当前源码重新构建、重新测量。295 个受支持的 M≥1024 shape 中，按中位数严格比较，OPUS 领先 **{len(wins)}** 项、落后 **{len(losses)}** 项；另 10 项超出现有寻址范围。", "",
         "| 差距分类 | shape 数 |", "|---|---:|",
         f"| OPUS 耗时低超过 3% | {len(strong_wins)} |",
         f"| 双方耗时差在 ±3% 内 | {len(close)} |",
         f"| OPUS 耗时高超过 3% | {len(strong_losses)} |", "",
         f"所有 shape 等权几何平均速度比（对手/OPUS）为 **{stats['geomean_reference_over_opus']:.4f}×**，不是模型总延迟加速比。",
         f"落后项目的耗时差中位数为 **{statistics.median(loss_gaps):.3f}%**，最大为 **{max(loss_gaps):.3f}%**。" if loss_gaps else "没有落后项目。", "",
         "差距定义为 `(OPUS_us / 对手_us - 1) × 100%`；正数表示 OPUS 更慢。OPUS kid 是该 shape 的最快合法 OPUS 候选，最终联合选择则由所有数值合格后端共同竞争。", "",
         "## 测量口径", "",
         "GPU 7 初始分片和 GPU 4 后续中断的补充分片因外部进程占用而整批排除。GPU 4 已成功完成的原分片保留 70 项，其 4 个边界 shape 连同中断批次的全部 25 项在 GPU 5/6 重新完整扫描。每个 shape 的候选和确认均在同一张 GPU 上。最终数值回放使用空闲 GPU 5/6，回放时间不影响性能结果。全部子批次与保留 shape 的哈希索引见 final_recovery.json、gpu*_complete_r3_run.json。", "",
         f"- 扫描 {full['candidates']:,} 个候选，每个合格候选测三轮；{stats['confirmation_shapes']} 个接近或轮次不一致的 shape 追加五轮确认。每个 shape 的所有候选固定在同一张卡。",
         "- 每轮使用当前仓库 run_perftest profiler，5 次预热、51 次迭代、自动轮换输入；不同候选的顺序随轮次轮换。",
         "- 公共 B shuffle、E8M0/FP32 scale 准备和参考计算不计时；后端内部转换计入，CPU launch 开销不计入。",
         "- 数据由同一 FP8/E8M0 数据集派生，所有后端先通过原 FP32 逐元素误差界检查才计时。",
         "- 五轮确认整体替换对应 shape 的三轮批次，没有跨批次挑最快值。±3% 和轮次不稳定项目应结合原始波动判断。",
         f"- 最终 295 个选择均通过原生 E8M0 接口回放。源码、CK 头文件及实际加载的二进制均冻结并核对哈希。共有 {summary['round_inconsistent']} 项在最终轮次中胜负不完全一致。", "",
         "## 各后端情况", "",
         "| 后端 | 候选数 | 全量三轮有效 | 排除 | 最终选中 shape |", "|---|---:|---:|---:|---:|"]
for lib in ("opus","ck","cktile","asm"):
    lines.append(f"| {lib} | {full['candidates_by_backend'].get(lib,0)} | {full['valid_candidates_by_backend'].get(lib,0)} | {full['rejected_candidates_by_backend'].get(lib,0)} | {stats['selected_backends'].get(lib,0)} |")
lines += ["", "排除的候选及数值/不支持原因见 `rejected_candidates.csv`；失败结果不参与性能选优。", "",
          "## 按最快 OPUS 分组的优化目标", "",
          "| OPUS kid | 落后数量 | 对应 pipeline |", "|---:|---:|---|"]
suffixes = {9000:"4wave",9010:"128x128",9011:"64x128",9012:"64x64",9020:"padded_m"}
for kid,suffix in suffixes.items():
    lines.append(f"| {kid} | {summary['losses_by_best_opus'].get('opus_'+str(kid),0)} | {suffix} |")
lines += ["", "9020 的 padded_m wrapper 复用 9000 的 4wave 计算主体。", "",
          "| N | K | 最快 OPUS | 数量 | M | OPUS 慢于对手 |", "|---:|---:|---:|---:|---|---:|"]
groups = defaultdict(list)
for r in losses:
    groups[int(r["N"]),int(r["K"]),r["opus"]].append(r)
group_rows = []
for (n,k,kid),local in sorted(groups.items()):
    gaps = [margin(r) for r in local]
    ms = m_list([int(r["M"]) for r in local])
    lines.append(f"| {n} | {k} | {kid.removeprefix('opus_')} | {len(local)} | {ms} | {min(gaps):.2f}～{max(gaps):.2f}% |")
    group_rows.append(dict(N=n,K=k,opus=kid,count=len(local),M=ms,min_gap_pct=min(gaps),max_gap_pct=max(gaps)))
with (HERE / "remaining_grouped.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=["N","K","opus","count","M","min_gap_pct","max_gap_pct"])
    writer.writeheader();writer.writerows(group_rows)
lines += ["", "## 落后项目逐项时间", "",
          "| M | N | K | OPUS | OPUS μs | 最快有效对手 | 对手 μs | OPUS 慢 |", "|---:|---:|---:|---:|---:|---|---:|---:|"]
for r in sorted(losses,key=margin,reverse=True):
    lines.append(f"| {r['M']} | {r['N']} | {r['K']} | {r['opus']} | {float(r['opus_us']):.3f} | {r['reference']} | {float(r['reference_us']):.3f} | {margin(r):.3f}% |")
lines += ["", "## 与同步 upstream 前的 208/87 对照", "",
          f"原负转胜 {transition.get('new_win',0)} 项，原胜转负 {transition.get('new_loss',0)} 项；旧时间没有参与本轮胜负判断。变化清单见 `transitions_from_previous.csv`。", "",
          "## 数据文件", "",
          "- [全部 295 个 shape 的最终差距和绝对时间差](local_gap_details.csv)",
          "- [优化目标及 OPUS 至少需要降时的百分比](optimization_targets.csv)",
          "- [全部落后项目](final_remaining_shapes.csv)",
          "- [按差距降序排列](remaining_by_gap.csv)",
          "- [最终选择](final_tuned.csv) / [每个 shape 最快 OPUS](final_opus_best.csv)",
          "- [全部候选扫描](profile.csv) / [未通过候选](rejected_candidates.csv)",
          "- [接近或轮次不稳定项目](final_close_shapes.csv)",
          "- [构建信息](build.json) / [完整扫描](full_r3_run.json) / [回放验证](replay.json)", "",
          "本轮结果对应上述本机 GEMM 调用和缓存模式；它不是模型端到端性能，也不是上游历史 CSV 时间的复现。生产源码和生产调度 CSV 未修改。", ""]
(HERE / "RESULTS.md").write_text("\n".join(lines))
p=HERE / "README.md"
s=p.read_text().replace("状态：构建和测量中，以 `workflow.json` 为准。",f"状态：测量、确认和回放已完成。OPUS {len(wins)} 胜 / {len(losses)} 负，详见 [RESULTS.md](RESULTS.md)。")
s=s.replace("物理 GPU 4–7 按 UUID 绑定，按 shape 分片。每个 shape 的全部后端在同一张卡上比较。",
            "最终有效计时来自物理 GPU 4–6 的空闲时段，按 UUID 绑定。GPU 7 初始批次和 GPU 4 后续中断的补测批次整批排除；迁移的 shape 在 GPU 5/6 完整重测。每个 shape 的全部候选和最终确认都在同一张卡上比较。最终数值回放使用空闲 GPU 5/6，回放时间不替代计时。")
p.write_text(s)
print(json.dumps(summary,indent=2))
