"""Export the completed small diagnostic without changing the tuning ledger."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
probe = json.loads((HERE / "cache_probe_r2.json").read_text())
assert probe["status"] == "passed" and len(probe["records"]) == 36
assert all(r["err"] == 0 for r in probe["records"])
comparison = list(csv.DictReader((ROOT / "reports/opus_mxfp8_optimization_targets_20260925/baseline_timing_comparison_295.csv").open()))
cktile = [r for r in comparison if r["baseline_lib"] == "cktile"]
assert len(cktile) == 138 and all(r["matching_candidate_status"] == "valid" for r in cktile)
gaps = [float(r["same_candidate_sweep_vs_csv_pct"]) for r in cktile]
sources = {}
for number in (5283, 5440):
    url = f"https://api.github.com/repos/ROCm/aiter/pulls/{number}"
    req = urllib.request.Request(url, headers={"User-Agent": "Codex-read-only-audit"})
    with urllib.request.urlopen(req, timeout=20) as stream:
        data = json.load(stream)
    sources[str(number)] = {k: data[k] for k in ("html_url", "title", "body", "merge_commit_sha")}
(HERE / "upstream_pr_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
rows = []
for s in probe["summaries"]:
    shape = tuple(str(s[k]) for k in ("M", "N", "K"))
    baseline = next(r for r in comparison if tuple(r[k] for k in ("M", "N", "K")) == shape)
    rows.append({**{k: s[k] for k in ("M", "N", "K", "kid", "protocol", "median_us")},
                 "baseline_csv_us": baseline["baseline_csv_us"],
                 "same_candidate_full_sweep_us": baseline["same_candidate_sweep_us"]})
with (HERE / "timing_probe.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# CKTile 发布计时与本机计时差异调查", "",
    "结论：测试条件确实影响耗时，固定同一个 CKTile 二进制已验证输入复用和预热/迭代设置的影响；尚未完整复现上游历史计时，不能将全部差异归因于缓存。", "",
    "## 上游记录与本轮协议", "",
    "- 上游 `(4096,2048,7168)` / CKTile 27 / splitK=0 的 62.0 μs 由 commit `7b481fbc`（PR #5283）写入。kernelName 与本轮候选匹配。",
    "- [PR #5283](https://github.com/ROCm/aiter/pull/5283) 明确写了 MI355X/gfx950/256CU、50 warmup、200 iters、7 组 ABBA、中位数；没有给出缓存轮换方式、精确计时脚本或完整构建/时钟信息。",
    "- [PR #5440](https://github.com/ROCm/aiter/pull/5440) 的另外 10 行也采用 50 warmup / 200 iters / 7 组 ABBA；CSV 经多次独立更新，不是一份统一条件下新跑出的全量报告。",
    "- 本轮全量扫描为 5 warmup / 51 iters / 自动参数轮换，3 轮中位数；17 个边界 shape 用同口径的 5 轮确认批次替换。",
    "- 自动轮换通过 deepcopy 多份输入再轮流调用，减少反复读取同一地址时的缓存复用。共同的 B shuffle、E8M0 到 FP32 scale 解码不在计时内。",
    "- CKTile 27 是八 wave / ColumnMajor AQ / B preshuffle 分支，直接使用已准备的 FP32 scale 指针，未额外计入 scale transpose。", "",
    "## 固定二进制的本机对照", "",
    "使用完整扫描时冻结的 CKTile `.so` 及当时的输入生成器、profiler，核对 SHA256；物理 GPU 6 / PCI E5 / UUID GPU-56f0ab624008ec65。三个 shape、四套参数，每项测三轮并取中位数，36 项输出均通过原 FP32 误差区间检查。此次只用于诊断旧报告，不是同步 upstream 后的构建性能验收。", "",
    "| M | N | K | CKTile | 上游 μs | 原全量同候选 μs | 轮换 5/51 | 复用 5/51 | 轮换 50/200 | 复用 50/200 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for m,n,k in ((4096,2048,7168),(1280,7168,7168),(6144,6144,7168)):
    selected = [r for r in rows if (r["M"],r["N"],r["K"]) == (m,n,k)]
    r = selected[0]
    measures = {x["protocol"]: x["median_us"] for x in selected}
    values = [measures[p] for p in ("rotate_5_51", "reuse_5_51", "rotate_50_200", "reuse_50_200")]
    lines.append(f"| {m} | {n} | {k} | {r['kid']} | {float(r['baseline_csv_us']):.3f} | {float(r['same_candidate_full_sweep_us']):.3f} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
lines += ["",
    "`5/51` 和 `50/200` 分别表示 warmup/iters。复用对应 num_rotate_args=1；轮换对应原 profiler 的自动模式 num_rotate_args=0。其余相同。此对照没有复现上游 ABBA 脚本，也没有证明上游使用了热缓存。",
    "",
    "首项轮换 85.266 → 复用 71.886 μs，下降 15.69%；复用并改为 50/200 后为 68.874 μs，较 85.266 下降 19.22%，但仍高于上游 62.0。大 shape 未呈现相同幅度，不能统一乘一个修正系数。除首项外，原全量样本分布在不同物理 GPU，本次全部固定在 GPU 6，跨批次结果只作背景对照。", "",
    "## 对优化清单的影响", "",
    f"- 295 个已测 shape 中，上游选择 CKTile 的有 138 个；138 个同名、同 kid、同 splitK 候选均在本轮有效。相对上游记录的耗时差中位数为 {statistics.median(gaps):.3f}%，有 82 项慢超过 3%，26 项快超过 3%。",
    "- 87 项清单比较的是同一轮、同一张卡上的最佳有效 CKTile 与 OPUS。它不代表已经超过上游 CSV 每行所记录的历史时间。",
    "- 最终全后端选择与上游 CKTile kid 可能不同，例如首项本轮最佳是 CKTile 29；本页固定 CKTile 27 后比较，避免把选型变化混作时间变化。",
    "- upstream 同步后仅完成重新构建和正确性回归，尚未对当前构建做全量性能验收。", "",
    "## 留档", "",
    "- `cache_probe_r2.json`、`timing_probe.csv`：最终完整对照。`cache_probe_r2.log` 是执行日志。",
    "- `cache_probe.py`：可复现脚本，要求使用新的输出 JSON 名，禁止覆盖或编译被冻结的模块。",
    "- `cache_probe.json`：首次运行在首个 shape 后的瞬时空闲检查处中止的留档；没有与最终批次混用。",
    "- `upstream_pr_sources.json`：上游 PR 描述的读取快照。",
    "- 原始全量、确认批次、87 项优化清单及 production 配置均未更新。", "",
]
(HERE / "README.md").write_text("\n".join(lines))
summary = dict(status="passed", root_cause_scope="measurement settings confirmed influential; historical residual cause unresolved",
               current_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               baseline_cktile_shapes=len(cktile), matched_valid=len(cktile), median_change_pct=statistics.median(gaps),
               diagnostic_measurements=len(probe["records"]), baseline_kernel_id=27,
               script_sha256=hashlib.sha256((HERE / "cache_probe.py").read_bytes()).hexdigest(),
               measured_binary_sha256=probe["binary_sha256"])
(HERE / "manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
