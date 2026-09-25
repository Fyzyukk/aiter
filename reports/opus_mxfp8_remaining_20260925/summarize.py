"""Reconstruct the progress ledger; read saved measurements without using a GPU."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "full": "reports/opus_mxfp8_small_opt_20260923/performance_20260923_gpu7/full_choices.csv",
    "narrow": "reports/opus_9011_schedule_20260923/joint_narrow_choices.csv",
    "9012": "reports/opus_9012_flow_20260924/comparison.csv",
    "final": "reports/opus_9010_9011_opt_20260924/harness/final_r5_choices.csv",
    "close": "reports/opus_9010_9011_opt_20260924/harness/close_margin_r5_choices.csv",
    "regression": "reports/opus_9010_9011_opt_20260924/validation/historical_rivals_r5b.json",
    "prior": "reports/opus_9010_9011_remote_handoff_20260924/remaining_all_89.csv",
    "latest": "reports/opus_9030_wide_n_20260924/resume_20260925_comparison.csv",
    "raw": "reports/opus_9030_wide_n_20260924/resume_20260925_r5_raw.csv",
    "checks": "reports/opus_9030_wide_n_20260924/resume_20260925_r5_correctness.csv",
    "latest_run": "reports/opus_9030_wide_n_20260924/resume_20260925_r5_run.json",
    "final_run": "reports/opus_9010_9011_opt_20260924/harness/final_r5_run.json",
    "variants": "reports/opus_9030_wide_n_20260924/variants.json",
    "deferred": "reports/opus_mxfp8_padded_m_20260922/deferred_shapes.csv",
    "cktile_instances": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_cktile_instance.py",
    "asm_evidence": "reports/opus_mxfp8_padded_m_20260922/joint_tune_20260923_gpu7/README.md",
}


def csv_rows(name):
    with (ROOT / SOURCES[name]).open(newline="") as f:
        return list(csv.DictReader(f))


def json_data(name):
    return json.loads((ROOT / SOURCES[name]).read_text())


def shape(row):
    return tuple(int(row[k]) for k in ("M", "N", "K"))


ledger = {}


def record(row, opus, opus_us, reference, reference_us, source, evidence):
    key = shape(row)
    o, r = float(opus_us), float(reference_us)
    assert math.isfinite(o) and o > 0 and math.isfinite(r) and r > 0
    ledger[key] = dict(
        zip(("M", "N", "K"), key),
        opus=opus, opus_us=o, reference=reference, reference_us=r,
        opus_slower_pct=(o / r - 1) * 100, opus_wins=o < r,
        evidence=evidence, source=SOURCES[source],
    )


for r in csv_rows("full"):
    record(r, r["new_opus"], r["new_opus_us"], r["reference"], r["reference_us"], "full", "historical")
assert len(ledger) == 295
counts = [("full", sum(not r["opus_wins"] for r in ledger.values()))]
for r in csv_rows("narrow"):
    record(r, r["new_best_opus"], r["new_best_opus_us"], r["best_reference"], r["best_reference_us"], "narrow", "historical")
counts.append(("narrow", sum(not r["opus_wins"] for r in ledger.values())))
for r in csv_rows("9012"):
    record(r, "opus_" + r["opus_kid"], r["opus_us"], "ck_" + r["ck_kid"] + "_split0", r["ck_us"], "9012", "accepted_9012_ck_subset")
counts.append(("9012", sum(not r["opus_wins"] for r in ledger.values())))
for source in ("final", "close"):
    for r in csv_rows(source):
        record(r, r["opus"], r["opus_us"], r["reference"], r["reference_us"], source, "accepted_finalists_r5")
counts.append(("final_and_close", sum(not r["opus_wins"] for r in ledger.values())))
for r in json_data("regression")["choices"]:
    med = r["medians_us"]
    opus = min((n for n in med if n.startswith("opus_")), key=med.get)
    ref = min((n for n in med if not n.startswith("opus_")), key=med.get)
    record(r, opus, med[opus], ref, med[ref], "regression", "accepted_saved_rival_r5")
prior = {shape(r) for r in csv_rows("prior")}
assert {k for k, r in ledger.items() if not r["opus_wins"]} == prior

# Independently recompute today's 49 candidate medians from all 245 raw rows.
raw = csv_rows("raw")
checks = csv_rows("checks")
assert len(raw) == 245 and all(r["status"] == "passed" for r in raw)
assert len(checks) == 450 and all(r["status"] == "passed" and r["guards"] == "True" for r in checks)
samples = defaultdict(list)
for r in raw:
    samples[shape(r), r["name"]].append(r)
medians = defaultdict(dict)
for (key, name), rs in samples.items():
    assert len(rs) == 5 and {int(r["round"]) for r in rs} == set(range(5))
    medians[key][name] = statistics.median(float(r["us"]) for r in rs)
assert len(samples) == 49
variants = {r["name"]: r["id"] for r in json_data("variants")}
for r in csv_rows("latest"):
    key = shape(r)
    assert key in prior
    med = medians[key]
    ref = min((n for n in med if n.startswith(("ck_", "cktile_", "asm_"))), key=med.get)
    opus = min((n for n in med if n.startswith("opus_")), key=med.get)
    wide = min((n for n in med if n in variants), key=med.get)
    assert (ref, opus, variants[wide]) == (r["reference"], r["existing"], int(r["wide_id"]))
    for name, field in ((ref, "reference_us"), (opus, "existing_us"), (wide, "wide_us")):
        assert math.isclose(med[name], float(r[field]), rel_tol=1e-12)
    record(r, opus, med[opus], ref, med[ref], "latest", "current_subset_r5_20260925")
    ledger[key].update(experimental_opus=variants[wide], experimental_us=med[wide],
                       best_including_experiment_slower_pct=(min(med[opus], med[wide]) / med[ref] - 1) * 100)
    assert min(med[opus], med[wide]) > med[ref]
counts.append(("latest", sum(not r["opus_wins"] for r in ledger.values())))

# Verify that the accepted and latest measurements still match workspace sources.
audits = {}
for source, field in (("final_run", "source_sha256"), ("regression", "source_sha256"), ("latest_run", "protected_sha256")):
    manifest = json_data(source)
    assert manifest["status"] == "passed"
    hashes = manifest[field]
    for p, expected in hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == expected, p
    audits[source] = {"matched_files": len(hashes), "mismatches": []}

rows = sorted(ledger.values(), key=lambda r: (r["N"], r["K"], r["M"]))
remaining = [r for r in rows if not r["opus_wins"]]
assert len(rows) == 295 and len(remaining) == 89
assert {shape(r) for r in remaining} == prior
assert Counter(r["evidence"] for r in remaining) == {"historical": 84, "current_subset_r5_20260925": 5}
backend_counts = {b: sum(r["reference"].split("_")[0] == b for r in remaining) for b in ("ck", "cktile", "asm")}
groups = defaultdict(list)
for r in remaining:
    groups[r["N"], r["K"]].append(r)
group_rows = [dict(N=n, K=k, count=len(rs), M=";".join(str(r["M"]) for r in rs),
                   min_slower_pct=min(r["opus_slower_pct"] for r in rs),
                   max_slower_pct=max(r["opus_slower_pct"] for r in rs))
              for (n, k), rs in groups.items()]


def write_csv(filename, data):
    fields = list(dict.fromkeys(k for row in data for k in row))
    with (HERE / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(data)


write_csv("all_shapes_ledger.csv", rows)
write_csv("remaining_shapes.csv", remaining)
write_csv("remaining_groups.csv", group_rows)
deferred = csv_rows("deferred")
assert len(deferred) == 10
summary = dict(
    kind="offline_progress_ledger", current_full_performance_retest=False,
    total_model_shapes=305, performance_shapes=295, recorded_opus_wins=206,
    remaining_performance_shapes=89, current_retested_losses=5, historical_pending_retest=84,
    deferred_addressing_shapes=len(deferred), best_reference_backend_counts=backend_counts,
    best_reference_kernel_counts=dict(Counter(r["reference"] for r in remaining)),
    best_recorded_opus_counts=dict(Counter(r["opus"] for r in remaining)),
    loss_count_by_overlay=counts, current_source_audits=audits,
    latest_raw_audit=dict(timing_rows=len(raw), candidate_medians=len(samples), correctness_checks=len(checks), rounds=5),
    sources={k: dict(path=p, sha256=hashlib.sha256((ROOT / p).read_bytes()).hexdigest()) for k, p in SOURCES.items()},
)
(HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

lines = [
    "# 2026-09-25：OPUS 尚未超过有效参考的 shape",
    "",
    "恢复历史全量、后续窄 N、9012、9010/9011 验收及今天 9030–9033 复测记录后，仍有 **89 项未超过记录**。按每项最快有效参考归属：**CK 0、CKTile 89、ASM 0**。其中 **5 项今天复测仍落后，84 项为历史落后、待当前版性能复测**。",
    "",
    "范围为 gfx950 / 256 CU、原 305 个 M≥1024 模型 shape：295 个有性能记录，206 项有超过记录，另 10 项因 2 GiB 寻址限制暂缓。206/89 是逐项更新的进度账本，不能作为当前版一次全量测量的胜负结果。",
    "",
    "统计按每个 shape 的最佳已测合法 OPUS 与同批最快有效参考比较，不按某个 OPUS kid 单独计数，也不是 CK/CKTile 分别对 OPUS 的独立胜场。每次更新整行替换，绝不跨批混用耗时。",
    "",
    "原 101 项经 9010/9011 正式验收新增解决 12 项后变为 89：N=2048/K=7168 的 M=1088、1152、1216、1280、1344、1408、1472、1600、1664、1728，以及 N=768/K=7168 的 M=6144、8192。今天五项复测未新增胜场。",
    "",
    "ASM 在已保存的 295-shape 扫描中 1770 个配置全部未通过原数值标准；因此没有有效胜出项，不代表已证明 OPUS 比 ASM 的裸计时快。数值诊断涉及 BF16 截断输出。",
    "",
    "所有剩余参考 ID 均属于 CKTile 192×256×128、8-wave 系列：11/27/28/29，以及 AQRowMajor 的 12/30/31/32。",
    "",
    "## 五项最新同批复测",
    "",
    "时间为五轮中位数，单位 µs；慢于参考 = (OPUS / 参考 − 1) × 100%。本轮正式候选覆盖合法 9000/9010/9011，参考为 CKTile 11/27/28/29 和保存的 CK 决赛候选；没有重新穷举所有后端，也未重测 9012/9020。此前正式 17 项比较覆盖全部合法 OPUS，这五项也均未胜出。",
    "",
    "| M | N | K | 最快参考 | 参考 µs | 本轮现有 OPUS | OPUS µs | 慢于参考 | 含实验候选后的最小差距 |",
    "|---:|---:|---:|---|---:|---|---:|---:|---:|",
]
for r in remaining:
    if r["evidence"] != "historical":
        lines.append(f'| {r["M"]} | {r["N"]} | {r["K"]} | {r["reference"]} | {r["reference_us"]:.3f} | {r["opus"]} | {r["opus_us"]:.3f} | {r["opus_slower_pct"]:.2f}% | {r["best_including_experiment_slower_pct"]:.2f}% |')
lines += [
    "",
    "实验 9033 在 (1088,7168,16384) 将差距从 7.06% 缩小到 3.25%，在 (1088,6144,7168) 从 8.81% 缩小到 8.37%；其余三项仍以现有 OPUS 更快。9030–9033 尚未集成。0.59% 的长 K 差距接近测量波动，应先确认稳定性。",
    "",
    "## 全部 89 项分组",
    "",
    "以下差距范围来自每项各自最新采用的同批记录；84 项仍是历史数据。区间形式的 M 步长均为 64。",
    "",
    "| N | K | 数量 | M | OPUS 耗时高出范围 |",
    "|---:|---:|---:|---|---:|",
]
for g in group_rows:
    ms = [int(m) for m in g["M"].split(";")]
    mtext = f"{ms[0]}～{ms[-1]}，步长64" if len(ms) > 1 and all(b-a == 64 for a, b in zip(ms, ms[1:])) else "、".join(map(str, ms))
    lines.append(f'| {g["N"]} | {g["K"]} | {g["count"]} | {mtext} | {g["min_slower_pct"]:.2f}%～{g["max_slower_pct"]:.2f}% |')
lines += [
    "",
    "N=7168 共 61 项，是剩余数量最多的一组；其中 K=384/768/1024 共 25 项。按历史最佳 OPUS 分：9020 有 58 项，9000 有 26 项，9011 有 3 项，9010 有 2 项。后续应先重测 84 项历史记录，再确定实际优化范围。",
    "",
    "另 10 项寻址暂缓：N=65536/K=1536，M=16384、20480、24576、28672、32768、40960、49152、57344、65536；以及 (65536,16384,1536)。它们不计入 89 项性能落后。",
    "",
    "## 数据与核对",
    "",
    "- [89 项逐 shape CSV](remaining_shapes.csv)：参考/OPUS 时间、差距、实验结果及来源。",
    "- [295 项进度账本](all_shapes_ledger.csv) · [分组 CSV](remaining_groups.csv) · [统计及来源哈希](summary.json)。",
    "- [复算脚本](summarize.py)：逐项重建后与旧 89 项清单集合核对，重新计算今天 245 条原始计时的 49 个五轮中位数，并核对 450 条正确性记录。",
    "- 当前源码与正式验收 107 项、历史回归 108 项、今天运行保护 39 项文件哈希均匹配。",
    "- [今天测试恢复记录](../opus_resume_20260925.md) · [此前完整状态](../opus_9010_9011_all_shapes_status_20260924.md)。",
    "",
    "本次只恢复记录并离线复算；没有发起新的 GPU 性能测试。复算命令：`python reports/opus_mxfp8_remaining_20260925/summarize.py`。",
]
(HERE / "README.md").write_text("\n".join(lines) + "\n")
print(json.dumps({k: summary[k] for k in ("recorded_opus_wins", "remaining_performance_shapes", "current_retested_losses", "historical_pending_retest", "best_reference_backend_counts", "loss_count_by_overlay")}, ensure_ascii=False, indent=2))
