"""Export every unresolved shape, per-OPUS times, and published baseline context."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "reports/opus_mxfp8_full_retune_4gpu_20260925"
BASE = ROOT / "aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
KIDS = (9000, 9010, 9011, 9012, 9020)
SOURCES = {
    9000: "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh",
    9010: "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_128x128_gfx950.cuh",
    9011: "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh",
    9012: "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh",
    9020: "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh",
}


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def shape(r):
    return tuple(int(r[k]) for k in ("M", "N", "K"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


base = {shape(r): r for r in read(BASE) if r["gfx"] == "gfx950" and
        r["cu_num"] == "256" and int(r["M"]) >= 1024}
comparisons = read(RUN / "final_comparison.csv")
remaining = read(RUN / "final_remaining_shapes.csv")
assert len(base) == 305 and len(comparisons) == 295 and len(remaining) == 87
assert all(shape(r) in base for r in comparisons)
profile = defaultdict(list)
for r in read(RUN / "profile.csv"):
    profile[shape(r), r["libtype"], r["kernelName"], int(r["splitK"])].append(r)
baseline_rows = []
for r in comparisons:
    b = base[shape(r)]
    candidates = profile[shape(r), b["libtype"], b["kernelName"], int(b["splitK"])]
    candidates.sort(key=lambda c: c["kernelId"] != b["kernelId"])
    candidate = candidates[0] if candidates else None
    us = float(candidate["us"]) if candidate and candidate["us"] else None
    baseline_rows.append(dict(
        zip(("M", "N", "K"), shape(r)), baseline_lib=b["libtype"],
        baseline_kid=int(b["kernelId"]), baseline_splitK=int(b["splitK"]),
        baseline_kernel_name=b["kernelName"], baseline_csv_us=float(b["us"]),
        matching_candidate_status="valid" if us else "rejected" if candidate else "not_enumerated",
        same_candidate_sweep_kid=candidate["kernelId"] if candidate else "",
        same_candidate_sweep_us=us if us else "",
        same_candidate_sweep_vs_csv_pct=(us/float(b["us"])-1)*100 if us else "",
        best_valid_reference=r["reference"], best_valid_reference_us=float(r["reference_us"]),
        best_opus=r["opus"], best_opus_us=float(r["opus_us"]),
        same_run_opus_slower_pct=float(r["opus_slower_pct"]),
        measurement_phase=r["measurement_phase"], rounds=int(r["rounds"]), gpu=int(r["gpu"]),
    ))
write("baseline_timing_comparison_295.csv", baseline_rows)
baseline_by = {shape(r): r for r in baseline_rows}

targets = []
for r in remaining:
    kid = int(r["opus"].split("_")[1])
    valid = {k: float(r[f"opus_{k}_us"]) for k in KIDS if r[f"opus_{k}_us"]}
    assert valid[kid] == min(valid.values()) and float(r["opus_us"]) > float(r["reference_us"])
    row = dict(zip(("M", "N", "K"), shape(r)), best_opus_kid=kid,
               best_opus_us=float(r["opus_us"]), reference=r["reference"],
               reference_us=float(r["reference_us"]),
               gap_us=float(r["opus_us"])-float(r["reference_us"]),
               opus_slower_pct=float(r["opus_slower_pct"]),
               opus_time_reduction_to_tie_pct=(1-float(r["reference_us"])/float(r["opus_us"]))*100,
               **{f"opus_{k}_us": valid.get(k, "") for k in KIDS},
               rounds=int(r["rounds"]), opus_faster_rounds=int(r["opus_faster_rounds"]),
               round_status_consistent=r["round_status_consistent"],
               measurement_phase=r["measurement_phase"], gpu=int(r["gpu"]),
               pipeline="csrc/opus_gemm/include/gfx950/" + SOURCES[kid],
               implementation_pipeline="csrc/opus_gemm/include/gfx950/" + SOURCES[9000 if kid == 9020 else kid],
               baseline_csv_lib=baseline_by[shape(r)]["baseline_lib"],
               baseline_csv_kid=baseline_by[shape(r)]["baseline_kid"],
               baseline_csv_us=baseline_by[shape(r)]["baseline_csv_us"])
    targets.append(row)
targets.sort(key=lambda r: (r["N"], r["K"], r["M"]))
write("optimization_targets_87.csv", targets)
write("optimization_targets_87_by_gap.csv", sorted(targets, key=lambda r: -r["opus_slower_pct"]))
groups = defaultdict(list)
for r in targets:
    groups[r["N"], r["K"], r["best_opus_kid"]].append(r)
group_rows = []
for (n, k, kid), rows in sorted(groups.items()):
    group_rows.append(dict(N=n, K=k, best_opus_kid=kid, count=len(rows),
                           M="、".join(str(r["M"]) for r in rows),
                           min_slower_pct=min(r["opus_slower_pct"] for r in rows),
                           max_slower_pct=max(r["opus_slower_pct"] for r in rows)))
assert sum(r["count"] for r in group_rows) == 87
write("grouped_by_opus.csv", group_rows)

lines = ["# 87 个 OPUS 尚未超过的 shape", "",
         "来源：四卡全量扫描加 17 项五轮确认，时间在 upstream/main 同步之前测得。同步后只完成了重新编译和正确性回归，当前分支尚未重新验收全量性能。", "",
         "本页的 OPUS kid 是该 shape 在全部合法 OPUS 候选中的最快者。联合选型中，这 87 项均由 CKTile 胜出。差距定义为 `(OPUS_us / CKTile_us - 1) × 100%`；若要算 OPUS 本身需要降时多少，CSV 另有 `opus_time_reduction_to_tie_pct` 列。", "",
         "上游 CSV 的 us 是发布时的记录，本轮 us 是实际重测结果；shape 一致不能推出计时一致。上游记录与本轮环境、版本、计时条件未对齐，不用其历史 us 判定下面的胜负。", "",
         "| 最佳 OPUS kid | 剩余数 | pipeline |", "|---:|---:|---|"]
counts = Counter(r["best_opus_kid"] for r in targets)
for kid in KIDS:
    lines.append(f"| {kid} | {counts[kid]} | [{SOURCES[kid]}](../../csrc/opus_gemm/include/gfx950/{SOURCES[kid]}) |")
lines += ["", "9020 通过 padded_m traits 复用 9000 的 4wave 主体；对应 wrapper 只有 include。9000/9020 合计覆盖 73/87 项。", "",
          "## 分组清单：覆盖全部 87 项", "",
          "| N | K | 最佳 OPUS | 数量 | M | 慢于 CKTile |", "|---:|---:|---:|---:|---|---:|"]
group_lines = []
for r in group_rows:
    gap = f"{r['min_slower_pct']:.2f}%" if r['count'] == 1 else f"{r['min_slower_pct']:.2f}～{r['max_slower_pct']:.2f}%"
    group_lines.append(f"| {r['N']} | {r['K']} | {r['best_opus_kid']} | {r['count']} | {r['M']} | {gap} |")
lines += group_lines
(HERE / "grouped_table.md").write_text("\n".join(group_lines) + "\n")
lines += ["", "## 逐项时间", "",
          "单位 μs；所有 CKTile splitK 均为 0。完整 CSV 保留全部五个 OPUS kid 的实测时间；空值表示该 shape 没有该 kid 的合法候选。", "",
          "| M | N | K | 最佳 OPUS | OPUS μs | CKTile kid | CKTile μs | OPUS 慢 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in targets:
    ref_kid = r["reference"].split("_")[1]
    lines.append(f"| {r['M']} | {r['N']} | {r['K']} | {r['best_opus_kid']} | {r['best_opus_us']:.3f} | {ref_kid} | {r['reference_us']:.3f} | {r['opus_slower_pct']:.2f}% |")
lines += ["", "- [完整 87 项及全部 OPUS 时间](optimization_targets_87.csv)",
          "- [按差距从大到小排序](optimization_targets_87_by_gap.csv)",
          "- [295 项与上游发布计时的对照](baseline_timing_comparison_295.csv)", ""]
(HERE / "README.md").write_text("\n".join(lines))
manifest = dict(status="passed", shapes=87, by_best_opus={str(k): counts[k] for k in KIDS},
                same_candidate_baseline_status=dict(Counter(r["matching_candidate_status"] for r in baseline_rows)),
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                               (BASE, RUN / "final_comparison.csv", RUN / "final_remaining_shapes.csv", RUN / "profile.csv")},
                outputs={p.name: sha(p) for p in HERE.iterdir() if p.suffix in (".csv", ".md")})
(HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k not in ("source_sha256", "outputs")}, indent=2))
