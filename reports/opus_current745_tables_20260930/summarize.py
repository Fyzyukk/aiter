#!/usr/bin/env python3
"""Summarize existing normal-tuner records; no GPU execution or source edits."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRENT = ROOT / "reports/opus_consolidate_20260930"
PREVIOUS = ROOT / "reports/opus_full745_register_20260930"
BASELINE = ROOT / "aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
SHAPE = ("gfx", "cu_num", "M", "N", "K")
EXTERNAL = ("ck", "cktile", "asm")


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert all(None not in r and None not in r.values() for r in rows), path
    return rows


def key(row):
    return (row["gfx"], *(int(row[k]) for k in SHAPE[1:]))


def valid(row):
    us, error = float(row["us"]), float(row["errRatio"])
    limit = 0 if row["libtype"] == "opus" else .05
    return math.isfinite(us) and us > 0 and math.isfinite(error) and 0 <= error <= limit


def write(name, rows):
    # UTF-8 BOM keeps the provenance labels readable when opened in Excel.
    with (HERE / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(ratios):
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return dict(shapes=len(ratios), faster=sum(r < 1 for r in ratios),
                slower=sum(r > 1 for r in ratios), tied=sum(r == 1 for r in ratios),
                geomean_latency_reduction_pct=100 * (1 - gm), geomean_speedup=1 / gm,
                max_latency_reduction_pct=100 * (1 - min(ratios)),
                min_latency_reduction_pct=100 * (1 - max(ratios)))


def timing_changes(ratios):
    """Current / historical latency; positive changes mean current is slower."""
    changes = [100 * (ratio - 1) for ratio in ratios]
    return dict(
        shapes=len(ratios),
        geomean_change_pct=100 * (math.exp(statistics.mean(map(math.log, ratios))) - 1),
        median_change_pct=statistics.median(changes),
        within_5pct=sum(abs(change) <= 5 for change in changes),
        faster_over_5pct=sum(change < -5 for change in changes),
        slower_over_5pct=sum(change > 5 for change in changes),
        min_change_pct=min(changes), max_change_pct=max(changes),
    )


def original_config_match(original, rows):
    """Require both the numeric configuration and its recorded kernel name."""
    identity = (original["libtype"], int(original["kernelId"]), int(original["splitK"]))
    same_id = [r for r in rows if
               (r["libtype"], int(r["kernelId"]), int(r["splitK"])) == identity]
    exact = [r for r in same_id if r["kernelName"] == original["kernelName"]]
    assert len(exact) <= 1
    if exact:
        return ("matched" if valid(exact[0]) else "measured_but_invalid"), exact[0], same_id
    if original["libtype"] not in EXTERNAL:
        return "backend_not_in_sweep", None, same_id
    return ("kernel_name_mismatch" if same_id else "configuration_not_measured"), None, same_id


def logical_shape(row):
    return " x ".join(str(row[dim]) for dim in ("M", "N", "K"))


def compact_rows(rows, baseline_field, baseline_heading):
    return [{"Logical shape": logical_shape(row),
             baseline_heading: f"{row[baseline_field] / 1000:.6f}",
             "OPUS (ms)": f"{row['opus_us'] / 1000:.6f}",
             "Speedup": f"{row[baseline_field] / row['opus_us']:.2f}x"}
            for row in rows]


def markdown_table(rows):
    headings = list(rows[0])
    return ["| " + " | ".join(headings) + " |",
            "|---|" + "---:|" * (len(headings) - 1),
            *("| " + " | ".join(str(row[col]) for col in headings) + " |" for row in rows)]


def main():
    previous_run = json.loads((PREVIOUS / "run.json").read_text())
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == previous_run["input_sha256"]
    assert json.loads((CURRENT / "verification.json").read_text())["status"] == "passed"
    baseline_rows = [r for r in read(BASELINE) if r["gfx"] == "gfx950" and int(r["cu_num"]) == 256]
    baseline = {key(r): r for r in baseline_rows}
    assert len(baseline_rows) == len(baseline) == 745 and all(valid(r) for r in baseline_rows)

    mapping = read(CURRENT / "candidate_mapping.csv")
    active = {int(r["kid"]): r for r in mapping if r["default_tuning"] == "True"}
    legacy = {int(r["kid"]): int(r["parent"]) for r in mapping if r["default_tuning"] != "True"}
    assert len(active) == 26 and len(legacy) == 13
    current_profile = read(CURRENT / "profile.csv")
    previous_profile = read(PREVIOUS / "profile.csv")
    selected_profiles = [(r, CURRENT.name, "本轮实测") for r in current_profile]
    selected_profiles += [(r, PREVIOUS.name, "上轮保留：M>2048") for r in previous_profile if int(r["M"]) > 2048]
    grouped = defaultdict(list)
    raw_grouped = defaultdict(list)
    provenance = {}
    measured = Counter()
    for row, source, label in selected_profiles:
        shape = key(row)
        assert shape in baseline
        assert (int(row["M"]) <= 2048) == (source == CURRENT.name)
        assert shape not in provenance or provenance[shape] == (source, label)
        provenance[shape] = source, label
        raw_grouped[shape].append(row)
        if row["libtype"] == "opus":
            kid = int(row["kernelId"])
            assert kid in active and row["kernelName"] == active[kid]["name"]
            assert valid(row)
            measured[kid] += 1
        if valid(row):
            grouped[shape].append(row)
    assert set(grouped) == set(baseline)

    candidate_counts = {kid: Counter() for kid in active}
    shape_rows = []
    ties = []
    for shape in sorted(baseline):
        rows = grouped[shape]
        opus_rows = [r for r in rows if r["libtype"] == "opus"]
        opus = min(opus_rows, key=lambda r: float(r["us"]))
        external = min((r for r in rows if r["libtype"] in ("ck", "cktile", "asm")),
                       key=lambda r: float(r["us"]))
        asm = min((r for r in rows if r["libtype"] == "asm"), key=lambda r: float(r["us"]))
        original = baseline[shape]
        us = float(opus["us"])
        original_us, external_us, asm_us = (float(r["us"]) for r in (original, external, asm))
        kid = int(opus["kernelId"])
        if sum(float(r["us"]) == us for r in opus_rows) > 1:
            ties.append(shape)
        count = candidate_counts[kid]
        count["opus_best"] += 1
        count["all_backend_best"] += us < external_us
        count["faster_than_original_csv"] += us < original_us
        count["faster_than_asm"] += us < asm_us
        count["current_sweep_opus_best"] += shape[2] <= 2048
        count["previous_sweep_opus_best"] += shape[2] > 2048
        range_name = "m_le512" if shape[2] <= 512 else "middle_m" if shape[2] < 1024 else "m_ge1024"
        count[f"{range_name}_opus_best"] += 1
        source, label = provenance[shape]
        row = dict(zip(SHAPE, shape), source_run=source, source_label=label,
                   opus_kernelId=kid, opus_us=us,
                   original_baseline_libtype=original["libtype"],
                   original_baseline_kernelId=int(original["kernelId"]),
                   original_baseline_splitK=int(original["splitK"]),
                   original_baseline_us=original_us,
                   vs_original_csv_saved_us=original_us-us,
                   vs_original_csv_latency_reduction_pct=100*(1-us/original_us),
                   vs_original_csv_speedup=original_us/us,
                   same_run_baseline_libtype=external["libtype"],
                   same_run_baseline_kernelId=int(external["kernelId"]),
                   same_run_baseline_splitK=int(external["splitK"]),
                   same_run_baseline_us=external_us,
                   vs_same_run_baseline_saved_us=external_us-us,
                   vs_same_run_baseline_latency_reduction_pct=100*(1-us/external_us),
                   vs_same_run_baseline_speedup=external_us/us,
                   same_run_asm_us=asm_us,
                   vs_same_run_asm_latency_reduction_pct=100*(1-us/asm_us),
                   opus_kernelName=opus["kernelName"],
                   original_baseline_kernelName=original["kernelName"],
                   same_run_baseline_kernelName=external["kernelName"])
        status, matched, same_id = original_config_match(original, raw_grouped[shape])
        matched_us = float(matched["us"]) if status == "matched" else None
        row.update(
            original_config_match_status=status,
            same_run_original_config_us=matched_us if matched_us is not None else "",
            original_config_timing_change_pct=100 * (matched_us / original_us - 1)
                if matched_us is not None else "",
            vs_same_run_original_config_speedup=matched_us / us
                if matched_us is not None else "",
            same_id_current_kernelNames="; ".join(sorted({r["kernelName"] for r in same_id})),
        )
        shape_rows.append(row)
    assert not ties, ("OPUS winner ties require an explicit counting rule", ties)
    candidate_rows = []
    for kid in sorted(active):
        count = candidate_counts[kid]
        candidate_rows.append(dict(
            kernelId=kid, measured_config_rows=measured[kid],
            **{k:count[k] for k in ("opus_best", "all_backend_best", "faster_than_original_csv",
                                   "faster_than_asm", "current_sweep_opus_best", "previous_sweep_opus_best",
                                   "m_le512_opus_best", "middle_m_opus_best", "m_ge1024_opus_best")},
            kernelName=active[kid]["name"]))
    assert sum(r["opus_best"] for r in candidate_rows) == 745
    assert sum(r["current_sweep_opus_best"] for r in candidate_rows) == 612
    assert sum(r["previous_sweep_opus_best"] for r in candidate_rows) == 133
    summary = dict(default_candidates=26, compatibility_ids=legacy, shapes=745,
                   timing_sources=dict(Counter(r["source_label"] for r in shape_rows)),
                   original_baseline_backends=dict(Counter(r["libtype"] for r in baseline_rows)),
                   original_csv=metrics([r["opus_us"]/r["original_baseline_us"] for r in shape_rows]),
                   same_run_ck_cktile_asm=metrics([r["opus_us"]/r["same_run_baseline_us"] for r in shape_rows]),
                   same_run_asm=metrics([r["opus_us"]/r["same_run_asm_us"] for r in shape_rows]),
                   notes=["612 current-sweep shapes plus 133 unchanged shapes from the prior sweep; not one new 745-shape sweep.",
                          "Original-CSV comparison uses historical saved latency, including 25 Triton records; it is cross-run.",
                          "Each same-run comparison pairs OPUS and CK/CKTile/ASM from the same source profile.",
                          "Latency reduction = 100*(baseline_us-opus_us)/baseline_us; positive means OPUS is faster."])
    for label, subset in (("current_sweep", [r for r in shape_rows if r["M"]<=2048]),
                          ("previous_sweep_unchanged", [r for r in shape_rows if r["M"]>2048])):
        summary[label] = {"original_csv": metrics([r["opus_us"]/r["original_baseline_us"] for r in subset]),
                          "same_run_ck_cktile_asm": metrics([r["opus_us"]/r["same_run_baseline_us"] for r in subset])}

    original_external = [r for r in shape_rows if r["original_baseline_libtype"] in EXTERNAL]
    original_triton = [r for r in shape_rows if r["original_baseline_libtype"] == "triton"]
    matched = [r for r in shape_rows if r["original_config_match_status"] == "matched"]
    assert len(original_external) == 720 and len(original_triton) == 25
    summary["original_csv_ck_cktile_asm_only"] = metrics(
        [r["opus_us"] / r["original_baseline_us"] for r in original_external])
    summary["retuned_best_vs_original_external_csv"] = timing_changes(
        [r["same_run_baseline_us"] / r["original_baseline_us"] for r in original_external])
    summary["original_config_match_status"] = dict(Counter(r["original_config_match_status"] for r in shape_rows))
    summary["matched_original_config_timing"] = timing_changes(
        [r["same_run_original_config_us"] / r["original_baseline_us"] for r in matched])
    summary["matched_original_config_by_backend"] = {
        backend: timing_changes([r["same_run_original_config_us"] / r["original_baseline_us"]
                                 for r in matched if r["original_baseline_libtype"] == backend])
        for backend in EXTERNAL}
    summary["matched_original_config_by_source"] = {
        source: timing_changes([r["same_run_original_config_us"] / r["original_baseline_us"]
                                for r in matched if r["source_run"] == source])
        for source in sorted({r["source_run"] for r in matched})}
    summary["notes"] += [
        "Original-config timing matches require backend, kernelId, splitK and kernelName to agree.",
        "The 5% band describes observed timing differences; it is not a statistical equivalence test.",
        "Historical compiler, binaries, clock/load state and full measurement settings are not recorded in the baseline CSV; causes of cross-run drift are unconfirmed.",
    ]
    write("candidate_wins.csv", candidate_rows)
    write("shape_comparison_745.csv", shape_rows)
    original_compact = compact_rows(original_external, "original_baseline_us", "Baseline CK/CKTile/ASM (ms)")
    same_run_compact = compact_rows(shape_rows, "same_run_baseline_us", "CK/CKTile/ASM best (ms)")
    write("original_baseline_720.csv", original_compact)
    write("same_run_comparison_745.csv", same_run_compact)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")

    lines = ["**gfx950 OPUS 最新候选与性能对比（2026-09-30）**", "",
             "当前默认参与 tune 的候选为 26 个，另保留 13 个历史兼容 ID。下表按合并后的父候选 ID 计数。", "",
             "Logical shape 按 M × N × K 排列，表示 A[M,K] 与 B[N,K] 的 GEMM。耗时统一为 ms，Speedup = Baseline / OPUS；大于 1 表示 OPUS 更快。倍率用未舍入的耗时计算，表中显示两位小数。", "",
             "数据来自已完成的正常 tuner：612 个 M≤2048 shape 使用 opus_consolidate_20260930 本轮实测；133 个 M>2048 shape 使用 opus_full745_register_20260930 上轮记录，其实现及生成 launcher 保持不变。每个 shape 的 OPUS 和同轮外部基线取自同一个 profile。", "",
             "原基线为 [dsv4 tuned CSV](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv) 中 gfx950/256-CU 的记录：ASM 133、CK 409、CKTile 178、Triton 25。主表严格使用前 720 条 CK/CKTile/ASM 的原 CSV 历史耗时，25 条 Triton 单列。该比较属于跨轮记录比较。", "",
             "[同轮重新 tune 的 745-shape 对比](SAME_RUN.md) 使用每个 shape 同轮所有有效 CK/CKTile/ASM 配置中的最快者，包含合法 split-K。OPUS 使用原生 E8M0 scale；外部后端沿用原 tuner 的 FP32 scale 数据、参考值与正确性检查。", "",
             "| 对照 | Shape 数 | OPUS 更快 | OPUS 更慢 | 几何平均加速倍率 |",
             "|---|---:|---:|---:|---:|"]
    for label, field in (("原 CSV 的 CK/CKTile/ASM 历史基线", "original_csv_ck_cktile_asm_only"),
                         ("同轮 CK/CKTile/ASM 最快者", "same_run_ck_cktile_asm"),
                         ("原 CSV 全部历史基线（含 Triton）", "original_csv")):
        s = summary[field]
        lines.append(f"| {label} | {s['shapes']} | {s['faster']} | {s['slower']} | {s['geomean_speedup']:.4f}x |")
    lines += ["", "**本轮 CK/CKTile/ASM 与原 CSV 是否一致**", "",
              "不能把它们视为没有差距。先固定原配置：要求 gfx、CU 数、M/N/K、后端、kernelId、splitK 和 kernelName 全部匹配，再比较本轮记录与原 CSV 的耗时。", "",
              "| 后端 | 原 CSV 条数 | 配置完全匹配条数 | 耗时几何平均变化 | 中位数变化 | 变化在 ±5% 内 |",
              "|---|---:|---:|---:|---:|---:|"]
    for backend in EXTERNAL:
        s = summary["matched_original_config_by_backend"][backend]
        lines.append(f"| {backend.upper() if backend != 'cktile' else 'CKTile'} | "
                     f"{summary['original_baseline_backends'][backend]} | {s['shapes']} | "
                     f"{s['geomean_change_pct']:+.2f}% | {s['median_change_pct']:+.2f}% | {s['within_5pct']} |")
    s = summary["matched_original_config_timing"]
    lines += [f"| **合计** | **720** | **{s['shapes']}** | **{s['geomean_change_pct']:+.2f}%** | **{s['median_change_pct']:+.2f}%** | **{s['within_5pct']}** |", "",
              "这里的耗时变化 = (本轮耗时 / 原 CSV 耗时 − 1) × 100%；正数表示本轮更慢。±5% 只是描述范围，单次记录不足以证明统计等价。", "",
              "720 条外部后端历史配置中，45 条缺少相同后端/ID/split-K 的本轮记录（ASM 41、CK 4），另有 1 条 ASM 的同 ID 名称不同。后者是 512×7168×1024、ASM ID 5、splitK=1：原 CSV 为 128x128，本轮为 48x128，因此不计入同配置统计。25 条历史 Triton 未参与当前外部后端枚举。明细 CSV 对这些情况保留明确状态和空值。", "",
              "再看重新互选：在上述 720 个 shape 上，本轮最快 CK/CKTile/ASM 对原 CSV 的耗时几何平均变化为 "
              f"{summary['retuned_best_vs_original_external_csv']['geomean_change_pct']:+.2f}%。这个数同时包含候选选择变化和跨轮计时变化，与固定原配置的统计含义不同。", "",
              "原 CSV 文件最后更新于提交 ded4e221（2026-09-17）；这些新计时来自 2026-09-30。CSV 未记录旧编译器、二进制、GPU 时钟/负载和完整测量参数，因此尚不能判定具体差异由哪一项造成。kernelName 相同也不证明二进制相同。本次 OPUS 工作没有修改 CK/CKTile/ASM 的 kernel 实现；两个新 profile 之间复用了哈希一致的外部二进制，但这不能外推为与历史 CSV 环境一致。", ""]
    lines += ["", "**26 个默认候选的胜出次数**", "",
              "OPUS 内最优：该候选是此 shape 最快的 OPUS。全后端最优：它同时快过该 shape 同轮所有有效 CK/CKTile/ASM 配置。每个 shape 只计入一个父候选；本次无并列最优。", "",
              "| 候选 ID | OPUS 内最优 | 全后端最优 | 其中快过原 CSV | 其中来自本轮 | 其中来自上轮保留 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in candidate_rows:
        lines.append(f"| {r['kernelId']} | {r['opus_best']} | {r['all_backend_best']} | {r['faster_than_original_csv']} | {r['current_sweep_opus_best']} | {r['previous_sweep_opus_best']} |")
    lines += ["| **合计** | **745** | **725** | **688** | **612** | **133** |", "",
              "历史兼容 ID：9048、9050、9056、9064–9069、9070–9073；它们不参与当前默认候选互选，未计入上表。", "",
              "[候选统计 CSV](candidate_wins.csv) · [745 行完整数据及原配置耗时核对](shape_comparison_745.csv) · [输入记录与校验](manifest.json)", "",
              "已完成的实现验证为 306 项测试通过、2 项 Graph 测试未运行；612-shape 正常调优中 OPUS 11,500 条、ASM 23,400 条全部有效。详见 [核验结果](../opus_consolidate_20260930/verification.json)、[测试命令](../opus_consolidate_20260930/validation.json) 和 [运行参数/源码哈希](../opus_consolidate_20260930/run.json)。此次整理仅做 CPU 数据分析，没有重新执行 GPU 测试或调优。", "",
              "重建本报告：`python3 reports/opus_current745_tables_20260930/summarize.py`。输入为仓库中的两个 profile、候选映射、原 CSV 与校验记录。", "",
              "**OPUS 对原 CSV 的 CK/CKTile/ASM 基线（720 个 shape）**", "",
              "[下载四列 CSV](original_baseline_720.csv)。Baseline 是原 CSV 保存的历史耗时；OPUS 是最新可用调优记录中的最快默认候选。每行的具体后端、ID、split-K 和来源见完整数据 CSV。", ""]
    lines += markdown_table(original_compact)
    lines += ["", "**原 CSV 的 25 条 Triton 历史记录（单列）**", "",
              "这些行补齐原 CSV 的全部 745 个 shape，但不计入上面的 CK/CKTile/ASM 历史基线表。对应 shape 的同轮 CK/CKTile/ASM 对比均已列入 SAME_RUN.md。", ""]
    lines += markdown_table(compact_rows(original_triton, "original_baseline_us", "Historical Triton (ms)"))
    (HERE / "REPORT.md").write_text("\n".join(lines)+"\n")

    same = summary["same_run_ck_cktile_asm"]
    lines = ["**OPUS 对同轮重新 tune 的 CK/CKTile/ASM（745 个 shape）**", "",
             "每个 shape 分别选择最快默认 OPUS 候选和最快有效 CK/CKTile/ASM 候选。两者取自同一个 profile，包含外部后端的合法 split-K；OPUS 的固定分区候选计时包含 producer 和 reduction。", "",
             "Logical shape 为 M × N × K；耗时为 ms。Speedup = CK/CKTile/ASM best / OPUS。倍率按未舍入的计时计算；小于 1 的结果也保留，接近 1 的结果可能显示为 1.00x。", "",
             f"OPUS 更快 {same['faster']} 个、更慢 {same['slower']} 个，几何平均加速 {same['geomean_speedup']:.4f}x。", "",
             "本表汇总最新可用记录：612 个 M≤2048 shape 来自合并后的正常调优，133 个 M>2048 shape 来自上轮且对应实现未变；不是一次新执行的 745-shape sweep。", "",
             "原 tuner 参数：`--libtype all --splitK --shape_grouped --mp 8 --warmup 5 --iters 51 --errRatio 0.05 --all`。OPUS 使用原生 E8M0 scale 并要求零超差；CK/CKTile/ASM 使用原 FP32 scale 数据及验收条件。", "",
             "[原 CSV 基线表与同配置耗时核对](REPORT.md) · [四列 CSV](same_run_comparison_745.csv) · [完整配置和来源](shape_comparison_745.csv)", ""]
    lines += markdown_table(same_run_compact)
    (HERE / "SAME_RUN.md").write_text("\n".join(lines)+"\n")

    # Validate the exported tables, including complete, unique shape coverage.
    exported = read(HERE / "shape_comparison_745.csv")
    assert len(exported) == 745 and {key(r) for r in exported} == set(baseline)
    assert len(read(HERE / "candidate_wins.csv")) == 26
    for filename, expected in (("original_baseline_720.csv", original_external),
                               ("same_run_comparison_745.csv", shape_rows)):
        table = read(HERE / filename)
        assert len(table) == len(expected)
        assert {r["Logical shape"] for r in table} == {logical_shape(r) for r in expected}
    assert len(matched) == 674
    assert summary["original_config_match_status"] == {
        "matched": 674, "configuration_not_measured": 45,
        "kernel_name_mismatch": 1, "backend_not_in_sweep": 25}
    inputs = [BASELINE, CURRENT/"profile.csv", PREVIOUS/"profile.csv", CURRENT/"candidate_mapping.csv",
              PREVIOUS/"run.json", CURRENT/"run.json", CURRENT/"verification.json", CURRENT/"validation.json"]
    manifest = dict(status="passed", generated_utc=datetime.now(timezone.utc).isoformat(),
                    operation="CPU-only summarization of existing data; no new GPU tests or tuning",
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    checks=dict(exact_745_shapes=True, no_duplicate_shapes=True, exact_26_candidates=True,
                                winners_sum_to_745=True, original_baseline_hash_matches=True,
                                same_run_pairing_per_shape=True, no_winner_ties=True,
                                exact_720_original_external_rows=True, triton_25_kept_separate=True,
                                exact_674_original_configuration_matches=True,
                                missing_or_renamed_configurations_not_imputed=True,
                                milliseconds_from_microseconds=True),
                    output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (HERE/"candidate_wins.csv", HERE/"shape_comparison_745.csv",
                                    HERE/"original_baseline_720.csv", HERE/"same_run_comparison_745.csv",
                                    HERE/"summary.json", HERE/"REPORT.md", HERE/"SAME_RUN.md")})
    (HERE / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
