#!/usr/bin/env python3
"""Freeze the verified public results and document completed experiments."""
import collections
import csv
import hashlib
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from prepare_opus_repeat import csv_rows, key, shape_row, pure_common, registry_metadata

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
DOC = LOCAL.parent
OUT = DOC / "optimization_summary_20261030"
LATEST = LOCAL / "stage17_pin_invariants54_20261030"
NAMES = {
    5: "候选扩展与复测", 6: "扩候选、kernel参数与调度", 7: "B-VGPR及pin主循环",
    8: "长K候选", 9: "窄N紧凑B流水", 10: "提前B预取", 11: "主循环展开",
    12: "scale缓存缩小", 13: "SFA局部临时值", 14: "按scale边界分段主循环",
    15: "pin_agpr展开2/4", 16: "矩阵请求提前、尾部临时值缩短", 17: "偏移不变量与标量传播",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    assert rows
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    receipt = json.loads((LATEST / "completion_receipt.json").read_text())
    assert (LATEST / "controller_exit_code").read_text().strip() == "0"
    assert receipt["pending"] == receipt["incorrect_experimental_measurements"] == 0
    assert receipt["public_validation"]["public_artifact_membership_verified"]
    rows = csv_rows(LATEST / "result/public_best_opus421.csv")
    losses = csv_rows(LATEST / "result/public_remaining_losses.csv")
    assert len(rows) == len({key(r) for r in rows}) == 421 and len(losses) == 84
    assert sum(r["status"] == "wins_csv" for r in rows) == 337
    baselines = {key(r): float(r["us"]) for r in csv_rows(DOC / "handoff_data/csv_baseline.csv")}
    metadata, _ = registry_metadata()
    common = pure_common()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    assert len(registry) == 48261
    for row in rows:
        shape, kid = key(row), int(row["kernelId"])
        assert shape[0:2] == ("gfx950", 256) and shape[5] == "128x128"
        assert metadata[kid]["kernelName"] == row["kernelName"]
        assert baselines[shape] == float(row["csv_baseline_us"])
        assert common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape[2:5])
        assert float(row["errRatio"]) < .05
    for name, digest in receipt["public_validation"]["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    module = LATEST / ".cache/public_jit/module_deepgemm_opus.so"
    assert sha(module) == receipt["public_validation"]["public_module_sha256"]
    compiled = json.loads((LATEST / ".cache/public_jit/build/compiled_kids_opus.json").read_text())
    assert all(k.startswith("a8w8_blockscale_bpreshuffle/") for k in compiled)
    assert {f"a8w8_blockscale_bpreshuffle/gfx950/256/{r['kernelId']}" for r in rows} <= set(compiled)

    OUT.mkdir(exist_ok=True)
    shutil.copy2(LATEST / "result/public_best_opus421.csv", OUT / "best_opus421.csv")
    shutil.copy2(LATEST / "result/public_remaining_losses.csv", OUT / "remaining_losses84.csv")
    shutil.copy2(ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json", OUT / "co_kernels_gfx950_snapshot.json")
    shutil.copy2(LATEST / "completion_receipt.json", OUT / "latest_completion_receipt.json")
    (OUT / "run_env.sh").write_text("#!/usr/bin/env bash\nsource " + str(LATEST / "public_env.sh") + "\n")

    pipeline_rows = []
    for pipeline in sorted({r["winner_pipeline"] for r in losses}, key=lambda p: -sum(r["winner_pipeline"] == p for r in losses)):
        selected = [r for r in losses if r["winner_pipeline"] == pipeline]
        values = [float(r["winner_slower_than_csv_pct"]) for r in selected]
        pipeline_rows.append({"pipeline": pipeline, "shapes": len(values),
            "median_slower_pct": statistics.median(values), "max_slower_pct": max(values),
            "below_1pct": sum(v < 1 for v in values), "below_3pct": sum(v < 3 for v in values),
            "below_5pct": sum(v < 5 for v in values), "at_least_10pct": sum(v >= 10 for v in values)})
        write_csv(OUT / f"remaining_{pipeline}.csv", selected)
    write_csv(OUT / "pipeline_summary.csv", pipeline_rows)
    loss_values = [float(r["winner_slower_than_csv_pct"]) for r in losses]
    buckets = [{"range_pct": label, "shapes": sum(low <= x < high for x in loss_values)}
        for label, low, high in (("0-1", 0, 1), ("1-3", 1, 3), ("3-5", 3, 5), ("5-10", 5, 10), (">=10", 10, float("inf")))]
    assert sum(r["shapes"] for r in buckets) == 84
    write_csv(OUT / "loss_margins.csv", buckets)
    groups = collections.defaultdict(list)
    for row in losses:
        groups[row["winner_pipeline"], int(row["N"]), int(row["K"])].append(float(row["winner_slower_than_csv_pct"]))
    write_csv(OUT / "remaining_by_pipeline_N_K.csv", [{"pipeline": p, "N": n, "K": k,
        "shapes": len(v), "median_slower_pct": statistics.median(v), "max_slower_pct": max(v)}
        for (p, n, k), v in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))])

    step4 = LOCAL / "step4_final_recovery_gpu56_v2_20261030/full421_final_report/best_opus.csv"
    original = {key(r): r for r in csv_rows(step4)}
    provenance = {shape: {"last_retested_campaign": "step4_final_recovery_gpu56_v2_20261030",
        "evidence_csv": str(step4.relative_to(DOC)), "rounds": 6,
        "kernelId": r["kernelId"], "us": float(r["us"])} for shape, r in original.items()}
    history, inventory, improvements = [], [], []
    for number in range(5, 18):
        directory = next(d for d in LOCAL.glob(f"stage{number}_*") if (d / "completion_receipt.json").exists())
        metrics = json.loads((directory / "result/public_metrics.json").read_text())
        campaign_receipt = json.loads((directory / "completion_receipt.json").read_text())
        integration = json.loads((directory / "integration/receipt.json").read_text())
        selected_ids = set(integration["selected_ids"])
        comparison_file = next((directory / "result").glob("public_final*_comparisons.csv"))
        comparisons = csv_rows(comparison_file)
        history.append({"stage": number, "direction": NAMES[number], "wins": metrics["combined421_wins"],
            "losses": metrics["combined421_losses"], "fresh_retested_shapes": len(comparisons),
            "faster_candidate_driven_new_wins": metrics["new_win_with_control_still_slower"],
            "old_control_retest_new_wins": metrics["wins_where_old_control_also_wins_now"],
            "new_co_count": len(selected_ids), "new_co_ids": ";".join(map(str, sorted(selected_ids))),
            "measurements": campaign_receipt["actual_measurements"], "evidence": str(directory.relative_to(DOC))})
        for r in comparisons:
            provenance[key(r)] = {"last_retested_campaign": directory.name,
                "evidence_csv": str(comparison_file.relative_to(DOC)), "rounds": 6,
                "kernelId": r["winner_kid"], "us": float(r["winner_us"])}
            if int(r["winner_kid"]) in selected_ids:
                improvements.append({"stage": number, **shape_row(key(r)), "kid": r["winner_kid"],
                    "pipeline": metadata[int(r["winner_kid"])]["geometry"][0],
                    "time_reduction_vs_control_pct": r["winner_gain_vs_control_pct"],
                    "paired_median_speedup_pct": r["paired_median_speedup_pct"],
                    "faster_rounds": r["paired_faster_rounds"], "wins_csv": r["wins_csv"],
                    "matched_paired_median_speedup_pct": r.get("matched_paired_median_speedup_pct", ""),
                    "evidence_csv": str(comparison_file.relative_to(DOC))})
        for kid in sorted(selected_ids):
            entry = registry[kid].bpreshuffle_co
            image = ROOT / "csrc/opus_gemm/gen_co/gfx950" / entry["image"]
            assert image.is_file()
            inventory.append({"kid": kid, "integrated_stage": number, "pipeline": entry["pipeline"],
                "kernelName": registry[kid].name, "current_best_shapes": sum(int(r["kernelId"]) == kid for r in rows),
                "supported_shapes": json.dumps(entry.get("supported_shapes"), separators=(",", ":")),
                "traits_header": entry.get("traits_header", ""), "pipeline_header": entry["pipeline_header"],
                "body_func": entry["body_func"], "body_args": json.dumps(entry["body_args"], separators=(",", ":")),
                "co_image": str(image.relative_to(ROOT)), "co_sha256": sha(image)})
    write_csv(OUT / "round_history.csv", history)
    write_csv(OUT / "integrated_co_inventory.csv", inventory)
    write_csv(OUT / "integrated_kernel_gains.csv", improvements)
    evidence_rows = []
    for r in rows:
        p = provenance[key(r)]
        assert p["kernelId"] == r["kernelId"] and p["us"] == float(r["us"]), key(r)
        evidence_rows.append({**shape_row(key(r)), "kernelId": r["kernelId"],
            "pipeline": metadata[int(r["kernelId"])]["geometry"][0], "status": r["status"],
            "us": r["us"], "last_retested_campaign": p["last_retested_campaign"],
            "rounds": 6, "evidence_csv": p["evidence_csv"]})
    write_csv(OUT / "shape_evidence421.csv", evidence_rows)
    assert sum(r["faster_candidate_driven_new_wins"] for r in history) == 53
    assert sum(r["old_control_retest_new_wins"] for r in history) == 23
    assert len(inventory) == len({r["kid"] for r in inventory}) == 27
    final_plan = json.loads((LATEST / "final/confirm/repeat_plan.json").read_text())
    tested = {key(s["shape"]) for s in final_plan["shapes"]}
    previous = {key(r): r for r in csv_rows(LOCAL / "stage16_pin_compute54_20261030/result/public_best_opus421.csv")}
    assert len(tested) == 54 and all(r == previous[key(r)] for r in rows if key(r) not in tested)

    summary = {"status": "optimization_stopped_results_organized", "organized_utc": datetime.now(timezone.utc).isoformat(),
        "latest_campaign": LATEST.name, "shapes": 421, "wins": 337, "losses": 84,
        "loss_median_pct": statistics.median(loss_values), "loss_max_pct": max(loss_values),
        "loss_buckets": buckets, "pipeline_summary": pipeline_rows, "public_registry_count": 48261,
        "jit_kid_base": 780, "integrated_new_co_count_stages5_17": 27,
        "starting_step4_wins": 261, "starting_step4_losses": 160,
        "faster_candidate_driven_new_wins": 53, "old_control_retest_new_wins": 23,
        "latest_fresh_retested_shapes": 54, "historical_rows_retained_latest": 367,
        "latest_measurements": 4634, "latest_measurements_incorrect": 0,
        "last_two_round_measurements": 5060 + 4634, "last_two_round_new_co_count": 0,
        "max_tested_A_input_bytes": max(int(r["M"]) * int(r["K"]) for r in rows),
        "max_tested_C_output_bytes": max(2 * int(r["M"]) * int(r["N"]) for r in rows),
        "A_2GiB_guard_unchanged": True, "only_tune": "Opus MXFP8 NT", "baseline": "fixed FlyDSL CSV, six-field identity",
        "compiled_family_only": "a8w8_blockscale_bpreshuffle", "public_compiled_routes": len(compiled),
        "physical_gpus": [0, 3, 5, 6], "logical_to_physical": [0, 3, 6, 5],
        "public_module_path": str(module), "public_module_sha256": sha(module),
        "source_hashes_verified": receipt["public_validation"]["source_sha256"],
        "cpu_only_organization": True, "no_new_gpu_measurements_during_organization": True}
    dump(OUT / "summary.json", summary)
    tables = "\n".join(f"| {r['pipeline']} | {r['shapes']} | {r['median_slower_pct']:.2f}% | {r['max_slower_pct']:.2f}% |" for r in pipeline_rows)
    rounds = "\n".join(f"| {r['stage']} | {r['direction']} | {r['wins']}/{r['losses']} | {r['new_co_count']} | {r['faster_candidate_driven_new_wins']}/{r['old_control_retest_new_wins']} |" for r in history)
    margin_table = "\n".join(f"| {r['range_pct']}% | {r['shapes']} |" for r in buckets)
    readme = f'''# Opus MXFP8 NT 优化结果整理

截至第17轮：**421项中337领先、84落后**。按用户要求停止继续优化，保留当前可用最优配置与已完成的实验。第16、17轮均未产生满足稳定标准的新CO；两轮共9694条测量全部正确，controller均退出0。

从第四步261/160推进到337/84，新增76项领先：53项在当轮原最优仍落后时由更快候选越线，23项当轮原最优自身复测也越线。53项包含既有候选重新选择，不能全部归因于新kernel源码。

当前421行来自各轮已确认数据合并。最后一轮只对54项pin_agpr落后做新六轮，其余367行逐字段保留；全部421项的最后复测来源在[shape_evidence421.csv](shape_evidence421.csv)。FlyDSL使用固定CSV，按gfx/CU/M/N/K/scale-block六字段匹配。

## 剩余落后

落后定义：`(Opus_us / 固定FlyDSL_us - 1) * 100%`。84项中位落后{summary['loss_median_pct']:.2f}%，最大{summary['loss_max_pct']:.2f}%。以下每个shape只按当前最优pipeline计一次。

| Pipeline | 项数 | 中位落后 | 最大落后 |
|---|---:|---:|---:|
{tables}

| 落后区间 | 项数 |
|---|---:|
{margin_table}

完整降序明细：[remaining_losses84.csv](remaining_losses84.csv)。按pipeline拆分的CSV与[remaining_by_pipeline_N_K.csv](remaining_by_pipeline_N_K.csv)同时保存。最近两轮定向处理最大类别pin_agpr，仍有54项；这表示当前尝试未见稳定收益，不代表已证明所有方向都无法提升。

## 已保留的实现

- [best_opus421.csv](best_opus421.csv)：当前可用的421项最优配置；领先与落后均保留。
- [integrated_co_inventory.csv](integrated_co_inventory.csv)：第5至17轮整合的27个CO，含当前选中shape数、traits、pipeline、shape范围、镜像路径及SHA256。历史CO继续保留。
- [integrated_kernel_gains.csv](integrated_kernel_gains.csv)：各轮整合新CO的逐shape收益与复测证据；同一shape可在多轮出现。
- [co_kernels_gfx950_snapshot.json](co_kernels_gfx950_snapshot.json)：本次停止时的公共CO清单快照。公共registry为48261，JIT编号起点固定780，旧配置身份保留。

近期有效改动包括：紧凑B预取流水；pin主循环展开8；SFA每次补载使用局部临时值；scale refill移至按缓存边界分段的外层；个别shape使用展开2。第13轮SFA改动带来两个shape收益，第14轮分段循环带来六个，第15轮展开2带来两个；对应源码与CO见清单，各轮之间存在shape重叠。

近期未采用的实验：第12轮缩小scale缓存无新最优CO，panel16曾出现大量spill；第16轮提前矩阵请求及缩短尾部临时值未达到稳定收益；第17轮暴露偏移不变量减少了部分寄存器，但伴随spill，同样无稳定新最优。完整实验、初筛明显慢候选、未通过编译的诊断都保存在原campaign目录；未作为当前最优整合。

## 轮次与归因

新增领先的两类分别为“更快候选越线 / 原control复测越线”。只列各轮新增数。

| 轮次 | 方向 | 领先/落后 | 新CO数 | 新领先归因 |
|---:|---|---:|---:|---:|
{rounds}

原始目录、复测数量、测量数及新CO编号：[round_history.csv](round_history.csv)。最新完成凭据：[latest_completion_receipt.json](latest_completion_receipt.json)。

## 复现与范围

在本workspace中使用已验证的公共模块和custom pin-AGPR LLVM wrapper：

```bash
source docs/gfx950_mxscale_20261029/optimization_summary_20261030/run_env.sh
```

环境为gfx950/256 CU、Opus NT；逻辑卡0/1/2/3经PCI核对对应物理0/3/6/5。该环境指向第17轮公开缓存，并设置`OPUS_COMPILE_FAMILY_ONLY=a8w8_blockscale_bpreshuffle`。后续本任务构建已限制为MXFP8 family，实验和公开生成清单均没有a16w16路由；其他任务未设置该变量时保持原构建范围。

A输入侧2GiB guard保持。输出使用已验证的tile寻址路径；421项中最大A输入为1GiB、最大C输出为8GiB，实际覆盖了超过4GiB的输出。此整理没有扩大A输入支持范围，也没有改动已有buffer描述符、硬件wait或线程同步。

[summary.json](summary.json)记录校验后的结果、来源哈希及公共模块哈希。整理过程仅作CPU读取、校验与文件汇总，没有新增GPU测量。当前优化进程均已退出。
'''
    (OUT / "README.md").write_text(readme)
    files = {str(p.relative_to(OUT)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in OUT.iterdir() if p.is_file() and p.name != "files.json"}
    dump(OUT / "files.json", files)
    print(json.dumps({"directory": str(OUT), "wins": 337, "losses": 84,
        "loss_median_pct": summary["loss_median_pct"], "pipeline_counts": {r["pipeline"]: r["shapes"] for r in pipeline_rows},
        "integrated_co_count": len(inventory), "summary_files": len(files), "public_hash_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
