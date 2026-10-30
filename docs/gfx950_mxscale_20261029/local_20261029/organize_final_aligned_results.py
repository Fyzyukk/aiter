#!/usr/bin/env python3
"""Freeze the final aligned winners, with shape identity and measurement evidence."""
import collections
import csv
import dataclasses
import hashlib
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from prepare_opus_repeat import csv_rows, key, shape_row, pure_common, registry_metadata
from prepare_co_experiment import load_common, registry

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
DOC = LOCAL.parent
STAGE = LOCAL / "stage18_remote_final84_20261030"
ALIGN = LOCAL / "remote_alignment_54bef351_20261030"
PREVIOUS = DOC / "optimization_summary_20261030"
OUT = DOC / "final_aligned_results_20261030"
REMOTE_COMMIT = "54bef35161139556602a39d0d2dcf367102193fc"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def yes(value):
    return str(value).lower() == "true"


def main():
    receipt = json.loads((STAGE / "completion_receipt.json").read_text())
    integration = json.loads((STAGE / "integration/receipt.json").read_text())
    alignment = json.loads((ALIGN / "set_alignment_summary.json").read_text())
    identity = json.loads((STAGE / "co_experiment/identity_and_points.json").read_text())
    assert (STAGE / "controller_exit_code").read_text().strip() == "0"
    assert (STAGE / "controller_resume_exit_code").read_text().strip() == "0"
    assert (STAGE / "controller_verified_exit_code").read_text().strip() == "0"
    assert receipt["pending"] == receipt["incorrect_experimental_measurements"] == 0
    assert receipt["public_validation"]["public_artifact_membership_verified"]
    final = {key(r): r for r in csv_rows(STAGE / "result/public_best_opus421.csv")}
    prior = {key(r): r for r in csv_rows(PREVIOUS / "best_opus421.csv")}
    aligned = {key(r): r for r in csv_rows(ALIGN / "all421_alignment.csv")}
    baseline_path = DOC / "handoff_data/csv_baseline.csv"
    baseline_rows = {key(r): r for r in csv_rows(baseline_path)}
    assert set(final) <= set(baseline_rows)
    baseline = {s: float(baseline_rows[s]["us"]) for s in final}
    assert len(final) == len(prior) == len(aligned) == 421
    assert set(final) == set(prior) == set(aligned) == set(baseline)
    prior_wins = {s for s, r in prior.items() if r["status"] == "wins_csv"}
    remote_wins = {s for s, r in aligned.items() if yes(r["remote_wins"])}
    final_wins = {s for s, r in final.items() if r["status"] == "wins_csv"}
    assert len(prior_wins) == 337 and len(remote_wins) == 236
    assert remote_wins <= prior_wins <= final_wins
    assert all(final[s] == prior[s] for s in prior_wins)
    assert len(final_wins) == receipt["combined421_wins"]
    assert sha(baseline_path) == alignment["remote_baseline_sha256"] == alignment["local_baseline_sha256"]

    md, _ = registry_metadata()
    common = pure_common()
    public = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    experiment = registry(load_common(STAGE / "co_experiment/.cache/experiment_repo", label="final_aligned_experiment"))
    new_ids = {p["new_kid"] for p in identity["points"]}
    old_ids = set(experiment) - new_ids
    assert len(old_ids) == identity["previous_ids_verified"] == 48261
    assert set(public) == old_ids | set(integration["selected_ids"])
    assert all(dataclasses.asdict(public[k]) == dataclasses.asdict(experiment[k]) for k in old_ids)
    for s, r in final.items():
        kid = int(r["kernelId"])
        assert s[0:2] == ("gfx950", 256) and s[5] == "128x128"
        assert r["kernelName"] == md[kid]["kernelName"]
        assert common.a8w8_mxscale_bpreshuffle_supports_shape(public[kid], *s[2:5])
        assert float(r["csv_baseline_us"]) == baseline[s]
        assert (float(r["us"]) < baseline[s]) == (s in final_wins)
        assert float(r["errRatio"]) <= .05
    for name, digest in receipt["public_validation"]["source_sha256"].items():
        assert sha(Path(name)) == digest
    module = STAGE / ".cache/public_jit/module_deepgemm_opus.so"
    assert sha(module) == receipt["public_validation"]["public_module_sha256"]
    compiled = json.loads((STAGE / ".cache/public_jit/build/compiled_kids_opus.json").read_text())
    assert all(k.startswith("a8w8_blockscale_bpreshuffle/") for k in compiled)
    assert {f"a8w8_blockscale_bpreshuffle/gfx950/256/{r['kernelId']}" for r in final.values()} <= set(compiled)

    comparisons = csv_rows(STAGE / "result/public_final84_comparisons.csv")
    tested = {key(r) for r in comparisons}
    assert tested == set(prior) - prior_wins and len(tested) == 84
    added = [dict(r) for r in comparisons if yes(r["wins_csv"])]
    assert {key(r) for r in added} == final_wins - prior_wins
    categories = collections.Counter()
    for r in added:
        if yes(r["control_wins_csv"]):
            cause = "old_control_also_crossed_on_retest"
        elif yes(r["new_co"]):
            cause = "new_kernel_crossed_while_control_still_slower"
        else:
            cause = "existing_candidate_crossed_while_control_still_slower"
        categories[cause] += 1
        r["new_win_attribution"] = cause
    losses = [r for r in comparisons if not yes(r["wins_csv"])]
    assert len(losses) == receipt["combined421_losses"]
    losses.sort(key=lambda r: float(r["winner_slower_than_csv_pct"]), reverse=True)

    OUT.mkdir(exist_ok=True)
    shutil.copy2(STAGE / "result/public_best_opus421.csv", OUT / "full421_tuned.csv")
    write(OUT / "leading_tuned.csv", [final[s] for s in sorted(final_wins)])
    write(OUT / "remaining_losses.csv", losses, list(comparisons[0]))
    write(OUT / "newly_leading_shapes.csv", added, list(comparisons[0]) + ["new_win_attribution"])
    shutil.copy2(STAGE / "result/public_final84_comparisons.csv", OUT / "last_round84_comparison.csv")
    shutil.copy2(STAGE / "completion_receipt.json", OUT / "last_round_completion_receipt.json")
    shutil.copy2(STAGE / "integration/receipt.json", OUT / "new_co_integration_receipt.json")
    shutil.copy2(ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json", OUT / "co_kernels_gfx950_snapshot.json")
    shutil.copytree(ALIGN / "remote_snapshot", OUT / "remote_snapshot", dirs_exist_ok=True)
    for name in ("all421_alignment.csv", "set_alignment_summary.json", "local_only_wins101.csv",
                 "both_losses84.csv", "remote_kernel_identity421.csv"):
        shutil.copy2(ALIGN / name, OUT / name)
    shutil.copy2(baseline_path, OUT / "fixed_baseline_source.csv")
    write(OUT / "fixed_gfx950_baseline.csv", [baseline_rows[s] for s in sorted(final)])
    (OUT / "run_env.sh").write_text("#!/usr/bin/env bash\nsource " + str(STAGE / "public_env.sh") + "\n")
    source_rows = []
    provenance = {key(r): r for r in csv_rows(PREVIOUS / "shape_evidence421.csv")}
    full_comparison = []
    for s in sorted(final):
        r, a = final[s], aligned[s]
        p = dict(provenance[s])
        if s in tested:
            p.update(last_retested_campaign=STAGE.name, evidence_csv=str((STAGE / "result/public_final84_comparisons.csv").relative_to(DOC)), rounds=6)
        p.update(kernelId=r["kernelId"], pipeline=md[int(r["kernelId"])]["geometry"][0], status=r["status"], us=r["us"])
        source_rows.append(p)
        full_comparison.append({**shape_row(s), "kernelId": r["kernelId"], "kernelName": r["kernelName"],
            "opus_us": float(r["us"]), "csv_baseline_us": baseline[s], "wins_csv": s in final_wins,
            "slower_than_csv_pct": (float(r["us"]) / baseline[s] - 1) * 100,
            "initial_local_wins": s in prior_wins, "remote_wins": s in remote_wins,
            "initial_category": a["category"], "last_retested_campaign": p["last_retested_campaign"],
            "measurement_scope": "last_round_fresh_six" if s in tested else "retained_verified_previous_result"})
    write(OUT / "shape_evidence421.csv", source_rows)
    write(OUT / "full421_comparison.csv", full_comparison)

    # Keep each isolated experiment's matched-body result, including trials which lost.
    raw = collections.defaultdict(dict)
    for file in (STAGE / "final/confirm/repeats").glob("gpu*.jsonl"):
        for line in file.read_text().splitlines():
            r = json.loads(line)
            assert r["correct"] and not r["failure"]
            raw[key(r["shape"]), r["kid"]][r["round"]] = r
    matches = {pair[1]: pair[0] for pair in identity["matched_controls"]}
    trials = []
    for point in identity["points"]:
        if point.get("comparison_only"):
            continue
        kid = point["new_kid"]
        for s in sorted(tested):
            if (s, kid) not in raw:
                continue
            base_id = matches[kid]
            a, b = raw[s, kid], raw[s, base_id]
            assert set(a) == set(b) == set(range(6))
            ratios = [b[i]["us"] / a[i]["us"] for i in range(6)]
            paired = (statistics.median(ratios) - 1) * 100
            faster = sum(x > 1 for x in ratios)
            trials.append({**shape_row(s), "new_kid": kid, "matched_kid": base_id,
                "pipeline": point["pipeline"], "axis": json.dumps(point["axis"], sort_keys=True),
                "new_us": statistics.median(x["us"] for x in a.values()),
                "matched_us": statistics.median(x["us"] for x in b.values()),
                "paired_median_speedup_pct": paired, "faster_rounds": faster,
                "matched_material_gain": paired >= .5 and faster >= 4,
                "selected_as_public_best": int(final[s]["kernelId"]) == kid})
    write(OUT / "last_round_matched_kernel_trials.csv", trials)
    for name in ("all22_co_resource_metadata.json", "schedule_invariants.json",
                 "interleave_B_first_isa_verification.json", "identity_and_points.json"):
        shutil.copy2(STAGE / "co_experiment" / name, OUT / name)
    for name in ("verification_reason.json", "measurement_verification_assembly.json"):
        shutil.copy2(STAGE / name, OUT / name)
    shutil.copy2(STAGE / "final/confirm/assembly_receipt.json", OUT / "verified_final_batch_assembly.json")
    new_co_rows = []
    manifest = json.loads((ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json").read_text())
    for entry in manifest["gfx950"]:
        if entry["kid"] not in integration["selected_ids"]:
            continue
        image = ROOT / "csrc/opus_gemm/gen_co/gfx950" / (public[entry["kid"]].name + ".co")
        new_co_rows.append({"kid": entry["kid"], "kernelName": public[entry["kid"]].name,
            "image": str(image.relative_to(ROOT)), "sha256": sha(image),
            "supported_shapes": json.dumps(entry["supported_shapes"]),
            "public_best_shapes": sum(int(r["kernelId"]) == entry["kid"] for r in final.values()),
            "public_best_wins": sum(int(final[s]["kernelId"]) == entry["kid"] for s in final_wins)})
    for name, digest in integration.get("headers", {}).items():
        assert sha(Path(name)) == digest
    for image in integration.get("images", []):
        assert sha(Path(image["destination"])) == image["sha256"]
    dump(OUT / "source_artifact_hashes.json", {
        "public_validation_sources": receipt["public_validation"]["source_sha256"],
        "new_headers": integration.get("headers", {}), "new_images": integration.get("images", []),
        "public_module": {"path": str(module), "sha256": sha(module)},
        "generator_and_tuner": {str(ROOT / rel): sha(ROOT / rel) for rel in
            ("csrc/opus_gemm/gen_instances.py", "csrc/opus_gemm/opus_gemm_tune.py")}})
    write(OUT / "new_integrated_co.csv", new_co_rows,
          ["kid", "kernelName", "image", "sha256", "supported_shapes", "public_best_shapes", "public_best_wins"])
    pipelines = []
    for pipeline in sorted({r["winner_pipeline"] for r in losses}):
        values = [float(r["winner_slower_than_csv_pct"]) for r in losses if r["winner_pipeline"] == pipeline]
        pipelines.append({"pipeline": pipeline, "shapes": len(values), "median_slower_pct": statistics.median(values), "max_slower_pct": max(values)})
    pipelines.sort(key=lambda r: -r["shapes"])
    write(OUT / "remaining_pipeline_summary.csv", pipelines)

    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "remote_commit": REMOTE_COMMIT,
        "remote_branch": "fyz/gfx950-search-identity-20261029", "alignment": alignment,
        "final_wins": len(final_wins), "final_losses": len(losses), "new_wins": len(added),
        "new_win_attribution": dict(categories), "last_round_metrics": receipt,
        "initial337_rows_preserved_exactly": True, "remote236_wins_all_in_final": True,
        "all421_shape_identity_verified": True, "old48261_kernel_identities_unchanged": True,
        "jit_kid_base": 780, "baseline_sha256": sha(baseline_path),
        "last_round_matched_trial_pairs": len(trials),
        "last_round_matched_material_gain_pairs": sum(r["matched_material_gain"] for r in trials),
        "remaining_pipelines": pipelines, "last_optimization_round": 18, "next_round": None,
        "measurement_scope": "84 initial shared losses freshly measured for six rounds; original 337 local wins retained unchanged",
        "leading_csv": "leading_tuned.csv", "full421_csv": "full421_tuned.csv"}
    dump(OUT / "summary.json", summary)
    lines = [f"最终合并结果：**{len(final_wins)} 项领先 / {len(losses)} 项落后**。全部领先项在 [leading_tuned.csv](leading_tuned.csv)，完整 421 项在 [full421_tuned.csv](full421_tuned.csv)。", "",
        "按完整六字段 gfx、cu_num、M、N、K、w_scale_block 对齐。固定基线的原字节 SHA-256 相同，421 个 shape 集合完全一致。远端 kernel 均按完整名称映射并核对 shape 支持。", "",
        "| 对齐类别 | 项数 |", "|---|---:|", "| 双方领先 | 236 |", "| 本地独有领先 | 101 |", "| 远端独有领先 | 0 |", "| 双方落后、进入最后优化 | 84 |", "",
        f"远端 `{REMOTE_COMMIT}` 的 236 个领先全部包含在本地 337 个领先中。本地有 101 个远端未领先的 shape。最终仍完整保留这 337 行，再加入最后一轮新增的 {len(added)} 个领先。", "",
        f"最后一轮只测原 84 个共同落后项：2 轮初筛、6 轮独立确认，每次确认计时 101 次；{receipt['actual_measurements']} 条性能记录全部正确、pending 0。原 337 个领先项沿用其已验证结果，逐字段不变。新 CO 已经过公共注册表、JIT 产物、正确性验证。", "",
        f"新增领先归因：新 kernel 且原 control 仍落后 {categories['new_kernel_crossed_while_control_still_slower']} 项；既有候选且原 control 仍落后 {categories['existing_candidate_crossed_while_control_still_slower']} 项；原 control 本次复测也越线 {categories['old_control_also_crossed_on_retest']} 项。成对比较和逐项证据见 [last_round84_comparison.csv](last_round84_comparison.csv) 与 [newly_leading_shapes.csv](newly_leading_shapes.csv)。", "",
        f"整合 {len(integration['selected_ids'])} 个新 CO：`{integration['selected_ids']}`。稳定整合要求最终最优、六轮全正确、至少 4/6 轮快于本地当前 control、成对中位 speedup 至少 0.5%，并对同配置原版满足同样门槛。comparison-only 镜像不整合。", "",
        "[optimization_records.md](optimization_records.md) 记录远端方案的借鉴、实际测量和未采用方向；[last_round_matched_kernel_trials.csv](last_round_matched_kernel_trials.csv) 保留每个已确认的新版本与同配置原版的比较。", "",
        f"剩余落后中位 {receipt['remaining_losses_median_pct']:.4f}%，最大 {receipt['remaining_losses_max_pct']:.4f}%。完整清单见 [remaining_losses.csv](remaining_losses.csv)。本轮结束后停止优化。", "",
        "| 剩余最优 pipeline | 项数 | 落后中位 % | 最大 % |", "|---|---:|---:|---:|"]
    lines += [f"| {r['pipeline']} | {r['shapes']} | {r['median_slower_pct']:.4f} | {r['max_slower_pct']:.4f} |" for r in pipelines]
    lines += ["", "文件入口：", "",
        "- [full421_comparison.csv](full421_comparison.csv)：最终对固定 gfx950 CSV 的比较，含初始领先分类。",
        "- [all421_alignment.csv](all421_alignment.csv)：优化前的双方结果对齐；远端耗时仅用于参考。",
        "- [shape_evidence421.csv](shape_evidence421.csv)：每项最终证据的轮次和来源。",
        "- [new_integrated_co.csv](new_integrated_co.csv)：本轮整合 CO 的身份、shape 范围和镜像 hash。",
        "- [summary.json](summary.json)、[last_round_completion_receipt.json](last_round_completion_receipt.json)：完整核验回执。",
        "- [run_env.sh](run_env.sh)：当前公开运行环境，使用本地 pin-AGPR LLVM wrapper。",
        "- [remote_snapshot/README.md](remote_snapshot/README.md)：固定远端提交的原始索引，可进入三条 kernel 路线文档。", "",
        "只 tune Opus MXFP8 NT gfx950/256 CU；FlyDSL 使用固定 CSV。公共旧 ID/name/config 保持，JIT 起点 780。A 端 2 GiB guard 和 C 端 64-bit base + tile buffer window 保留。远端混合使用 Top8/101 次与 Top4/11 次计时，不能把跨环境耗时差直接归因于 kernel。"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    details = ["最后一轮优化记录（stage18）", "",
        "远端来源：[optimization_records.md](remote_snapshot/optimization_records.md)、[README.md](remote_snapshot/README.md)，固定提交 `54bef351`。远端的 14 项回退及 171 项 CPU 计划用于选择方向；先与本地最终结果对齐，只处理仍落后的 84 项。", "",
        "本轮构建 12 个新版本、10 个同配置对照；加入远端及回退分析中与本地 84 项相关的 98 条 shape/candidate 记录，去重后形成初筛计划。4wave 方向覆盖提前 scale 加载；interleave 比较未消费 A 行裁剪与独立 B-first；narrow 方向补齐精确既有候选，并按本地当前 winner 测试 m-repeat A_RING=2。", "",
        "提前 scale 加载：在 chunked 主循环 C00 之后发出下一 panel 的 raw 读取，在同一 iteration C11 之后保留原等待和 barrier，再发布。没有改变 MFMA/矩阵数据流。panel32/64、loops1–200 共 400 个调度模型检查通过。提前加载增加寄存器占用，panel64 个别版本存在 SGPR spill，实际资源已逐个从 linked CO 的 ELF notes 读取。", "",
        "interleave：I1 只跳过 BM192 中没有 consumer 的 producer 行 96–127，仍保留真正 tail-M 的零写入；I3 独立按 B0/B1/A0/A1 发出请求。6 个 producer/consumer 覆盖模型通过。两个 B-first linked CO 的 24 条 matrix async-load 指令全部核对，3 组从 AAAABBBB 改为 BBBBAAAA。I1 的静态指令数未减少，其 wave guard 减少实际执行请求。所有 interleave 版本 VGPR182、无 spill、LDS160 KiB。", "",
        "m-repeat：只用 derived traits 将 A_RING 从 3 改为 2，保留原流水线和同步；E_M3/4、ring2/3 的 slot ownership 模型通过。编译后 LDS 与各自原版相同，因为总 workspace 被其他区域/epilogue 约束；不能据 ring2 推断驻留率提升。全部 m-repeat 对照和新版本无 spill。", "",
        "窄 N 远端代表 `(1152,768,7168)`、`(16384,768,7168)`、`(16384,1536,7168)` 仍落后，加入其精确现有候选；`(10240,2048,7168)` 本地已经领先，保持已有行。远端小 M 长 K 代表在本地当前最快属于 m-repeat，因此 interleave 改动同时与同配置原版和本地当前最优比较。", "",
        "测量使用物理 GPU0/3/6/5，PCI 验证；逻辑0/1/2/3映射分别为0000:05:00.0、0000:75:00.0、0000:e5:00.0、0000:95:00.0。原初筛到确认边界的 GPU5 空闲 preflight 读取78%而停止；四个初筛 worker 均退出0，之后重新核对空闲，仅恢复确认，不重跑初筛。原始日志和恢复回执均保留。", "",
        "六轮汇总后发现 GPU5 上六个 shape 的当前 winner/control 计时最大/最小比超过1.5，其中 `(8192,1536,7168)` 同一旧 kernel 在90和210 us间跳变。按照同一规则选出六项，保持候选不变，转到物理 GPU0/3/6 补做完整六轮；全部354条正确，该旧 kernel 恢复至约90.26 us，六轮比约1.012。每个受影响 shape 的整组结果全部替换，未按最快批次挑选；原始4794条及首次公开回执保存在 stage18/initial_verification_snapshot。", "",
        f"本轮实际 {receipt['actual_measurements']} 条性能测量全部正确（初筛3948 + 原确认4794 + 换卡核对354）；最终统计接受初筛3948和确认4794，共8742条，354条受波动影响的原确认记录仅归档。799个 shape/candidate 组合六轮完整。公开最优的稳定候选收益 {receipt['stable_candidate_improved_shapes']} 项；新 CO 成为 {receipt['new_co_best_shapes']} 项最优，新增领先 {len(added)} 项。详见逐项 CSV，最终收益不从远端计时推导。", "",
        "| 新版本 pipeline | 六轮 matched 比较对数 | 达到≥0.5%、≥4/6快 | 成为公开最优 |", "|---|---:|---:|---:|"]
    for p in sorted({r["pipeline"] for r in trials}):
        rs = [r for r in trials if r["pipeline"] == p]
        details.append(f"| {p} | {len(rs)} | {sum(r['matched_material_gain'] for r in rs)} | {sum(r['selected_as_public_best'] for r in rs)} |")
    details += ["", "实际整合 kernel 的两项收益：", "",
        "| M / N / K | 对本地当前最优耗时下降 % | 对同配置原版耗时下降 % | 更快轮数 | 最终仍落后 % |",
        "|---|---:|---:|---:|---:|"]
    for r in comparisons:
        if yes(r["new_co"]):
            details.append(f"| {r['M']} / {r['N']} / {r['K']} | {float(r['winner_gain_vs_control_pct']):.4f} | {float(r['matched_time_reduction_pct']):.4f} | {r['paired_faster_rounds']}/6 | {float(r['winner_slower_than_csv_pct']):.4f} |")
    details += ["", "表中 matched 收益包括未成为本地最优的实验候选，不能等同新增领先或整合收益。全部试验保留，最终只使用公共可运行候选。", "",
        "第18轮为本次最后一轮；剩余落后项仅归档，不继续扩新候选或启动第19轮。"]
    (OUT / "optimization_records.md").write_text("\n".join(details) + "\n")
    dump(OUT / "files.json", {str(p.relative_to(OUT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                             for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "files.json"})
    assert len(csv_rows(OUT / "leading_tuned.csv")) == len(final_wins)
    assert len(csv_rows(OUT / "full421_tuned.csv")) == 421
    print(json.dumps({"directory": str(OUT), "wins": len(final_wins), "losses": len(losses), "new_wins": len(added), "new_win_attribution": dict(categories), "new_co_ids": integration["selected_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
