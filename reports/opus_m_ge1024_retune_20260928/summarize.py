#!/usr/bin/env python3
"""Summarize this one recorded sweep using CSV/log/file reads only.

Run after run.sh exits. No torch, tuner imports, subprocesses, or GPU work.
Exit 0 means the recorded sweep passes the explicit completeness checks;
exit 1 leaves the same four reports with incomplete/failure evidence.
"""

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SHAPE_KEYS = ("gfx", "cu_num", "M", "N", "K")
BACKENDS = ("ck", "cktile", "asm", "opus")
OLD_KIDS = {9000, 9020}
NEW_KIDS = {9060, 9061, 9062, 9063, 9064}
GROUPS = ("opus", "external", "new5", "original2")
EXPECTED_COUNT = 305


def shape_dict(shape):
    return dict(zip(SHAPE_KEYS, shape))


def parse_int(value):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"not an integer: {value!r}")
    return int(number)


def parse_shape(row):
    shape = (row["gfx"], *(parse_int(row[key]) for key in SHAPE_KEYS[1:]))
    if not shape[0] or min(shape[1:]) <= 0:
        raise ValueError(f"invalid shape: {shape}")
    return shape


def read_csv(path, problems):
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header")
            return list(reader)
    except (OSError, ValueError, csv.Error) as exc:
        problems.append(f"{path.name}: {exc}")
        return []


def read_text(path):
    return path.read_text(errors="replace").strip() if path.is_file() else None


def parse_result_rows(raw_rows, label, problems):
    results = []
    for line, row in enumerate(raw_rows, 2):
        try:
            shape = parse_shape(row)
            lib = row["libtype"].strip()
            kid, split = parse_int(row["kernelId"]), parse_int(row["splitK"])
            if lib not in BACKENDS or kid < 0 or split < 0:
                raise ValueError("invalid backend/kernelId/splitK")
            us, error = float(row["us"]), float(row["errRatio"])
            if not math.isfinite(us):
                status = "invalid_timing_or_worker_failure"
            elif us <= 0:
                # CSV alone cannot separate an unsupported shape from runtime failure.
                status = "unsupported_or_runtime_failure"
            elif not math.isfinite(error) or error < 0:
                status = "invalid_error_metadata"
            elif error != 0:
                status = "numerical_rejection"
            else:
                status = "valid"
            results.append({
                "shape": shape, "libtype": lib, "kernelId": kid,
                "splitK": split, "kernelName": row.get("kernelName", ""),
                "us": us, "errRatio": error, "status": status, "line": line,
            })
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{label}:{line}: {exc}")
    return results


def identity(row):
    return (row["libtype"], row["kernelId"], row["splitK"], row["kernelName"])


def brief(row):
    if row is None:
        return None
    return {
        **shape_dict(row["shape"]),
        **{key: row[key] for key in ("libtype", "kernelId", "splitK", "kernelName")},
        "us": row["us"] if math.isfinite(row["us"]) else None,
        "errRatio": row["errRatio"] if math.isfinite(row["errRatio"]) else None,
        "status": row["status"],
    }


def in_group(row, group):
    if group == "external":
        return row["libtype"] in ("ck", "cktile", "asm")
    if row["libtype"] != "opus":
        return False
    return group == "opus" or row["kernelId"] in (
        NEW_KIDS if group == "new5" else OLD_KIDS
    )


def fastest(rows):
    valid = [row for row in rows if row["status"] == "valid"]
    return min(valid, key=lambda row: (row["us"], identity(row))) if valid else None


def compare_groups(expected, best, lhs, rhs):
    paired, counts = [], Counter()
    for shape in expected:
        left, right = best[shape][lhs], best[shape][rhs]
        if left is None or right is None:
            key = "lhs_only_valid" if left else "rhs_only_valid" if right else "neither_valid"
            counts[key] += 1
            continue
        ratio = left["us"] / right["us"]
        counts["faster" if ratio < 1 else "slower" if ratio > 1 else "tie"] += 1
        paired.append({
            **shape_dict(shape), "lhs": brief(left), "rhs": brief(right),
            "lhs_over_rhs": ratio, "rhs_over_lhs_speedup": 1 / ratio,
            "lhs_time_reduction_pct": 100 * (1 - ratio),
        })
    gm = math.exp(sum(math.log(item["lhs_over_rhs"]) for item in paired) / len(paired)) if paired else None
    return {
        "lhs": lhs, "rhs": rhs, "comparable_shapes": len(paired),
        **{key: counts[key] for key in ("faster", "tie", "slower", "lhs_only_valid", "rhs_only_valid", "neither_valid")},
        "geomean_lhs_over_rhs": gm,
        "geomean_speedup_rhs_over_lhs": 1 / gm if gm is not None else None,
        "geomean_lhs_time_reduction_pct": 100 * (1 - gm) if gm is not None else None,
        "worst_relative": sorted(paired, key=lambda item: item["lhs_over_rhs"], reverse=True)[:10],
        "slowest_lhs_us": sorted(paired, key=lambda item: item["lhs"]["us"], reverse=True)[:10],
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    directory = Path(__file__).resolve().parent
    record = json.loads((directory / "run.json").read_text())
    repo = Path(record["cwd"])
    command = record["command"]
    input_path = Path(command[command.index("-i") + 1])
    if not input_path.is_absolute():
        input_path = repo / input_path
    problems = []
    input_rows = read_csv(input_path, problems)
    expected_list = []
    for line, row in enumerate(input_rows, 2):
        try:
            expected_list.append(parse_shape(row))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"input:{line}: {exc}")
    expected = sorted(set(expected_list))
    expected_set = set(expected)
    profile_raw = read_csv(directory / "profile.csv", problems)
    tuned_raw = read_csv(directory / "tuned.csv", problems)
    profile = parse_result_rows(profile_raw, "profile.csv", problems)
    tuned = parse_result_rows(tuned_raw, "tuned.csv", problems)
    profile_by_shape, tuned_by_shape = defaultdict(list), defaultdict(list)
    for row in profile:
        profile_by_shape[row["shape"]].append(row)
    for row in tuned:
        tuned_by_shape[row["shape"]].append(row)

    duplicates = Counter((row["shape"], identity(row)) for row in profile)
    duplicate_profile = [{**shape_dict(shape), "candidate": candidate, "count": count}
                         for (shape, candidate), count in duplicates.items() if count > 1]
    missing_tuned = sorted(expected_set - tuned_by_shape.keys())
    missing_profile = sorted(expected_set - profile_by_shape.keys())
    extra_tuned = sorted(tuned_by_shape.keys() - expected_set)
    extra_profile = sorted(profile_by_shape.keys() - expected_set)
    choice_issues = []
    for shape in expected:
        rows = tuned_by_shape[shape]
        if not rows:
            # Coverage is checked separately; absence is not an invalid saved choice.
            continue
        if len(rows) != 1:
            choice_issues.append({**shape_dict(shape), "issue": f"expected one tuned row, got {len(rows)}"})
            continue
        selected = rows[0]
        valid = [row for row in profile_by_shape[shape] if row["status"] == "valid"]
        if selected["status"] != "valid":
            choice_issues.append({**shape_dict(shape), "issue": "selected row is invalid", "row": brief(selected)})
        elif not valid:
            choice_issues.append({**shape_dict(shape), "issue": "no valid profile candidate"})
        elif selected["us"] != min(row["us"] for row in valid):
            choice_issues.append({**shape_dict(shape), "issue": "selected time is not the minimum valid profile time", "row": brief(selected)})
        elif not any(identity(row) == identity(selected) and row["us"] == selected["us"] for row in valid):
            choice_issues.append({**shape_dict(shape), "issue": "selected candidate/time absent from valid profile rows", "row": brief(selected)})

    best = {shape: {group: fastest([row for row in profile_by_shape[shape] if in_group(row, group)])
                    for group in GROUPS} for shape in expected}
    comparisons = {
        "opus_vs_external": compare_groups(expected, best, "opus", "external"),
        "new5_vs_original2": compare_groups(expected, best, "new5", "original2"),
        "all7_vs_original2": compare_groups(expected, best, "opus", "original2"),
    }
    participation = {}
    for lib in BACKENDS:
        rows = [row for row in profile if row["libtype"] == lib and row["shape"] in expected_set]
        shapes = {row["shape"] for row in rows}
        valid_shapes = {row["shape"] for row in rows if row["status"] == "valid"}
        participation[lib] = {
            "candidate_rows": len(rows), "shapes_with_candidates": len(shapes),
            "shapes_with_valid_candidates": len(valid_shapes),
            "row_status_counts": dict(Counter(row["status"] for row in rows)),
            "shapes_without_candidates": [shape_dict(shape) for shape in expected if shape not in shapes],
            "shapes_without_valid_candidates": [shape_dict(shape) for shape in expected if shape not in valid_shapes],
        }

    hashes = []
    for name, before in record.get("source_sha256", {}).items():
        path = repo / name
        after = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        hashes.append({"path": name, "before": before, "after": after, "unchanged": after == before})

    log_path = directory / "tune.log"
    log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", read_text(log_path) or "")
    finish_counts = [int(value) for value in re.findall(r"Tuning Finished\. tune (\d+) shapes", log)]
    patterns = {
        "completion": r"Tuning Finished\.|Processing Status.*100\.0%",
        "fatal": r"error in batch|tune_summary failed|synchronous build_module FAILED|subset-compile rebuild failed|Tuning (?:Error|Interrupted)\.",
        "worker_failure": r"timed out after|likely stuck|Segmentation fault|Memory access fault|HIP error|CUDA error|GPU Runtime Error",
        "incomplete_selection": r"\[Tuning not Finished\]|No kernel can be used",
        "candidate_warning": r"run gpu func warning|Runtime Error in process",
        "missing_asm_list": r"ASM kernel list file not exist",
        "distribution": r"Distributing \d+ task groups across \d+ GPUs",
    }
    evidence = {}
    for name, pattern in patterns.items():
        matches = [line for line in log.splitlines() if re.search(pattern, line, re.IGNORECASE)]
        evidence[name] = {"count": len(matches), "last_lines": matches[-12:]}
    exit_text = read_text(directory / "exit_code.txt")
    exit_code_source = "exit_code.txt" if exit_text is not None else "run.json"
    exit_value = exit_text
    if exit_value is None:
        for source in (record, record.get("process", {})):
            for key in ("exit_code", "returncode", "return_code"):
                if source.get(key) is not None:
                    exit_value = source[key]
                    break
            if exit_value is not None:
                break
    try:
        exit_code = parse_int(exit_value) if exit_value is not None else None
    except (TypeError, ValueError):
        exit_code = None
        problems.append(f"invalid process exit code: {exit_value!r}")

    unexpected_opus = [brief(row) for row in profile if row["libtype"] == "opus" and row["kernelId"] not in OLD_KIDS | NEW_KIDS]
    checks = {
        "input_has_exactly_305_unique_rows": len(expected) == len(input_rows) == EXPECTED_COUNT,
        "input_matches_recorded_count": len(expected) == record.get("shape_count"),
        "input_is_gfx950_256cu_m_ge1024": bool(expected) and all(shape[0] == "gfx950" and shape[1] == 256 and shape[2] >= 1024 for shape in expected),
        "csv_rows_parse": not problems,
        "profile_covers_input_exactly": not missing_profile and not extra_profile and bool(expected),
        "tuned_covers_input_exactly": not missing_tuned and not extra_tuned and len(tuned) == len(expected) and bool(expected),
        "no_repeated_profile_candidates": not duplicate_profile,
        "all_selected_rows_are_valid_minima_with_zero_error": not choice_issues and bool(tuned),
        "all_four_backends_have_candidate_records": all(participation[lib]["candidate_rows"] > 0 for lib in BACKENDS),
        "only_requested_opus_ids": not unexpected_opus,
        "protected_source_hashes_unchanged": bool(hashes) and all(item["unchanged"] for item in hashes),
        "process_exit_zero": exit_code == 0,
        "log_confirms_305_finished_shapes": EXPECTED_COUNT in finish_counts,
        "no_fatal_or_worker_failure_log_evidence": evidence["fatal"]["count"] == evidence["worker_failure"]["count"] == 0,
        "no_unresolved_shapes_log_evidence": evidence["incomplete_selection"]["count"] == 0,
        "no_missing_asm_list": evidence["missing_asm_list"]["count"] == 0,
    }
    limitations = [
        "One sweep only: 5 warmups and 51 timing iterations per candidate; these are not independent tuning rounds or evidence of stable wins across rounds.",
        "OPUS uses native E8M0 scales; CK/CKTile/ASM use independently random FP32 scales and each dataset has its own reference. Cross-backend timings compare backend workloads, not mathematically identical input tensors.",
        "Only this profile's valid candidates are compared; historical CSV timings are not used. Geometric means weight each comparable shape equally; ratio is lhs_us/rhs_us, speedup is rhs_us/lhs_us.",
        "No OPUS profile rows means no OPUS candidate was recorded for that shape. If the sweep is incomplete, absence does not establish lack of kernel support.",
        "us<=0 combines unsupported candidates and runtime failures because CSV alone cannot distinguish them. Positive finite timings with nonzero finite errRatio are numerical rejections.",
        "Recorded backend participation and shape coverage do not independently prove that every possible registered candidate was enumerated.",
    ]
    selected_valid = [row for row in tuned if row["shape"] in expected_set and row["status"] == "valid"]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "complete_and_valid": all(checks.values()), "checks": checks,
        "input_file": str(input_path), "input_shapes": len(expected),
        "profile_rows": len(profile_raw), "parsed_profile_rows": len(profile),
        "tuned_rows": len(tuned_raw), "parsed_tuned_rows": len(tuned),
        "profile_status_counts": dict(Counter(row["status"] for row in profile)),
        "winner_backend_counts": dict(Counter(row["libtype"] for row in selected_valid)),
        "winner_opus_id_counts": dict(Counter(str(row["kernelId"]) for row in selected_valid if row["libtype"] == "opus")),
        "coverage": {"missing_tuned": list(map(shape_dict, missing_tuned)), "missing_profile": list(map(shape_dict, missing_profile)),
                     "extra_tuned": list(map(shape_dict, extra_tuned)), "extra_profile": list(map(shape_dict, extra_profile))},
        "choice_issues": choice_issues, "duplicate_profile_candidates": duplicate_profile,
        "unexpected_opus_candidates": unexpected_opus, "read_or_parse_problems": problems,
        "backend_participation": participation, "comparisons": comparisons,
        "opus_without_candidates": participation["opus"]["shapes_without_candidates"],
        "opus_without_valid_candidates": participation["opus"]["shapes_without_valid_candidates"],
        "slowest_selected_us": [brief(row) for row in sorted(selected_valid, key=lambda row: row["us"], reverse=True)[:10]],
        "source_sha256": hashes,
        "process": {"exit_code": exit_code, "exit_code_source": exit_code_source if exit_value is not None else None,
                    "recorded_status": record.get("status"), "started_utc": read_text(directory / "started_utc.txt"),
                    "ended_utc": read_text(directory / "ended_utc.txt")},
        "log_evidence": evidence, "limitations": limitations,
    }
    comparison_rows = []
    for shape in expected:
        row = shape_dict(shape)
        selected = tuned_by_shape[shape][0] if len(tuned_by_shape[shape]) == 1 else None
        for key in ("libtype", "kernelId", "us", "errRatio"):
            row[f"selected_{key}"] = brief(selected)[key] if selected else ""
        for group in GROUPS:
            candidates = [item for item in profile_by_shape[shape] if in_group(item, group)]
            winner = best[shape][group]
            row[f"{group}_candidate_rows"] = len(candidates)
            row[f"{group}_valid_rows"] = sum(item["status"] == "valid" for item in candidates)
            row[f"{group}_numerical_rejections"] = sum(item["status"] == "numerical_rejection" for item in candidates)
            row[f"{group}_state"] = "valid" if winner else "no_valid_candidate" if candidates else "no_recorded_candidate"
            for key in ("libtype", "kernelId", "splitK", "us"):
                row[f"{group}_{key}"] = winner[key] if winner else ""
        for name, (left, right) in {"opus_vs_external": ("opus", "external"), "new5_vs_original2": ("new5", "original2"), "all7_vs_original2": ("opus", "original2")}.items():
            lhs, rhs = best[shape][left], best[shape][right]
            row[f"{name}_lhs_over_rhs"] = lhs["us"] / rhs["us"] if lhs and rhs else ""
        comparison_rows.append(row)
    comparison_fields = list(SHAPE_KEYS) + [f"selected_{key}" for key in ("libtype", "kernelId", "us", "errRatio")]
    comparison_fields += [f"{group}_{key}" for group in GROUPS for key in ("candidate_rows", "valid_rows", "numerical_rejections", "state", "libtype", "kernelId", "splitK", "us")]
    comparison_fields += ["opus_vs_external_lhs_over_rhs", "new5_vs_original2_lhs_over_rhs", "all7_vs_original2_lhs_over_rhs"]
    write_csv(directory / "comparison.csv", comparison_rows, comparison_fields)

    usage = defaultdict(list)
    for row in profile:
        if row["shape"] in expected_set:
            # ASM kernelId is assigned locally per shape; name/splitK is its stable identity.
            key = (row["libtype"], None if row["libtype"] == "asm" else row["kernelId"], row["splitK"], row["kernelName"])
            usage[key].append(row)
    usage_rows = []
    for (lib, kid, split, name), rows in sorted(usage.items()):
        valid = [row for row in rows if row["status"] == "valid"]
        selected_shapes = {row["shape"] for row in selected_valid if row["libtype"] == lib and row["splitK"] == split
                           and row["kernelName"] == name and (kid is None or row["kernelId"] == kid)}
        usage_rows.append({
            "libtype": lib, "kernelId": "per-shape" if kid is None else kid,
            "kernelIds_seen": ";".join(map(str, sorted({row["kernelId"] for row in rows}))),
            "splitK": split, "kernelName": name,
            "family": "new5" if lib == "opus" and kid in NEW_KIDS else "original2" if lib == "opus" and kid in OLD_KIDS else lib,
            "candidate_rows": len(rows), "candidate_shapes": len({row["shape"] for row in rows}),
            "valid_rows": len(valid), "valid_shapes": len({row["shape"] for row in valid}),
            "unsupported_or_runtime_failure": sum(row["status"] == "unsupported_or_runtime_failure" for row in rows),
            "numerical_rejection": sum(row["status"] == "numerical_rejection" for row in rows),
            "other_invalid": sum(row["status"] in ("invalid_timing_or_worker_failure", "invalid_error_metadata") for row in rows),
            "selected_shapes": len(selected_shapes),
        })
    usage_fields = ("libtype", "kernelId", "kernelIds_seen", "splitK", "kernelName", "family", "candidate_rows", "candidate_shapes", "valid_rows", "valid_shapes", "unsupported_or_runtime_failure", "numerical_rejection", "other_invalid", "selected_shapes")
    write_csv(directory / "candidate_usage.csv", usage_rows, usage_fields)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    fmt = lambda value: "N/A" if value is None else f"{value:.6f}"
    report = [
        "# 305-shape MXFP8 B-preshuffle retune", "",
        f"验证结果：{'本轮记录完整且通过全部列明检查' if summary['complete_and_valid'] else '未获得全部输入的有效选择；见下方缺失项与检查结果'}。",
        f"输入 {len(expected)} shape；profile {len(profile_raw)} 行；tuned {len(tuned_raw)} 行；进程退出码 {exit_code}。", "",
        f"最优后端计数：`{json.dumps(summary['winner_backend_counts'], ensure_ascii=False)}`。",
        f"OPUS 最优 ID 计数：`{json.dumps(summary['winner_opus_id_counts'], ensure_ascii=False)}`。",
        f"候选状态计数：`{json.dumps(summary['profile_status_counts'], ensure_ascii=False)}`。", "",
        "| 比较（左方 vs 右方） | 可比 shape | 左快 / 平 / 左慢 | 等权几何平均 左/右 | 几何平均 speedup 右/左 | 左方耗时降低 % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in comparisons.items():
        report.append(f"| {name} | {result['comparable_shapes']} | {result['faster']} / {result['tie']} / {result['slower']} | {fmt(result['geomean_lhs_over_rhs'])} | {fmt(result['geomean_speedup_rhs_over_lhs'])} | {fmt(result['geomean_lhs_time_reduction_pct'])} |")
    report += ["", "仅双方都有本批有效候选的 shape 进入对应比较；没有有效候选的数量和全部逐 shape 数值见 summary.json、comparison.csv。", "",
               "| 后端 | 候选记录 | 有候选 shape | 有有效候选 shape |", "|---|---:|---:|---:|"]
    for lib, data in participation.items():
        report.append(f"| {lib} | {data['candidate_rows']} | {data['shapes_with_candidates']} | {data['shapes_with_valid_candidates']} |")
    report += ["", "检查失败项：" + (", ".join(key for key, ok in checks.items() if not ok) or "无") + "。",
               f"受保护文件 SHA256：{sum(item['unchanged'] for item in hashes)}/{len(hashes)} 一致；具体路径与前后哈希见 summary.json。", ""]
    report += ["未获得有效最优项的 shape（M/N/K）：", ""]
    report.extend(f"- `{shape[2]}/{shape[3]}/{shape[4]}`" for shape in missing_tuned)
    if not missing_tuned:
        report.append("无。")
    report.append("")
    for key in ("completion", "fatal", "worker_failure", "incomplete_selection", "missing_asm_list", "distribution"):
        report.append(f"日志 {key}：{evidence[key]['count']} 条。")
        report.extend(f"- `{line}`" for line in evidence[key]["last_lines"][-2:])
        report.append("")
    report += ["OPUS 无候选记录的 shape（gfx/CU/M/N/K）：", ""]
    absent = participation["opus"]["shapes_without_candidates"]
    report.extend(f"- `{item['gfx']}/{item['cu_num']}/{item['M']}/{item['N']}/{item['K']}`" for item in absent)
    if not absent:
        report.append("无。")
    for name, result in comparisons.items():
        report += ["", f"{name} 相对最慢的 5 项（左/右越大越慢；完整前 10 项及绝对耗时最慢项见 summary.json）：", ""]
        for item in result["worst_relative"][:5]:
            report.append(f"- M/N/K={item['M']}/{item['N']}/{item['K']}：{item['lhs']['us']:.4f} / {item['rhs']['us']:.4f} us，左/右={item['lhs_over_rhs']:.6f}。")
        if not result["worst_relative"]:
            report.append("没有双方均有效的 shape。")
    report += ["", "解释限制：", ""] + [f"- {item}" for item in limitations]
    (directory / "RESULTS.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"complete_and_valid": summary["complete_and_valid"], "input_shapes": len(expected),
                      "profile_rows": len(profile_raw), "tuned_rows": len(tuned_raw),
                      "failed_checks": [key for key, ok in checks.items() if not ok]}, ensure_ascii=False))
    return 0 if summary["complete_and_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
