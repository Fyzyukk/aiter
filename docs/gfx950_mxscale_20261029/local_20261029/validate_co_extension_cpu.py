#!/usr/bin/env python3
"""CPU mechanism fixtures only: no extension, compilation, or GPU work."""

import copy
import json
import tempfile
from pathlib import Path

import extend_co_experiment as extension


def main():
    local = extension.LOCAL
    repo = local / "co_round1/.cache/experiment_repo"
    manifest_path = repo / extension.MANIFEST_REL
    manifest_sha = extension.digest(manifest_path)
    document = json.loads(manifest_path.read_text())
    if len(document["gfx950"]) == 804:
        base = extension.load_common(repo, label="extend_fixture_base")
    else:
        # Once root applies round2, the live clone is larger. Use its retained
        # pre-extension snapshot to repeat this CPU mechanism check without
        # changing the live manifest or dropping any new configurations.
        snapshot = local / "co_round2/base_manifest.json"
        document = json.loads(snapshot.read_text())
        if len(document["gfx950"]) != 804:
            raise ValueError("CPU fixture needs the retained original804 snapshot")
        base = extension.load_common(repo, manifest=snapshot, label="extend_fixture_snapshot_base")
    report_path = local / "co_round1/repeat_report/report.json"
    report = json.loads(report_path.read_text())
    baseline = {extension.key(row): float(row["us"]) for row in extension.csv_rows(extension.DATA / "csv_baseline.csv")}
    audited = extension.audit_repeat(report, extension.resolve_files(report["summary"]["files"]), baseline,
                                     extension.registry_metadata(base), .05)
    fixture_reason = "CPU mechanism fixture only; fabricated selection is not a GPU optimization plan"
    plan = {"source_report": str(report_path), "experiments": [
        {"seed_kid": 100001, "vary": {"GroupM": 16}, "shapes": [[8192, 7168, 16384]], "reason": fixture_reason},
        {"seed_kid": 100001, "vary": {"Panel": 32}, "shapes": [[8192, 7168, 16384]], "reason": fixture_reason}],
        "paired_experiments": [{"b_seed_kid": 100020, "overlap_seed_kid": 100021, "vary": {"NB": 2},
                                "shapes": [[2048, 768, 7168], [8192, 768, 7168]], "reason": fixture_reason}]}
    proposed, points = extension.prepare_document(document, plan)
    with tempfile.TemporaryDirectory(prefix="extend_fixture_", dir=local) as temporary:
        path = Path(temporary) / "manifest.json"
        extension.dump(path, proposed)
        extended = extension.load_common(repo, manifest=path, label="extend_fixture_proposed")
        identity = extension.check_identity(base, extended, [point["new_kid"] for point in points])
        extension.validate_points(base, extended, points, baseline, audited)
        helper = extension.make_repeat_plan(points, extended, baseline, report_path)
        assert len(helper["shapes"]) == 3
        for point in points:
            for shape in point["shapes"]:
                item = next(item for item in helper["shapes"] if [item["shape"][axis] for axis in ("M", "N", "K")] == shape)
                assert {point["new_kid"], point["seed_kid"]} <= {row["kid"] for row in item["candidates"]}
        bad_point = copy.deepcopy(next(point for point in points if point["seed_kid"] == 100020))
        bad_point["shapes"] = [[4096, 1536, 7168]]
        try:
            extension.validate_points(base, extended, [bad_point], baseline, audited)
        except ValueError:
            illegal_scope_rejected = True
        else:
            raise AssertionError("Old seed scope was widened")
    checks = {"actual_round1_candidates_checked": len(audited), "actual_round1_all_correct": sum(row["valid"] for row in audited.values()),
              "old_ids_preserved": identity, "new_and_seed_helper_coverage": True,
              "GroupM16_Panel32_B_NB2_all_legal_with_runtimeK_Unroll2": True,
              "illegal_seed_scope_rejected": illegal_scope_rejected}
    entry = next(row for row in document["gfx950"] if row["kid"] == 100001)
    for name, vary in (("multi_axis", {"GroupM": 16, "Panel": 32}), ("GroupM32", {"GroupM": 32}), ("unchanged_GroupM", {"GroupM": 8})):
        try:
            extension.transform(entry, vary, [[8192, 7168, 16384]], 100001)
        except ValueError:
            checks[name + "_rejected"] = True
        else:
            raise AssertionError(name)
    for name, seed, vary in (("existing_device_config", 100000, {"GroupM": 8}), ("Unroll4_outside_current_scope", 100001, {"Unroll": 4})):
        bad = {"source_report": str(report_path), "experiments": [{"seed_kid": seed, "vary": vary,
               "shapes": [[8192, 7168, 16384]], "reason": fixture_reason}]}
        try:
            extension.prepare_document(document, bad)
        except ValueError:
            checks[name + "_rejected"] = True
        else:
            raise AssertionError(name)
    b = next(point for point in points if point["seed_kid"] == 100020)
    assert b["before"]["B_AHEAD"] == b["after"]["B_AHEAD"] == 2 and b["after"]["LDS_BYTES"] == 41984
    checks["NB4_to2_BA2_LDS41984"] = True
    with tempfile.TemporaryDirectory(prefix="build_merge_fixture_", dir=local) as temporary:
        a, b, merged = (Path(temporary) / filename for filename in ("a.json", "b.json", "merged.json"))
        extension.dump(a, {"llvm_bin": "fixture", "kernels": [{"kid": 100000, "status": "ready"}]})
        extension.dump(b, {"llvm_bin": "fixture", "kernels": [{"kid": 100024, "status": "ready"}]})
        extension.merge_reports([a, b], merged)
        assert [row["kid"] for row in json.loads(merged.read_text())["kernels"]] == [100000, 100024]
        checks["separate_reports_merge_without_loss"] = True
        try:
            extension.merge_reports([a, a], Path(temporary) / "duplicate.json")
        except ValueError:
            checks["duplicate_build_evidence_rejected"] = True
        else:
            raise AssertionError("Duplicate build evidence was overwritten")
    assert extension.digest(manifest_path) == manifest_sha
    result = {"fixture_only": True, "gpu_used": False, "build_used": False, "manifest_extended": False, "checks": checks}
    extension.dump(local / "co_extension_cpu_fixture_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
