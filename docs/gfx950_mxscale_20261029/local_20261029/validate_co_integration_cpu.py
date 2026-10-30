#!/usr/bin/env python3
"""CPU-only integration fixtures; all fake images/public mutations are temporary.

Synthetic timing and ELF-header fixtures are explicitly not GPU evidence. The
only retained output is co_integration_cpu_fixture_report.json.
"""

import argparse
import copy
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import integrate_co_winners as integration
from summarize_opus_repeat import candidate_statistics


def row_and_events(kid, times, *, rotate=0, error=.001, missing=False, correct=True):
    events = {index: {"kid": kid, "round": index, "us": time, "errRatio": error,
                      "correct": correct, "rotate_args": rotate, "gpu_logical_id": 0,
                      "failure": None, "csv_baseline_us": 20.0,
                      "position": int(kid >= 100000) if index % 2 == 0 else int(kid < 100000)}
              for index, time in enumerate(times)}
    if missing:
        events.pop(5)
    shape = ("gfx950", 256, 8192, 6144, 7168, "128x128")
    row = candidate_statistics(shape, kid, {"kernelName": "fixture", "kind": "CO"},
                               events, 20.0, argparse.Namespace(rounds=6, err_ratio=.05))
    return row, events


def main():
    results = {}
    parent_row, parent_events = row_and_events(0, [10.0] * 6)
    parent = integration.audit_candidate(parent_row, parent_events, 20.0)
    cases = {"stable_six_round_win": ([9.0] * 6, {}, True),
             "equal_median_rejected": ([10.0] * 6, {}, False),
             "slower_median_rejected": ([11.0] * 6, {}, False),
             "missing_round_rejected": ([9.0] * 6, {"missing": True}, False),
             "error_rounds_rejected": ([9.0] * 6, {"error": .1}, False),
             "false_correct_rejected": ([9.0] * 6, {"correct": False}, False),
             "zero_time_rejected": ([0.0] * 6, {}, False),
             "nan_time_rejected": ([float("nan")] * 6, {}, False),
             "hot_cache_rejected": ([9.0] * 6, {"rotate": 1}, False)}
    for name, (times, options, expected) in cases.items():
        row, events = row_and_events(100000, times, **options)
        candidate = integration.audit_candidate(row, events, 20.0)
        assert integration.compare_pair(candidate, parent)["stable_improvement"] is expected, name
        results[name] = "passed"
    a, b = row_and_events(0, [10, 10, 10, 100, 100, 100])
    c, d = row_and_events(100000, [11, 11, 11, 90, 90, 90])
    comparison = integration.compare_pair(integration.audit_candidate(c, d, 20.0), integration.audit_candidate(a, b, 20.0))
    assert comparison["new_median_us"] < comparison["parent_median_us"] and not comparison["stable_improvement"]
    assert comparison["paired_round_wins"] == 3
    results["lower_median_but_only_three_paired_wins_rejected"] = "passed"
    c, d = row_and_events(100000, [9.0] * 6)
    d[0]["gpu_logical_id"] = 1
    assert not integration.compare_pair(integration.audit_candidate(c, d, 20.0), parent)["stable_improvement"]
    results["paired_gpu_mismatch_rejected"] = "passed"
    c, d = row_and_events(100000, [9.0] * 6)
    for event in d.values():
        event["position"] = 1
    constant_parent = copy.deepcopy(parent)
    for event in constant_parent["records"].values():
        event["position"] = 0
    order_result = integration.compare_pair(integration.audit_candidate(c, d, 20.0), constant_parent)
    assert not order_result["stable_improvement"] and "paired_round_order_not_alternated" in order_result["reasons"]
    results["missing_order_reversal_rejected"] = "passed"
    c, d = row_and_events(100000, [9.0] * 6)
    c["median_us"] = 8.9
    try:
        integration.audit_candidate(c, d, 20.0)
    except ValueError:
        results["forged_report_median_rejected"] = "passed"
    else:
        raise AssertionError("Forged median was accepted")

    original_root = integration.ROOT
    original_hashes = {str(rel): integration.digest(original_root / rel) for rel in (integration.COMMON_REL, integration.MANIFEST_REL)}
    with tempfile.TemporaryDirectory(prefix="co_integration_fixture_", dir=integration.LOCAL) as temp:
        temporary = Path(temp)
        public = temporary / "fixture_public"
        experiment = temporary / "fixture_experiment"
        repo = experiment / ".cache/experiment_repo"
        for relative in (Path("csrc/opus_gemm"), Path("csrc/include")):
            for source in (original_root / relative).rglob("*"):
                if source.is_file() and source.suffix in integration.SOURCE_SUFFIXES:
                    destination = public / source.relative_to(original_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        (public / integration.MANIFEST_REL).write_bytes((original_root / integration.MANIFEST_REL).read_bytes())
        shutil.copytree(public, repo)
        doc = json.loads((public / integration.MANIFEST_REL).read_text())
        plan = {"experiments": [{"seed_kid": 0, "vary": {"GroupM": 4},
            "shapes": [[8192, 6144, 7168], [6144, 6144, 7168]],
            "reason": "CPU fixture only, fabricated data; never selected performance evidence"}]}
        extended_doc, points = integration.prepare_manifest(doc, plan, 1)
        integration.dump(repo / integration.MANIFEST_REL, extended_doc)
        common_source = (repo / integration.COMMON_REL).read_text()
        (repo / integration.COMMON_REL).write_text(common_source.replace(integration.JIT_BASE_LINE, integration.FROZEN_BASE_LINE))
        common = integration.load_common(repo, label="integration_fixture")
        metadata = integration.registry_metadata(common)
        points[0]["kernelName"] = metadata[100000]["kernelName"]
        integration.dump(repo / integration.MANIFEST_REL, extended_doc)
        integration.dump(experiment / "selection_plan.json", plan)
        integration.dump(experiment / "identity_and_points.json", {"identity": {"original_config_digest": integration.ORIGINAL_DIGEST}, "points": points})
        image_entry = common.gfx950_co_kernels_declared[100000]
        image = repo / integration.MANIFEST_REL.parent / "gfx950" / image_entry["image"]
        image.parent.mkdir(parents=True)
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (224).to_bytes(2, "little")
        header[48:52] = (0x4F).to_bytes(4, "little")
        image.write_bytes(header)
        build_row = {"kid": 100000, "status": "ready", "symbol": image_entry["symbol"], "image": image_entry["image"],
                     "offload_arch": "gfx950", "elf_machine": 224, "elf_processor_id": 0x4F, "undefined_symbols": [],
                     "fixture_only": True, "co_sha256": integration.digest(image)}
        integration.dump(experiment / "new_co_build_info.json", {"kernels": [build_row], "fixture_only": True})
        baseline = {integration.key(row): float(row["us"]) for row in integration.csv_rows(integration.DATA / "csv_baseline.csv")}
        candidates, raw = [], []
        for mnk, new_time in (([8192, 6144, 7168], 9.0), ([6144, 6144, 7168], 11.0)):
            shape = ("gfx950", 256, *mnk, "128x128")
            for kid, time in ((0, 10.0), (100000, new_time)):
                events = {index: {"shape": integration.shape_row(shape), "kid": kid, "round": index,
                                  "kernelName": metadata[kid]["kernelName"], "kind": "CO", "us": time,
                                  "errRatio": .001, "correct": True, "rotate_args": 0, "gpu_logical_id": 0,
                                  "failure": None, "csv_baseline_us": baseline[shape],
                                  "position": int(kid >= 100000) if index % 2 == 0 else int(kid < 100000)} for index in range(6)}
                stat = candidate_statistics(shape, kid, metadata[kid], events, baseline[shape], argparse.Namespace(rounds=6, err_ratio=.05))
                stat.update(metadata[kid])
                candidates.append(stat)
                raw.extend(events.values())
        raw_path = experiment / "fixture_repeat.jsonl"
        raw_path.write_text("".join(json.dumps(event) + "\n" for event in raw))
        report_path = experiment / "fixture_report.json"
        report = {"summary": {"rounds": 6, "pending_shapes": 0, "files": [str(raw_path)]}, "candidates": candidates}
        integration.dump(report_path, report)
        def reject_raw(name, first, second=None):
            first_path = experiment / (name + "_shard0.jsonl")
            first_path.write_text("".join(json.dumps(event) + "\n" for event in first))
            files = [first_path]
            if second is not None:
                second_path = experiment / (name + "_shard1.jsonl")
                second_path.write_text("".join(json.dumps(event) + "\n" for event in second))
                files.append(second_path)
            try:
                integration.audit_repeat(report, files, baseline, metadata, .05)
            except ValueError:
                results[name] = "passed"
            else:
                raise AssertionError(name)
        reject_raw("duplicate_raw_round_rejected", raw + [raw[0]])
        missing_identity = copy.deepcopy(raw)
        missing_identity[0]["shape"].pop("gfx")
        reject_raw("implicit_shape_identity_rejected", missing_identity)
        duplicate_position = copy.deepcopy(raw)
        duplicate_position[6]["position"] = duplicate_position[0]["position"]
        reject_raw("duplicate_raw_position_rejected", duplicate_position)
        reject_raw("shape_split_across_files_rejected", raw[1:], [raw[0]])
        split_gpu = copy.deepcopy(raw)
        split_gpu[0]["gpu_logical_id"] = 1
        reject_raw("shape_split_across_gpus_rejected", split_gpu)
        args = argparse.Namespace(experiment_dir=experiment, report=report_path, repeat_jsonl=None,
                                  baseline=integration.DATA / "csv_baseline.csv", shapes=integration.DOC / "shapes_421.csv",
                                  err_ratio=.05, require_csv_win=False, expand_scope="nk", apply=False)
        integration.ROOT = public
        receipt, proposed, proposed_source = integration.prepare(args)
        assert receipt["selected_new_ids"] == [100000]
        assert receipt["identity"]["original_ids_verified"] == 48212
        enabled = {tuple(row["supported_shapes"][0]) for row in proposed["gfx950"] if row["kid"] == 100000}
        scopes = receipt["new_kernel_scopes"]
        assert any(row["gpu_validation_pending"] for row in scopes)
        assert any(row["M"] == 8192 and row["gpu_validated_stable_improvement"] for row in scopes)
        assert not any(row["M"] == 6144 and row["N"] == 6144 and row["K"] == 7168 for row in scopes)
        results["full_prepare_identity_and_scope_fixture"] = {"passed": True, "summary": receipt["summary"]}
        assert {str(rel): integration.digest(public / rel) for rel in (integration.COMMON_REL, integration.MANIFEST_REL)} == original_hashes
        results["check_only_did_not_change_fixture_public"] = "passed"
        # No GPU work occurs here: apply exercises only a temporary fake public
        # repo, and the fabricated image disappears with this TemporaryDirectory.
        apply_out = temporary / "fixture_apply_receipt"
        apply_out.mkdir()
        integration.apply(receipt, proposed, proposed_source, apply_out)
        assert receipt["status"] == "applied" and len(receipt["backups"]) == 2
        assert integration.digest(Path(receipt["images"][0]["destination"])) == integration.digest(image)
        results["apply_only_to_temporary_fixture_public"] = "passed"
        # Restore only the temporary fixture so a second transaction can prove
        # rollback after the source freeze and image copy have already occurred.
        for rel in (integration.MANIFEST_REL, integration.COMMON_REL):
            (public / rel).write_bytes(Path(receipt["backups"][str(rel)]).read_bytes())
        Path(receipt["images"][0]["destination"]).unlink()
        receipt["status"] = "checked"
        rollback_out = temporary / "fixture_rollback_receipt"
        rollback_out.mkdir()
        original_write = integration.atomic_write
        failed = False
        def one_failure(path, content):
            nonlocal failed
            if Path(path) == public / integration.MANIFEST_REL and not failed:
                failed = True
                raise RuntimeError("intentional CPU fixture transaction failure")
            original_write(path, content)
        integration.atomic_write = one_failure
        try:
            integration.apply(receipt, proposed, proposed_source, rollback_out)
        except RuntimeError:
            assert receipt["status"] == "rolled_back"
        else:
            raise AssertionError("Expected fixture rollback")
        finally:
            integration.atomic_write = original_write
        assert {str(rel): integration.digest(public / rel) for rel in (integration.COMMON_REL, integration.MANIFEST_REL)} == original_hashes
        assert not Path(receipt["images"][0]["destination"]).exists()
        results["transaction_failure_restored_both_files_and_removed_new_image"] = "passed"
        integration.ROOT = original_root
    assert {str(rel): integration.digest(original_root / rel) for rel in (integration.COMMON_REL, integration.MANIFEST_REL)} == original_hashes
    output = {"fixture_only": True, "gpu_used": False, "build_used": False,
              "real_public_files_unchanged": True, "checks": results}
    integration.dump(integration.LOCAL / "co_integration_cpu_fixture_report.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
