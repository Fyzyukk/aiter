#!/usr/bin/env python3
"""Prepare the focused NT follow-up on CPU, without compilation or GPU work.

The fixed 56-shape selection covers all M>=2048 in the three priority N/K
families and four controls each for N768/N2048. Geometries come only from
five CSV-confirmed COs, current best original kernels, and mapped historical
best kernels. A proposal registry is declared-only; regenerate the candidate
receipt against the final public manifest before compiling/running.
"""

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import shlex
from collections import Counter, defaultdict
from pathlib import Path

import prepare_opus_repeat as metadata_helper
from prepare_co_experiment import ROOT, LOCAL, MANIFEST_REL, JIT_BASE_LINE, load_common, registry

DOC = ROOT / "docs/gfx950_mxscale_20261029"
SHAPE_FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
GEO_FIELDS = ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN")
CONFIRMED_CO_IDS = (100003, 100024, 100025, 100030, 100031)
REPORTS = ("step1_report/report.json", "step2_report/report.json",
           "step3_existing_round1/repeat_report/report.json", "co_round1/repeat_report/report.json",
           "co_round2/repeat_report/report.json")


def read_csv(path):
    return metadata_helper.csv_rows(path)


def mnk(row):
    return tuple(int(row[field]) for field in ("M", "N", "K"))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for field, value in row.items() if field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST_REL)
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "focused_expansion")
    parser.add_argument("--receipt-only", action="store_true",
                        help="Preserve shapes/geometries and refresh receipts against the final manifest")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    public_source = (ROOT / "csrc/opus_gemm/opus_gemm_common.py").read_text()
    common = load_common(ROOT, manifest=manifest, freeze=JIT_BASE_LINE in public_source, label="focused_expansion")
    declared = registry(common)
    using_public_manifest = manifest == (ROOT / MANIFEST_REL).resolve()
    runtime_available = set(common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)) if using_public_manifest else None
    missing = set(CONFIRMED_CO_IDS) - declared.keys()
    if missing:
        parser.error(f"Manifest lacks CSV-confirmed COs {sorted(missing)}; pass the integration proposal or final public manifest")
    # Proposal images are not stored relative to the proposal file. CPU name
    # mapping must use declared identities; the final tuner checks real images.
    common.get_registry = lambda family, gfx, cu_num: declared
    metadata_helper.pure_common = lambda: common
    metadata, _ = metadata_helper.registry_metadata()
    indexes = metadata_helper.name_indexes(metadata)
    full_rows = {mnk(row): row for row in read_csv(DOC / "shapes_421.csv")}
    remote = {mnk(row): row for row in read_csv(DOC / "all_shapes_421.csv")}
    baseline = {mnk(row): float(row["us"]) for row in read_csv(DOC / "handoff_data/csv_baseline.csv")
                if row.get("gfx") == "gfx950" and int(row["cu_num"]) == 256 and row["w_scale_block"] == "128x128"}
    evidence_dir = LOCAL / "step3_inputs"
    observed = {mnk(row) for row in read_csv(evidence_dir / "observed_losses_shapes.csv")}
    pending = {mnk(row) for row in read_csv(evidence_dir / "historical_pending_losses_shapes.csv")}
    if len(observed) != 16 or len(pending) != 69 or observed & pending:
        raise ValueError("Expected the frozen initial16 observed +69 historical-pending union")
    selected = {shape for shape in full_rows if shape[0] >= 2048 and shape[1:]
                in ((1536, 7168), (7168, 7168), (6144, 7168))}
    selected.update((m, 768, 7168) for m in (2048, 8192, 12288, 32768))
    selected.update((m, 2048, 7168) for m in (1984, 2048, 12288, 32768))
    if len(selected) != 56 or not selected <= full_rows.keys():
        raise ValueError("Fixed56 selection is outside the complete421 namespace")
    observed_by_candidate, measured_shapes = {}, set()
    previous_current_winners = defaultdict(set)
    report_hashes = {}
    for relative in REPORTS:
        path = LOCAL / relative
        report = json.loads(path.read_text())
        report_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in report["shapes"]:
            if row.get("repeat_complete"):
                shape = mnk(row)
                measured_shapes.add(shape)
                winner = row.get("best_repeat_kid")
                if winner in declared and shape in selected and common.a8w8_mxscale_bpreshuffle_supports_shape(declared[winner], *shape):
                    if row["best_repeat_name"] != declared[winner].name:
                        raise ValueError(f"Previous current winner identity differs for kid{winner}")
                    previous_current_winners[shape].add(winner)
        for row in report["candidates"]:
            if not row["all_rounds_correct"] or not row["timing_matches_tuner"]:
                continue
            shape, kid = mnk(row), int(row["kid"])
            if shape not in selected or kid not in declared:
                continue
            if row["kernelName"] != declared[kid].name:
                raise ValueError(f"Completed candidate identity differs for kid{kid}")
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(declared[kid], *shape):
                continue  # A retired experimental loss may not be published.
            observed_by_candidate[(shape, kid)] = {**row, "source_report": str(path)}
    geo_rows, details, known_rows, candidate_rows, shape_details = [], [], [], [], []
    unique_union = set()
    for shape in sorted(selected, key=lambda value: (value[1], value[2], value[0])):
        current = [row for (candidate_shape, _), row in observed_by_candidate.items() if candidate_shape == shape]
        original = [row for row in current if row["kid"] < 100000]
        original_best = min(original, key=lambda row: row["median_us"]) if original else None
        current_best = min(current, key=lambda row: row["median_us"]) if current else None
        shape_full = ("gfx950", 256, *shape, "128x128")
        mapping = metadata_helper.map_remote(remote[shape]["opus_kernel"], metadata, indexes, shape_full)
        if not mapping["mapped"]:
            raise ValueError(f"Historical known winner is not fully mapped/legal for {shape}: {mapping}")
        known = set(mapping["legal_current_kids"]) | previous_current_winners[shape]
        if original_best:
            known.add(original_best["kid"])
        if current_best:
            known.add(current_best["kid"])
        new = {kid for kid in CONFIRMED_CO_IDS
               if common.a8w8_mxscale_bpreshuffle_supports_shape(declared[kid], *shape)}
        if not new:
            raise ValueError(f"No CSV-confirmed CO with published hostscope fits focused shape {shape}")
        for kid in new:
            instance = declared[kid]
            raw_entry = copy.deepcopy(instance.bpreshuffle_co)
            raw_entry.pop("supported_shapes", None)
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(dataclasses.replace(instance, bpreshuffle_co=raw_entry), *shape):
                raise ValueError(f"Published scope bypasses an actual device guard for kid{kid}/{shape}")
        geometries = {tuple(metadata[kid]["geometry"]) for kid in new | known}
        counts = Counter()
        selected_kids = set()
        for kid, instance in declared.items():
            if tuple(metadata[kid]["geometry"]) in geometries and common.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape):
                selected_kids.add(kid)
                counts[tuple(metadata[kid]["geometry"])] += 1
        if (new | known) - selected_kids:
            raise ValueError("Requested Top0 geometry omitted a new/known candidate")
        if runtime_available is not None and selected_kids - runtime_available:
            raise ValueError(f"Final public candidate images are unavailable: {sorted(selected_kids - runtime_available)}")
        unique_union.update(selected_kids)
        for geo in sorted(geometries):
            row = dict(zip(GEO_FIELDS, (*shape, *geo)))
            geo_rows.append(row)
            details.append({**row, "candidate_count": counts[geo],
                "new_csv_co_geometry": any(tuple(metadata[kid]["geometry"]) == geo for kid in new),
                "known_winner_geometry": any(tuple(metadata[kid]["geometry"]) == geo for kid in known)})
        for kid in sorted(new | known):
            role = "csv_confirmed_co" if kid in new else "known_winner"
            if kid in new and kid in known:
                role += "+known_winner"
            known_rows.append({"M": shape[0], "N": shape[1], "K": shape[2], "kid": kid,
                "kernelName": metadata[kid]["kernelName"], "kind": metadata[kid]["kind"],
                "role": role, "geometry": metadata[kid]["geometry"], "traits": metadata[kid]["traits"]})
        for kid in sorted(selected_kids):
            candidate_rows.append({"M": shape[0], "N": shape[1], "K": shape[2], "kid": kid,
                                   "kernelName": metadata[kid]["kernelName"], "kind": metadata[kid]["kind"]})
        shape_details.append({**full_rows[shape], "selection_class": "initial_observed_loss" if shape in observed
            else "historical_pending_loss" if shape in pending else "other_same_family_control",
            "previous_complete_gpu_measurement": shape in measured_shapes,
            "new_csv_co_kids": sorted(new), "known_winner_kids": sorted(known),
            "requested_geometry_count": len(geometries), "candidate_count": len(selected_kids),
            "best_current_kid": current_best["kid"] if current_best else None,
            "best_current_us": current_best["median_us"] if current_best else None,
            "current_wins_csv": bool(current_best and current_best["median_us"] < baseline[shape]),
            "csv_baseline_us": baseline[shape]})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shape_csv = [full_rows[shape] for shape in sorted(selected, key=lambda value: (value[1], value[2], value[0]))]
    if args.receipt_only:
        if {mnk(row) for row in read_csv(out / "shapes.csv")} != selected:
            raise ValueError("Receipt refresh would change the fixed shape selection")
        old_geos = {tuple(row[field] for field in GEO_FIELDS) for row in read_csv(out / "geometries.csv")}
        fresh_geos = {tuple(str(row[field]) for field in GEO_FIELDS) for row in geo_rows}
        if old_geos != fresh_geos:
            raise ValueError("Final manifest changes the requested new/known geometry set; inspect instead of silently replacing fixed inputs")
    else:
        write_csv(out / "shapes.csv", shape_csv, SHAPE_FIELDS)
        write_csv(out / "geometries.csv", geo_rows, GEO_FIELDS)
    write_csv(out / "geometry_candidate_counts.csv", details, (*GEO_FIELDS, "candidate_count", "new_csv_co_geometry", "known_winner_geometry"))
    write_csv(out / "known_winners.csv", known_rows, ("M", "N", "K", "kid", "kernelName", "kind", "role", "geometry", "traits"))
    write_csv(out / "candidate_identities.csv", candidate_rows, ("M", "N", "K", "kid", "kernelName", "kind"))
    write_csv(out / "shape_selection.csv", shape_details, tuple(shape_details[0]))
    summary = {"shapes": len(selected), "initial_observed_losses_included": len(selected & observed),
        "historical_pending_losses_included": len(selected & pending), "original85_included": len(selected & (observed | pending)),
        "other_same_family_shapes": len(selected - observed - pending),
        "previous_complete_gpu_shapes": len(selected & measured_shapes), "new_current_gpu_shapes": len(selected - measured_shapes),
        "requested_geometry_rows": len(geo_rows), "geometry_counts_per_shape": dict(Counter(row["requested_geometry_count"] for row in shape_details)),
        "candidate_shape_groups": len(candidate_rows), "unique_candidate_union": len(unique_union),
        "forced_new_and_known_candidate_shape_groups": len(known_rows), "known_winner_identity_failures": 0,
        "by_NK": dict(Counter(f"N{shape[1]}/K{shape[2]}" for shape in selected)),
        "M_ge8192": sum(shape[0] >= 8192 for shape in selected)}
    receipt = {"summary": summary, "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "registry_mode": "final_public_available_images_verified" if using_public_manifest
            else "CPU_declared_proposal; refresh against final public manifest before actual compile/run",
        "csv_confirmed_co_ids": list(CONFIRMED_CO_IDS), "report_hashes": report_hashes,
        "policy": "NT; Top0 specified new/known geometries; screen21iters; Top5 plus every known winner101iters six rounds; GPUs0-3",
        "gpu_used": False, "compiled": False, "shapes": shape_details}
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    launcher = ("#!/usr/bin/env bash\nset -euo pipefail\nsource " + shlex.quote(str(LOCAL / "env.sh")) + "\n"
        + "cd " + shlex.quote(str(ROOT)) + "\nexport AITER_REBUILD=0\n"
        + 'export HIP_VISIBLE_DEVICES=0,1,2,3\nexport ROCR_VISIBLE_DEVICES=0,1,2,3\n'
        + "export OPUS_TUNE_PROGRESS_FILE=" + shlex.quote(str(out / "progress.jsonl")) + "\n"
        + "export OPUS_TUNE_CANDIDATE_PROGRESS_DIR=" + shlex.quote(str(out / "candidates")) + "\n"
        + '"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py '
        + '--mxscale_table --libtype opus --mp 4 --shape_grouped --opus_top_geometries 0 --iters 21 --timeout 7200 '
        + '--opus_geometry_file ' + shlex.quote(str(out / "geometries.csv")) + ' -i ' + shlex.quote(str(out / "shapes.csv"))
        + ' -o ' + shlex.quote(str(out / "tuned.csv")) + ' -o2 ' + shlex.quote(str(out / "profile.csv")) + "\n")
    (out / "run.sh").write_text(launcher)
    (out / "run.sh").chmod(0o755)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
