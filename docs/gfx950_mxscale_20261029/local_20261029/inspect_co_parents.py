#!/usr/bin/env python3
"""Read a finished profile on CPU and report existing CO parent identities.

Reports all measured legal 4wave panel/PAD/WideOutput parents and top measured
B/overlap CO parents with exact paired peer and actual B_AHEAD. It never picks
an experimental configuration or writes a selection plan automatically.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_progress import DATA, SHAPE_FIELDS, key, csv_rows
from prepare_co_experiment import ROOT, MANIFEST_REL, load_common, registry, b_metadata, pin_wrapper


def peer_key(entry):
    flags = entry.get("device_flags")
    if isinstance(flags, list):
        flags = tuple(flags)
    return (tuple(entry["traits"]["args"][:17]), tuple(entry["tile"]), tuple(entry["wave_layout"]),
            tuple(entry.get("launch_bounds", ())), flags, entry.get("grid_layout"), entry.get("pin_mask"),
            entry.get("pad_m"), entry.get("wide_output"), entry.get("fixed_k"), entry.get("max_k"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--shape", help="Limit to M,N,K")
    parser.add_argument("--topk", type=int, default=3, help="Existing B/overlap parent candidates per shape/pipeline")
    parser.add_argument("--err-ratio", type=float, default=.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    chosen = tuple(map(int, args.shape.split(","))) if args.shape else None
    if chosen is not None and len(chosen) != 3 or args.topk < 1:
        parser.error("Use M,N,K and positive topk")
    doc = json.loads((ROOT / MANIFEST_REL).read_text())
    entries = {entry["kid"]: entry for entry in doc["gfx950"]}
    common = load_common(ROOT, label="parent_inspection")
    reg = registry(common)
    paired = defaultdict(lambda: defaultdict(list))
    for kid, entry in entries.items():
        if entry["pipeline"] in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            paired[peer_key(entry)][entry["pipeline"]].append(kid)
    baseline = {key(row): float(row["us"]) for row in csv_rows(DATA / "csv_baseline.csv")}
    latest, invalid = {}, []
    for index, row in enumerate(csv_rows(args.profile), 2):
        if row.get("libtype") != "opus":
            continue
        shape = key(row)
        if chosen is not None and shape[2:5] != chosen:
            continue
        try:
            kid, us, err = int(row["kernelId"]), float(row["us"]), float(row["errRatio"])
        except (ValueError, TypeError):
            invalid.append({"line": index, "reason": "invalid ID/time/error"})
            continue
        if kid not in entries:
            continue  # JIT is useful in first-pass comparisons but is not an existing CO parent.
        if row.get("kernelName") != reg[kid].name:
            raise ValueError(f"Current CO profile name/ID mismatch at line {index}: {kid}")
        latest[(shape, kid)] = (row, index, us, err)
    by_shape_pipeline = defaultdict(list)
    for (shape, kid), (row, line, us, err) in latest.items():
        if not math.isfinite(us) or us <= 0 or not math.isfinite(err) or not 0 <= err <= args.err_ratio:
            invalid.append({"line": line, "kid": kid, "shape": dict(zip(SHAPE_FIELDS, shape)), "reason": "failed correctness/time"})
            continue
        entry = entries[kid]
        if not common.a8w8_mxscale_bpreshuffle_supports_shape(reg[kid], *shape[2:5]):
            raise ValueError(f"Measured CO parent is not legal for shape {shape}: {kid}")
        record = {"shape": dict(zip(SHAPE_FIELDS, shape)), "kid": kid, "kernelName": reg[kid].name,
                  "pipeline": entry["pipeline"], "us": us, "errRatio": err,
                  "csv_baseline_us": baseline[shape], "wins_csv": us < baseline[shape],
                  "source": {"profile": str(args.profile), "line": line},
                  "tile": entry["tile"], "wave_layout": entry["wave_layout"],
                  "traits": entry["traits"], "body_func": entry["body_func"], "body_args": entry["body_args"]}
        if entry["pipeline"] == "pin_agpr":
            _, wrapper = pin_wrapper(entry)
            record.update(panel=int(entry["traits"]["args"][0]), unroll=int(entry["traits"]["args"][1]),
                          pad_m=bool(entry["pad_m"]), wide_output=wrapper[4] == "true", group_m=int(wrapper[5]))
        else:
            record["actual_schedule"] = b_metadata(entry["traits"]["args"])
            other = "prefetch_b_vgpr_overlap" if entry["pipeline"] == "prefetch_b_vgpr" else "prefetch_b_vgpr"
            record["strict_base_trait_peers"] = paired[peer_key(entry)][other]
            record["publish_m"] = int(entry["traits"]["args"][17]) if entry["pipeline"] == "prefetch_b_vgpr_overlap" else -1
        by_shape_pipeline[(shape, entry["pipeline"])].append(record)
    output = []
    for (_, pipeline), records in sorted(by_shape_pipeline.items()):
        ordered = sorted(records, key=lambda record: (record["us"], record["kid"]))
        output.extend(ordered if pipeline == "pin_agpr" else ordered[:args.topk])
    report = {"profile": str(args.profile), "selection": "latest profile observation per shape/kid; correct candidates only",
              "parent_policy": "All existing 4wave parents, and topk existing B/overlap CO parents. No automatic choice or new CO plan.",
              "limitations": "JIT parents are omitted because this extension mechanism derives variants from the preserved 780 COs. Existing JIT stage2/3/4 results remain available in the original profile.",
              "parents": output, "invalid_measurements": invalid}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
