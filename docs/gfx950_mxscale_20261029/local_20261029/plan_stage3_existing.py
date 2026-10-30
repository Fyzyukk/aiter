#!/usr/bin/env python3
"""CPU proposal for ~19 existing-configuration NT direction-finding shapes.

Default is a stdout preview. --write requires complete step1/2 observations,
complete step1 balanced repeats and no currently partial repeat confirmations.
No GPU, compilation, new CO selection or core-source mutation is performed.
"""

import argparse
import csv
import json
import shlex
from collections import Counter
from pathlib import Path

from audit_progress import DATA, DOC, SHAPE_FIELDS, csv_rows, key, m_class, pure_tuner, write_csv

LOCAL = Path(__file__).resolve().parent
GEO_FIELDS = ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=LOCAL / "step3_inputs/evidence.json")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step3_existing_round1")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    summary = evidence["summary"]
    current = {key(row): row for row in evidence["current76"]}
    history_pending = {key(row): row for row in csv_rows(args.evidence.parent / "historical_pending_losses_status.csv")}
    pending_repeats = summary.get("repeat_confirmation_pending_shapes", 0)
    step1 = {key(row) for row in csv_rows(DATA / "m_repeat_32_shapes.csv")}
    completed_step1_repeats = sum(current[shape].get("has_complete_repeat", False) for shape in step1)
    complete = summary["completed_unique_shapes"] == 76 and completed_step1_repeats == 32 and pending_repeats == 0
    if args.write and not complete:
        parser.error(f"Final inputs require completed step1/2 and repeats: current76={summary['completed_unique_shapes']}/76, step1_repeat={completed_step1_repeats}/32, partial_repeat_shapes={pending_repeats}; use stdout preview while work runs")
    shape_rows = {key(row): row for row in csv_rows(DOC / "shapes_421.csv")}
    baseline = {key(row): float(row["us"]) for row in csv_rows(DATA / "csv_baseline.csv")}
    losses = {shape: row for shape, row in current.items() if row["status"] == "observed_loss"}
    pool = {**history_pending, **losses}
    selected = {}

    def add(shape, reason):
        if shape not in shape_rows:
            raise ValueError(f"Shape outside the fixed421 list: {shape}")
        state = current.get(shape) or history_pending.get(shape) or {}
        old = selected.get(shape)
        if old:
            old["selection_reasons"].append(reason)
            return
        selected[shape] = {**dict(zip(SHAPE_FIELDS, shape)), "csv_baseline_us": baseline[shape],
                           "current_status": state.get("status", "not_measured_current_control"),
                           "current_us": state.get("opus_us"), "current_kind": state.get("kind"),
                           "historical_sources": state.get("historical_sources", []),
                           "selection_reasons": [reason], "m_class": m_class(shape[2])}

    missing_narrow_categories = []
    for n in (768, 1536, 2048):
        for label, predicate, preferred_m in (("small M loss", lambda m: m <= 6144, 6144),
                                              ("large M loss", lambda m: m >= 8192, 8192)):
            options = [shape for shape in pool if shape[3:5] == (n, 7168) and predicate(shape[2])]
            if not options:
                missing_narrow_categories.append({"N": n, "K": 7168, "category": label})
                continue
            chosen = min(options, key=lambda shape: (shape not in losses, abs(shape[2] - preferred_m), -shape[2]))
            add(chosen, f"N{n}/K7168 {label}; current observed loss preferred, then closest M{preferred_m} historical pending")
    for n, k in ((6144, 7168), (7168, 7168), (7168, 16384), (7168, 4096)):
        for m in (6144, 8192, 32768):
            add(("gfx950", 256, m, n, k, "128x128"), f"4wave direction control N{n}/K{k}, M{m}; compare panel/PAD/WideOutput before adding GM4/8")
    add(("gfx950", 256, 1664, 65536, 1536, "128x128"), "wide-N un-won target; compare legal m-repeat grouping plus B/overlap")
    common, tuner = pure_tuner()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    if len(registry) != 48212:
        raise ValueError("Existing-config first round expects the original48212 available identities")
    geometry_fn = tuner["_opus_geometry"]
    geometries, candidate_details = [], []
    for shape, row in selected.items():
        m, n, k = shape[2:5]
        picked = []
        for kid, instance in registry.items():
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k):
                continue
            geo = geometry_fn(instance)
            pipeline, bm, bn, bk, tm, tn = geo
            keep = (pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap")
                    and bm in (64, 96, 128) and bn in (64, 128) and tm == 1 and tn in (2, 4))
            keep = keep or pipeline == "pin_agpr"
            keep = keep or (pipeline == "pingpong_lds" and bm in (128, 192, 256))
            if keep:
                picked.append((kid, instance))
        counts = Counter(geometry_fn(instance) for kid, instance in picked)
        if not counts:
            raise ValueError(f"No requested existing candidates fit {shape}")
        co_4wave = [kid for kid, instance in picked if instance.bpreshuffle_co is not None and geometry_fn(instance)[0] == "pin_agpr"]
        pipeline_counts = Counter(geometry_fn(instance)[0] for kid, instance in picked)
        stage_counts = Counter()
        for kid, instance in picked:
            entry = instance.bpreshuffle_entry or instance.bpreshuffle_co
            if entry["pipeline"] not in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
                continue
            # Existing JIT stages are encoded in traits text; existing COs in args.
            raw = entry["traits"]["args"] if instance.bpreshuffle_co is not None else entry["traits"].split("<", 1)[1].rstrip(">").split(",")
            stage_counts[int(str(raw[2]).strip())] += 1
        row.update(opus_candidates=len(picked), geometry_count=len(counts),
                   by_pipeline=dict(pipeline_counts), b_overlap_stages=dict(stage_counts),
                   legal_4wave_co_count=len(co_4wave), legal_4wave_co_ids=co_4wave)
        for geo, count in sorted(counts.items()):
            base = dict(zip(GEO_FIELDS, (*shape[2:5], *geo)))
            geometries.append(base)
            candidate_details.append({**base, "candidate_count": count})
    report = {"mode": "final_existing_plan" if args.write else "provisional_stdout_preview",
              "current_evidence": str(args.evidence), "current76_completed": summary["completed_unique_shapes"],
              "step1_repeat_completed": completed_step1_repeats, "repeat_partial_shapes": pending_repeats,
              "inputs_complete": complete, "shape_count": len(selected),
              "opus_candidates": sum(row["opus_candidates"] for row in selected.values()),
              "geometry_rows": len(geometries), "missing_narrow_loss_categories": missing_narrow_categories,
              "baseline": str(DATA / "csv_baseline.csv"), "libtype": "opus", "top_geometries": 0,
              "pipeline_policy": "B/overlap BM64/96/128, BN64/128, WN2/4, every legal BK and parameter; all legal existing4wave CO; m-repeat BM128/192/256; no interleave",
              "parameter_policy": "Geometry Top0 retains every legal existing parameter, including B/overlap stage2/3/4. No newCO or Cartesian extension.",
              "shapes": list(selected.values())}
    if args.write:
        out = args.output_dir.resolve()
        if not out.is_relative_to(LOCAL):
            parser.error("Write plans only under local_20261029")
        if out.exists():
            parser.error(f"Output exists; choose a fresh round directory: {out}")
        out.mkdir(parents=True)
        write_csv(out / "shapes.csv", selected.values(), SHAPE_FIELDS)
        write_csv(out / "geometries.csv", geometries, GEO_FIELDS)
        write_csv(out / "geometry_candidate_counts.csv", candidate_details, GEO_FIELDS + ("candidate_count",))
        (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
        script = ("#!/usr/bin/env bash\nset -euo pipefail\nsource " + shlex.quote(str(LOCAL / "env.sh")) + "\n"
                  'export AITER_REBUILD=0\n'
                  f"export OPUS_TUNE_PROGRESS_FILE={shlex.quote(str(out / 'progress.jsonl'))}\n"
                  f"export OPUS_TUNE_CANDIDATE_PROGRESS_DIR={shlex.quote(str(out / 'candidates'))}\n"
                  '"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py '
                  '--mxscale_table --libtype opus --mp 4 --shape_grouped --opus_top_geometries 0 '
                  '--opus_geometry_file ' + shlex.quote(str(out / "geometries.csv"))
                  + ' -i ' + shlex.quote(str(out / "shapes.csv"))
                  + ' -o ' + shlex.quote(str(out / "tuned.csv"))
                  + ' -o2 ' + shlex.quote(str(out / "profile.csv")) + "\n")
        (out / "run.sh").write_text(script)
        (out / "run.sh").chmod(0o755)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
