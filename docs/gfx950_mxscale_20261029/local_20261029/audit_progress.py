#!/usr/bin/env python3
"""CPU-only CSV-baseline progress accounting and exact tuner Top8 audit.

Examples:
  python3 audit_progress.py progress run.progress.jsonl --expected ../handoff_data/m_repeat_32_shapes.csv
  python3 audit_progress.py progress a.jsonl b.jsonl --selection best --output merged_summary.json
  python3 audit_progress.py top8 --output-dir top8_audit

The coverage audit extracts pure functions from the actual tuner with AST;
it imports the standard-library-only OPUS registry, never torch/aiter/GPU code.
Historical IDs are never treated as current identities.
"""

import argparse
import ast
import contextlib
import csv
import importlib
import io
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/gfx950_mxscale_20261029"
DATA = DOC / "handoff_data"
SHAPE_FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")


def key(row):
    return (str(row.get("gfx", "gfx950")), int(row.get("cu_num", 256)),
            int(row["M"]), int(row["N"]), int(row["K"]),
            str(row.get("w_scale_block", "128x128")))


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def m_class(m):
    if m < 128:
        return "M<128"
    if m < 1024:
        return "128<=M<1024"
    if m < 8192:
        return "1024<=M<8192"
    return "M>=8192"


def progress(args):
    baseline = {key(row): float(row["us"]) for row in csv_rows(args.baseline)}
    expected = {key(row) for row in csv_rows(args.expected)} if args.expected else None
    events = defaultdict(list)
    partial_lines = []
    for path in args.files:
        with Path(path).open() as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Live progress writers may be midway through their final line.
                    partial_lines.append(f"{path}:{lineno}")
                    continue
                shape = key(event["shape"])
                if expected is None or shape in expected:
                    events[shape].append(event)
    detail = []
    for shape, records in sorted(events.items()):
        valid = []
        for event in records:
            opus = event.get("best_opus")
            if opus and math.isfinite(float(opus["us"])) and float(opus["us"]) > 0:
                err = float(opus.get("errRatio", 1))
                if math.isfinite(err) and err <= args.err_ratio:
                    valid.append(event)
        event = (min(valid, key=lambda r: float(r["best_opus"]["us"]))
                 if args.selection == "best" and valid else records[-1])
        opus = event.get("best_opus")
        correct = event in valid
        us = float(opus["us"]) if correct else None
        csv_us = baseline.get(shape)
        wins = correct and csv_us is not None and us < csv_us
        row = dict(zip(SHAPE_FIELDS, shape))
        row.update(completed_utc=event.get("completed_utc"), events=len(records),
                   correct=correct, opus_us=us, csv_baseline_us=csv_us,
                   wins_csv=wins, speedup=csv_us / us if correct and csv_us else None,
                   kernelName=opus.get("kernelName") if opus else None,
                   kernelId=opus.get("kernelId") if opus else None,
                   best_libtype=event.get("best_libtype"))
        detail.append(row)
    losses = [row for row in detail if row["csv_baseline_us"] is not None and not row["wins_csv"]]
    nk = Counter(f'N{row["N"]}/K{row["K"]}' for row in losses)
    mc = Counter(m_class(row["M"]) for row in losses)
    summary = dict(selection=args.selection, correctness_limit=args.err_ratio,
                   files=[str(p) for p in args.files], completed_shapes=len(detail),
                   expected_shapes=len(expected) if expected is not None else None,
                   pending_shapes=len(expected - events.keys()) if expected is not None else None,
                   wins_csv=sum(row["wins_csv"] for row in detail),
                   losses_csv=len(losses),
                   missing_baselines=sum(row["csv_baseline_us"] is None for row in detail),
                   no_correct_opus=sum(not row["correct"] for row in detail),
                   losses_by_n_k=dict(nk.most_common()), losses_by_m_class=dict(mc),
                   ignored_partial_lines=partial_lines)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"summary": summary, "shapes": detail}, indent=2) + "\n")
    if args.losses_csv:
        write_csv(args.losses_csv, losses, SHAPE_FIELDS + ("opus_us", "csv_baseline_us", "speedup", "kernelName"))


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pure_tuner():
    sys.path.insert(0, str(ROOT))
    common = importlib.import_module("csrc.opus_gemm.opus_gemm_common")
    names = {"_opus_geometry", "_normalize_opus_geometry", "load_opus_geometry_file",
             "opus_candidates", "_opus_jit_full_name", "opus_kids_by_name"}
    tree = ast.parse((ROOT / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py").read_text())
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name in names], type_ignores=[])
    scope = dict(math=math, defaultdict=defaultdict, ast=ast, csv=csv, json=json, re=re)
    exec(compile(selected, "<pure tuner audit>", "exec"), scope)
    return common, scope


def name_geometry(name):
    root = "opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_"
    if name.startswith(root):
        match = re.match(r"(.+?)_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)", name[len(root):])
        if match:
            return (match[1], *map(int, match.groups()[1:]))
    co = re.match(r"opus_gemm_gfx950_(4wave_pin_agpr_co|b_vgpr_co|b_vgpr_overlap_co)_\d+x(\d+)x(\d+)x(\d+)_(\d+)x(\d+)_", name)
    if co:
        pipeline = {"4wave_pin_agpr_co": "pin_agpr", "b_vgpr_co": "prefetch_b_vgpr",
                    "b_vgpr_overlap_co": "prefetch_b_vgpr_overlap"}[co[1]]
        return (pipeline, *map(int, co.groups()[1:]))
    return None


def old_m_repeat_traits(name):
    """Normalize a proven old16-argument name only when all removed pins were false."""
    root = "opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_pingpong_lds_"
    if not name.startswith(root):
        return None
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_(.+)", name[len(root):])
    if not match:
        return None
    args = tuple(value.lower() for value in match[6].split("_"))
    if len(args) != 16 or args[9:12] != ("false", "false", "false"):
        return None
    if any(not re.fullmatch(r"-?\d+|true|false", value) for value in args):
        return None
    geometry = ("pingpong_lds", *map(int, match.groups()[:5]))
    return geometry, args[:9] + args[12:]


def top8(args):
    common, tuner = pure_tuner()
    reg = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    declared = common.KERNEL_REGISTRY["a8w8_blockscale_bpreshuffle"]["gfx950"][256]
    geom_fn = tuner["_opus_geometry"]
    all_geom = {kid: geom_fn(instance) for kid, instance in reg.items()}
    by_name = tuner["opus_kids_by_name"](reg)
    m_repeat_by_traits = defaultdict(set)
    for kid, instance in reg.items():
        entry = instance.bpreshuffle_entry
        if entry and entry["pipeline"] == "pingpong_lds":
            match = re.fullmatch(r"opus_gemm_bpreshuffle_pingpong_lds_traits<([^<>]+)>", entry["traits"])
            if match:
                traits = tuple(value.strip().lower() for value in match[1].split(","))
                m_repeat_by_traits[(all_geom[kid], traits)].add(kid)
    shapes = csv_rows(DOC / "shapes_421.csv")
    remote = {key(row): row for row in csv_rows(DOC / "all_shapes_421.csv")}
    baseline = {key(row): float(row["us"]) for row in csv_rows(DATA / "csv_baseline.csv")}
    known = []
    for row in remote.values():
        geo = (row["best_opus_pipeline"], *map(int, row["best_opus_tile"].split("x")),
               *map(int, row["best_opus_waves"].split("x")))
        known.append(("remote421", key(row), geo, row["opus_kernel"],
                      float(row["opus_us"]) < baseline[key(row)]))
    for filename in ("local_full185_progress.jsonl", "local_priority22_progress.jsonl"):
        with (DATA / filename).open() as stream:
            for line in stream:
                event = json.loads(line)
                opus = event.get("best_opus")
                if opus:
                    geo = name_geometry(opus["kernelName"])
                    if geo:
                        shape = key(event["shape"])
                        known.append((filename, shape, geo, opus["kernelName"], float(opus["us"]) < baseline[shape]))
    for filename in ("m_repeat_32_geometries.csv", "regressions_46_geometries.csv"):
        for mnk, geometries in tuner["load_opus_geometry_file"](DATA / filename).items():
            shape = ("gfx950", 256, *mnk, "128x128")
            for geo in geometries:
                known.append((filename, shape, geo, "", False))
    current_sources, read_warnings = set(), []

    def current_geometry(name):
        matches = by_name.get(name, set())
        return all_geom[min(matches)] if matches else name_geometry(name)

    def valid_current_result(result):
        try:
            us, err = float(result["us"]), float(result["errRatio"])
        except (TypeError, KeyError, ValueError):
            return False
        return math.isfinite(us) and us > 0 and math.isfinite(err) and 0 <= err <= args.err_ratio and result.get("correct", True)

    def add_current(source, shape, name, us):
        geo = current_geometry(name)
        if geo is None:
            read_warnings.append(f"Unrecognized current geometry: {source} {shape} {name}")
            return
        current_sources.add(source)
        known.append((source, shape, geo, name, float(us) < baseline[shape]))

    for path in args.current_evidence:
        source = f"current_evidence:{path.parent.name}/{path.name}"
        document = json.loads(path.read_text())
        for row in document["current76"]:
            us = row.get("opus_us")
            if row.get("observed") and row.get("kernelName") and us is not None and math.isfinite(float(us)) and float(us) > 0:
                # update_losses only writes a best candidate after correctness
                # checks and complete-repeat median replacement.
                add_current(source, key(row), row["kernelName"], us)
    for path in args.current_progress:
        source = f"current_progress:{path.parent.name}/{path.name}"
        latest = {}
        with path.open() as stream:
            for line, text in enumerate(stream, 1):
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    read_warnings.append(f"Ignored incomplete JSON: {path}:{line}")
                    continue
                latest[key(event["shape"])] = event
        for shape, event in latest.items():
            result = event.get("best_opus")
            if result and valid_current_result(result):
                add_current(source, shape, result["kernelName"], result["us"])
    repeat_groups = defaultdict(dict)
    for path in args.current_repeat:
        with path.open() as stream:
            for line, text in enumerate(stream, 1):
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    read_warnings.append(f"Ignored incomplete JSON: {path}:{line}")
                    continue
                if int(result.get("rotate_args", 0)) != 0:
                    continue
                repeat_groups[(key(result["shape"]), result["kernelName"])][int(result["round"])] = result
    repeat_best = {}
    for (shape, name), rounds in repeat_groups.items():
        if set(range(args.repeat_rounds)).issubset(rounds):
            records = [rounds[i] for i in range(args.repeat_rounds)]
            if all(valid_current_result(result) for result in records):
                us = statistics.median(float(result["us"]) for result in records)
                old = repeat_best.get(shape)
                if old is None or us < old[1]:
                    repeat_best[shape] = (name, us)
    for shape, (name, us) in repeat_best.items():
        add_current(f"current_repeat:{args.repeat_rounds}_round_median", shape, name, us)
    by_shape = defaultdict(list)
    for item in known:
        by_shape[item[1]].append(item)
    coverage = []
    shape_counts = []
    # Reuse an available registry and candidate list inside the original tuner
    # function; avoid stat'ing the same 780 CO images for each call.
    original_get = common.get_registry
    original_candidates = common.candidate_kids_for_shape
    common.get_registry = lambda family, gfx, cu_num: reg
    try:
        for index, row in enumerate(shapes, 1):
            shape = key(row)
            m, n, k = shape[2:5]
            candidates = [kid for kid, instance in reg.items()
                          if common.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k)]
            common.candidate_kids_for_shape = lambda *a, **kw: candidates
            with contextlib.redirect_stdout(io.StringIO()):
                picked = tuner["opus_candidates"](m, n, k, cu_num=256, top_geometries=8)
            selected = {kid for kid, instance in picked}
            legal_geometries = {all_geom[kid] for kid in candidates}
            selected_geometries = {all_geom[kid] for kid in selected}
            shape_counts.append({**dict(zip(SHAPE_FIELDS, shape)), "legal_candidates": len(candidates),
                                 "top8_candidates": len(selected), "legal_geometries": len(legal_geometries),
                                 "top8_geometries": len(selected_geometries)})
            for source, _, geo, name, won in by_shape[shape]:
                exact = by_name.get(name, set()) & set(candidates)
                normalized = old_m_repeat_traits(name)
                normalized_kids = m_repeat_by_traits.get(normalized, set()) & set(candidates) if normalized else set()
                coverage.append({**dict(zip(SHAPE_FIELDS, shape)), "source": source,
                                 "historical_wins_csv": won, "geometry": json.dumps(geo),
                                 "geometry_legal": geo in legal_geometries,
                                 "geometry_retained": geo in selected_geometries,
                                 "exact_name_matches_legal": len(exact),
                                 "exact_name_retained": bool(exact & selected),
                                 "normalized_m_repeat_matches_legal": len(normalized_kids),
                                 "normalized_m_repeat_retained": bool(normalized_kids & selected),
                                 "normalized_m_repeat_current_kids": json.dumps(sorted(normalized_kids)),
                                 "historical_kernelName": name})
            if index % 50 == 0:
                print(f"audited {index}/{len(shapes)} shapes", file=sys.stderr, flush=True)
    finally:
        common.get_registry = original_get
        common.candidate_kids_for_shape = original_candidates
    totals = {}
    for source in sorted({row["source"] for row in coverage}):
        rows = [row for row in coverage if row["source"] == source]
        wins = [row for row in rows if row["historical_wins_csv"]]
        totals[source] = dict(records=len(rows), geometries_legal=sum(r["geometry_legal"] for r in rows),
                              geometries_retained=sum(r["geometry_retained"] for r in rows),
                              historical_csv_winners=len(wins),
                              winner_geometries_retained=sum(r["geometry_retained"] for r in wins),
                              exact_names_legal=sum(bool(r["exact_name_matches_legal"]) for r in rows),
                              exact_names_retained=sum(r["exact_name_retained"] for r in rows),
                              normalized_m_repeat_traits_legal=sum(bool(r["normalized_m_repeat_matches_legal"]) for r in rows),
                              normalized_m_repeat_traits_retained=sum(r["normalized_m_repeat_retained"] for r in rows))
    current_totals = {}
    for source in sorted(current_sources):
        rows = [row for row in coverage if row["source"] == source]
        wins = [row for row in rows if row["historical_wins_csv"]]
        current_totals[source] = dict(records=len(rows), csv_winner_records=len(wins),
                                      geometries_legal=sum(row["geometry_legal"] for row in rows),
                                      geometries_retained=sum(row["geometry_retained"] for row in rows),
                                      winner_geometries_retained=sum(row["geometry_retained"] for row in wins),
                                      exact_names_legal=sum(bool(row["exact_name_matches_legal"]) for row in rows),
                                      exact_names_retained=sum(row["exact_name_retained"] for row in rows))
    dropped = [row for row in coverage if row["geometry_legal"] and not row["geometry_retained"]]
    unique_dropped = {(row["M"], row["N"], row["K"], row["geometry"]) for row in dropped}
    winner_dropped = {(row["M"], row["N"], row["K"], row["geometry"]) for row in dropped
                      if row["historical_wins_csv"]}
    current_winner_dropped = {(row["M"], row["N"], row["K"], row["geometry"]) for row in dropped
                              if row["historical_wins_csv"] and row["source"] in current_sources}
    fallback = []
    for m, n, k, encoded in sorted(unique_dropped):
        geo = json.loads(encoded)
        fallback.append(dict(zip(("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"), (m, n, k, *geo))))
    summary = {"shape_count": len(shapes), "declared_candidates": len(declared),
               "available_candidates": len(reg), "available_co": sum(i.bpreshuffle_co is not None for i in reg.values()),
               "top8_seeds": 0, "sources": totals, "fallback_geometry_rows": len(fallback),
               "fallback_shapes": len({(r["M"], r["N"], r["K"]) for r in fallback}),
               "winner_fallback_geometry_rows": len(winner_dropped),
               "winner_fallback_shapes": len({r[:3] for r in winner_dropped}),
               "current_sources": current_totals, "current_read_warnings": read_warnings,
               "current_input_files": {"evidence": [str(path) for path in args.current_evidence],
                                       "progress": [str(path) for path in args.current_progress],
                                       "repeat": [str(path) for path in args.current_repeat]},
               "current_winner_fallback_shapes": len({r[:3] for r in current_winner_dropped}),
               "current_winner_fallback_geometry_rows": len(current_winner_dropped),
               "limitations": ["Historical full-name mappings may fail after intentional interface/name changes; geometry coverage is audited independently.",
                               "Old16-argument m-repeat names additionally match current13 traits only when removed PinA/B/C were all false. This audit-only normalization does not change the tuner's automatic seed-name resolution.",
                               "This audits Top8 with no restored seeds. The actual tuner can additionally retain proven full-name seeds."]}
    out = Path(args.output_dir)
    write_csv(out / "coverage.csv", coverage, SHAPE_FIELDS + ("source", "historical_wins_csv", "geometry", "geometry_legal", "geometry_retained", "exact_name_matches_legal", "exact_name_retained", "normalized_m_repeat_matches_legal", "normalized_m_repeat_retained", "normalized_m_repeat_current_kids", "historical_kernelName"))
    write_csv(out / "shape_candidate_counts.csv", shape_counts, SHAPE_FIELDS + ("legal_candidates", "top8_candidates", "legal_geometries", "top8_geometries"))
    write_csv(out / "fallback_geometries.csv", fallback, ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"))
    fallback_mnk = {(r["M"], r["N"], r["K"]) for r in fallback}
    write_csv(out / "fallback_shapes.csv", [r for r in shapes if (int(r["M"]), int(r["N"]), int(r["K"])) in fallback_mnk], SHAPE_FIELDS)
    winner_fallback = []
    for m, n, k, encoded in sorted(winner_dropped):
        winner_fallback.append(dict(zip(("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"),
                                       (m, n, k, *json.loads(encoded)))))
    write_csv(out / "winner_fallback_geometries.csv", winner_fallback, ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"))
    winner_mnk = {r[:3] for r in winner_dropped}
    write_csv(out / "winner_fallback_shapes.csv", [r for r in shapes if (int(r["M"]), int(r["N"]), int(r["K"])) in winner_mnk], SHAPE_FIELDS)
    current_fallback = []
    for m, n, k, encoded in sorted(current_winner_dropped):
        current_fallback.append(dict(zip(("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"),
                                        (m, n, k, *json.loads(encoded)))))
    write_csv(out / "current_winner_fallback_geometries.csv", current_fallback, ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"))
    current_mnk = {r[:3] for r in current_winner_dropped}
    write_csv(out / "current_winner_fallback_shapes.csv", [r for r in shapes if (int(r["M"]), int(r["N"]), int(r["K"])) in current_mnk], SHAPE_FIELDS)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("progress")
    p.add_argument("files", nargs="+")
    p.add_argument("--baseline", default=DATA / "csv_baseline.csv")
    p.add_argument("--expected")
    p.add_argument("--selection", choices=("latest", "best"), default="latest")
    p.add_argument("--err-ratio", type=float, default=0.05)
    p.add_argument("--output")
    p.add_argument("--losses-csv")
    p.set_defaults(func=progress)
    p = sub.add_parser("top8")
    p.add_argument("--output-dir", default=Path(__file__).parent / "top8_audit")
    p.add_argument("--current-evidence", type=Path, nargs="*", default=[], help="Current update_losses evidence.json files")
    p.add_argument("--current-progress", type=Path, nargs="*", default=[], help="Current completed-shape tuning JSONL files")
    p.add_argument("--current-repeat", type=Path, nargs="*", default=[], help="Current balanced-repeat JSONL files; audit fastest all-correct median per shape")
    p.add_argument("--repeat-rounds", type=int, default=6)
    p.add_argument("--err-ratio", type=float, default=.05)
    p.set_defaults(func=top8)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
