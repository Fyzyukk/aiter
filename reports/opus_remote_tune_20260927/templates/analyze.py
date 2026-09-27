#!/usr/bin/env python3
"""Audit one fresh batch and compare a merged OPUS pool with its old16 controls.

Example: python reports/opus_remote_run/analyze.py --batch full295_r3
Default new IDs: 21000,21310,21311,21220,21221. Reports go to BATCH/analysis.
Only the selected batch supplies timing values. In finalists mode, a completed
full sweep from this generated harness supplies external identities only.
Uses the Python standard library and does not import or execute GPU code.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
def find_repo_root():
    for directory in HERE.parents:
        if (directory / "csrc/opus_gemm/opus_gemm_common.py").is_file():
            return directory
    raise FileNotFoundError("Cannot find repository root containing csrc/opus_gemm/opus_gemm_common.py")


ROOT = find_repo_root()
OLD_REGISTERED = {9000, 9010, 9011, 9012, 9020}
OLD_RUNTIME_LIBRARIES = {
    13163: "long_epilogue_sync", 20000: "long_runtime", 20010: "short_runtime", 20011: "short_runtime",
    20020: "n224_runtime", 20100: "long_runtime_grid", 20124: "n224_runtime_loop",
    20125: "short_runtime_unified", 20126: "short_runtime_unified", 20128: "long_runtime_fused",
    20131: "short_runtime_group4_cache2",
}
OLD_RUNTIME = set(OLD_RUNTIME_LIBRARIES)
DEFAULT_NEW_IDS = (21000, 21310, 21311, 21220, 21221)
NEW_LIBRARIES = {"main", "small", "narrow"}
OLD16 = OLD_REGISTERED | OLD_RUNTIME
PRESERVED = {9000, 9020}
EXTERNAL = {"ck", "cktile", "asm"}
OPUS = {"opus", "experiment"}
IDENTITY_FIELDS = ("M", "N", "K", "name", "lib", "kid", "splitK", "kernelName", "library")
CSV_KINDS = {"candidates", "check", "reject", "raw", "summary", "comparison", "selection"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(path, directory, description):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    require(path.is_relative_to(directory), f"{description} must belong to {directory}: {path}")
    return path


class Audit:
    def __init__(self):
        self.files = {}

    def verify(self, path, expected=None):
        path = str(Path(path).resolve())
        if path not in self.files:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            self.files[path] = digest.hexdigest()
        actual = self.files[path]
        require(expected is None or actual == expected, f"Artifact changed: {path}")
        return actual

    def json(self, path, expected=None):
        self.verify(path, expected)
        return json.loads(Path(path).read_text())

    def csv(self, path, expected=None):
        self.verify(path, expected)
        with Path(path).open(newline="") as stream:
            return list(csv.DictReader(stream))


def shape(row):
    return tuple(int(row[key]) for key in ("M", "N", "K"))


def identity(row):
    return row["lib"], int(row["kid"]), int(row["splitK"]), row["kernelName"], row.get("library", "")


def candidate(row):
    return row.get("library") or "registered_opus", int(row["kid"])


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[shape(row)].append(row)
    return result


def us(row):
    return float(row["median_us"])


def best(rows, predicate):
    return min((row for row in rows if predicate(row)), key=lambda row: (us(row), row["name"]), default=None)


def delta(value, baseline):
    return (value / baseline - 1) * 100


def ratio_stats(values):
    values = list(values)
    if not values:
        return dict(shapes=0, geomean_delta_pct=None, max_delta_pct=None, min_delta_pct=None,
                    slower_shapes=0, faster_shapes=0, tied_shapes=0)
    return dict(shapes=len(values), geomean_delta_pct=delta(math.exp(statistics.fmean(map(math.log, values))), 1),
                max_delta_pct=delta(max(values), 1), min_delta_pct=delta(min(values), 1),
                slower_shapes=sum(v > 1 for v in values), faster_shapes=sum(v < 1 for v in values),
                tied_shapes=sum(v == 1 for v in values))


def supports(variant, target):
    m, n, k = target
    return (not variant.get("fixed_k") or k == int(variant["fixed_k"])) and all(
        value % int(variant.get(field, default)) == 0
        for value, field, default in ((m, "m_multiple", 64), (n, "n_multiple", 256), (k, "k_multiple", 128))
    ) and k >= int(variant.get("min_k") or 0) and k <= int(variant.get("max_k") or k)


def write_csv(path, rows, fallback=IDENTITY_FIELDS):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prefixed(prefix, row):
    return {f"{prefix}_{key}": row.get(key, "") for key in ("name", "lib", "library", "kid")} | {
        f"{prefix}_us": us(row)}


def external_finalists(audit, worker, run, local, full_batch):
    """Verify local full-sweep finalist identities without reusing its timings."""
    source = run.get("full_input")
    require(source and source.get("path") == worker.get("full_manifest"), "Missing/mismatched full input")
    source_path = within(source["path"], full_batch, "Full-sweep worker manifest")
    require(source_path.parent == full_batch, "Full-sweep worker must belong to the declared full batch")
    prior_launch = audit.json(full_batch / "launch.json")
    require(prior_launch.get("status") == "passed" and prior_launch.get("external_mode") == "full",
            "Finalists require a completed full-mode batch in this generated harness")
    prior_workers = [item for item in prior_launch["workers"]
                     if Path(item["run_manifest"]).resolve() == source_path]
    require(len(prior_workers) == 1 and prior_workers[0].get("status") == "passed"
            and prior_workers[0].get("exit_code") == 0 and prior_workers[0]["run_sha256"] == source["sha256"],
            "Full-sweep worker is missing, incomplete, or changed")
    prior = audit.json(source_path, source["sha256"])
    require(prior.get("status") == "passed" and prior.get("external_mode") == "full",
            "Finalists require a passed full sweep")
    require(prior["gpu_uuid"] == run["gpu_uuid"] and prior["gpu"] == run["gpu"], "Finalist GPU changed")
    require(prior_workers[0]["gpu_uuid"] == prior["gpu_uuid"] and prior_workers[0]["gpu"] == prior["gpu"],
            "Full-sweep worker GPU identity differs from its launcher")
    require(local <= {tuple(s) for s in prior["shapes"]}, "Full sweep does not cover finalist shapes")
    # The current orchestration source necessarily differs. The archived
    # numerical/timing helpers, external source tree, and external binaries do not.
    prior_refs = {k: v for k, v in prior["reference_sha256"].items() if k != "measurement_harness"}
    current_refs = {k: v for k, v in run["reference_sha256"].items() if k != "measurement_harness"}
    require(prior_refs == current_refs, "External finalist reference sources/binaries changed")
    rows = {kind: grouped(audit.csv(within(prior["outputs"][kind], full_batch, "Full-sweep output"),
                                     prior["output_sha256"][kind]))
            for kind in ("summary", "candidates")}
    result = {}
    for target in local:
        valid = [r for r in rows["summary"][target] if r["lib"] in EXTERNAL]
        require(valid and all(int(r["samples"]) == prior["rounds"] and math.isfinite(us(r)) and us(r) > 0
                              for r in valid), f"No valid full-sweep external reference: {target}")
        threshold = min(map(us, valid)) * 1.05
        result[target] = ({identity(r) for r in rows["candidates"][target] if r["lib"] in EXTERNAL},
                          {identity(r) for r in valid if us(r) <= threshold})
    return result


def audit_shape(target, rows, rounds, meta, common, finalists):
    candidates, checks, raw, summaries = (rows[kind][target] for kind in ("candidates", "check", "raw", "summary"))
    candidate_ids = {identity(r) for r in candidates}
    canonical = {identity(r): r for r in candidates}
    require(len(candidate_ids) == len(candidates) == len({r["name"] for r in candidates}),
            f"Duplicate candidate identity/name: {target}")
    require(all(r["lib"] in EXTERNAL | OPUS and r["selected"] in {"True", "False"} for r in candidates),
            f"Invalid backend/selection flag: {target}")
    active = [r for r in candidates if r["selected"] == "True"]
    active_ids = {identity(r) for r in active}
    require(all(r["selected"] == "True" for r in candidates if r["lib"] in OPUS),
            f"An OPUS candidate was silently skipped: {target}")
    expected_runtime = {key for key, variant in meta.items() if key[0] != "registered_opus" and supports(variant, target)}
    require({candidate(r) for r in candidates if r["lib"] == "experiment"} == expected_runtime,
            f"Runtime candidate coverage differs: {target}")
    expected_registered = {kid for kid in OLD_REGISTERED if common.a8w8_mxscale_bpreshuffle_supports_shape(
        common.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid], *target)}
    require({int(r["kid"]) for r in candidates if r["lib"] == "opus"} == expected_registered,
            f"Registered old16 controls are incomplete: {target}")
    external_ids = {identity(r) for r in candidates if r["lib"] in EXTERNAL}
    active_external = {identity(r) for r in active if r["lib"] in EXTERNAL}
    if finalists is None:
        require(active_ids == candidate_ids, f"Full sweep skipped candidates: {target}")
    else:
        registry, expected_finalists = finalists[target]
        require(external_ids == registry and active_external == expected_finalists,
                f"External registry/finalist membership differs: {target}")
    require(active_external, f"No active external reference: {target}")
    preflight = [r for r in checks if r["phase"] == "preflight"]
    require(len(preflight) == len(active) and {identity(r) for r in preflight} == active_ids,
            f"Incomplete/duplicate numerical preflight: {target}")
    require(all(r["status"] in {"passed", "failed"} and (
        (r["phase"] == "preflight" and int(r["round"]) == -1) or
        (r["phase"] == "timed" and int(r["round"]) in range(rounds))) for r in checks),
        f"Invalid check phase/status/round: {target}")
    require(all(identity(r) in active_ids for r in checks + raw + summaries), f"Foreign/inactive record: {target}")
    require(all(r["name"] == canonical[identity(r)]["name"] for r in checks + raw + summaries),
            f"Candidate name changed between records: {target}")
    require(all(float(r["error"]) == 0 and r["guards"] == "True" for r in checks if r["status"] == "passed"),
            f"Numerical pass violates original FP32/guard contract: {target}")
    timed = {(identity(r), int(r["round"])): r for r in checks if r["phase"] == "timed"}
    require(len(timed) == len(raw) == len(checks) - len(preflight), f"Duplicate/missing timed check: {target}")
    samples, raw_by_id = defaultdict(dict), defaultdict(list)
    for row in raw:
        key, turn = identity(row), int(row["round"])
        require(turn in range(rounds) and turn not in samples[key], f"Duplicate/invalid timing round: {target}")
        check = timed.get((key, turn))
        require(check and all(check[field] == row[field] for field in ("status", "error", "reason")),
                f"Raw/check record mismatch: {target}")
        value = float(row["us"]) if row["us"] else None
        if row["status"] == "passed":
            require(value is not None and math.isfinite(value) and value > 0, f"Invalid passed timing: {target}")
        samples[key][turn] = value
        raw_by_id[key].append(row)
    for turn in range(rounds):
        orders = [int(r["order"]) for r in raw if int(r["round"]) == turn]
        require(sorted(orders) == list(range(len(orders))), f"Timing order is incomplete/duplicated: {target}")
    failed = {identity(r) for r in checks if r["status"] != "passed"}
    for row in preflight:
        key = identity(row)
        records = sorted(raw_by_id[key], key=lambda r: int(r["round"]))
        if row["status"] == "failed":
            require(not records, f"Preflight-rejected candidate was timed: {target}")
        else:
            require(records and [int(r["round"]) for r in records] == list(range(len(records))),
                    f"Missing timing round without explicit failure: {target}")
            bad = [r for r in records if r["status"] == "failed"]
            require((not bad and len(records) == rounds) or (len(bad) == 1 and bad[0] is records[-1]),
                    f"Incomplete rounds or continued timing after failure: {target}")
    summary_ids = {identity(r) for r in summaries}
    require(len(summary_ids) == len(summaries) and summary_ids == active_ids - failed,
            f"Summary must include every valid candidate and exclude every failure: {target}")
    for row in summaries:
        values = samples[identity(row)]
        require(set(values) == set(range(rounds)) and int(row["samples"]) == rounds,
                f"Summary has incomplete rounds: {target}")
        require(us(row) == statistics.median(values.values()) and float(row["min_us"]) == min(values.values())
                and float(row["max_us"]) == max(values.values()), f"Summary differs from raw timing: {target}")
    # These harness files choose over the whole measurement union. Audit them,
    # then recompute the new pool independently below; never reuse this winner.
    union = best(summaries, lambda r: r["lib"] in OPUS)
    external = best(summaries, lambda r: r["lib"] in EXTERNAL)
    old5 = best(summaries, lambda r: r["lib"] == "opus")
    require(union and external and old5, f"Missing valid control/reference candidate: {target}")
    winner = union if us(union) < us(external) else external
    selections, comparisons = rows["selection"][target], rows["comparison"][target]
    require(len(selections) == len(comparisons) == 1, f"Multiple/missing harness choices: {target}")
    require(int(selections[0]["rounds"]) == rounds and int(comparisons[0]["rounds"]) == rounds,
            f"Recorded choice/comparison rounds differ: {target}")
    require(identity(selections[0]) == identity(winner) and us(selections[0]) == us(winner),
            f"Recorded union selection differs: {target}")
    comp = comparisons[0]
    win_rounds = sum(samples[identity(union)][turn] < samples[identity(external)][turn] for turn in range(rounds))
    require(comp["opus"] == union["name"] and comp["reference"] == external["name"]
            and float(comp["opus_us"]) == us(union) and float(comp["reference_us"]) == us(external)
            and float(comp["old_opus_us"]) == us(old5) and int(comp["opus_faster_rounds"]) == win_rounds,
            f"Recorded union comparison differs: {target}")
    return candidates, summaries, checks, failed, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--new-ids", nargs="+", default=[",".join(map(str, DEFAULT_NEW_IDS))],
                        help="new runtime IDs, separated by commas or spaces (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, help="fresh directory; default: BATCH/analysis")
    args = parser.parse_args()
    new_list = [int(value) for item in args.new_ids for value in item.split(",") if value]
    new_ids = set(new_list)
    require(new_ids and len(new_ids) == len(new_list) and not new_ids & OLD16, "--new-ids must be unique new IDs")
    batch = args.batch
    if not batch.is_absolute() and not batch.exists():
        batch = HERE / batch
    batch = batch.resolve()
    require(batch.parent == HERE, f"--batch must be a batch directory in this generated harness: {HERE}")
    output = (args.output_dir or batch / "analysis").resolve()
    require(not output.exists(), f"Use a fresh output directory: {output}")
    audit = Audit()
    audit.verify(__file__)
    launch_path = batch / "launch.json"
    launch = audit.json(launch_path)
    expected_rows = audit.csv(launch["shapes_file"], launch["shapes_sha256"])
    expected_shapes = {shape(r) for r in expected_rows}
    require(expected_shapes and len(expected_shapes) == len(expected_rows), "Expected unique nonempty target shapes")
    if len(expected_shapes) == 295:
        require(expected_shapes == {shape(r) for r in audit.csv(HERE / "shapes295.csv")},
                "A full295 batch must cover the canonical 295 shapes")
    require(launch.get("status") == "passed" and launch.get("shape_count") == len(expected_shapes)
            and launch.get("completed_shapes") == len(expected_shapes), "Batch is not completely passed")
    require(len(launch["shapes"]) == len(expected_shapes) and {tuple(s) for s in launch["shapes"]} == expected_shapes,
            "Launch shape membership differs")
    require(launch["rounds"] in (3, 5) and launch["external_mode"] in {"full", "finalists"}, "Invalid rounds/mode")
    full_batch = None
    if launch["external_mode"] == "finalists":
        require(launch.get("full_batch"), "Finalists mode requires a full-batch manifest")
        full_batch = within(launch["full_batch"], HERE, "Full batch")
        require(full_batch.parent == HERE and full_batch != batch,
                "Full batch must be a separate completed batch from this generated harness")
    else:
        require(launch.get("full_batch") is None, "Full mode must not depend on another batch")
    audit.verify(HERE / "benchmark.py", launch["benchmark_sha256"])
    audit.verify(HERE / "launch.py", launch["launcher_sha256"])
    config = audit.json(launch["experiments_file"], launch["experiments_sha256"])
    registered = list(map(int, config["registered_opus_ids"]))
    require(set(registered) == OLD_REGISTERED and len(registered) == len(OLD_REGISTERED),
            "Registered old16 control pool differs")
    config_libraries = {entry["name"]: entry for entry in config["libraries"]}
    require(len(config_libraries) == len(config["libraries"])
            and len({library["name"] for library in launch["libraries"]}) == len(launch["libraries"])
            and set(config_libraries) == {library["name"] for library in launch["libraries"]},
            "Missing/duplicate experiment library")
    meta, by_id = {}, {}
    for library in launch["libraries"]:
        entry = config_libraries[library["name"]]
        directory = (Path(launch["experiments_file"]).parent / entry["directory"]).resolve()
        require(Path(library["directory"]).resolve() == directory
                and Path(library["binary"]).resolve() == directory / "experiments.so",
                f"Library path differs from configuration: {library['name']}")
        require(len(entry["ids"]) == len(set(map(int, entry["ids"]))),
                f"Duplicate configured runtime ID: {library['name']}")
        for path, digest in library["protected_sha256"].items():
            audit.verify(path, digest)
        for filename in ("variants.json", "experiments.so", "build_manifest.json"):
            require(str(directory / filename) in library["protected_sha256"],
                    f"Missing protected runtime artifact: {library['name']}/{filename}")
        variants = audit.json(directory / "variants.json")
        selected_variants = [variant for variant in variants if int(variant["id"]) in set(map(int, entry["ids"]))]
        require(selected_variants == library["variants"], f"Runtime metadata changed: {library['name']}")
        build = audit.json(directory / "build_manifest.json")
        require(build.get("status") == "passed"
                and build["binary_sha256"] == library["protected_sha256"][str(directory / "experiments.so")],
                f"Runtime binary differs from its successful build: {library['name']}")
        for variant in library["variants"]:
            key, kid = (library["name"], int(variant["id"])), int(variant["id"])
            require(key not in meta and kid not in by_id, "Ambiguous/duplicate runtime ID")
            require(not variant.get("fixed_k") and variant.get("runtime_k", True), f"Fixed-K runtime candidate: {key}")
            meta[key] = dict(variant, library=library["name"], directory=library["directory"])
            by_id[kid] = key
    require(set(by_id) == OLD_RUNTIME | new_ids, "Runtime pool must contain exactly old11 plus --new-ids")
    require(all(by_id[kid][0] == library for kid, library in OLD_RUNTIME_LIBRARIES.items()),
            "Retained old11 library identities differ")
    require(all(by_id[kid][0] in NEW_LIBRARIES for kid in new_ids),
            "New runtime candidates must belong to main, small, or narrow")
    if "merged_opus_ids" in config:
        require(set(map(int, config["merged_opus_ids"])) == PRESERVED | new_ids,
                "Configured merged OPUS pool differs from 9000/9020 plus --new-ids")
    configured = {(entry["name"], int(kid)) for entry in config["libraries"] for kid in entry["ids"]}
    require(configured == set(meta), "Launch/config runtime metadata differs")
    for kid in OLD_REGISTERED:
        key = "registered_opus", kid
        meta[key] = dict(id=kid, name=f"opus_{kid}", library="registered_opus")
        by_id[kid] = key
    old_keys = {by_id[kid] for kid in OLD16}
    new_keys = {by_id[kid] for kid in PRESERVED | new_ids}
    common_path = ROOT / "csrc/opus_gemm/opus_gemm_common.py"
    audit.verify(common_path)
    spec = importlib.util.spec_from_file_location("_remote_analysis_common", common_path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    exec(compile(common_path.read_text(), str(common_path), "exec"), common.__dict__)

    usage = {key: Counter() for key in meta}
    comparisons, selections, old_selections, new_choices, old_choices = [], [], [], [], []
    status_rows, valid_rows, rejects, runs = [], [], [], []
    per_new, impact, transition = defaultdict(list), defaultdict(list), Counter()
    covered, common_reference = set(), None
    for worker in launch["workers"]:
        require(worker.get("status") == "passed" and worker.get("exit_code") == 0, "Every worker must pass")
        run_path = within(worker["run_manifest"], batch, "Current worker manifest")
        require(run_path.parent == batch, "Current worker must belong to the selected batch")
        run = audit.json(run_path, worker["run_sha256"])
        local = {tuple(s) for s in worker["shapes"]}
        require(not covered & local and len(local) == worker["shape_count"] == len(worker["shapes"]),
                "Overlapping/duplicate worker shard")
        covered.update(local)
        require(run.get("status") == "passed" and run["completed_shapes"] == len(local)
                and len(run["shapes"]) == len(local) and {tuple(s) for s in run["shapes"]} == local,
                "Worker did not complete its shard")
        require(run["gpu"] == worker["gpu"] and run["gpu_uuid"] == worker["gpu_uuid"]
                and run["rounds"] == launch["rounds"] and run["external_mode"] == launch["external_mode"],
                "Worker identity/rounds/mode differs")
        require(run["experiments_file"] == launch["experiments_file"] and run["libraries"] == launch["libraries"],
                "Worker experiment configuration differs")
        require(Path(run["outputs"]["run"]).resolve() == run_path, "Worker output manifest path differs")
        shape_path = within(run["shapes_file"], batch, "Current worker shapes")
        require(shape_path == Path(worker["shapes_file"]).resolve(), "Worker shape input differs")
        require(str(shape_path) in run["protected_sha256"], "Worker shape file is not protected")
        shard_rows = audit.csv(shape_path, run["protected_sha256"][str(shape_path)])
        require(len(shard_rows) == len(local) and {shape(row) for row in shard_rows} == local,
                "Worker shape file differs from its declared shard")
        if full_batch is None:
            require(run.get("full_input") is None and worker.get("full_manifest") is None,
                    "Full-mode worker must not depend on another batch")
        for path, digest in run["protected_sha256"].items():
            audit.verify(path, digest)
        require(run["protected_sha256"].get(launch["experiments_file"]) == launch["experiments_sha256"]
                and run["reference_sha256"]["measurement_harness"] == launch["benchmark_sha256"],
                "Worker harness/configuration hash differs")
        require(common_reference is None or run["reference_sha256"] == common_reference,
                "Workers used differing references or timing helpers")
        common_reference = run["reference_sha256"]
        require(CSV_KINDS | {"gpu"} == set(run["output_sha256"]), "Missing/unexpected output artifacts")
        artifacts = {}
        for kind, digest in run["output_sha256"].items():
            path = within(run["outputs"][kind], batch, "Current worker output")
            artifacts[kind] = audit.csv(path, digest) if kind in CSV_KINDS else audit.verify(path, digest)
        require(len(artifacts["check"]) == run["checks"] and len(artifacts["reject"]) == run["rejected_checks"]
                and len(artifacts["raw"]) == run["timing_records"], "Worker artifact counters differ")
        require(artifacts["reject"] == [r for r in artifacts["check"] if r["status"] != "passed"],
                "Rejected-check ledger differs")
        rows = {kind: grouped(artifacts[kind]) for kind in CSV_KINDS}
        require(all(set(values) <= local for values in rows.values()), "Artifact contains foreign shape")
        require(all(set(rows[kind]) == local for kind in CSV_KINDS - {"reject"}), "Incomplete per-shape artifacts")
        require(all(int(row["gpu"]) == run["gpu"] and row["gpu_uuid"] == run["gpu_uuid"]
                    and int(row["rounds"]) == run["rounds"] and row["external_mode"] == run["external_mode"]
                    for kind in ("selection", "comparison") for row in artifacts[kind]),
                "Recorded selection/comparison GPU, rounds, or external mode differs")
        finalists = external_finalists(audit, worker, run, local, full_batch) if full_batch is not None else None
        provenance = dict(gpu=run["gpu"], gpu_uuid=run["gpu_uuid"], rounds=run["rounds"],
                          external_mode=run["external_mode"], run_manifest=worker["run_manifest"], run_sha256=worker["run_sha256"])
        for target in sorted(local):
            candidates, summaries, checks, failed, samples = audit_shape(target, rows, run["rounds"], meta, common, finalists)
            new = best(summaries, lambda r: r["lib"] in OPUS and candidate(r) in new_keys)
            old = best(summaries, lambda r: r["lib"] in OPUS and candidate(r) in old_keys)
            external = best(summaries, lambda r: r["lib"] in EXTERNAL)
            require(new and old and external, f"No valid new-pool/old16/external choice: {target}")
            winner = new if us(new) < us(external) else external
            old_winner = old if us(old) < us(external) else external
            new_rounds = sum(samples[identity(new)][t] < samples[identity(old)][t] for t in range(run["rounds"]))
            external_rounds = sum(samples[identity(new)][t] < samples[identity(external)][t] for t in range(run["rounds"]))
            comp = dict(zip(("M", "N", "K"), target)) | provenance
            comp.update(prefixed("new_pool", new) | prefixed("old16", old) | prefixed("external", external)
                        | prefixed("new_overall", winner) | prefixed("old16_overall", old_winner))
            comp.update(new_pool_delta_old16_pct=delta(us(new), us(old)),
                        new_pool_delta_external_pct=delta(us(new), us(external)),
                        delta_old16_pct=delta(us(new), us(old)),
                        delta_external_pct=delta(us(new), us(external)),
                        new_overall_delta_old16_overall_pct=delta(us(winner), us(old_winner)),
                        new_pool_faster_old16_rounds=new_rounds, new_pool_faster_external_rounds=external_rounds,
                        new_pool_overall_selected=winner is new, old16_overall_selected=old_winner is old,
                        failed_new_ids=",".join(map(str, sorted({int(r["kid"]) for r in candidates
                            if r["lib"] == "experiment" and int(r["kid"]) in new_ids and identity(r) in failed}))))
            comparisons.append(comp)
            selections.append({field: winner[field] for field in IDENTITY_FIELDS} | provenance | dict(median_us=us(winner)))
            old_selections.append({field: old_winner[field] for field in IDENTITY_FIELDS} | provenance | dict(median_us=us(old_winner)))
            new_choices.append(dict(new, **provenance, delta_old16_pct=delta(us(new), us(old)),
                                    delta_external_pct=delta(us(new), us(external)), overall_selected=winner is new))
            old_choices.append(dict(old, **provenance, delta_new_pool_pct=delta(us(old), us(new)),
                                    delta_external_pct=delta(us(old), us(external)), overall_selected=old_winner is old))
            summary_by_id = {identity(r): r for r in summaries}
            for row in candidates:
                key, row_id = candidate(row), identity(row)
                is_opus = row["lib"] in OPUS
                active = row["selected"] == "True"
                valid = row_id in summary_by_id
                failed_checks = [r for r in checks if identity(r) == row_id and r["status"] == "failed"]
                ledger = {field: row[field] for field in IDENTITY_FIELDS} | provenance
                ledger.update(in_old16=is_opus and key in old_keys, in_new_pool=is_opus and key in new_keys,
                              new_kernel=row["lib"] == "experiment" and int(row["kid"]) in new_ids,
                              selected_for_measurement=active,
                              status="valid" if valid else "failed" if row_id in failed else "not_selected_external",
                              completed_passed_rounds=sum(r["phase"] == "timed" and r["status"] == "passed"
                                                          for r in checks if identity(r) == row_id),
                              failure_reasons=" | ".join(dict.fromkeys(r["reason"] for r in failed_checks)))
                status_rows.append(ledger)
                if valid:
                    value = summary_by_id[row_id]
                    metrics = dict(ledger, median_us=us(value), min_us=float(value["min_us"]), max_us=float(value["max_us"]),
                                   delta_old16_pct=delta(us(value), us(old)), delta_external_pct=delta(us(value), us(external)))
                    valid_rows.append(metrics)
                    if ledger["new_kernel"]:
                        per_new[key].append(metrics)
                if is_opus:
                    usage[key]["legal_shapes"] += 1
                    usage[key]["valid_shapes"] += valid
                    usage[key]["failed_shapes"] += row_id in failed
            usage[candidate(new)]["new_pool_best_shapes"] += 1
            usage[candidate(old)]["old16_best_shapes"] += 1
            if winner is new:
                usage[candidate(new)]["new_pool_overall_selected_shapes"] += 1
            if old_winner is old:
                usage[candidate(old)]["old16_overall_selected_shapes"] += 1
            impact[candidate(old)].append(comp)
            transition[candidate(old), candidate(new)] += 1
        for row in artifacts["reject"]:
            rejects.append(dict(row, **provenance,
                                in_old16=row["lib"] in OPUS and candidate(row) in old_keys,
                                in_new_pool=row["lib"] in OPUS and candidate(row) in new_keys,
                                new_kernel=row["lib"] == "experiment" and int(row["kid"]) in new_ids))
        runs.append(dict(path=worker["run_manifest"], sha256=worker["run_sha256"], gpu=run["gpu"], gpu_uuid=run["gpu_uuid"],
                         completed_shapes=run["completed_shapes"], start_time=run["start_time"], end_time=run["end_time"]))
    require(covered == expected_shapes and len(comparisons) == len(expected_shapes), "Incomplete batch coverage")

    usage_fields = ("legal_shapes", "valid_shapes", "failed_shapes", "old16_best_shapes", "old16_overall_selected_shapes",
                    "new_pool_best_shapes", "new_pool_overall_selected_shapes")
    usage_rows = [dict(library=key[0], kid=key[1], name=variant["name"], in_old16=key in old_keys, in_new_pool=key in new_keys,
                       **{field: usage[key][field] for field in usage_fields})
                  for key, variant in sorted(meta.items(), key=lambda item: item[0][1])]
    kernel_metrics = []
    for kid in sorted(new_ids):
        key = by_id[kid]
        values = per_new[key]
        stats = ratio_stats(1 + row["delta_old16_pct"] / 100 for row in values)
        ext_stats = ratio_stats(1 + row["delta_external_pct"] / 100 for row in values)
        worst = max(values, key=lambda row: row["delta_old16_pct"], default=None)
        chosen = [row for row in comparisons if int(row["new_pool_kid"]) == kid]
        kernel_metrics.append(dict(library=key[0], kid=kid, **{field: usage[key][field] for field in usage_fields},
            **{f"valid_shapes_vs_old16_{field}": value for field, value in stats.items()},
            **{f"valid_shapes_vs_external_{field}": value for field, value in ext_stats.items()},
            **{f"selected_shapes_vs_old16_{field}": value for field, value in ratio_stats(
                row["new_pool_us"] / row["old16_us"] for row in chosen).items()},
            worst_shape="x".join(map(str, shape(worst))) if worst else ""))
    impact_rows = []
    for key, values in sorted(impact.items(), key=lambda item: item[0][1]):
        worst = max(values, key=lambda row: row["new_pool_delta_old16_pct"])
        impact_rows.append(dict(old16_library=key[0], old16_kid=key[1], old16_name=meta[key]["name"],
            removed_from_new_pool=key not in new_keys,
            **ratio_stats(row["new_pool_us"] / row["old16_us"] for row in values),
            old16_overall_selected_shapes=sum(row["old16_overall_selected"] for row in values),
            new_pool_overall_selected_shapes=sum(row["new_pool_overall_selected"] for row in values),
            worst_shape="x".join(map(str, shape(worst))),
            replacement_counts=json.dumps({f"{new_key[0]}:{new_key[1]}": count
                for (old_key, new_key), count in sorted(transition.items()) if old_key == key}, sort_keys=True)))
    selected_keys = {key for key in new_keys if usage[key]["new_pool_overall_selected_shapes"] > 0}
    selected_config = dict(registered_opus_ids=sorted(key[1] for key in selected_keys if key[0] == "registered_opus"), libraries=[])
    for entry in config["libraries"]:
        ids = [int(kid) for kid in entry["ids"] if (entry["name"], int(kid)) in selected_keys]
        if ids:
            directory = (Path(launch["experiments_file"]).parent / entry["directory"]).resolve()
            selected_config["libraries"].append(dict(name=entry["name"], directory=os.path.relpath(directory, output), ids=ids))
    failed_new = [row for row in status_rows if row["new_kernel"] and row["status"] == "failed"]
    summary = dict(status="passed", audit_status="passed", shape_count=len(expected_shapes), rounds=launch["rounds"],
        external_mode=launch["external_mode"], analysis_scope="full295" if len(expected_shapes) == 295 else "subset",
        timing_policy="All comparison timings are current-batch medians on the same GPU per shape; prior full data only selects external identities",
        selection_policy="new_pool=9000/9020 plus --new-ids; old16 controls never enter new_pool selection",
        old16_ids=sorted(OLD16), new_ids=sorted(new_ids), new_pool_ids=sorted(PRESERVED | new_ids),
        excluded_measured_runtime_ids=sorted(set(by_id) - OLD16 - new_ids), configured_old16_candidates=16,
        configured_new_pool_candidates=len(new_keys), new_pool_overall_used_candidates=len(selected_keys),
        new_pool_opus_wins=sum(row["new_pool_overall_selected"] for row in comparisons),
        old16_opus_wins=sum(row["old16_overall_selected"] for row in comparisons),
        new_candidates_all_valid=not failed_new, failed_new_candidate_shapes=len(failed_new),
        failed_new_ids=sorted({int(row["kid"]) for row in failed_new}),
        new_pool_vs_old16=ratio_stats(row["new_pool_us"] / row["old16_us"] for row in comparisons),
        new_pool_vs_external=ratio_stats(row["new_pool_us"] / row["external_us"] for row in comparisons),
        new_overall_vs_old16_overall=ratio_stats(row["new_overall_us"] / row["old16_overall_us"] for row in comparisons),
        new_pool_faster_old16_rounds=dict(Counter(row["new_pool_faster_old16_rounds"] for row in comparisons)),
        new_pool_faster_external_rounds=dict(Counter(row["new_pool_faster_external_rounds"] for row in comparisons)),
        candidate_status_counts=dict(Counter(row["status"] for row in status_rows)),
        selected_config=selected_config, launch_manifest=str(launch_path), launch_sha256=audit.verify(launch_path),
        run_manifests=runs, elapsed_seconds=launch["end_time"] - launch["start_time"], verified_artifacts=dict(audit.files))
    output.mkdir(parents=True, exist_ok=False)
    tables = {"comparison.csv": comparisons, "new_pool_overall_selection.csv": selections,
        "overall_selection.csv": selections,
        "old16_overall_selection.csv": old_selections, "new_pool_choices.csv": new_choices,
        "best_old16.csv": old_choices, "old16_choices.csv": old_choices,
        "candidate_usage.csv": usage_rows, "old16_usage.csv": [r for r in usage_rows if r["in_old16"]],
        "new_pool_usage.csv": [r for r in usage_rows if r["in_new_pool"]], "new_kernel_metrics.csv": kernel_metrics,
        "merge_impact.csv": impact_rows, "candidate_status.csv": status_rows, "valid_candidates.csv": valid_rows,
        "failed_candidates.csv": rejects, "failed_new_candidates.csv": [r for r in rejects if r["new_kernel"]],
        "regressions_vs_old16.csv": [r for r in comparisons if r["new_pool_delta_old16_pct"] > 0],
        "remaining_external_losses.csv": [r for r in comparisons if not r["new_pool_overall_selected"]],
        "remaining_losses.csv": [r for r in comparisons if not r["new_pool_overall_selected"]]}
    for name, values in tables.items():
        if values and all("M" in row for row in values):
            values.sort(key=lambda row: (shape(row)[1], shape(row)[2], shape(row)[0], row.get("name", "")))
        write_csv(output / name, values)
    (output / "selected_experiments.json").write_text(json.dumps(selected_config, indent=2) + "\n")
    text = [f"# 同批合并流程对照：{len(expected_shapes)} 项", "",
        f"完整通过审计，{launch['rounds']} 轮，外部模式 `{launch['external_mode']}`。",
        f"old16 固定 16 个；新池 {len(new_keys)} 个，只含 9000/9020 和参数指定的新 ID。",
        f"相对有效外部：新池胜 {summary['new_pool_opus_wins']} 项，old16 胜 {summary['old16_opus_wins']} 项。",
        f"新池相对 old16：几何平均 {summary['new_pool_vs_old16']['geomean_delta_pct']:+.4f}%，"
        f"最大退化 {summary['new_pool_vs_old16']['max_delta_pct']:+.4f}%。",
        f"含有效外部的最终选择相对 old16+外部：几何平均 "
        f"{summary['new_overall_vs_old16_overall']['geomean_delta_pct']:+.4f}%。",
        f"新候选失败 {len(failed_new)} 个候选/shape，失败 ID：{summary['failed_new_ids']}。失败候选已排除选型并逐条保留。", "",
        "所有百分比为耗时变化，正值表示变慢；几何平均按 shape 等权。候选指标只覆盖该候选合法且有效的 shape，"
        "其失败数单列；选择后的指标覆盖该候选成为新池赢家的 shape。",
        "full 模式重测全部外部候选；finalists 模式仅从本新机已有 full 批次读取候选身份。"
        "每个 shape 的新池、old16、外部比较耗时均来自本批次的同一张卡。", "",
        "| ID | 库 | 合法 | 有效 | 失败 | old16最快 | old16最终 | 新池最快 | 新池最终 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    text += [f"| {r['kid']} | {r['library']} | {r['legal_shapes']} | {r['valid_shapes']} | {r['failed_shapes']} | "
             f"{r['old16_best_shapes']} | {r['old16_overall_selected_shapes']} | {r['new_pool_best_shapes']} | "
             f"{r['new_pool_overall_selected_shapes']} |" for r in usage_rows]
    text += ["", "comparison.csv 为逐项三方比较；old16_usage.csv / new_pool_usage.csv 为并列用量；"
             "merge_impact.csv 按原 old16 赢家归因；new_kernel_metrics.csv 为各新 ID 的几何差与最大退化。",
             "candidate_status.csv 保留未测外部项、有效项和失败项；failed_new_candidates.csv 保留新 ID 的具体失败记录。",
             "new_pool_choices.csv 为新池最快候选；new_pool_overall_selection.csv 和 selected_experiments.json "
             "仅从新池加本批有效外部中选择。old16_overall_selection.csv 列出 old16 加相同外部参考的选择。",
             "友好别名：old16_choices.csv = best_old16.csv；overall_selection.csv = new_pool_overall_selection.csv；"
             "remaining_losses.csv = remaining_external_losses.csv。comparison.csv 的 delta_old16_pct / delta_external_pct "
             "分别等于 new_pool_delta_old16_pct / new_pool_delta_external_pct，均为 (new_us / baseline_us - 1) × 100。"]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n")
    summary["output_sha256"] = {path.name: audit.verify(path) for path in output.iterdir() if path.is_file()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "shape_count", "new_pool_opus_wins", "old16_opus_wins",
        "new_candidates_all_valid", "failed_new_candidate_shapes", "new_pool_vs_old16", "new_overall_vs_old16_overall")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
