#!/usr/bin/env python3
"""Measure registered and experimental OPUS against valid CK/CKTile/ASM.

Uses existing binaries only. Full mode enumerates every external candidate;
finalists mode reuses the valid external candidates within 5% of a successful
same-GPU full sweep. Every selected candidate is checked on its target shape.
"""

import argparse
from collections import defaultdict
import ctypes
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "csrc/opus_gemm/opus_gemm_common.py").is_file())
JIT = HERE / "jit"
import base
GPU_MAP = base.GPU_MAP
EXTERNAL = {"ck", "cktile", "asm"}
OPUS = {"opus", "experiment"}
OLD_KIDS = {9000, 9010, 9011, 9012, 9020}
IDENTITY = ["M", "N", "K", "name", "lib", "kid", "splitK", "kernelName", "library"]
CHECK_FIELDS = IDENTITY + ["phase", "round", "status", "error", "guards", "reason"]
RAW_FIELDS = IDENTITY + ["round", "order", "us", "status", "error", "reason"]
SUMMARY_FIELDS = IDENTITY + ["median_us", "min_us", "max_us", "samples"]
COMPARISON_FIELDS = ["M", "N", "K", "gpu", "gpu_uuid", "rounds", "external_mode",
                     "reference", "reference_lib", "reference_us", "opus", "opus_lib", "opus_library", "opus_kid",
                     "opus_us", "speedup", "opus_improvement_pct", "opus_faster", "opus_faster_rounds",
                     "old_opus", "old_opus_us", "registered_opus", "registered_opus_us",
                     "ck", "ck_us", "cktile", "cktile_us", "asm", "asm_us"]
SELECTION_FIELDS = IDENTITY + ["gpu", "gpu_uuid", "median_us", "rounds", "external_mode",
                               "dtype", "scale_dtype", "outdtype", "preshuffleB"]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(row):
    return row["lib"], int(row["kid"]), int(row["splitK"]), row["kernelName"], row.get("library", "")


def shape_key(row):
    return tuple(int(row[key]) for key in ("M", "N", "K"))


def read_shapes(path):
    result = base.read_shapes(path, False)
    if len(base.read_csv(path)) != len(result):
        raise ValueError("Shapes must be unique")
    return result


def experiments(path):
    config = json.loads(path.read_text())
    entries = config["libraries"] if isinstance(config, dict) else config
    result, names = [], set()
    for entry in entries:
        directory = (path.parent / entry["directory"]).resolve()
        name = entry.get("name", directory.name)
        if name in names:
            raise ValueError(f"Duplicate experiment library name: {name}")
        names.add(name)
        variants_path, binary_path = directory / "variants.json", directory / "experiments.so"
        variants = json.loads(variants_path.read_text())
        requested = set(map(int, entry["ids"])) if "ids" in entry else None
        if requested is not None and not requested <= {int(v["id"]) for v in variants}:
            raise ValueError(f"Unknown requested IDs in {name}")
        variants = [v for v in variants if requested is None or int(v["id"]) in requested]
        if len({int(v["id"]) for v in variants}) != len(variants):
            raise ValueError(f"Duplicate variant IDs in {name}")
        build_path = directory / "build_manifest.json"
        build = json.loads(build_path.read_text())
        if build.get("status") != "passed" or base.sha256(binary_path) != build["binary_sha256"]:
            raise ValueError(f"Experiment binary is not the successful frozen build: {name}")
        protected = {str(p): base.sha256(p) for p in directory.rglob("*")
                     if p.is_file() and p.suffix in {".py", ".cuh", ".hip", ".hpp", ".h", ".json", ".so"}}
        result.append(dict(name=name, directory=str(directory), binary=str(binary_path),
                           variants=variants, protected_sha256=protected))
    return result


def supports(variant, shape):
    m, n, k = shape
    fixed_k = int(variant.get("fixed_k") or 0)
    return (not fixed_k or k == fixed_k) and all(
        value % int(variant.get(field, default)) == 0
        for value, field, default in ((m, "m_multiple", 64), (n, "n_multiple", 256), (k, "k_multiple", 128))
    ) and k >= int(variant.get("min_k") or 0) and k <= int(variant.get("max_k") or k)


def external_finalists(path, references, shapes):
    run = json.loads(path.read_text())
    if run.get("status") != "passed" or run.get("external_mode") != "full":
        raise ValueError("Finalists require a successful full-mode manifest")
    if run.get("gpu_uuid") != base.GPU_UUID or {k: v for k, v in run.get("reference_sha256", {}).items() if k != "measurement_harness"} != {k: v for k, v in references.items() if k != "measurement_harness"}:
        raise ValueError("Full-mode GPU or reference/timing artifacts differ")
    if not set(shapes) <= {tuple(s) for s in run["shapes"]}:
        raise ValueError("Full-mode manifest does not cover all requested shapes")
    artifacts = {}
    for kind in ("summary", "candidates"):
        source = Path(run["outputs"][kind])
        if base.sha256(source) != run["output_sha256"][kind]:
            raise ValueError(f"Full-mode output changed: {source}")
        artifacts[kind] = base.read_csv(source)
    # The numerical comparator and external source/binary hashes match above;
    # only this adapter differs from the completed full-sweep harness.
    selected, registry = {}, {}
    for shape in shapes:
        rows = [r for r in artifacts["summary"] if shape_key(r) == shape and r["lib"] in EXTERNAL]
        if not rows or any(int(r["samples"]) != run["rounds"] for r in rows):
            raise ValueError(f"No complete valid external reference in full mode: {shape}")
        threshold = min(float(r["median_us"]) for r in rows) * 1.05
        selected[shape] = {identity(r) for r in rows if float(r["median_us"]) <= threshold}
        registry[shape] = {identity(r) for r in artifacts["candidates"]
                           if shape_key(r) == shape and r["lib"] in EXTERNAL}
    return selected, registry


class Output:
    def __init__(self, prefix, manifest):
        csv_fields = dict(candidates=IDENTITY + ["selected", "selection_reason"],
                          check=CHECK_FIELDS, reject=CHECK_FIELDS, raw=RAW_FIELDS,
                          summary=SUMMARY_FIELDS, comparison=COMPARISON_FIELDS, selection=SELECTION_FIELDS)
        self.paths = {name: Path(f"{prefix}_{name}.csv") for name in csv_fields}
        self.paths.update(run=Path(f"{prefix}_run.json"), gpu=Path(f"{prefix}_gpu.jsonl"))
        if any(path.exists() for path in self.paths.values()):
            raise FileExistsError(f"Use a fresh output prefix: {prefix}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        manifest["outputs"] = {k: str(v) for k, v in self.paths.items()}
        with self.paths["run"].open("x") as stream:
            json.dump(manifest, stream, indent=2)
        self.streams, self.writers = {}, {}
        for name, fields in csv_fields.items():
            self.streams[name] = self.paths[name].open("x", newline="")
            self.writers[name] = csv.DictWriter(self.streams[name], fieldnames=fields)
            self.writers[name].writeheader()
            self.streams[name].flush()
        self.streams["gpu"] = self.paths["gpu"].open("x")

    def row(self, kind, row):
        self.writers[kind].writerow(row)
        self.streams[kind].flush()

    def sample(self, row):
        self.streams["gpu"].write(json.dumps(row) + "\n")
        self.streams["gpu"].flush()

    def save(self):
        atomic_json(self.paths["run"], self.manifest)

    def close(self):
        for stream in self.streams.values():
            stream.close()
        self.manifest["output_sha256"] = {k: base.sha256(p) for k, p in self.paths.items() if k != "run"}
        self.save()


def compare(output, shape, summaries, records, rounds, mode):
    def best(predicate):
        return min((r for r in summaries if predicate(r)), key=lambda r: (r["median_us"], r["name"]), default=None)
    reference = best(lambda r: r["lib"] in EXTERNAL)
    opus = best(lambda r: r["lib"] in OPUS)
    if reference is None or opus is None:
        raise RuntimeError(f"No complete valid external or OPUS candidate for {shape}")
    old = best(lambda r: r["lib"] == "opus" and int(r["kid"]) in OLD_KIDS)
    registered = best(lambda r: r["lib"] == "opus")
    opus_rounds = {r["round"]: r["us"] for r in records if r["name"] == opus["name"]}
    ref_rounds = {r["round"]: r["us"] for r in records if r["name"] == reference["name"]}
    row = dict(zip(("M", "N", "K"), shape))
    row.update(gpu=base.GPU_INDEX, gpu_uuid=base.GPU_UUID, rounds=rounds, external_mode=mode,
               reference=reference["name"], reference_lib=reference["lib"], reference_us=reference["median_us"],
               opus=opus["name"], opus_lib=opus["lib"], opus_library=opus["library"], opus_kid=opus["kid"],
               opus_us=opus["median_us"], speedup=reference["median_us"] / opus["median_us"],
               opus_improvement_pct=(1 - opus["median_us"] / reference["median_us"]) * 100,
               opus_faster=opus["median_us"] < reference["median_us"],
               opus_faster_rounds=sum(opus_rounds[t] < ref_rounds[t] for t in range(rounds)))
    for name, candidate in [("old_opus", old), ("registered_opus", registered)] + [
            (lib, best(lambda r, lib=lib: r["lib"] == lib)) for lib in ("ck", "cktile", "asm")]:
        if candidate:
            row.update({name: candidate["name"], f"{name}_us": candidate["median_us"]})
    output.row("comparison", row)
    winner = opus if row["opus_faster"] else reference
    choice = {field: winner[field] for field in IDENTITY}
    choice.update(gpu=base.GPU_INDEX, gpu_uuid=base.GPU_UUID, median_us=winner["median_us"], rounds=rounds,
                  external_mode=mode, dtype="fp8", scale_dtype="e8m0", outdtype="bf16", preshuffleB=True)
    output.row("selection", choice)
    print(json.dumps(row), flush=True)
    return row


def run(args):
    from build import assert_source_roots, verify_build
    verify_build()
    shapes = read_shapes(args.shapes)
    libraries = experiments(args.experiments)
    sources, binaries, references = base.snapshot(JIT)
    references["measurement_harness"] = base.sha256(__file__)
    references["archived_harness_helpers"] = base.sha256(HERE / "base.py")
    selected, prior_registry = None, None
    if args.external_mode == "finalists":
        selected, prior_registry = external_finalists(args.full_manifest, references, shapes)
    protected = {str(ROOT / path): digest for path, digest in sources.items()} | binaries
    protected.update({str(Path(__file__).resolve()): base.sha256(__file__),
                      str(HERE / "base.py"): references["archived_harness_helpers"],
                      str(args.experiments): base.sha256(args.experiments), str(args.shapes): base.sha256(args.shapes)})
    for name in ("launch.py", "preflight.py", "machine.py", "machine_config.json", "build.json"):
        path = HERE / name
        protected[str(path)] = base.sha256(path)
    for library in libraries:
        protected.update(library["protected_sha256"])
    manifest = dict(status="running", start_time=time.time(), pid=os.getpid(), argv=sys.argv,
                    gpu=base.GPU_INDEX, gpu_uuid=base.GPU_UUID, gpu_bus=base.GPU_BUS,
                    rounds=args.rounds, external_mode=args.external_mode, shapes=shapes,
                    shapes_file=str(args.shapes), experiments_file=str(args.experiments), libraries=libraries,
                    source_sha256=sources, binary_sha256=binaries, reference_sha256=references,
                    protected_sha256=protected, full_input=None if args.full_manifest is None else
                    dict(path=str(args.full_manifest), sha256=base.sha256(args.full_manifest)),
                    numerical_contract="Original FP32 elementwise interval; error=0 and 64-row NaN output guards",
                    timing="Original run_perftest profiler; warmup=5, iters=51; automatic full input/output storage rotation",
                    candidate_policy="Original registered controls plus configured retained controls and merged single-flow candidates; external finalists are identities within 5% of the latest complete full sweep and all timings are measured anew")
    output = Output(args.prefix, manifest)
    check_count, rejected, timing_count, comparisons = 0, 0, 0, []
    try:
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(key, None)
        os.environ.update(ROCR_VISIBLE_DEVICES=base.GPU_UUID, AITER_AOT_IMPORT="1", AITER_REBUILD="0",
                          AITER_JIT_DIR=str(JIT), GPU_ARCHS="gfx950", CU_NUM="256", OMP_NUM_THREADS="2",
                          AITER_LOG_MORE="0", AITER_SMI_MONITOR="0", AITER_META_DIR=str(ROOT),
                          CK_DIR=str(ROOT / "3rdparty/composable_kernel"),
                          OPUS_GEN_CO_DIR=str(ROOT / "csrc/opus_gemm/gen_co"))
        sys.path.insert(0, str(ROOT))
        import torch
        import aiter
        from aiter.utility import dtypes
        from aiter.ops import gemm_op_a8w8
        from aiter.jit import core
        from aiter.test_common import run_perftest
        assert_source_roots(core)
        aiter.dtypes = dtypes
        for name in ("gemm_a8w8_blockscale_bpreshuffle_tune", "gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
                     "gemm_a8w8_blockscale_bpreshuffle_asm"):
            setattr(aiter, name, getattr(gemm_op_a8w8, name))
        def forbid_build(*_a, **_kw):
            raise RuntimeError("Only existing frozen binaries may be measured")
        core.build_module = forbid_build
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected one UUID-bound GPU")
        prop = torch.cuda.get_device_properties(0)
        if ((prop.pci_bus_id, prop.pci_domain_id, prop.pci_device_id, prop.multi_processor_count)
                != (base.GPU_BUS, base.GPU_DOMAIN, base.GPU_DEVICE, 256) or not prop.gcnArchName.startswith("gfx950")):
            raise RuntimeError(f"Unexpected GPU: {prop}")
        for name in base.MODULES:
            if not name.endswith("_asm") and Path(core.get_module(name).__file__).resolve() != JIT / f"{name}.so":
                raise RuntimeError(f"Module loaded outside frozen JIT: {name}")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest.update(gpu_properties=str(prop), torch_version=torch.__version__, hip_version=torch.version.hip)
        tune = base.load_tune()
        tuner = tune.OpusMxscaleBpreshuffleTuner()
        handles = {}
        for library in libraries:
            handle = ctypes.CDLL(library["binary"])
            handle.launch.argtypes = [ctypes.c_int, *([ctypes.c_void_p] * 5),
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            handle.launch.restype = ctypes.c_int
            handles[library["name"]] = handle
        entries = {}
        keys = ("x", "w", "out", "x_scale", "w_scale", "x_scale_t_fp32", "x_scale_fp32", "w_scale_fp32", "zero_bias")
        def invoke(name, x, w, storage, sa, sb, sat32, sar32, sb32, zero_bias):
            m, k = x.shape
            n = storage.shape[1]
            out = storage[64:64 + m]
            candidate = entries[name]
            if isinstance(candidate, tuple):
                library_name, variant = candidate
                code = handles[library_name].launch(int(variant["id"]), x.data_ptr(), w.data_ptr(),
                                                     sa.data_ptr(), sb.data_ptr(), out.data_ptr(), m, n, k,
                                                     torch.cuda.current_stream().cuda_stream)
                if code:
                    raise RuntimeError(f"HIP launch error {code}")
            else:
                data = dict(x=x, w=w, weight_shuffle=w, out=out, x_scale=sa, w_scale=sb,
                            x_scale_t_fp32=sat32, x_scale_fp32=sar32, w_scale_fp32=sb32, zero_bias=zero_bias)
                candidate.func(*candidate.arguments(data), **candidate.kwargs)
            return storage
        def checked(reference, storage, m):
            result = base.check_output(torch, tune, reference, storage[64:64 + m])
            guards = bool(torch.isnan(storage[:64]).all() and torch.isnan(storage[64 + m:]).all())
            if not guards:
                result.update(status="failed", reason="output canary overwritten")
            return {key: result[key] for key in ("status", "error", "reason")} | {"guards": guards}
        base.require_idle(torch, output, "start")
        for shape_index, shape in enumerate(shapes):
            m, n, k = shape
            registered = [candidate for candidate in base.candidates_for_shape(tune, tuner, shape)
                          if candidate.row["lib"] != "opus" or int(candidate.row["kid"]) in OLD_KIDS]
            candidates = []
            for candidate in registered:
                row = dict(candidate.row, library="")
                candidates.append(row)
                entries[row["name"]] = candidate
            current_external = {identity(row) for row in candidates if row["lib"] in EXTERNAL}
            if prior_registry is not None and current_external != prior_registry[shape]:
                raise ValueError(f"External candidate registry changed since full mode: {shape}")
            for library in libraries:
                for variant in library["variants"]:
                    if supports(variant, shape):
                        name = f"experiment:{library['name']}:{variant['name']}"
                        row = dict(zip(("M", "N", "K"), shape))
                        row.update(name=name, lib="experiment", kid=int(variant["id"]), splitK=0,
                                   kernelName=variant["name"], library=library["name"])
                        candidates.append(row)
                        entries[name] = (library["name"], variant)
            if len({row["name"] for row in candidates}) != len(candidates):
                raise ValueError("Candidate names must be unique within a shape")
            active = []
            for row in candidates:
                take = args.external_mode == "full" or row["lib"] in OPUS or identity(row) in selected[shape]
                output.row("candidates", dict(row, selected=take, selection_reason=
                           "all OPUS" if row["lib"] in OPUS else "full external sweep" if args.external_mode == "full"
                           else "within 5% of valid full winner" if take else "outside valid full finalist set"))
                if take:
                    active.append(row)
            data = tune.generate_data(m, n, k, device="cuda:0")
            reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
            data["out"] = torch.full((m + 128, n), float("nan"), device="cuda", dtype=torch.bfloat16)
            failed, records = set(), []
            for row in active:
                data["out"].fill_(float("nan"))
                try:
                    storage = invoke(row["name"], *(data[key] for key in keys))
                    result = checked(reference, storage, m)
                except Exception as exc:
                    result = dict(status="failed", error="", guards="", reason=f"{type(exc).__name__}: {exc}")
                check = dict(row, phase="preflight", round=-1, **result)
                output.row("check", check)
                check_count += 1
                if result["status"] != "passed":
                    output.row("reject", check)
                    rejected += 1
                    failed.add(row["name"])
                    torch.cuda.synchronize()  # Stop if a candidate poisoned the device context.
            for turn in range(args.rounds):
                base.require_idle(torch, output, f"before_{shape}_round{turn}")
                good = [row for row in active if row["name"] not in failed]
                if not good:
                    break
                shift = (shape_index + turn) % len(good)
                order = good[shift:] + good[:shift]
                if turn % 2:
                    order.reverse()
                for position, row in enumerate(order):
                    us = None
                    data["out"].fill_(float("nan"))
                    try:
                        storage, us = run_perftest(invoke, row["name"], *(data[key] for key in keys),
                                                   num_warmup=5, num_iters=51, num_rotate_args=0)
                        us = float(us)
                        result = checked(reference, storage, m)
                        if not math.isfinite(us) or us <= 0:
                            result.update(status="failed", reason=f"invalid timing: {us}")
                            us = None
                    except Exception as exc:
                        result = dict(status="failed", error="", guards="", reason=f"{type(exc).__name__}: {exc}")
                    check = dict(row, phase="timed", round=turn, **result)
                    output.row("check", check)
                    check_count += 1
                    raw = dict(row, round=turn, order=position, us=us,
                               **{key: result[key] for key in ("status", "error", "reason")})
                    output.row("raw", raw)
                    timing_count += 1
                    if result["status"] == "passed":
                        records.append(raw)
                    else:
                        output.row("reject", check)
                        rejected += 1
                        failed.add(row["name"])
                        torch.cuda.synchronize()
                base.require_idle(torch, output, f"after_{shape}_round{turn}")
            summaries = []
            for row in active:
                values = [record["us"] for record in records if record["name"] == row["name"]]
                if row["name"] in failed or len(values) != args.rounds:
                    continue
                summary = dict(row, median_us=statistics.median(values), min_us=min(values),
                               max_us=max(values), samples=len(values))
                output.row("summary", summary)
                summaries.append(summary)
            comparisons.append(compare(output, shape, summaries, records, args.rounds, args.external_mode))
            manifest.update(completed_shapes=len(comparisons), checks=check_count, rejected_checks=rejected, timing_records=timing_count)
            output.save()
            del data, reference, summaries, records
            if "storage" in locals():
                del storage
            torch.cuda.empty_cache()
        base.require_idle(torch, output, "end")
        for path, digest in protected.items():
            if base.sha256(path) != digest:
                raise RuntimeError(f"Protected source/binary changed during run: {path}")
        manifest.update(status="passed", opus_wins=sum(row["opus_faster"] for row in comparisons))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(end_time=time.time(), completed_shapes=len(comparisons), checks=check_count,
                        rejected_checks=rejected, timing_records=timing_count)
        output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, default=HERE / "shapes295.csv")
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--list-shapes", action="store_true", help="list shapes without importing torch or querying a GPU")
    parser.add_argument("--rounds", type=int, choices=(3, 5), default=5)
    parser.add_argument("--external-mode", choices=("full", "finalists"), default="full")
    parser.add_argument("--full-manifest", type=Path)
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments.json")
    args = parser.parse_args()
    if args.list_shapes:
        print("M,N,K")
        for shape in read_shapes(args.shapes):
            print(",".join(map(str, shape)))
        return
    if args.prefix is None:
        parser.error("--prefix is required for measurements")
    if (args.external_mode == "finalists") != (args.full_manifest is not None):
        parser.error("--full-manifest is required exactly in finalists mode")
    args.shapes, args.experiments = args.shapes.resolve(), args.experiments.resolve()
    args.prefix = (HERE / args.prefix).resolve()
    if args.full_manifest:
        args.full_manifest = args.full_manifest.resolve()
    run(args)


if __name__ == "__main__":
    main()
