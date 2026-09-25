#!/usr/bin/env python3
"""Sweep CK/CKTile/ASM/OPUS, or remeasure valid close rivals and all legal OPUS.

No torch/aiter import occurs for --help or --list-shapes. Execution is pinned to
an assigned physical GPU by UUID and requires already prepared, separately built modules.
"""

import argparse
from bootstrap import load_tune
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SHAPES = ROOT / "reports/opus_mxfp8_remaining_20260924/remaining_shapes.csv"
GPU_MAP = {
    4: ("GPU-23a6cd0d658d72b6", 0x85),
    5: ("GPU-2d57f9bd7c2ee0fe", 0x95),
    6: ("GPU-56f0ab624008ec65", 0xE5),
    7: ("GPU-5ff36708541c8ec0", 0xF5),
}
GPU_INDEX = int(os.environ["OPUS_TUNE_GPU_INDEX"])
GPU_UUID, GPU_BUS = GPU_MAP[GPU_INDEX]
MODULES = (
    "module_aiter_core",
    "module_deepgemm_opus",
    "module_gemm_a8w8_blockscale_bpreshuffle_tune",
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    "module_gemm_a8w8_blockscale_bpreshuffle_asm",
)
IDENTITY = ["M", "N", "K", "name", "lib", "kid", "splitK", "kernelName"]
CHECK_FIELDS = IDENTITY + [
    "phase", "round", "status", "error", "reason", "nonfinite", "mismatches",
    "max_abs_error",
]
RAW_FIELDS = IDENTITY + ["round", "order", "us", "error", "status", "reason"]
SUMMARY_FIELDS = IDENTITY + ["median_us", "min_us", "max_us", "samples"]
CHOICE_FIELDS = [
    "M", "N", "K", "reference", "reference_us", "opus", "opus_us",
    "speedup", "opus_faster", "opus_9010_us", "opus_9011_us",
    "best_9010_9011", "best_9010_9011_us", "target_speedup", "target_faster",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_shapes(path, default_selection):
    rows = read_csv(path)
    if default_selection:
        rows = [row for row in rows if row.get("best_recorded_opus") in
                {"opus_9010", "opus_9011"}]
    shapes = set()
    for row in rows:
        shape = tuple(int(row[key]) for key in ("M", "N", "K"))
        if min(shape) <= 0 or shape[1] % 16 or shape[2] % 128:
            raise ValueError(f"Invalid MXFP8 B-preshuffle shape: {shape}")
        shapes.add(shape)
    if not shapes or (default_selection and len(shapes) != 17):
        raise ValueError(f"Expected {'17 default' if default_selection else 'nonempty'} shapes; got {len(shapes)}")
    return sorted(shapes, key=lambda shape: (shape[1], shape[2], shape[0]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true", help="measure every tuner candidate")
    mode.add_argument("--final", type=Path, metavar="SWEEP_RUN_JSON",
                      help="remeasure each valid rival within 5%% of the sweep winner")
    parser.add_argument("--prefix", type=Path, help="fresh output prefix; relative paths start in this harness directory")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--shapes", type=Path, help="CSV with M,N,K; default: the 17 remaining opus_9010/9011 shapes")
    parser.add_argument("--jit-dir", type=Path, default=HERE.parent / "jit_integrated",
                        help="directory containing prebuilt OPUS, CK, CKTile and ASM modules")
    parser.add_argument("--list-shapes", action="store_true", help="print selected M,N,K without importing torch or using a GPU")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    args.shapes_path = (args.shapes or DEFAULT_SHAPES).resolve()
    try:
        args.shape_list = read_shapes(args.shapes_path, args.shapes is None)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.list_shapes:
        return args
    if not (args.sweep or args.final) or args.prefix is None:
        parser.error("provide --sweep or --final SWEEP_RUN_JSON, and a fresh --prefix")
    args.prefix = (HERE / args.prefix).resolve()
    args.jit_dir = args.jit_dir.resolve()
    if args.final:
        args.final = args.final.resolve()
    return args


class Output:
    """Reserve a prefix once; append every check and raw perftest measurement."""

    def __init__(self, prefix, manifest):
        self.paths = {kind: Path(f"{prefix}_{kind}.{extension}") for kind, extension in
                      (("run", "json"), ("raw", "csv"), ("correctness", "csv"),
                       ("candidates", "csv"), ("summary", "csv"), ("choices", "csv"),
                       ("gpu", "jsonl"))}
        for path in self.paths.values():
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite results: {path}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        manifest["outputs"] = {key: str(path) for key, path in self.paths.items()}
        with self.paths["run"].open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        self.streams, self.writers = {}, {}
        for name, fields in (("raw", RAW_FIELDS), ("correctness", CHECK_FIELDS),
                             ("candidates", IDENTITY + ["selected", "selection_reason"]),
                             ("summary", SUMMARY_FIELDS), ("choices", CHOICE_FIELDS)):
            stream = self.paths[name].open("x", newline="")
            self.streams[name] = stream
            writer = self.writers[name] = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            stream.flush()
        self.streams["gpu"] = self.paths["gpu"].open("x")

    def row(self, kind, row):
        self.writers[kind].writerow(row)
        self.streams[kind].flush()

    def sample(self, value):
        self.streams["gpu"].write(json.dumps(value) + "\n")
        self.streams["gpu"].flush()

    def save(self):
        temporary = self.paths["run"].with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.paths["run"])

    def close(self):
        for stream in self.streams.values():
            stream.close()


@dataclass
class Candidate:
    row: dict
    func: object
    keys: tuple
    extra: tuple
    kwargs: dict

    @property
    def key(self):
        return candidate_identity(self.row)

    def arguments(self, data):
        return (*(data[key] for key in self.keys), *self.extra)


def candidate_identity(row):
    # A registry name need not encode every instance parameter. In particular,
    # distinct CK IDs can share a name, so keep both ID and name. ASM IDs depend
    # on the shape/split enumeration, which is fixed for each compared shape.
    return row["lib"], int(row["kid"]), row["kernelName"], int(row["splitK"])


def candidates_for_shape(tune, tuner, shape):
    info = ("gfx950", 256, *shape)
    tasks = tuner.get_gemm_a8w8_blockscale_tune_task(info, True, 0, True, {})
    tasks += tuner.get_gemm_a8w8_blockscale_cktile_tune_task(
        info, True, 0, True, list(range(1, tune.generic_tune.BLOCK_PER_CU_MAX + 1)), {})
    tasks += tuner.get_gemm_a8w8_blockscale_asm_tune_task(info, True, 0, True, {})
    tasks += tuner.get_gemm_a8w8_blockscale_opus_tune_task(info, 0, True, {})
    result = []
    for task in tasks:
        task_info, _, _, func, args, kwargs, *_ = task
        _, kid, split, kernel_name, lib, preshuffle = task_info
        if not preshuffle:
            raise ValueError("Non-preshuffle task returned by tuner")
        if not kernel_name:
            kernel_name = tuner.getKernelName(kid, lib, True)
        if not kernel_name:
            raise ValueError(f"Missing kernel name: {task_info}")
        row = dict(zip(("M", "N", "K"), shape))
        row.update(name=f"opus_{kid}" if lib == "opus" else f"{lib}_{kid}_split{split}",
                   lib=lib, kid=int(kid), splitK=int(split), kernelName=kernel_name)
        result.append(Candidate(row, func, tuple(args[0]), tuple(args[1:]), kwargs))
    if len({candidate.key for candidate in result}) != len(result):
        raise ValueError(f"Duplicate tuner candidates for {shape}")
    for lib in ("ck", "cktile", "opus"):
        if not any(candidate.row["lib"] == lib for candidate in result):
            raise ValueError(f"No {lib} candidates for {shape}")
    return result


def snapshot(jit_dir):
    source_paths = set()
    for directory in ("csrc/opus_gemm", "csrc/ck_gemm_a8w8_blockscale",
                      "csrc/ck_gemm_a8w8_blockscale_bpreshuffle", "csrc/include",
                      "3rdparty/composable_kernel/include", "3rdparty/composable_kernel/library/include", "3rdparty/ck_helper",
                      "aiter/ops/opus", "aiter/jit/utils"):
        source_paths.update(path for path in (ROOT / directory).rglob("*")
                            if path.is_file() and path.suffix in
                            {".py", ".h", ".hpp", ".cuh", ".cu", ".cpp"}
                            and "build" not in path.parts and "__pycache__" not in path.parts)
    for name in ("aiter/test_common.py", "aiter/jit/core.py", "aiter/jit/optCompilerConfig.json",
                 "aiter/ops/gemm_op_a8w8.py", "aiter/ops/opus/gemm_op_a8w8.py",
                 "aiter/ops/shuffle.py", "aiter/utility/mp_tuner.py", "aiter/utility/base_tuner.py", "aiter/utility/dtypes.py",
                 "csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu", "hsa/codegen.py"):
        source_paths.add(ROOT / name)
    asm_dir = ROOT / "hsa/gfx950/fp8gemm_blockscale"
    asm_csv = asm_dir / "fp8gemm_bf16_blockscale.csv"
    source_paths.add(asm_csv)
    if not asm_csv.is_file():
        raise FileNotFoundError(f"ASM registry is required for a complete sweep: {asm_csv}")
    asm_paths = {asm_dir / row["co_name"] for row in read_csv(asm_csv)
                 if int(row["bpreshuffle"]) == 1}
    if not asm_paths:
        raise ValueError("ASM registry has no B-preshuffle kernels")
    required_modules = [jit_dir / f"{name}.so" for name in MODULES]
    for path in required_modules + list(asm_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Prebuilt artifact required; build outside this harness: {path}")
    source_paths.update(HERE / name for name in ("tune_adapter.py", "bootstrap.py"))
    source_paths.update((ROOT / "csrc/pybind").glob("*a8w8*bpreshuffle*"))
    source_paths.add(ROOT / "csrc/pybind/aiter_core_pybind.cu")
    source_paths = {p for p in source_paths if p.is_file()}
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(source_paths)}
    binary_paths = set(jit_dir.glob("*.so")) | asm_paths
    binaries = {str(path.resolve()): sha256(path) for path in sorted(binary_paths)}
    # Freeze the reference side across sweep/final, while permitting an OPUS
    # optimization between them. The FP32 comparator is explicitly frozen.
    references = {name: digest for name, digest in sources.items()
                  if not name.startswith(("csrc/opus_gemm/", "csrc/include/opus/", "aiter/ops/opus/"))}
    comparator = str((HERE / "tune_adapter.py").relative_to(ROOT))
    references[comparator] = sources[comparator]
    for path, digest in binaries.items():
        path = Path(path)
        if path.name != "module_deepgemm_opus.so":
            references[f"{'jit' if path.parent == jit_dir else 'asm'}/{path.name}"] = digest
    return sources, binaries, references


def load_sweep(path, references, shapes):
    manifest = json.loads(path.read_text())
    if manifest.get("mode") != "sweep" or manifest.get("status") != "passed":
        raise ValueError("--final requires a successfully completed sweep manifest")
    if manifest.get("reference_sha256") != references:
        raise ValueError("Reference sources/binaries changed since sweep; run a fresh complete sweep")
    if manifest.get("gpu_uuid") != GPU_UUID:
        raise ValueError("Sweep GPU UUID does not match assigned GPU")
    if not set(shapes) <= {tuple(shape) for shape in manifest["shapes"]}:
        raise ValueError("Requested shapes were not all included in sweep")
    rows = {}
    for kind in ("raw", "correctness", "candidates"):
        source = Path(manifest["outputs"][kind])
        if sha256(source) != manifest["output_sha256"][kind]:
            raise ValueError(f"Sweep artifact was modified: {source}")
        rows[kind] = read_csv(source)
    grouped = defaultdict(list)
    failed = set()
    for row in rows["correctness"] + rows["raw"]:
        key = tuple(int(row[key]) for key in ("M", "N", "K")) + candidate_identity(row)
        if row["status"] != "passed" or float(row["error"] or "nan") != 0:
            failed.add(key)
    for row in rows["raw"]:
        key = tuple(int(row[key]) for key in ("M", "N", "K")) + candidate_identity(row)
        us = float(row["us"] or "nan")
        if key not in failed and row["lib"] != "opus" and math.isfinite(us) and us > 0:
            grouped[key].append((int(row["round"]), us))
    medians = {key: statistics.median(us for _, us in values) for key, values in grouped.items()
               if len(values) == manifest["rounds"] and
               {turn for turn, _ in values} == set(range(manifest["rounds"]))}
    selections = {}
    for shape in shapes:
        local = {key[3:]: us for key, us in medians.items() if key[:3] == shape}
        if not local:
            raise ValueError(f"Sweep has no fully measured valid reference for {shape}")
        threshold = min(local.values()) * 1.05
        selections[shape] = {key for key, us in local.items() if us <= threshold}
    return selections, rows["candidates"], {"path": str(path), "sha256": sha256(path)}


def check_output(torch, tune, reference, actual):
    error = float(tune.compare_outputs(reference, actual, printLog=False))
    result = dict(error=error, status="passed" if error == 0 else "failed",
                  reason="" if error == 0 else "outside original FP32 accumulation bounds")
    if error:
        expected, magnitude = reference.unbind(0)
        nonfinite = int((~torch.isfinite(actual)).sum().item())
        bound = 1e-4 + 5e-5 * magnitude
        lower, upper = (expected - bound).to(actual.dtype).float(), (expected + bound).to(actual.dtype).float()
        value = actual.float()
        result.update(nonfinite=nonfinite,
                      mismatches=int(((value < lower) | (value > upper) | ~torch.isfinite(value)).sum().item()))
        if nonfinite:
            result["reason"] = "nonfinite output (including unwritten NaN-prefilled elements)"
        else:
            result["max_abs_error"] = float((value - expected).abs().max().item())
    return result


def require_idle(torch, output, label):
    torch.cuda.synchronize()
    time.sleep(0.4)
    recent = []
    for _ in range(20):
        sample = json.loads(subprocess.check_output(
            ["rocm-smi", "-d", str(GPU_INDEX), "--showuse", "--showmemuse", "--json"], text=True))[f"card{GPU_INDEX}"]
        output.sample(dict(time=time.time(), label=label, **sample))
        recent.append(sample)
        window = recent[-3:]
        if (len(window) == 3 and not any(int(item["GPU use (%)"]) for item in window)
                and len({item["GFX Activity"] for item in window}) == 1):
            return
        time.sleep(0.35)
    raise RuntimeError("External GPU activity detected; preserve partial results and stop")


def summarize(output, candidates, records, failed, rounds):
    summaries = []
    for candidate in candidates:
        values = [row["us"] for row in records if row["name"] == candidate.row["name"]]
        if candidate.key in failed or len(values) != rounds:
            continue
        summary = dict(candidate.row, median_us=statistics.median(values),
                       min_us=min(values), max_us=max(values), samples=len(values))
        output.row("summary", summary)
        summaries.append(summary)
    reference = min((row for row in summaries if row["lib"] != "opus"),
                    key=lambda row: row["median_us"], default=None)
    opus = min((row for row in summaries if row["lib"] == "opus"),
               key=lambda row: row["median_us"], default=None)
    if reference is None or opus is None:
        raise RuntimeError("Shape has no fully measured valid reference or OPUS candidate")
    targets = [row for row in summaries if row["name"] in {"opus_9010", "opus_9011"}]
    target = min(targets, key=lambda row: row["median_us"], default=None)
    choice = {key: opus[key] for key in ("M", "N", "K")}
    choice.update(reference=reference["name"], reference_us=reference["median_us"],
                  opus=opus["name"], opus_us=opus["median_us"],
                  speedup=reference["median_us"] / opus["median_us"],
                  opus_faster=opus["median_us"] < reference["median_us"])
    for row in targets:
        choice[f"{row['name']}_us"] = row["median_us"]
    if target:
        choice.update(best_9010_9011=target["name"], best_9010_9011_us=target["median_us"],
                      target_speedup=reference["median_us"] / target["median_us"],
                      target_faster=target["median_us"] < reference["median_us"])
    output.row("choices", choice)
    print(json.dumps(choice), flush=True)
    return choice


def run(args):
    sources, binaries, references = snapshot(args.jit_dir)
    selected, sweep_candidates, sweep_input = None, None, None
    if args.final:
        selected, sweep_candidates, sweep_input = load_sweep(args.final, references, args.shape_list)
    manifest = dict(
        schema_version=1, status="running", mode="sweep" if args.sweep else "final",
        start_time=time.time(), pid=os.getpid(), argv=sys.argv,
        gpu=GPU_INDEX, gpu_uuid=GPU_UUID, gpu_bus=GPU_BUS, rounds=args.rounds, shapes=args.shape_list,
        shapes_file=str(args.shapes_path), shapes_sha256=sha256(args.shapes_path),
        jit_dir=str(args.jit_dir), harness_sha256=sha256(Path(__file__)),
        source_sha256=sources, binary_sha256=binaries, reference_sha256=references,
        sweep_input=sweep_input, num_warmup=5, num_iters=51, num_rotate_args=0,
        timing="Original run_perftest profiler GPU time, automatic buffer rotation; each raw row is one returned perftest measurement",
        timing_scope="Preprocessing, B shuffle, exact scale decoding and FP32 reference excluded; backend internal transforms included; CPU launch overhead excluded",
        numerical_contract="Original tune.compare_outputs; error == 0 for every element; FP32 interval 1e-4 + 5e-5 * sum(abs(A_i*B_i)), endpoints rounded to BF16; no relaxed tolerance",
        inputs="One deterministic FP8/E8M0 dataset per shape; shared shuffled B; OPUS native E8M0, CK/CKTile/ASM exact FP32 scale decodes",
        candidate_policy="Reuse all four MXFP8 tuner task factories, all BlockPerCu, useSplitK=True; CK/CKTile/OPUS splitK=0, ASM valid aligned partitions through 8",
        selection_policy="All candidates" if args.sweep else "All legal OPUS plus every valid rival with sweep median <= 1.05 * fastest valid rival median",
    )
    output = Output(args.prefix, manifest)
    checks, timing_count, failure_count, choices = 0, 0, 0, []
    try:
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(key, None)
        os.environ.update(ROCR_VISIBLE_DEVICES=GPU_UUID, AITER_AOT_IMPORT="1", AITER_REBUILD="0",
                          AITER_JIT_DIR=str(args.jit_dir), GPU_ARCHS="gfx950", CU_NUM="256", OMP_NUM_THREADS="2",
                          AITER_LOG_MORE="0", AITER_SMI_MONITOR="0")
        sys.path.insert(0, str(ROOT))
        import torch
        import aiter
        from aiter.utility import dtypes
        from aiter.ops import gemm_op_a8w8

        aiter.dtypes = dtypes
        for name in ("gemm_a8w8_blockscale_bpreshuffle_tune",
                     "gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
                     "gemm_a8w8_blockscale_bpreshuffle_asm"):
            setattr(aiter, name, getattr(gemm_op_a8w8, name))
        tune = load_tune()
        from aiter.test_common import run_perftest
        from aiter.jit import core

        # A missing or incompatible module must fail, rather than silently build
        # another binary while candidate measurements are in progress.
        def forbid_build(*_args, **_kwargs):
            raise RuntimeError("Harness does not build modules; prepare --jit-dir separately")

        core.build_module = forbid_build
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one GPU after UUID binding")
        properties = torch.cuda.get_device_properties(0)
        if (properties.pci_bus_id != GPU_BUS or properties.pci_domain_id != 0
                or properties.pci_device_id != 0 or properties.multi_processor_count != 256
                or not properties.gcnArchName.startswith("gfx950")):
            raise RuntimeError(f"Unexpected GPU mapping: {properties}")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest.update(gpu_name=properties.name, gpu_properties=str(properties),
                        torch_version=torch.__version__, hip_version=torch.version.hip)
        # Load pybind modules eagerly and verify that Python resolved this JIT directory.
        for module_name in MODULES:
            if module_name.endswith("_asm"):
                continue  # ASM is a ctypes module and has no PyInit symbol.
            module = core.get_module(module_name)
            if Path(module.__file__).resolve() != (args.jit_dir / f"{module_name}.so").resolve():
                raise RuntimeError(f"Module loaded from unexpected path: {module.__file__}")
        output.save()
        require_idle(torch, output, "start")
        tuner = tune.OpusMxscaleBpreshuffleTuner()
        for shape_index, shape in enumerate(args.shape_list):
            candidates = candidates_for_shape(tune, tuner, shape)
            if args.final:
                previous = {candidate_identity(row)
                            for row in sweep_candidates if
                            tuple(int(row[key]) for key in ("M", "N", "K")) == shape
                            and row["lib"] != "opus"}
                current = {candidate.key for candidate in candidates if candidate.row["lib"] != "opus"}
                if previous != current:
                    raise RuntimeError(f"Rival candidate registry changed since sweep: {shape}")
            for candidate in candidates:
                take = args.sweep or candidate.row["lib"] == "opus" or candidate.key in selected[shape]
                output.row("candidates", dict(candidate.row, selected=bool(take), selection_reason=
                           "full sweep" if args.sweep else "all legal OPUS" if candidate.row["lib"] == "opus"
                           else "within 5% of sweep winner" if take else "outside sweep 5% window or invalid in sweep"))
            if args.final:
                candidates = [candidate for candidate in candidates
                              if candidate.row["lib"] == "opus" or candidate.key in selected[shape]]
            data = tune.generate_data(*shape, device="cuda:0")
            reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
            good, failed, records = [], set(), []
            for candidate in candidates:
                data["out"].fill_(float("nan"))
                try:
                    actual = candidate.func(*candidate.arguments(data), **candidate.kwargs)
                    result = check_output(torch, tune, reference, actual)
                except Exception as exc:
                    result = dict(status="failed", error="", reason=f"{type(exc).__name__}: {exc}")
                output.row("correctness", dict(candidate.row, phase="preflight", round=-1, **result))
                checks += 1
                if result["status"] == "passed":
                    good.append(candidate)
                else:
                    failure_count += 1
                    failed.add(candidate.key)
                    # An unsupported host-side candidate may be skipped. A
                    # poisoned device context cannot support a trustworthy run.
                    torch.cuda.synchronize()
            for turn in range(args.rounds):
                require_idle(torch, output, f"before_{shape}_round{turn}")
                active = [candidate for candidate in good if candidate.key not in failed]
                if not active:
                    break
                shift = (shape_index + turn) % len(active)
                order = active[shift:] + active[:shift]
                if turn % 2:
                    order.reverse()
                for position, candidate in enumerate(order):
                    us = None
                    try:
                        actual, us = run_perftest(candidate.func, *candidate.arguments(data),
                                                 num_warmup=5, num_iters=51, num_rotate_args=0,
                                                 **candidate.kwargs)
                        us = float(us)
                        result = check_output(torch, tune, reference, actual)
                        if not math.isfinite(us) or us <= 0:
                            result.update(status="failed", reason=f"invalid timing: {us}")
                    except Exception as exc:
                        result = dict(status="failed", error="", reason=f"{type(exc).__name__}: {exc}")
                    check = dict(candidate.row, phase="timed", round=turn, **result)
                    output.row("correctness", check)
                    checks += 1
                    raw = dict(candidate.row, round=turn, order=position, us=us,
                               **{key: result[key] for key in ("error", "status", "reason")})
                    output.row("raw", raw)
                    timing_count += 1
                    if result["status"] != "passed":
                        failed.add(candidate.key)
                        failure_count += 1
                        torch.cuda.synchronize()
                    else:
                        records.append(raw)
                require_idle(torch, output, f"after_{shape}_round{turn}")
            choices.append(summarize(output, candidates, records, failed, args.rounds))
            manifest.update(completed_shapes=len(choices), correctness_checks=checks,
                            excluded_checks=failure_count, timing_records=timing_count)
            output.save()
            del data, reference, good, candidates, records
            if "actual" in locals():
                del actual
            if "order" in locals():
                del order
            torch.cuda.empty_cache()
        require_idle(torch, output, "end")
        for relative, digest in sources.items():
            if sha256(ROOT / relative) != digest:
                raise RuntimeError(f"Source changed during run: {relative}")
        for path, digest in binaries.items():
            if sha256(path) != digest:
                raise RuntimeError(f"Binary changed during run: {path}")
        manifest.update(status="passed", all_shapes_opus_faster=all(row["opus_faster"] for row in choices),
                        all_shapes_9010_9011_faster=all(row.get("target_faster", False) for row in choices))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(end_time=time.time(), correctness_checks=checks,
                        excluded_checks=failure_count, timing_records=timing_count)
        output.close()
        manifest["output_sha256"] = {kind: sha256(path) for kind, path in output.paths.items() if kind != "run"}
        output.save()
    print(json.dumps({key: manifest[key] for key in
                      ("status", "completed_shapes", "correctness_checks", "excluded_checks",
                       "timing_records", "all_shapes_opus_faster", "all_shapes_9010_9011_faster")}), flush=True)


def main():
    args = parse_args()
    if args.list_shapes:
        writer = csv.writer(sys.stdout)
        writer.writerow(("M", "N", "K"))
        writer.writerows(args.shape_list)
        return
    run(args)


if __name__ == "__main__":
    main()
