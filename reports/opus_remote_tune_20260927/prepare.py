#!/usr/bin/env python3
"""Prepare a fresh portable main/small/narrow tuning directory.

Discovers physical card identities with rocm-smi and KFD, without torch or GPU
allocations. --gpu-map-json accepts a previously recorded mapping for offline
preparation; launch always verifies the real machine and idle state.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "csrc/opus_gemm/opus_gemm_common.py").is_file())
TEMPLATES = HERE / "templates"
OLD_REGISTERED = [9000, 9010, 9011, 9012, 9020]
RETAINED = {
    "long_epilogue_sync": [13163], "long_runtime": [20000],
    "short_runtime": [20010, 20011], "n224_runtime": [20020],
    "long_runtime_grid": [20100], "n224_runtime_loop": [20124],
    "short_runtime_unified": [20125, 20126], "long_runtime_fused": [20128],
    "short_runtime_group4_cache2": [20131],
}
NEW = {"main": [21000], "small": [21310, 21311], "narrow": [21220, 21221]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/opus_remote_run")
    parser.add_argument("--gpus", required=True, help="comma-separated physical rocm-smi card indices")
    parser.add_argument("--opus-clang-path", type=Path, required=True, help="pin-op-dst clang++ bin directory")
    parser.add_argument("--rocm-path", type=Path, default=Path("/opt/rocm"))
    parser.add_argument("--stock-clang-path", type=Path, help="default: ROCM_PATH/llvm/bin")
    parser.add_argument("--gpu-map-json", type=Path, help='offline mapping: {"0":{"uuid":"GPU-...","pci_bus":"0000:05:00.0"}}')
    parser.add_argument("--max-jobs", type=int, default=12)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.parent != ROOT / "reports" or output.exists():
        parser.error("--output-dir must be a new direct child of this checkout's reports/")
    try:
        gpus = [int(v) for v in args.gpus.split(",")]
        if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
            raise ValueError
    except ValueError:
        parser.error("--gpus must be unique nonnegative physical card indices")
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    sys.path.insert(0, str(TEMPLATES))
    from preflight import discover, pci_parts
    if args.gpu_map_json:
        supplied = json.loads(args.gpu_map_json.read_text())
        devices = {}
        for gpu in gpus:
            row = supplied[str(gpu)]
            pci = row["pci_bus"].lower()
            domain, bus, device, function = pci_parts(pci)
            uuid = row["uuid"]
            if not re.fullmatch(r"GPU-[0-9a-fA-F]{16}", uuid):
                raise ValueError(f"Invalid ROCr UUID for GPU {gpu}: {uuid!r}")
            devices[gpu] = dict(uuid=uuid.lower().replace("gpu-", "GPU-"), pci_bus=pci,
                                domain=domain, bus=bus, device=device, function=function)
    else:
        devices = discover(gpus)
    if len({v["uuid"] for v in devices.values()}) != len(devices):
        raise ValueError("GPU UUIDs must be unique")
    files = {}
    for path in TEMPLATES.glob("*"):
        if path.is_file() and path.suffix in {".py", ".json"}:
            files[path.name] = path.read_bytes()
    libraries = []
    for group, candidates in (("retained", RETAINED), ("current", NEW)):
        for name, ids in candidates.items():
            source = (ROOT / "csrc/opus_gemm/mxfp8_bpreshuffle_retained" / name
                      if group == "retained" else TEMPLATES / "kernels" / name)
            destination = Path("libraries") / name
            paths = [p for p in source.iterdir() if p.suffix in {".cuh", ".hip"} or p.name == "variants.json"]
            if not {"launch.hip", "variants.json"} <= {p.name for p in paths}:
                raise FileNotFoundError(f"Incomplete source library: {source}")
            variants = json.loads((source / "variants.json").read_text())
            if {int(v["id"]) for v in variants} != set(ids):
                raise ValueError(f"Unexpected candidate IDs: {source}")
            for path in paths:
                files[str(destination / path.name)] = path.read_bytes()
            libraries.append(dict(name=name, directory=str(destination), ids=ids, role=group))
    for name in ("shapes295.csv", "pilot62.csv", "shape_groups.csv"):
        files[name] = (HERE / name).read_bytes()
    for path in (HERE / "baseline").glob("*"):
        if path.is_file():
            files[str(Path("baseline") / path.name)] = path.read_bytes()
    config = dict(gpus=devices, opus_clang_path=str(args.opus_clang_path.resolve()),
                  rocm_path=str(args.rocm_path.resolve()), max_jobs=args.max_jobs,
                  stock_clang_path=str((args.stock_clang_path or args.rocm_path / "llvm/bin").resolve()),
                  expected_arch="gfx950", expected_cu=256, identity_source="provided JSON" if args.gpu_map_json else "rocm-smi + KFD")
    experiments = dict(registered_opus_ids=OLD_REGISTERED, libraries=libraries,
                       merged_opus_ids=[9000, 9020, 21000, 21310, 21311, 21220, 21221])
    files["machine_config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    files["experiments.json"] = (json.dumps(experiments, indent=2) + "\n").encode()
    for name, data in files.items():
        if name.endswith(".py"):
            ast.parse(data, filename=name)
    preparation = dict(status="prepared", gpu_executed=False, binaries_copied=False,
        prepared_files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        target_ids=experiments["merged_opus_ids"], old16_ids=OLD_REGISTERED + sum(RETAINED.values(), []),
        note="Compile every backend and private library here, then run a full sweep on this machine.")
    files["preparation.json"] = (json.dumps(preparation, indent=2) + "\n").encode()
    output.mkdir()
    try:
        for name, data in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    except BaseException:
        shutil.rmtree(output)
        raise
    print(f"Prepared {output}; no compilation or GPU allocations. Build with: python {output / 'build.py'}")


if __name__ == "__main__":
    main()
