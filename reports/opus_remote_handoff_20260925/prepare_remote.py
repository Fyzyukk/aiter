#!/usr/bin/env python3
"""Prepare fresh, machine-specific copies of the archived MXFP8 harness.

Uses only the standard library. Does not query a GPU, compile, or run benchmarks.
Run from any directory; output must be a new direct child of this repo's reports/.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
FIXED = ROOT / "reports/opus_fixedk_gpu_20260925"
BASE = ROOT / "reports/opus_local_gap_current_20260925"
KIDS = "{9000, 9010, 9011, 9012, 9020, 9030, 9031, 9032, 9033, 9040, 9041, 9042, 9050, 9051}"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Archived template changed: expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/opus_remote_run")
    parser.add_argument("--gpu-index", type=int, required=True, help="physical rocm-smi card index")
    parser.add_argument("--gpu-uuid", required=True, help="ROCr UUID of that same physical card")
    parser.add_argument("--pci-bus", type=lambda value: int(value, 16), required=True,
                        help="hex bus byte, e.g. 85 for PCI 0000:85:00.0")
    parser.add_argument("--opus-clang-path", type=Path, required=True,
                        help="directory containing the pin-op-dst clang++")
    parser.add_argument("--stock-clang-path", type=Path, default=Path("/opt/rocm/llvm/bin"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.parent != ROOT / "reports" or output.exists():
        parser.error("--output-dir must be a new direct child of this repo's reports/")
    if args.gpu_index < 0 or not 0 <= args.pci_bus <= 255:
        parser.error("invalid physical GPU index or PCI bus")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]+", args.gpu_uuid):
        parser.error("--gpu-uuid must be a ROCr GPU-<hex> UUID")

    sources = {}

    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return data.decode()

    files = {name: read(FIXED / name) for name in
             ("benchmark.py", "bootstrap.py", "tune_adapter.py", "validate.py")}
    # This baseline builder compiles EVERY backend from source. The later fixed-K
    # builder copies old-machine .so files, so it is intentionally not the template.
    build = read(BASE / "build.py")
    build = replace_once(build, "{9000, 9010, 9011, 9012, 9020}", KIDS)
    build = replace_once(build, 'STOCK = "/opt/rocm/llvm/bin"',
                         f"STOCK = {str(args.stock_clang_path.resolve())!r}")
    build = replace_once(build, 'OPUS = "/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin"',
                         f"OPUS = {str(args.opus_clang_path.resolve())!r}")
    build = replace_once(build, 'ROCR_VISIBLE_DEVICES="GPU-5ff36708541c8ec0"',
                         f"ROCR_VISIBLE_DEVICES={args.gpu_uuid!r}")
    files["build.py"] = build

    benchmark, count = re.subn(
        r"GPU_MAP = \{\n.*?\n\}",
        f"GPU_MAP = {{{args.gpu_index}: ({args.gpu_uuid!r}, {args.pci_bus})}}",
        files["benchmark.py"], count=1, flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Archived benchmark GPU_MAP changed")
    benchmark = replace_once(benchmark, 'GPU_INDEX = int(os.environ["OPUS_TUNE_GPU_INDEX"])',
                             f"GPU_INDEX = {args.gpu_index}")
    files["benchmark.py"] = benchmark
    validate = replace_once(files["validate.py"], 'os.environ["OPUS_TUNE_GPU_INDEX"] = "7"',
                            f'os.environ["OPUS_TUNE_GPU_INDEX"] = "{args.gpu_index}"')
    files["validate.py"] = replace_once(validate, 'gpu=7, gpu_uuid=GPU_UUID',
                                         f'gpu={args.gpu_index}, gpu_uuid=GPU_UUID')
    for name in ("first25.csv", "second8.csv", "shapes33.csv"):
        files[name] = read(FIXED / name)
    files["shapes87.csv"] = read(ROOT / "reports/opus_9030_targets87_20260925/shapes.csv")
    files["shapes295.csv"] = read(BASE / "shapes.csv")
    for name, source in files.items():
        if name.endswith(".py"):
            ast.parse(source, filename=name)
    files["remote_config.json"] = json.dumps({
        "gpu_index": args.gpu_index, "gpu_uuid": args.gpu_uuid,
        "pci_bus": args.pci_bus, "expected_arch": "gfx950", "expected_cu": 256,
        "pci_domain": 0, "pci_device": 0,
        "opus_clang_path": str(args.opus_clang_path.resolve()),
        "stock_clang_path": str(args.stock_clang_path.resolve()),
        "source_sha256": sources,
        "note": "Prepared only; build, GPU correctness and performance must run on the target server.",
    }, indent=2) + "\n"
    output.mkdir()
    for name, source in files.items():
        (output / name).write_text(source)
    print(f"Prepared {output}; no GPU access or compilation performed.")


if __name__ == "__main__":
    main()
