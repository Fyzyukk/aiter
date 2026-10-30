#!/usr/bin/env python3
"""Compare saved K21504 CO artifacts without importing or initializing a GPU.

This uses only the Python standard library. It compares the installed historical
build snapshot with current CO files and compares the cached host launchers.
An equal snapshot is evidence about saved files, not a receipt identifying the
CO image that was actually loaded for the historical timing measurement.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess


SYMBOLS = {
    693: (
        "opus_gemm_gfx950_b_vgpr_overlap_co_256x128x256x128_1x4_"
        "16x16x128_1x128x128_p4_bd1_k21504_pinA0B0C1"
    ),
    501: (
        "opus_gemm_gfx950_b_vgpr_co_256x128x256x128_1x4_"
        "16x16x128_1x128x128_p4_bd1_k21504_pinA0B0C1"
    ),
}
SECTION_NAMES = (".text", ".rodata", ".note")
RESOURCE_FIELDS = (
    "vgpr_count", "sgpr_count", "agpr_count", "vgpr_spill_count",
    "sgpr_spill_count", "private_segment_fixed_size",
    "group_segment_fixed_size", "pin_mask_verified",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_evidence(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data)}


def read_elf_sections(path):
    """Extract file-backed sections from an ELF64 little-endian CO image."""
    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
        raise ValueError(f"Expected ELF64 little-endian image: {path}")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, section_count, names_index = struct.unpack_from("<HHH", data, 58)
    if entry_size < 64 or section_count == 0 or names_index >= section_count:
        raise ValueError(f"Unsupported ELF section table: {path}")
    if section_offset + entry_size * section_count > len(data):
        raise ValueError(f"Truncated ELF section table: {path}")
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + index * entry_size)
        for index in range(section_count)
    ]
    names_header = headers[names_index]
    names_start, names_size = names_header[4], names_header[5]
    if names_start + names_size > len(data):
        raise ValueError(f"Truncated ELF section names: {path}")
    names = data[names_start:names_start + names_size]
    sections = {}
    for header in headers:
        name_offset, section_type = header[0], header[1]
        if name_offset >= len(names):
            raise ValueError(f"Invalid ELF section name offset: {path}")
        name_end = names.find(b"\0", name_offset)
        if name_end < 0:
            raise ValueError(f"Unterminated ELF section name: {path}")
        name = names[name_offset:name_end].decode("ascii")
        start, size = header[4], header[5]
        if section_type == 8:  # SHT_NOBITS has no bytes stored in the file.
            continue
        if start + size > len(data):
            raise ValueError(f"Truncated ELF section {name}: {path}")
        sections[name] = data[start:start + size]
    return sections


def kernel_metadata(report, symbol):
    records = [row for row in report["kernels"] if row.get("symbol") == symbol]
    if len(records) != 1:
        raise ValueError(f"Expected one build metadata record for {symbol}")
    record = records[0]
    return {
        "kid": record["kid"], "status": record["status"],
        "symbol": symbol,
        "resources": {field: record.get(field) for field in RESOURCE_FIELDS},
    }


def compare_kernel(kid, symbol, snapshot, historical, current_run, repo,
                   old_report, current_report):
    old_image = snapshot / "co" / (symbol + ".co")
    current_image = repo / "csrc/opus_gemm/gen_co/gfx950" / (symbol + ".co")
    old_sections = read_elf_sections(old_image)
    current_sections = read_elf_sections(current_image)
    sections = {}
    for name in SECTION_NAMES:
        if name not in old_sections or name not in current_sections:
            raise ValueError(f"Missing required {name} in image comparison for kid {kid}")
        old_data, current_data = old_sections[name], current_sections[name]
        sections[name] = {
            "equal": old_data == current_data,
            "historical_bytes": len(old_data), "current_bytes": len(current_data),
            "historical_sha256": sha256(old_data),
            "current_sha256": sha256(current_data),
        }
    launcher_relative = Path("aiter_jit/build/module_deepgemm_opus/blob/impl") / (symbol + ".cuh")
    old_launcher = historical / launcher_relative
    current_launcher = current_run / launcher_relative
    old_metadata = kernel_metadata(old_report, symbol)
    current_metadata = kernel_metadata(current_report, symbol)
    return {
        "current_kid": kid, "symbol": symbol,
        "historical_build_local_kid": old_metadata["kid"],
        "historical_image": file_evidence(old_image),
        "current_image": file_evidence(current_image),
        "whole_elf_equal": old_image.read_bytes() == current_image.read_bytes(),
        "sections": sections,
        "device_sections_equal": all(row["equal"] for row in sections.values()),
        "launcher": {
            "equal": old_launcher.read_bytes() == current_launcher.read_bytes(),
            "historical": file_evidence(old_launcher),
            "current": file_evidence(current_launcher),
        },
        "metadata": {
            "resources_equal": old_metadata["resources"] == current_metadata["resources"],
            "historical": old_metadata, "current": current_metadata,
        },
    }


def build_report(workspace):
    repo = workspace / "aiter"
    work = workspace / "retune_dsv4_20261028"
    snapshot = work / "pin_only_co_rebuild_20261028T160234Z"
    historical = work / "tune_nonpin_20261028T162327Z_8gpu"
    current_run = work / "gpu_stage3_20261029T161551Z/stage4_full421"
    old_report_path = snapshot / "build_report.json"
    current_report_path = repo / "csrc/opus_gemm/gen_co/gfx950/build_info.json"
    old_report = json.loads(old_report_path.read_text())
    current_report = json.loads(current_report_path.read_text())
    old_status = json.loads((snapshot / "status.json").read_text())
    tune_status = json.loads((historical / "status.json").read_text())
    with (historical / "comparison.csv").open(newline="") as stream:
        timing_records = [
            row for row in csv.DictReader(stream)
            if tuple(int(row[key]) for key in ("M", "N", "K")) == (3072, 7168, 21504)
        ]
    if len(timing_records) != 1:
        raise ValueError("Expected one historical timing record for (3072,7168,21504)")
    historical_receipts = sorted(str(path) for path in historical.rglob("*receipt*"))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    comparisons = [
        compare_kernel(kid, symbol, snapshot, historical, current_run, repo,
                       old_report, current_report)
        for kid, symbol in SYMBOLS.items()
    ]
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "CPU file inspection only; standard library; no GPU imports or execution",
        "current_repository_head": head,
        "historical_build": {
            "snapshot": str(snapshot), "status_file": str(snapshot / "status.json"),
            "installed_at_utc": old_status.get("installed_at_utc"),
            "installed_co_dir": old_status.get("installed_co_dir"),
            "build_metadata": file_evidence(old_report_path),
            "llvm_bin": old_report.get("llvm_bin"),
        },
        "historical_tune": {
            "directory": str(historical), "status_file": str(historical / "status.json"),
            "started_at_utc": tune_status.get("started_at_utc"),
            "physical_gpus": tune_status.get("physical_gpus"),
            "timing_file": str(historical / "comparison.csv"),
            "timing_record": timing_records[0],
        },
        "current_build_metadata": file_evidence(current_report_path),
        "comparisons": comparisons,
        "all_device_sections_equal": all(row["device_sections_equal"] for row in comparisons),
        "all_launchers_equal": all(row["launcher"]["equal"] for row in comparisons),
        "all_resource_metadata_equal": all(row["metadata"]["resources_equal"] for row in comparisons),
        "historical_runtime_identity": {
            "co_load_receipt_available": False,
            "saved_receipt_paths": historical_receipts,
            "limitation": (
                "The historical tune retained a JIT compiled-kids receipt, but no receipt "
                "recording the CO image SHA actually loaded for each timing. The CO loader "
                "allows OPUS_GEN_CO_DIR to override its image directory, and the historical "
                "status does not preserve that full environment. Equal saved device "
                "sections and host launchers therefore do not strictly authenticate the "
                "image loaded for the 410.7753 us measurement."
            ),
        },
        "interpretation": (
            "The saved historical build and current artifacts have identical compared "
            "device instructions, kernel descriptors, AMDGPU notes, host launchers, and "
            "resource metadata. Whole-ELF SHA differences alone do not establish changed "
            "device code. This CPU comparison does not identify the cause of timing "
            "changes or establish equal GPU clocks, inputs, or measurement conditions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/fyz"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "longk_artifact_comparison.json",
    )
    args = parser.parse_args()
    result = build_report(args.workspace.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "current_repository_head": result["current_repository_head"],
        "all_device_sections_equal": result["all_device_sections_equal"],
        "all_launchers_equal": result["all_launchers_equal"],
        "all_resource_metadata_equal": result["all_resource_metadata_equal"],
        "historical_co_load_receipt_available": False,
    }, indent=2))


if __name__ == "__main__":
    main()
