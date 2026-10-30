#!/usr/bin/env python3
"""Temporary-file checks for sparse-preserving CLI generation; no GPU/build."""

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
GENERATOR = ROOT / "csrc/opus_gemm/gen_co/gen_co_kernels_gfx950.py"
MANIFEST = ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json"
COMMON = ROOT / "csrc/opus_gemm/opus_gemm_common.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(output, *extra):
    return subprocess.run([sys.executable, str(GENERATOR), "--output", str(output), *extra],
                          cwd=ROOT, capture_output=True, text=True)


def main():
    before = {str(path): sha(path) for path in (MANIFEST, COMMON)}
    existing = json.loads(MANIFEST.read_text())
    base = copy.deepcopy(existing)
    base["gfx950"] = [row for row in base["gfx950"] if row["kid"] < 100000]
    base["image_suffixes"] = {str(row["kid"]): base["image_suffixes"][str(row["kid"])] for row in base["gfx950"]}
    for name in list(base):
        if name not in ("_generation", "defaults", "device_flag_sets", "image_suffixes", "gfx950"):
            base.pop(name)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="manifest_generator_cpu_", dir=LOCAL) as temporary:
        temporary = Path(temporary)
        for label, fixture in (("current802", existing), ("original780", base)):
            output = temporary / (label + ".json")
            output.write_text(json.dumps(fixture, indent=2) + "\n")
            process = invoke(output)
            assert process.returncode == 0, process.stderr
            assert json.loads(output.read_text()) == fixture
            checks[label + "_complete_entries_suffixes_metadata_preserved"] = True
        fresh = temporary / "fresh.json"
        process = invoke(fresh)
        assert process.returncode == 0, process.stderr
        assert json.loads(fresh.read_text()) == base
        checks["fresh_output_remains_original_pure780_generation"] = True
        count_output = temporary / "count_only.json"
        count_output.write_text(json.dumps(existing, indent=2) + "\n")
        count_before = sha(count_output)
        process = invoke(count_output, "--count-only")
        assert process.returncode == 0 and sha(count_output) == count_before
        checks["count_only_never_writes_existing_manifest"] = True
        extra_metadata = copy.deepcopy(existing)
        extra_metadata["_future_provenance"] = {"raw_paths": ["round1", "round2"], "pending": True}
        extra_metadata["device_flag_sets"]["flags_unused_extension"] = ["fixture_extra"]
        extra_output = temporary / "extra_metadata.json"
        extra_output.write_text(json.dumps(extra_metadata, indent=2) + "\n")
        process = invoke(extra_output)
        assert process.returncode == 0 and json.loads(extra_output.read_text()) == extra_metadata
        checks["additional_extension_flag_sets_and_unknown_metadata_preserved"] = True

        def rejected(label, mutate):
            fixture = copy.deepcopy(existing)
            mutate(fixture)
            output = temporary / (label + ".json")
            output.write_text(json.dumps(fixture, indent=2) + "\n")
            original = sha(output)
            process = invoke(output)
            assert process.returncode != 0 and "Cannot preserve existing manifest" in process.stderr
            assert sha(output) == original, "Rejected CLI changed its output"
            checks[label + "_rejected_without_writing"] = True

        rejected("base_traits_changed", lambda doc: doc["gfx950"][0]["traits"]["args"].__setitem__(0, "32"))
        rejected("base_scope_changed", lambda doc: doc["gfx950"][0].__setitem__("supported_shapes", [[8192, 7168, 7168]]))
        rejected("base_defaults_changed", lambda doc: doc["defaults"].__setitem__("tag", "changed"))
        rejected("base_flags_changed", lambda doc: doc["device_flag_sets"]["flags_00"].append("changed"))
        rejected("base_image_name_changed", lambda doc: doc["image_suffixes"].__setitem__("0", "changed"))
        rejected("sparse_id_moved_into_JIT_range", lambda doc: doc["gfx950"][-1].__setitem__("kid", 800))
        rejected("frozen_JIT_base_shifted", lambda doc: doc.__setitem__("jit_kid_base", 802))
        rejected("duplicate_sparse_image_symbol", lambda doc: doc["image_suffixes"].__setitem__("100001", doc["image_suffixes"]["100000"]))
        rejected("dangling_image_suffix", lambda doc: doc["image_suffixes"].__setitem__("199999", "missing"))
    old_source = subprocess.run(["git", "show", "HEAD:csrc/opus_gemm/gen_co/gen_co_kernels_gfx950.py"],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for name in ("generate_document", "validate_document"):
        def node(source):
            return next(item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name)
        assert ast.dump(node(old_source), include_attributes=False) == ast.dump(node(GENERATOR.read_text()), include_attributes=False)
        checks[name + "_pure_behavior_code_unchanged"] = True
    assert before == {str(path): sha(path) for path in (MANIFEST, COMMON)}
    report = {"cpu_only": True, "gpu_used": False, "build_used": False,
              "public_manifest_common_unchanged": True, "base_co": 780,
              "preserved_sparse_co": len(existing["gfx950"]) - 780, "checks": checks}
    (LOCAL / "co_manifest_generator_cpu_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
