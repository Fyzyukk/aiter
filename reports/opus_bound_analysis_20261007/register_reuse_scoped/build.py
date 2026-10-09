#!/usr/bin/env python3
"""CPU-only isolated alias opt-in build; does not change the tested full experiment."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_manifest = json.loads((OUT / "register_reuse/build_manifest.json").read_text())
    old_audit = json.loads((OUT / "register_reuse/device_audit.json").read_text())
    variants = json.loads((OUT / "register_reuse/variants.json").read_text())["variants"]
    headers = ["opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh",
               "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh",
               "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh", "launch.hip"]
    alias_ids = {}
    aliases = re.findall(r"using\s+(\w+)\s*=\s*opus_gemm_small_register_traits_gfx950<([^;]+)>;",
        (OUT / "register_reuse/baseline" / headers[0]).read_text())
    for alias, args in aliases:
        ints = [int(value.strip()) for value in args.split(",")]
        full = ints + [0] * (8 - len(ints))
        matches = [v["variant"] for v in variants if full == [v["B_M"], v["B_N"], v["T_M"], v["T_N"],
            v["prefetch"], v["wave_k"], v["output"], v["B_cache"]]]
        assert len(matches) == 1, (alias, ints, matches)
        alias_ids[alias] = matches[0]
    assert set(alias_ids.values()) == {9040, 9041, 9042, 9050, 9051, 9052, 9053, 9054}
    builds = []
    for side in ["baseline", "candidate"]:
        folder = HERE / side
        folder.mkdir(parents=True, exist_ok=True)
        for header in headers:
            shutil.copyfile(OUT / "register_reuse/baseline" / header, folder / header)
        if side == "candidate":
            text = (folder / headers[0]).read_text()
            for alias, args in aliases:
                if alias_ids[alias] in {9042, 9053, 9054}:
                    original = f"opus_gemm_small_register_traits_gfx950<{args}>"
                    explicit = args + (", 0" if len(args.split(",")) == 7 else "")
                    replacement = f"opus_gemm_small_register_traits_gfx950<{explicit}, 0, false, true>"
                    assert text.count(original) == 1
                    text = text.replace(original, replacement)
            # Default remains false; other aliases, explicit codegen defaults
            # and production files are not edited.
            assert "bool ReuseBScale = false>" in text
            (folder / headers[0]).write_text(text)
        old = next(value for value in old_manifest["libraries"] if value["mode"] == side)
        old_prefix = str(OUT / "register_reuse" / side)
        command = [part.replace(old_prefix, str(folder)) for part in old["command"]]
        started = time.time()
        result = subprocess.run(command, env=ENV, cwd=ROOT, text=True, capture_output=True)
        (folder / "build.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (side, result.returncode)
        builds.append({"mode": side, "command": command, "returncode": 0,
            "seconds": time.time() - started, "pre_audit_binary_sha256": sha(folder / "experiments.so"),
            "source_sha256": {name: sha(folder / name) for name in headers}})
    # Reuse the already reviewed extractor with HERE rebound to this directory.
    spec = importlib.util.spec_from_file_location("scoped_register_audit", OUT / "register_reuse/build.py")
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    extractor.HERE = HERE
    audited = [extractor.audit(side, variants) for side in ["baseline", "candidate"]]
    reference = {lib["mode"]: {k["variant"]: k for k in lib["kernels"]} for lib in old_audit["libraries"]}
    comparisons = []
    for lib in audited:
        for kernel in lib["kernels"]:
            expected_side = "candidate" if lib["mode"] == "candidate" and kernel["variant"] in {9042, 9053, 9054} else "baseline"
            expected = reference[expected_side][kernel["variant"]]
            equal = kernel["instruction_sha256"] == expected["instruction_sha256"] and kernel["resources"] == expected["resources"]
            comparisons.append({"side": lib["mode"], "variant": kernel["variant"],
                "expected_original_side": expected_side, "instruction_equal": kernel["instruction_sha256"] == expected["instruction_sha256"],
                "resources_equal": kernel["resources"] == expected["resources"],
                "matches_expected": equal})
        next(b for b in builds if b["mode"] == lib["mode"])["binary_sha256"] = lib["binary_sha256"]
    status = "scoped_aliases_match_tested_machine_code" if all(c["matches_expected"] for c in comparisons) else "CPU_built_codegen_diff_requires_new_paired_timing"
    result = {"status": status, "gpu_executed": False, "production_modified": False,
        "opt_in_alias_ids": [9042, 9053, 9054], "default_reuse_b_scale": False,
        "other_aliases_false": [9040, 9041, 9050, 9051, 9052], "libraries": audited,
        "comparisons_to_existing_tested_bundles": comparisons,
        "scope_limit": "Private8-entry traits-copy proof; production parent TUs still require exact device hash/resource comparison after integration. Fixed9070/9072 already true and N48 explicitly false are outside default change."}
    (HERE / "device_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "build_manifest.json").write_text(json.dumps({"status": status, "builds": builds,
        "gpu_executed": False, "production_modified": False, "device_audit_sha256": sha(HERE / "device_audit.json")}, indent=2) + "\n")
    print(json.dumps({"status": status, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
