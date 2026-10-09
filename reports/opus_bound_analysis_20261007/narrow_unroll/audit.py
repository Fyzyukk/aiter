#!/usr/bin/env python3
"""CPU audit and exact official baseline identity for narrow unroll AB."""
from pathlib import Path
import importlib.util
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compute_metadata", OUT.parent / "audit_current_compute_metadata.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
official_manifest = json.loads((OUT.parent / "official_compute_metadata.json").read_text())
official = {v["name"]: v for p in official_manifest["kernels"]
            if p["parent_id"] in (9023, 9024) for v in p["variants"]}
result = {"cpu_only": True, "gpu_executed": False, "production_modified": False,
          "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "status": "passed", "variants": [],
          "dynamic_frequency_note": "Backward branch regions contain static opcounts. Executed path/tail/refill frequency requires control-flow interpretation and GPU confirmation; full disassembly count is not dynamic count."}
for mode in ("baseline", "candidate"):
    image = OUT / mode / "device.co"
    image.write_bytes(common.module_image(OUT / mode / "experiments.so"))
    metadata, rows = common.summarize_image(image.read_bytes())
    common.narrow_spill_audit(image, rows)
    disassembly = image.with_suffix(".s").read_text()
    for row in rows:
        body = disassembly[disassembly.index("<"+row["name"]+">:"):]
        next_symbol = re.search(r"\n[0-9a-f]+ <", body)
        if next_symbol:
            body = body[:next_symbol.start()]
        instructions = []
        for line in body.splitlines():
            pc = re.search(r"//\s+([0-9A-F]+):", line)
            if pc:
                instructions.append((int(pc[1], 16), line))
        loops = []
        for pc, line in instructions:
            branch = re.search(r"\b(s_(?:c?branch\w*))\s+(\d+)", line)
            if not branch:
                continue
            delta = int(branch[2])
            delta = delta-65536 if delta>=32768 else delta
            target = pc+4+4*delta
            if target < pc:
                region = "\n".join(text for addr,text in instructions if target<=addr<=pc)
                loops.append({"branch_opcode": branch[1], "branch_pc": pc, "target_pc": target,
                              "static_mfma": region.count("v_mfma_scale_f32"),
                              "v_writelane_b32": region.count("v_writelane_b32"),
                              "v_readlane_b32": region.count("v_readlane_b32"),
                              "barriers": region.count("s_barrier")})
        row.update(mode=mode, static_mfma=body.count("v_mfma_scale_f32"),
                   hardware_barriers=body.count("s_barrier"), backward_branch_regions=loops)
        comparison = (official[row["name"]] if mode=="baseline" else
                      next(v for v in result["variants"] if v["mode"]=="baseline" and v["name"]==row["name"]))
        same = {"instructions": row["instruction_sha256"]==comparison["instruction_sha256"],
                "metadata": row["metadata"]==comparison["metadata"],
                "normalized_descriptor": row["descriptor_normalized_sha256"]==comparison["descriptor_normalized_sha256"]}
        row["official_baseline_identical" if mode=="baseline" else "baseline_identical"] = same
        if mode=="baseline" or "7168" in row["demangled"]:
            assert all(same.values()), "Expected exact official baseline/fixed-K identity"
        result["variants"].append(row)
    (OUT / mode / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
(OUT / "isa_comparison.json").write_text(json.dumps(result, indent=2)+"\n")
for row in result["variants"]:
    m=row["metadata"]
    print(json.dumps({"mode": row["mode"], "kernel": row["demangled"],
                      "vgpr": m[".vgpr_count"], "sgpr": m[".sgpr_count"],
                      "sgpr_spill": m[".sgpr_spill_count"], "bytes": row["instruction_bytes"],
                      "static_mfma": row["static_mfma"],
                      "lane_write": row["scalar_spill_isa"]["v_writelane_b32"],
                      "lane_read": row["scalar_spill_isa"]["v_readlane_b32"]}))
