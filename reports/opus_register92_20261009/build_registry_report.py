#!/usr/bin/env python3
"""Export the registered configurations using scalar-only Python and saved TUs.

This script does not compile, import a GPU runtime, inspect GPUs, or load a
library. Generated files are documentation; production selection uses common.py
and the new-variant Python table.
"""

import csv
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import re
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class NoGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "triton", "flydsl"}:
            raise RuntimeError(f"GPU runtime import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, NoGpuImports())
sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
from opus_gemm_common import (  # noqa: E402
    A8W8_BPRESHUFFLE_FAMILY_BY_KID,
    A8W8_BPRESHUFFLE_LEGACY_KIDS,
    A8W8_BPRESHUFFLE_PIN_AGPR_KIDS,
    A8W8_BPRESHUFFLE_TUNING_KIDS,
    a8w8_mxscale_bpreshuffle_candidate_kids,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    metadata = json.loads((HERE / "codegen/all105_metadata.json").read_text())
    rows = []
    for saved in metadata:
        kid, name = saved["kid"], saved["name"]
        instance = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        variant = instance.bpreshuffle_variant
        impl = HERE / "codegen/impl" / f"{name}.cuh"
        device = HERE / "codegen/instances" / f"{name}_Cbf16_t.device.cu"
        assert sha(impl) == saved["generated_impl_sha256"]
        assert sha(device) == saved["generated_device_tu_sha256"]
        assert instance.name == name
        text, device_text = impl.read_text(), device.read_text()
        kernels = re.findall(r"template __global__ void\s+(\w+)", device_text)
        aliases = re.findall(r"using\s+(\w+)\s*=\s*([^;]+);", text)
        primary = dict(aliases)[name + "_Traits"].strip()
        pipeline = variant.pipeline_header if variant else re.search(
            r'#include "([^"]*opus_gemm_pipeline_[^"]+)"', text).group(1)
        traits_header = re.search(r'#ifdef OPUS_FUSED_HOST_TU\s*#include "([^"]+)"', text).group(1)
        sfa_align = re.findall(r"reinterpret_cast<uintptr_t>\(x_scale.data_ptr\(\)\) % (\d+)", text)
        c_align = re.findall(r"output % (\d+)", text)
        family = A8W8_BPRESHUFFLE_FAMILY_BY_KID[kid]
        rows.append(dict(
            kid=kid,
            role="new_tuning" if variant else "retained_tuning" if kid in A8W8_BPRESHUFFLE_TUNING_KIDS else "legacy_internal",
            family=family,
            label=variant.label if variant else instance.name_tag or "default",
            compiler="pin_clang24" if kid in A8W8_BPRESHUFFLE_PIN_AGPR_KIDS else "baseline_clang23",
            pipeline_header=pipeline,
            traits_header=traits_header,
            primary_kernel=kernels[0],
            primary_traits=primary,
            additional_traits=json.dumps([value.strip() for alias, value in aliases if alias != name + "_Traits"]),
            tile_m=instance.B_M, tile_n=instance.B_N, tile_k=instance.B_K,
            block_size=instance.BLOCK_SIZE,
            m_alignment=instance.m_align,
            n_alignment=instance.GROUP_N if instance.bpreshuffle_pad_n else math.lcm(instance.GROUP_N, instance.B_N),
            max_m=instance.max_m,
            max_k=instance.max_k,
            fixed_k=instance.bpreshuffle_fixed_k,
            split_k=instance.bpreshuffle_split_k,
            workspace_dtype=instance.splitk_workspace_dtype or "none",
            sfa_pointer_alignment=int(sfa_align[0]) if sfa_align else 1,
            c_pointer_alignment=int(c_align[0]),
            max_tensor_bytes=instance.max_tensor_bytes,
            large_output_only=family == "large_output",
            output_dtype="bf16",
            internal_dispatch=json.dumps(instance.bpreshuffle_dispatch),
            legacy_parent=A8W8_BPRESHUFFLE_LEGACY_KIDS.get(kid),
            name=name,
            generated_impl_sha256=saved["generated_impl_sha256"],
            generated_device_tu_sha256=saved["generated_device_tu_sha256"],
        ))
    public = [row for row in rows if row["role"] != "legacy_internal"]
    assert len(rows) == 105 and len(public) == 92
    assert sum(row["role"] == "new_tuning" for row in rows) == 64
    for filename, selected in (("registry92.csv", public), ("registry105.csv", rows)):
        with (HERE / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    families = {}
    for row in public:
        group = families.setdefault(row["family"], dict(count=0, kids=[], primary_kernel_templates=[]))
        group["count"] += 1
        group["kids"].append(row["kid"])
        if row["primary_kernel"] not in group["primary_kernel_templates"]:
            group["primary_kernel_templates"].append(row["primary_kernel"])
    formal = ROOT / "aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv"
    counts = []
    with formal.open(newline="") as stream:
        for row in csv.DictReader(stream):
            counts.append(len(a8w8_mxscale_bpreshuffle_candidate_kids(
                row["gfx"], int(row["M"]), int(row["N"]), int(row["K"]))))
    assert len(counts) == 745
    groups = json.loads((HERE / "header_promotion/template_groups92.json").read_text())
    assert len({row["primary_kernel"] for row in public}) == groups["distinct_main_kernel_templates"] == 18
    summary = dict(
        status="all92_registered_scalar_catalog_exported_gpu_validation_pending",
        tuning_candidates=92, total_registered_ids=105,
        retained_tuning_candidates=28, new_tuning_candidates=64,
        earlier_all_series_variants=62, earlier_shortk_variants=2, legacy_internal_ids=13,
        primary_gemm_templates=18, shared_reducer_templates=1,
        retained_primary_gemm_templates=10, new_primary_gemm_templates=8,
        new_grouped_traits_headers=4,
        compiler_counts_public=dict(pin_clang24=15, baseline_clang23=77),
        compiler_counts_all=dict(pin_clang24=15, baseline_clang23=90),
        families=dict(sorted(families.items())),
        selection_745_scalar_only=dict(shapes=len(counts), total_eligible_cases=sum(counts),
            minimum=min(counts), maximum=max(counts), median=statistics.median(counts),
            measured=False, source_sha256=sha(formal)),
        runtime_sources=dict(new64="csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py",
            original41_and_combined_registry="csrc/opus_gemm/opus_gemm_common.py"),
        provenance_json="csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_promoted_variants.json",
        provenance_json_runtime_use=False,
        gpu_queries=0, gpu_runtime_imports=0, library_loads=0, kernel_launches=0,
        numerical_validation="not_run_gpu_stopped", performance_validation="not_run_gpu_stopped",
        catalog_sha256={name: sha(HERE / name) for name in ("registry92.csv", "registry105.csv")},
        source_sha256={name: sha(ROOT / name) for name in (
            "csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py",
            "csrc/opus_gemm/opus_gemm_common.py",
            "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
            "csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py")},
        generator_sha256=sha(__file__),
    )
    write_json(HERE / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "status", "tuning_candidates", "total_registered_ids", "primary_gemm_templates", "selection_745_scalar_only")}, indent=2))


if __name__ == "__main__":
    main()
