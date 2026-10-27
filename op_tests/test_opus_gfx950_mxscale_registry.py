# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU coverage for gfx950 MXFP8 local IDs and tuned codegen selection."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from csrc.opus_gemm import opus_gemm_common as registry


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "a8w8_blockscale_bpreshuffle"
TABLE = ROOT / "aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"


def test_same_local_kid_resolves_distinct_gfx950_families():
    for cu in (128, 256):
        mxscale = registry.get_kernel_instance("gfx950", FAMILY, 1, cu_num=cu)
        a16w16 = registry.get_kernel_instance("gfx950", "a16w16", 1, cu_num=cu)
        assert mxscale.kernel_tag == "a8w8_mxscale_gemm_bpreshuffle"
        assert a16w16.kernel_tag == "a16w16"
        assert mxscale.name != a16w16.name
        assert registry.get_kernel_instance("gfx950", FAMILY, 9501, cu_num=cu) is None


def test_tuned_mxscale_ids_match_kernel_names_shapes_and_workspace_contracts():
    with TABLE.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["libtype"] == "opus"]
    assert rows
    for row in rows:
        kid, cu = int(row["kernelId"]), int(row["cu_num"])
        instance = registry.get_kernel_instance(row["gfx"], FAMILY, kid, cu_num=cu)
        assert instance is not None
        assert registry.a8w8_mxscale_bpreshuffle_kernel_name(instance) == row["kernelName"]
        shape = tuple(int(row[key]) for key in ("M", "N", "K"))
        assert registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)
        assert registry.kernel_needs_external_workspace(row["gfx"], FAMILY, kid, cu_num=cu) == bool(
            registry.bpreshuffle_workspace_bytes(instance, *shape)
        )


def test_codegen_keeps_legacy_tuned_names_in_each_cu_namespace(tmp_path):
    with TABLE.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        row = next(row for row in reader if row["libtype"] == "opus" and row["kernelId"] == "1951")
    instance = registry.get_registry(FAMILY, "gfx950", 256)[1951]
    assert row["kernelName"] != instance.name
    tune_file = tmp_path / "tuned.csv"
    with tune_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict(row, cu_num=cu) for cu in (128, 256))
    names = {}
    for kid, co in registry.gfx950_co_kernels_declared.items():
        names.setdefault(co["symbol"], []).append(kid)
    alias_kids = next(kids[:2] for kids in names.values() if len(kids) > 1)
    alias_keys = {
        registry.KernelKey(FAMILY, "gfx950", cu, kid).encode()
        for cu in (128, 256) for kid in alias_kids
    }
    subprocess.run(
        [
            sys.executable, str(ROOT / "csrc/opus_gemm/gen_instances.py"),
            "--working_path", str(tmp_path / "generated"),
            "--targets", "gfx950/128", "gfx950/256",
            "--tune_files", str(tune_file),
            "--extra_kernel_keys", *sorted(alias_keys),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
    )
    keys = set(json.loads((tmp_path / "generated/compiled_kids.json").read_text()))
    assert alias_keys <= keys
    header = (tmp_path / "generated/opus_gemm_a8w8_kid_dispatch.h").read_text()
    for cu in (128, 256):
        assert registry.KernelKey(FAMILY, "gfx950", cu, 1951).encode() in keys
        assert f"{{ {cu}, 1951 }}" in header
    assert instance.name in header
