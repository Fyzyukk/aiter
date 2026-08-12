# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Task2 interface goldens through B6 family validation and documentation.

B2 removes the legacy C++/pybind raw tune symbol. The old Python public name
remains only as a warning compatibility wrapper around the canonical launcher.
B3 gives each A8W8 physical contract its own generated exact-kid dispatch and
keeps empty per-arch bpreshuffle capabilities explicit. B4 removes the generic
``opus_gemm`` C++/pybind/Python mega entry after the family numerical parity
checkpoint.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    HEURISTIC_DEFAULT_KIDS,
    OPUS_KERNEL_TAGS_BY_ARCH_FAMILY,
    OPUS_MANDATORY_A8_KIDS,
    a8w8_kernels_list,
    a8w8_scale_kernels_list,
    gfx1250_clusterlaunch_kernels_list,
    gfx1250_kernels_list,
    gfx1250_splitk_fuse_kernels_list,
    gfx942_a8w8_kernels_list,
    gfx942_nosplit_kernels_list,
    gfx942_splitk_kernels_list,
    get_kernel_instance,
    kernels_list,
)


_ROOT = Path(__file__).resolve().parents[1]
_BPRESHUFFLE_TAG = "a8w8_blockscale_bpreshuffle_singlebuf"


def _instance_arch(instance) -> str:
    return (instance.arch_prefix or "gfx950").lower()


def _parameter_names(callable_) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_).parameters)


def _python_definition_parameter_names(
    source_path: Path, function_name: str
) -> tuple[str, ...]:
    tree = ast.parse(source_path.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    arguments = definition.args
    return tuple(
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    )


def _cpp_parameter_names(source: str, function_name: str) -> tuple[str, ...]:
    declaration = re.search(
        rf"\bvoid\s+{re.escape(function_name)}\s*\((.*?)\)\s*;",
        source,
        flags=re.DOTALL,
    )
    assert declaration is not None, f"missing C++ declaration for {function_name}"
    names = []
    for parameter in declaration.group(1).split(","):
        name = re.search(r"([A-Za-z_]\w*)\s*$", parameter.strip())
        assert name is not None, f"cannot parse {function_name} parameter {parameter!r}"
        names.append(name.group(1))
    return tuple(names)


def _cpp_function_pointer_parameters(source: str, alias: str) -> tuple[str, ...]:
    declaration = re.search(
        rf"\busing\s+{re.escape(alias)}\s*=\s*void\s*"
        rf"\(\s*\*\s*\)\s*\((.*?)\)\s*;",
        source,
        flags=re.DOTALL,
    )
    assert declaration is not None, f"missing function pointer alias {alias}"
    return tuple(
        re.sub(r"\s+", " ", parameter.strip())
        for parameter in declaration.group(1).split(",")
    )


def _pybind_parameter_names(source: str, macro_name: str) -> tuple[str, ...]:
    start = source.index(f"#define {macro_name}")
    end = source.find("\n#define ", start + 1)
    block = source[start:] if end < 0 else source[start:end]
    return tuple(re.findall(r'py::arg\("([^"]+)"\)', block))


def test_a16_python_high_level_and_compat_signature_golden():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")

    assert _parameter_names(gemm.gemm_a16w16_opus) == (
        "A",
        "B",
        "bias",
        "dtype",
        "kernelId",
        "splitK",
        "out",
    )
    high_level = inspect.signature(gemm.gemm_a16w16_opus).parameters
    assert tuple(
        name
        for name, parameter in high_level.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ) == ("kernelId", "splitK", "out")

    assert _parameter_names(gemm.opus_gemm_a16w16_tune) == (
        "XQ",
        "WQ",
        "Y",
        "bias",
        "kernelId",
        "splitK",
        "workspace",
    )
    tune_wrapper = inspect.signature(gemm.opus_gemm_a16w16_tune).parameters
    assert tune_wrapper["workspace"].kind is inspect.Parameter.KEYWORD_ONLY


def test_cpp_pybind_signatures_and_removed_legacy_raw_symbols():
    header = (_ROOT / "csrc/opus_gemm/include/opus_gemm.h").read_text()
    pybind = (_ROOT / "csrc/include/rocm_ops.hpp").read_text()
    registration = (_ROOT / "csrc/pybind/opus_gemm_pybind.cu").read_text()
    implementation = (_ROOT / "csrc/opus_gemm/opus_gemm.cu").read_text()
    expected = {
        "opus_gemm_a8w8_launch": (
            "XQ",
            "WQ",
            "Y",
            "kid",
        ),
        "opus_gemm_a8w8_blockscale_launch": (
            "XQ",
            "WQ",
            "Y",
            "x_scale",
            "w_scale",
            "kid",
        ),
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch": (
            "XQ",
            "WQ",
            "x_scale",
            "w_scale",
            "Y",
            "kid",
        ),
    }
    pybind_macros = {
        "opus_gemm_a8w8_launch": "OPUS_GEMM_A8W8_LAUNCH_PYBIND",
        "opus_gemm_a8w8_blockscale_launch": (
            "OPUS_GEMM_A8W8_BLOCKSCALE_LAUNCH_PYBIND"
        ),
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch": (
            "OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_LAUNCH_PYBIND"
        ),
    }

    for function_name, parameter_names in expected.items():
        assert _cpp_parameter_names(header, function_name) == parameter_names
        assert (
            _pybind_parameter_names(pybind, pybind_macros[function_name])
            == parameter_names
        )

    assert re.search(r"\bvoid\s+opus_gemm\s*\(", header) is None
    assert re.search(r"\bvoid\s+opus_gemm\s*\(", implementation) is None
    assert "OPUS_GEMM_PYBIND" not in pybind
    assert "OPUS_GEMM_PYBIND;" not in registration
    assert 'm.def("opus_gemm"' not in pybind
    assert "OpusScaleKernel" not in implementation
    assert "OpusNoscaleKernel" not in implementation
    assert "opus_dispatch_scale" not in implementation
    assert "opus_dispatch_a8w8" not in implementation

    assert "void opus_gemm_a16w16_tune(" not in header
    assert "void opus_gemm_a16w16_tune(" not in implementation
    assert "OPUS_GEMM_A16W16_TUNE_PYBIND" not in pybind
    assert "OPUS_GEMM_A16W16_TUNE_PYBIND" not in registration
    assert 'm.def("opus_gemm_a16w16_tune"' not in pybind
    assert "void opus_gemm_a8w8_blockscale_bpreshuffle_tune(" not in header
    assert "void opus_gemm_a8w8_blockscale_bpreshuffle_tune(" not in implementation
    assert "OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_TUNE_PYBIND" not in pybind
    assert 'm.def("opus_gemm_a8w8_blockscale_bpreshuffle_tune"' not in pybind


def test_canonical_a16_python_cpp_and_pybind_signatures():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    opus = importlib.import_module("aiter.ops.opus")
    expected = ("XQ", "WQ", "Y", "bias", "workspace", "kid", "split_k")

    assert "opus_gemm_a16w16_launch" in gemm.__all__
    assert "opus_gemm_a16w16_launch" in opus.__all__
    assert callable(opus.opus_gemm_a16w16_launch)

    assert _parameter_names(gemm.opus_gemm_a16w16_launch) == (
        "XQ",
        "WQ",
        "Y",
        "bias",
        "kid",
        "split_k",
        "workspace",
    )
    public = inspect.signature(gemm.opus_gemm_a16w16_launch).parameters
    assert tuple(
        name
        for name, parameter in public.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ) == ("kid", "split_k", "workspace")
    assert public["kid"].default is inspect.Parameter.empty
    assert public["split_k"].default == 0
    assert public["workspace"].default is None

    assert _python_definition_parameter_names(
        _ROOT / "aiter/ops/opus/gemm_op_a16w16.py",
        "_opus_gemm_a16w16_launch_raw",
    ) == expected

    header = (_ROOT / "csrc/opus_gemm/include/opus_gemm.h").read_text()
    pybind = (_ROOT / "csrc/include/rocm_ops.hpp").read_text()
    registration = (_ROOT / "csrc/pybind/opus_gemm_pybind.cu").read_text()
    assert _cpp_parameter_names(header, "opus_gemm_a16w16_launch") == expected
    assert (
        _pybind_parameter_names(pybind, "OPUS_GEMM_A16W16_LAUNCH_PYBIND")
        == expected
    )
    assert "OPUS_GEMM_A16W16_LAUNCH_PYBIND;" in registration
    assert not hasattr(gemm, "_opus_gemm_a16w16_tune_raw")


@pytest.mark.parametrize(
    ("arch", "supported"),
    [("gfx942", True), ("gfx950", True), ("gfx1250", True), ("gfx999", False)],
)
def test_opus_package_exports_supported_arches_and_unsupported_stubs(
    monkeypatch, arch, supported
):
    monkeypatch.setenv("GPU_ARCHS", arch)
    module_name = f"aiter.ops.opus._b7_init_fixture_{arch}"
    init_path = _ROOT / "aiter/ops/opus/__init__.py"
    loader = SourceFileLoader(module_name, str(init_path))
    spec = importlib.util.spec_from_loader(module_name, loader, is_package=False)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    gated_exports = (
        "gemm_a16w16_opus",
        "opus_gemm_a8w8_launch",
        "opus_gemm_a8w8_blockscale_launch",
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
        "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
        "opus_gemm_a16w16_launch",
        "opus_gemm_a16w16_tune",
    )
    assert module.__all__ == [*gated_exports, "opus_gemm_workspace_init"]
    assert module._arch_ok is supported
    assert module._detected_arch == arch
    assert all(callable(getattr(module, name)) for name in module.__all__)

    if supported:
        assert all(
            not (inspect.getdoc(getattr(module, name)) or "").startswith("Stub:")
            for name in gated_exports
        )
        return

    for name in gated_exports:
        with pytest.raises(
            RuntimeError,
            match=rf"{name} requires GPU arch.*detected 'gfx999'",
        ):
            getattr(module, name)()


def test_unsupported_opus_does_not_truncate_top_level_aiter_import():
    # gfx90a is known to the wider AITER package but intentionally unsupported
    # by OPUS, so it isolates the OPUS stub behavior from unknown-arch failures
    # in unrelated build-time helpers.
    env = os.environ.copy()
    env["GPU_ARCHS"] = "gfx90a"
    env.pop("AITER_REBUILD", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import aiter
import aiter.ops.opus as opus

assert opus._arch_ok is False
assert opus._detected_arch == "gfx90a"
assert (aiter.opus_gemm_a16w16_launch.__doc__ or "").startswith("Stub:")
try:
    aiter.opus_gemm_a16w16_launch()
except RuntimeError as exc:
    assert "detected 'gfx90a'" in str(exc)
else:
    raise AssertionError("unsupported OPUS stub did not raise")

post_opus = (
    "rmsnorm2d_fwd_with_add",
    "topk_plain",
    "fused_split_gdr_update",
)
assert all(callable(getattr(aiter, name)) for name in post_opus)
assert hasattr(aiter, "mla")

star_namespace = {}
exec("from aiter import *", star_namespace)
assert all(callable(star_namespace[name]) for name in post_opus)
assert "mla" in star_namespace
""",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("call_style", ["legacy-positionals", "legacy-keywords"])
def test_a16_tune_compat_warns_once_and_calls_canonical(monkeypatch, call_style):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    calls = []

    def canonical(XQ, WQ, Y, bias=None, *, kid, split_k=0, workspace=None):
        calls.append((XQ, WQ, Y, bias, kid, split_k, workspace))
        return Y

    monkeypatch.setattr(gemm, "opus_gemm_a16w16_launch", canonical)
    XQ = torch.empty((1, 16, 128), dtype=torch.bfloat16)
    WQ = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 16, 32), dtype=torch.bfloat16)
    workspace = torch.empty((2, 32, 32), dtype=torch.bfloat16)

    with pytest.warns(DeprecationWarning, match="use opus_gemm_a16w16_launch") as seen:
        if call_style == "legacy-positionals":
            result = gemm.opus_gemm_a16w16_tune(
                XQ, WQ, Y, 20000, 2, workspace=workspace
            )
        else:
            result = gemm.opus_gemm_a16w16_tune(
                XQ,
                WQ,
                Y,
                kernelId=20000,
                splitK=2,
                workspace=workspace,
            )

    assert len(seen) == 1
    assert result is Y
    assert calls == [(XQ, WQ, Y, None, 20000, 2, workspace)]


def test_deepgemm_compat_warns_once_and_calls_canonical(monkeypatch):
    deepgemm = importlib.import_module("aiter.ops.deepgemm")
    calls = []

    def canonical(XQ, WQ, Y, bias=None, *, kid, split_k=0, workspace=None):
        calls.append((XQ, WQ, Y, bias, kid, split_k, workspace))
        return Y

    monkeypatch.setattr(deepgemm, "_opus_launch", canonical)
    XQ = torch.empty((1, 16, 128), dtype=torch.bfloat16)
    WQ = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 16, 32), dtype=torch.bfloat16)

    with pytest.warns(DeprecationWarning, match="has moved") as seen:
        result = deepgemm.opus_gemm_a16w16_tune(XQ, WQ, Y, 300, 0)

    assert len(seen) == 1
    assert result is Y
    assert calls == [(XQ, WQ, Y, None, 300, 0, None)]


def test_production_and_tuner_imports_use_canonical_a16_launch():
    expected_aliases = {
        "aiter/tuned_gemm.py": "_opus_launch",
        "csrc/opus_gemm/opus_gemm_tune.py": "_opus_gemm_a16w16_launch",
        "csrc/gemm_a16w16/gemm_a16w16_tune.py": "_opus_gemm_a16w16_launch",
        "aiter/ops/deepgemm.py": "_opus_launch",
    }
    for relative_path, expected_alias in expected_aliases.items():
        tree = ast.parse((_ROOT / relative_path).read_text())
        imports = {
            (alias.name, alias.asname)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "aiter.ops.opus.gemm_op_a16w16"
            for alias in node.names
        }
        # deepgemm uses a package-relative import; normalize that one explicitly.
        if relative_path == "aiter/ops/deepgemm.py":
            imports = {
                (alias.name, alias.asname)
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "opus.gemm_op_a16w16"
                for alias in node.names
            }
        assert ("opus_gemm_a16w16_launch", expected_alias) in imports
        assert all(name != "opus_gemm_a16w16_tune" for name, _alias in imports)


@pytest.mark.parametrize(
    (
        "arch",
        "cu_num",
        "kid",
        "split_k",
        "M",
        "N",
        "K",
        "actual_kid",
        "allocation_split_k",
        "launch_split_k",
        "workspace_shape",
        "workspace_dtype",
    ),
    [
        pytest.param(
            "gfx950",
            256,
            200,
            3,
            65,
            33,
            512,
            200,
            3,
            3,
            (3, 1, 128, 64),
            torch.float32,
            id="gfx950-workspace",
        ),
        pytest.param(
            "gfx950",
            256,
            300,
            0,
            192,
            64,
            128,
            300,
            1,
            0,
            None,
            None,
            id="gfx950-non-workspace",
        ),
        pytest.param(
            "gfx942",
            304,
            10210,
            3,
            128,
            768,
            4096,
            10200,
            3,
            3,
            (3, 1, 128, 768),
            torch.float32,
            id="gfx942-redirect-workspace",
        ),
        pytest.param(
            "gfx942",
            304,
            10000,
            0,
            128,
            128,
            4096,
            10000,
            1,
            0,
            None,
            None,
            id="gfx942-non-workspace",
        ),
        pytest.param(
            "gfx1250",
            256,
            20000,
            2,
            17,
            33,
            512,
            20000,
            2,
            2,
            (2, 32, 64),
            torch.bfloat16,
            id="gfx1250-two-stage",
        ),
        pytest.param(
            "gfx1250",
            256,
            21003,
            2,
            17,
            64,
            1024,
            21003,
            5,
            5,
            (2, 2, 4, 16, 32),
            torch.bfloat16,
            id="gfx1250-fused",
        ),
    ],
)
def test_compat_and_canonical_explicit_launch_contracts_match(
    monkeypatch,
    arch,
    cu_num,
    kid,
    split_k,
    M,
    N,
    K,
    actual_kid,
    allocation_split_k,
    launch_split_k,
    workspace_shape,
    workspace_dtype,
):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    monkeypatch.setattr(
        gemm, "_device_arch_and_cu", lambda _device: (arch, cu_num)
    )

    selected = []
    real_select = gemm.select_launch_config

    def record_select(**kwargs):
        config = real_select(**kwargs)
        selected.append(config)
        return config

    monkeypatch.setattr(gemm, "select_launch_config", record_select)
    calls = []

    def record_raw(XQ, WQ, Y, bias, workspace, resolved_kid, resolved_split_k):
        calls.append((XQ, WQ, Y, bias, workspace, resolved_kid, resolved_split_k))
        return Y

    monkeypatch.setattr(
        gemm, "_opus_gemm_a16w16_launch_ctypes_raw", record_raw
    )

    XQ = torch.empty((1, M, K), dtype=torch.bfloat16)
    WQ = torch.empty((1, N, K), dtype=torch.bfloat16)
    old_Y = torch.empty((1, M, N), dtype=torch.bfloat16)
    new_Y = torch.empty_like(old_Y)
    with pytest.warns(DeprecationWarning) as seen:
        assert (
            gemm.opus_gemm_a16w16_tune(
                XQ, WQ, old_Y, kernelId=kid, splitK=split_k
            )
            is old_Y
        )
    assert len(seen) == 1
    assert (
        gemm.opus_gemm_a16w16_launch(
            XQ, WQ, new_Y, kid=kid, split_k=split_k
        )
        is new_Y
    )

    assert selected[0] == selected[1]
    assert selected[0].actual_kid == actual_kid
    assert selected[0].allocation_split_k == allocation_split_k
    assert selected[0].launch_split_k == launch_split_k
    old_call, new_call = calls[0], calls[1]
    assert old_call[0] is new_call[0] is XQ
    assert old_call[1] is new_call[1] is WQ
    assert old_call[2] is old_Y
    assert new_call[2] is new_Y
    assert old_call[3] is new_call[3] is None
    assert old_call[5:] == new_call[5:] == (actual_kid, launch_split_k)

    old_workspace, new_workspace = old_call[4], new_call[4]
    if workspace_shape is None:
        assert old_workspace is new_workspace is None
        return

    assert old_workspace is not new_workspace
    for workspace in (old_workspace, new_workspace):
        assert tuple(workspace.shape) == workspace_shape
        assert workspace.dtype == workspace_dtype
        assert workspace.device.type == "cpu"

    caller_workspace = torch.empty_like(old_workspace)
    old_caller_Y = torch.empty_like(old_Y)
    new_caller_Y = torch.empty_like(old_Y)
    with pytest.warns(DeprecationWarning) as seen:
        assert (
            gemm.opus_gemm_a16w16_tune(
                XQ,
                WQ,
                old_caller_Y,
                kernelId=kid,
                splitK=split_k,
                workspace=caller_workspace,
            )
            is old_caller_Y
        )
    assert len(seen) == 1
    assert (
        gemm.opus_gemm_a16w16_launch(
            XQ,
            WQ,
            new_caller_Y,
            kid=kid,
            split_k=split_k,
            workspace=caller_workspace,
        )
        is new_caller_Y
    )
    assert selected[2] == selected[3] == selected[0]
    assert calls[2][4] is caller_workspace
    assert calls[3][4] is caller_workspace


def test_canonical_raw_fake_registration_is_torch_compile_visible():
    from torch._subclasses.fake_tensor import FakeTensorMode

    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    expected = ("XQ", "WQ", "Y", "bias", "workspace", "kid", "split_k")
    schema = torch.ops.aiter._opus_gemm_a16w16_launch_raw.default._schema
    assert tuple(argument.name for argument in schema.arguments) == expected
    assert _parameter_names(
        gemm._gen_opus_gemm_a16w16_launch_fake_tensors
    ) == expected

    mode = FakeTensorMode()
    with mode:
        XQ = torch.empty((1, 16, 128), dtype=torch.bfloat16)
        WQ = torch.empty((1, 32, 128), dtype=torch.bfloat16)
        Y = torch.empty((1, 16, 32), dtype=torch.bfloat16)
        workspace = torch.empty((2, 32, 32), dtype=torch.bfloat16)

        def raw_call(XQ, WQ, Y, workspace):
            return gemm._opus_gemm_a16w16_launch_raw(
                XQ, WQ, Y, None, workspace, 20000, 2
            )

        assert raw_call(XQ, WQ, Y, workspace) is Y
        compiled = torch.compile(raw_call, backend="eager", fullgraph=True)
        result = compiled(XQ, WQ, Y, workspace)
        assert result is Y
        assert result.fake_mode is mode


def test_a8_canonical_raw_fake_registration_is_torch_compile_visible():
    from torch._subclasses.fake_tensor import FakeTensorMode

    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    raw_specs = {
        "_opus_gemm_a8w8_launch_raw": (
            "_gen_opus_gemm_a8w8_launch_fake_tensors",
            ("XQ", "WQ", "Y", "kid"),
        ),
        "_opus_gemm_a8w8_blockscale_launch_raw": (
            "_gen_opus_gemm_a8w8_blockscale_launch_fake_tensors",
            ("XQ", "WQ", "Y", "x_scale", "w_scale", "kid"),
        ),
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw": (
            "_gen_opus_gemm_a8w8_blockscale_bpreshuffle_launch_fake_tensors",
            ("XQ", "WQ", "x_scale", "w_scale", "Y", "kid"),
        ),
    }
    source = _ROOT / "aiter/ops/opus/gemm_op_a8w8.py"
    for raw_name, (fake_name, expected) in raw_specs.items():
        schema = getattr(torch.ops.aiter, raw_name).default._schema
        # compile_ops may prepend an internal dummy Tensor to schemas without
        # optional Tensor arguments. It is not part of the Python/C++ ABI.
        schema_names = tuple(
            argument.name
            for argument in schema.arguments
            if argument.name != "dummy"
        )
        assert schema_names == expected
        assert _python_definition_parameter_names(source, raw_name) == expected
        assert _parameter_names(getattr(a8, fake_name)) == expected

    mode = FakeTensorMode()
    with mode:
        XQ = torch.empty((128, 256))
        WQ = torch.empty((256, 256))
        x_scale = torch.empty((128, 2))
        w_scale = torch.empty((2, 2))
        Y = torch.empty((128, 256), dtype=torch.bfloat16)

        def raw_call(XQ, WQ, x_scale, w_scale, Y):
            return a8._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
                XQ, WQ, x_scale, w_scale, Y, 11000
            )

        assert raw_call(XQ, WQ, x_scale, w_scale, Y) is Y
        compiled = torch.compile(raw_call, backend="eager", fullgraph=True)
        result = compiled(XQ, WQ, x_scale, w_scale, Y)
        assert result is Y
        assert result.fake_mode is mode


def test_python_a8_canonical_and_legacy_signatures():
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    a16_source = (_ROOT / "aiter/ops/opus/gemm_op_a16w16.py").read_text()
    assert "_opus_gemm_bf16_dispatch" not in vars(a16)
    assert "_gen_opus_gemm_bf16_dispatch_fake_tensors" not in vars(a16)
    assert not hasattr(torch.ops.aiter, "_opus_gemm_bf16_dispatch")
    assert "group_layout" not in a16_source

    source = _ROOT / "aiter/ops/opus/gemm_op_a8w8.py"
    assert _python_definition_parameter_names(
        source, "_opus_gemm_a8w8_launch_raw"
    ) == ("XQ", "WQ", "Y", "kid")
    assert _python_definition_parameter_names(
        source, "_opus_gemm_a8w8_blockscale_launch_raw"
    ) == ("XQ", "WQ", "Y", "x_scale", "w_scale", "kid")
    assert _python_definition_parameter_names(
        source, "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw"
    ) == ("XQ", "WQ", "x_scale", "w_scale", "Y", "kid")

    assert _parameter_names(a8.opus_gemm_a8w8_launch) == (
        "XQ", "WQ", "Y", "kid"
    )
    assert _parameter_names(a8.opus_gemm_a8w8_blockscale_launch) == (
        "XQ", "WQ", "Y", "x_scale", "w_scale", "kid"
    )
    assert _parameter_names(
        a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch
    ) == ("XQ", "WQ", "x_scale", "w_scale", "Y", "kid")
    canonical = inspect.signature(
        a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch
    ).parameters
    assert canonical["kid"].default is None
    assert canonical["kid"].kind is inspect.Parameter.KEYWORD_ONLY

    assert _parameter_names(a8.opus_gemm_a8w8_blockscale_bpreshuffle_tune) == (
        "XQ",
        "WQ",
        "x_scale",
        "w_scale",
        "Y",
        "kernelId",
    )
    public = inspect.signature(
        a8.opus_gemm_a8w8_blockscale_bpreshuffle_tune
    ).parameters
    assert public["Y"].default is None
    assert public["kernelId"].default == 11000
    assert "_opus_gemm_a8w8_blockscale_bpreshuffle_tune_raw" not in vars(a8)


def test_a8_python_wrappers_pass_only_resolved_exact_kids(monkeypatch):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    monkeypatch.setattr(a8, "_device_arch", lambda _device: "gfx950")
    calls = []
    monkeypatch.setattr(
        a8,
        "_opus_gemm_a8w8_launch_raw",
        lambda *args: calls.append(("noscale", args)),
    )
    monkeypatch.setattr(
        a8,
        "_opus_gemm_a8w8_blockscale_launch_raw",
        lambda *args: calls.append(("blockscale", args)),
    )

    XQ = torch.empty((1, 256, 256))
    WQ = torch.empty((1, 256, 256))
    Y = torch.empty((1, 256, 256), dtype=torch.float32)
    x_scale = torch.empty((1, 256, 2))
    w_scale = torch.empty((1, 2, 2))

    assert a8.opus_gemm_a8w8_launch(XQ, WQ, Y) is Y
    assert a8.opus_gemm_a8w8_blockscale_launch(
        XQ, WQ, Y, x_scale, w_scale
    ) is Y
    assert calls[0][0] == "noscale" and calls[0][1][-1] == 2
    assert calls[1][0] == "blockscale" and calls[1][1][-1] == 1

    with pytest.raises(TypeError):
        a8.opus_gemm_a8w8_blockscale_launch(XQ, WQ, Y, x_scale)
    with pytest.raises(TypeError, match="w_scale"):
        a8.opus_gemm_a8w8_blockscale_launch(
            XQ, WQ, Y, x_scale, None  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="no registered OPUS kernel"):
        a8.opus_gemm_a8w8_launch(XQ, WQ, Y, kid=1)


def test_a8_device_arch_cache_is_scoped_by_explicit_device(monkeypatch):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    monkeypatch.setattr(a8, "_DEVICE_ARCH_CACHE", {})
    current_device = [0]
    property_reads = []

    class Properties:
        def __init__(self, arch):
            self.gcnArchName = arch

    arches = {
        torch.device("cuda", 0): "gfx942:sramecc+:xnack-",
        torch.device("cuda", 1): "gfx950:sramecc+:xnack-",
    }

    def get_device_properties(device):
        explicit = torch.device(device)
        property_reads.append(explicit)
        return Properties(arches[explicit])

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device[0])

    assert a8._device_arch(torch.device("cuda", 0)) == "gfx942"
    assert a8._device_arch(torch.device("cuda", 0)) == "gfx942"
    assert a8._device_arch(torch.device("cuda")) == "gfx942"

    current_device[0] = 1
    assert a8._device_arch(torch.device("cuda")) == "gfx950"
    assert a8._device_arch(torch.device("cuda", 1)) == "gfx950"
    assert property_reads == [torch.device("cuda", 0), torch.device("cuda", 1)]


def test_a8_registered_kid_cache_only_memoizes_success(monkeypatch, request):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    a8._require_registered_kid_cached.cache_clear()
    request.addfinalizer(a8._require_registered_kid_cached.cache_clear)
    registry_reads = []

    def get_kernel_instance(arch, family, kid, output_dtype):
        key = (arch, family, kid, output_dtype)
        registry_reads.append(key)
        return object() if kid == 2 else None

    monkeypatch.setattr(a8, "get_kernel_instance", get_kernel_instance)
    kwargs = {
        "arch": "gfx950",
        "family": "a8w8",
        "output_dtype": torch.float32,
    }

    assert a8._require_registered_kid(kid=2, **kwargs) == 2
    assert a8._require_registered_kid(kid=2, **kwargs) == 2
    assert registry_reads.count(("gfx950", "a8w8", 2, torch.float32)) == 1

    for _ in range(2):
        with pytest.raises(ValueError, match="no registered OPUS kernel"):
            a8._require_registered_kid(kid=1, **kwargs)
    assert registry_reads.count(("gfx950", "a8w8", 1, torch.float32)) == 2


def test_bpreshuffle_python_resolution_and_empty_capabilities(monkeypatch):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    arch = "gfx942"
    monkeypatch.setattr(a8, "_device_arch", lambda _device: arch)
    monkeypatch.setattr(
        a8, "_lookup_opus_bpreshuffle_tuned_kid", lambda *_args: None
    )
    calls = []
    monkeypatch.setattr(
        a8,
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
        lambda *args: calls.append(args),
    )

    XQ = torch.empty((128, 256))
    WQ = torch.empty((256, 256))
    x_scale = torch.empty((128, 2))
    w_scale = torch.empty((2, 2))
    Y = torch.empty((128, 256), dtype=torch.bfloat16)

    def must_not_lookup(*_args):
        raise AssertionError("tuned lookup ran before explicit kid resolution")

    monkeypatch.setattr(a8, "_lookup_opus_bpreshuffle_tuned_kid", must_not_lookup)
    assert a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
        XQ, WQ, x_scale, w_scale, Y, kid=11000
    ) is Y
    assert calls[-1][-1] == 11000

    monkeypatch.setattr(
        a8, "_lookup_opus_bpreshuffle_tuned_kid", lambda *_args: None
    )
    assert a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
        XQ, WQ, x_scale, w_scale, Y
    ) is Y
    assert calls[-1][-1] == 11000

    monkeypatch.setattr(
        a8, "_lookup_opus_bpreshuffle_tuned_kid", lambda *_args: 11000
    )
    assert a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
        XQ, WQ, x_scale, w_scale, Y, kid=None
    ) is Y
    assert calls[-1][-1] == 11000

    monkeypatch.setattr(
        a8, "_lookup_opus_bpreshuffle_tuned_kid", lambda *_args: None
    )
    for empty_arch in ("gfx950", "gfx1250"):
        arch = empty_arch
        with pytest.raises(
            RuntimeError,
            match=rf"no registered OPUS kernel.*arch='{empty_arch}'",
        ):
            a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
                XQ, WQ, x_scale, w_scale, Y
            )
        for foreign_kid in (1, 2, 11000, 99999):
            with pytest.raises(ValueError, match="invalid explicit"):
                a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
                    XQ, WQ, x_scale, w_scale, Y, kid=foreign_kid
                )

    arch = "gfx942"
    wrong_Y = torch.empty_like(Y, dtype=torch.float32)
    with pytest.raises(ValueError, match="Y.dtype=torch.float32"):
        a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
            XQ, WQ, x_scale, w_scale, wrong_Y, kid=11000
        )


def test_a8_legacy_wrapper_warns_once_and_forwards_to_canonical(monkeypatch):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    calls = []

    def fake_launch(XQ, WQ, x_scale, w_scale, Y, *, kid):
        calls.append((XQ, WQ, x_scale, w_scale, Y, kid))
        return Y

    monkeypatch.setattr(
        a8, "opus_gemm_a8w8_blockscale_bpreshuffle_launch", fake_launch
    )
    XQ = torch.empty((128, 256))
    WQ = torch.empty((256, 256))
    x_scale = torch.empty((128, 2))
    w_scale = torch.empty((2, 2))
    Y = torch.empty((128, 256), dtype=torch.bfloat16)
    with pytest.warns(DeprecationWarning) as seen:
        result = a8.opus_gemm_a8w8_blockscale_bpreshuffle_tune(
            XQ, WQ, x_scale, w_scale, Y, kernelId=11000
        )
    assert len(seen) == 1
    assert result is Y
    assert calls == [(XQ, WQ, x_scale, w_scale, Y, 11000)]


def _a8_route_inputs(*, scale_dtype=torch.float32):
    XQ = torch.empty((128, 256))
    WQ = torch.empty((256, 256))
    x_scale = torch.empty((128, 2), dtype=scale_dtype)
    w_scale = torch.empty((2, 2), dtype=scale_dtype)
    return XQ, WQ, x_scale, w_scale


def test_gfx942_opus_row_routes_to_canonical_a8_launch(monkeypatch):
    high = importlib.import_module("aiter.ops.gemm_op_a8w8")
    low = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    monkeypatch.setattr(high, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(high, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(
        high,
        "get_CKGEMM_config",
        lambda *_args, **_kwargs: {
            "libtype": "opus",
            "kernelId": 11000,
            "kernelName": "",
        },
    )
    calls = []

    def fake_opus(XQ, WQ, x_scale, w_scale, Y, *, kid):
        calls.append((XQ, WQ, x_scale, w_scale, Y, kid))
        return Y

    monkeypatch.setattr(
        low, "opus_gemm_a8w8_blockscale_bpreshuffle_launch", fake_opus
    )
    result = high.gemm_a8w8_blockscale_bpreshuffle(
        *_a8_route_inputs(), dtype=torch.bfloat16
    )
    assert result is calls[0][4]
    assert calls[0][-1] == 11000


@pytest.mark.parametrize("libtype", ["ck", "cktile", "asm", "triton"])
def test_gfx950_non_opus_rows_never_call_opus_raw(monkeypatch, libtype):
    high = importlib.import_module("aiter.ops.gemm_op_a8w8")
    low = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    monkeypatch.setattr(high, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(high, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(
        high,
        "get_CKGEMM_config",
        lambda *_args, **_kwargs: {
            "libtype": libtype,
            "kernelId": 7,
            "kernelName": "fixture",
            "splitK": 1,
        },
    )
    monkeypatch.setattr(
        low,
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
        lambda *_args, **_kwargs: pytest.fail("non-OPUS row reached OPUS raw"),
    )
    calls = []

    def fake_backend(*args, **kwargs):
        calls.append((args, kwargs))
        if libtype == "triton":
            return torch.empty((128, 256), dtype=torch.bfloat16)
        return args[2] if libtype == "asm" else args[4]

    if libtype == "ck":
        monkeypatch.setattr(
            high, "gemm_a8w8_blockscale_bpreshuffle_ck", fake_backend
        )
    elif libtype == "cktile":
        monkeypatch.setattr(
            high, "gemm_a8w8_blockscale_bpreshuffle_cktile", fake_backend
        )
    elif libtype == "asm":
        monkeypatch.setattr(
            high, "gemm_a8w8_blockscale_bpreshuffle_asm", fake_backend
        )
    else:
        triton = importlib.import_module(
            "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
        )
        monkeypatch.setattr(
            triton, "gemm_a8w8_blockscale_preshuffle", fake_backend
        )
    result = high.gemm_a8w8_blockscale_bpreshuffle(
        *_a8_route_inputs(), dtype=torch.bfloat16
    )
    assert calls and result is not None
    if libtype == "triton":
        assert calls[0][1]["backend"] is None


def test_gfx1250_flydsl_row_never_calls_opus_raw(monkeypatch):
    from aiter.utility import dtypes

    high = importlib.import_module("aiter.ops.gemm_op_a8w8")
    low = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    monkeypatch.setattr(high, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(high, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(
        high,
        "get_CKGEMM_config",
        lambda *_args, **_kwargs: {
            "libtype": "flydsl",
            "kernelName": "fixture",
        },
    )
    monkeypatch.setattr(
        low,
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
        lambda *_args, **_kwargs: pytest.fail("FlyDSL row reached OPUS raw"),
    )
    calls = []

    def fake_flydsl(XQ, WQ, x_scale, w_scale, Y, config):
        calls.append((XQ, WQ, x_scale, w_scale, Y, config))
        return Y

    monkeypatch.setattr(
        high, "gemm_a8w8_mxfp8_128_bpreshuffle_flydsl", fake_flydsl
    )
    result = high.gemm_a8w8_blockscale_bpreshuffle(
        *_a8_route_inputs(scale_dtype=dtypes.fp8_e8m0),
        dtype=torch.bfloat16,
    )
    assert calls and result is calls[0][4]


@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gfx1250_triton_or_gluon_row_never_calls_opus_raw(monkeypatch, backend):
    high = importlib.import_module("aiter.ops.gemm_op_a8w8")
    low = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    triton = importlib.import_module(
        "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
    )
    monkeypatch.setattr(high, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(high, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(
        high,
        "get_CKGEMM_config",
        lambda *_args, **_kwargs: {
            "libtype": "triton",
            "kernelName": backend,
        },
    )
    monkeypatch.setattr(
        low,
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
        lambda *_args, **_kwargs: pytest.fail("Triton row reached OPUS raw"),
    )
    calls = []

    def fake_triton(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty((128, 256), dtype=torch.bfloat16)

    monkeypatch.setattr(triton, "gemm_a8w8_blockscale_preshuffle", fake_triton)
    result = high.gemm_a8w8_blockscale_bpreshuffle(
        *_a8_route_inputs(), dtype=torch.bfloat16
    )
    assert calls and result.shape == (128, 256)
    assert calls[0][1]["backend"] == backend


def test_canonical_registry_counts_match_b0_golden():
    assert len(kernels_list) == 2039
    assert Counter(_instance_arch(instance) for instance in kernels_list.values()) == {
        "gfx950": 142,
        "gfx942": 23,
        "gfx1250": 1874,
    }
    assert {
        arch: sum(
            _instance_arch(instance) == arch
            and instance.kernel_tag.startswith("a16w16")
            for instance in kernels_list.values()
        )
        for arch in ("gfx950", "gfx942", "gfx1250")
    } == {"gfx950": 140, "gfx942": 22, "gfx1250": 1874}
    assert {
        "gfx950_a8w8_blockscale_plain": len(a8w8_scale_kernels_list),
        "gfx950_a8w8_no_scale": len(a8w8_kernels_list),
        "gfx942_a16w16_non_workspace": len(gfx942_nosplit_kernels_list),
        "gfx942_a16w16_workspace": len(gfx942_splitk_kernels_list),
        "gfx942_a8w8_bpreshuffle": len(gfx942_a8w8_kernels_list),
        "gfx1250_two_stage_plain": len(gfx1250_kernels_list),
        "gfx1250_two_stage_clusterlaunch": len(
            gfx1250_clusterlaunch_kernels_list
        ),
        "gfx1250_fused": len(gfx1250_splitk_fuse_kernels_list),
    } == {
        "gfx950_a8w8_blockscale_plain": 1,
        "gfx950_a8w8_no_scale": 1,
        "gfx942_a16w16_non_workspace": 14,
        "gfx942_a16w16_workspace": 8,
        "gfx942_a8w8_bpreshuffle": 1,
        "gfx1250_two_stage_plain": 28,
        "gfx1250_two_stage_clusterlaunch": 468,
        "gfx1250_fused": 1378,
    }


def _a8_contract(instance) -> tuple[object, ...]:
    return (
        _instance_arch(instance),
        instance.kernel_tag,
        tuple(instance.output_dtypes),
        (instance.B_M, instance.B_N, instance.B_K),
        (instance.GROUP_M, instance.GROUP_N, instance.GROUP_K),
    )


def test_existing_a8_family_contracts_are_kid_and_arch_scoped():
    assert set(a8w8_scale_kernels_list) == {1}
    assert set(a8w8_kernels_list) == {2}
    assert set(gfx942_a8w8_kernels_list) == {11000}
    assert _a8_contract(a8w8_scale_kernels_list[1]) == (
        "gfx950",
        "a8w8_scale",
        ("fp32_t",),
        (256, 256, 128),
        (1, 128, 128),
    )
    assert _a8_contract(a8w8_kernels_list[2]) == (
        "gfx950",
        "a8w8",
        ("fp32_t",),
        (256, 256, 128),
        (0, 0, 0),
    )
    assert _a8_contract(gfx942_a8w8_kernels_list[11000]) == (
        "gfx942",
        _BPRESHUFFLE_TAG,
        ("bf16_t",),
        (128, 128, 128),
        (1, 128, 128),
    )

    assert get_kernel_instance("gfx950", "a8w8", 2, torch.float32) is (
        a8w8_kernels_list[2]
    )
    assert get_kernel_instance(
        "gfx950", "a8w8_blockscale", 1, torch.float32
    ) is a8w8_scale_kernels_list[1]
    assert get_kernel_instance(
        "gfx942",
        "a8w8_blockscale_bpreshuffle",
        11000,
        torch.bfloat16,
    ) is gfx942_a8w8_kernels_list[11000]
    assert get_kernel_instance(
        "gfx942", "a8w8_blockscale_bpreshuffle", 11000, torch.float32
    ) is None
    assert get_kernel_instance("gfx950", "a8w8", 1, torch.float32) is None
    assert get_kernel_instance(
        "gfx950", "a8w8_blockscale_bpreshuffle", 11000, torch.bfloat16
    ) is None


def test_a8_capability_slots_and_mandatory_compile_set_are_explicit():
    assert {
        arch: OPUS_KERNEL_TAGS_BY_ARCH_FAMILY[arch][
            "a8w8_blockscale_bpreshuffle"
        ]
        for arch in ("gfx942", "gfx950", "gfx1250")
    } == {
        "gfx942": frozenset({_BPRESHUFFLE_TAG}),
        "gfx950": frozenset(),
        "gfx1250": frozenset(),
    }
    assert OPUS_MANDATORY_A8_KIDS == {
        "gfx950": frozenset({1, 2}),
        "gfx942": frozenset({11000}),
        "gfx1250": frozenset(),
    }


def test_subset_compile_formula_arch_filter_and_mandatory_kids(tmp_path):
    csv_path = tmp_path / "tuned.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,libtype,solidx\n"
        "gfx950,256,16,16,128,opus,4\n"
        "gfx950,256,16,16,128,ck,5\n"
        "gfx942,304,16,16,128,opus,10000\n"
        "gfx950,256,16,16,128,opus,999999\n"
    )
    sidecar_path = tmp_path / "compiled_kids.json"
    sidecar_kids = {6, 10000, 999999}
    sidecar_path.write_text(json.dumps(sorted(sidecar_kids)))
    working_path = tmp_path / "generated"
    working_path.mkdir()

    env = os.environ.copy()
    env["GPU_ARCHS"] = "gfx950"
    completed = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "csrc/opus_gemm/gen_instances.py"),
            "--working_path",
            str(working_path),
            "--tune_files",
            str(csv_path),
            "--compiled_kids_sidecar",
            str(sidecar_path),
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    valid_kids = set(kernels_list)
    csv_opus_kids = {4, 10000, 999999}
    expected = (
        csv_opus_kids | sidecar_kids | set(HEURISTIC_DEFAULT_KIDS)
    ) & valid_kids
    expected = {
        kid
        for kid in expected
        if _instance_arch(kernels_list[kid]) == "gfx950"
    }
    expected |= set(OPUS_MANDATORY_A8_KIDS["gfx950"]) & valid_kids

    actual = set(json.loads(sidecar_path.read_text()))
    assert actual == expected
    assert {1, 2, 4, 6} <= actual
    assert {5, 10000, 999999}.isdisjoint(actual)


def test_opus_bpreshuffle_capability_matrix_b0_golden():
    capability = {
        arch: {
            kid
            for kid, instance in kernels_list.items()
            if _instance_arch(instance) == arch
            and instance.kernel_tag == _BPRESHUFFLE_TAG
        }
        for arch in ("gfx942", "gfx950", "gfx1250")
    }
    assert capability == {"gfx942": {11000}, "gfx950": set(), "gfx1250": set()}


@pytest.mark.parametrize(
    ("arch", "kid", "output_dtype", "dtype_suffix"),
    [
        ("gfx950", 900001, "bf16_t", "BF16"),
        ("gfx1250", 900002, "fp32_t", "FP32"),
    ],
)
def test_empty_bpreshuffle_codegen_slot_expands_with_registry_and_emitter_only(
    tmp_path, monkeypatch, arch, kid, output_dtype, dtype_suffix
):
    """A future arch kernel needs registry/emitter data, not public ABI edits.

    This is a structural codegen fixture, not a claim that the synthetic
    instance is a runnable physical kernel.  Its minimal emitter supplies a
    valid host launcher in the temporary output tree so the shared typed table
    can be exercised without changing any production header or wrapper.
    """
    monkeypatch.syspath_prepend(str(_ROOT / "csrc/opus_gemm"))
    from codegen.common import EMIT_REGISTRY
    from gen_instances import opus_gemm_codegen

    public_paths = (
        _ROOT / "csrc/opus_gemm/include/opus_gemm.h",
        _ROOT / f"csrc/opus_gemm/include/{arch}/opus_gemm_arch_{arch}.cuh",
        _ROOT / "csrc/include/rocm_ops.hpp",
        _ROOT / "csrc/pybind/opus_gemm_pybind.cu",
        _ROOT / "aiter/ops/opus/gemm_op_a8w8.py",
        _ROOT / "aiter/ops/opus/__init__.py",
    )
    public_before = {path: path.read_bytes() for path in public_paths}
    macro = (
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_"
        f"{arch.upper()}_{dtype_suffix}"
    )
    other_suffix = "FP32" if dtype_suffix == "BF16" else "BF16"
    other_macro = (
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_"
        f"{arch.upper()}_{other_suffix}"
    )

    assert not OPUS_KERNEL_TAGS_BY_ARCH_FAMILY[arch][
        "a8w8_blockscale_bpreshuffle"
    ]
    assert get_kernel_instance(
        arch, "a8w8_blockscale_bpreshuffle", kid, output_dtype
    ) is None
    baseline_codegen = opus_gemm_codegen(str(tmp_path), False)
    baseline_codegen.gen_a8w8_kid_dispatch(kernels_list)
    baseline_header = (tmp_path / "opus_gemm_a8w8_kid_dispatch.h").read_text()
    assert f"#define {macro}_SIZE 0" in baseline_header
    assert _macro_body(baseline_header, macro) == ""

    instance = replace(
        gfx942_a8w8_kernels_list[11000],
        arch_prefix=arch,
        name_tag=f"synthetic_{arch}_bpreshuffle",
        output_dtypes=[output_dtype],
    )
    monkeypatch.setitem(kernels_list, kid, instance)
    monkeypatch.setitem(
        OPUS_KERNEL_TAGS_BY_ARCH_FAMILY[arch],
        "a8w8_blockscale_bpreshuffle",
        frozenset({_BPRESHUFFLE_TAG}),
    )
    assert get_kernel_instance(
        arch, "a8w8_blockscale_bpreshuffle", kid, output_dtype
    ) is instance

    emitted = []

    def synthetic_emit(
        codegen,
        kernel,
        make_a8w8_bpreshuffle_host_decl,
        **_unused,
    ):
        emitted.append((_instance_arch(kernel), kernel.kernel_tag, kernel.name))
        (Path(codegen.impl_path) / f"{kernel.name}.cuh").write_text(
            f'''#include "aiter_tensor.h"
template <typename D_C>
void {kernel.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y)
{{
    (void)XQ; (void)WQ; (void)x_scale; (void)w_scale; (void)Y;
}}
'''
        )
        for dtype in kernel.output_dtypes:
            codegen._host_instantiations.append(
                {
                    "kid_name": kernel.name,
                    "dtype": dtype,
                    "host_decl": make_a8w8_bpreshuffle_host_decl(
                        kernel.name, dtype, ""
                    ),
                }
            )

    monkeypatch.setitem(
        EMIT_REGISTRY, (arch, _BPRESHUFFLE_TAG), synthetic_emit
    )
    codegen = opus_gemm_codegen(str(tmp_path), False)
    codegen.gen_instances({kid: instance})

    assert emitted == [(arch, _BPRESHUFFLE_TAG, instance.name)]
    assert (tmp_path / "instances" / f"all_instances_host_{arch}.cu").is_file()
    header = (tmp_path / "opus_gemm_a8w8_kid_dispatch.h").read_text()
    assert _macro_kids(header, macro) == (kid,)
    assert f"#define {macro}_SIZE 1" in header
    assert f"&{instance.name}<{output_dtype}>" in _macro_body(header, macro)
    assert _macro_kids(header, other_macro) == ()
    assert f"#define {other_macro}_SIZE 0" in header
    assert _macro_body(header, other_macro) == ""
    assert {path: path.read_bytes() for path in public_paths} == public_before


_A16_DISPATCH_SET_GOLDEN = {
    "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX950_BF16": (
        92,
        "f9743fd6634ab6d010798208d4fc0526f941ccb50be1eb064d2912719f1e8994",
    ),
    "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX950_FP32": (
        92,
        "f9743fd6634ab6d010798208d4fc0526f941ccb50be1eb064d2912719f1e8994",
    ),
    "GENERATE_A16W16_WORKSPACE_KID_DISPATCH_GFX950": (
        48,
        "64dc006db4356f018ec42fddec710ce8be6ef5c5e4b7c47356abeb46eb93a6a6",
    ),
    "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX942_BF16": (
        14,
        "62ba8933000e2392d38f368f555882a26361369e8966ca46d1e43e7633638dab",
    ),
    "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX942_FP32": (
        1,
        "39e5b4830d4d9c14db7368a95b65d5463ea3d09520373723430c03a5a453b5df",
    ),
    "GENERATE_A16W16_WORKSPACE_KID_DISPATCH_GFX942": (
        8,
        "3d34ca7ffe881e360cf767711d6ecefaaa8d7838d84c840e9f56a9ba0d4f6f3e",
    ),
    "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX1250_BF16": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX1250_FP32": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "GENERATE_A16W16_WORKSPACE_KID_DISPATCH_GFX1250": (
        1874,
        "c63348e05fcabc4bac52228dbae4ad46c851ef19ad23d79cbf1a80ef3e22639a",
    ),
}


def _macro_body(source: str, macro_name: str) -> str:
    macro = re.search(
        rf"^#define {re.escape(macro_name)}(?:\(CTYPE\))?"
        rf"(?: \\\n(?P<body>.*?))?\n\n",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert macro is not None, f"missing generated macro {macro_name}"
    return macro["body"] or ""


def _macro_kids(source: str, macro_name: str) -> tuple[int, ...]:
    return tuple(
        int(kid)
        for kid in re.findall(r"\{\s*(\d+)\s*,", _macro_body(source, macro_name))
    )


def _kid_set_digest(kids: tuple[int, ...]) -> str:
    assert len(kids) == len(set(kids)), "generated dispatch contains duplicate kids"
    payload = ",".join(str(kid) for kid in sorted(kids))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_generated_dispatch_kid_sets_match_b0_golden(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / "csrc/opus_gemm"))
    from gen_instances import opus_gemm_codegen

    codegen = opus_gemm_codegen(str(tmp_path), False)
    codegen.gen_a16w16_kid_dispatch(kernels_list)
    codegen.gen_a8w8_kid_dispatch(kernels_list)

    a16_path = tmp_path / "opus_gemm_a16w16_kid_dispatch.h"
    a8_path = tmp_path / "opus_gemm_a8w8_kid_dispatch.h"
    first_a16 = a16_path.read_bytes()
    first_a8 = a8_path.read_bytes()
    codegen.gen_a16w16_kid_dispatch(kernels_list)
    codegen.gen_a8w8_kid_dispatch(kernels_list)
    assert a16_path.read_bytes() == first_a16
    assert a8_path.read_bytes() == first_a8

    assert not (tmp_path / "opus_gemm_lookup.h").exists()
    assert not (tmp_path / "opus_gemm_a16w16_tune_lookup.h").exists()
    assert not (tmp_path / "opus_gemm_a8w8_tune_lookup.h").exists()

    a16_header = first_a16.decode()
    assert "TUNE_LOOKUP" not in a16_header
    assert "GENERATE_OPUS_LOOKUP_TABLE" not in a16_header
    for macro_name, (expected_count, expected_digest) in (
        _A16_DISPATCH_SET_GOLDEN.items()
    ):
        kids = _macro_kids(a16_header, macro_name)
        declared_size = re.search(
            rf"^#define {re.escape(macro_name)}_SIZE (\d+)$",
            a16_header,
            flags=re.MULTILINE,
        )
        assert declared_size is not None
        assert int(declared_size.group(1)) == len(kids) == expected_count
        assert _kid_set_digest(kids) == expected_digest

    a8_header = first_a8.decode()
    expected_a8 = {
        "GENERATE_A8W8_NOSCALE_KID_DISPATCH_GFX950": (2,),
        "GENERATE_A8W8_BLOCKSCALE_KID_DISPATCH_GFX950": (1,),
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_BF16": (
            11000,
        ),
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_FP32": (),
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16": (),
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_FP32": (),
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_BF16": (),
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_FP32": (),
    }
    for macro_name, expected_kids in expected_a8.items():
        kids = _macro_kids(a8_header, macro_name)
        assert kids == expected_kids
        declared_size = re.search(
            rf"^#define {re.escape(macro_name)}_SIZE (\d+)$",
            a8_header,
            flags=re.MULTILINE,
        )
        assert declared_size is not None
        assert int(declared_size.group(1)) == len(expected_kids)

    expected_pointer_parameters = {
        "OpusA16W16Kernel": (
            "aiter_tensor_t&",
            "aiter_tensor_t&",
            "aiter_tensor_t&",
            "std::optional<aiter_tensor_t>",
            "int",
        ),
        "OpusA16W16WorkspaceKernel": (
            "aiter_tensor_t&",
            "aiter_tensor_t&",
            "aiter_tensor_t&",
            "aiter_tensor_t&",
            "std::optional<aiter_tensor_t>",
            "int",
        ),
        "OpusA8W8Kernel": ("aiter_tensor_t&",) * 3,
        "OpusA8W8BlockscaleKernel": ("aiter_tensor_t&",) * 5,
        "OpusA8W8BlockscaleBpreshuffleKernel": ("aiter_tensor_t&",) * 5,
    }
    for arch in ("gfx942", "gfx950", "gfx1250"):
        arch_header = (
            _ROOT
            / f"csrc/opus_gemm/include/{arch}/opus_gemm_arch_{arch}.cuh"
        ).read_text()
        assert {
            alias: _cpp_function_pointer_parameters(arch_header, alias)
            for alias in expected_pointer_parameters
        } == expected_pointer_parameters
        assert re.search(
            r"struct\s+OpusA16W16KidEntry\s*\{.*?"
            r"OpusA16W16Kernel\s+func\s*;",
            arch_header,
            flags=re.DOTALL,
        )
        assert re.search(
            r"struct\s+OpusA16W16WorkspaceKidEntry\s*\{.*?"
            r"OpusA16W16WorkspaceKernel\s+func\s*;",
            arch_header,
            flags=re.DOTALL,
        )
        assert re.search(
            r"OpusA8W8KidEntry<\s*"
            r"OpusA8W8BlockscaleBpreshuffleKernel\s*>",
            arch_header,
        )
        for macro_name in (*_A16_DISPATCH_SET_GOLDEN, *expected_a8):
            if arch.upper() in macro_name:
                assert macro_name in arch_header

    gfx950_header = (
        _ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh"
    ).read_text()
    for family_type in ("OpusA8W8Kernel", "OpusA8W8BlockscaleKernel"):
        assert re.search(
            rf"OpusA8W8KidEntry<\s*{family_type}\s*>", gfx950_header
        )


def test_b6_generated_a8_launchers_own_exact_physical_contracts(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(_ROOT / "csrc/opus_gemm"))
    from gen_instances import opus_gemm_codegen

    selected = {
        1: a8w8_scale_kernels_list[1],
        2: a8w8_kernels_list[2],
        11000: gfx942_a8w8_kernels_list[11000],
    }
    codegen = opus_gemm_codegen(str(tmp_path), False)
    codegen.gen_instances(selected)

    scale = (
        tmp_path / "impl" / f"{selected[1].name}.cuh"
    ).read_text()
    for contract in (
        "XQ/WQ/Y must be 3D",
        "expected fp8 XQ/WQ",
        "expected fp32 Y",
        "XQ/WQ must be K-contiguous",
        "tensor shapes must be",
        "loops_ >= 2",
        "loops_ % 2 == 0",
        "K % 2 == 0",
        "expects fp32 scales",
        "expects contiguous scales",
        "x_scale must be",
        "w_scale must be",
    ):
        assert contract in scale

    noscale = (
        tmp_path / "impl" / f"{selected[2].name}.cuh"
    ).read_text()
    for contract in (
        "XQ/WQ/Y must be 3D",
        "XQ/WQ must be K-contiguous",
        "loops_ >= 2",
        "loops_ % 2 == 0",
        "K % 2 == 0",
    ):
        assert contract in noscale

    bpreshuffle = (
        tmp_path / "impl" / f"{selected[11000].name}.cuh"
    ).read_text()
    for contract in (
        "XQ must be",
        "WQ must be",
        "Y must be",
        "supports batch=1 only",
        "expected fp8 XQ/WQ",
        "expected bf16 Y",
        "expects fp32 scales",
        "expects contiguous ",
        "exact N/K tiles",
        "x_scale must use ",
        "transposed storage contract",
        "w_scale must be ",
        "row-major [N/128,K/128]",
    ):
        assert contract in bpreshuffle


def test_b6_public_router_is_arch_neutral_and_errors_use_launch_terms():
    implementation = (_ROOT / "csrc/opus_gemm/opus_gemm.cu").read_text()
    start = implementation.index(
        "void opus_gemm_a8w8_blockscale_bpreshuffle_launch("
    )
    body = implementation[start : implementation.index("\n}\n", start) + 2]
    for arch_physical_rule in (
        "batch == 1",
        "N % 128",
        "K % 128",
        "x_scale.dtype",
        "w_scale.dtype",
    ):
        assert arch_physical_rule not in body

    assert "Host-side family routers and strict exact-kid dispatch" in implementation
    assert "lookup table + heuristic" not in implementation
    assert "opus_a16w16_tune_dispatch" not in implementation

    for arch in ("gfx942", "gfx950", "gfx1250"):
        header = (
            _ROOT
            / f"csrc/opus_gemm/include/{arch}/opus_gemm_arch_{arch}.cuh"
        ).read_text()
        assert "opus_a16w16_tune_dispatch" not in header
        assert f"opus_a16w16_kid_dispatch_{arch}" in header
        assert "Kernel id" not in header
        assert "unknown kid" in header


def test_b6_python_api_documents_bpreshuffle_content_semantics():
    module = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    doc = inspect.getdoc(
        module.opus_gemm_a8w8_blockscale_bpreshuffle_launch
    )
    assert doc is not None
    assert "content/layout semantic" in doc
    assert "cannot be proven from Tensor shape or strides" in doc
    assert "shuffle_weight(WQ, layout=(16, 16))" in doc


def test_b6_readmes_match_family_policy_and_task1_workspace_state():
    python_readme = (_ROOT / "aiter/ops/opus/README.md").read_text()
    cpp_readme = (_ROOT / "csrc/opus_gemm/README.md").read_text()
    combined = python_readme + cpp_readme

    for launch in (
        "opus_gemm_a16w16_launch",
        "opus_gemm_a8w8_launch",
        "opus_gemm_a8w8_blockscale_launch",
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
    ):
        assert launch in combined
    for required in (
        "[tiles_m, tiles_n, fuse_split_k - 1, B_M, B_N]",
        "1378",
        "runtime policy",
        "subset compile",
        "opus_gemm_a16w16_tune",
        "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
    ):
        assert required in combined
    for stale in (
        "WorkspacePlan",
        "`_workspace.py`",
        "`_workspace_a16w16.py`",
        "_opus_gemm_bf16_dispatch",
        "opus_gemm_lookup.h",
        "tune_lookup",
        "generic BF16",
    ):
        assert stale not in combined


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0].lower()


def _gfx950_a16_raw_case():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires idle gfx950 hardware; a skip is not a pass")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    config = gemm.select_launch_config(
        arch="gfx950",
        M=64,
        N=64,
        K=512,
        batch=1,
        cu_num=torch.cuda.get_device_properties(device).multi_processor_count,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        explicit_kid=200,
        explicit_split_k=2,
        tuned_lookup=lambda **_kwargs: None,
    )
    XQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    WQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), device=device, dtype=torch.bfloat16)
    workspace = gemm._init_a16w16_workspace(config, XQ, Y)
    assert workspace is not None
    return gemm, config, XQ, WQ, Y, workspace


@pytest.mark.parametrize("failure", ["missing", "dtype", "short"])
def test_gfx950_canonical_raw_workspace_errors(failure):
    gemm, config, XQ, WQ, Y, allocated = _gfx950_a16_raw_case()
    if failure == "missing":
        workspace = None
        expected = "requires a workspace tensor"
    elif failure == "dtype":
        workspace = torch.empty(
            allocated.numel(), device=Y.device, dtype=torch.bfloat16
        )
        expected = "workspace dtype must be"
    else:
        workspace = torch.empty(
            allocated.numel() - 1, device=Y.device, dtype=allocated.dtype
        )
        expected = "workspace capacity"

    with pytest.raises(RuntimeError) as error:
        gemm._opus_gemm_a16w16_launch_raw(
            XQ,
            WQ,
            Y,
            None,
            workspace,
            config.actual_kid,
            config.launch_split_k,
        )
    assert expected in str(error.value)


def test_gfx950_canonical_raw_numerical_result():
    gemm, config, XQ, WQ, Y, workspace = _gfx950_a16_raw_case()
    gemm._opus_gemm_a16w16_launch_raw(
        XQ,
        WQ,
        Y,
        None,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )
    torch.cuda.synchronize(Y.device)
    golden = torch.bmm(XQ.float(), WQ.float().transpose(1, 2))
    torch.testing.assert_close(Y.float(), golden, rtol=0.03, atol=0.5)


def _plain_a8_inputs(device: torch.device, M: int, N: int, K: int):
    from aiter import dtypes

    XQ = (
        torch.arange(M * K, device=device, dtype=torch.int32).remainder(5).sub(2)
    ).reshape(M, K).to(dtypes.fp8)
    WQ = (
        torch.arange(N * K, device=device, dtype=torch.int32).remainder(7).sub(3)
    ).reshape(N, K).to(dtypes.fp8)
    return XQ, WQ


def _blockscale_reference(XQ, WQ, x_scale, w_scale, output_dtype):
    M, K = XQ.shape
    N = WQ.shape[0]
    result = torch.zeros((M, N), device=XQ.device, dtype=torch.float32)
    for block_k in range(K // 128):
        partial = XQ[:, block_k * 128 : (block_k + 1) * 128].float() @ WQ[
            :, block_k * 128 : (block_k + 1) * 128
        ].float().T
        result.add_(
            partial
            * x_scale[:, block_k].unsqueeze(1)
            * w_scale[:, block_k].repeat_interleave(128).unsqueeze(0)
        )
    return result.to(output_dtype)


def test_gfx950_a8w8_no_scale_raw_numerical_golden():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires idle gfx950 hardware; a skip is not a pass")
    device = torch.device("cuda", torch.cuda.current_device())
    M = N = K = 256
    XQ, WQ = _plain_a8_inputs(device, M, N, K)
    Y = torch.empty((1, M, N), device=device, dtype=torch.float32)
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")

    a8._opus_gemm_a8w8_launch_raw(XQ.unsqueeze(0), WQ.unsqueeze(0), Y, 2)
    expected = XQ.float() @ WQ.float().T
    torch.testing.assert_close(Y[0], expected, rtol=0, atol=0)
    assert Y[0, :3, :3].cpu().tolist() == [
        [-1.0, 5.0, 4.0],
        [-7.0, 3.0, -8.0],
        [-3.0, 11.0, -10.0],
    ]
    assert Y.sum().item() == -3.0


def test_gfx950_a8w8_plain_blockscale_raw_numerical_golden():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires idle gfx950 hardware; a skip is not a pass")
    device = torch.device("cuda", torch.cuda.current_device())
    M = N = K = 256
    XQ, WQ = _plain_a8_inputs(device, M, N, K)
    x_scale = torch.ones((M, K // 128), device=device, dtype=torch.float32)
    x_scale[1::2].mul_(0.5)
    x_scale[:, 1].mul_(2.0)
    w_scale = torch.ones((N // 128, K // 128), device=device, dtype=torch.float32)
    w_scale[1].mul_(2.0)
    w_scale[:, 1].mul_(0.25)
    Y = torch.empty((1, M, N), device=device, dtype=torch.float32)
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    a8._opus_gemm_a8w8_blockscale_launch_raw(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y,
        x_scale.unsqueeze(0),
        w_scale.unsqueeze(0),
        1,
    )
    expected = _blockscale_reference(XQ, WQ, x_scale, w_scale, torch.float32)
    torch.testing.assert_close(Y[0], expected, rtol=0, atol=0)
    assert Y[0, :3, :3].cpu().tolist() == [
        [0.5, 1.0, 8.5],
        [-1.25, -0.5, -3.25],
        [-8.0, 12.0, -6.5],
    ]
    assert Y.sum().item() == -4.75


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("scale_shape", "x_scale must be"),
        ("w_scale_shape", "w_scale must be"),
        ("scale_dtype", "expects fp32 scales"),
        ("w_scale_dtype", "expects fp32 scales"),
        ("output_dtype", "expected fp32 Y"),
        ("wrong_kid", "unknown kid 2.*a8w8_blockscale"),
        ("empty_bpreshuffle", "no registered kernel.*bpreshuffle.*gfx950"),
        ("prefetch_min", "must be >= 2"),
        ("prefetch_even", "must be even.*prefetch constraint"),
        ("noscale_shape", "tensor shapes must be"),
        ("noscale_layout", "K-contiguous"),
        ("noscale_wrong_kid", "unknown kid 1.*a8w8 on gfx950"),
    ],
)
def test_gfx950_a8_family_negative_contracts(failure, message):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    device = torch.device("cuda", torch.cuda.current_device())
    XQ, WQ = _plain_a8_inputs(device, 256, 256, 256)
    XQ3 = XQ.unsqueeze(0)
    WQ3 = WQ.unsqueeze(0)
    Y = torch.empty((1, 256, 256), device=device, dtype=torch.float32)
    x_scale = torch.ones((1, 256, 2), device=device, dtype=torch.float32)
    w_scale = torch.ones((1, 2, 2), device=device, dtype=torch.float32)
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")

    with pytest.raises(RuntimeError, match=message):
        if failure == "scale_shape":
            a8._opus_gemm_a8w8_blockscale_launch_raw(
                XQ3, WQ3, Y, x_scale[:, :-1], w_scale, 1
            )
        elif failure == "w_scale_shape":
            a8._opus_gemm_a8w8_blockscale_launch_raw(
                XQ3, WQ3, Y, x_scale, w_scale[:, :-1], 1
            )
        elif failure == "scale_dtype":
            a8._opus_gemm_a8w8_blockscale_launch_raw(
                XQ3, WQ3, Y, x_scale.to(torch.bfloat16), w_scale, 1
            )
        elif failure == "w_scale_dtype":
            a8._opus_gemm_a8w8_blockscale_launch_raw(
                XQ3, WQ3, Y, x_scale, w_scale.to(torch.bfloat16), 1
            )
        elif failure == "output_dtype":
            a8._opus_gemm_a8w8_launch_raw(
                XQ3, WQ3, Y.to(torch.bfloat16), 2
            )
        elif failure == "wrong_kid":
            a8._opus_gemm_a8w8_blockscale_launch_raw(
                XQ3, WQ3, Y, x_scale, w_scale, 2
            )
        elif failure == "empty_bpreshuffle":
            a8._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
                XQ,
                WQ,
                x_scale[0],
                w_scale[0],
                Y[0].to(torch.bfloat16),
                11000,
            )
        elif failure in ("prefetch_min", "prefetch_even"):
            test_k = 128 if failure == "prefetch_min" else 384
            test_XQ, test_WQ = _plain_a8_inputs(device, 256, 256, test_k)
            test_Y = torch.empty(
                (1, 256, 256), device=device, dtype=torch.float32
            )
            test_x_scale = torch.ones(
                (1, 256, test_k // 128),
                device=device,
                dtype=torch.float32,
            )
            test_w_scale = torch.ones(
                (1, 2, test_k // 128),
                device=device,
                dtype=torch.float32,
            )
            a8._opus_gemm_a8w8_blockscale_launch_raw(
                test_XQ.unsqueeze(0),
                test_WQ.unsqueeze(0),
                test_Y,
                test_x_scale,
                test_w_scale,
                1,
            )
        elif failure == "noscale_shape":
            a8._opus_gemm_a8w8_launch_raw(XQ3, WQ3[:, :-1], Y, 2)
        elif failure == "noscale_layout":
            a8._opus_gemm_a8w8_launch_raw(
                XQ3[:, :, ::2], WQ3[:, :, ::2], Y, 2
            )
        else:
            a8._opus_gemm_a8w8_launch_raw(XQ3, WQ3, Y, 1)


def test_gfx950_a8_public_scale_device_single_scale_and_empty_capability():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    device = torch.device("cuda", torch.cuda.current_device())
    XQ, WQ = _plain_a8_inputs(device, 256, 256, 256)
    XQ3 = XQ.unsqueeze(0)
    WQ3 = WQ.unsqueeze(0)
    Y = torch.empty((1, 256, 256), device=device, dtype=torch.float32)
    x_scale = torch.ones((1, 256, 2), device=device, dtype=torch.float32)
    w_scale = torch.ones((1, 2, 2), device=device, dtype=torch.float32)
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")

    # Both scales are mandatory; a one-scale call cannot silently select the
    # no-scale family or reinterpret the remaining positional argument.
    with pytest.raises(TypeError, match="w_scale"):
        a8.opus_gemm_a8w8_blockscale_launch(XQ3, WQ3, Y, x_scale)
    with pytest.raises(TypeError, match="w_scale must be Tensor"):
        a8.opus_gemm_a8w8_blockscale_launch(
            XQ3, WQ3, Y, x_scale, None  # type: ignore[arg-type]
        )

    if torch.cuda.device_count() < 2:
        pytest.skip("scale device negatives require two visible gfx950 devices")
    other_device = torch.device("cuda", 1)
    for scale_name, foreign_scale in (
        ("x_scale", x_scale.to(other_device)),
        ("w_scale", w_scale.to(other_device)),
    ):
        args = {
            "x_scale": x_scale,
            "w_scale": w_scale,
        }
        args[scale_name] = foreign_scale
        with pytest.raises(ValueError, match="all tensors must be on one device"):
            a8.opus_gemm_a8w8_blockscale_launch(
                XQ3, WQ3, Y, args["x_scale"], args["w_scale"], kid=1
            )

    # The stable reserved public entry exists on gfx950, but kid=None must
    # report the empty typed capability instead of falling into another family.
    with pytest.raises(
        RuntimeError,
        match="no registered OPUS kernel.*arch='gfx950'.*bpreshuffle",
    ):
        a8.opus_gemm_a8w8_blockscale_bpreshuffle_launch(
            XQ,
            WQ,
            x_scale[0],
            w_scale[0],
            torch.empty((256, 256), device=device, dtype=torch.bfloat16),
            kid=None,
        )


def test_gfx942_a8w8_bpreshuffle_raw_numerical_golden():
    if _runtime_arch() != "gfx942":
        pytest.skip("requires idle gfx942 hardware; a skip is not a pass")
    from aiter.ops.shuffle import shuffle_weight

    device = torch.device("cuda", torch.cuda.current_device())
    M, N, K = 128, 256, 256
    XQ, WQ = _plain_a8_inputs(device, M, N, K)
    x_scale = torch.ones((M, K // 128), device=device, dtype=torch.float32)
    x_scale[1::2].mul_(0.5)
    x_scale[:, 1].mul_(2.0)
    w_scale = torch.ones((N // 128, K // 128), device=device, dtype=torch.float32)
    w_scale[1].mul_(2.0)
    w_scale[:, 1].mul_(0.25)
    # kid 11000 consumes the existing CK bpreshuffle scale layout.
    x_scale_storage = x_scale.T.contiguous().view_as(x_scale)
    WQ_storage = shuffle_weight(WQ, layout=(16, 16))
    Y = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    raw = importlib.import_module(
        "aiter.ops.opus.gemm_op_a8w8"
    )._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw

    raw(XQ, WQ_storage, x_scale_storage, w_scale, Y, 11000)
    expected = _blockscale_reference(XQ, WQ, x_scale, w_scale, torch.bfloat16)
    torch.testing.assert_close(Y, expected, rtol=0, atol=0)

    # The same exact launcher accepts explicit 3D tensors when batch is one.
    Y3 = torch.empty((1, M, N), device=device, dtype=torch.bfloat16)
    raw(
        XQ.unsqueeze(0),
        WQ_storage.unsqueeze(0),
        x_scale_storage,
        w_scale,
        Y3,
        11000,
    )
    torch.testing.assert_close(Y3[0], expected, rtol=0, atol=0)
    assert Y[:3, :3].float().cpu().tolist() == [
        [0.5, 1.0, 8.5],
        [-1.25, -0.5, -3.25],
        [-8.0, 12.0, -6.5],
    ]
    assert Y.float().sum().item() == -11.75


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("batch", "supports batch=1 only"),
        ("scale_dtype", "expects fp32 scales"),
        ("scale_shape", "x_scale must use"),
        ("n_tile", "exact N/K tiles"),
        ("k_tile", "exact N/K tiles"),
        ("wq_layout", "expects contiguous XQ/WQ/Y"),
        ("wrong_kid", "unknown kid 1.*bpreshuffle.*gfx942"),
    ],
)
def test_gfx942_a8w8_bpreshuffle_negative_contracts(failure, message):
    if _runtime_arch() != "gfx942":
        pytest.skip("requires gfx942 hardware")
    device = torch.device("cuda", torch.cuda.current_device())
    M = 128
    N = K = 256
    if failure == "n_tile":
        N = 192
    elif failure == "k_tile":
        K = 192

    XQ, WQ = _plain_a8_inputs(device, M, N, K)
    x_scale = torch.ones(
        (M, max(1, K // 128)), device=device, dtype=torch.float32
    )
    w_scale = torch.ones(
        (max(1, N // 128), max(1, K // 128)),
        device=device,
        dtype=torch.float32,
    )
    Y = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    raw = importlib.import_module(
        "aiter.ops.opus.gemm_op_a8w8"
    )._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw

    with pytest.raises(RuntimeError, match=message):
        if failure == "batch":
            raw(
                XQ.unsqueeze(0).repeat(2, 1, 1),
                WQ.unsqueeze(0).repeat(2, 1, 1),
                x_scale,
                w_scale,
                Y.unsqueeze(0).repeat(2, 1, 1),
                11000,
            )
        elif failure == "scale_dtype":
            raw(XQ, WQ, x_scale.to(torch.bfloat16), w_scale, Y, 11000)
        elif failure == "scale_shape":
            raw(XQ, WQ, x_scale[:-1], w_scale, Y, 11000)
        elif failure == "wq_layout":
            raw(XQ, WQ.T, x_scale, w_scale, Y, 11000)
        elif failure == "wrong_kid":
            raw(XQ, WQ, x_scale, w_scale, Y, 1)
        else:
            raw(XQ, WQ, x_scale, w_scale, Y, 11000)
