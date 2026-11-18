# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reuse the mapped HIP file when dlopen hooks redirect soname lookups."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "aiter/jit/utils/hip_runtime.py"
_SPEC = importlib.util.spec_from_file_location("hip_runtime_under_test", _SOURCE)
hip_runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hip_runtime)


def _maps(*paths):
    return "".join(f"1000-2000 r-xp 00000000 00:01 1 {path}\n" for path in paths)


@pytest.fixture(autouse=True)
def fresh_loader():
    hip_runtime.load_hip_runtime.cache_clear()
    yield
    hip_runtime.load_hip_runtime.cache_clear()


def test_loader_uses_mapped_path_before_redirected_sonames(monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    core = "/venv/_rocm_sdk_core/lib/libamdhip64.so.7"
    monkeypatch.setattr(
        hip_runtime, "open", mock.mock_open(read_data=_maps(core)), raising=False
    )
    loaded = object()

    def redirected_dlopen(name, mode=0):
        if mode & os.RTLD_NOLOAD:
            if name == core:
                return loaded
            raise OSError("hook resolved name to an unloaded devel copy")
        pytest.fail("loading another HIP runtime can abort on duplicate COMGR")

    cdll = mock.Mock(side_effect=redirected_dlopen)
    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", cdll)
    assert hip_runtime.load_hip_runtime() is loaded
    assert hip_runtime.load_hip_runtime() is loaded
    cdll.assert_called_once_with(core, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)


def test_maps_preserve_spaces_deduplicate_segments_and_ignore_other_files(monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=None))
    path = "/core SDK/lib/libamdhip64.so.7.15.26333"
    text = _maps("/lib/libc.so.6", path, path, "/tmp/not-libamdhip64.so.7", "[heap]")
    monkeypatch.setattr(
        hip_runtime, "open", mock.mock_open(read_data=text), raising=False
    )
    assert hip_runtime._loaded_hip_runtime_paths() == [path]


def test_maps_prefer_the_imported_torch_hip_major(monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip="8.1.2"))
    )
    legacy, current = "/legacy/libamdhip64.so.7", "/core/libamdhip64.so.8.1.2"
    monkeypatch.setattr(
        hip_runtime,
        "open",
        mock.mock_open(read_data=_maps(legacy, current)),
        raising=False,
    )
    assert hip_runtime._loaded_hip_runtime_paths() == [current, legacy]


def test_maps_strip_the_deleted_marker(monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    path = "/core/libamdhip64.so.7"
    monkeypatch.setattr(
        hip_runtime,
        "open",
        mock.mock_open(read_data=_maps(path + " (deleted)")),
        raising=False,
    )
    assert hip_runtime._loaded_hip_runtime_paths() == [path]


@pytest.mark.parametrize("missing_maps", [True, False])
def test_soname_fallback_remains_available(monkeypatch, missing_maps):
    monkeypatch.setattr(hip_runtime.sys, "platform", "linux")
    mapped = "/unloaded/libamdhip64.so.7"
    maps = mock.mock_open(read_data=_maps(mapped))
    if missing_maps:
        maps.side_effect = OSError("proc is unavailable")
    monkeypatch.setattr(hip_runtime, "open", maps, raising=False)
    loaded = object()

    def cdll(name, mode=0):
        assert mode & os.RTLD_NOLOAD
        if name == "libamdhip64.so.7":
            return loaded
        raise OSError("not loaded")

    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", cdll)
    assert hip_runtime.load_hip_runtime() is loaded


def test_windows_keeps_dll_loading_without_proc_queries(monkeypatch):
    monkeypatch.setattr(hip_runtime.sys, "platform", "win32")
    monkeypatch.setattr(hip_runtime, "_library_names", lambda: ("amdhip64_7.dll",))
    read_maps = mock.Mock(side_effect=AssertionError("Windows has no proc maps"))
    monkeypatch.setattr(hip_runtime, "open", read_maps, raising=False)
    cdll = mock.Mock(return_value="dll")
    monkeypatch.setattr(hip_runtime.ctypes, "CDLL", cdll)
    assert hip_runtime.load_hip_runtime() == "dll"
    cdll.assert_called_once_with("amdhip64_7.dll")
    read_maps.assert_not_called()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux dlopen regression"
)
@pytest.mark.parametrize("visibility", ["local", "global"])
def test_real_loaded_file_wins_over_a_different_inode_copy(tmp_path, visibility):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("needs a C compiler for the tiny HIP stand-in")
    # Maps paths must preserve spaces and filesystem bytes outside UTF-8.
    core, devel = tmp_path / ("core SDK_" + chr(0xDCFF)), tmp_path / "devel SDK"
    core.mkdir()
    devel.mkdir()
    source = tmp_path / "hip.c"
    source.write_text("int hipRuntimeGetVersion(int *v) { *v = 70000000; return 0; }\n")
    library = core / "libamdhip64.so.7"
    subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            "-Wl,-soname,libamdhip64.so.7",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    shutil.copyfile(library, devel / library.name)
    (devel / "libamdhip64.so").symlink_to(library.name)
    assert not library.samefile(devel / library.name)
    script = """
import ctypes, importlib.util, os, sys
from pathlib import Path
source, core, devel, visibility = sys.argv[1:]
spec = importlib.util.spec_from_file_location('hip_runtime', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_cdll = ctypes.CDLL
loaded = original_cdll(core, mode=ctypes.RTLD_GLOBAL if visibility == 'global' else ctypes.RTLD_LOCAL)
calls = []
def redirected_cdll(name, mode=0):
    calls.append((name, mode))
    if name is not None and not os.path.isabs(name):
        name = str(Path(devel) / name)
    return original_cdll(name, mode=mode)
module.ctypes.CDLL = redirected_cdll
reused = module.load_hip_runtime()
assert reused._handle == loaded._handle, calls
assert calls == [(core, os.RTLD_NOLOAD | os.RTLD_LAZY)], calls
assert module.load_hip_runtime() is reused
print('PASS: reused loaded HIP file without mapping the devel copy')
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(_SOURCE),
            str(library),
            str(devel),
            visibility,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
