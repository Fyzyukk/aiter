"""CPU integration of parameter JSON, tuning tasks, CSV, replay, and prepare.

Actual tuner methods are extracted with AST, retaining their inheritance and
using real CPU pandas. Only data generation, measurement and module preparation
are replaced by inert fakes; torch, aiter, HIP and HSA imports are rejected.
"""

import argparse
import ast
import builtins
from contextlib import contextmanager, redirect_stdout
import importlib.abc
import io
import itertools
import functools
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, MappingProxyType, ModuleType
from typing import NamedTuple
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OPUS = ROOT / "csrc/opus_gemm"
TUNER = OPUS / "opus_gemm_mxscale_bpreshuffle_tune.py"
GENERIC = ROOT / "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py"
BASE = ROOT / "aiter/utility/base_tuner.py"


class _RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "hip", "hsa", "flydsl"}:
            raise AssertionError(f"CPU tuner integration attempted GPU import: {fullname}")
        return None


@contextmanager
def _cpu_imports():
    guard = _RejectGPUImports()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


with _cpu_imports():
    import pandas as pd
    sys.path.insert(0, str(OPUS))
    try:
        import opus_gemm_bpreshuffle_config as catalog
        import opus_gemm_common as common
        import opus_gemm_bpreshuffle_policy as policy
    finally:
        sys.path.remove(str(OPUS))


def _class(path, original, renamed, base, methods, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == original)
    selected = [node for node in node.body
                if isinstance(node, ast.FunctionDef) and node.name in methods]
    found = {node.name for node in selected}
    if found != set(methods):
        raise AssertionError(f"Missing methods in {path}: {set(methods) - found}")
    extracted = ast.ClassDef(name=renamed, bases=[ast.Name(id=base, ctx=ast.Load())],
                             keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[renamed]


class _Output:
    def fill_(self, value):
        self.fill_value = value
        return self

    def view(self, *shape):
        return self


def _generate_data(m, n, k, seed, flydsl=False, device=None):
    del m, n, k, seed, device
    out = _Output()
    data = {"x": object(), "w": object(), "w_reference": object(),
            "out": out, "x_scale": object(), "w_scale": object()}
    if flydsl:
        data.update({key: out if key == "out_bmm" else object() for key in (
            "x_bmm", "w_bmm", "out_bmm", "x_scale_bmm", "w_scale_bmm", "x_scale_shuf", "w_scale_shuf"
        )})
    return data


def _flydsl_scalar_runtime():
    """Run the real BMM name grammar and legality checker without GPU imports."""
    namespace = dict(re=re, functools=functools, itertools=itertools,
                     MappingProxyType=MappingProxyType, NamedTuple=NamedTuple,
                     ceildiv=lambda a, b: (a + b - 1) // b, _LDS_BYTES=160 * 1024)
    paths = (
        (ROOT / "aiter/ops/flydsl/kernels/bmm_a8w8_mxscale_gfx950.py",
         {"_lds_layout", "_padded_k", "check_bmm_config", "_byte_slots", "_scale_panels"},
         {"WAVE", "MFMA_M", "MFMA_K", "SCALE_GROUP", "SCALE_BLOCKS"}),
        (ROOT / "aiter/ops/flydsl/batched_gemm_a8w8_gfx950.py",
         {"_launch_config", "parse_bmm_kernel_name", "bmm_kernel_name", "_runs", "_variants",
          "_max_split", "pick_bmm_kernel_name"},
         {"BMM_MFMA_NAME_PREFIX", "_BMM_KERNEL_NAME_RE", "_FALLBACK_FIELDS", "_FALLBACK"}),
    )
    for path, functions, constants in paths:
        nodes = []
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                nodes.append(node)
            elif isinstance(node, ast.ClassDef) and node.name == "_LdsLayout":
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets
            ):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _flydsl_catalog():
    path = ROOT / "aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py"
    module = ModuleType("_flydsl_cpu_catalog")
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def _reference(*args, **kwargs):
    return (args, kwargs)


def _compare(*args, **kwargs):
    del args, kwargs
    return 0.0


def _namespace():
    namespace = {
        "argparse": argparse, "Path": Path, "pd": pd, "math": math, "os": os,
        "itertools": itertools,
        "torch": SimpleNamespace(float8_e4m3fn="torch.float8_e4m3fn", bfloat16="torch.bfloat16"),
        "logger": SimpleNamespace(info=lambda *args: None, error=lambda *args: None),
        "_read_csv": pd.read_csv, "object": object,
        "BpreshuffleConfig": catalog.BpreshuffleConfig,
        "config_identity": catalog.config_identity,
        "construct_config": catalog.construct_config,
        "config_supports_shape": catalog.config_supports_shape,
        "validate_saved_config": catalog.validate_saved_config,
        "config_from_legacy_kid": catalog.config_from_legacy_kid,
        "kernel_instance_from_config": catalog.kernel_instance_from_config,
        "pipeline_configs": catalog.pipeline_configs,
        "bpreshuffle_candidate_split_k": common.bpreshuffle_candidate_split_k,
        "bpreshuffle_launch_plan": common.bpreshuffle_launch_plan,
        "canonical_output_dtype": common.canonical_output_dtype,
        "a8w8_mxscale_gemm_bpreshuffle_kernels_list": common.a8w8_mxscale_gemm_bpreshuffle_kernels_list,
        "generate_data": _generate_data, "run_torch": _reference, "compare_outputs": _compare,
        "compare_flydsl_outputs": _compare,
        "kernels_list_flydsl": {}, "fits_shape_flydsl": None,
        "generate_data_e8m0": _generate_data, "run_torch_e8m0": _reference,
        "run_gemm_a8w8_blockscale_flydsl": _reference,
        "dtypes": SimpleNamespace(bf16="torch.bfloat16"),
        "_BENCH_KEYS": ("x", "w", "out", "x_scale", "w_scale"),
        "_REF_KEYS": ("x", "w_reference", "x_scale", "w_scale"),
        "_CK_REF_KEYS": ("x", "weight", "x_scale", "w_scale"),
        "_SCALE_GROUP_K": 128,
    }
    tree = ast.parse(TUNER.read_text())
    constants = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in {
            "_NATIVE_LIBTYPES", "_SUPPORTED_LIBTYPES", "_FLYDSL_BMM_KEYS", "_FLYDSL_KEYS"
        } for target in node.targets
    )]
    exec(compile(ast.Module(body=constants, type_ignores=[]), str(TUNER), "exec"), namespace)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"load_opus_configs", "prepare_opus_configs", "run_config_bench",
                                   "selected_backends", "flydsl_modules", "validate_flydsl_bmm",
                                   "flydsl_bmm_candidate_names", "prepare_flydsl_data"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TUNER), "exec"), namespace)
    _class(BASE, "TunerCommon", "ScalarBase", "object",
           {"get_tuned_gemm_list", "post_process"}, namespace)
    _class(BASE, "GemmCommonTuner", "ScalarGemm", "ScalarBase", {"calculate"}, namespace)
    _class(GENERIC, "GemmA8W8BlockScaleTuner", "ScalarGeneric", "ScalarGemm",
           {"calculate", "result_to_df", "get_gemm_a8w8_blockscale_flydsl_tune_task"}, namespace)
    tuner = _class(
        TUNER, "OpusMxscaleBpreshuffleTuner", "ScalarOpus", "ScalarGeneric",
        {"_normalize_rows", "_config_columns", "get_tuned_gemm_list", "_candidate_configs",
         "_candidate_kids", "_make_task", "get_gemm_a8w8_blockscale_opus_tune_task",
         "get_gemm_a8w8_blockscale_flydsl_bmm_tune_task", "get_gemm_a8w8_blockscale_flydsl_tune_task",
         "tune", "post_process", "getKernelName", "calculate", "result_to_df",
         "run_config", "_error_limit"}, namespace,
    )
    return namespace, tuner


class BpreshuffleTunerConfigCPU(unittest.TestCase):
    def setUp(self):
        self.guard = _cpu_imports()
        self.guard.__enter__()
        self.addCleanup(self.guard.__exit__, None, None, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.namespace, self.tuner_type = _namespace()
        self.tuner = self.tuner_type()
        self.tuner.keys = ["gfx", "cu_num", "M", "N", "K"]
        self.tuner.columns = [*self.tuner.keys, "libtype", "kernelId", "splitK", "us",
                              "kernelName", "tflops", "bw", "errRatio", "pipeline", "config"]
        self.tuner.get_gfx = lambda: "gfx950"
        self.tuner.get_cu_num = lambda: 256
        self.tuner.update_config_files = lambda path, name: str(path)
        self.tuner.name = "scalar_opus"
        self.tuner.opus_kids = None
        self.tuner.opus_families = None
        self.tuner.opus_pipelines = frozenset({"register"})
        self.tuner.opus_configs = ()
        self.tuner.INVALID_TIME, self.tuner.INF_TIME, self.tuner.topk = -1, math.inf, 0
        self.prepared, self.bench_calls, self.measurements = [], [], []
        original_import = builtins.__import__

        def ensure_config(config):
            self.prepared.append(config)
            return SimpleNamespace(config=config)

        def launch(*args, **kwargs):
            self.bench_calls.append((args, kwargs))
            return args[2]

        def measure(bench, *args, **kwargs):
            self.measurements.append(kwargs)
            return bench(*args), 7.25

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aiter.ops.opus.bpreshuffle_runtime":
                return SimpleNamespace(ensure_config=ensure_config)
            if name == "aiter.ops.opus":
                return SimpleNamespace(opus_gemm_bpreshuffle=launch)
            if name == "aiter.test_common":
                return SimpleNamespace(checkAllclose=_compare, run_perftest=measure)
            return original_import(name, globals, locals, fromlist, level)

        self.import_patch = patch("builtins.__import__", fake_import)
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def _json(self, rows, name="configs.json"):
        path = Path(self.temp.name) / name
        path.write_text(json.dumps(rows))
        return path

    def _configs(self):
        axes = {"tile_m": 16, "tile_n": 32, "runtime_split_k": True, "prefetch": 5}
        path = self._json([{"pipeline": "register", "compile_params": axes}] * 2)
        configs = self.namespace["load_opus_configs"](path)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].legacy_kid, -1)
        return configs

    def _args(self, **kwargs):
        return argparse.Namespace(libtype="opus", opus_kids=None,
                                  opus_pipelines=frozenset({"register"}), warmup=2, iters=5,
                                  splitK=0, mp=1, shape_grouped=True, errRatio=0.05,
                                  timeout=60, verbose=False, profile_file="", **kwargs)

    def _install_flydsl(self):
        runtime, catalog = _flydsl_scalar_runtime(), _flydsl_catalog()

        def bench(*args):
            self.bench_calls.append(args)
            return args[2]

        self.namespace.update(
            flydsl_modules=lambda backend: runtime if backend == "flydsl_bmm" else (runtime, catalog),
            run_flydsl_bmm_bench=bench, run_flydsl_bench=bench,
            kernels_list_flydsl=catalog.kernels_list, fits_shape_flydsl=catalog.fits_shape,
            run_gemm_a8w8_blockscale_flydsl=lambda x, w, xs, ws, out, name: out,
        )
        return runtime, catalog

    def test_all_and_native_selection_dispatch_and_share_native_input_cache_key(self):
        runtime, _ = self._install_flydsl()
        name = runtime.bmm_kernel_name(16, 32, 256, 1, 1, 2, splits=2)
        key = ("gfx950", 256, 16, 256, 1024)
        frame = pd.DataFrame([dict(zip(self.tuner.keys, key))])
        for libtype, expected in (
            ("all", {"ck", "cktile", "asm", "opus", "flydsl", "flydsl_bmm"}),
            ("opus_flydsl", {"opus", "flydsl", "flydsl_bmm"}),
        ):
            args = self._args()
            args.libtype, args.blockPerCu = libtype, [1]
            args.flydsl_bmm_kernels = (name,)
            for backend, method in (
                ("ck", "get_gemm_a8w8_blockscale_tune_task"),
                ("cktile", "get_gemm_a8w8_blockscale_cktile_tune_task"),
                ("asm", "get_gemm_a8w8_blockscale_asm_tune_task"),
            ):
                setattr(self.tuner, method, lambda *a, backend=backend: [
                    self.tuner._make_task((key, 0, 0, "external", backend, True),
                                         16, 256, 1024, 0, _reference, (), (), {})
                ])
            dispatched = []

            def mp_tuner(tasks, groups, *a, **kw):
                self.assertEqual(groups, [(len(tasks), ())])
                dispatched.extend(tasks)
                return []

            self.namespace["generic_tune"] = SimpleNamespace(mp_tuner=mp_tuner)
            self.tuner.tune(frame, pd.DataFrame(), args)
            self.assertEqual({task[0][4] for task in dispatched}, expected)
            native = [task for task in dispatched if task[0][4] in {"opus", "flydsl", "flydsl_bmm"}]
            self.assertEqual({(id(task[1]), task[2]) for task in native},
                             {(id(_generate_data), (16, 256, 1024, 0, True))})
            self.assertTrue(all(task[6] is _reference and task[12] is _compare for task in native))

    def test_bmm_parameter_search_uses_real_legality_and_explicit_names_bypass_pruning(self):
        runtime, _ = self._install_flydsl()
        search = self.namespace["flydsl_bmm_candidate_names"]
        names = search(runtime, 16, 256, 1024, 256)
        self.assertGreater(len(names), 1)
        configs = [self.namespace["validate_flydsl_bmm"](runtime, name, 16, 256, 1024) for name in names]
        self.assertGreater(len({config["num_buffers"] for config in configs}), 1)
        self.assertGreater(len({config["splits"] for config in configs}), 1)
        self.assertEqual({config["b_direct"] for config in configs}, {False, True})
        self.assertEqual({config["scale_preload"] for config in configs}, {False, True})
        self.assertEqual(len(names), len(set(names)))
        exact = runtime.bmm_kernel_name(128, 64, 128, 1, 4, 2)
        self.assertNotIn(exact, names)  # default search avoids extreme M padding
        self.assertEqual(search(runtime, 16, 256, 1024, 256, (exact,)), (exact,))
        self.assertTrue(set(names).issubset(search(runtime, 16, 256, 1024, 256, search="full")))
        with self.assertRaises(ValueError):
            search(runtime, 16, 256, 1024, 256, ("invalid",))

    def test_mixed_numeric_and_legacy_bmm_csv_roundtrip_and_exact_replay(self):
        runtime, catalog = self._install_flydsl()
        key = ("gfx950", 256, 16, 256, 1024)
        name = runtime.bmm_kernel_name(16, 32, 256, 1, 1, 2, splits=2)
        self.tuner.flydsl_bmm_kernels = (name,)
        bmm = self.tuner.get_gemm_a8w8_blockscale_flydsl_bmm_tune_task(key, 0, True, {})[0]
        self.assertEqual(bmm[14], ("out_bmm",))  # NaN-init must reach the 3-D output argument
        numeric = next(task for task in self.tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(key, 0, True, {})
                       if task[0][2] == 2)
        self.assertIs(numeric[3], self.namespace["run_gemm_a8w8_blockscale_flydsl"])
        expected = [(kid, ki.name) for kid, ki in catalog.candidates_for("fp8", "fp8", 16, 256, 1024)]
        actual = [(task[0][1], task[0][3]) for task in
                  self.tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(key, 0, True, {})]
        self.assertEqual(actual, expected)
        self.assertEqual(numeric[0][3], catalog.kernels_list[numeric[0][1]].name)
        infos = [bmm[0], numeric[0]]
        frame = self.tuner.result_to_df([(info, 7.25, 0.0) for info in infos])
        frame.loc[frame.libtype.eq("flydsl_bmm"), "libtype"] = "flydsl"  # upstream spelling
        path = Path(self.temp.name) / "mixed.csv"
        frame.to_csv(path, index=False)
        loaded = self.tuner.get_tuned_gemm_list(path)
        self.assertEqual(list(loaded.libtype), ["flydsl_bmm", "flydsl"])
        self.assertEqual(list(loaded.kernelId), ["bmm", numeric[0][1]])
        normalized = self.tuner._normalize_rows(loaded, saved=True)
        self.assertEqual(list(normalized.scale_dtype), ["e8m0", "e8m0"])
        self.tuner.untunedf = loaded
        result = self.tuner.run_config(self._args())
        self.assertEqual([row["status"] for row in result], ["ok", "ok"])
        self.assertEqual([call[-1] for call in self.bench_calls], [name, numeric[0][3]])
        self.assertFalse(self.prepared)  # FlyDSL-only replay does not compile OPUS

    def test_mixed_profile_selection_keeps_raw_flydsl_errors_and_backend_thresholds(self):
        runtime, catalog = self._install_flydsl()
        key = ("gfx950", 256, 16, 256, 1024)
        name = runtime.bmm_kernel_name(16, 32, 256, 1, 1, 2)
        kid, kernel = catalog.candidates_for("fp8", "fp8", 16, 256, 1024)[0]
        config = self._configs()[0]
        infos = [
            (key, config, 1, self.namespace["kernel_instance_from_config"](config).name, "opus", True),
            (key, "bmm", 1, name, "flydsl_bmm", True),
            (key, kid, kernel.split_k, kernel.name, "flydsl", True),
        ]
        results = [(infos[0], 0.5, 0.0001), (infos[1], 0.75, 0.06), (infos[2], 1.0, 0.01)]
        profile = Path(self.temp.name) / "mixed_profile.csv"
        args = self._args()
        args.profile_file = str(profile)
        with redirect_stdout(io.StringIO()):
            selected = self.tuner.post_process(results, args, topk=1)
        self.assertEqual(selected.at[0, "libtype"], "flydsl")
        raw = pd.read_csv(profile)
        self.assertEqual(set(raw.libtype), {"opus", "flydsl", "flydsl_bmm"})
        self.assertEqual(sorted(raw.us), [0.5, 0.75, 1.0])
        self.assertEqual(sorted(raw.errRatio), [0.0001, 0.01, 0.06])

    def test_stale_flydsl_metadata_is_rejected_before_any_build_or_measurement(self):
        runtime, _ = self._install_flydsl()
        key = ("gfx950", 256, 16, 256, 1024)
        name = runtime.bmm_kernel_name(16, 32, 256, 1, 1, 2, splits=2)
        self.tuner.flydsl_bmm_kernels = (name,)
        bmm = self.tuner.get_gemm_a8w8_blockscale_flydsl_bmm_tune_task(key, 0, True, {})[0][0]
        numeric = self.tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(key, 0, True, {})[0][0]
        frame = self.tuner.result_to_df([(bmm, 7.25, 0.0), (numeric, 7.25, 0.0)])
        for index, column, value in (
            (0, "splitK", 1), (0, "kernelName", "wrong"), (0, "kernelId", 1),
            (0, "N", 144), (1, "splitK", 0), (1, "kernelName", name), (1, "kernelId", 999999),
        ):
            with self.subTest(index=index, column=column):
                modified = frame.copy()
                modified.at[index, column] = value
                self.tuner.untunedf = modified
                with self.assertRaises(ValueError):
                    self.tuner.run_config(self._args())
                self.assertFalse(self.prepared)
                self.assertFalse(self.bench_calls)
                self.assertFalse(self.measurements)

    def test_missing_flydsl_fails_all_before_opus_prepare_and_dispatch(self):
        args = self._args()
        args.libtype = "all"
        self.tuner.opus_configs = self._configs()
        frame = pd.DataFrame([dict(gfx="gfx950", cu_num=256, M=16, N=256, K=1024)])
        with self.assertRaisesRegex(AssertionError, "CPU tuner integration attempted GPU import"):
            self.tuner.tune(frame, pd.DataFrame(), args)
        self.assertFalse(self.prepared)
        self.assertFalse(self.measurements)

    def test_config_json_tasks_result_csv_and_replay_preserve_exact_tuple(self):
        configs = self._configs()
        config = configs[0]
        self.tuner.opus_configs = configs
        shape_key = ("gfx950", 256, 16, 256, 7168)
        tasks = self.tuner.get_gemm_a8w8_blockscale_opus_tune_task(
            shape_key, 19, True, {"num_warmup": 2, "num_iters": 5}
        )
        custom = [task for task in tasks if task[0][1] == config]
        self.assertEqual([task[0][2] for task in custom], list(range(1, 17)))
        for task in custom:
            self.assertIs(task[4][1], config)
            self.assertEqual(task[4][2], task[0][2])
            self.assertEqual(task[2], (16, 256, 7168, 19))
        task = next(task for task in custom if task[0][2] == 3)
        info = task[0]
        frame = self.tuner.result_to_df([(info, 7.25, 0.0)])
        self.assertEqual((frame.at[0, "kernelId"], frame.at[0, "pipeline"], frame.at[0, "config"]),
                         (-1, config.pipeline, config.to_json()))
        for omit_id in (False, True):
            path = Path(self.temp.name) / f"saved{omit_id}.csv"
            (frame.drop(columns=["kernelId"]) if omit_id else frame).to_csv(path, index=False)
            loaded = self.tuner.get_tuned_gemm_list(path)
            self.assertEqual(list(loaded.columns), self.tuner.columns)
            self.assertEqual(loaded.at[0, "kernelId"], -1)
            self.tuner.untunedf = loaded
            self.prepared.clear()
            self.bench_calls.clear()
            self.assertEqual(self.tuner.run_config(self._args())[0]["status"], "ok")
            self.assertEqual(self.prepared, [config])
            self.assertEqual(self.bench_calls[0][1],
                             {"pipeline": "register", "config": config, "split_k": 3})
            self.assertEqual(policy.select_config((16, 256, 7168), tuned_file=path).config, config)

    def test_tune_prepares_before_dispatching_config_tasks(self):
        configs = self._configs()
        self.tuner.opus_configs = configs
        frame = pd.DataFrame([dict(gfx="gfx950", cu_num=256, M=16, N=256, K=7168)])
        dispatched = []

        def mp_tuner(tasks, *args, **kwargs):
            del args, kwargs
            self.assertIn(configs[0], self.prepared)
            dispatched.extend(tasks)
            return [(task[0], 7.25, 0.0) for task in tasks]

        self.namespace["generic_tune"] = SimpleNamespace(mp_tuner=mp_tuner)
        results = self.tuner.tune(frame, pd.DataFrame(), self._args())
        self.assertTrue(dispatched)
        custom = [info for info, _, _ in results if info[1] == configs[0]]
        self.assertEqual(len(custom), 16)
        self.assertEqual(len(self.prepared), len({(c.pipeline, c.to_json()) for c in self.prepared}))

    def test_profile_sort_accepts_multiple_config_objects_and_preserves_errors(self):
        one = self._configs()[0]
        two = catalog.construct_config("register", tile_m=16, tile_n=32,
                                       runtime_split_k=True, prefetch=7)
        key = ("gfx950", 256, 16, 256, 7168)
        infos = [(key, config, 3, catalog.kernel_instance_from_config(config).name, "opus", True)
                 for config in (one, two)]
        results = [(infos[1], 2.0, 0.01), (infos[0], 7.25, 0.0)]
        profile = Path(self.temp.name) / "profile.csv"
        args = self._args()
        args.profile_file = str(profile)
        with redirect_stdout(io.StringIO()):
            selected = self.tuner.post_process(results, args, topk=1)
        self.assertEqual(selected.at[0, "config"], one.to_json())
        raw = pd.read_csv(profile)
        self.assertEqual(set(raw["config"]), {one.to_json(), two.to_json()})
        self.assertEqual(sorted(raw["errRatio"]), [0.0, 0.01])
        self.assertEqual(sorted(raw["us"]), [2.0, 7.25])

    def test_json_and_stale_saved_metadata_fail_before_prepare_or_measurement(self):
        bad = [[], {}, [{"pipeline": "register"}], [{"pipeline": "register", "compile_params": []}],
               [{"pipeline": "register", "compile_params": {"tile_m": 17, "tile_n": 32}}]]
        for index, payload in enumerate(bad):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.namespace["load_opus_configs"](self._json(payload, f"bad{index}.json"))
        config = self._configs()[0]
        row = dict(gfx="gfx950", cu_num=256, M=16, N=256, K=7168, libtype="opus",
                   kernelId=92320, splitK=3, pipeline="register", config=config.to_json())
        self.tuner.untunedf = pd.DataFrame([row])
        with self.assertRaisesRegex(ValueError, "does not match kernelId"):
            self.tuner.run_config(self._args())
        self.assertFalse(self.prepared)
        self.assertFalse(self.bench_calls)
        self.assertFalse(self.measurements)

    def test_oct9_extra_config_metadata_loads_and_rejects_stale_identity_or_id(self):
        config = self._configs()[0]
        row = {"pipeline": config.pipeline, "compile_params": json.loads(config.to_json()),
               "legacy_kid": -1, "identity": catalog.config_identity(config)}
        self.assertEqual(self.namespace["load_opus_configs"](self._json([row])), (config,))
        for column, value in (("identity", "stale"), ("legacy_kid", 92320),
                              ("legacy_kid", "-1"), ("unknown", 1)):
            with self.subTest(column=column):
                bad = {**row, column: value}
                with self.assertRaises(ValueError):
                    self.namespace["load_opus_configs"](self._json([bad]))


if __name__ == "__main__":
    unittest.main()
