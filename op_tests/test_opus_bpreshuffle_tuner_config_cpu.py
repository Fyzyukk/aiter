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
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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


def _generate_data(m, n, k, seed, device=None):
    del m, n, k, seed, device
    return {"x": object(), "w": object(), "w_reference": object(),
            "out": _Output(), "x_scale": object(), "w_scale": object()}


def _reference(*args, **kwargs):
    return (args, kwargs)


def _compare(*args, **kwargs):
    del args, kwargs
    return 0.0


def _namespace():
    namespace = {
        "argparse": argparse, "Path": Path, "pd": pd, "math": math, "os": os,
        "torch": SimpleNamespace(float8_e4m3fn="torch.float8_e4m3fn", bfloat16="torch.bfloat16"),
        "logger": SimpleNamespace(info=lambda *args: None, error=lambda *args: None),
        "_read_csv": pd.read_csv, "object": object,
        "BpreshuffleConfig": catalog.BpreshuffleConfig,
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
        "_BENCH_KEYS": ("x", "w", "out", "x_scale", "w_scale"),
        "_REF_KEYS": ("x", "w_reference", "x_scale", "w_scale"),
        "_CK_REF_KEYS": ("x", "weight", "x_scale", "w_scale"),
        "_SCALE_GROUP_K": 128, "_SUPPORTED_LIBTYPES": frozenset({"opus", "ck", "cktile", "asm"}),
    }
    tree = ast.parse(TUNER.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"load_opus_configs", "prepare_opus_configs", "run_config_bench"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TUNER), "exec"), namespace)
    _class(BASE, "TunerCommon", "ScalarBase", "object",
           {"get_tuned_gemm_list", "post_process"}, namespace)
    _class(BASE, "GemmCommonTuner", "ScalarGemm", "ScalarBase", {"calculate"}, namespace)
    _class(GENERIC, "GemmA8W8BlockScaleTuner", "ScalarGeneric", "ScalarGemm",
           {"calculate", "result_to_df"}, namespace)
    tuner = _class(
        TUNER, "OpusMxscaleBpreshuffleTuner", "ScalarOpus", "ScalarGeneric",
        {"_normalize_rows", "_config_columns", "get_tuned_gemm_list", "_candidate_configs",
         "_candidate_kids", "_make_task", "get_gemm_a8w8_blockscale_opus_tune_task",
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


if __name__ == "__main__":
    unittest.main()
