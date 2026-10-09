"""CPU-only checks for pipeline selection and old/new tuning CSV metadata.

Run with ``PYTHONDONTWRITEBYTECODE=1 python3 op_tests/test_opus_bpreshuffle_config_cpu.py``.
Only scalar production modules and AST-extracted tuner methods are imported.
"""

import ast
import argparse
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OPUS = ROOT / "csrc/opus_gemm"


class _RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "hip", "hsa", "flydsl"}:
            raise AssertionError(f"CPU config check attempted GPU import: {fullname}")
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
    sys.path.insert(0, str(OPUS))
    try:
        import opus_gemm_bpreshuffle_config as catalog
        import opus_gemm_common as common
    finally:
        sys.path.remove(str(OPUS))


class _Rows:
    """Tiny DataFrame substitute for exercising the real scalar CSV method."""
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def copy(self):
        return _Rows(self.rows)

    def __contains__(self, key):
        return all(key in row for row in self.rows)

    def __setitem__(self, key, values):
        if isinstance(key, str):
            values = [values] * len(self.rows) if isinstance(values, str) else values
            for row, value in zip(self.rows, values):
                row[key] = value
        else:
            raise AssertionError(key)

    def itertuples(self, index=False):
        return iter(SimpleNamespace(**row) for row in self.rows)


def _tuner_definitions():
    path = OPUS / "opus_gemm_mxscale_bpreshuffle_tune.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"parse_opus_kids", "parse_opus_pipelines"}]
    tuner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "OpusMxscaleBpreshuffleTuner")
    selected += [node for node in tuner.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_config_columns", "_candidate_configs", "_candidate_kids"}]
    namespace = {
        "argparse": argparse, "BPRESHUFFLE_PIPELINE_NAMES": catalog.BPRESHUFFLE_PIPELINE_NAMES,
        "pd": SimpleNamespace(notna=lambda value: value is not None),
        "validate_saved_config": catalog.validate_saved_config,
        "config_from_legacy_kid": catalog.config_from_legacy_kid,
        "pipeline_configs": catalog.pipeline_configs,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class BpreshuffleConfigCPU(unittest.TestCase):
    def test_all_exact_ids_have_unique_frozen_configs_and_five_pipelines(self):
        self.assertEqual(set(catalog.CONFIGS_BY_LEGACY_KID), set(common.a8w8_mxscale_gemm_bpreshuffle_kernels_list))
        self.assertEqual(len(catalog.CONFIGS_BY_LEGACY_KID), 105)
        self.assertEqual(set(catalog.BPRESHUFFLE_PIPELINES), {"pin", "tiled", "register", "lds", "large_output"})
        self.assertEqual({name: len(pipeline.active_configs) for name, pipeline in catalog.BPRESHUFFLE_PIPELINES.items()},
                         {"pin": 15, "tiled": 20, "register": 13, "lds": 38, "large_output": 3})
        unique = {(config.pipeline, config.to_json()) for config in catalog.CONFIGS_BY_LEGACY_KID.values()}
        self.assertEqual(len(unique), 105)
        config = catalog.config_from_legacy_kid(9000)
        with self.assertRaises(FrozenInstanceError):
            config.tile_m = 128
        with self.assertRaises(TypeError):
            config.compile_params["tile_m"] = 128

    def test_complete_payload_roundtrip_and_no_numeric_abi_keys(self):
        for kid, config in catalog.CONFIGS_BY_LEGACY_KID.items():
            with self.subTest(kid=kid):
                payload = config.to_json()
                self.assertEqual(payload, json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")))
                self.assertEqual(set(json.loads(payload)), set(config.compile_params))
                self.assertNotIn("kid", payload)
                self.assertNotIn("pipeline", json.loads(payload))
                self.assertIs(catalog.resolve_config(config.pipeline, payload), config)
                self.assertIs(catalog.validate_saved_config(config.pipeline, payload, kid), config)
                self.assertIs(catalog.resolve_config(config.pipeline, config), config)

    def test_filtering_reproduces_legacy_registry_for_shape_boundaries(self):
        shapes = [(1, 128, 128), (16, 256, 384), (64, 256, 7168),
                  (256, 256, 7168), (512, 1024, 16384), (2048, 128, 1536),
                  (0, 128, 128), (262144, 8192, 1536)]
        for shape in shapes:
            for include_legacy in (False, True):
                expected = common.a8w8_mxscale_bpreshuffle_candidate_kids("gfx950", *shape, include_legacy=include_legacy)
                actual = catalog.pipeline_configs(shape, include_legacy=include_legacy)
                self.assertEqual([config.legacy_kid for config in actual], expected)
                for pipeline in catalog.BPRESHUFFLE_PIPELINES:
                    filtered = catalog.pipeline_configs(shape, [pipeline], include_legacy=include_legacy)
                    self.assertEqual(filtered, tuple(config for config in actual if config.pipeline == pipeline))
        self.assertEqual(catalog.pipeline_configs((256, 256, 7168), gfx="gfx942"), ())
        self.assertEqual(catalog.pipeline_configs((256, 256, 7168), outdtype="fp32"), ())
        with self.assertRaises(ValueError):
            catalog.pipeline_configs((256, 256, 7168), ["unknown"])
        with self.assertRaises(ValueError):
            catalog.pipeline_configs((256, 256, 7168), [])

    def test_compile_parameters_retain_actual_axes_and_runtime_split_separation(self):
        config = catalog.config_from_legacy_kid
        self.assertEqual((config(9043).stages, config(9043).cluster), (8, 2))
        self.assertEqual((config(9055).stages, config(9055).cluster), (12, 4))
        self.assertTrue(config(9049).register_scales)
        self.assertTrue(config(9049).xor_lds)
        self.assertEqual((config(9071).wave_k, config(9071).output_mode, config(9071).prefetch), (4, 4, 4))
        self.assertTrue(config(9071).pad_n)
        self.assertEqual(config(92003).scale_panel, 8)
        self.assertEqual(config(92011).group_m, 8)
        self.assertEqual(config(92103).scale_panel, 64)
        self.assertTrue(config(92113).scale_reset)
        self.assertEqual(config(9020).scale_panel, 128)
        self.assertEqual(config(9030).scale_panel, 128)
        self.assertEqual((config(92501).b_direct_sets, config(92501).c_chunk_rows), (2, 96))
        self.assertEqual(config(92501).b_ahead, 0)
        for kid in (92310, 92311, 92320, 92321, 92330, 92340, 92410, 92420, 92430):
            self.assertIsNone(config(kid).split_k)
            self.assertTrue(config(kid).runtime_split_k)
        for alias in (92411, 92421, 92431):
            self.assertEqual(config(alias).legacy_kid, alias)
            self.assertEqual(config(alias).split_k, 2)
            self.assertFalse(config(alias).runtime_split_k)

    def test_resolver_rejects_unsupported_ambiguous_and_malformed_payloads(self):
        config = catalog.config_from_legacy_kid(92310)
        self.assertIs(catalog.resolve_config("register", tile_m=16, tile_n=16, wave_k=1), config)
        for params in ({}, {"tile_m": 16}, {"tile_m": 17}, {"tile_m": 16.0}, {"kid": 92310}):
            with self.assertRaises(ValueError):
                catalog.resolve_config("register", **params)
        mutations = [dict(config.compile_params, tile_m=32),
                     dict(config.compile_params, tile_m=16.0),
                     dict(config.compile_params, runtime_split_k=1),
                     {"tile_m": 16}]
        for payload in mutations:
            with self.assertRaises(ValueError):
                catalog.validate_saved_config("register", payload, 92310)
        with self.assertRaises(ValueError):
            catalog.validate_saved_config("lds", config.to_json(), 92310)
        for payload in ('[]', 'null', '{"tile_m":16,"tile_m":32}', 'broken'):
            with self.assertRaises(ValueError):
                catalog.resolve_config("register", payload)
        fine = catalog.config_from_legacy_kid(9060)
        values = json.loads(fine.to_json())
        values["specializations"][0][0] = float(values["specializations"][0][0])
        with self.assertRaises(ValueError):
            catalog.validate_saved_config("lds", values, 9060)

    def test_tuner_pipeline_flags_and_legacy_subset_intersection(self):
        methods = _tuner_definitions()
        self.assertEqual(methods["parse_opus_pipelines"]("register,lds"), {"register", "lds"})
        self.assertEqual(methods["parse_opus_kids"]("92310,92410"), {92310, 92410})
        for value in ("", "register,register", "register,unknown", "register,"):
            with self.assertRaises(argparse.ArgumentTypeError):
                methods["parse_opus_pipelines"](value)
        tuner = SimpleNamespace(opus_kids={92310, 92410}, opus_families=None,
                                opus_pipelines={"register"})
        tuner._candidate_configs = lambda *args: methods["_candidate_configs"](tuner, *args)
        self.assertEqual(methods["_candidate_kids"](tuner, "gfx950", 16, 128, 128), [92310])
        tuner.opus_kids = {92411}
        tuner.opus_pipelines = {"lds"}
        self.assertEqual(methods["_candidate_kids"](tuner, "gfx950", 16, 128, 128), [92411])

    def test_tuner_csv_legacy_backfill_new_validation_and_backend_blanks(self):
        normalize = _tuner_definitions()["_config_columns"]
        legacy = _Rows([{"libtype": "opus", "kernelId": 92310},
                        {"libtype": "opus", "kernelId": 92411},
                        {"libtype": "ck", "kernelId": 1}])
        result = normalize(legacy, validate_saved=True)
        self.assertEqual(result.rows[0]["pipeline"], "register")
        self.assertEqual(result.rows[1]["config"], catalog.config_from_legacy_kid(92411).to_json())
        self.assertEqual((result.rows[2]["pipeline"], result.rows[2]["config"]), ("", ""))
        self.assertEqual(normalize(result, validate_saved=True).rows, result.rows)
        for changed in ({"pipeline": "lds"}, {"config": ""}, {"kernelId": 92311}):
            row = dict(result.rows[0], **changed)
            with self.assertRaises(ValueError):
                normalize(_Rows([row]), validate_saved=True)
        with self.assertRaises(ValueError):
            normalize(_Rows([dict(result.rows[0], libtype="ck")]), validate_saved=True)
        # Serialization fills metadata for a newly measured result frame.
        self.assertEqual(normalize(legacy).rows, result.rows)

    def test_public_wrapper_resolves_parameters_and_forwards_launch_arguments(self):
        path = ROOT / "aiter/ops/opus/__init__.py"
        node = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "opus_gemm_bpreshuffle")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        calls = []

        def dispatch(*args, **kwargs):
            calls.append((args, kwargs))
            return args[4]

        namespace = {"_opus_dispatch": dispatch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"), namespace)
        wrapper = namespace["opus_gemm_bpreshuffle"]
        tensors = [object() for _ in range(5)]
        workspace = object()
        config = catalog.config_from_legacy_kid(92310)
        package_module = "csrc.opus_gemm.opus_gemm_bpreshuffle_config"
        with _cpu_imports(), patch.dict(sys.modules, {package_module: catalog}):
            for payload in (config, config.to_json(), dict(config.compile_params)):
                self.assertIs(wrapper(*tensors, pipeline="register", config=payload,
                                      split_k=3, workspace=workspace), tensors[2])
            self.assertIs(wrapper(*tensors, pipeline="register", tile_m=16, tile_n=16,
                                  wave_k=1, split_k=1), tensors[2])
            self.assertEqual(len(calls), 4)
            for args, kwargs in calls:
                self.assertEqual(args, ("opus_gemm", 2, *tensors[:3]))
                self.assertEqual(kwargs["kid"], 92310)
                self.assertEqual(kwargs["layout"], "bpreshuffle")
                self.assertIs(kwargs["x_scale"], tensors[3])
                self.assertIs(kwargs["w_scale"], tensors[4])
            for _, kwargs in calls[:3]:
                self.assertEqual(kwargs["split_k"], 3)
                self.assertIs(kwargs["workspace"], workspace)
            self.assertEqual(calls[3][1]["split_k"], 1)
            self.assertIsNone(calls[3][1]["workspace"])
            for bad in ({"pipeline": "unknown"}, {"pipeline": "register", "tile_m": 17},
                        {"pipeline": "register", "config": "broken"},
                        {"pipeline": "lds", "config": config},
                        {"pipeline": "register", "config": config, "tile_m": 16}):
                with self.assertRaises(ValueError):
                    wrapper(*tensors, **bad)
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
