"""CPU-only contracts for OPUS B-preshuffle tuned selection and fallback.

Run ``PYTHONDONTWRITEBYTECODE=1 python3 op_tests/test_opus_bpreshuffle_policy_cpu.py``.
GPU packages are rejected during scalar imports and all policy calls.
"""

from contextlib import contextmanager
import csv
from dataclasses import FrozenInstanceError
import importlib.abc
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OPUS = ROOT / "csrc/opus_gemm"


class _RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "hip", "hsa", "flydsl"}:
            raise AssertionError(f"CPU policy check attempted GPU import: {fullname}")
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
        import opus_gemm_bpreshuffle_policy as policy
        import opus_gemm_common as common
    finally:
        sys.path.remove(str(OPUS))


def _row(kid=9041, *, shape=(1, 768, 7168), split=0, **values):
    m, n, k = shape
    return {"gfx": "gfx950", "cu_num": 256, "M": m, "N": n, "K": k,
            "libtype": "opus", "kernelId": kid, "splitK": split, "us": 5,
            **values}


class BpreshufflePolicyCPU(unittest.TestCase):
    def setUp(self):
        self.import_guard = _cpu_imports()
        self.import_guard.__enter__()
        self.environment = patch.dict("os.environ", {policy.TUNED_CONFIG_ENV: ""})
        self.environment.start()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.import_guard.__exit__, None, None, None)

    def _csv(self, rows, name="tuned.csv", columns=None):
        path = Path(self.temp.name) / name
        if columns is None:
            columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _without_tuned(self, shape, **kwargs):
        with patch.object(policy, "DEFAULT_TUNED_CONFIG", Path(self.temp.name) / "missing.csv"):
            return policy.select_config(shape, **kwargs)

    def _assert_legal(self, selection, shape):
        instance = policy._instance(selection.config)
        self.assertTrue(common.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape))
        plan = common.bpreshuffle_launch_plan(instance, *shape, selection.split_k)
        if selection.config.runtime_split_k:
            if selection.split_k:
                self.assertEqual(selection.split_k, plan.split_k)
            self.assertGreaterEqual(selection.split_k, 0)
        else:
            self.assertEqual(selection.split_k, 0)

    def test_production_csv_replays_all_693_opus_rows_and_external_fallbacks(self):
        with policy.DEFAULT_TUNED_CONFIG.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 745)
        opus_rows = [row for row in rows if row["libtype"] == "opus"]
        self.assertEqual(len(opus_rows), 693)
        for row in opus_rows:
            shape = tuple(int(row[name]) for name in ("M", "N", "K"))
            with self.subTest(shape=shape):
                selected = policy.select_config(shape, gfx=row["gfx"], cu_num=int(row["cu_num"]))
                self.assertEqual(selected.source, "tuned")
                self.assertEqual(selected.config.legacy_kid, int(row["kernelId"]))
                self.assertEqual(selected.split_k, int(row["splitK"]))
                self._assert_legal(selected, shape)
        for row in rows:
            if row["libtype"] == "opus":
                continue
            shape = tuple(int(row[name]) for name in ("M", "N", "K"))
            with self.subTest(backend=row["libtype"], shape=shape):
                selected = policy.select_config(shape)
                self.assertEqual(selected.source, "heuristic")
                self._assert_legal(selected, shape)

    def test_complete_shape_and_device_key_and_pipeline_filter(self):
        row = _row()
        path = self._csv([row])
        selected = policy.select_config((1, 768, 7168), tuned_file=path)
        self.assertEqual(selected.config.pipeline, "register")
        self.assertEqual(selected.source, "tuned")
        self.assertEqual(policy.select_config((1, 768, 7168), cu_num=304, tuned_file=path).source,
                         "heuristic")
        self.assertEqual(policy.select_config((2, 768, 7168), tuned_file=path).source, "heuristic")
        selected = policy.select_config((1, 768, 7168), pipeline="lds", tuned_file=path)
        self.assertEqual((selected.config.pipeline, selected.source), ("lds", "heuristic"))
        selected = policy.select_config((1, 768, 7168), pipeline="register,lds", tuned_file=path)
        self.assertEqual(selected.source, "tuned")
        selected = policy.select_config((1, 768, 7168), pipeline=catalog.BPRESHUFFLE_PIPELINES["register"],
                                        tuned_file=path)
        self.assertEqual(selected.source, "tuned")
        with self.assertRaisesRegex(ValueError, "gfx950"):
            policy.select_config((1, 768, 7168), gfx="gfx942", tuned_file=path)

    def test_old_and_new_metadata_and_runtime_literal_split(self):
        config = catalog.config_from_legacy_kid(92320)
        shape = (16, 256, 7168)
        path = self._csv([_row(92320, shape=shape, split=3,
                               pipeline=config.pipeline, config=config.to_json())])
        selected = policy.select_config(shape, tuned_file=path)
        self.assertEqual((selected.config, selected.split_k, selected.source), (config, 3, "tuned"))
        selected = policy.select_config(shape, pipeline="register", split_k=5, tuned_file=path)
        self.assertEqual((selected.config.pipeline, selected.split_k, selected.source),
                         ("register", 5, "heuristic"))
        path = self._csv([_row(92320, shape=shape, split=0)])
        self.assertEqual(policy.select_config(shape, tuned_file=path).split_k, 0)
        short_path = self._csv([_row(92320, shape=(16, 256, 128), split=0)], name="short.csv")
        self.assertEqual(policy.select_config((16, 256, 128), tuned_file=short_path).split_k, 0)
        self.assertEqual(policy.select_config((16, 256, 128), config=config, split_k=0).split_k, 0)
        # A CSV with only public config metadata needs no internal kernelId.
        row = _row(92320, shape=shape, split=3, pipeline=config.pipeline, config=config.to_json())
        del row["kernelId"]
        path = self._csv([row])
        self.assertEqual(policy.select_config(shape, tuned_file=path).config, config)
        row["kernelId"] = -1
        path = self._csv([row], name="parameter_only.csv")
        self.assertEqual(policy.select_config(shape, tuned_file=path).config, config)

    def test_explicit_config_bypasses_files_and_validates_shape(self):
        config = catalog.config_from_legacy_kid(92320)
        shape = (16, 256, 7168)
        missing = Path(self.temp.name) / "missing.csv"
        selected = policy.select_config(shape, config=config, split_k=3, tuned_file=missing)
        self.assertEqual((selected.config, selected.split_k, selected.source), (config, 3, "explicit"))
        selected = policy.select_config(shape, pipeline="register", config=config.to_json(), split_k=3)
        self.assertEqual(selected.config, config)
        selected = policy.select_config(shape, pipeline="register", compile_params=dict(config.compile_params),
                                        split_k=3)
        self.assertEqual(selected.config, config)
        with self.assertRaises(FrozenInstanceError):
            selected.split_k = 8
        with self.assertRaises(ValueError):
            policy.select_config((16, 250, 7168), config=config)
        with self.assertRaises(ValueError):
            policy.select_config(shape, pipeline="lds", config=config)
        with self.assertRaises(ValueError):
            policy.select_config(shape, config=config.to_json())
        with self.assertRaises(ValueError):
            policy.select_config(shape, pipeline="register", config=config, compile_params={"tile_m": 16})
        bounded = catalog.construct_config(
            "register", tile_m=16, tile_n=32, prefetch=5,
            runtime_split_k=True, max_tensor_bytes=1024,
        )
        self.assertFalse(catalog.config_supports_shape(bounded, *shape))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            policy.select_config(shape, config=bounded, split_k=3)

    def test_fallback_uses_legal_configs_with_actual_splits_and_grid_cu(self):
        shapes = [(1, 128, 128), (17, 256, 7168), (64, 256, 7168),
                  (256, 256, 7168), (512, 1024, 16384), (2048, 128, 1536),
                  (262144, 8192, 1536)]
        for shape in shapes:
            selected = self._without_tuned(shape)
            with self.subTest(shape=shape):
                self.assertEqual(selected.source, "heuristic")
                self._assert_legal(selected, shape)
        self.assertEqual(self._without_tuned((1, 128, 128)).config.pipeline, "register")
        self.assertEqual(self._without_tuned((64, 256, 7168)).config.pipeline, "lds")
        self.assertEqual(self._without_tuned((512, 1024, 7168)).config.pipeline, "pin")
        self.assertEqual(self._without_tuned((262144, 8192, 1536)).config.pipeline, "large_output")
        config = catalog.config_from_legacy_kid(92320)
        low = policy.select_config((16, 256, 7168), config=config, cu_num=8)
        high = policy.select_config((16, 256, 7168), config=config, cu_num=256)
        self.assertEqual((low.split_k, high.split_k), (1, 16))
        short = policy.select_config((16, 256, 128), config=config)
        self.assertEqual(short.split_k, 1)
        fixed = catalog.config_from_legacy_kid(9060)
        self.assertEqual(policy.select_config((64, 256, 7168), config=fixed).split_k, 0)
        with self.assertRaises(ValueError):
            policy.select_config((64, 256, 7168), config=fixed, split_k=1)
        with self.assertRaises(ValueError):
            policy.select_config((16, 256, 128), config=config, split_k=2)

    def test_runtime_dynamic_lds_uses_legal_split_and_rejects_unsupported_k(self):
        config = catalog.config_from_legacy_kid(92430)
        shape = (96, 128, 16384)
        selected = policy.select_config(shape, config=config, cu_num=1)
        self.assertEqual(selected.split_k, 1)
        self._assert_legal(selected, shape)
        self.assertLessEqual(common.bpreshuffle_launch_plan(policy._instance(config), *shape,
                                                           selected.split_k).lds_bytes, 160 * 1024)
        with self.assertRaisesRegex(ValueError, "unsupported|no legal"):
            policy.select_config((96, 128, 131072), config=config)

    def test_new_unregistered_tuple_supports_explicit_and_parameter_csv(self):
        shape = (16, 256, 7168)
        axes = {"tile_m": 16, "tile_n": 32, "runtime_split_k": True, "prefetch": 5}
        config = catalog.construct_config("register", **axes)
        self.assertEqual(config.legacy_kid, -1)
        explicit = policy.select_config(shape, config=config, split_k=3)
        named = policy.select_config(shape, pipeline="register", compile_params=axes, split_k=3)
        self.assertEqual((explicit.config, named.config), (config, config))
        self.assertEqual((explicit.split_k, named.split_k), (3, 3))
        self._assert_legal(explicit, shape)
        row = _row(-1, shape=shape, split=3, pipeline="register", config=config.to_json())
        for metadata in (row, {key: value for key, value in row.items() if key != "kernelId"}):
            path = self._csv([metadata], name=f"tuple{len(metadata)}.csv")
            with self.subTest(columns=list(metadata)):
                selected = policy.select_config(shape, tuned_file=path)
                self.assertEqual((selected.config, selected.split_k, selected.source),
                                 (config, 3, "tuned"))
                self._assert_legal(selected, shape)

    def test_new_runtime_lds_plan_respects_stage_scale_and_xor_axes(self):
        shape = (64, 128, 7168)
        axes_list = [
            {"tile_m": 64, "tile_n": 128, "runtime_split_k": True, "stages": 3},
            {"tile_m": 64, "tile_n": 128, "runtime_split_k": True, "stages": 5},
            {"tile_m": 64, "tile_n": 128, "runtime_split_k": True,
             "register_scales": True, "xor_lds": True},
        ]
        for axes in axes_list:
            with self.subTest(axes=axes):
                config = catalog.construct_config("lds", **axes)
                selected = policy.select_config(shape, config=config, split_k=3)
                plan = common.bpreshuffle_launch_plan(policy._instance(config), *shape, 3)
                loops = (shape[2] // 128 + 2) // 3
                stage_bytes = ((config.tile_m + config.tile_n) // 8
                               * (1024 + (0 if config.xor_lds else 32)))
                scale_bytes = 0 if config.register_scales else (config.tile_m + 1) * loops
                self.assertEqual(plan.lds_bytes, min(loops, config.stages) * stage_bytes + scale_bytes)
                self._assert_legal(selected, shape)

    def test_explicit_files_fail_clearly_and_default_stale_rows_fallback(self):
        malformed = [
            _row(kernelId="9041.0"), _row(kernelId="999999"), _row(splitK="-1"),
            _row(pipeline="register", config="{}"), _row(pipeline="register", config=""),
            _row(9000), _row(-1, pipeline="register", config='{"tile_m":16,"tile_n":32}'),
        ]
        for index, row in enumerate(malformed):
            with self.subTest(row=row):
                path = self._csv([row], name=f"bad{index}.csv")
                with self.assertRaisesRegex(policy.TunedConfigError, r"bad\d+\.csv:2"):
                    policy.select_config((1, 768, 7168), tuned_file=path)
                with patch.object(policy, "DEFAULT_TUNED_CONFIG", path):
                    self.assertEqual(policy.select_config((1, 768, 7168)).source, "heuristic")
        bad_schema = self._csv([{"M": 1}], name="schema.csv")
        with self.assertRaisesRegex(policy.TunedConfigError, "missing columns"):
            policy.select_config((1, 768, 7168), tuned_file=bad_schema)
        with patch.object(policy, "DEFAULT_TUNED_CONFIG", bad_schema):
            self.assertEqual(policy.select_config((1, 768, 7168)).source, "heuristic")
        with self.assertRaisesRegex(policy.TunedConfigError, "cannot read"):
            policy.select_config((1, 768, 7168), tuned_file=Path(self.temp.name) / "missing.csv")

    def test_environment_override_and_file_replacements_are_observed(self):
        path = self._csv([_row(9040)])
        with patch.dict("os.environ", {policy.TUNED_CONFIG_ENV: str(path)}):
            self.assertEqual(policy.select_config((1, 768, 7168)).config.legacy_kid, 9040)
            replacement = self._csv([_row(9041)], name="replacement.csv")
            replacement.replace(path)
            self.assertEqual(policy.select_config((1, 768, 7168)).config.legacy_kid, 9041)
            with patch.object(policy, "DEFAULT_TUNED_CONFIG", Path(self.temp.name) / "missing.csv"):
                path.unlink()
                with self.assertRaises(policy.TunedConfigError):
                    policy.select_config((1, 768, 7168))

    def test_duplicate_rows_prefer_fastest_valid_timing_and_skip_default_stale(self):
        path = self._csv([_row(9040, us=10), _row(9041, us=5), _row(999999, us=1)])
        with patch.object(policy, "DEFAULT_TUNED_CONFIG", path):
            self.assertEqual(policy.select_config((1, 768, 7168)).config.legacy_kid, 9041)
        with self.assertRaises(policy.TunedConfigError):
            policy.select_config((1, 768, 7168), tuned_file=path)
        path = self._csv([_row(9040, us=10), _row(9041, us="nan")])
        self.assertEqual(policy.select_config((1, 768, 7168), tuned_file=path).config.legacy_kid, 9040)

    def test_scalar_validation_and_no_legal_shape(self):
        for shape in [None, 1, (), (1, 2), (1, 2, 3, 4), (0, 128, 128),
                      (True, 128, 128), (1.0, 128, 128)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                policy.select_config(shape)
        for cu_num in [0, -1, True, 256.0]:
            with self.subTest(cu_num=cu_num), self.assertRaises(ValueError):
                policy.select_config((16, 256, 7168), cu_num=cu_num)
        for split in [True, 1.0, -2, "3"]:
            with self.subTest(split=split), self.assertRaises(ValueError):
                policy.select_config((16, 256, 7168), split_k=split)
        for pipeline in ["bogus", [], "", ["register", "bogus"]]:
            with self.subTest(pipeline=pipeline), self.assertRaises(ValueError):
                self._without_tuned((16, 256, 7168), pipeline=pipeline)
        with self.assertRaisesRegex(ValueError, "no legal"):
            self._without_tuned((16, 250, 7168))


if __name__ == "__main__":
    unittest.main()
