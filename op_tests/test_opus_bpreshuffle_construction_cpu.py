# SPDX-License-Identifier: MIT
"""CPU checks that scalar configurations really construct producer traits."""
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
OPUS = ROOT / 'csrc/opus_gemm'
if str(OPUS) not in sys.path:
    sys.path.insert(0, str(OPUS))
import opus_gemm_bpreshuffle_config as catalog
from opus_gemm_common import bpreshuffle_launch_plan
from gen_instances import opus_gemm_codegen


class ConfigConstructionCPU(unittest.TestCase):
    @staticmethod
    def examples():
        return (
            catalog.construct_config('pin', tile_m=256, tile_n=256, scale_panel=32),
            catalog.construct_config('tiled', tile_m=96, tile_n=128, stages=3),
            catalog.construct_config('register', tile_m=16, tile_n=32, runtime_split_k=True, prefetch=5),
            catalog.construct_config('lds', tile_m=64, tile_n=128, runtime_split_k=True, stages=5),
            catalog.construct_config('large_output', tile_m=192, tile_n=256, b_direct=True, c_chunk_rows=64),
        )

    def test_catalog_is_independent_of_registry_import(self):
        source = (OPUS / 'opus_gemm_bpreshuffle_config.py').read_text()
        prefix = source[:source.index('def kernel_instance_from_config')]
        self.assertNotIn('import opus_gemm_common', prefix)
        self.assertEqual(len(catalog.CONFIGS_BY_LEGACY_KID), 105)
        self.assertEqual(sum(len(p.active_configs) for p in catalog.BPRESHUFFLE_PIPELINES.values()), 89)

    def test_new_tuples_roundtrip_and_emit_real_traits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for config in self.examples():
                with self.subTest(pipeline=config.pipeline):
                    self.assertEqual(config.legacy_kid, -1)
                    restored = catalog.resolve_config(config.pipeline, config.to_json())
                    self.assertEqual(restored, config)
                    instance = catalog.kernel_instance_from_config(restored)
                    self.assertIs(instance.bpreshuffle_config, restored)
                    self.assertTrue(instance.bpreshuffle_config_codegen)
                    output = Path(tmp) / config.pipeline
                    output.mkdir()
                    opus_gemm_codegen(str(output)).gen_instances({1: instance})
                    impl = (output / 'impl' / (instance.name + '.cuh')).read_text()
                    self.assertIn(catalog.config_traits(config), impl)
                    self.assertIn('opus_gemm_mxscale_bpreshuffle_' + config.pipeline + '_kernel', impl)
                    self.assertEqual(len(list((output / 'instances').glob('*.device.cu'))), 1)

    def test_invalid_axes_are_rejected_without_rewriting_explicit_values(self):
        invalid = (
            ('register', dict(tile_m=16, tile_n=32, prefetch=5, runtime_split_k=True, split_k=1)),
            ('register', dict(tile_m=16, tile_n=32, prefetch=5, runtime_split_k=True, schedule=0)),
            ('register', dict(tile_m=16, tile_n=32, prefetch=5, stages=4)),
            ('pin', dict(tile_m=256, tile_n=256, scale_panel=32, stages=2)),
            ('pin', dict(tile_m=256, tile_n=256, scale_panel=32, specializations=((384,2,1,4,128,0),))),
            ('tiled', dict(tile_m=96, tile_n=128, stages=3, reduce_vec=16)),
            ('lds', dict(tile_m=64, tile_n=128, stages=12, runtime_split_k=True)),
            ('lds', dict(tile_m=64, tile_n=128, stages=5, runtime_split_k=True, specializations=((384,4,1,16,128,2),))),
            ('large_output', dict(tile_m=192, tile_n=256, b_direct=True, c_chunk_rows=80)),
        )
        for pipeline, axes in invalid:
            with self.subTest(pipeline=pipeline, axes=axes), self.assertRaises(ValueError):
                catalog.construct_config(pipeline, **axes)

    def test_mutating_a_legacy_config_drops_compatibility_id(self):
        seed = catalog.config_from_legacy_kid(9040)
        changed = catalog.validate_config(replace(seed, prefetch=5))
        self.assertEqual(changed.legacy_kid, -1)
        self.assertIn('config_', catalog.kernel_instance_from_config(changed).name)
        with self.assertRaises(ValueError):
            catalog.validate_saved_config(changed.pipeline, changed.to_json(), seed.legacy_kid)

    def test_runtime_lds_planner_matches_new_ring_extent(self):
        config = self.examples()[3]
        instance = catalog.kernel_instance_from_config(config)
        plan = bpreshuffle_launch_plan(instance, 64, 256, 7168, 1)
        self.assertEqual(plan.lds_bytes, 5 * (64 + 128) // 8 * 1056 + 65 * 56)
        self.assertEqual(plan.workspace_bytes, 0)
        self.assertEqual(plan.grid, (2, 1, 1))
        expected = catalog.config_identity(config)
        self.assertEqual(catalog.config_identity(catalog.resolve_config(config.pipeline, config.to_json())), expected)


if __name__ == '__main__':
    unittest.main()
