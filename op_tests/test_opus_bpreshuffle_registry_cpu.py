"""CPU-only registration checks against the retained experiment contracts.

Run directly with ``python3 op_tests/test_opus_bpreshuffle_registry_cpu.py``.
Only scalar registry code is imported. Launchers are generated as text; no HIP
compiler, runtime, device query, shared library, or tensor package is needed.
"""

import ast
from contextlib import contextmanager
import csv
from dataclasses import fields
import fnmatch
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OPUS = ROOT / "csrc/opus_gemm"
REPORT = ROOT / "reports/opus_register92_20261009"
EXPERIMENTS = ROOT / "reports/opus_flydsl_all_20261009"
FAMILY = "a8w8_blockscale_bpreshuffle"
INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1
RUNTIME_IDS = frozenset({92310, 92311, 92320, 92321, 92330, 92340, 92410, 92420, 92430})
RUNTIME_ALIASES = {92411: 92410, 92421: 92420, 92431: 92430}
HYBRID_ACTUALS = (9043, 9044, 9045, 9046, 9047, 9049, 9055, 9056)
SMALL_IDS = {
    110: 92310, 111: 92311, 120: 92320, 121: 92321,
    130: 92330, 140: 92340,
    210: 92410, 211: 92411, 220: 92420, 221: 92421,
    230: 92430, 231: 92431,
}


class _RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "hip", "hsa"}:
            raise AssertionError(f"CPU registry check attempted GPU import: {fullname}")
        return None


@contextmanager
def _cpu_imports():
    guard = _RejectGPUImports()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through this key.
    spec.loader.exec_module(module)
    return module


def _extract(path, names, namespace):
    """Execute only named top-level definitions, with explicit scalar globals."""
    tree = ast.parse(path.read_text())
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & names:
                selected.append(node)
                found |= targets & names
    if found != names:
        raise AssertionError(f"Missing AST definitions in {path}: {names - found}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _promoted_id(key):
    package, token = key.split(":", 1)
    if package == "main_variants":
        return int(token)
    if package == "small_split":
        return SMALL_IDS[int(token)]
    if package == "hybrid_small":
        actual, ahead = re.fullmatch(r"(\d+)@direct_b_ahead([123])", token).groups()
        return 92200 + 3 * HYBRID_ACTUALS.index(int(actual)) + int(ahead) - 1
    if package == "large9030":
        return {"panel16": 92500, "directb": 92501}[token]
    raise AssertionError(key)


def _frozen_configs(before):
    """Metadata oracle from the frozen catalog and experiment traits/contracts."""
    result = {}

    def add(kid, family, bm, bn, wm, wn, **overrides):
        result[kid] = dict(
            family=family, tile_m=bm, tile_n=bn, wave_m=wm, wave_n=wn,
            wave_k=1, m_align=16, max_m=None, fixed_k=0, split_k=1,
            dynamic_lds=False, pin_agpr=False, sfa_alignment=16, c_alignment=16,
            reduce_vec=16, reduce_block=128,
        )
        result[kid].update(overrides)

    catalog = json.loads((EXPERIMENTS / "candidate_catalog.json").read_text())
    for row in catalog:
        kid = _promoted_id(row["candidate_key"])
        detail = row["details"]
        package = row["package"]
        if package == "main_variants":
            family = ("tile_order" if kid in (92011, 92012) else
                      "pad_pin_fixed" if kid == 92120 else
                      "pin_fixed" if detail["pin"] else "geometry")
            add(kid, family, detail["tile_M"], detail["tile_N"], 2, 2,
                fixed_k=detail["fixed_K"], pin_agpr=detail["pin"],
                m_align=256 if family == "pin_fixed" else 16)
        elif package == "hybrid_small":
            actual = detail["actual_kid"]
            old = before.a8w8_mxscale_gemm_bpreshuffle_kernels_list[actual]
            add(kid, "small_direct_b", old.B_M, old.B_N, old.T_M, old.T_N,
                m_align=1, max_m=512 if actual in (9047, 9049) else 2048,
                dynamic_lds=True, sfa_alignment=1, c_alignment=8)
        elif package == "small_split":
            variant = detail["variant"]
            if detail["pool"] == "register":
                bm = 32 if variant in (130, 140) else 16
                bn = {110: 16, 111: 16, 120: 32, 121: 32, 130: 32, 140: 64}[variant]
                add(kid, "register_split", bm, bn, 1, 1, m_align=1,
                    wave_k=2 if variant in (111, 121) else 1,
                    max_m=512, split_k=4, sfa_alignment=1)
            else:
                bm, bn, wm, wn = ((48, 64, 1, 4) if variant < 220 else
                                  (64 if variant < 230 else 96, 128, 2, 2))
                add(kid, "fine_lds", bm, bn, wm, wn, m_align=1, max_m=2048,
                    split_k=1 if variant % 10 == 0 else 2, dynamic_lds=True)
        else:
            add(kid, "large_output", 192, 256, 4, 2, m_align=64, fixed_k=1536)
    # Earlier BM160 short-K experiment is independent of BM96 geometry above.
    add(92020, "shortk", 160, 128, 2, 2, fixed_k=384)
    add(92021, "shortk", 160, 128, 2, 2, fixed_k=768)
    return result


def _contract_accepts(config, m, n, k):
    """Scalar transcription of immutable contract.h guards, using oracle data."""
    if m <= 0 or n <= 0 or not 128 <= k <= 16384 or k % 128 or n % 128:
        return False
    if m % config["m_align"] or n % config["tile_n"]:
        return False
    if config["max_m"] is not None and m > config["max_m"]:
        return False
    if config["fixed_k"] and k != config["fixed_k"]:
        return False
    if config["family"] == "large_output":
        # private9030_shape_valid divides before multiplying C's byte extent.
        return (m * k <= INT32_MAX and n * k <= INT32_MAX
                and 191 * n + 256 <= INT32_MAX // 2
                and m > (INT32_MAX // 2) // n
                and m <= (INT64_MAX // 2) // n)
    bytes_per_output = 4 if config["split_k"] > 1 else 2
    return (m * k <= INT32_MAX and n * k <= INT32_MAX
            and m * n * bytes_per_output <= INT32_MAX)


class BpreshuffleRegistryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules_before = set(sys.modules)
        with _cpu_imports():
            variants = _load_file("_opus_registry_cpu_variants", OPUS / "opus_gemm_bpreshuffle_variants.py")
            with patch.dict(sys.modules, {"opus_gemm_bpreshuffle_variants": variants}):
                cls.common = _load_file("_opus_registry_cpu_common", OPUS / "opus_gemm_common.py")
            cls.before = _load_file("_opus_registry_cpu_before", REPORT / "before/csrc/opus_gemm/opus_gemm_common.py")
        cls.variants = variants.NEW_BPRESHUFFLE_VARIANTS_BY_KID
        cls.registry = cls.common.a8w8_mxscale_gemm_bpreshuffle_kernels_list
        cls.expected = _frozen_configs(cls.before)
        cls.promoted = json.loads((OPUS / "include/gfx950/opus_gemm_mxscale_bpreshuffle_promoted_variants.json").read_text())["variants"]
        cls.cases = _csv(EXPERIMENTS / "candidate_cases334.csv")
        cls.coverage = _csv(EXPERIMENTS / "coverage334.csv")
        helpers = _extract(OPUS / "gen_instances.py", {
            "_INSTANCE_IMPL_PREAMBLE_TEMPLATE", "instance_impl_preamble",
            "_make_a8w8_bpreshuffle_host_decl",
        }, {})
        emitter = _extract(OPUS / "codegen/gen_instances_gfx950.py", {
            "_bpreshuffle_compact_traits", "gen_mxscale_bpreshuffle_instance",
        }, {
            "os": os, "Path": Path, "OpusGemmInstance": cls.common.OpusGemmInstance,
            "a8w8_mxscale_gemm_bpreshuffle_kernels_list": cls.registry,
        })
        cls.generated = {}
        cls.device_decls = {}
        cls.host_decls = {}
        with tempfile.TemporaryDirectory() as directory:
            cg = SimpleNamespace(impl_path=directory, _kid_pipeline_header={},
                                 _host_instantiations=[], _device_instantiations=[])
            for kid in sorted(cls.variants):
                instance = cls.registry[kid]
                emitter["gen_mxscale_bpreshuffle_instance"](
                    cg, instance, pipeline_header="unused", traits_header="unused",
                    kernel_func="unused", traits_name="unused",
                    kargs_name="opus_gemm_mxscale_bpreshuffle_kargs",
                    instance_impl_preamble=helpers["instance_impl_preamble"],
                    make_a8w8_bpreshuffle_host_decl=helpers["_make_a8w8_bpreshuffle_host_decl"],
                )
                cls.generated[kid] = (Path(directory) / f"{instance.name}.cuh").read_text()
                cls.device_decls[kid] = cg._device_instantiations[-1]["device_decl"]
                cls.host_decls[kid] = cg._host_instantiations[-1]["host_decl"]

    def test_counts_ids_names_and_legacy_preservation(self):
        old = self.before.a8w8_mxscale_gemm_bpreshuffle_kernels_list
        self.assertEqual(len(old), 41)
        self.assertEqual(set(self.variants), set(self.expected))
        self.assertEqual(len(self.variants), 64)
        self.assertFalse(set(old) & set(self.variants))
        self.assertEqual(set(self.registry), set(old) | set(self.expected))
        self.assertEqual(len(self.registry), 105)
        self.assertEqual(len(self.common.A8W8_BPRESHUFFLE_TUNING_KIDS), 89)
        self.assertEqual(self.common.A8W8_BPRESHUFFLE_LEGACY_KIDS,
                         {**self.before.A8W8_BPRESHUFFLE_LEGACY_KIDS, **RUNTIME_ALIASES})
        self.assertEqual(len({instance.name for instance in self.registry.values()}), 105)
        old_fields = [field.name for field in fields(self.before.OpusGemmInstance)]
        for kid, frozen in old.items():
            current = self.registry[kid]
            with self.subTest(kid=kid):
                self.assertEqual({name: getattr(current, name) for name in old_fields},
                                 {name: getattr(frozen, name) for name in old_fields})
                self.assertEqual(current.name, frozen.name)
                self.assertEqual(current.m_align, frozen.m_align)
                self.assertIsNone(current.bpreshuffle_variant)

    def test_new_metadata_matches_frozen_experiments_and_promoted_headers(self):
        self.assertEqual({row["id"] for row in self.promoted}, set(self.expected))
        for kid, expected in self.expected.items():
            descriptor = self.variants[kid]
            instance = self.registry[kid]
            with self.subTest(kid=kid):
                self.assertEqual({name: getattr(descriptor, name) for name in expected}, expected)
                self.assertIs(instance.bpreshuffle_variant, descriptor)
                self.assertEqual((instance.B_M, instance.B_N, instance.T_M, instance.T_N),
                                 (expected["tile_m"], expected["tile_n"], expected["wave_m"], expected["wave_n"]))
                self.assertEqual(instance.BLOCK_SIZE, 64 * expected["wave_m"] * expected["wave_n"] * expected["wave_k"])
                self.assertEqual(instance.output_dtypes, ["bf16_t"])
                self.assertEqual(instance.scale_dtype, "e8m0")
                self.assertEqual(instance.splitk_workspace_dtype, "fp32_t" if expected["split_k"] > 1 else None)
                self.assertEqual(instance.max_tensor_bytes, INT64_MAX if expected["family"] == "large_output" else INT32_MAX)
        for row in self.promoted:
            descriptor = self.variants[row["id"]]
            with self.subTest(kid=row["id"]):
                if row["id"] in RUNTIME_IDS:
                    self.assertTrue(descriptor.runtime_split_k)
                    self.assertTrue((OPUS / "include" / descriptor.pipeline_header).is_file())
                    self.assertTrue((OPUS / "include" / descriptor.traits_header).is_file())
                else:
                    for name in ("pipeline_header", "traits_header", "kernel", "traits"):
                        self.assertEqual(getattr(descriptor, name), row[name])
                if row["id"] in (92011, 92012):
                    self.assertEqual(row["report_source"], "existing production pipeline")
                else:
                    self.assertTrue((ROOT / row["report_source"]).is_file())
                self.assertTrue((OPUS / "include" / row["pipeline_header"]).is_file())
                self.assertTrue((OPUS / "include" / row["traits_header"]).is_file())
        # BM160 stays distinct from the similarly named BM96 geometry entries.
        self.assertEqual(self.variants[92020].fixed_k, self.variants[92003].fixed_k)
        self.assertEqual(self.variants[92020].kernel, self.variants[92003].kernel)
        self.assertNotEqual(self.variants[92020].traits, self.variants[92003].traits)
        self.assertEqual((self.variants[92020].tile_m, self.variants[92003].tile_m), (160, 96))

    def test_shape_boundaries_match_frozen_contracts(self):
        shapes = {
            (m, n, k)
            for m in (0, 1, 15, 16, 63, 64, 96, 160, 255, 256, 512, 513, 2048, 2049)
            for n in (0, 127, 128, 256)
            for k in (0, 127, 128, 384, 768, 1536, 3072, 7168, 16384, 16512)
        }
        shapes |= {
            (32768, 32768, 1536), (32768, 33024, 1536),
            (65536, 65536, 1536), (64, 5_620_992, 1536),
            (1_398_144, 256, 1536), (1_398_208, 256, 1536),
            (256, 1_398_144, 1536), (256, 1_398_272, 1536),
            (512, 1_048_576, 128), (512, 1_048_448, 128),
            (2048, 262_144, 128), (2048, 262_016, 128),
            (16, 16_777_216, 128), (16, 16_777_088, 128),
            (2**62, 256, 1536), (64, 2**62, 1536),
        }
        for kid, config in self.expected.items():
            for shape in sorted(shapes):
                with self.subTest(kid=kid, shape=shape):
                    contract = dict(config)
                    if kid in RUNTIME_IDS:
                        # Runtime registrations admit the direct BF16 shape;
                        # each requested FP32 workspace plane has its own check.
                        contract["split_k"] = 1
                    self.assertEqual(self.common.a8w8_mxscale_bpreshuffle_supports_shape(
                        self.registry[kid], *shape), _contract_accepts(contract, *shape))

    def test_traits_parameters_match_the_frozen_catalog(self):
        catalog = json.loads((EXPERIMENTS / "candidate_catalog.json").read_text())
        prefix = "opus_gemm_mxscale_bpreshuffle_"
        for row in catalog:
            kid = _promoted_id(row["candidate_key"])
            detail = row["details"]
            config = self.expected[kid]
            if row["package"] == "main_variants":
                traits = detail["traits"]
                for old, new in (
                    ("opus_private_geometry_traits", prefix + "geometry_traits"),
                    ("opus_private_pin_fixed_traits", prefix + "pin_fixed_traits"),
                    ("opus_private_pad_pin_fixed_traits", prefix + "pad_pin_fixed_traits"),
                ):
                    traits = traits.replace(old, new)
            elif row["package"] == "hybrid_small":
                traits = prefix + f"small_direct_b_traits<{detail['actual_kid']},{detail['b_ahead']}>"
            elif config["family"] == "register_split":
                traits = prefix + f"register_split_traits<{config['tile_m']},{config['tile_n']},{config['wave_k']}>"
            elif config["family"] == "fine_lds":
                traits = prefix + f"narrow_fine_traits<{config['tile_m']},{config['tile_n']},{config['wave_m']},{config['wave_n']},{config['split_k']}>"
            else:
                traits = prefix + "large_output_" + ("panel16" if detail["library"] == "panel16" else "direct_b") + "_traits"
            with self.subTest(kid=kid):
                if kid not in RUNTIME_IDS:
                    self.assertEqual(self.variants[kid].traits, traits)
        for kid, fixed in ((92020, 384), (92021, 768)):
            self.assertEqual(self.variants[kid].traits, prefix + f"shortk_traits<{fixed}>")
        short_traits = (ROOT / "reports/opus_flydsl_gap_20261009/shortk9022/candidate/traits.cuh").read_text()
        self.assertIn("FIXED_K == 384 || FIXED_K == 768", short_traits)
        self.assertIn("SCALE_PANEL = 8", short_traits)

    def test_all_1452_historical_cases_and_334_loser_shapes_remain_callable(self):
        self.assertEqual(len(self.cases), 1452)
        self.assertEqual(len(self.coverage), 334)
        actual_keys = set()
        for row in self.cases:
            shape = tuple(int(row[name]) for name in ("M", "N", "K"))
            kid = _promoted_id(row["candidate_key"])
            actual_keys.add((shape, kid))
            with self.subTest(kid=kid, shape=shape):
                self.assertTrue(_contract_accepts(self.expected[kid], *shape))
                self.assertIn(kid, self.common.a8w8_mxscale_bpreshuffle_candidate_kids("gfx950", *shape, include_legacy=True))
                if kid in RUNTIME_IDS:
                    plan = self.common.bpreshuffle_launch_plan(self.registry[kid], *shape, split_k=0)
                    self.assertEqual(plan.split_k, self.expected[kid]["split_k"])
                detail = json.loads(row["details_json"])
                if row["package"] == "small_split":
                    if "descriptor" in detail:
                        calls = detail["descriptor"]["kernel_calls"]
                    else:
                        calls = detail["kernel_calls"]
                        self.assertEqual(int(detail["global_splitK"]), self.variants[kid].split_k)
                        self.assertEqual(int(detail["workspace_bytes"]),
                                         4 * self.variants[kid].split_k * shape[0] * shape[1] if self.variants[kid].split_k > 1 else 0)
                    self.assertEqual(int(calls), 2 if self.variants[kid].split_k > 1 else 1)
        self.assertEqual(len(actual_keys), 1452)
        covered_keys = {
            (tuple(int(row[name]) for name in ("M", "N", "K")), _promoted_id(key))
            for row in self.coverage for key in row["candidate_keys"].split(";")
        }
        self.assertEqual(actual_keys, covered_keys)
        self.assertEqual(len({shape for shape, kid in actual_keys}), 334)

    def test_filtering_arch_dtype_legacy_and_pipeline_family(self):
        choose = self.common.a8w8_mxscale_bpreshuffle_candidate_kids
        shape = (256, 256, 7168)
        defaults = set(choose("gfx950", *shape))
        all_kids = set(choose("gfx950", *shape, include_legacy=True))
        self.assertFalse(defaults & self.common.A8W8_BPRESHUFFLE_LEGACY_KIDS.keys())
        self.assertEqual(all_kids - defaults, set(self.common.A8W8_BPRESHUFFLE_LEGACY_KIDS))
        for dtype in ("fp32", "float32", "fp16", "unknown"):
            self.assertEqual(choose("gfx950", *shape, outdtype=dtype), [])
        self.assertEqual(choose("gfx942", *shape), [])
        self.assertEqual(choose("gfx1250", *shape), [])
        self.assertEqual(choose("gfx950", 0, 256, 7168), [])
        self.assertEqual(choose("gfx950", *shape, families=[]), [])
        with self.assertRaises(ValueError):
            choose("gfx950", *shape, families=["unknown"])
        self.assertEqual(set(self.common.A8W8_BPRESHUFFLE_FAMILY_BY_KID), set(self.registry))
        for family in self.common.A8W8_BPRESHUFFLE_FAMILIES:
            result = set(choose("gfx950", *shape, families=[family]))
            self.assertEqual(result, {kid for kid in defaults if self.common.A8W8_BPRESHUFFLE_FAMILY_BY_KID[kid] == family})
        wrapper = _extract(OPUS / "opus_gemm_mxscale_bpreshuffle_tune.py", {"candidate_kids_for_shape"},
                           {"a8w8_mxscale_bpreshuffle_candidate_kids": choose})["candidate_kids_for_shape"]
        self.assertEqual(wrapper("gfx950", *shape, families=["geometry"]),
                         choose("gfx950", *shape, families=["geometry"]))

    def test_exact_registry_lookup_and_external_workspace_capability(self):
        for kid, instance in self.registry.items():
            with self.subTest(kid=kid):
                self.assertIs(self.common.get_kernel_instance("gfx950", FAMILY, kid, "bf16"), instance)
                self.assertIsNone(self.common.get_kernel_instance("gfx942", FAMILY, kid, "bf16"))
                self.assertIsNone(self.common.get_kernel_instance("gfx950", FAMILY, kid, "fp32"))
                self.assertEqual(self.common.kernel_needs_external_workspace("gfx950", FAMILY, kid),
                                 kid in RUNTIME_IDS or instance.bpreshuffle_split_k > 1)
        with self.assertRaises(KeyError):
            self.common.kernel_needs_external_workspace("gfx950", FAMILY, -1)

    def test_generated_launch_contract_and_alignment_for_every_new_id(self):
        for kid, config in self.expected.items():
            if kid in RUNTIME_IDS:
                continue  # Runtime launch ABI is checked by its dedicated suite.
            source = self.generated[kid]
            instance = self.registry[kid]
            descriptor = self.variants[kid]
            with self.subTest(kid=kid):
                self.assertIn(f'using {instance.name}_Traits = '
                              f'opus_gemm_mxscale_bpreshuffle_pipeline_traits<{descriptor.traits}, {descriptor.schedule}>;', source)
                self.assertIn(f'#include "{descriptor.pipeline_header}"', source)
                self.assertEqual(source.count(f'#include "{descriptor.traits_header}"'), 2)
                self.assertIn("static_assert(std::is_same_v<D_C, bf16_t>);", source)
                self.assertIn("x_scale.stride(0) == 1 && x_scale.stride(1) == m", source)
                self.assertIn("w_scale must be row-major [N/128,K/128]", source)
                self.assertIn("k <= 16384", source)
                self.assertIn(f"n % {max(config['tile_n'], 128)} == 0", source)
                if config["m_align"] > 1:
                    self.assertIn(f"m % {config['m_align']} == 0", source)
                if config["fixed_k"]:
                    self.assertIn(f"k == {config['fixed_k']}", source)
                if config["max_m"] is not None:
                    self.assertIn(f"m <= {config['max_m']}", source)
                self.assertIn(f"output % {config['c_alignment']} == 0", source)
                if config["sfa_alignment"] > 1:
                    self.assertIn("reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0", source)
                else:
                    self.assertNotIn("reinterpret_cast<uintptr_t>(x_scale.data_ptr()) %", source)
                expected_lds = f"{instance.name}_Traits::lds_bytes(k)" if config["dynamic_lds"] else "0"
                self.assertIn(f"grid, dim3({instance.BLOCK_SIZE}), {expected_lds}, aiter::getCurrentHIPStream()", source)
                self.assertIn("Y must not overlap input storage", source)
                self.assertIn("std::optional<aiter_tensor_t> workspace", self.host_decls[kid])
                self.assertIn(descriptor.kernel, self.device_decls[kid])

    def test_split_k_emits_complete_call_with_fp32_workspace_and_bf16_reducer(self):
        for kid, config in self.expected.items():
            if kid in RUNTIME_IDS:
                continue  # Runtime producer/reducer coverage belongs to its suite.
            source = self.generated[kid]
            with self.subTest(kid=kid):
                if config["split_k"] == 1:
                    self.assertIn("!workspace.has_value()", source)
                    self.assertNotIn("opus_gemm_mxscale_bpreshuffle_reduce_kernel<", self.device_decls[kid])
                    continue
                split = config["split_k"]
                self.assertIn(f"const dim3 grid(n / {config['tile_n']}, tiles_m, {split});", source)
                self.assertIn(f"opus_checked_extent_product({{{split}, size_t(m), size_t(n)}}", source)
                self.assertIn("AITER_DTYPE_fp32, required, 16, entry", source)
                self.assertIn("m <= (byte_limit / sizeof(float)) / n", source)
                self.assertIn("workspace must not overlap input/output storage", source)
                self.assertIn("args.ptr_c = partials;", source)
                reducer = f"opus_gemm_mxscale_bpreshuffle_reduce_kernel<{split}, 16, 128>"
                self.assertIn(reducer, self.device_decls[kid])
                self.assertIn("(const float*, opus::bf16_t*, int)", self.device_decls[kid])
                producer_offset = source.index(f"{self.variants[kid].kernel}<{self.registry[kid].name}_Traits><<<")
                reducer_offset = source.index(reducer + "<<<")
                self.assertLess(producer_offset, reducer_offset)
                complete_call = source[producer_offset:]
                self.assertEqual(complete_call.count("aiter::getCurrentHIPStream()"), 2)
                self.assertIn("reinterpret_cast<const float*>(args.ptr_c)", complete_call)
                self.assertIn("reinterpret_cast<opus::bf16_t*>(Y.data_ptr())", complete_call)
        reducer_source = (OPUS / "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_lds_gfx950.cuh").read_text()
        reducer_source = reducer_source[reducer_source.index("void opus_gemm_mxscale_bpreshuffle_reduce_kernel"):]
        self.assertIn("static_for<SplitK>", reducer_source)
        self.assertIn("cast<bf16_t>", reducer_source)
        self.assertIn("static_cast<int64_t>(decltype(split)::value) * elements", reducer_source)
        register_source = (OPUS / "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_gfx950.cuh").read_text()
        self.assertIn("reinterpret_cast<float*>(args.ptr_c) + static_cast<int64_t>(split)", register_source)
        self.assertIn("__shared__ float partials", register_source)

    def test_large_output_keeps_64bit_c_base_and_signed_tile_span(self):
        for kid in (92500, 92501):
            source = self.generated[kid]
            with self.subTest(kid=kid):
                self.assertIn("output_byte_limit = 9223372036854775807", source)
                self.assertIn("m > (input_byte_limit / sizeof(D_C)) / n", source)
                self.assertIn("191 * n + 256 <= input_byte_limit / sizeof(D_C)", source)
                self.assertIn("args.stride_c_batch = 0;", source)
                self.assertTrue(self.common.a8w8_mxscale_bpreshuffle_supports_shape(self.registry[kid], 65536, 65536, 1536))
                self.assertFalse(self.common.a8w8_mxscale_bpreshuffle_supports_shape(self.registry[kid], 32704, 32768, 1536))

    def test_compiler_override_covers_only_the_15_explicit_pin_entries(self):
        expected_pin = {9000, 9001, 9010, 9011, 92120} | set(range(92100, 92105)) | set(range(92110, 92115))
        self.assertEqual(self.common.A8W8_BPRESHUFFLE_PIN_AGPR_KIDS, expected_pin)
        with _cpu_imports():
            compiler = _load_file("_opus_registry_cpu_compiler", ROOT / "aiter/jit/utils/opus_compiler.py")
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "clang++").touch()
                with patch.dict(os.environ, {"OPUS_BASELINE_HIP_CLANG_PATH": "baseline", "OPUS_HIP_CLANG_PATH": directory}), \
                     patch.dict(sys.modules, {"csrc.opus_gemm.opus_gemm_common": self.common}):
                    commands = compiler.opus_compiler_commands_per_source()
                    self.assertEqual(len(commands), 15)
                    matched = {kid for kid, instance in self.registry.items()
                               if any(fnmatch.fnmatch(instance.name + "_C0.device.cu", pattern) for pattern in commands)}
                    self.assertEqual(matched, expected_pin)
                    for command in commands.values():
                        self.assertEqual(command[command.index("--compiler") + 1], str(Path(directory) / "clang++"))
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(compiler.opus_compiler_commands_per_source(), {})

    def test_imports_remain_cpu_only_and_snapshot_oracle_is_unchanged(self):
        imported = set(sys.modules) - self.modules_before
        self.assertFalse({name for name in imported if name.split(".")[0] in {"torch", "aiter", "hip", "hsa"}})
        before_path = REPORT / "before/csrc/opus_gemm/opus_gemm_common.py"
        self.assertEqual(hashlib.sha256(before_path.read_bytes()).hexdigest(),
                         "7f9e80cfdc6d81b0f03e0e90e2b0c3da8bc09c8d11d4cb1059c3a4c8a3c31e2e")


if __name__ == "__main__":
    unittest.main(verbosity=2)
