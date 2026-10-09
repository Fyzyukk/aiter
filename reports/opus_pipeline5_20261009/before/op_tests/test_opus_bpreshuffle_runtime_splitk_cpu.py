"""CPU contracts for literal runtime split-K and its complete-call launch ABI.

Only pure registry modules are imported. GPU-bearing wrappers and codegen are
read using restricted AST extraction; kernels are inspected as source text.
"""

import ast
import builtins
from contextlib import contextmanager
from dataclasses import fields, FrozenInstanceError
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OPUS = ROOT / "csrc/opus_gemm"
INCLUDE = OPUS / "include/gfx950"
REPORT = ROOT / "reports/opus_runtime_splitk_20261009"
BEFORE = REPORT / "before"
RUNTIME_IDS = frozenset({92310, 92311, 92320, 92321, 92330, 92340, 92410, 92420, 92430})
FINE_IDS = frozenset({92410, 92420, 92430})
ALIAS_IDS = {92411: 92410, 92421: 92420, 92431: 92430}
INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1


class _RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "hip", "hsa"}:
            raise AssertionError(f"CPU test attempted GPU import: {fullname}")
        return None


@contextmanager
def _cpu_imports():
    guard = _RejectGPUImports()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _definitions(path, names, namespace):
    selected = []
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            identifiers = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if identifiers & names:
                selected.append(node)
                found |= identifiers & names
    if found != names:
        raise AssertionError(f"Missing definitions in {path}: {names - found}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _partition_reference(total, parts):
    """Fairly deal tiles, then place the resulting counts contiguously."""
    buckets = [[] for _ in range(parts)]
    for tile in range(total):
        buckets[tile % parts].append(tile)
    begin = 0
    result = []
    for bucket in buckets:
        result.append((begin, len(bucket)))
        begin += len(bucket)
    return result


def _fine_lds_reference(bm, bn, k, split):
    lengths = [count for begin, count in _partition_reference(k // 128, split)]
    loops = max(lengths)
    # Four stages, 32-byte LDS row padding, byte scales indexed per K128 tile.
    stage_bytes = ((bm + bn) // 8) * (1024 + 32)
    scale_bytes = (bm + math.ceil(bn / 128)) * loops
    return min(loops, 4) * stage_bytes + scale_bytes


def _bf16_round(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


class _TensorMetadata:
    """Inert tensor metadata for AST-extracted host routing checks."""

    def __init__(self, shape, dtype, strides=None, device="mock-device"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        if strides is None:
            strides = []
            pitch = 1
            for extent in reversed(self.shape):
                strides.append(pitch)
                pitch *= extent
            strides = tuple(reversed(strides))
        self._strides = tuple(strides)

    def dim(self):
        return len(self.shape)

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]

    def is_contiguous(self):
        pitch = 1
        for extent, stride in zip(reversed(self.shape), reversed(self._strides)):
            if stride != pitch:
                return False
            pitch *= extent
        return True

    def unsqueeze(self, index):
        shape = list(self.shape)
        shape.insert(index, 1)
        return _TensorMetadata(shape, self.dtype, device=self.device)


class _Column:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def eq(self, value):
        return [entry == value for entry in self.values]


class _NormalizedRows:
    """Minimal normalized-table protocol consumed by saved OPUS replay."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __getattr__(self, column):
        return _Column([getattr(row, column) for row in self.rows])

    def __getitem__(self, mask):
        return _NormalizedRows([row for row, keep in zip(self.rows, mask) if keep])

    @property
    def empty(self):
        return not self.rows

    def itertuples(self, index=False):
        return iter(self.rows)


def _generate(common, source_path, helper_path, kids):
    helpers = _definitions(helper_path, {
        "_INSTANCE_IMPL_PREAMBLE_TEMPLATE", "instance_impl_preamble", "_make_a8w8_bpreshuffle_host_decl",
    }, {})
    registry = common.a8w8_mxscale_gemm_bpreshuffle_kernels_list
    generator = _definitions(source_path, {"_bpreshuffle_compact_traits", "gen_mxscale_bpreshuffle_instance"}, {
        "os": os, "Path": Path, "OpusGemmInstance": common.OpusGemmInstance,
        "a8w8_mxscale_gemm_bpreshuffle_kernels_list": registry,
    })["gen_mxscale_bpreshuffle_instance"]
    outputs = {}
    with tempfile.TemporaryDirectory() as directory:
        cg = SimpleNamespace(impl_path=directory, _kid_pipeline_header={},
                             _host_instantiations=[], _device_instantiations=[])
        for kid in sorted(kids):
            instance = registry[kid]
            generator(cg, instance, pipeline_header="unused", traits_header="unused",
                      kernel_func="unused", traits_name="unused",
                      kargs_name="opus_gemm_mxscale_bpreshuffle_kargs_gfx950",
                      instance_impl_preamble=helpers["instance_impl_preamble"],
                      make_a8w8_bpreshuffle_host_decl=helpers["_make_a8w8_bpreshuffle_host_decl"])
            outputs[kid] = SimpleNamespace(
                source=(Path(directory) / f"{instance.name}.cuh").read_text(),
                host=cg._host_instantiations[-1]["host_decl"],
                device=cg._device_instantiations[-1]["device_decl"],
            )
    return outputs


class RuntimeSplitKCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules_before = set(sys.modules)
        with _cpu_imports():
            variants = _load("_runtime_split_cpu_variants", OPUS / "opus_gemm_bpreshuffle_variants.py")
            with patch.dict(sys.modules, {"opus_gemm_bpreshuffle_variants": variants}):
                cls.common = _load("_runtime_split_cpu_common", OPUS / "opus_gemm_common.py")
            before_variants = _load("_runtime_split_cpu_before_variants", BEFORE / "csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py")
            with patch.dict(sys.modules, {"opus_gemm_bpreshuffle_variants": before_variants}):
                cls.before = _load("_runtime_split_cpu_before_common", BEFORE / "csrc/opus_gemm/opus_gemm_common.py")
        cls.variants = variants.NEW_BPRESHUFFLE_VARIANTS_BY_KID
        cls.registry = cls.common.a8w8_mxscale_gemm_bpreshuffle_kernels_list
        cls.generated = _generate(cls.common, OPUS / "codegen/gen_instances_gfx950.py", OPUS / "gen_instances.py", cls.registry)
        cls.generated_before = _generate(cls.before, BEFORE / "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
                                         BEFORE / "csrc/opus_gemm/gen_instances.py", cls.registry)

    def test_nine_runtime_candidates_and_three_legacy_aliases(self):
        self.assertEqual({kid for kid, variant in self.variants.items() if variant.runtime_split_k}, RUNTIME_IDS)
        self.assertEqual(len(self.registry), 105)
        self.assertEqual(len(self.common.A8W8_BPRESHUFFLE_TUNING_KIDS), 89)
        self.assertEqual(len(self.common.A8W8_BPRESHUFFLE_LEGACY_KIDS), 16)
        for alias, canonical in ALIAS_IDS.items():
            self.assertEqual(self.common.A8W8_BPRESHUFFLE_LEGACY_KIDS[alias], canonical)
            self.assertNotIn(alias, self.common.A8W8_BPRESHUFFLE_TUNING_KIDS)
            self.assertIn(canonical, self.common.A8W8_BPRESHUFFLE_TUNING_KIDS)
        for kid in RUNTIME_IDS:
            self.assertEqual(self.variants[kid].split_k, 1 if kid in FINE_IDS else 4)

    def test_nonruntime_metadata_names_and_generated_launchers_are_preserved(self):
        old_fields = [field.name for field in fields(self.before.OpusGemmInstance)]
        descriptor_fields = [field.name for field in fields(next(iter(self.before.NEW_BPRESHUFFLE_VARIANTS)))]
        for kid, frozen in self.before.a8w8_mxscale_gemm_bpreshuffle_kernels_list.items():
            if kid in RUNTIME_IDS:
                continue
            current = self.registry[kid]
            with self.subTest(kid=kid):
                for name in old_fields:
                    if name == "bpreshuffle_variant" and frozen.bpreshuffle_variant is not None:
                        self.assertEqual({key: getattr(current.bpreshuffle_variant, key) for key in descriptor_fields},
                                         {key: getattr(frozen.bpreshuffle_variant, key) for key in descriptor_fields})
                    else:
                        self.assertEqual(getattr(current, name), getattr(frozen, name), name)
                self.assertEqual(current.name, frozen.name)
                self.assertEqual(self.generated[kid].source, self.generated_before[kid].source)
                self.assertEqual(self.generated[kid].host, self.generated_before[kid].host)
                self.assertEqual(self.generated[kid].device, self.generated_before[kid].device)

    def test_all_literal_partitions_are_enumerated_and_cap_at_16_k128_tiles(self):
        choose = self.common.bpreshuffle_candidate_split_k
        for kid in sorted(RUNTIME_IDS):
            instance = self.registry[kid]
            for total in range(1, 129):
                with self.subTest(kid=kid, total=total):
                    expected = list(range(1, min(16, total) + 1))
                    self.assertEqual(choose(instance, 1, 128, total * 128), expected)
                    self.assertEqual(choose(instance, 1, 128, total * 128, cu_num=1), expected)
        instance = self.registry[92310]
        for requested in (1, 3, 5, 16):
            self.assertEqual(choose(instance, 1, 128, 2048, requested=requested), [requested])
        for requested in (-2, 17, 3):
            with self.assertRaises(ValueError):
                self.common.bpreshuffle_launch_plan(instance, 1, 128, 128, split_k=requested)
        for requested in (True, 1.5, "3"):
            with self.assertRaises(ValueError):
                self.common.bpreshuffle_launch_plan(instance, 1, 128, 2048, split_k=requested)

    def test_default_zero_retains_short_k_fourway_register_partitions(self):
        for kid in sorted(RUNTIME_IDS):
            instance = self.registry[kid]
            default = 1 if kid in FINE_IDS else 4
            for total in (1, 2, 3, 4, 16, 128):
                plan = self.common.bpreshuffle_launch_plan(instance, 1, 128, total * 128)
                with self.subTest(kid=kid, total=total):
                    self.assertEqual(plan.split_k, default)
                    self.assertEqual(plan.grid[2], default)
                    self.assertEqual(plan.workspace_elements, default * 128 if default > 1 else 0)
                    self.assertEqual(self.common.bpreshuffle_candidate_split_k(instance, 1, 128, total * 128, requested=0), [default])

    def test_auto_split_uses_caller_cu_count_and_literal_grid_capacity(self):
        for kid in sorted(RUNTIME_IDS):
            instance = self.registry[kid]
            for shape in ((1, 128, 384), (1, 128, 16384), (512, 65536, 16384)):
                m, n, k = shape
                base_grid = math.ceil(m / instance.B_M) * (n // instance.B_N)
                for cu_num in (1, 8, 64, 256, 1024):
                    expected = max(1, min(16, k // 128, math.ceil(cu_num / base_grid)))
                    with self.subTest(kid=kid, shape=shape, cu_num=cu_num):
                        plan = self.common.bpreshuffle_launch_plan(instance, *shape, split_k=-1, cu_num=cu_num)
                        self.assertEqual(plan.split_k, expected)
                        self.assertEqual(self.common.bpreshuffle_candidate_split_k(instance, *shape, requested=-1, cu_num=cu_num), [expected])
        for cu_num in (0, -1, True, 256.5):
            with self.assertRaises(ValueError):
                self.common.bpreshuffle_launch_plan(self.registry[92310], 1, 128, 2048, split_k=-1, cu_num=cu_num)
        # Auto must fall back to the direct path when a partial plane is too big.
        oversized_plane = (512, 1_048_576, 256)
        for kid in (92310, 92410):
            plan = self.common.bpreshuffle_launch_plan(self.registry[kid], *oversized_plane,
                                                       split_k=-1, cu_num=2**30)
            self.assertEqual(plan.split_k, 1)
            self.assertEqual(plan.workspace_bytes, 0)

    def test_launch_grid_lds_workspace_match_independent_partition_oracle(self):
        for kid in sorted(RUNTIME_IDS):
            instance = self.registry[kid]
            for total in range(1, 129):
                for split in range(1, min(16, total) + 1):
                    m, n, k = 17, 384, total * 128
                    plan = self.common.bpreshuffle_launch_plan(instance, m, n, k, split_k=split)
                    with self.subTest(kid=kid, total=total, split=split):
                        self.assertEqual(plan.grid, (n // instance.B_N, math.ceil(m / instance.B_M), split))
                        self.assertEqual(plan.workspace_elements, split * m * n if split > 1 else 0)
                        self.assertEqual(plan.workspace_bytes, 4 * plan.workspace_elements)
                        if kid in FINE_IDS:
                            expected_lds = _fine_lds_reference(instance.B_M, instance.B_N, k, split)
                            self.assertEqual(plan.lds_bytes, expected_lds)
                            self.assertLessEqual(plan.lds_bytes, 163840)
                        else:
                            # The launch field excludes compile-time local-wave LDS.
                            self.assertEqual(plan.lds_bytes, 0)
                        # Local WK2 reductions may have an empty second wave.
                        ranges = _partition_reference(total, split)
                        covered = []
                        for begin, count in ranges:
                            for local_begin, local_count in _partition_reference(count, self.variants[kid].wave_k):
                                covered.extend(range(begin + local_begin, begin + local_begin + local_count))
                        self.assertEqual(covered, list(range(total)))
        plan = self.common.bpreshuffle_launch_plan(self.registry[92410], 1, 128, 384, split_k=3)
        with self.assertRaises(FrozenInstanceError):
            plan.split_k = 2

    def test_fp32_partition_signed_limit_and_total_workspace_64bit_extent(self):
        for kid in (92310, 92410):
            instance = self.registry[kid]
            # C fits signed-int bytes; one FP32 workspace plane does not.
            shape = (512, 1_048_576, 256)
            self.assertEqual(self.common.bpreshuffle_candidate_split_k(instance, *shape), [1])
            self.assertEqual(self.common.bpreshuffle_launch_plan(instance, *shape, split_k=1).workspace_bytes, 0)
            with self.assertRaises(ValueError):
                self.common.bpreshuffle_launch_plan(instance, *shape, split_k=2)
            # Each plane remains <2GiB while all16 planes exceed2GiB.
            shape = (512, 1_048_448, 2048)
            plan = self.common.bpreshuffle_launch_plan(instance, *shape, split_k=16)
            self.assertLess(4 * shape[0] * shape[1], INT32_MAX)
            self.assertGreater(plan.workspace_bytes, INT32_MAX)
            self.assertEqual(plan.workspace_bytes, 16 * shape[0] * shape[1] * 4)
            self.assertLessEqual(plan.workspace_bytes, INT64_MAX)

    def test_illegal_geometry_and_nonruntime_split_overrides_are_rejected(self):
        runtime = self.registry[92310]
        for shape in ((0, 128, 128), (513, 128, 128), (1, 127, 128), (1, 128, 127),
                      (1, 128, 16512), (512, 16_777_216, 128)):
            with self.subTest(shape=shape):
                self.assertEqual(self.common.bpreshuffle_candidate_split_k(runtime, *shape), [])
                with self.assertRaises(ValueError):
                    self.common.bpreshuffle_launch_plan(runtime, *shape)
        for kid in set(self.registry) - RUNTIME_IDS:
            instance = self.registry[kid]
            k = instance.bpreshuffle_fixed_k or 7168
            m = instance.m_align
            n = max(instance.B_N, 128)
            if instance.name_tag == "large_output" or (instance.bpreshuffle_variant and instance.bpreshuffle_variant.family == "large_output"):
                m, n, k = 65536, 65536, instance.bpreshuffle_fixed_k or 1536
            with self.subTest(kid=kid):
                self.assertEqual(self.common.bpreshuffle_candidate_split_k(instance, m, n, k), [0])
                self.common.bpreshuffle_launch_plan(instance, m, n, k)
                for split in (-1, 1, 2):
                    with self.assertRaises(ValueError):
                        self.common.bpreshuffle_launch_plan(instance, m, n, k, split_k=split)

    def test_reducer_rounds_bf16_once_after_all_fp32_planes(self):
        # Rounding each partial first would produce1.0 instead of1.0078125.
        self.assertEqual(_bf16_round(1.003 + 0.003), 1.0078125)
        self.assertNotEqual(_bf16_round(_bf16_round(1.003) + _bf16_round(0.003)), _bf16_round(1.003 + 0.003))
        source = (INCLUDE / "opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh").read_text()
        reducer = source[source.index("void opus_gemm_mxscale_bpreshuffle_reduce_runtime_kernel"):]
        self.assertIn("for (int split = 0; split < split_k; ++split)", reducer)
        self.assertIn("workspace + static_cast<int64_t>(split) * elements", reducer)
        self.assertIn("vector_t<float, Vec> acc{}", reducer)
        self.assertLess(reducer.index("acc[offset"), reducer.index("cast<bf16_t>"))
        self.assertEqual(reducer.count("cast<bf16_t>"), 1)
        self.assertNotIn("atomic", reducer)

    def test_runtime_codegen_has_direct_and_partial_branches_with_one_reducer(self):
        for kid in sorted(RUNTIME_IDS):
            instance = self.registry[kid]
            variant = self.variants[kid]
            output = self.generated[kid]
            source = output.source
            with self.subTest(kid=kid):
                self.assertIn("workspace, int split_k", output.host)
                self.assertIn("opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950", output.device)
                self.assertIn(f"effective_split_k = split_k == 0 ? {variant.split_k} : split_k", source)
                self.assertIn("explicit split_k must not exceed K/128", source)
                self.assertIn("args.split_k = effective_split_k", source)
                self.assertIn(f"grid(n / {instance.B_N}, tiles_m, effective_split_k)", source)
                self.assertIn("{size_t(effective_split_k), size_t(m), size_t(n)}", source)
                self.assertIn("AITER_DTYPE_fp32, required, 16", source)
                self.assertIn("required <= uint64_t(9223372036854775807) / sizeof(float)", source)
                self.assertIn("workspace must not overlap input/output storage", source)
                self.assertIn("split_k=1 does not use workspace", source)
                self.assertIn("(effective_split_k > 1 ? sizeof(float) : sizeof(D_C))", source)
                self.assertIn("if (effective_split_k > 1)", source)
                reducer = "opus_gemm_mxscale_bpreshuffle_reduce_runtime_kernel<16, 128>"
                self.assertEqual(source.count(reducer + "<<<"), 1)
                self.assertIn(reducer + "(const float*, opus::bf16_t*, int, int)", output.device)
                producer = source.index(variant.kernel + "<" + instance.name + "_Traits")
                reduce_launch = source.index(reducer + "<<<")
                self.assertLess(producer, reduce_launch)
                self.assertIn("m * n, effective_split_k", source[reduce_launch:])
                if kid in FINE_IDS:
                    self.assertIn(instance.name + "_Traits, false>", output.device)
                    self.assertIn(instance.name + "_Traits, true>", output.device)
                    self.assertIn("lds_bytes(k, effective_split_k)", source)
                    self.assertIn("lds_bytes <= 160 * 1024", source)
                else:
                    self.assertIn("grid, dim3(" + str(instance.BLOCK_SIZE) + "), 0, aiter::getCurrentHIPStream()", source)
                launches = source[producer:]
                self.assertEqual(launches.count("aiter::getCurrentHIPStream()"), 3 if kid in FINE_IDS else 2)

    def test_dispatch_generation_and_manifest_keep_runtime_and_fixed_abis_separate(self):
        names = {"gen_a8w8_kid_dispatch", "gen_manifest_head"}
        namespace = _definitions(OPUS / "codegen/common.py", {
            "W3_KERNEL_PAIRS", "_NOSPLIT", "_GFX942_SPLITK_ONLY", "_SPLITK",
            "_GFX942_A16W16_TAGS", "_A16W16_CO_TAGS", "_A16W16_TAGS", "kid_arch",
        }, {})
        namespace["_kid_arch_common"] = namespace["kid_arch"]
        namespace.update(os=os, SPLITK_REDUCE_ARCHES=("gfx950", "gfx942", "gfx1250"))
        _definitions(OPUS / "gen_instances.py", {
            "A16W16_KID_DISPATCH_TAGS", "SPLITK_TAGS", "A8W8_BPRESHUFFLE_TAGS",
        }, namespace)

        def generate(source, kernels):
            tree = ast.parse(source.read_text())
            codegen = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "opus_gemm_codegen")
            methods = [node for node in codegen.body
                       if isinstance(node, ast.FunctionDef) and node.name in names]
            self.assertEqual({node.name for node in methods}, names)
            extracted = dict(namespace)
            exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), extracted)
            with tempfile.TemporaryDirectory() as directory, _cpu_imports():
                cg = SimpleNamespace(working_path=directory)
                # Reversing the registry ensures the generator sorts each table itself.
                unsorted = dict(reversed(list(kernels.items())))
                for name in names:
                    extracted[name](cg, unsorted)
                dispatch = (Path(directory) / "opus_gemm_a8w8_kid_dispatch.h").read_text()
                manifest = (Path(directory) / "opus_gemm_manifest.h").read_text()
            tables = {}
            for match in re.finditer(r"#define (\w+)_SIZE (\d+)\n(.*?)\n\n", dispatch, re.S):
                name, size, body = match.groups()
                self.assertTrue(body.startswith(f"#define {name}"))
                rows = [(int(kid), symbol, dtype) for kid, symbol, dtype in
                        re.findall(r"\{\s*(\d+),\s*&(\w+)<(\w+)>\s*\}", body)]
                self.assertEqual(int(size), len(rows), name)
                self.assertEqual([row[0] for row in rows], sorted({row[0] for row in rows}), name)
                tables[name] = (rows, match.group(0))
            manifest_rows = re.findall(
                r"template <typename D_C>\nvoid\n(\w+)\((.*?)\);", manifest, re.S)
            declarations = dict(manifest_rows)
            self.assertEqual(len(manifest_rows), len(kernels))
            self.assertEqual(set(declarations), {kernel.name for kernel in kernels.values()})
            for symbol, signature in manifest_rows:
                self.assertEqual(signature, declarations[symbol])
            return tables, declarations

        tables, declarations = generate(OPUS / "gen_instances.py", self.common.kernels_list)
        frozen_tables, frozen_declarations = generate(
            BEFORE / "csrc/opus_gemm/gen_instances.py", self.before.kernels_list)
        fixed_name = "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16"
        runtime_name = "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_RUNTIME_KID_DISPATCH_GFX950_BF16"
        runtime_rows, fixed_rows = tables[runtime_name][0], tables[fixed_name][0]
        runtime_ids, fixed_ids = {row[0] for row in runtime_rows}, {row[0] for row in fixed_rows}
        self.assertEqual(runtime_ids, RUNTIME_IDS)
        self.assertEqual(len(fixed_ids), 96)
        self.assertFalse(runtime_ids & fixed_ids)
        self.assertEqual(runtime_ids | fixed_ids, set(self.registry))
        self.assertEqual(set(tables) - set(frozen_tables), {runtime_name})
        self.assertEqual(set(frozen_tables) - set(tables), set())
        for name, (rows, block) in tables.items():
            if name not in {fixed_name, runtime_name}:
                self.assertEqual(block, frozen_tables[name][1], name)
        self.assertEqual(fixed_rows, [row for row in frozen_tables[fixed_name][0]
                                     if row[0] not in RUNTIME_IDS])
        self.assertEqual([row[0] for row in tables[
            "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_BF16"][0]], [11000])

        for kid, symbol, dtype in runtime_rows + fixed_rows:
            self.assertEqual(symbol, self.registry[kid].name)
            self.assertEqual(dtype, "bf16_t")
            signature = declarations[symbol]
            self.assertEqual(signature.count("aiter_tensor_t &"), 5)
            if kid in RUNTIME_IDS:
                self.assertEqual(signature, frozen_declarations[self.before.kernels_list[kid].name]
                                 .replace("workspace", "workspace, int split_k"))
                self.assertEqual(len(signature.split(",")), 7)
            else:
                self.assertEqual(signature, frozen_declarations[symbol])
                self.assertEqual(len(signature.split(",")), 6)
        gfx942_symbol = self.common.kernels_list[11000].name
        self.assertEqual(declarations[gfx942_symbol], frozen_declarations[gfx942_symbol])
        self.assertNotIn("split_k", declarations[gfx942_symbol])

        def pointer_signature(source, name):
            match = re.search(r"using " + name + r" = void \(\*\)\((.*?)\);", source, re.S)
            self.assertIsNotNone(match, name)
            return re.sub(r"\s+", "", match.group(1))

        arch = (INCLUDE / "opus_gemm_arch_gfx950.cuh").read_text()
        frozen_arch = (BEFORE / "csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh").read_text()
        gfx942_arch = (OPUS / "include/gfx942/opus_gemm_arch_gfx942.cuh").read_text()
        old_pointer = "OpusA8W8BlockscaleBpreshuffleKernel"
        fixed_signature = pointer_signature(arch, old_pointer)
        self.assertEqual(fixed_signature, pointer_signature(frozen_arch, old_pointer))
        self.assertEqual(fixed_signature, pointer_signature(gfx942_arch, old_pointer))
        self.assertEqual(pointer_signature(arch, old_pointer.replace("Kernel", "RuntimeKernel")),
                         fixed_signature + ",int")
        self.assertNotIn("BpreshuffleRuntimeKernel", gfx942_arch)
        self.assertNotIn("BPRESHUFFLE_RUNTIME_KID_DISPATCH", gfx942_arch)

    def test_kernel_sources_use_shared_partition_helper_and_runtime_count(self):
        fine = (INCLUDE / "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_runtime_gfx950.cuh").read_text()
        register = (INCLUDE / "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_runtime_gfx950.cuh").read_text()
        self.assertIn("gemm_a8w8_mxfp8_scale_small_lds_runtime_kernel", fine)
        self.assertIn("opus_gemm_mxscale_bpreshuffle_register_runtime_kernel", register)
        self.assertIn("balanced_partition(total_loops", fine)
        self.assertIn("balanced_partition(total_tiles", register)
        self.assertIn("args.split_k", fine)
        self.assertIn("args.split_k", register)
        self.assertIn("balanced_partition(global.count", register)

    def test_tuner_records_each_literal_split_and_forwards_it_to_the_operation(self):
        path = OPUS / "opus_gemm_mxscale_bpreshuffle_tune.py"
        calls = []
        bench = _definitions(path, {"run_bench"}, {
            "opus_gemm": lambda *args, **kwargs: calls.append((args, kwargs)),
        })["run_bench"]
        tensors = [object() for _ in range(5)]
        self.assertIs(bench(*tensors, 92310, 3), tensors[2])
        self.assertEqual(calls[0][1]["split_k"], 3)

        # Extract only task construction, injecting inert data/benchmark callbacks.
        tree = ast.parse(path.read_text())
        tuner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OpusMxscaleBpreshuffleTuner")
        names = {"_make_task", "get_gemm_a8w8_blockscale_opus_tune_task"}
        methods = [node for node in tuner.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in methods}, names)
        extracted_class = ast.ClassDef(name="ScalarTuner", bases=[], keywords=[], body=methods, decorator_list=[])
        inert = lambda *args, **kwargs: None
        namespace = {
            "generate_data": inert, "run_torch": inert, "compare_outputs": inert,
            "run_bench": bench, "_REF_KEYS": ("reference",), "_BENCH_KEYS": ("inputs",),
            "a8w8_mxscale_gemm_bpreshuffle_kernels_list": self.registry,
            "bpreshuffle_candidate_split_k": self.common.bpreshuffle_candidate_split_k,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted_class], type_ignores=[])), str(path), "exec"), namespace)
        scalar = namespace["ScalarTuner"]()
        scalar._candidate_kids = lambda *args: [92310, 92410, 9000]
        tasks = scalar.get_gemm_a8w8_blockscale_opus_tune_task(("gfx950", 256, 256, 256, 384), 0, True, {})
        choices = [(task[0][1], task[0][2]) for task in tasks]
        self.assertEqual(choices, [(92310, 1), (92310, 2), (92310, 3),
                                   (92410, 1), (92410, 2), (92410, 3), (9000, 0)])
        for task in tasks:
            self.assertEqual(task[4][1:], (task[0][1], task[0][2]))
        self.assertEqual(scalar.get_gemm_a8w8_blockscale_opus_tune_task(("gfx950", 256, 256, 256, 384), 0, False, {}), [])

    def test_public_contract_accepts_runtime_literals_and_rejects_direct_workspace(self):
        family = "a8w8_blockscale_bpreshuffle"
        fake_torch = SimpleNamespace(dtype=object, bfloat16="bf16", float32="fp32")
        validate = _definitions(ROOT / "aiter/ops/opus/launch_plan.py", {"_validate_a8w8_public_contract"}, {
            "torch": fake_torch, "_FP8_DTYPES": {"fp8"},
            "_A8W8_FAMILY_BY_TAG": {"a8w8_mxscale_gemm_bpreshuffle": family},
            "_A8W8_FAMILY_LAYOUT": {family: "bpreshuffle"},
            "_A8W8_BPRESHUFFLE_FAMILY": family, "_A8W8_FAMILY": "a8w8",
            "_A8W8_MXSCALE_BMM_FAMILY": "a8w8_mxscale_bmm",
        })["_validate_a8w8_public_contract"]
        base = dict(kernel_tag="a8w8_mxscale_gemm_bpreshuffle", kid=92310,
                    input_dtype="fp8", weight_dtype="fp8", output_dtype="bf16",
                    layout="bpreshuffle", has_x_scale=True, has_w_scale=True,
                    has_bias=False, has_workspace=False, bpreshuffle_runtime_split_k=True,
                    bpreshuffle_split_k=4)
        for split in (-1, 0, 1, 3, 5, 16):
            self.assertEqual(validate(**base, split_k=split), family)
        for split in (-2, 17, True, 1.5, "3"):
            with self.assertRaises(ValueError):
                validate(**base, split_k=split)
        with self.assertRaises(ValueError):
            validate(**{**base, "has_workspace": True}, split_k=1)
        with self.assertRaises(ValueError):
            validate(**{**base, "has_workspace": True, "bpreshuffle_split_k": 1}, split_k=0)
        for split in (0, 3, -1):
            self.assertEqual(validate(**{**base, "has_workspace": True}, split_k=split), family)
        for overrides in ({"has_x_scale": False, "has_w_scale": False}, {"has_bias": True},
                          {"output_dtype": "fp32"}, {"layout": "plain"}):
            with self.assertRaises(ValueError):
                validate(**{**base, **overrides}, split_k=3)
        fixed = {**base, "bpreshuffle_runtime_split_k": False}
        self.assertEqual(validate(**fixed, split_k=0), family)
        for split in (-1, 1, 2):
            with self.assertRaises(ValueError):
                validate(**fixed, split_k=split)

    def test_workspace_capability_uses_the_effective_plan_split(self):
        family = "a8w8_blockscale_bpreshuffle"
        for kid in sorted(RUNTIME_IDS):
            self.assertTrue(self.common.kernel_needs_external_workspace("gfx950", family, kid))
            self.assertTrue(self.common.kernel_needs_external_workspace("gfx950", family, kid, split_k=-1))
            for split in (1, 3, 5, 16):
                plan = self.common.bpreshuffle_launch_plan(self.registry[kid], 1, 128, 2048, split_k=split)
                self.assertEqual(self.common.kernel_needs_external_workspace("gfx950", family, kid, split_k=plan.split_k),
                                 plan.workspace_elements > 0)

    def test_public_launcher_allocates_from_plan_and_preserves_default_sentinel(self):
        allocations, launches, queries = [], [], []

        def empty(elements, *, device, dtype):
            allocations.append(elements)
            return _TensorMetadata((elements,), dtype, device=device)

        def device_properties(device):
            queries.append(device)
            return SimpleNamespace(multi_processor_count=256)

        fake_torch = SimpleNamespace(float8_e4m3fn="fp8", bfloat16="bf16", float32="fp32",
                                     empty=empty, cuda=SimpleNamespace(get_device_properties=device_properties))
        launcher = _definitions(ROOT / "aiter/ops/opus/gemm_op_a8w8.py", {"_launch_a8w8_blockscale_bpreshuffle_gemm"}, {
            "Tensor": _TensorMetadata, "torch": fake_torch, "_E8M0_DTYPES": {"e8m0", "uint8"},
            "_A8W8_BPRESHUFFLE_FAMILY": "a8w8_blockscale_bpreshuffle",
            "_opus_gemm_bpreshuffle_runtime_raw": lambda *args: launches.append(args),
            "_launch_a8w8_backend": lambda *args: self.fail("runtime candidate used legacy backend"),
        })["_launch_a8w8_blockscale_bpreshuffle_gemm"]
        m, n, k = 17, 384, 2048
        x = _TensorMetadata((m, k), "fp8")
        w = _TensorMetadata((n, k), "fp8")
        y = _TensorMetadata((m, n), "bf16")
        sa = _TensorMetadata((m, k // 128), "e8m0", (1, m))
        sb = _TensorMetadata((n // 128, k // 128), "e8m0")
        with _cpu_imports(), patch.dict(sys.modules, {"csrc.opus_gemm.opus_gemm_common": self.common}):
            for kid, requested, expected_split, abi_split in (
                (92310, 1, 1, 1), (92310, 3, 3, 3),
                (92310, 0, 4, 0), (92410, 0, 1, 0),
                (92410, -1, 16, 16),
            ):
                allocations.clear()
                launches.clear()
                queries.clear()
                result = launcher(x, w, sa, sb, y, kid=kid, split_k=requested,
                                  instance=self.registry[kid])
                with self.subTest(kid=kid, split=requested):
                    self.assertIs(result, y)
                    self.assertEqual(allocations, [expected_split * m * n] if expected_split > 1 else [])
                    self.assertEqual(len(launches), 1)
                    self.assertEqual(launches[0][-1], abi_split)
                    self.assertEqual(queries, [x.device] if requested == -1 else [])
                    if expected_split == 1:
                        self.assertIsNone(launches[0][-2])
                    else:
                        self.assertIsNotNone(launches[0][-2])
            provided = _TensorMetadata((3 * m * n,), "fp32")
            allocations.clear()
            launches.clear()
            launcher(x, w, sa, sb, y, kid=92310, split_k=3, workspace=provided,
                     instance=self.registry[92310])
            self.assertEqual(allocations, [])
            self.assertIs(launches[0][-2], provided)
            with self.assertRaises(ValueError):
                launcher(x, w, sa, sb, y, kid=92310, split_k=1, workspace=provided,
                         instance=self.registry[92310])
            allocations.clear()
            with self.assertRaises(ValueError):
                launcher(x, w, _TensorMetadata(sa.shape, "e8m0"), sb, y,
                         kid=92310, split_k=3, instance=self.registry[92310])
            self.assertEqual(allocations, [])

    def test_public_dispatch_forwards_explicit_and_auto_split_to_runtime_family(self):
        launches = []
        family_module = SimpleNamespace(
            _launch_a8w8_blockscale_bpreshuffle_gemm=lambda *args, **kwargs: launches.append(kwargs) or args[4]
        )
        family = "a8w8_blockscale_bpreshuffle"

        def resolve(kid, *args):
            return family, self.registry[kid], family_module

        dispatch = _definitions(ROOT / "aiter/ops/opus/dispatch.py", {"_opus_dispatch"}, {
            "Tensor": _TensorMetadata, "_resolve_contract": resolve, "GFX950": "gfx950",
        })["_opus_dispatch"]
        x = _TensorMetadata((17, 2048), "fp8")
        w = _TensorMetadata((384, 2048), "fp8")
        y = _TensorMetadata((17, 384), "bf16")
        sa = _TensorMetadata((17, 16), "e8m0", (1, 17))
        sb = _TensorMetadata((3, 16), "e8m0")
        for split in (0, 1, 3, 16, -1):
            result = dispatch("opus_gemm", 2, x, w, y, kid=92310, split_k=split,
                              layout="bpreshuffle", x_scale=sa, w_scale=sb)
            self.assertIs(result, y)
            self.assertEqual(launches[-1]["split_k"], split)
            self.assertIs(launches[-1]["instance"], self.registry[92310])
        for kid, split in ((9000, -1), (92310, -2), (92310, True)):
            with self.assertRaises(ValueError):
                dispatch("opus_gemm", 2, x, w, y, kid=kid, split_k=split,
                         layout="bpreshuffle", x_scale=sa, w_scale=sb)

    def test_saved_replay_preserves_exact_literal_split_zero_defaults_and_alias_ids(self):
        path = OPUS / "opus_gemm_mxscale_bpreshuffle_tune.py"
        calls, generated, compiled = [], [], []
        real_import = builtins.__import__

        def importer(name, *args, **kwargs):
            if name == "aiter.test_common":
                return SimpleNamespace(
                    checkAllclose=lambda *args, **kwargs: 0,
                    run_perftest=lambda bench, *args, **kwargs: (bench(*args), 1.25),
                )
            return real_import(name, *args, **kwargs)

        def data(m, n, k, seed, *, device):
            generated.append((m, n, k))
            out = SimpleNamespace(fill_=lambda value: None)
            return {"x": object(), "w": object(), "x_scale": object(),
                    "w_scale": object(), "out": out}

        namespace = {
            "__builtins__": {**vars(builtins), "__import__": importer},
            "_SUPPORTED_LIBTYPES": {"opus", "ck", "cktile", "asm"},
            "candidate_kids_for_shape": self.common.a8w8_mxscale_bpreshuffle_candidate_kids,
            "bpreshuffle_launch_plan": self.common.bpreshuffle_launch_plan,
            "a8w8_mxscale_gemm_bpreshuffle_kernels_list": self.registry,
            "pd": SimpleNamespace(isna=lambda value: value is None,
                                  notna=lambda value: value is not None),
            "_ensure_kids_compiled": lambda kids: compiled.append(kids),
            "generate_data": data, "run_torch": lambda *args, **kwargs: object(),
            "compare_outputs": lambda *args, **kwargs: 0,
            "_REF_KEYS": ("x", "w", "x_scale", "w_scale"),
            "_BENCH_KEYS": ("x", "w", "out", "x_scale", "w_scale"),
            "math": math,
            "opus_gemm": lambda *args, **kwargs: calls.append((kwargs["kid"], kwargs["split_k"])),
        }
        _definitions(path, {"run_bench"}, namespace)
        tree = ast.parse(path.read_text())
        tuner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OpusMxscaleBpreshuffleTuner")
        method = next(node for node in tuner.body if isinstance(node, ast.FunctionDef) and node.name == "run_config")
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        scalar = SimpleNamespace(
            untunedf=None, _normalize_rows=lambda rows, **kwargs: rows,
            get_gfx=lambda: "gfx950", get_cu_num=lambda: 256,
            getKernelName=lambda kid, *args: self.registry[kid].name,
            _error_limit=lambda *args: 0,
        )
        args = SimpleNamespace(warmup=1, iters=2, verbose=False, errRatio=0)

        def row(kid, split, m=1, n=128, k=2048):
            return SimpleNamespace(gfx="gfx950", cu_num=256, M=m, N=n, K=k,
                                   libtype="opus", kernelId=kid, splitK=split,
                                   outdtype="bf16", kernelName=self.registry[kid].name)

        saved = [row(92310, 1), row(92310, 3), row(92310, 5),
                 row(92310, 0, k=128), row(92410, 0, k=128),
                 *[row(alias, 0, k=128) for alias in ALIAS_IDS]]
        scalar.untunedf = _NormalizedRows(saved)
        with _cpu_imports():
            results = namespace["run_config"](scalar, args)
        self.assertEqual(calls, [(entry.kernelId, entry.splitK) for entry in saved])
        self.assertEqual(compiled, [{entry.kernelId for entry in saved}])
        self.assertEqual(len(generated), len(saved))
        self.assertEqual(len(results), len(saved))
        self.assertEqual(len({result["shape"] for result in results}), len(saved))
        for entry, result in zip(saved, results):
            self.assertIn(f"kid={entry.kernelId},splitK={entry.splitK}", result["shape"])
            self.assertEqual(result["status"], "ok")

        # Validation must finish before creating data or compiling any saved ID.
        invalid = [row(9000, 1, m=256, n=256, k=384), row(92310, -1),
                   row(92310, 3, k=128), row(92411, 2, k=128)]
        for entry in invalid:
            generated.clear()
            compiled.clear()
            calls.clear()
            scalar.untunedf = _NormalizedRows([entry])
            with self.subTest(kid=entry.kernelId, split=entry.splitK), _cpu_imports():
                with self.assertRaises(ValueError):
                    namespace["run_config"](scalar, args)
                self.assertEqual(generated, [])
                self.assertEqual(compiled, [])
                self.assertEqual(calls, [])

    def test_imports_and_test_execution_do_not_load_gpu_packages(self):
        imported = set(sys.modules) - self.modules_before
        self.assertFalse({name for name in imported if name.split(".")[0] in {"torch", "aiter", "hip", "hsa"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
