"""CPU checks for per-configuration JIT identity and real raw-call conversion.

The production runtime is loaded by path, and compile_ops is AST-extracted
with tensor/JIT substitutes. No torch, aiter initialization, or HIP query runs.
"""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class _RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "hip", "hsa", "flydsl"}:
            raise AssertionError(f"CPU JIT check attempted GPU import: {fullname}")
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
    spec = importlib.util.spec_from_file_location(
        "_bpreshuffle_jit_cpu_runtime", ROOT / "aiter/ops/opus/bpreshuffle_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)


@dataclass(frozen=True)
class _Config:
    tile_m: int
    runtime_split_k: bool = False
    _legacy_kid: int = -1
    pipeline: str = "lds"
    pin_agpr: bool = False

    @property
    def legacy_kid(self):
        return self._legacy_kid

    def to_json(self):
        return json.dumps({"tile_m": self.tile_m, "runtime_split_k": self.runtime_split_k},
                          sort_keys=True, separators=(",", ":"))


class _Tensor:
    def __init__(self, name):
        self.name = name


@dataclass
class _Pod:
    name: str


def _real_compile_ops(core):
    """Execute the actual develop=True wrapper with isolated host substitutes."""
    path = ROOT / "aiter/jit/core.py"
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "compile_ops")
    namespace = {
        "Callable": typing.Callable, "Any": typing.Any, "Optional": typing.Optional,
        "functools": functools, "types": types, "typing": typing,
        "torch_compile_guard": lambda **_kw: lambda func: func,
        "AITER_REBUILD": 0, "AITER_LOG_MORE": 0, "rebuilded_list": [], "__mds": {},
        "get_module": core.get_module, "get_args_of_build": core.get_args_of_build,
        "build_module": core.build_module, "get_asm_dir": lambda: "unused",
        "_pybind_develop_hooks": lambda: (
            lambda tensor: _Pod(tensor.name), _Tensor, lambda _device: 456, lambda: 0),
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["compile_ops"]


class _Core:
    def __init__(self, directory):
        self.bd_dir = str(directory / "build")
        self.AITER_CSRC_DIR = str(directory / "csrc")
        Path(self.AITER_CSRC_DIR, "include").mkdir(parents=True)
        Path(self.AITER_CSRC_DIR, "opus_gemm").mkdir()
        self.header = Path(self.AITER_CSRC_DIR, "include/producer.hpp")
        self.header.write_text("producer revision 1\n")
        self.modules = {runtime._BASE_MODULE: types.ModuleType(runtime._BASE_MODULE)}
        self.calls = []
        self.builds = []
        self.lookups = []
        self.compile_ops = _real_compile_ops(self)

    def get_args_of_build(self, module, *, overrides=None):
        self.assert_base(module)
        args = {
            "srcs": ["legacy.cpp"], "md_name": module,
            "flags_extra_cc": [], "flags_extra_hip": ["-mllvm --amdgpu-mfma-vgpr-form"],
            "blob_gen_cmd": "legacy-generator", "extra_include": [], "extra_ldflags": None,
            "verbose": False, "is_python_module": True, "is_standalone": False,
            "torch_exclude": True, "hip_clang_path": None, "third_party": [],
            "flags_extra_hip_per_source": {"*.device.cu": ["-D__HIPCC_RTC__"]},
            "hip_compiler_commands_per_source": {},
        }
        args.update(overrides or {})
        return args

    def assert_base(self, module):
        if module != runtime._BASE_MODULE:
            raise AssertionError(f"unexpected build template: {module}")

    def get_module(self, module):
        self.lookups.append(module)
        if module not in self.modules:
            raise ModuleNotFoundError(module)
        return self.modules[module]

    def build_module(self, name, *args, **kwargs):
        self.builds.append((name, args, kwargs))
        module = types.ModuleType(name)
        module._set_current_hip_stream = lambda stream: self.calls.append((name, "stream", stream))
        for suffix in ("launch", "workspace_launch", "runtime_launch"):
            function_name = f"{runtime._PREFIX}_{suffix}"

            def raw(*call_args, _name=function_name):
                self.calls.append((name, _name, call_args))

            # check_args sees a pybind-style non-signature doc and skips its
            # introspection path while the production conversion executes.
            raw.__doc__ = "Members: CPU substitute"
            setattr(module, function_name, raw)
        self.modules[name] = module

    def executable_path(self, name):
        return str(Path(self.bd_dir).parent / "toolchain/bin" / name)

    def get_hip_version(self):
        return "7.0.0"


class BpreshuffleJitCPU(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.core = _Core(self.root)
        self.torch = types.ModuleType("torch")
        self.torch.Tensor = _Tensor
        runtime._prepared.clear()
        runtime._compiler_identities.clear()
        self.patches = [
            patch.object(runtime, "_core", return_value=self.core),
            patch.object(runtime, "_canonical_config", side_effect=lambda config: config),
            patch.object(runtime, "_compiler_plan", side_effect=lambda _config, _core: (
                {"*.device.cu": ["cpu-pin-compiler"]},
                {"compiler": os.environ.get("OPUS_HIP_CLANG_PATH", "default")})),
            patch.dict(sys.modules, {"torch": self.torch}),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def test_two_configurations_build_distinct_modules_and_convert_real_wrapper_args(self):
        tensors = tuple(_Tensor(name) for name in ("x", "w", "sx", "sw", "y"))
        first = runtime.ensure_config(_Config(32, _legacy_kid=9000))
        second = runtime.ensure_config(_Config(64))
        self.assertNotEqual(first.module_name, second.module_name)
        self.assertEqual(len(self.core.builds), 2)
        self.assertNotIn(runtime._BASE_MODULE, self.core.lookups)
        for prepared in (first, second):
            self.assertIs(prepared.launch(*tensors), tensors[-1])
        launches = [call for call in self.core.calls if call[1] != "stream"]
        self.assertEqual([call[0] for call in launches], [first.module_name, second.module_name])
        self.assertEqual([call[2][-1] for call in launches], [9000, 1])
        for call in launches:
            self.assertEqual([pod.name for pod in call[2][:-1]], [tensor.name for tensor in tensors])
            self.assertTrue(all(isinstance(pod, _Pod) for pod in call[2][:-1]))
        self.assertEqual([call[2] for call in self.core.calls if call[1] == "stream"], [456, 456])

    def test_static_workspace_and_runtime_split_select_the_matching_raw_abi(self):
        tensors = tuple(_Tensor(name) for name in ("x", "w", "sx", "sw", "y"))
        workspace = _Tensor("partials")
        runtime.launch_config(_Config(32), *tensors, workspace)
        runtime.launch_config(_Config(64, runtime_split_k=True), *tensors, workspace, 7)
        launches = [call for call in self.core.calls if call[1] != "stream"]
        self.assertTrue(launches[0][1].endswith("_workspace_launch"))
        self.assertIsInstance(launches[0][2][-1], _Pod)
        self.assertTrue(launches[1][1].endswith("_runtime_launch"))
        self.assertEqual(launches[1][2][-1], 7)
        self.assertIsInstance(launches[1][2][-2], _Pod)

    def test_explicit_prepare_invalidates_source_and_compiler_identity_in_same_process(self):
        config = _Config(32)
        first = runtime.ensure_config(config)
        self.assertIs(runtime.ensure_config(config), first)
        self.core.header.write_text("producer revision 2\n")
        second = runtime.ensure_config(config)
        self.assertNotEqual(first.module_name, second.module_name)
        catalog = Path(self.core.AITER_CSRC_DIR, "opus_gemm/catalog.json")
        catalog.write_text('{"revision": 1}\n')
        third = runtime.ensure_config(config)
        self.assertNotEqual(second.module_name, third.module_name)
        catalog.write_text('{"revision": 2}\n')
        fourth = runtime.ensure_config(config)
        self.assertNotEqual(third.module_name, fourth.module_name)
        producer = Path(self.core.AITER_CSRC_DIR, "opus_gemm/include/producer.cuh")
        producer.parent.mkdir(parents=True)
        producer.write_text("producer traits revision 1\n")
        fifth = runtime.ensure_config(config)
        self.assertNotEqual(fourth.module_name, fifth.module_name)
        producer.write_text("producer traits revision 2\n")
        sixth = runtime.ensure_config(config)
        self.assertNotEqual(fifth.module_name, sixth.module_name)
        with patch.dict(os.environ, {"OPUS_HIP_CLANG_PATH": "/cpu/compiler-v2"}):
            seventh = runtime.ensure_config(config)
        self.assertNotEqual(sixth.module_name, seventh.module_name)
        self.assertEqual(len(self.core.builds), 7)

    def test_hot_launch_has_no_source_or_compiler_probe_and_does_not_mutate_environment(self):
        config = _Config(32)
        runtime.ensure_config(config)
        before = dict(os.environ)
        with patch.object(runtime, "_build_arguments", side_effect=AssertionError("unexpected rebuild")):
            tensors = tuple(_Tensor(name) for name in ("x", "w", "sx", "sw", "y"))
            for _ in range(2):
                runtime.launch_config(config, *tensors)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(len(self.core.builds), 1)

    def test_request_is_complete_and_blob_command_preserves_spaces_and_braces(self):
        self.core.bd_dir = str(self.root / "build {odd} 'quoted' directory")
        config = _Config(32)
        prepared = runtime.ensure_config(config)
        _, args, _ = self.core.builds[-1]
        self.assertEqual(args[0], [])
        self.assertIn("-DENABLE_CK=0", args[1])
        self.assertIn("-DENABLE_CK=0", args[2])
        command = shlex.split(args[3].format("/cpu/staging"))
        request = Path(command[command.index("--bpreshuffle_config") + 1])
        self.assertEqual(request.stem, prepared.module_name)
        self.assertEqual(request.parent.name, "opus_bpreshuffle_requests")
        payload = json.loads(request.read_text())
        self.assertEqual(payload, {"pipeline": "lds", "compile_params": json.loads(config.to_json())})
        self.assertEqual(command[-1], str(Path(self.core.bd_dir) / prepared.module_name / "blob.staging"))

    def test_compiler_plan_keeps_pin_override_and_resources_separate_without_env_changes(self):
        # Run the production compiler selection rather than the build fixture's
        # simplified fingerprint, using filesystem-only compiler substitutes.
        production_plan = self.patches[2].temp_original
        for name in ("pin", "baseline"):
            directory = self.root / name
            (directory / "bin").mkdir(parents=True)
            (directory / "bin/clang++").write_text("CPU compiler substitute\n")
            (directory / "resource/include").mkdir(parents=True)
            (directory / "resource/include/stddef.h").write_text(f"{name} headers\n")
        catalog = types.SimpleNamespace(get_config_dependencies=lambda config: ())
        identity = lambda compiler: {"path": str(compiler), "version": "clang version 23"}
        settings = {
            "OPUS_HIP_CLANG_PATH": str(self.root / "pin/bin"),
            "OPUS_BASELINE_HIP_CLANG_PATH": str(self.root / "baseline/bin"),
            "OPUS_HIP_RESOURCE_DIR": str(self.root / "pin/resource"),
            "OPUS_BASELINE_HIP_RESOURCE_DIR": str(self.root / "baseline/resource"),
        }
        with patch.dict(os.environ, settings), patch.object(runtime.importlib, "import_module", return_value=catalog), \
                patch.object(runtime, "_compiler_identity", side_effect=identity):
            before = dict(os.environ)
            commands, metadata = production_plan(_Config(32, pin_agpr=True), self.core)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(list(commands), ["*.device.cu", "*.cu"])
        self.assertIn(str(self.root / "pin/bin/clang++"), commands["*.device.cu"])
        self.assertIn(str(self.root / "baseline/bin/clang++"), commands["*.cu"])
        self.assertEqual(commands["*.device.cu"][-3:], ["--", "-mllvm", "--amdgpu-mfma-vgpr-form"])
        self.assertEqual(metadata["resources"]["pin"]["files"][0]["path"],
                         str(self.root / "pin/resource/include/stddef.h"))

    def test_baseline_only_build_template_skips_aggregate_pin_helper_before_eval(self):
        path = ROOT / "aiter/jit/core.py"
        tree = ast.parse(path.read_text())
        definition = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "get_args_of_build")
        directory = self.root / "jit-config"
        directory.mkdir()
        (directory / "optCompilerConfig.json").write_text(json.dumps({runtime._BASE_MODULE: {
            "md_name": repr(runtime._BASE_MODULE),
            "hip_compiler_commands_per_source": "aggregate_pin_helper()",
        }}))

        def aggregate_pin_helper():
            raise FileNotFoundError("aggregate pin environment is absent")

        namespace = {"json": json, "this_dir": str(directory),
                     "aggregate_pin_helper": aggregate_pin_helper}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        builder = namespace["get_args_of_build"]
        settings = {"OPUS_BASELINE_HIP_CLANG_PATH": "/cpu/baseline"}
        with patch.dict(os.environ, settings, clear=True):
            resolved = builder(runtime._BASE_MODULE, overrides={"hip_compiler_commands_per_source": {}})
            self.assertEqual(resolved["hip_compiler_commands_per_source"], {})
            with self.assertRaisesRegex(FileNotFoundError, "aggregate pin"):
                builder(runtime._BASE_MODULE)
        production_plan = self.patches[2].temp_original
        compiler = self.root / "baseline/bin/clang++"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("CPU compiler substitute\n")
        resources = self.root / "baseline/resource/include"
        resources.mkdir(parents=True)
        settings = {"OPUS_BASELINE_HIP_CLANG_PATH": str(compiler.parent),
                    "OPUS_BASELINE_HIP_RESOURCE_DIR": str(resources.parent)}
        catalog = types.SimpleNamespace(get_config_dependencies=lambda _config: ())
        with patch.dict(os.environ, settings, clear=True), \
                patch.object(runtime.importlib, "import_module", return_value=catalog), \
                patch.object(runtime, "_compiler_identity", side_effect=lambda executable: {
                    "path": str(executable), "version": "clang version 23"}):
            commands, _metadata = production_plan(_Config(32), self.core)
        self.assertEqual(list(commands), ["*.cu"])
        self.assertIn(str(compiler), commands["*.cu"])

    def test_runtime_import_and_builder_have_no_eager_gpu_or_global_environment_mutation(self):
        tree = ast.parse((ROOT / "aiter/ops/opus/bpreshuffle_runtime.py").read_text())
        eager = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in eager:
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module]
            self.assertFalse(any(name.split(".")[0] in {"torch", "aiter", "hip", "hsa"} for name in names))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(target, ast.Subscript)
                                     and ast.unparse(target.value) == "os.environ" for target in targets))


class BpreshuffleSingleConfigCodegenCPU(unittest.TestCase):
    def test_all_legacy_configs_and_unregistered_producer_axes_generate_only_selected_sources(self):
        path = str(ROOT / "csrc/opus_gemm")
        with _cpu_imports():
            sys.path.insert(0, path)
            try:
                import gen_instances
                import opus_gemm_bpreshuffle_config as catalog
                new = (
                    catalog.construct_config("pin", tile_m=256, tile_n=256, scale_panel=32),
                    catalog.construct_config("tiled", tile_m=96, tile_n=128, stages=3),
                    catalog.construct_config("register", tile_m=16, tile_n=32, runtime_split_k=True, prefetch=5),
                    catalog.construct_config("lds", tile_m=64, tile_n=128, runtime_split_k=True, stages=5),
                    catalog.construct_config("large_output", tile_m=192, tile_n=256, b_direct=True, c_chunk_rows=64),
                )
                self.assertTrue(all(config.legacy_kid == -1 for config in new))
                configs = (*catalog.CONFIGS_BY_LEGACY_KID.values(), *new)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    for index, config in enumerate(configs):
                        with self.subTest(pipeline=config.pipeline, kid=config.legacy_kid):
                            request = root / "request.json"
                            request.write_text(json.dumps({"pipeline": config.pipeline,
                                                           "compile_params": json.loads(config.to_json())}))
                            output = root / str(index)
                            generated = gen_instances.generate_bpreshuffle_config(request, output)
                            self.assertEqual(generated.to_json(), config.to_json())
                            files = sorted(output.rglob("*.cu"))
                            device = [item for item in files if item.name.endswith(".device.cu")]
                            self.assertGreaterEqual(len(device), 1)
                            self.assertTrue(all("a8w8_mxscale_gemm_bpreshuffle" in item.name for item in device))
                            self.assertFalse((output / "compiled_kids_opus.json").exists())
                            self.assertFalse((output / "opus_gemm_a16w16_kid_dispatch.h").exists())
                            manifest = (output / "opus_gemm_manifest.h").read_text()
                            self.assertNotIn("a16w16", manifest)
                            self.assertNotIn("bmm", manifest)
                            receipt = json.loads((output / "bpreshuffle_config.json").read_text())
                            self.assertEqual(receipt["abi_kid"], config.legacy_kid if config.legacy_kid >= 0 else 1)
                            router = (output / "bpreshuffle_config_dispatch.cu").read_text()
                            self.assertIn(f"kid == {receipt['abi_kid']}", router)
                            self.assertIn("g_aiter_can_throw = true", router)
                            self.assertIn("current HIP device must match tensor device", router)
            finally:
                sys.path.remove(path)


if __name__ == "__main__":
    with _cpu_imports():
        unittest.main()
