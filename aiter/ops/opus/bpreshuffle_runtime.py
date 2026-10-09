# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""On-demand, configuration-specific OPUS MXFP8 extension modules.

This module is safe to load while examining scalar configurations: tensor,
HIP, and JIT imports happen only when a validated launch or explicit prepare
requests a compiled module. Each complete configuration has its own module
name, including source and compiler identity, so a process never needs to
replace an already-loaded legacy subset extension.
"""

from dataclasses import dataclass
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading


_BASE_MODULE = "module_deepgemm_opus"
_PREFIX = "opus_gemm_a8w8_blockscale_bpreshuffle"
_DEFAULT_PIN_COMPILER = "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin"
_ROOT = Path(__file__).resolve().parents[3]
_ENV_KEYS = (
    "OPUS_HIP_CLANG_PATH", "OPUS_HIP_RESOURCE_DIR",
    "OPUS_BASELINE_HIP_CLANG_PATH", "OPUS_BASELINE_HIP_RESOURCE_DIR",
    "HIP_CLANG_PATH", "AITER_HIP_RESOURCE_DIR", "ROCM_PATH", "ROCM_HOME",
    "GPU_ARCHS", "CXX", "CC", "PATH", "AITER_META_DIR", "AITER_JIT_DIR",
    "AITER_DISABLE_KERNARG_PRELOAD", "AITER_FP4x2", "AITER_ASM_DEBUG",
    "ENABLE_ROPE_POSITIONS_INT32", "AITER_REBUILD",
)
_prepared = {}
_compiler_identities = {}
_lock = threading.RLock()


def _core():
    return importlib.import_module("aiter.jit.core")


def _canonical_config(config):
    catalog = importlib.import_module("csrc.opus_gemm.opus_gemm_bpreshuffle_config")
    return catalog.construct_config(config.pipeline, config=config)


def _environment_key():
    return tuple((key, os.environ.get(key)) for key in _ENV_KEYS)


@functools.lru_cache(maxsize=1024)
def _config_key(config):
    return config.pipeline, config.to_json()


def _file_identity(path):
    """Hash contents on explicit prepare, including rapid same-size edits."""
    path = Path(path).resolve()
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def _source_identity(core):
    """Cover generation, producers, tensor ABI, and the JIT implementation."""
    csrc = Path(core.AITER_CSRC_DIR)
    paths = {Path(__file__).resolve()}
    paths.update(path for path in (csrc / "opus_gemm").glob("*")
                 if path.is_file() and path.suffix in {".py", ".json"})
    for directory in (csrc / "opus_gemm/codegen", csrc / "opus_gemm/include", csrc / "include"):
        paths.update(path for path in directory.rglob("*")
                     if path.is_file() and path.suffix in {".py", ".h", ".hpp", ".cuh", ".json"})
    jit = _ROOT / "aiter/jit"
    paths.update((jit / "core.py", jit / "optCompilerConfig.json"))
    paths.update(jit / "utils" / filename for filename in (
        "cpp_extension.py", "_cpp_extension_versioner.py", "jit_cache.py",
        "file_baton.py", "torch_guard.py", "opus_compiler.py", "opus_hip_compile.py",
        "chip_info.py",
    ))
    return [_file_identity(path) for path in sorted(paths)]


def _hip_header_identity(core):
    """Record the HIP headers used by the installed ROCm toolchain."""
    roots = {Path(value).resolve() for value in (
        os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"),
    ) if value}
    roots.add(Path(core.executable_path("hipcc")).resolve().parents[1])
    extension = sys.modules.get("cpp_extension")
    if extension is not None and getattr(extension, "ROCM_HOME", None):
        roots.add(Path(extension.ROCM_HOME).resolve())
    paths = set()
    for root in roots:
        for directory in (root / "include/hip", root / "hip/include/hip"):
            if directory.is_dir():
                paths.update(path for path in directory.rglob("*") if path.is_file())
    return [_file_identity(path) for path in sorted(paths)]


def _compiler_identity(executable):
    path = Path(executable).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OPUS compiler does not exist: {path}")
    stat = path.stat()
    token = str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    identity = _compiler_identities.get(token)
    if identity is None:
        version = subprocess.check_output([str(path), "--version"], text=True).strip()
        identity = {"path": str(path), "file": list(token[1:]), "version": version}
        _compiler_identities[token] = identity
    return identity


def _clang_in(directory, variable):
    compiler = Path(directory) / "clang++"
    if not compiler.is_file():
        raise FileNotFoundError(f"Invalid {variable}: expected {compiler}")
    return compiler


def _resource_directory(compiler_identity, requested, core):
    resource = requested
    match = re.search(r"clang version (\d+)", compiler_identity["version"])
    if not resource and match and int(match[1]) >= 24 and core.get_hip_version().startswith("7.0."):
        rocm = (os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
                or "/opt/rocm")
        resource = next((str(path) for path in (
            Path(rocm) / "lib/llvm/lib/clang/20", Path(rocm) / "llvm/lib/clang/20",
        ) if (path / "include").is_dir()), None)
        if resource is None:
            raise RuntimeError("LLVM 24 with ROCm 7.0 requires OPUS_HIP_RESOURCE_DIR")
    if resource and not (Path(resource) / "include").is_dir():
        raise FileNotFoundError(f"Invalid OPUS HIP resource directory: {resource}")
    return str(Path(resource).resolve()) if resource else None


def _compiler_plan(config, core):
    """Use explicit argv overrides without mutating shared compiler settings."""
    catalog = importlib.import_module("csrc.opus_gemm.opus_gemm_bpreshuffle_config")
    dependencies = catalog.get_config_dependencies(config)
    needs_pin = any(item.pin_agpr for item in (config, *dependencies))
    pin_directory = os.environ.get("OPUS_HIP_CLANG_PATH")
    baseline_directory = os.environ.get("OPUS_BASELINE_HIP_CLANG_PATH")
    if needs_pin and not pin_directory:
        pin_directory = _DEFAULT_PIN_COMPILER
    # An explicit OPUS compiler also applies to unpinned producers unless the
    # user selected a separate baseline compiler, matching the tuner policy.
    primary_directory = baseline_directory or pin_directory
    primary = (_clang_in(primary_directory, "OPUS_BASELINE_HIP_CLANG_PATH" if baseline_directory
                         else "OPUS_HIP_CLANG_PATH") if primary_directory else None)
    pin = _clang_in(pin_directory, "OPUS_HIP_CLANG_PATH") if needs_pin else None
    identities = {}
    resources = {}
    commands = {}
    launcher = str(_ROOT / "aiter/jit/utils/opus_hip_compile.py")

    def command(compiler, requested_resource, label):
        identity = _compiler_identity(compiler)
        resource = _resource_directory(identity, requested_resource, core)
        identities[label] = identity
        resource_headers = resource
        if not resource_headers:
            resource_headers = subprocess.check_output(
                [identity["path"], "-print-resource-dir"], text=True).strip()
        resources[label] = ({"path": resource_headers,
                             "files": [_file_identity(path) for path in sorted(Path(resource_headers, "include").rglob("*"))
                                       if path.is_file()]} if resource_headers else None)
        argv = [sys.executable, launcher, "--compiler", identity["path"]]
        if resource:
            argv += ["--resource-dir", resource]
        # core probes flags with the process's default hipcc. Preserve OPUS's
        # required register form when this explicitly chosen compiler accepts
        # a flag that those shared probes may have filtered from global flags.
        return argv + ["--", "-mllvm", "--amdgpu-mfma-vgpr-form"]

    if pin is not None:
        pin_resource = os.environ.get("OPUS_HIP_RESOURCE_DIR") or os.environ.get("AITER_HIP_RESOURCE_DIR")
        commands["*.device.cu"] = command(pin, pin_resource, "pin")
    if primary is not None:
        primary_resource = (os.environ.get("OPUS_BASELINE_HIP_RESOURCE_DIR") if baseline_directory
                            else os.environ.get("OPUS_HIP_RESOURCE_DIR") or os.environ.get("AITER_HIP_RESOURCE_DIR"))
        commands["*.cu"] = command(primary, primary_resource, "primary")
    # The default hipcc still performs flag probes; CXX links the extension.
    identities["hipcc"] = _compiler_identity(core.executable_path("hipcc"))
    cxx = os.environ.get("CXX") or ("cl" if sys.platform == "win32" else "c++")
    resolved_cxx = shutil.which(cxx)
    if resolved_cxx:
        identities["cxx"] = _compiler_identity(resolved_cxx)
    return commands, {"compilers": identities, "resources": resources,
                      "hip_version": core.get_hip_version()}


def _write_request(path, config):
    payload = json.dumps({"pipeline": config.pipeline,
                          "compile_params": json.loads(config.to_json())},
                         sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == payload:
        return
    # Requests are immutable for their module key. Atomic installation lets
    # another process safely read the same request before taking the build lock.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".config-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_arguments(config, core):
    args = dict(core.get_args_of_build(
        _BASE_MODULE, overrides={"hip_compiler_commands_per_source": {}},
    ))
    commands, compiler_identity = _compiler_plan(config, core)
    args.update(srcs=[], hip_clang_path=None, hip_compiler_commands_per_source=commands)
    # The aggregate's old source glob predates these generated symbol names.
    # This module contains only B-preshuffle producers, so its device TUs all
    # use the producer's RTC and LICM policy directly.
    args["flags_extra_hip_per_source"] = {
        "*.device.cu": ["-D__HIPCC_RTC__", "-mllvm --disable-machine-licm"],
    }
    # OPUS producers and the tensor ABI use the existing lightweight shim;
    # there is no CK producer dependency in a configuration-specific module.
    for key in ("flags_extra_cc", "flags_extra_hip"):
        args[key] = [flag for flag in args[key] if "ENABLE_CK=" not in flag] + ["-DENABLE_CK=0"]
    args["third_party"] = []
    identity = {
        "schema": 1, "pipeline": config.pipeline,
        "compile_params": json.loads(config.to_json()),
        "sources": _source_identity(core), "hip_headers": _hip_header_identity(core),
        "compiler": compiler_identity,
        "build": {key: value for key, value in args.items() if key not in {"md_name", "blob_gen_cmd"}},
        "environment": _environment_key(),
        "python": {"version": list(sys.version_info[:3]), "executable": sys.executable},
    }
    try:
        identity["pybind11"] = importlib.metadata.version("pybind11")
    except importlib.metadata.PackageNotFoundError:
        identity["pybind11"] = None
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args["md_name"] = f"module_opus_bpreshuffle_{digest}"
    # build_module can clear its own op directory under AITER_REBUILD=1.
    # Keep the immutable generation request outside that disposable tree.
    request = Path(core.bd_dir) / "opus_bpreshuffle_requests" / f"{args['md_name']}.json"
    _write_request(request, config)
    generator = str(Path(core.AITER_CSRC_DIR) / "opus_gemm/gen_instances.py")
    # core's transactional staging directory is deterministic. Quote the
    # complete path before its format/shlex pass so spaces and literal braces
    # in AITER_JIT_DIR do not turn into extra CLI arguments.
    quote = lambda value: shlex.quote(str(value)).replace("{", "{{").replace("}", "}}")
    staging = Path(core.bd_dir) / args["md_name"] / "blob.staging"
    args["blob_gen_cmd"] = (f"{quote(generator)} --bpreshuffle_config {quote(request)} "
                            f"--working_path {quote(staging)}")
    return args


def _build_module(core, args):
    core.build_module(
        args["md_name"], args["srcs"], args["flags_extra_cc"], args["flags_extra_hip"],
        args["blob_gen_cmd"], args["extra_include"], args["extra_ldflags"],
        args["verbose"], args["is_python_module"], args["is_standalone"],
        args["torch_exclude"], args["third_party"], args.get("hipify", False),
        flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
        hip_compiler_commands_per_source=args.get("hip_compiler_commands_per_source", {}),
    )


def _make_launchers(core, args):
    # Eager Tensor annotations preserve the raw custom op's tensor dispatch
    # identity and avoid introducing a dummy CUDA tensor at every launch.
    from torch import Tensor

    def direct(XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor,
               Y: Tensor, kid: int) -> None:
        pass

    def workspace(XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor,
                  Y: Tensor, kid: int, workspace: Tensor) -> None:
        pass

    def runtime(XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor,
                Y: Tensor, kid: int, workspace: Tensor | None, split_k: int) -> None:
        pass

    launchers = {}
    for kind, stub in (("direct", direct), ("workspace", workspace), ("runtime", runtime)):
        suffix = "launch" if kind == "direct" else f"{kind}_launch"
        stub.__name__ = f"_{args['md_name']}_{suffix}"
        launchers[kind] = core.compile_ops(
            _BASE_MODULE, fc_name=f"{_PREFIX}_{suffix}",
            gen_func=lambda *_args, **_kwargs: dict(args),
            gen_fake=lambda *_args, **_kwargs: None, develop=True,
        )(stub)
    return launchers


@dataclass(frozen=True)
class PreparedConfig:
    config: object
    module_name: str
    abi_kid: int
    launchers: dict

    def launch(self, XQ, WQ, x_scale, w_scale, Y, workspace=None, split_k=0):
        args = XQ, WQ, x_scale, w_scale, Y, self.abi_kid
        if self.config.runtime_split_k:
            self.launchers["runtime"](*args, workspace, split_k)
        elif workspace is not None:
            self.launchers["workspace"](*args, workspace)
        else:
            self.launchers["direct"](*args)
        return Y


def ensure_config(config):
    """Compile/load an exact configuration without launching a GPU kernel.

    Rechecks build identity, so an explicit prepare after source/compiler
    changes obtains a new module even in the same Python process. Repeated
    launches use the prepared object without filesystem or compiler probes.
    """
    config = _canonical_config(config)
    with _lock:
        core = _core()
        args = _build_arguments(config, core)
        key = _config_key(config), _environment_key()
        cached = _prepared.get(key)
        if cached is not None and cached.module_name == args["md_name"]:
            return cached
        try:
            core.get_module(args["md_name"])
        except ModuleNotFoundError:
            _build_module(core, args)
            core.get_module(args["md_name"])
        prepared = PreparedConfig(config, args["md_name"],
                                  config.legacy_kid if config.legacy_kid >= 0 else 1,
                                  _make_launchers(core, args))
        _prepared[key] = prepared
        return prepared


def launch_config(config, XQ, WQ, x_scale, w_scale, Y, workspace=None, split_k=0):
    """Launch after the public adapter has validated tensors and workspace."""
    key = _config_key(config), _environment_key()
    prepared = _prepared.get(key)
    if prepared is None:
        prepared = ensure_config(config)
    return prepared.launch(XQ, WQ, x_scale, w_scale, Y, workspace, split_k)


__all__ = ["PreparedConfig", "ensure_config", "launch_config"]
