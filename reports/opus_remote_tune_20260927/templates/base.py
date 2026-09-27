"""Local numerical, timing and registry helpers for the portable harness."""

from bootstrap import load_tune
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

from machine import GPU_MAP, GPU_INDEX, GPU_UUID, GPU_BUS, GPU_DOMAIN, GPU_DEVICE

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "csrc/opus_gemm/opus_gemm_common.py").is_file())
MODULES = (
    "module_aiter_core",
    "module_deepgemm_opus",
    "module_gemm_a8w8_blockscale_bpreshuffle_tune",
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    "module_gemm_a8w8_blockscale_bpreshuffle_asm",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_shapes(path, default_selection):
    rows = read_csv(path)
    if default_selection:
        rows = [row for row in rows if row.get("best_recorded_opus") in
                {"opus_9010", "opus_9011"}]
    shapes = set()
    for row in rows:
        shape = tuple(int(row[key]) for key in ("M", "N", "K"))
        if min(shape) <= 0 or shape[1] % 16 or shape[2] % 128:
            raise ValueError(f"Invalid MXFP8 B-preshuffle shape: {shape}")
        shapes.add(shape)
    if not shapes or (default_selection and len(shapes) != 17):
        raise ValueError(f"Expected {'17 default' if default_selection else 'nonempty'} shapes; got {len(shapes)}")
    return sorted(shapes, key=lambda shape: (shape[1], shape[2], shape[0]))


@dataclass
class Candidate:
    row: dict
    func: object
    keys: tuple
    extra: tuple
    kwargs: dict

    @property
    def key(self):
        return candidate_identity(self.row)

    def arguments(self, data):
        return (*(data[key] for key in self.keys), *self.extra)


def candidate_identity(row):
    # A registry name need not encode every instance parameter. In particular,
    # distinct CK IDs can share a name, so keep both ID and name. ASM IDs depend
    # on the shape/split enumeration, which is fixed for each compared shape.
    return row["lib"], int(row["kid"]), row["kernelName"], int(row["splitK"])


def candidates_for_shape(tune, tuner, shape):
    info = ("gfx950", 256, *shape)
    tasks = tuner.get_gemm_a8w8_blockscale_tune_task(info, True, 0, True, {})
    tasks += tuner.get_gemm_a8w8_blockscale_cktile_tune_task(
        info, True, 0, True, list(range(1, tune.generic_tune.BLOCK_PER_CU_MAX + 1)), {})
    tasks += tuner.get_gemm_a8w8_blockscale_asm_tune_task(info, True, 0, True, {})
    tasks += tuner.get_gemm_a8w8_blockscale_opus_tune_task(info, 0, True, {})
    result = []
    for task in tasks:
        task_info, _, _, func, args, kwargs, *_ = task
        _, kid, split, kernel_name, lib, preshuffle = task_info
        if not preshuffle:
            raise ValueError("Non-preshuffle task returned by tuner")
        if not kernel_name:
            kernel_name = tuner.getKernelName(kid, lib, True)
        if not kernel_name:
            raise ValueError(f"Missing kernel name: {task_info}")
        row = dict(zip(("M", "N", "K"), shape))
        row.update(name=f"opus_{kid}" if lib == "opus" else f"{lib}_{kid}_split{split}",
                   lib=lib, kid=int(kid), splitK=int(split), kernelName=kernel_name)
        result.append(Candidate(row, func, tuple(args[0]), tuple(args[1:]), kwargs))
    if len({candidate.key for candidate in result}) != len(result):
        raise ValueError(f"Duplicate tuner candidates for {shape}")
    for lib in ("ck", "cktile", "opus"):
        if not any(candidate.row["lib"] == lib for candidate in result):
            raise ValueError(f"No {lib} candidates for {shape}")
    return result


def snapshot(jit_dir):
    source_paths = set()
    for directory in ("csrc/opus_gemm", "csrc/ck_gemm_a8w8_blockscale",
                      "csrc/ck_gemm_a8w8_blockscale_bpreshuffle", "csrc/include",
                      "3rdparty/composable_kernel/include", "3rdparty/composable_kernel/library/include", "3rdparty/ck_helper",
                      "aiter/ops/opus", "aiter/jit/utils"):
        source_paths.update(path for path in (ROOT / directory).rglob("*")
                            if path.is_file() and path.suffix in
                            {".py", ".h", ".hpp", ".cuh", ".cu", ".cpp"}
                            and "build" not in path.parts and "__pycache__" not in path.parts)
    for name in ("aiter/test_common.py", "aiter/jit/core.py", "aiter/jit/optCompilerConfig.json",
                 "aiter/ops/gemm_op_a8w8.py", "aiter/ops/opus/gemm_op_a8w8.py",
                 "aiter/ops/shuffle.py", "aiter/utility/mp_tuner.py", "aiter/utility/base_tuner.py", "aiter/utility/dtypes.py",
                 "csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu", "hsa/codegen.py"):
        source_paths.add(ROOT / name)
    asm_dir = ROOT / "hsa/gfx950/fp8gemm_blockscale"
    asm_csv = asm_dir / "fp8gemm_bf16_blockscale.csv"
    source_paths.add(asm_csv)
    if not asm_csv.is_file():
        raise FileNotFoundError(f"ASM registry is required for a complete sweep: {asm_csv}")
    asm_paths = {asm_dir / row["co_name"] for row in read_csv(asm_csv)
                 if int(row["bpreshuffle"]) == 1}
    if not asm_paths:
        raise ValueError("ASM registry has no B-preshuffle kernels")
    required_modules = [jit_dir / f"{name}.so" for name in MODULES]
    for path in required_modules + list(asm_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Prebuilt artifact required; build outside this harness: {path}")
    source_paths.update(HERE / name for name in ("tune_adapter.py", "bootstrap.py"))
    source_paths.update((ROOT / "csrc/pybind").glob("*a8w8*bpreshuffle*"))
    source_paths.add(ROOT / "csrc/pybind/aiter_core_pybind.cu")
    source_paths = {p for p in source_paths if p.is_file()}
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(source_paths)}
    binary_paths = set(jit_dir.glob("*.so")) | asm_paths
    binaries = {str(path.resolve()): sha256(path) for path in sorted(binary_paths)}
    # Freeze the reference side across sweep/final, while permitting an OPUS
    # optimization between them. The FP32 comparator is explicitly frozen.
    references = {name: digest for name, digest in sources.items()
                  if not name.startswith(("csrc/opus_gemm/", "csrc/include/opus/", "aiter/ops/opus/"))}
    comparator = str((HERE / "tune_adapter.py").relative_to(ROOT))
    references[comparator] = sources[comparator]
    for path, digest in binaries.items():
        path = Path(path)
        if path.name != "module_deepgemm_opus.so":
            references[f"{'jit' if path.parent == jit_dir else 'asm'}/{path.name}"] = digest
    return sources, binaries, references


def check_output(torch, tune, reference, actual):
    error = float(tune.compare_outputs(reference, actual, printLog=False))
    result = dict(error=error, status="passed" if error == 0 else "failed",
                  reason="" if error == 0 else "outside original FP32 accumulation bounds")
    if error:
        expected, magnitude = reference.unbind(0)
        nonfinite = int((~torch.isfinite(actual)).sum().item())
        bound = 1e-4 + 5e-5 * magnitude
        lower, upper = (expected - bound).to(actual.dtype).float(), (expected + bound).to(actual.dtype).float()
        value = actual.float()
        result.update(nonfinite=nonfinite,
                      mismatches=int(((value < lower) | (value > upper) | ~torch.isfinite(value)).sum().item()))
        if nonfinite:
            result["reason"] = "nonfinite output (including unwritten NaN-prefilled elements)"
        else:
            result["max_abs_error"] = float((value - expected).abs().max().item())
    return result


def require_idle(torch, output, label):
    torch.cuda.synchronize()
    time.sleep(0.4)
    recent = []
    for _ in range(20):
        sample = json.loads(subprocess.check_output(
            ["rocm-smi", "-d", str(GPU_INDEX), "--showuse", "--showmemuse", "--json"], text=True))[f"card{GPU_INDEX}"]
        output.sample(dict(time=time.time(), label=label, **sample))
        recent.append(sample)
        window = recent[-3:]
        if (len(window) == 3 and not any(int(item["GPU use (%)"]) for item in window)
                and len({item["GFX Activity"] for item in window}) == 1):
            return
        time.sleep(0.35)
    raise RuntimeError("External GPU activity detected; preserve partial results and stop")
