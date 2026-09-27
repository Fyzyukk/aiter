"""Keep the archived MXFP8 tuner adapter outside the production source tree."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "csrc/opus_gemm/opus_gemm_common.py").is_file())

def load_tune():
    sys.path[:0] = [str(ROOT), str(ROOT / "csrc/ck_gemm_a8w8_blockscale")]
    name = "csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "tune_adapter.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
