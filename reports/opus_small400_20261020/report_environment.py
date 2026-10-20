"""Use recorded compiler defaults with paths relative to this checkout."""
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[1]

def configure():
    recorded = json.loads((P / 'environment.json').read_text())
    for key, value in recorded.items():
        os.environ.setdefault(key, value)
    os.environ['PYTHONPATH'] = str(ROOT)
    os.environ['OPUS_GEN_CO_DIR'] = str(ROOT / 'csrc/opus_gemm/gen_co')
    os.environ['AITER_JIT_DIR'] = str(P / 'jit')
    os.environ['TMPDIR'] = str(P / 'tmp')
    os.environ['FLYDSL_RUNTIME_CACHE_DIR'] = str(P / 'flydsl_cache')
    os.environ['AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE'] = str(
        ROOT / 'aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv')
    if 'CK_DIR' not in os.environ or not Path(os.environ['CK_DIR']).is_dir():
        os.environ['CK_DIR'] = str(ROOT / '3rdparty/composable_kernel')
    (P / 'tmp').mkdir(exist_ok=True)
