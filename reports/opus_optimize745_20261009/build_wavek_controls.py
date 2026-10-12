#!/usr/bin/env python3
"""Prebuild exact public controls for local-wave global-fusion experiments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    os.environ.update(plan['env'])
    os.environ.update(HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
    lock=(HERE/'cache_build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    from aiter.ops.opus import bpreshuffle_runtime as runtime
    from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config
    import torch
    core=runtime._core()
    build_path=HERE/'fused_wavek/build.json';build=json.loads(build_path.read_text())
    assert build['returncode']==0
    destination=HERE/'fused_wavek/controls.json'
    assert not destination.exists()
    rows=[]
    for row in build['configurations']:
        started=time.monotonic();config=construct_config(row['pipeline'],config=row['compile_params'])
        args=runtime._build_arguments(config,core);assert args['is_python_module']
        artifact=Path(plan['env']['AITER_JIT_DIR'])/(args['md_name']+'.so')
        if not artifact.exists():runtime._build_module(core,args)
        assert artifact.is_file() and not torch.cuda.is_initialized()
        for fd in Path('/proc/self/fd').iterdir():
            try:target=fd.resolve()
            except FileNotFoundError:continue
            assert str(target)!='/dev/kfd'
        rows.append({**row,'path':str(artifact),'module':args['md_name'],
                     'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'seconds':time.monotonic()-started})
        report={'completed':len(rows),'total':len(build['configurations']),'rows':rows,
                'kernel_launches':0,'config_library_loads':0,'gpu_contexts':0,
                'build_sha256':hashlib.sha256(build_path.read_bytes()).hexdigest()}
        temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(destination)
        print(row['id'],args['md_name'],flush=True)
    lock.close()


if __name__=='__main__':
    main()
