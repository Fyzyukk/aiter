#!/usr/bin/env python3
"""Offline Clang23 compile/link only. Built HIP libraries are never loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CLANG=Path('/opt/rocm-llvm23-46fcb339/bin/clang++')
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    options=argparse.ArgumentParser(description=__doc__)
    options.add_argument('--dry-run',action='store_true')
    args=options.parse_args()
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    common=[str(CLANG),'-x','hip','--rocm-path=/opt/rocm','--hip-path=/opt/rocm','--offload-arch=gfx950',
            '-std=c++20','-O3','-fPIC','-fvisibility=hidden','-D__HIP_PLATFORM_AMD__=1','-DOPUS_ENABLE_RUNTIME_QUERY=0',
            '-fgpu-flush-denormals-to-zero','-fno-offload-uniform-block','-fno-gpu-rdc',
            '-mllvm','--amdgpu-kernarg-preload-count=32','-mllvm','--amdgpu-mfma-vgpr-form',
            '-mllvm','--lsr-drop-solution=1','-mllvm','-amdgpu-early-inline-all=true',
            '-mllvm','-amdgpu-function-calls=false','-mllvm','-enable-post-misched=0','-mllvm','-verify-machineinstrs',
            '-I'+str(HERE),'-I'+str(HERE/'frozen'),'-I'+str(HERE/'frozen/gemm_include'),'-I'+str(HERE/'frozen/gemm_include/gfx950')]
    builds=[]
    for ahead in [0,1,2,3]:
        side='baseline' if ahead==0 else f'direct_b_ahead{ahead}'
        folder=HERE/'build'/side
        obj=folder/'launch.o'; lib=folder/'experiments.so'
        compile_argv=common+([] if ahead==0 else [f'-DSMALL_DIRECT_B_AHEAD={ahead}'])+['-c',str(HERE/'launch.hip'),'-o',str(obj)]
        link_argv=[str(CLANG),'-shared',str(obj),'-L/opt/rocm/lib','-lamdhip64','-Wl,--build-id=sha1','-o',str(lib)]
        builds.append(dict(side=side,b_ahead=ahead,object=str(obj),library=str(lib),compile_argv=compile_argv,link_argv=link_argv))
    if args.dry_run:
        print(json.dumps(dict(cpu_only=True,builds=builds),indent=2)); return
    if (HERE/'build').exists() or (HERE/'build_receipt.json').exists(): raise SystemExit('Refusing to overwrite build')
    for row in manifest['frozen_files']:
        assert sha(ROOT/row['original'])==row['sha256']==sha(HERE/row['frozen'])
    for row in manifest['controls']: assert sha(Path(row['path']))==row['sha256']
    receipt=dict(status='building',cpu_only=True,gpu_operations=0,registered=False,
                 numerical_validation='not_run_gpu_stopped',performance_validation='not_run_gpu_stopped',
                 compiler=dict(path=str(CLANG),realpath=str(CLANG.resolve()),sha256=sha(CLANG),
                               version=subprocess.check_output([str(CLANG),'--version'],env=ENV,text=True),
                               resource_dir=subprocess.check_output([str(CLANG),'-print-resource-dir'],env=ENV,text=True).strip()),
                 source_manifest_sha256=sha(HERE/'source_manifest.json'),
                 sources={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in {'.hip','.h','.hpp','.cuh','.py'}},
                 builds=builds,controls=manifest['controls'])
    for build in builds:
        folder=Path(build['object']).parent; folder.mkdir(parents=True)
        for phase in ['compile','link']:
            start=time.monotonic()
            run=subprocess.run(build[phase+'_argv'],cwd=HERE,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            log=folder/(phase+'.log'); log.write_text(run.stdout)
            build.update({phase+'_returncode':run.returncode,phase+'_seconds':time.monotonic()-start,phase+'_log_sha256':sha(log)})
            if run.returncode:
                receipt['status']='failed_'+phase
                (HERE/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print(run.stdout); raise SystemExit(run.returncode)
        build['object_sha256']=sha(Path(build['object'])); build['library_sha256']=sha(Path(build['library']))
        print(json.dumps(dict(side=build['side'],offline_compile_link='passed',library_sha256=build['library_sha256'])),flush=True)
    assert all(sha(ROOT/r['original'])==r['sha256'] for r in manifest['frozen_files'])
    assert all(sha(Path(r['path']))==r['sha256'] for r in manifest['controls'])
    receipt.update(status='offline_build_passed_unvalidated_gpu_stopped',production_sources_unchanged=True,controls_unchanged=True)
    (HERE/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__': main()
