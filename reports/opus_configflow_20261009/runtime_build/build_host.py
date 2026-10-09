"""Compile generated scalar OPUS host/pybind TUs without importing GPU packages."""
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=Path('/opt/rocm-llvm23-46fcb339/bin/clang++')
PIN=Path('/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++')
PYBIND=Path('/opt/venv/lib/python3.10/site-packages/pybind11/include')

def build(pipeline):
    directory=OUT/pipeline
    compiler=PIN if pipeline=='pin' else BASE
    common=[str(compiler),'-x','hip','--offload-host-only','--rocm-path=/opt/rocm','--hip-path=/opt/rocm',
            '-std=c++20','-O2','-fPIC','-DENABLE_CK=0','-D_GLIBCXX_USE_CXX11_ABI=1',
            '-DAITER_EXTENSION_NAME=module_offline_'+pipeline,'-DTORCH_EXTENSION_NAME=module_offline_'+pipeline,
            '-I'+str(ROOT/'csrc/include'),'-I'+str(ROOT/'csrc/opus_gemm/include'),'-I'+str(directory),
            '-I'+str(PYBIND),'-I/usr/include/python3.10']
    if pipeline=='pin':common+=['-resource-dir=/opt/rocm/lib/llvm/lib/clang/20']
    commands=[]
    for filename in ('instances/all_instances_host_gfx950.cu','bpreshuffle_config_dispatch.cu','bpreshuffle_config_pybind.cu'):
        obj=directory/(Path(filename).stem+'.o')
        argv=common+[str(directory/filename),'-c','-o',str(obj)]
        commands.append(argv)
        result=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (directory/(Path(filename).stem+'.compile.log')).write_text(result.stdout)
        if result.returncode:
            (directory/'host_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            return {'pipeline':pipeline,'returncode':result.returncode,'failed':filename,'log':str(directory/(Path(filename).stem+'.compile.log'))}
    (directory/'host_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    return {'pipeline':pipeline,'returncode':0}

with ThreadPoolExecutor(max_workers=3) as executor:results=list(executor.map(build,('pin','tiled','register','lds','large_output')))
(OUT/'host_result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
sys.exit(any(result['returncode'] for result in results))
