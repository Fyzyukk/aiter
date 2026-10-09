#!/usr/bin/env python3
"""CPU-only build of private baseline and per-wave wait experiment."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LLVM=Path('/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin')
flags=['-x','hip','-D__HIPCC_RTC__','-D__HIP_PLATFORM_AMD__=1','-O3','-std=c++20','--offload-arch=gfx950','--rocm-path=/opt/rocm','-fPIC','-shared','-fno-gpu-rdc','-fgpu-flush-denormals-to-zero','-fno-offload-uniform-block','-fvisibility=hidden','-mllvm','--amdgpu-kernarg-preload-count=32','-mllvm','--amdgpu-mfma-vgpr-form','-mllvm','--lsr-drop-solution=1','-mllvm','-amdgpu-early-inline-all=true','-mllvm','-amdgpu-function-calls=false','-mllvm','-enable-post-misched=0','-mllvm','-verify-machineinstrs']
env=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
rows=[]
for name in ['baseline','candidate']:
 d=HERE/name;cmd=[str(LLVM/'clang++'),*flags,'-I'+str(d/'include'),str(d/'launch.hip'),'-L/opt/rocm/lib','-lamdhip64','-o',str(d/'experiments.so')]
 t=time.time()
 with (d/'build.log').open('w') as log:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 row={'side':name,'command':cmd,'exit_code':r.returncode,'seconds':time.time()-t,'gpu_executed':False}
 if r.returncode==0:row['binary_sha256']=hashlib.sha256((d/'experiments.so').read_bytes()).hexdigest()
 row['source_sha256']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file() and p.suffix in ['.hip','.cuh','.hpp']}
 rows.append(row);(HERE/'build_manifest.json').write_text(json.dumps({'status':'passed' if all(z['exit_code']==0 for z in rows) else 'failed','gpu_executed':False,'builds':rows},indent=2)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ['command','source_sha256']}),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
