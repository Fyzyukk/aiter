#!/usr/bin/env python3
"""Offline-only builds; no HIP libraries are loaded or GPU calls made."""
import hashlib,json,os,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
CLANG23='/opt/rocm-llvm23-46fcb339/bin/clang++'
CLANG24='/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if (H/'build').exists():raise SystemExit('Refusing existing offline build')
 m=json.loads((H/'source_manifest.json').read_text())
 assert all(sha(R/x['original'])==sha(H/x['frozen'])==x['sha256'] for x in m['frozen_files'])
 receipt={'status':'building','gpu_operations':0,'registered':False,'numerical_validation':'not_run_gpu_stopped','performance_validation':'not_run_gpu_stopped','sources':{str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and p.suffix in ('.hip','.h','.cuh','.hpp')},'builds':[]}
 for label,compiler,source in [('main23',CLANG23,'launch.hip'),('pin24',CLANG24,'launch_pin.hip')]:
  folder=H/'build'/label;folder.mkdir(parents=True)
  common=[compiler,'-x','hip','--rocm-path=/opt/rocm','--hip-path=/opt/rocm','--offload-arch=gfx950','-std=c++20','-O3','-fPIC','-fvisibility=hidden','-D__HIP_PLATFORM_AMD__=1','-DOPUS_ENABLE_RUNTIME_QUERY=0','-fgpu-flush-denormals-to-zero','-fno-offload-uniform-block','-fno-gpu-rdc','-mllvm','--amdgpu-kernarg-preload-count=32','-mllvm','--amdgpu-mfma-vgpr-form','-mllvm','--lsr-drop-solution=1','-mllvm','-amdgpu-early-inline-all=true','-mllvm','-amdgpu-function-calls=false','-mllvm','-enable-post-misched=0','-mllvm','-verify-machineinstrs','-I'+str(H),'-I'+str(H/'candidate'),'-I'+str(H/'frozen'),'-I'+str(H/'frozen/gemm_include'),'-I'+str(H/'frozen/gemm_include/gfx950')]
  if label=='pin24':common+=['-resource-dir=/opt/rocm/lib/llvm/lib/clang/20']
  obj=folder/'launch.o';lib=folder/'experiments.so'
  compile=common+['-c',str(H/source),'-o',str(obj)]
  link=[compiler,'-shared',str(obj),'-L/opt/rocm/lib','-lamdhip64','-Wl,--build-id=sha1','-o',str(lib)]
  entry={'side':label,'compile_argv':compile,'link_argv':link,'compiler_sha256':sha(compiler),'compiler_version':subprocess.check_output([compiler,'--version'],env=ENV,text=True),'object':str(obj),'library':str(lib)}
  receipt['builds'].append(entry)
  for name,argv in [('compile',compile),('link',link)]:
   start=time.monotonic();done=subprocess.run(argv,cwd=H,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   (folder/(name+'.log')).write_text(done.stdout)
   entry[name+'_returncode']=done.returncode;entry[name+'_seconds']=time.monotonic()-start
   entry[name+'_log_sha256']=sha(folder/(name+'.log'))
   if done.returncode:
    receipt['status']='failed_'+label+'_'+name;(H/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(done.stdout[-8000:]);raise SystemExit(done.returncode)
  entry['object_sha256']=sha(obj);entry['library_sha256']=sha(lib)
  print(label,'compiled and linked',flush=True)
 assert all(sha(R/p)==d for p,d in m['formal_inputs'].items())
 receipt['status']='offline_build_passed_gpu_stopped'
 (H/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
