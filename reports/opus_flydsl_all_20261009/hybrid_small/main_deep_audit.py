#!/usr/bin/env python3
"""Independent CPU-only main geometry/pin scale layout audit; no HIP runtime load."""
import argparse,difflib,hashlib,json,os,re,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;MAIN=H.parent/'main_variants'
LLVM=Path('/opt/rocm-llvm23-46fcb339/bin')
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-name',default='main_deep_cpu_v1');a=p.parse_args()
 assert Path(a.output_name).name==a.output_name
 out=H/a.output_name
 if out.exists():raise SystemExit('Refusing overwrite')
 out.mkdir()
 sources={}
 for name in ['candidate/traits.cuh','candidate/pin.cuh','candidate/pad_pin.cuh','candidate/geometry.cuh','contract.h','launch.hip','launch_pin.hip']:
  src=MAIN/name;target=out/'review_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);sources[name]=sha(src)
 shutil.copyfile(MAIN/'candidate/traits.cuh',out/'main_traits.cuh')
 host=out/'host_include/opus';host.mkdir(parents=True)
 original=(MAIN/'frozen/opus/opus.hpp').read_text()
 begin=original.index('template<typename FDim, typename Target, index_t I0');end=original.index('#define OPUS_KP_',begin)
 adapted=original[:begin]+original[begin:end].replace('OPUS_D ','OPUS_H_D ')+original[end:]
 (host/'opus.hpp').write_text(adapted)
 (out/'host_attributes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True))))
 for name in ['dtypes.hpp','hip_minimal.hpp']:shutil.copyfile(MAIN/'frozen/opus'/name,host/name)
 source=(MAIN/'frozen/gemm_include/gfx950/opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh').read_text()
 begin=source.index('template<class T>\n__device__ inline auto make_layout_ga_scale')
 helpers=source[begin:].replace('__device__','__host__')
 (out/'helpers.h').write_text('#pragma once\n#include <opus/hip_minimal.hpp>\n#include <opus/opus.hpp>\nusing opus::operator""_I;\nnamespace checked {\n'+helpers+'}\n')
 obj=out/'check.o';binary=out/'check'
 compile_argv=[str(LLVM/'clang++'),'-x','hip','--offload-host-only','--rocm-path=/opt/rocm','--hip-path=/opt/rocm',
               '-std=c++20','-O2','-D__HIPCC_RTC__=1','-I'+str(out),'-I'+str(out/'host_include'),
               '-I'+str(MAIN/'frozen'),'-I'+str(MAIN/'frozen/gemm_include/gfx950'),'-c',str(H/'main_deep_check.cpp'),'-o',str(obj)]
 done=subprocess.run(compile_argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'compile.log').write_text(done.stdout)
 if done.returncode:raise RuntimeError(done.stdout)
 link_argv=[str(LLVM/'clang++'),str(obj),'-o',str(binary)]
 done=subprocess.run(link_argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'link.log').write_text(done.stdout)
 if done.returncode:raise RuntimeError(done.stdout)
 needed=subprocess.check_output([str(LLVM/'llvm-readelf'),'-d',str(binary)],env=ENV,text=True);(out/'dependencies.txt').write_text(needed)
 assert 'amdhip' not in needed.lower() and 'hsa-runtime' not in needed.lower()
 done=subprocess.run([str(binary)],cwd=out,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/'check.log').write_text(done.stdout)
 receipt=dict(status='passed' if done.returncode==0 else 'host_assertion_failed',cpu_only=True,gpu_operations=0,
              no_hip_hsa_dependencies=True,compile_argv=compile_argv,link_argv=link_argv,host_returncode=done.returncode,
              input_main_sha256=sources,host_source_sha256=sha(H/'main_deep_check.cpp'),object_sha256=sha(obj),binary_sha256=sha(binary),
              helper_source_sha256=sha(out/'helpers.h'),compiler_sha256=sha(LLVM/'clang++'),stdout=done.stdout,
              result=json.loads(done.stdout) if done.returncode==0 else None)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2));assert done.returncode==0
if __name__=='__main__':main()
