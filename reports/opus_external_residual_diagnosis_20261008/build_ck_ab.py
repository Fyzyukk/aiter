#!/usr/bin/env python3
"""Isolate CK ID17 dependency/compiler changes without changing the tune build."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, shlex, subprocess, time
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
BASE=ROOT/'reports/opus_clang23_mixed_retune_20261008'
BUILD=BASE/'jit/build/module_gemm_a8w8_blockscale_bpreshuffle_tune/build'
CK=ROOT/'3rdparty/composable_kernel'
NAME='module_gemm_a8w8_blockscale_bpreshuffle_tune'
STEM='a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1_dBF16_eBF16'
COMMANDS=[shlex.split(x) for x in (BASE/f'{NAME}_commands.txt').read_text().splitlines() if x.strip()]
INSTANCE=next(x for x in COMMANDS if any(y.endswith('/'+STEM+'.cpp') for y in x))
PYBIND=next(x for x in COMMANDS if any(y.endswith('_tune_pybind.cu') for y in x))
LINK=COMMANDS[-1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd,cwd,env,log):
    log.write('$ '+shlex.join(cmd)+'\n');log.flush()
    subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=log,check=True)
def build(item):
    module,revision,compiler=item; begin=time.time(); dest=OUT/module;dest.mkdir(exist_ok=True)
    extra=['-DCK_USE_LLVM_BUILTIN_BF16=0'] if module.endswith('_bf16off') else []
    source=CK
    if revision!='current':
        source=OUT/('ck_'+revision[:12]); source.mkdir(exist_ok=True)
        archive=OUT/(revision[:12]+'.tar')
        if not archive.exists():
            with archive.open('wb') as f:subprocess.run(['git','-C',str(CK),'archive',revision],stdout=f,check=True)
            subprocess.run(['tar','xf',str(archive),'-C',str(source)],check=True)
    env={**os.environ,'HIP_CLANG_PATH':compiler,'GPU_ARCHS':'gfx950'}
    commands=[]
    with (dest/'build.log').open('w') as log:
        cmd=[* [x.replace(str(CK),str(source)) for x in INSTANCE], *extra]
        cmd[cmd.index('-o')+1]=str(dest/(STEM+'.cuda.o'));commands.append(cmd)
        if not (dest/(STEM+'.cuda.o')).exists():run(cmd,dest,env,log)
        # The host entry point stays Clang23 for all variants.
        # Old CK represents BF16 as uint16_t, current CK as __bf16. Compile a
        # fixed-ID host entry against its own CK headers to preserve the ABI.
        symbol=STEM.removesuffix('_dBF16_eBF16')
        binding=dest/'fixed_id17_pybind.cu'
        binding.write_text('#include "gemm_a8w8_blockscale_bpreshuffle_common.cuh"\n#include "gemm_a8w8_blockscale_bpreshuffle_manifest.h"\nPYBIND11_MODULE(AITER_EXTENSION_NAME,m) { m.def("gemm_a8w8_blockscale_bpreshuffle_tune", [](torch::Tensor& a,torch::Tensor& b,torch::Tensor& sa,torch::Tensor& sb,torch::Tensor& out,int kid,int split) { TORCH_CHECK(kid==17 && split==0,"fixed ID17 only"); return '+symbol+'<F32,B16>(a,b,sa,sb,out); }); }\n')
        cmd=[* [x.replace(NAME,module).replace(str(CK),str(source)) if x.startswith('-DAITER_EXTENSION_NAME=') or x.startswith('-DTORCH_EXTENSION_NAME=') else x.replace(str(CK),str(source)) for x in PYBIND], *extra]
        cmd[cmd.index('-c')+1]=str(binding)
        cmd[cmd.index('-o')+1]=str(dest/'pybind.cuda.o');commands.append(cmd)
        run(cmd,dest,{**env,'HIP_CLANG_PATH':'/opt/rocm-llvm23-46fcb339/bin'},log)
        objects=[str(dest/(STEM+'.cuda.o')),str(dest/'pybind.cuda.o')]
        cmd=[LINK[0],*objects,*LINK[2:]];cmd[cmd.index('-o')+1]=str(OUT/(module+'.so'))
        commands.append(cmd);run(cmd,dest,env,log)
    (dest/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    result={'module':module,'ck_revision':subprocess.check_output(['git','-C',str(CK),'rev-parse','HEAD'],text=True).strip() if revision=='current' else revision,'device_compiler':compiler,'host_compiler':'/opt/rocm-llvm23-46fcb339/bin','extra_flags':extra,'path':str(OUT/(module+'.so')),'sha256':sha(OUT/(module+'.so')),'seconds':time.time()-begin,'instance':STEM,'host_entry':'direct fixed ID17, same generated template and common wrapper; original tune entry included as control'}
    print(json.dumps(result),flush=True);return result
def main():
    variants=[('module_ck_old3238','33b62ed0878369db891a85d743576605e62b3d1c','/opt/rocm-llvm23-46fcb339/bin'),('module_ck_old3383','83566edb0fded5e1c618c2c19110adbb74532762','/opt/rocm-llvm23-46fcb339/bin'),('module_ck_current20','current','/opt/rocm/llvm/bin'),('module_ck_current23','current','/opt/rocm-llvm23-46fcb339/bin')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(build,variants))
    (OUT/'ck_build_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
