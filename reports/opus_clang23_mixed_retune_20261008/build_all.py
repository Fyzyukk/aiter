from pathlib import Path
import os,sys,time,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
BASE='/opt/rocm-llvm23-46fcb339/bin'
PIN='/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin'
os.environ.update(AITER_AOT_IMPORT='1',AITER_JIT_DIR=str(OUT/'jit'),GPU_ARCHS='gfx950',CU_NUM='256',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HIP_CLANG_PATH=BASE,OPUS_BASELINE_HIP_CLANG_PATH=BASE,OPUS_HIP_CLANG_PATH=PIN,OPUS_HIP_RESOURCE_DIR='/opt/rocm/lib/llvm/lib/clang/20',MAX_JOBS='32')
os.environ.pop('AITER_HIP_RESOURCE_DIR',None)
from aiter.jit import core
# CPU-only build: runtime dtype selection normally probes rocminfo despite hidden GPUs.
# The target is explicitly gfx950 and this process never executes tensor kernels.
from aiter.jit.utils import chip_info
chip_info._detect_native=lambda: ['gfx950']
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
records=[]
compilers={}
for tag,directory in [('baseline',BASE),('pin_agpr',PIN)]:
 p=Path(directory)/'clang++'
 compilers[tag]={'path':str(p),'version':subprocess.check_output([str(p),'--version'],text=True),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/'compiler_manifest.json').write_text(json.dumps({'compilers':compilers,'pin_kids':[9000,9001,9010,9011],'other_opus_kids':'24 public candidates plus support/device/host TUs','baseline_backends':['ck','cktile','asm host wrapper'],'asm_device':'existing precompiled .co','cpu_build_runtime_probe':'explicit gfx950 target; no GPU initialization'},indent=2)+'\n')
start=time.monotonic()
tune._ensure_kids_compiled(tune.A8W8_BPRESHUFFLE_TUNING_KIDS)
p=OUT/'jit/module_deepgemm_opus.so'
records.append({'name':'module_deepgemm_opus','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seconds':time.monotonic()-start})
(OUT/'build_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
for name in ['module_aiter_core','module_gemm_a8w8_blockscale_bpreshuffle_tune','module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune','module_gemm_a8w8_blockscale_bpreshuffle_asm']:
 args=core.get_args_of_build(name)
 start=time.monotonic()
 core.build_module(name,args['srcs'],args['flags_extra_cc'],args['flags_extra_hip'],args['blob_gen_cmd'],args['extra_include'],args['extra_ldflags'],False,args['is_python_module'],args['is_standalone'],args['torch_exclude'],args['third_party'],flags_extra_hip_per_source=args.get('flags_extra_hip_per_source',{}),hip_compiler_commands_per_source=args.get('hip_compiler_commands_per_source',{}))
 p=OUT/'jit'/f'{name}.so'
 records.append({'name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'build_args':args})
 (OUT/'build_manifest.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
 print(json.dumps(records[-1],default=str),flush=True)
print('BUILD_COMPLETED',flush=True)
