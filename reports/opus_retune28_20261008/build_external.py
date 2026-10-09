from pathlib import Path
import os,sys,time,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
os.environ.update(AITER_AOT_IMPORT='1',AITER_JIT_DIR=str(OUT/'jit'),GPU_ARCHS='gfx950',CU_NUM='256',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HIP_CLANG_PATH='/opt/rocm/llvm/bin',MAX_JOBS='12')
from aiter.jit import core
names=['module_gemm_a8w8_blockscale_bpreshuffle_tune','module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune','module_gemm_a8w8_blockscale_bpreshuffle_asm']
records=[]
for name in names:
 args=core.get_args_of_build(name)
 start=time.monotonic()
 core.build_module(name,args['srcs'],args['flags_extra_cc'],args['flags_extra_hip'],args['blob_gen_cmd'],args['extra_include'],args['extra_ldflags'],False,args['is_python_module'],args['is_standalone'],args['torch_exclude'],args['third_party'],flags_extra_hip_per_source=args.get('flags_extra_hip_per_source',{}))
 path=OUT/'jit'/f'{name}.so'
 records.append({'name':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'build_args':args})
 (OUT/'external_build.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
 print(json.dumps(records[-1],default=str),flush=True)
