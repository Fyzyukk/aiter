import json,pathlib,sys,subprocess,os,concurrent.futures,time
root=pathlib.Path.cwd();sys.path.insert(0,str(root/'csrc/opus_gemm'))
from opus_gemm_bpreshuffle_config import *
from gen_instances import opus_gemm_codegen
out=root/'reports/opus_configflow_20261009/device_build'
examples=[construct_config('pin',tile_m=256,tile_n=256,scale_panel=32),construct_config('tiled',tile_m=96,tile_n=128,stages=3),construct_config('register',tile_m=16,tile_n=32,runtime_split_k=True,prefetch=5),construct_config('lds',tile_m=64,tile_n=128,runtime_split_k=True,stages=5),construct_config('large_output',tile_m=192,tile_n=256,b_direct=True,c_chunk_rows=64)]
ref=json.loads((root/'reports/opus_flydsl_all_20261009/main_variants/build_receipt.json').read_text())
flags={}
for b in ref['builds']:
 flags[b['side']]=[]
 for t in b['compile_argv']:
  if t=='-c':break
  if not t.startswith('-I'):flags[b['side']].append(t)
env=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def run(c):
 path=out/c.pipeline;path.mkdir(exist_ok=True)
 inst=kernel_instance_from_config(c);cg=opus_gemm_codegen(str(path));cg.gen_instances({1:inst})
 src=next((path/'instances').glob('*.device.cu'));obj=path/'device.o';log=path/'device.log'
 argv=flags['pin24' if c.pin_agpr else 'main23']+['-mllvm', '--disable-machine-licm', f'-I{root}/csrc/include',f'-I{root}/csrc/opus_gemm/include',f'-I{path}','-c',str(src),'-o',str(obj)]
 started=time.monotonic();p=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True);log.write_text(p.stdout+p.stderr)
 row=dict(pipeline=c.pipeline,compile_params=dict(c.compile_params),legacy_kid=c.legacy_kid,argv=argv,returncode=p.returncode,seconds=time.monotonic()-started,object=str(obj),log=str(log),source=str(src))
 print(json.dumps({k:row[k] for k in ('pipeline','returncode','seconds')}),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as p: rows=list(p.map(run,examples))
(out/'build_receipt.json').write_text(json.dumps(dict(status='all_objects_passed' if all(r['returncode']==0 for r in rows) else 'failed', gpu_queries=0,library_loads=0,kernel_launches=0,builds=rows),indent=2)+'\n')
raise SystemExit(any(x['returncode'] for x in rows))
