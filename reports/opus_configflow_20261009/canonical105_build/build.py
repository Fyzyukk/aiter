import json,pathlib,sys,subprocess,os,concurrent.futures,time,hashlib
root=pathlib.Path.cwd();sys.path.insert(0,str(root/'csrc/opus_gemm'))
from opus_gemm_bpreshuffle_config import *
from gen_instances import opus_gemm_codegen
out=root/'reports/opus_configflow_20261009/canonical105_build'
out.mkdir(exist_ok=True)
examples=list(CONFIGS_BY_LEGACY_KID.values())
ref=json.loads((root/'reports/opus_flydsl_all_20261009/main_variants/build_receipt.json').read_text())
flags={}
for b in ref['builds']:
 flags[b['side']]=[]
 for t in b['compile_argv']:
  if t=='-c':break
  if not t.startswith('-I'):flags[b['side']].append(t)
env=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def run(c):
 path=out/str(c.legacy_kid);path.mkdir(exist_ok=True)
 inst=kernel_instance_from_config(c);cg=opus_gemm_codegen(str(path));cg.gen_instances({1:inst})
 src=next((path/'instances').glob('*.device.cu'));obj=path/'device.o';log=path/'device.log'
 argv=flags['pin24' if c.pin_agpr else 'main23']+['-mllvm', '--disable-machine-licm', f'-I{root}/csrc/include',f'-I{root}/csrc/opus_gemm/include',f'-I{path}','-c',str(src),'-o',str(obj)]
 started=time.monotonic();p=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True);log.write_text(p.stdout+p.stderr)
 row=dict(pipeline=c.pipeline,compile_params=dict(c.compile_params),legacy_kid=c.legacy_kid,argv=argv,returncode=p.returncode,seconds=time.monotonic()-started,object=str(obj),log=str(log),source=str(src))
 row.update(source_sha256=sha(src), object_sha256=sha(obj) if obj.exists() else None, log_sha256=sha(log), impl_sha256=sha(next((path/'impl').glob('*.cuh'))))
 print(json.dumps({k:row[k] for k in ('legacy_kid','pipeline','returncode','seconds')}),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p: rows=list(p.map(run,examples))
rows.sort(key=lambda r:r['legacy_kid'])
receipt=dict(status='all105_canonical_object_compilation_passed' if all(r['returncode']==0 for r in rows) else 'failed', compile_count=len(rows), gpu_queries=0, library_loads=0, kernel_launches=0, builds=rows)
receipt['compiler_sha256']={argv[0]:sha(argv[0]) for argv in flags.values()}
receipt['source_identity_sha256']={str(p.relative_to(root)):sha(p) for p in [root/'csrc/opus_gemm/opus_gemm_bpreshuffle_catalog.json',root/'csrc/opus_gemm/opus_gemm_bpreshuffle_config.py',root/'csrc/opus_gemm/codegen/gen_instances_gfx950.py',root/'csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh']}
receipt['build_script_sha256']=sha(__file__)
(out/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(any(x['returncode'] for x in rows))
