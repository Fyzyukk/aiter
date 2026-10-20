"""Seeded whole-shape retune, followed by paired three-backend confirmation.

Ordinary public calls, automatic tensor rotation and full-output checking.
Default OPUS includes its automatic stream-local workspace handling.
"""
import argparse,csv,gc,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[1]

def run_opus(x,w,xs,ws,out,kid):
 from aiter.ops.opus import opus_gemm
 return opus_gemm(x,w,out,x_scale=xs,w_scale=ws,layout='bpreshuffle',kid=kid)

def worker(a):
 from report_environment import configure
 configure();os.environ['ROCR_VISIBLE_DEVICES']=str(a.part+4)
 for q in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):os.environ.pop(q,None)
 sys.path.insert(0,str(ROOT));import aiter
 from aiter.utility import dtypes
 aiter.dtypes=dtypes
 import torch
 from torch.nn import functional as F
 from aiter.test_common import run_perftest
 from csrc.gemm_a8w8_mxfp8_bpreshuffle import gemm_a8w8_mxfp8_bpreshuffle_tune as t
 from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_gemm_bpreshuffle_kernels_list as reg
 torch.set_num_threads(4)
 fp=json.loads((P/'build_fingerprint.json').read_text());assert all(hashlib.sha256((ROOT/q).read_bytes()).hexdigest()==sha for q,sha in fp.items())
 plan=json.loads((P/a.plan).read_text())
 for idx,s in enumerate(plan[a.part::4]):
  m,n,k=s['shape'];dest=P/a.output/f'{m}_{n}_{k}.json'
  if dest.exists():continue
  start=time.time();d=t.generate_data(m,n,k,173,'128x128');expected=F.linear(d['x_deq'],d['w_deq']);bound=1e-4+5e-5*F.linear(d['x_deq'].abs(),d['w_deq'].abs());lo=(expected-bound).to(dtypes.bf16).float();hi=(expected+bound).to(dtypes.bf16).float();del expected,bound,d['x_deq'],d['w_deq']
  def check(y):return int(((~torch.isfinite(y))|(y.float()<lo)|(y.float()>hi)).sum().item())
  choices=[];rows=[]
  for kid in s['kids']:choices.append((str(kid),'opus',kid,run_opus,[d[q]for q in ('x','weight_shuffle','x_scale_opus','w_scale','out')]+[kid]))
  for ext in s.get('external',[]):
   fn=t.run_bmm_flydsl if ext['tag']=='bmm'else t.run_gemm_flydsl
   choices.append((ext['name'],ext['tag'],ext['name'],fn,[d[q]for q in ('x','weight_shuffle','x_scale','w_scale','out')]+[ext['name']]))
  valid=[]
  for tag,lib,kid,fn,values in choices:
   d['out'].fill_(float('nan'));y=fn(*values);e=check(y);assert not e,(m,n,k,kid,e);rows.append(dict(tag=tag,lib=lib,kid=kid,stage='precheck',errors=e));valid.append((tag,lib,kid,fn,values))
  # First screen every seeded candidate, then repeat the best three of each
  # family plus the prior OPUS winner. Keep all raw search observations.
  for repeat in range(3):
   offset=(idx+repeat)%len(valid)
   for tag,lib,kid,fn,values in valid[offset:]+valid[:offset]:
    d['out'].fill_(float('nan'));y,us=run_perftest(fn,*values,num_warmup=5,num_iters=51 if repeat==0 and a.mode=='search' else 101)
    e=check(y);assert not e,(m,n,k,kid,e);rows.append(dict(tag=tag,lib=lib,kid=kid,stage='timing',repeat=repeat,us=float(us),errors=e))
   if repeat==0 and a.mode=='search':
    chosen={str(s['baseline'])}
    for lib in ('opus','bmm','pr6603'):
     rank=sorted([r for r in rows if r.get('repeat')==0 and r['lib']==lib],key=lambda r:r['us'])[:3];chosen.update(r['tag']for r in rank)
    valid=[v for v in valid if v[0]in chosen]
  med={tag:statistics.median([r['us']for r in rows if r['tag']==tag and r.get('repeat',-1)>= (1 if a.mode=='search' else 0)])for tag,_,_,_,_ in valid}
  best={lib:min([v for v in valid if v[1]==lib],key=lambda v:med[v[0]])[0]for lib in ('opus','bmm','pr6603')if any(v[1]==lib for v in valid)}
  result=dict(shape=s['shape'],gpu=a.part+4,seed=173,rows=rows,medians=med,best=best,baseline=s['baseline'],timing='ordinary production calls; automatic tensor rotation; no graph; cache initialized by warmup',elapsed=time.time()-start)
  dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(shape=s['shape'],best={lib:dict(tag=tag,us=med[tag])for lib,tag in best.items()},elapsed=result['elapsed'])),flush=True)
  del y,d,lo,hi,choices,valid;gc.collect();torch.cuda.empty_cache()

def launch(a):
 sys.path.insert(0,str(P));from device_occupancy import require_idle_devices,inspect_devices
 state=require_idle_devices();out=P/a.output;out.mkdir(exist_ok=False,parents=True);(out/'occupancy_before.json').write_text(json.dumps(state,indent=2));procs=[]
 for part in range(4):
  with (out/f'gpu{part}.log').open('w')as f:procs.append(subprocess.Popen([sys.executable,'-u',__file__,'--plan',a.plan,'--output',a.output,'--mode',a.mode,'--part',str(part)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT))
 (out/'processes.json').write_text(json.dumps([p.pid for p in procs]));codes=[p.wait()for p in procs];(out/'exit_codes.json').write_text(json.dumps(codes));(out/'occupancy_after.json').write_text(json.dumps(inspect_devices(),indent=2));print(codes);raise SystemExit(any(codes))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',default='tune_plan.json');ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['search','confirm'],default='search');ap.add_argument('--part',type=int);a=ap.parse_args();worker(a)if a.part is not None else launch(a)
