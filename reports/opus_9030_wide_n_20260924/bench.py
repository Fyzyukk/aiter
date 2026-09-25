"""Validate M/K tails and compare wide-N candidates against saved rivals."""
import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARNESS = ROOT/'reports/opus_9010_9011_opt_20260924/harness'
sys.path.insert(0, str(HARNESS))
from benchmark import GPU_UUID, candidates_for_shape, check_output, require_idle, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--variants', type=int, nargs='+', default=[9030,9031,9032,9033])
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    variants = {v['id']:v for v in json.loads((HERE/'variants.json').read_text())}
    assert set(args.variants) <= variants.keys() and args.rounds > 0
    prefix = HERE/args.prefix
    paths = {kind:Path(f'{prefix}_{kind}.{ext}') for kind,ext in
             [('run','json'),('correctness','csv'),('raw','csv'),('summary','csv'),('gpu','jsonl')]}
    for p in paths.values():
        if p.exists(): raise FileExistsError(p)
    build = json.loads((HERE/'build_manifest.json').read_text())
    assert build['status'] == 'passed'
    protected = dict(build['source_sha256'])
    protected.update({str(HERE/name):sha256(HERE/name) for name in
                      ['experiments.so','variants.json','targets.csv','bench.py','build_manifest.json']})
    for p,h in protected.items(): assert sha256(p)==h,p
    manifest = dict(status='starting', start_time=time.time(), pid=os.getpid(), argv=sys.argv,
                    gpu=7, gpu_uuid=GPU_UUID, protected_sha256=protected,
                    warmup=5, iters=51, rounds=args.rounds, variants=args.variants,
                    numerical_contract='Unchanged FP32 accumulation interval; exact E8M0 scale decodes for rivals',
                    timing='run_perftest automatic argument rotation; complete guarded output rotated',
                    reference_scope='CKTile 11/27/28/29, saved final CK finalists, legal OPUS 9000/9010/9011')
    paths['run'].write_text(json.dumps(manifest,indent=2)+'\n')
    def save(): paths['run'].write_text(json.dumps(manifest,indent=2)+'\n')
    class Telemetry:
        def sample(self,value):
            with paths['gpu'].open('a') as f: f.write(json.dumps(value)+'\n')
    telemetry=Telemetry()
    checks=[]; records=[]; rejected=set()
    def append(kind,row):
        with paths[kind].open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if f.tell()==0:writer.writeheader()
            writer.writerow(row)
    try:
        samples=[]
        for i in range(3):
            card=json.loads(subprocess.check_output(['rocm-smi','-d','7','--showuse','--showmemuse','--json'],text=True))['card7']
            samples.append(card);telemetry.sample(dict(time=time.time(),phase='before_gpu_init',card=card))
            if int(card['GPU use (%)']) or int(card['GPU Memory Allocated (VRAM%)'])>1:
                raise RuntimeError('GPU 7 is occupied; no GPU work started')
            if i<2:time.sleep(5)
        if len({s['GFX Activity'] for s in samples})!=1:raise RuntimeError('GPU 7 activity changed')
        for key in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:os.environ.pop(key,None)
        jit=ROOT/'reports/opus_9010_9011_opt_20260924/jit_final'
        os.environ.update(ROCR_VISIBLE_DEVICES=GPU_UUID,AITER_AOT_IMPORT='1',AITER_REBUILD='0',
                          AITER_JIT_DIR=str(jit),GPU_ARCHS='gfx950',CU_NUM='256',OMP_NUM_THREADS='2',AITER_LOG_MORE='0')
        sys.path.insert(0,str(ROOT))
        import torch
        import aiter
        from aiter.utility import dtypes
        from aiter.ops import gemm_op_a8w8
        aiter.dtypes=dtypes
        for name in ['gemm_a8w8_blockscale_bpreshuffle_tune','gemm_a8w8_blockscale_bpreshuffle_cktile_tune',
                     'gemm_a8w8_blockscale_bpreshuffle_asm']:
            setattr(aiter,name,getattr(gemm_op_a8w8,name))
        from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
        from aiter.test_common import run_perftest
        from aiter.jit import core
        def forbid_build(*a,**kw):raise RuntimeError('Benchmark must use prebuilt modules')
        core.build_module=forbid_build
        assert torch.cuda.device_count()==1
        prop=torch.cuda.get_device_properties(0)
        assert prop.pci_bus_id==0xF5 and prop.multi_processor_count==256 and prop.gcnArchName.startswith('gfx950')
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        manifest.update(status='running',gpu_properties=str(prop),gpu_work_started=True);save()
        lib=ctypes.CDLL(str(HERE/'experiments.so'))
        lib.launch.argtypes=[ctypes.c_int,*([ctypes.c_void_p]*5),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        lib.launch.restype=ctypes.c_int
        tuner=tune.OpusMxscaleBpreshuffleTuner()
        targets=[tuple(int(r[k]) for k in ['M','N','K']) for r in csv.DictReader((HERE/'targets.csv').open())]
        # Exercise all three M-tail residues, both B scale halves, short loops,
        # both panel boundaries and multiple panels before accepting timings.
        edges=[(m,256,k) for m in [64,128,192,256,320] for k in [128,256,384]]
        edges += [(m,512,k) for m in [64,192,256,1088] for k in [8064,8192,8320,16256,16384,16512]]
        if args.smoke:edges=[(64,256,128),(192,256,256),(256,512,384)]
        final_rows=list(csv.DictReader((HARNESS/'final_r5_summary.csv').open()))
        guard=64
        entries={}
        def invoke(name,x,w,storage,sa,sb,sat32,sar32,sb32):
            m,n,k=x.shape[0],storage.shape[1],x.shape[1]
            out=storage[guard:guard+m]
            if name.startswith('wide_'):
                code=lib.launch(entries[name],x.data_ptr(),w.data_ptr(),sa.data_ptr(),sb.data_ptr(),out.data_ptr(),
                                m,n,k,torch.cuda.current_stream().cuda_stream)
                if code:raise RuntimeError(f'HIP launch error {code}: {name}')
            else:
                candidate=entries[name]
                data=dict(x=x,w=w,weight_shuffle=w,out=out,x_scale=sa,w_scale=sb,
                          x_scale_t_fp32=sat32,x_scale_fp32=sar32,w_scale_fp32=sb32)
                candidate.func(*candidate.arguments(data),**candidate.kwargs)
            return storage
        keys=['x','w','out','x_scale','w_scale','x_scale_t_fp32','x_scale_fp32','w_scale_fp32']
        def verify(ref,storage,m):
            r=check_output(torch,tune,ref,storage[guard:guard+m])
            safe=bool(torch.isnan(storage[:guard]).all() and torch.isnan(storage[guard+m:]).all())
            if not safe:r.update(status='failed',reason='output canary overwritten')
            return r,safe
        def check_record(shape,name,phase,turn,r,safe):
            row=dict(zip(['M','N','K'],shape),name=name,phase=phase,round=turn,status=r['status'],
                     error=r['error'],guards=safe,reason=r.get('reason',''),mismatches=r.get('mismatches',''),
                     max_abs_error=r.get('max_abs_error',''))
            checks.append(row);append('correctness',row)
        require_idle(torch,telemetry,'start')
        for shape in edges+targets:
            m,n,k=shape
            data=tune.generate_data(m,n,k,device='cuda:0')
            ref=tune.run_torch(*(data[key] for key in tune._REF_KEYS),with_bounds=True)
            data['out']=torch.full((m+2*guard,n),float('nan'),device='cuda',dtype=torch.bfloat16)
            names=[]
            for v in args.variants:
                name=variants[v]['name'];entries[name]=v
                if name not in rejected:names.append(name)
            if shape in targets and not args.correctness_only:
                saved_ck={r['name'] for r in final_rows if tuple(int(r[x]) for x in ['M','N','K'])==shape and r['lib']=='ck'}
                for c in candidates_for_shape(tune,tuner,shape):
                    r=c.row
                    if (r['lib']=='cktile' and r['kid'] in [11,27,28,29] or
                        r['lib']=='ck' and r['name'] in saved_ck or
                        r['lib']=='opus' and r['kid'] in [9000,9010,9011]):
                        names.append(r['name']);entries[r['name']]=c
            for name in names:
                data['out'].fill_(float('nan'))
                actual=invoke(name,*(data[key] for key in keys))
                r,safe=verify(ref,actual,m)
                check_record(shape,name,'preflight',-1,r,safe)
                if r['status']!='passed':
                    rejected.add(name);print('REJECTED',shape,name,r,flush=True)
            if shape in targets and not args.correctness_only:
                for turn in range(args.rounds):
                    require_idle(torch,telemetry,f'{shape}_round{turn}')
                    active=[name for name in names if name not in rejected]
                    shift=turn%len(active) if active else 0
                    active=active[shift:]+active[:shift]
                    if turn%2:active.reverse()
                    for name in active:
                        data['out'].fill_(float('nan'))
                        actual,us=run_perftest(invoke,name,*(data[key] for key in keys),num_warmup=5,num_iters=51)
                        r,safe=verify(ref,actual,m)
                        if not math.isfinite(us) or us<=0:r.update(status='failed',reason='invalid timing')
                        check_record(shape,name,'timed',turn,r,safe)
                        row=dict(zip(['M','N','K'],shape),name=name,round=turn,us=float(us),status=r['status'])
                        records.append(row);append('raw',row)
                        if r['status']!='passed':rejected.add(name)
                medians={name:statistics.median(r['us'] for r in records if tuple(r[x] for x in ['M','N','K'])==shape and r['name']==name)
                         for name in names if name not in rejected}
                print(shape,{k:round(v,3) for k,v in medians.items()},flush=True)
            else:print('checked',shape,flush=True)
            del data,ref
            if 'actual' in locals():del actual
            torch.cuda.empty_cache()
            manifest.update(correctness_checks=len(checks),timing_records=len(records),rejected=sorted(rejected));save()
        for shape in targets:
            for name in sorted({r['name'] for r in records} - rejected):
                values=[r['us'] for r in records if tuple(r[x] for x in ['M','N','K'])==shape and r['name']==name and r['status']=='passed']
                if len(values)==args.rounds:
                    append('summary',dict(zip(['M','N','K'],shape),name=name,median_us=statistics.median(values),
                                          min_us=min(values),max_us=max(values),samples=len(values)))
        require_idle(torch,telemetry,'end')
        for p,h in protected.items():assert sha256(p)==h,p
        manifest['status']='completed_with_rejections' if rejected else 'passed'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                        error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(end_time=time.time(),correctness_checks=len(checks),timing_records=len(records),rejected=sorted(rejected));save()
    print(json.dumps({k:manifest[k] for k in ['status','correctness_checks','timing_records','rejected']}),flush=True)

if __name__=='__main__':main()
