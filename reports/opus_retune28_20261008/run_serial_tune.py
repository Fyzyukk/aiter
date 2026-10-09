#!/usr/bin/env python3
"""Official task enumeration/worker/profiler in one KFD-owned process.

Large reference and comparison expressions are evaluated by row chunks; only
large timed argument pools are capped. No registry or candidate is pruned.
"""
from pathlib import Path
import argparse, collections, hashlib, json, math, os, sys, time, gc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from aiter.utility import mp_tuner as mp_impl
from aiter import test_common
from aiter.jit import core

OUT=Path(__file__).resolve().parent
LARGE_ELEMENTS=32*1024*1024
CHUNK=256
original_opus_ref=tune.run_torch
original_external_ref=tune.generic_tune.run_torch
original_compare=tune.compare_outputs
original_allclose=mp_impl.checkAllclose
original_perf=test_common.run_perftest
run_meta={'started_unix':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timing':'unchanged torch profiler run_perftest; large rotation cap8/free-memory bound','reference_chunk_rows':CHUNK,'large_elements_threshold':LARGE_ELEMENTS,'single_pid':os.getpid(),'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID'),'shape_progress':[],'large_rotation_counts':{}}


def opus_ref(x,w,x_scale,w_scale,*,with_bounds=False):
    if x.shape[0]*w.shape[0]<=LARGE_ELEMENTS:
        return original_opus_ref(x,w,x_scale,w_scale,with_bounds=with_bounds)
    result=torch.empty(((2,x.shape[0],w.shape[0]) if with_bounds else (x.shape[0],w.shape[0])),device=x.device,dtype=torch.float32)
    for b in range(0,x.shape[0],CHUNK):
        e=min(b+CHUNK,x.shape[0])
        chunk=original_opus_ref(x[b:e],w,x_scale[b:e],w_scale,with_bounds=with_bounds)
        if with_bounds:result[:,b:e].copy_(chunk)
        else:result[b:e].copy_(chunk)
        del chunk
    return result


def external_ref(x,w,x_scale,w_scale,bias=None,dtype=tune.dtypes.bf16):
    if x.shape[0]*w.shape[0]<=LARGE_ELEMENTS:
        return original_external_ref(x,w,x_scale,w_scale,bias,dtype)
    result=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=dtype)
    for b in range(0,x.shape[0],CHUNK):
        e=min(b+CHUNK,x.shape[0]);chunk=original_external_ref(x[b:e],w,x_scale[b:e],w_scale,bias,dtype)
        result[b:e].copy_(chunk);del chunk
    return result


def compare(ref,out,**kwargs):
    if out.numel()<=LARGE_ELEMENTS:return original_compare(ref,out,**kwargs)
    # Zero-outlier gate: return positive for any failing chunk, matching the
    # official ceil-to-four-decimals contract and full-output denominator.
    for b in range(0,out.shape[0],CHUNK):
        e=min(b+CHUNK,out.shape[0])
        if original_compare(ref[:,b:e],out[b:e],**{**kwargs,'printLog':False}):return 0.0001
    return 0.0


def allclose(a,b,**kwargs):
    if a.numel()<=LARGE_ELEMENTS:return original_allclose(a,b,**kwargs)
    count=0;finite=True;ref_max=1.0;delta_max=0.0
    for begin in range(0,a.shape[0],CHUNK):
        end=min(begin+CHUNK,a.shape[0]);aa=a[begin:end];bb=b[begin:end]
        count+=int((~torch.isclose(aa,bb,rtol=kwargs.get('rtol',1e-2),atol=kwargs.get('atol',1e-2))).sum().item())
        finite=finite and bool(torch.isfinite(aa).all()) and bool(torch.isfinite(bb).all())
        ref_max=max(ref_max,float(bb.abs().max().item()))
        delta_max=max(delta_max,float((aa-bb).abs().max().item()))
    ratio=count/a.numel()
    if not count:return 0.0
    # Official silent path returns immediately when ratio >= threshold.
    if ratio>=kwargs.get('tol_err_ratio',.05):return ratio
    if kwargs.get('max_abs_delta') is not None:
        return 1.0 if delta_max>kwargs['max_abs_delta'] else ratio
    if kwargs.get('catastrophic_check',False) and (not finite or delta_max>ref_max*test_common._CATASTROPHIC_REL_THRESHOLD):return 1.0
    return ratio



def perftest(func,*args,**kwargs):
    tensors=[a for a in args if isinstance(a,torch.Tensor)]
    out_tensors=[a for a in tensors if a.dtype==torch.bfloat16 and a.ndim==2]
    if any(a.numel()>LARGE_ELEMENTS for a in out_tensors):
        torch.cuda.empty_cache();footprint=sum(a.untyped_storage().nbytes() for a in tensors)
        free=torch.cuda.mem_get_info()[0]
        count=max(1,min(8,kwargs.get('num_iters',51),int(free*.5/max(1,footprint))))
        kwargs['num_rotate_args']=count
        shape=tuple(out_tensors[0].shape)
        run_meta['large_rotation_counts'].setdefault(str(shape),set()).add(count)
    return original_perf(func,*args,**kwargs)


def save_meta(path):
    path=Path(path)
    value={**run_meta,'large_rotation_counts':{k:sorted(v) for k,v in run_meta['large_rotation_counts'].items()}}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def serial_mp(tasks,in_datas,mp_num=1,fast_mode=True,shape_grouped=True,err_ratio=.05,timeout=None,verbose=False):
    assert mp_num==1 and fast_mode and shape_grouped
    groups=collections.OrderedDict()
    for task in tasks:groups.setdefault(task[0][0],[]).append(task)
    assert len(groups)==len(in_datas)
    results=[]
    for shape,group in groups.items():
        stamp=time.monotonic()
        # Put all original OPUS candidates together, randomize their order by
        # shape, and interleave variants with their original family members.
        # External enumeration is preserved; timing is sequential throughout.
        import random
        external=[t for t in group if t[0][4]!='opus'];opus=[t for t in group if t[0][4]=='opus']
        random.Random(0x2800+sum(int(x) for x in shape[2:])).shuffle(opus)
        group=external+opus
        result=mp_impl.work_group({os.getpid():0},True,err_ratio,(len(group),()),group,verbose=False)
        results.extend(result)
        valid=sum(math.isfinite(t[1]) and t[1]>0 and t[2]<=(0 if t[0][4]=='opus' else err_ratio) for t in result)
        progress={'shape':list(shape),'tasks':len(group),'valid':valid,'seconds':time.monotonic()-stamp}
        run_meta['shape_progress'].append(progress);save_meta(run_meta['metadata_path'])
        print(json.dumps({'event':'shape_complete',**progress,'completed':len(run_meta['shape_progress'])}),flush=True)
        gc.collect();torch.cuda.empty_cache()
    return results


def main():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--plan',type=Path,required=True);pre.add_argument('--metadata',type=Path,required=True)
    own,other=pre.parse_known_args();plan=json.loads(own.plan.read_text());run_meta['metadata_path']=own.metadata
    # Preserve serialization of path fields.
    run_meta['metadata_path']=str(own.metadata)
    run_meta['gpu']=str(torch.cuda.get_device_properties(0))
    import ctypes,ctypes.util
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or '/opt/rocm/lib/libamdhip64.so')
    buf=ctypes.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0
    def normalize(value):
        domain,bus,devfun=value.lower().split(':');dev,fun=devfun.split('.')
        return f'{int(domain,16):04x}:{int(bus,16):02x}:{int(dev,16):02x}.{int(fun)}'
    bdf=normalize(buf.value.decode());expected=normalize(os.environ['OPUS_EXPECTED_GPU_BDF'])
    assert expected==bdf,(expected,bdf)
    run_meta['gpu_bdf']=bdf
    run_meta['plan_sha256']=hashlib.sha256(own.plan.read_bytes()).hexdigest()
    run_meta['libraries']={}
    for name,path in plan['libraries'].items():
        path=Path(path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if name=='opus':assert sha==plan['official_sha256']
        run_meta['libraries'][name]={'path':str(path),'sha256':sha}
    for name in ['module_aiter_core','module_deepgemm_opus','module_gemm_a8w8_blockscale_bpreshuffle_tune','module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune']:
        module=core.get_module(name);path=Path(module.__file__).resolve()
        assert path.parent==Path(os.environ['AITER_JIT_DIR']).resolve()
    assert tune._ensure_kids_compiled(tune.A8W8_BPRESHUFFLE_TUNING_KIDS) is False
    assert len(tune.A8W8_BPRESHUFFLE_TUNING_KIDS)==28
    tune.run_torch=opus_ref;tune.generic_tune.run_torch=external_ref;tune.compare_outputs=compare
    mp_impl.checkAllclose=allclose;test_common.run_perftest=perftest;tune.generic_tune.mp_tuner=serial_mp
    tuner=tune.OpusMxscaleBpreshuffleTuner();args=tuner.parser.parse_args(other)
    # Save the complete legal OPUS enumeration before running.
    import pandas as pd
    input_rows=pd.read_csv(args.untune_file);inventory=[]
    for row in input_rows.itertuples(index=False):
        for kid in tuner._candidate_kids(row.gfx,row.M,row.N,row.K):
            inventory.append({'M':row.M,'N':row.N,'K':row.K,'kernelId':kid})
    pd.DataFrame(inventory).to_csv(own.metadata.with_name(own.metadata.stem+'_expected_opus_candidates.csv'),index=False)
    run_meta['expected_opus_tasks']=len(inventory);run_meta['expected_opus_kids']=sorted({r['kernelId'] for r in inventory})
    for old_path in (Path(args.tune_file),Path(args.profile_file)):
        if old_path.exists():old_path.rename(old_path.with_name(old_path.stem+'.attempt_'+str(time.time_ns())+old_path.suffix))
    save_meta(own.metadata)
    try:
        tuner.run(args);run_meta['status']='completed'
    except BaseException:
        run_meta['status']='failed';raise
    finally:
        run_meta['ended_unix']=time.time();save_meta(own.metadata)
    assert len(run_meta['shape_progress'])==len(input_rows)

if __name__=='__main__':main()
