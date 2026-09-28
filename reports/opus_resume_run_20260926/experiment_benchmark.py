#!/usr/bin/env python3
"""Compare kernel variants on target shapes with prior CKTile winners remeasured.

Uses the original FP32 comparator and run_perftest(5 warmups, 51 iterations,
automatic argument rotation). Compilation must finish before invoking this file.
"""
import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import benchmark as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('shortk', 'k1536', 'shortk_4wave', 'k1536_4wave',
                                             'shortk_xor', 'k1536_xor',
                                             'shortk_epilogue', 'k1536_epilogue'), required=True)
    parser.add_argument('--shapes', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    assert args.rounds >= 5
    exp = ROOT / 'reports/opus_resume_20260926' / (args.family + '_exp')
    variants = json.loads((exp / 'variants.json').read_text())
    assert isinstance(variants, list) and variants
    control_names = {v['name'] for v in variants if v['id'] == v['base_kid']}
    shapes = [tuple(shape) for shape in base.read_shapes(args.shapes, False)
              if shape[2] in {int(v['fixed_k']) for v in variants}]
    assert shapes
    jit = HERE / 'jit'
    sources, binaries, references = base.snapshot(jit)
    historical_path = ROOT / 'reports/opus_local_gap_current_20260925/final_comparison.csv'
    historical = {tuple(int(row[k]) for k in ('M','N','K')):row
                  for row in base.read_csv(historical_path)}
    assert all(shape in historical for shape in shapes)
    protected = {str(ROOT / p): h for p, h in sources.items()} | binaries
    protected[str(Path(__file__))] = base.sha256(__file__)
    protected[str(historical_path)] = base.sha256(historical_path)
    protected.update({str(p): base.sha256(p) for p in exp.rglob('*')
                      if p.is_file() and p.suffix in ('.py', '.cuh', '.hip', '.json', '.so')})
    assert (exp / 'experiments.so').is_file()
    prefix = (HERE / args.prefix).resolve()
    paths = {kind: Path(f'{prefix}_{kind}.{ext}') for kind, ext in
             [('run','json'), ('correctness','csv'), ('raw','csv'),
              ('summary','csv'), ('comparison','csv'), ('gpu','jsonl')]}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', start_time=time.time(), pid=os.getpid(),
                    argv=sys.argv, gpu=base.GPU_INDEX, gpu_uuid=base.GPU_UUID,
                    family=args.family, shapes=shapes, rounds=args.rounds,
                    variants=variants, shapes_file=str(args.shapes.resolve()),
                    protected_sha256=protected, outputs={k:str(v) for k,v in paths.items()},
                    warmup=5, iters=51, num_rotate_args=0,
                    numerical_contract='Original FP32 interval, error==0, NaN-prefilled output and 64-row canaries',
                    timing='Original run_perftest GPU profiler, automatic full storage/input rotation; no compilation',
                    selection='All legal registered OPUS, independent experiments, CKTile 27/28/29 and prior fastest CKTile, all remeasured together; no full backend sweep')
    paths['run'].write_text(json.dumps(manifest, indent=2)+'\n')
    fields = {
        'correctness': ['M','N','K','name','phase','round','status','error','guards','reason'],
        'raw': ['M','N','K','name','lib','kid','round','order','us','status'],
        'summary': ['M','N','K','name','lib','kid','median_us','min_us','max_us','samples'],
        'comparison': ['M','N','K','reference','reference_us','old_opus','old_opus_us',
                       'registered_opus','registered_opus_us','experiment','experiment_us',
                       'control','control_us','experiment_vs_control_pct',
                       'experiment_vs_old_pct','experiment_vs_registered_pct','experiment_vs_reference_pct'],
    }
    streams = {key: paths[key].open('x', newline='') for key in fields}
    writers = {key:csv.DictWriter(streams[key], fieldnames=value) for key,value in fields.items()}
    for writer in writers.values():
        writer.writeheader()
    def append(kind, row):
        writers[kind].writerow(row)
        streams[kind].flush()
    def save():
        temporary = paths['run'].with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(paths['run'])
    class Telemetry:
        def sample(self, row):
            with paths['gpu'].open('a') as stream:
                stream.write(json.dumps(row)+'\n')
    telemetry = Telemetry()
    checks, records, comparisons = [], [], []
    try:
        for key in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):
            os.environ.pop(key, None)
        os.environ.update(ROCR_VISIBLE_DEVICES=base.GPU_UUID, AITER_AOT_IMPORT='1',
                          AITER_REBUILD='0', AITER_JIT_DIR=str(jit), GPU_ARCHS='gfx950',
                          CU_NUM='256', OMP_NUM_THREADS='2', AITER_LOG_MORE='0', AITER_SMI_MONITOR='0')
        sys.path.insert(0, str(ROOT))
        import torch
        import aiter
        from aiter.utility import dtypes
        from aiter.ops import gemm_op_a8w8
        from aiter.jit import core
        from aiter.test_common import run_perftest
        aiter.dtypes = dtypes
        for name in ('gemm_a8w8_blockscale_bpreshuffle_tune',
                     'gemm_a8w8_blockscale_bpreshuffle_cktile_tune',
                     'gemm_a8w8_blockscale_bpreshuffle_asm'):
            setattr(aiter, name, getattr(gemm_op_a8w8, name))
        def forbid_build(*a, **kw):
            raise RuntimeError('Experiments require prebuilt modules')
        core.build_module = forbid_build
        assert torch.cuda.device_count() == 1
        prop = torch.cuda.get_device_properties(0)
        assert (prop.pci_bus_id, prop.pci_domain_id, prop.pci_device_id,
                prop.multi_processor_count) == (base.GPU_BUS, 0, 0, 256)
        assert prop.gcnArchName.startswith('gfx950')
        for name in base.MODULES:
            if not name.endswith('_asm'):
                assert Path(core.get_module(name).__file__).resolve() == jit / (name+'.so')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest.update(gpu_properties=str(prop), torch_version=torch.__version__, hip_version=torch.version.hip)
        save()
        tune = base.load_tune()
        tuner = tune.OpusMxscaleBpreshuffleTuner()
        lib = ctypes.CDLL(str(exp / 'experiments.so'))
        lib.launch.argtypes = [ctypes.c_int, *([ctypes.c_void_p]*5),
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.launch.restype = ctypes.c_int
        entries = {}
        keys = ['x','w','out','x_scale','w_scale','x_scale_t_fp32','x_scale_fp32','w_scale_fp32']
        def invoke(name, x, w, storage, sa, sb, sat32, sar32, sb32):
            m, k = x.shape
            n = storage.shape[1]
            out = storage[64:64+m]
            candidate = entries[name]
            if isinstance(candidate, dict):
                code = lib.launch(candidate['id'], x.data_ptr(), w.data_ptr(), sa.data_ptr(),
                                  sb.data_ptr(), out.data_ptr(), m, n, k,
                                  torch.cuda.current_stream().cuda_stream)
                if code:
                    raise RuntimeError(f'HIP launch {name}: {code}')
            else:
                data = dict(x=x,w=w,weight_shuffle=w,out=out,x_scale=sa,w_scale=sb,
                            x_scale_t_fp32=sat32,x_scale_fp32=sar32,w_scale_fp32=sb32)
                candidate.func(*candidate.arguments(data), **candidate.kwargs)
            return storage
        def verify(shape, name, phase, turn, reference, storage):
            m, n, k = shape
            checked = base.check_output(torch, tune, reference, storage[64:64+m])
            guards = bool(torch.isnan(storage[:64]).all() and torch.isnan(storage[64+m:]).all())
            if not guards:
                checked.update(status='failed', reason='output canary overwritten')
            row = dict(M=m,N=n,K=k,name=name,phase=phase,round=turn,guards=guards,
                       **{key:checked[key] for key in ('status','error','reason')})
            append('correctness', row)
            checks.append(row)
            assert row['status'] == 'passed', row
        base.require_idle(torch, telemetry, 'start')
        for shape_index, shape in enumerate(shapes):
            m, n, k = shape
            data = tune.generate_data(m,n,k,device='cuda:0')
            reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS),with_bounds=True)
            data['out'] = torch.full((m+128,n),float('nan'),dtype=torch.bfloat16,device='cuda')
            names, metadata = [], {}
            for variant in variants:
                if variant['fixed_k'] == k:
                    name = variant['name']
                    entries[name] = variant
                    names.append(name)
                    metadata[name] = dict(lib='experiment',kid=variant['id'])
            if shape in shapes:
                candidates = base.candidates_for_shape(tune,tuner,shape)
                for candidate in candidates:
                    row = candidate.row
                    if (row['lib'] == 'opus' or row['lib'] == 'cktile' and
                            (row['kid'] in (27,28,29) or row['name'] == historical[shape]['cktile_best'])):
                        names.append(row['name'])
                        entries[row['name']] = candidate
                        metadata[row['name']] = dict(lib=row['lib'],kid=row['kid'])
            for name in names:
                data['out'].fill_(float('nan'))
                actual = invoke(name,*(data[key] for key in keys))
                verify(shape,name,'preflight',-1,reference,actual)
            if shape in shapes:
                local = []
                for turn in range(args.rounds):
                    base.require_idle(torch,telemetry,f'before_{shape}_{turn}')
                    shift = (shape_index + turn) % len(names)
                    order = names[shift:] + names[:shift]
                    if turn % 2:
                        order.reverse()
                    for position,name in enumerate(order):
                        data['out'].fill_(float('nan'))
                        actual, us = run_perftest(invoke,name,*(data[key] for key in keys),
                                                 num_warmup=5,num_iters=51,num_rotate_args=0)
                        verify(shape,name,'timed',turn,reference,actual)
                        assert math.isfinite(us) and us > 0
                        row = dict(M=m,N=n,K=k,name=name,**metadata[name],round=turn,
                                   order=position,us=float(us),status='passed')
                        append('raw',row)
                        records.append(row)
                        local.append(row)
                    base.require_idle(torch,telemetry,f'after_{shape}_{turn}')
                summaries = []
                for name in names:
                    values = [r['us'] for r in local if r['name'] == name]
                    assert len(values) == args.rounds
                    summary = dict(M=m,N=n,K=k,name=name,**metadata[name],median_us=statistics.median(values),
                                   min_us=min(values),max_us=max(values),samples=len(values))
                    append('summary',summary)
                    summaries.append(summary)
                def best(predicate):
                    return min((r for r in summaries if predicate(r)),key=lambda r:r['median_us'])
                ref = best(lambda r:r['lib'] not in ('opus','experiment'))
                old = best(lambda r:r['lib']=='opus' and r['kid'] in (9000,9010,9011,9012,9020))
                registered = best(lambda r:r['lib']=='opus')
                experiment = best(lambda r:r['lib']=='experiment' and r['name'] not in control_names)
                control = best(lambda r:r['name'] in control_names)
                comp = dict(M=m,N=n,K=k,reference=ref['name'],reference_us=ref['median_us'],
                            old_opus=old['name'],old_opus_us=old['median_us'],
                            registered_opus=registered['name'],registered_opus_us=registered['median_us'],
                            experiment=experiment['name'],experiment_us=experiment['median_us'],
                            control=control['name'],control_us=control['median_us'])
                for label, rival in [('old',old),('registered',registered),('reference',ref),('control',control)]:
                    comp[f'experiment_vs_{label}_pct'] = (experiment['median_us']/rival['median_us']-1)*100
                append('comparison',comp)
                comparisons.append(comp)
                print(json.dumps(comp),flush=True)
            del data, reference, actual
            torch.cuda.empty_cache()
            manifest.update(correctness_checks=len(checks),timing_records=len(records),completed_shapes=len(comparisons))
            save()
        base.require_idle(torch,telemetry,'end')
        for path,digest in protected.items():
            assert base.sha256(path) == digest,path
        manifest['status'] = 'passed'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        for stream in streams.values():
            stream.close()
        manifest.update(end_time=time.time(),correctness_checks=len(checks),timing_records=len(records),
                        completed_shapes=len(comparisons),output_sha256={k:base.sha256(p) for k,p in paths.items()
                                                                       if k!='run' and p.is_file()})
        save()


if __name__ == '__main__':
    main()
