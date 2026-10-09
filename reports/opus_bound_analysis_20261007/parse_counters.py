#!/usr/bin/env python3
"""Parse rocprofv3 long counter CSV without joining unrelated PMC passes.

Only stdlib is imported. No GPU access. Timestamps are SDK nanoseconds.
Custom OPUS_GFX950 counters are scalar reductions and must occur once per
dispatch; raw instance rows are preserved and reduced explicitly.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile

ALIAS_PREFIX = 'OPUS_GFX950_'

def raw_name(name):
    if name.startswith(ALIAS_PREFIX):
        return name[len(ALIAS_PREFIX):].rsplit('_', 1)[0]
    return name

def divide(numerator, denominator):
    return numerator / denominator if denominator else None

def metrics(counters, duration_ns, cu_count=None, simd_count=None):
    out = {'profiled_dispatch_ns': duration_ns}
    def ratio(name, numerator, denominator, factor=1):
        if numerator in counters and denominator in counters:
            value = divide(counters[numerator] * factor, counters[denominator])
            if value is not None: out[name] = value
    ratio('mean_wave_life_cycles_estimate', 'SQ_WAVE_CYCLES', 'SQ_WAVES', 4)
    ratio('wave_any_wait_pct_estimate', 'SQ_WAIT_ANY', 'SQ_WAVE_CYCLES', 100)
    ratio('wave_issue_wait_pct_estimate', 'SQ_WAIT_INST_ANY', 'SQ_WAVE_CYCLES', 100)
    ratio('wave_execute_pct_estimate', 'SQ_ACTIVE_INST_ANY', 'SQ_WAVE_CYCLES', 100)
    for suffix, counter in [('read', 'TCC_EA0_RDREQ_DRAM_32B'),
                            ('write', 'TCC_EA0_WRREQ_WRITE_DRAM_32B')]:
        if counter in counters:
            out[f'dram_{suffix}_bytes'] = 32 * counters[counter]
            out[f'dram_{suffix}_tb_s_profiled'] = out[f'dram_{suffix}_bytes'] / duration_ns / 1000
    if all(f'dram_{suffix}_bytes' in out for suffix in ('read', 'write')):
        out['dram_total_bytes'] = out['dram_read_bytes'] + out['dram_write_bytes']
        out['dram_total_tb_s_profiled'] = out['dram_total_bytes'] / duration_ns / 1000
    sizes = ['TCC_EA0_RDREQ_32B', 'TCC_EA0_RDREQ_64B', 'TCC_EA0_RDREQ_128B']
    if all(name in counters for name in sizes):
        out['ea_total_read_bytes'] = sum(counters[name]*size for name,size in zip(sizes,[32,64,128]))
        out['ea_total_read_tb_s_profiled'] = out['ea_total_read_bytes'] / duration_ns / 1000
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in counters:
        out['issued_f8_operations'] = 512 * counters['SQ_INSTS_VALU_MFMA_MOPS_F8']
        out['issued_f8_tflops_profiled'] = out['issued_f8_operations'] / duration_ns / 1000
    ratio('ta_paper_addr_rate_estimate', 'TA_BUFFER_WAVEFRONTS', 'TA_BUFFER_TOTAL_CYCLES', 4)
    ratio('ta_mean_buffer_cycles_per_wave', 'TA_BUFFER_TOTAL_CYCLES', 'TA_BUFFER_WAVEFRONTS')
    ratio('ta_downstream_stall_pct_estimate', 'TA_ADDR_STALLED_BY_TC_CYCLES', 'TA_TA_BUSY', 100)
    ratio('tcc_tag_stall_pct_estimate', 'TCC_TAG_STALL', 'TCC_BUSY', 100)
    ratio('sq_vmem_addr_stall_pct_estimate', 'SQ_VMEM_TA_ADDR_FIFO_FULL', 'SQ_BUSY_CYCLES', 100)
    for numerator, name in [
        ('TCP_TCP_TA_ADDR_STALL_CYCLES', 'tcp_ta_addr_stall'),
        ('TCP_TCR_TCP_STALL_CYCLES', 'tcp_tcr_stall'),
        ('TCP_LFIFO_STALL_CYCLES', 'tcp_lfifo_stall'),
        ('TCP_RFIFO_STALL_CYCLES', 'tcp_rfifo_stall'),
        ('TCP_READ_TAGCONFLICT_STALL_CYCLES', 'tcp_read_tagconflict_stall'),
        ('TCP_UTCL1_SERIALIZATION_STALL', 'utcl1_serialization_stall'),
        ('TCP_UTCL1_THRASHING_STALL', 'utcl1_thrashing_stall'),
        ('TCP_UTCL1_STALL_INFLIGHT_MAX', 'utcl1_inflight_max_stall')]:
        ratio(name + '_pct_of_interface_clock_estimate', numerator, 'TCP_GATE_EN1', 100)
    if 'TCC_HIT' in counters and 'TCC_MISS' in counters:
        value = divide(counters['TCC_HIT'] * 100, counters['TCC_HIT'] + counters['TCC_MISS'])
        if value is not None: out['tcc_hit_pct'] = value
    tags = [f'TCP_TAGRAM{i}_REQ' for i in range(4)]
    if all(name in counters for name in tags) and sum(counters[name] for name in tags):
        total = sum(counters[name] for name in tags)
        for i,name in enumerate(tags): out[f'tcp_tagram{i}_request_share_pct'] = 100*counters[name]/total
    if simd_count:
        ratio('mfma_util_pct_estimate', 'SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE', 100/simd_count)
    if cu_count:
        ratio('mean_resident_waves_per_cu_estimate', 'SQ_WAVE_CYCLES', 'GRBM_GUI_ACTIVE', 4/cu_count)
    return out

def parse_file(path, kernel_regex, last, cu_count=None, simd_count=None):
    source = str(path.resolve())
    dispatched = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Dispatch_Id','Kernel_Name','Counter_Name','Counter_Value','Start_Timestamp','End_Timestamp'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: requires rocprofv3 long CSV columns {sorted(required)}')
        for line,row in enumerate(reader,2):
            if not kernel_regex.search(row['Kernel_Name']): continue
            key = tuple(row.get(name,'') for name in ['Process_Id','Agent_Id','Queue_Id','Dispatch_Id','Kernel_Id','Correlation_Id'])
            start,end = int(row['Start_Timestamp']),int(row['End_Timestamp'])
            if end <= start: raise ValueError(f'{path}:{line}: invalid SDK nanosecond duration {start}:{end}')
            item = dispatched.setdefault(key, {'source_file':source,'identity':dict(zip(
                ['process','agent','queue','dispatch','kernel_id','correlation'],key)),
                'kernel_name':row['Kernel_Name'],'start_ns':start,'end_ns':end,
                'metadata':{name:row.get(name,'') for name in ['Grid_Size','Workgroup_Size','VGPR_Count','SGPR_Count','LDS_Block_Size','Scratch_Size']},
                'counter_rows':defaultdict(list)})
            if (item['kernel_name'],item['start_ns'],item['end_ns']) != (row['Kernel_Name'],start,end):
                raise ValueError(f'{path}:{line}: inconsistent metadata within dispatch {key}')
            value=float(row['Counter_Value'])
            if not math.isfinite(value): raise ValueError(f'{path}:{line}: nonfinite counter value')
            item['counter_rows'][row['Counter_Name']].append({'value':value,'csv_line':line,
                'instance':{name:row[name] for name in row if 'instance' in name.lower() or 'dimension' in name.lower()}})
    kernels = defaultdict(list)
    for item in dispatched.values():
        counters = {}
        for counter,rows in item['counter_rows'].items():
            if counter.startswith(ALIAS_PREFIX) and len(rows)!=1:
                raise ValueError(f'{path}: scalar custom counter {counter} occurred {len(rows)} times in {item["identity"]}')
            name=raw_name(counter)
            if name in counters: raise ValueError(f'{path}: raw and derived duplicate for {name}')
            values=[row['value'] for row in rows]
            counters[name]=max(values) if name.startswith('GRBM_') else sum(values)
        item['counters']=counters
        item['metrics']=metrics(counters,item['end_ns']-item['start_ns'],cu_count,simd_count)
        kernels[item['kernel_name']].append(item)
    result=[]
    for kernel,items in sorted(kernels.items()):
        items.sort(key=lambda item:(item['start_ns'],item['end_ns'],item['identity']['dispatch']))
        if last and len(items)<last: raise ValueError(f'{path}: {kernel} has {len(items)} dispatches, expected >= {last}')
        selected=items[-last:] if last else items
        names=set.intersection(*(set(item['metrics']) for item in selected))
        counter_names=set.intersection(*(set(item['counters']) for item in selected))
        result.append({'source_file':source,'kernel_name':kernel,'dispatches_before_selection':len(items),
            'dispatches_selected':len(selected),'median_metrics':{n:statistics.median(i['metrics'][n] for i in selected) for n in sorted(names)},
            'median_counters':{n:statistics.median(i['counters'][n] for i in selected) for n in sorted(counter_names)},
            'samples':selected})
    return result

def split_k_pairs(groups):
    by_source=defaultdict(list)
    for group in groups: by_source[group['source_file']].append(group)
    summaries=[]
    for source,entries in by_source.items():
        if len(entries)!=2: raise ValueError(f'{source}: whole-call pairing requires exactly producer and reducer kernels, got {len(entries)}')
        reducers=[g for g in entries if 'reduce_kernel' in g['kernel_name']]
        if len(reducers)!=1: raise ValueError(f'{source}: could not identify one production reducer')
        reducer=reducers[0]; producer=next(g for g in entries if g is not reducer)
        if len(producer['samples'])!=len(reducer['samples']): raise ValueError(f'{source}: unequal producer/reducer counts')
        pairs=[]
        for index,(p,r) in enumerate(zip(producer['samples'],reducer['samples'])):
            if (any(p['identity'][key]!=r['identity'][key] for key in ['process','agent','queue'])
                    or p['end_ns']>r['start_ns']
                    or (index+1<len(producer['samples']) and r['end_ns']>producer['samples'][index+1]['start_ns'])):
                raise ValueError(f'{source}: producer/reducer ordering or agent/queue mismatch')
            pair={'producer_dispatch':p['identity'],'reducer_dispatch':r['identity'],
                'profiled_kernel_time_sum_ns':p['metrics']['profiled_dispatch_ns']+r['metrics']['profiled_dispatch_ns'],
                'profiled_call_span_ns':r['end_ns']-p['start_ns']}
            if 'dram_total_bytes' in p['metrics'] and 'dram_total_bytes' in r['metrics']:
                pair['dram_total_bytes']=p['metrics']['dram_total_bytes']+r['metrics']['dram_total_bytes']
                pair['dram_total_tb_s_profiled_kernel_sum']=pair['dram_total_bytes']/pair['profiled_kernel_time_sum_ns']/1000
                pair['dram_total_tb_s_profiled_call_span']=pair['dram_total_bytes']/pair['profiled_call_span_ns']/1000
            pairs.append(pair)
        summaries.append({'source_file':source,'producer':producer['kernel_name'],'reducer':reducer['kernel_name'],
            'paired_samples':pairs,'median_metrics':{n:statistics.median(p[n] for p in pairs)
                for n in pairs[0] if n not in ['producer_dispatch','reducer_dispatch']}})
    return summaries

def self_test():
    fields=['Process_Id','Agent_Id','Queue_Id','Dispatch_Id','Kernel_Id','Correlation_Id','Kernel_Name','Counter_Name','Counter_Value','Start_Timestamp','End_Timestamp']
    with tempfile.TemporaryDirectory() as directory:
        paths=[]
        for index in range(2):
            path=Path(directory)/f'pass{index}.csv'; paths.append(path)
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
                for dispatch in range(3):
                    base=dict(zip(fields[:7],['1','gpu','q',str(dispatch),'k',str(dispatch),'producer']))
                    base.update(Start_Timestamp=str(dispatch*1000),End_Timestamp=str(dispatch*1000+100))
                    for counter,values in [('SQ_WAVE_CYCLES',[10,20]),('SQ_WAVES',[2,3]),('TCC_EA0_RDREQ_DRAM_32B',[100+index])]:
                        for value in values: writer.writerow({**base,'Counter_Name':counter,'Counter_Value':value})
        groups=[g for p in paths for g in parse_file(p,re.compile('producer'),2)]
        assert len(groups)==2 and all(g['dispatches_selected']==2 for g in groups)
        assert groups[0]['samples'][0]['identity']['dispatch']=='1'
        assert groups[0]['median_metrics']['mean_wave_life_cycles_estimate']==24
        assert groups[0]['median_metrics']['dram_read_tb_s_profiled']==0.032
        assert groups[0]['source_file']!=groups[1]['source_file']
        assert len(groups[0]['samples'][0]['counter_rows']['SQ_WAVE_CYCLES'])==2
    print(json.dumps({'status':'passed','cpu_fixture_checks':['source isolation','raw instances','last samples','quad-cycle normalization','ns bandwidth units']}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path,nargs='*')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--kernel-regex',default='gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel')
    parser.add_argument('--last',type=int,default=51,help='last N matching dispatches per source/kernel; 0 retains all')
    parser.add_argument('--cu-count',type=int,help='verified active CU count in selected partition')
    parser.add_argument('--simd-count',type=int,help='verified hardware simd_count in selected partition')
    parser.add_argument('--pair-split-k',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test(); return
    if not args.inputs or not args.output: parser.error('inputs and --output are required')
    if args.last<0: parser.error('--last must be >= 0')
    paths=sorted({file.resolve() for path in args.inputs for file in
        ([path] if path.is_file() else path.rglob('*counter_collection.csv'))})
    if not paths: parser.error('no counter CSV files found')
    groups=[g for path in paths for g in parse_file(path,re.compile(args.kernel_regex),args.last,args.cu_count,args.simd_count)]
    if not groups: raise ValueError('No kernel matched; do not interpret an empty collection')
    result={'status':'parsed','timestamp_unit':'ns','selected_last_per_source_kernel':args.last,
        'cu_count':args.cu_count,'simd_count':args.simd_count,
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'groups':groups,'split_k_calls':split_k_pairs(groups) if args.pair_split_k else [],
        'interpretation':'Each PMC subprocess remains separate. Profiled rates are diagnostic; speedups require unprofiled timings. Estimates have the definition/window limitations in counter_evidence.json.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'parsed','sources':len(paths),'kernel_groups':len(groups),'output':str(args.output)}))

if __name__=='__main__':
    main()
