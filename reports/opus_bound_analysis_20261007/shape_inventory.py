#!/usr/bin/env python3
"""CPU-only current codegen/traits inventory for the historical 745 winners.

Uses the actual scalar registry, emitted launch branches, host-only compiled
traits, and existing official ELF metadata. No torch import or HIP runtime call.
"""
import argparse
import ast
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
COMMON=ROOT/'csrc/opus_gemm/opus_gemm_common.py'
CODEGEN=ROOT/'csrc/opus_gemm/codegen/gen_instances_gfx950.py'
GENERATED=OUT/'jit_baseline/build/module_deepgemm_opus/blob.staging/impl'
EVIDENCE=OUT/'shape_inventory_evidence'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load_cpu_registry():
    sys.path.insert(0,str(ROOT/'csrc/opus_gemm'))
    import opus_gemm_common
    from codegen import gen_instances_gfx950
    if 'torch' in sys.modules: raise RuntimeError('Inventory must not import torch')
    return opus_gemm_common,gen_instances_gfx950

def balanced(source,start,left='(',right=')'):
    assert source[start]==left
    depth=1;index=start+1
    while depth:
        if source[index]==left: depth+=1
        elif source[index]==right: depth-=1
        index+=1
    return source[start+1:index-1],index

def branches(source,path=()):
    """Read generated scalar if/else launch tree, retaining every path guard."""
    source=source.strip()
    if source.startswith('if'):
        begin=source.index('(')
        condition,end=balanced(source,begin)
        body,end=balanced(source,source.index('{',end),'{','}')
        result=branches(body,path+(condition,))
        remainder=source[end:].strip()
        if remainder.startswith('else'):
            remainder=remainder[4:].strip()
            if remainder.startswith('{'): remainder,_=balanced(remainder,0,'{','}')
            result+=branches(remainder,path+(f'!({condition})',))
        return result
    match=re.search(r'(gemm_a8w8_mxfp8\w*)<\s*(\w+)\s*><<<([\s\S]*?)>>>\(args\)',source)
    if not match: raise ValueError(f'No kernel leaf in {source[:160]}')
    return [{'conditions':list(path),'kernel_function':match[1],'traits_alias':match[2],
             'launch_arguments':match[3].strip()}]

def cpp_eval(expression,m,n,k):
    expression=expression.replace('&&',' and ').replace('||',' or ')
    expression=re.sub(r'!(?!=)',' not ',expression).replace('/','//').strip()
    tree=ast.parse(expression,mode='eval')
    allowed=(ast.Expression,ast.BoolOp,ast.UnaryOp,ast.BinOp,ast.Compare,ast.Name,ast.Load,
        ast.Constant,ast.And,ast.Or,ast.Not,ast.Add,ast.Sub,ast.Mult,ast.FloorDiv,ast.Mod,
        ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.USub,ast.UAdd)
    if any(not isinstance(node,allowed) for node in ast.walk(tree)):
        raise ValueError(f'Unexpected scalar expression {expression}')
    return bool(eval(compile(tree,'cpp-condition','eval'),{'__builtins__':{}},{'m':m,'n':n,'k':k}))

def generate_launches(registry,codegen):
    output={}
    emitted=EVIDENCE/'generated';emitted.mkdir(parents=True,exist_ok=True)
    for kid in sorted(registry.A8W8_BPRESHUFFLE_TUNING_KIDS):
        instance=registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        cg=SimpleNamespace(impl_path=str(emitted),_kid_pipeline_header={},
                           _host_instantiations=[],_device_instantiations=[])
        codegen.gen_mxscale_bpreshuffle_instance(cg,instance,
            codegen.PIPELINE_HEADER_MAP[instance.kernel_tag],codegen.TRAITS_HEADER_MAP[instance.kernel_tag],
            codegen.KERNEL_FUNC_MAP[instance.kernel_tag],codegen.TRAITS_NAME_MAP[instance.kernel_tag],
            codegen.KARGS_NAME_MAP[instance.kernel_tag],lambda text:'',lambda *args:'')
        path=emitted/(instance.name+'.cuh');source=path.read_text()
        official=GENERATED/(instance.name+'.cuh')
        if not official.is_file(): raise ValueError(f'Missing current generated official impl {official}')
        # Preamble callback differs, so compare complete traits/host launch body
        # starting at the architecture header split, not file boilerplate.
        start=source.index('#ifdef OPUS_FUSED_HOST_TU')
        official_source=official.read_text();official_start=official_source.index('#ifdef OPUS_FUSED_HOST_TU')
        if source[start:]!=official_source[official_start:]:
            raise ValueError(f'Current codegen differs from built official launcher {kid}')
        aliases=dict(re.findall(r'using\s+(\w+)\s*=\s*([^;]+);',source))
        tail=source[source.index('const dim3 grid'):]
        tail=tail[tail.index(';')+1:]
        leaves=branches(tail)
        for index,leaf in enumerate(leaves):
            suffix=leaf['traits_alias'].removeprefix(instance.name)
            leaf.update(index=index,traits_cpp=aliases[leaf['traits_alias']],branch_label=suffix)
        headers=re.findall(r'#include "(gfx950/[^\"]+traits[^\"]+)"',source)
        output[kid]={'parent_name':instance.name,'registry':asdict(instance),
            'generated_impl':str(official),'generated_impl_sha256':sha(official),
            'codegen_launch_body_identical':True,'traits_headers':headers,'branches':leaves}
    return output

FIELDS=['B_M','B_N','B_K','BLOCK_SIZE','T_M','T_N','T_K','NUM_WAVES','NUM_STAGES','CLUSTER',
        'OUTPUT','PREFETCH','FIXED_K','SPLIT_K','REDUCE_VEC','REDUCE_BLOCK','STORE_CACHE','SCALE_PANEL',
        'SCALE_PANEL_K_CAPACITY','A_STAGE','B_STAGE','B_GROUPS','REGISTER_SCALES','XOR_LDS','EARLY_SCALE_LOADS',
        'FINE_M_LOADS','LDS_BYTES','WAVE_K','BLOCK_GROUP_M']

def probe_traits(launches,rebuild):
    types=sorted({leaf['traits_cpp'] for parent in launches.values() for leaf in parent['branches']})
    probe=EVIDENCE/'traits_probe.hip';exe=EVIDENCE/'traits_probe'
    headers=sorted({header for parent in launches.values() for header in parent['traits_headers']})
    lines=['#include <cstdio>','#include <typeinfo>']+[f'#include "{h}"' for h in headers]
    for field in FIELDS:
        lines.append(f'template<class T> long long get_{field}() {{ if constexpr(requires{{T::{field};}}) return T::{field}; else return -1; }}')
    lines.append('template<class T> void show(int id) { std::printf("%d",id);')
    lines += [f'std::printf(",%lld",get_{field}<T>());' for field in FIELDS]
    lines.append('std::printf("|%s\\n",typeid(T).name()); }\nint main() {')
    lines += [f'show<{name}>({index});' for index,name in enumerate(types)]
    lines.append('}')
    source='\n'.join(lines)+'\n'
    changed=not probe.exists() or probe.read_text()!=source
    probe.write_text(source)
    compiler='/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++'
    host_object=EVIDENCE/'traits_probe_host.o'
    command=[compiler,'-x','hip','--offload-host-only','--offload-arch=gfx950','--rocm-path=/opt/rocm',
        '-resource-dir=/opt/rocm/lib/llvm/lib/clang/20','-D__HIPCC_RTC__','-std=c++20','-O0',
        '-I'+str(ROOT/'csrc/include'),'-I'+str(ROOT/'csrc/opus_gemm/include'),'-c',str(probe),'-o',str(host_object)]
    link_command=[compiler,str(host_object),'-o',str(exe)]
    if rebuild or changed or not exe.exists():
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        (EVIDENCE/'traits_probe_build.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(f'CPU traits host-only compile failed; see {EVIDENCE}/traits_probe_build.log')
        subprocess.run(link_command,cwd=ROOT,check=True)
    dependencies=subprocess.check_output(['ldd',str(exe)],text=True)
    if any(name in dependencies for name in ['libamdhip64','libhsa-runtime']):
        raise ValueError('Traits probe must link as a plain host executable without HIP/HSA libraries; rerun --rebuild-probe')
    rows=subprocess.check_output([str(exe)],text=True).splitlines()
    values={}
    for row in rows:
        numeric,mangled=row.split('|',1)
        numbers=[int(value) for value in numeric.split(',')]
        values[types[numbers[0]]]={field:(None if value==-1 else value) for field,value in zip(FIELDS,numbers[1:])}
        values[types[numbers[0]]]['canonical_type']=subprocess.check_output(['c++filt','-t',mangled],text=True).strip()
    if len(values)!=len(types): raise ValueError('Traits probe missing types')
    manifest={'status':'host_only_cpu_passed_no_hip_runtime_calls','command':command,'link_command':link_command,
        'linked_dependencies':dependencies,
        'compiler_sha256':sha(Path(compiler)),'probe_sha256':sha(probe),'executable_sha256':sha(exe),
        'traits':values}
    (EVIDENCE/'traits_probe.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return values

def canonical(value):
    value=re.sub(r'\s+','',value)
    value=value.replace('true','1').replace('false','0')
    return re.sub(r'u(?=[,>])','',value)

def resources_for_parents(launches):
    spec=importlib.util.spec_from_file_location('opus_inventory_elf',OUT/'audit_current_compute_metadata.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result={}
    build=OUT/'jit_baseline/build/module_deepgemm_opus/build'
    for kid,parent in launches.items():
        object_path=build/(parent['parent_name']+'_Cbf16_t.device.cuda.o')
        if not object_path.exists(): raise ValueError(f'No current official object {object_path}')
        image=module.device_bundle(object_path)
        metadata,rows=module.summarize_image(image)
        result[kid]={'object':str(object_path),'object_sha256':sha(object_path),'variants':rows}
    (EVIDENCE/'official_resources.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def match_resource(leaf,values,resources):
    # RTTI from the host-only traits probe expands every default and alias.
    expression=leaf['kernel_function']+'<'+values['canonical_type']+'>'
    for row in resources['variants']:
        demangle=row['demangled']
        if canonical(expression) in canonical(demangle): return row
    return None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rebuild-probe',action='store_true');args=parser.parse_args()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    registry,codegen=load_cpu_registry()
    launches=generate_launches(registry,codegen)
    traits=probe_traits(launches,args.rebuild_probe)
    resources=resources_for_parents(launches)
    table=ROOT/'reports/opus_current745_tables_20260930/shape_comparison_745.csv'
    history=list(csv.DictReader(table.open(encoding='utf-8-sig')))
    rows=[]
    for original in history:
        kid=int(original['opus_kernelId']);m,n,k=(int(original[name]) for name in ['M','N','K'])
        parent=launches[kid]
        candidates=[leaf for leaf in parent['branches'] if all(cpp_eval(condition,m,n,k) for condition in leaf['conditions'])]
        if len(candidates)!=1: raise ValueError(f'Expected one actual launch branch {kid} {(m,n,k)}, found {len(candidates)}')
        leaf=candidates[0];values=traits[leaf['traits_cpp']]
        parent_instance=registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        if not registry.a8w8_mxscale_bpreshuffle_supports_shape(parent_instance,m,n,k):
            raise ValueError(f'Historical chosen parent outside current support domain {kid} {(m,n,k)}')
        bm,bn,bk,block=(values[name] for name in ['B_M','B_N','B_K','BLOCK_SIZE'])
        waves=block//64;wavek=values['WAVE_K'] or values['T_K'] or 1;split=values['SPLIT_K'] or 1
        gm=(m+bm-1)//bm;gn=(n+bn-1)//bn
        leaf_fixed=values['FIXED_K'] or 0
        if leaf_fixed and leaf_fixed!=k: raise ValueError('Selected fixed K does not match shape')
        if 'lds_bytes(k)' in leaf['launch_arguments']:
            loops=(k//bk+split-1)//split
            active_stages=min(loops,values['NUM_STAGES'])
            dynamic_lds=active_stages*(values['A_STAGE']+values['B_STAGE'])
            if not values['REGISTER_SCALES']: dynamic_lds+=(bm+values['B_GROUPS'])*loops
        else: dynamic_lds=0;active_stages=None
        resource=match_resource(leaf,values,resources[kid])
        meta=resource['metadata'] if resource else {}
        scalar_fields={name:meta.get('.'+name) for name in ['vgpr_count','agpr_count','sgpr_count',
            'vgpr_spill_count','sgpr_spill_count','group_segment_fixed_size','private_segment_fixed_size']}
        logical_ab=(m+n)*k;scales=m*(k//128)+(n//128)*(k//128);output=2*m*n
        workspace=4*split*m*n if split>1 else 0
        workspace_traffic=2*workspace
        useful=2*m*n*k;ideal_bytes=logical_ab+scales+output
        logical_with_ws=ideal_bytes+workspace_traffic
        repeated_full=gm*gn*(bm+bn)*k
        repeated_valid=(gn*m+gm*n)*k
        total_tiles=k//128
        split_tilecounts=[total_tiles//split+(s<total_tiles%split) for s in range(split)]
        wave_tilecounts=[[count//wavek+(w<count%wavek) for w in range(wavek)] for count in split_tilecounts]
        choice=re.search(r'_Choice([0-9]+)',leaf['traits_alias'])
        actual_kid=(parent['registry']['bpreshuffle_dispatch'][int(choice[1])][1] if choice else kid)
        is_register=leaf['kernel_function']=='gemm_a8w8_mxfp8_scale_small_register_kernel'
        is_dynamic='lds_bytes(k)' in leaf['launch_arguments']
        cluster=values['CLUSTER']
        padded_flops=2*gm*bm*gn*bn*k
        row={'M':m,'N':n,'K':k,'parent_id':kid,'actual_branch_index':leaf['index'],
            'actual_branch_label':leaf['branch_label'],
            'actual_configuration_id':actual_kid,'actual_conditions':leaf['conditions'],
            'actual_traits':leaf['traits_cpp'],'kernel_function':leaf['kernel_function'],
            'canonical_traits_type':values['canonical_type'],
            'BM':bm,'BN':bn,'BK':bk,'waves':waves,'wave_k':wavek,'split_k':split,
            'stages':values['NUM_STAGES'] if values['NUM_STAGES'] is not None else (2 if kid in [9000,9010] else None),
            'matrix_ring':not is_register,'cluster':cluster,'cluster_source':'traits' if values['CLUSTER'] is not None else ('not_applicable' if is_register else 'no CLUSTER parameter'),
            'register_prefetch':values['PREFETCH'],'fixed_k':leaf_fixed,
            'scale_panel_k_tiles':values['SCALE_PANEL'] or values['SCALE_PANEL_K_CAPACITY'],
            'dynamic_lds_bytes':dynamic_lds,'active_matrix_lds_stages':active_stages,
            'static_lds_bytes_metadata':meta.get('.group_segment_fixed_size'),
            'total_lds_bytes':None if not meta else dynamic_lds+meta['.group_segment_fixed_size'],
            'producer_grid':[gn,gm,split],'producer_workgroups':gn*gm*split,
            'producer_waves_launched':gn*gm*split*waves,'tile_efficiency':m*n/(gm*bm*gn*bn),
            'k128_tiles_per_wave_partition_by_global_split':wave_tilecounts,'k128_tiles_per_global_split':split_tilecounts,
            'reduce_vec':values['REDUCE_VEC'] if split>1 else None,'reduce_block':values['REDUCE_BLOCK'] if split>1 else None,
            'reducer_grid':[(m*n+values['REDUCE_VEC']*values['REDUCE_BLOCK']-1)//(values['REDUCE_VEC']*values['REDUCE_BLOCK']),1,1] if split>1 else None,
            'useful_flops':useful,'padded_tile_flops_estimate':padded_flops,
            'padded_mfma_instruction_count_estimate':padded_flops//65536,
            'dynamic_matrix_ring_maximum_stages':None if not is_dynamic else values['NUM_STAGES'],
            'logical_ab_bytes':logical_ab,'logical_scale_bytes':scales,'logical_output_bytes':output,
            'ideal_logical_bytes_including_scales':ideal_bytes,'workspace_capacity_bytes':workspace,
            'workspace_write_read_bytes':workspace_traffic,'ideal_logical_bytes_including_workspace_traffic':logical_with_ws,
            'logical_ai_flop_per_byte':useful/ideal_bytes,'logical_ai_with_workspace_flop_per_byte':useful/logical_with_ws,
            'approx_repeated_ab_full_tiles_bytes':repeated_full,'approx_repeated_ab_valid_extents_bytes':repeated_valid,
            'approx_ab_request_lower_valid_extent_bytes':repeated_valid,
            'approx_register_wave_ab_bytes_before_oob':(gm*gn*k*(bm*values['T_N']+bn*values['T_M'])) if is_register else None,
            'approx_tile_ab_ai_flop_per_byte':useful/repeated_full,
            'historical_winner_us':float(original['opus_us']),'historical_source_run':original['source_run'],
            'historical_useful_tflops':useful/(float(original['opus_us'])*1e6),
            'historical_logical_tb_s_including_scales':ideal_bytes/(float(original['opus_us'])*1e6),
            'metadata_match':'exact_official' if resource else 'unknown',**scalar_fields,
            'official_kernel_symbol':resource['name'] if resource else None,
            'official_instruction_sha256':resource['instruction_sha256'] if resource else None}
        rows.append(row)
    if len(rows)!=745 or len({(r['M'],r['N'],r['K']) for r in rows})!=745: raise ValueError('Expected745 distinct shapes')
    summary={'rows':len(rows),'parents':len({r['parent_id'] for r in rows}),
        'exact_official_resources':sum(r['metadata_match']=='exact_official' for r in rows),
        'unknown_resources':sum(r['metadata_match']=='unknown' for r in rows),
        'source_runs':{run:sum(r['historical_source_run']==run for r in rows) for run in sorted({r['historical_source_run'] for r in rows})}}
    summary['actual_configurations']={str(kid):sum(row['actual_configuration_id']==kid for row in rows)
                                      for kid in sorted({row['actual_configuration_id'] for row in rows})}
    if any(row['total_lds_bytes'] is not None and row['total_lds_bytes']>163840 for row in rows):
        raise ValueError('Inventory total LDS exceeds current traits contract')
    result={'status':'cpu_current_codegen_traits_inventory','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'gpu_imported_or_initialized':False,'registry_sha256':sha(COMMON),'codegen_sha256':sha(CODEGEN),
        'history_table_sha256':sha(table),'summary':summary,'parents':launches,'rows':rows,
        'units':{'time':'us historical kernel timings','FLOPs':'logical2MNK','bytes':'logical model estimates, not HBM counter traffic'},
        'limits':['Only historical chosen parents are inventoried for each shape, not every candidate alternative.',
            'Logical AB/scales/output minima assume one copy; workspace traffic models mainwrite+reduceread.',
            'Repeated AB budgets count cross-WG tile duplication; not per-wave register-path duplication, line amplification, hits, or physical HBM.',
            'DynamicLDS uses current traits launcher ceil-split allocation; actual individual split loops may be smaller.',
            'Metadata from current official device ELF. VGPR_count is total gfx950 vector allocation; do not add AGPR again.',
            'Kernel occupancy and bottleneck classification are unknown without GPU measurement.']}
    (OUT/'shape_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'shape_inventory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows: writer.writerow({name:json.dumps(value) if isinstance(value,(list,dict)) else value for name,value in row.items()})
    print(json.dumps(summary))

if __name__=='__main__': main()
