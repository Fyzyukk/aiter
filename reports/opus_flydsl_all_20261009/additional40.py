#!/usr/bin/env python3
"""Historical fastest-legal-OPUS comparison and existing candidate guards, CPU only."""
import ast,collections,csv,dataclasses,hashlib,json,math,os,re,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HIST=ROOT/'reports/opus_flydsl_resume_20261009/historical_745_best_per_shape.csv'
PROFILE=ROOT/'reports/opus_clang23_mixed_retune_20261008/profile.csv'
OLD=ROOT/'reports/opus_flydsl_gap_20261009/losers294.csv'
COMMON=ROOT/'csrc/opus_gemm/opus_gemm_common.py'
CODEGEN=ROOT/'csrc/opus_gemm/codegen/gen_instances_gfx950.py'
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
HYBRID_IDS=[9043,9044,9045,9046,9047,9049,9055,9056]
SMALL_IDS=[110,111,120,121,130,140,210,211,220,221,230,231]
REG_PARENTS={9040,9041,9042,9051,9052,9053,9054}
FINE_PARENTS={9060,9061,9062,9063}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def key(r):return tuple(int(r[x]) for x in ['M','N','K'])
def snapshot():
    return {str(p.relative_to(HERE)):sha(p) for pkg in ['hybrid_small','small_split','main_variants','large9030']
            for p in sorted((HERE/pkg).rglob('*')) if p.is_file()}

def registry(source):
    tree=ast.parse(source)
    names={'OpusGemmInstance','_a8w8_mxscale_gemm_bpreshuffle','_a8w8_mxscale_gemm_bpreshuffle_merged',
           '_a8w8_mxscale_gemm_bpreshuffle_small','_a8w8_mxscale_gemm_bpreshuffle_fine',
           '_a8w8_mxscale_gemm_bpreshuffle_register_tail','a8w8_mxscale_bpreshuffle_supports_shape'}
    assigns={'a8w8_mxscale_gemm_bpreshuffle_kernels_list','_bpreshuffle_dispatch','_BMM_M_ALIGN_TILES'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names or
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assigns for t in n.targets)]
    g={'dataclass':dataclasses.dataclass,'field':dataclasses.field}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_scalar_registry','exec'),g)
    return g

def condition(cpp,m,n,k):
    # All dimensions are positive, so Python // matches C++ integer division.
    expression=cpp.replace('&&',' and ').replace('||',' or ')
    expression=re.sub(r'!(?!=)',' not ',expression).replace('/','//')
    return bool(eval(expression.strip(),{'__builtins__':{}},{'m':m,'n':n,'k':k}))

def main():
    out=HERE/'additional40_cpu_audit_v2'
    if out.exists() or (HERE/'additional40.csv').exists() or (HERE/'additional40_audit.json').exists():
        raise SystemExit('Refusing to overwrite additional40 evidence')
    before=snapshot();out.mkdir()
    controls=json.loads((HERE/'hybrid_small/source_manifest.json').read_text())['controls']
    for path in [COMMON,CODEGEN]:
        expected=next(r['sha256'] for r in controls if r['path']==str(path))
        assert sha(path)==expected,'Dispatch differs from frozen hybrid control'
        (out/path.name).write_bytes(path.read_bytes())
    g=registry((out/'opus_gemm_common.py').read_text())
    reg=g['a8w8_mxscale_gemm_bpreshuffle_kernels_list'];supports=g['a8w8_mxscale_bpreshuffle_supports_shape']
    dispatch=g['_bpreshuffle_dispatch']
    hist=read(HIST);profile=read(PROFILE);old=read(OLD)
    assert len(hist)==745 and len({key(r) for r in hist})==745 and len(old)==294
    legal=[];best={};counts=collections.Counter()
    for line,r in enumerate(profile,2):
        if r['libtype']!='opus':continue
        counts['opus_rows']+=1
        m,n,k=key(r);kid=int(r['kernelId']);lat=float(r['us']);err=float(r['errRatio'])
        if not math.isfinite(lat) or lat<=0 or not math.isfinite(err) or err!=0:
            counts['bad_time_or_error']+=1;continue
        if int(r['splitK'])!=0 or not supports(reg[kid],m,n,k):
            counts['illegal']+=1;continue
        row={**r,'line':line};legal.append(row)
        if (m,n,k) not in best or (lat,kid,line)<(float(best[m,n,k]['us']),int(best[m,n,k]['kernelId']),best[m,n,k]['line']):
            best[m,n,k]=row
    assert counts['opus_rows']==len(legal)==13027 and counts['illegal']==0
    assert len({(key(r),int(r['kernelId']),int(r['splitK'])) for r in legal})==len(legal)
    assert all(key(h) in best for h in hist)
    losing={key(h) for h in hist if float(best[key(h)]['us'])>float(h['flydsl_us'])}
    oldkeys={key(r) for r in old};assert len(losing)==334 and oldkeys<=losing
    extra=sorted([h for h in hist if key(h) in losing-oldkeys],key=key)
    assert len(extra)==40 and all(h['selected_backend']!='opus' for h in extra)
    assert {key(h) for h in hist if h['selected_backend']!='opus' and key(h) in losing}=={key(h) for h in extra}
    # Query the actual source contracts in an ordinary CPU executable. No HIP
    # includes, runtime library, pointers dereferenced, allocation, or kernel.
    variants=json.loads((HERE/'main_variants/variants.json').read_text())
    main_candidates=[v for v in variants if 92000<=v['id']<=92012 or 92100<=v['id']<=92120]
    cpp=['#include <iostream>','#include <cstdint>',
         '#include "hybrid_small/contract.h"','#include "small_split/contract.h"',
         '#include "main_variants/contract.h"','#include "large9030/contract.h"',
         'int main(){int index,m,n,k;while(std::cin>>index>>m>>n>>k){',
         'const void* a=reinterpret_cast<const void*>(uintptr_t(0x100000000));',
         'const void* b=reinterpret_cast<const void*>(uintptr_t(0x200000000));',
         'const void* sa=reinterpret_cast<const void*>(uintptr_t(0x300000000));',
         'const void* sb=reinterpret_cast<const void*>(uintptr_t(0x400000001));',
         'void* c=reinterpret_cast<void*>(uintptr_t(0x500000000));',
         'void* ws=reinterpret_cast<void*>(uintptr_t(0x600000000));']
    for actual in HYBRID_IDS:
        cpp.append(f'std::cout<<index<<" hybrid {actual} "<<(opus_private_small_shape_valid({actual},m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\\n";')
    for candidate in SMALL_IDS:
        cpp.append(f'{{PrivateConfig cfg{{}};bool ok=private_config({candidate},true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small {candidate} "<<ok<<" "<<bytes<<"\\n";}}')
    for v in main_candidates:
        extra_guard=' && m%256==0' if v['pin'] and v['id']!=92120 else ''
        cpp.append(f'std::cout<<index<<" main {v["id"]} "<<(main_shape_valid(m,n,k,{v["tile_N"]},{v["fixed_K"]}){extra_guard}&&main_pointers_valid(a,b,sa,sb,c))<<" 0\\n";')
    cpp+=['std::cout<<index<<" large 0 "<<(private9030_shape_valid(m,n,k,true)&&private9030_pointer_valid(a,b,sa,sb,c))<<" 0\\n";',
          '}}']
    source=out/'contract_check.cpp';source.write_text('\n'.join(cpp)+'\n')
    binary=out/'contract_check';argv=['/opt/rocm-llvm23-46fcb339/bin/clang++','-std=c++20','-O2','-I'+str(HERE),str(source),'-o',str(binary)]
    done=subprocess.run(argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'compile.log').write_text(done.stdout);assert done.returncode==0,done.stdout
    dynamic=subprocess.check_output(['readelf','-d',str(binary)],env=ENV,text=True);(out/'dynamic.txt').write_text(dynamic)
    needed=re.findall(r'Shared library: \[([^\]]+)\]',dynamic);assert all('hip' not in x.lower() and 'hsa' not in x.lower() for x in needed)
    stdin=''.join(f'{i} {h["M"]} {h["N"]} {h["K"]}\n' for i,h in enumerate(extra))
    (out/'shapes.txt').write_text(stdin)
    result=subprocess.check_output([str(binary)],input=stdin,env=ENV,text=True);(out/'contract_results.txt').write_text(result)
    guard={}
    for line in result.splitlines():
        i,pkg,cid,ok,bytes=line.split();guard[int(i),pkg,int(cid)]=(bool(int(ok)),int(bytes))
    assert len(guard)==40*(len(HYBRID_IDS)+len(SMALL_IDS)+len(main_candidates)+1)
    # Bind every mapping to an already built library and audited producer.
    library={};audit={}
    for pkg in ['hybrid_small','small_split','main_variants','large9030']:
        receipt=json.loads((HERE/pkg/'build_receipt.json').read_text());audit[pkg]=json.loads((HERE/pkg/'cpu_audit.json').read_text())
        for b in receipt['builds']:
            assert sha(b['library'])==b['library_sha256']
            library[pkg,b['side']]={'path':str(Path(b['library']).relative_to(ROOT)),'sha256':b['library_sha256']}
    compiled_hybrid={r['side']:{k['actual_kid'] for k in r['kernels']} for r in audit['hybrid_small']['libraries']}
    assert all(set(HYBRID_IDS)==compiled_hybrid[f'direct_b_ahead{ahead}'] for ahead in [1,2,3])
    candidate_library_symbols={r['side']:[k['name'] for k in r['kernels']] for r in audit['main_variants']['libraries']}
    for v in main_candidates:
        trait=v['traits'].replace(' ','')
        if trait.startswith('opus_private_geometry_traits<') and trait.count(',')==2:
            trait=trait[:-1]+',0>'
        assert any(trait in n.replace(' ','') for n in candidate_library_symbols['pin24' if v['pin'] else 'main23']),v
    assert len(next(r for r in audit['small_split']['libraries'] if r['side']=='candidate')['kernels'])==14
    rows=[];case_rows=[];catalog={};preferred_count=collections.Counter();all_count=collections.Counter()
    for i,h in enumerate(extra):
        m,n,k=key(h);r=best[m,n,k];parent=int(r['kernelId']);actual=parent;chosen_condition=''
        for cpp_condition,destination in dispatch.get(parent,()):
            if condition(cpp_condition,m,n,k):actual=destination;chosen_condition=cpp_condition;break
        assert supports(reg[parent],m,n,k)
        same_hybrid=actual if actual in HYBRID_IDS else None
        alltokens=[];preferred=[]
        def add(pkg,side,cid,workspace=0,prefer=False):
            token=f'{pkg}/{side}:{cid}';alltokens.append(token)
            if prefer:preferred.append(token);preferred_count[pkg]+=1
            all_count[pkg]+=1
            descriptor={'package':pkg,'side':side,'launch_id':cid,'library':library[pkg,side]['path'],
                        'library_sha256':library[pkg,side]['sha256'],'workspace_bytes':workspace,'kernel_calls':2 if workspace else 1}
            catalog[token]={x:y for x,y in descriptor.items() if x!='workspace_bytes'}
            case_rows.append({'M':m,'N':n,'K':k,'best_opus_parent':parent,'best_opus_actual':actual,'candidate_token':token,
                              'preferred_same_family':prefer,**descriptor,'contract_checked':'CPU actual header and launcher constraints',
                              'numerical_validation':'pending_gpu_stop','performance_validation':'pending_gpu_stop'})
        for cid in HYBRID_IDS:
            ok,bytes=guard[i,'hybrid',cid]
            if ok:
                for ahead in [1,2,3]:add('hybrid_small',f'direct_b_ahead{ahead}',cid,0,cid==same_hybrid)
        for cid in SMALL_IDS:
            ok,bytes=guard[i,'small',cid]
            if ok:add('small_split','candidate',cid,bytes,(parent in REG_PARENTS and cid<200) or (parent in FINE_PARENTS and cid>=200))
        for v in main_candidates:
            ok,bytes=guard[i,'main',v['id']]
            if ok:add('main_variants','pin24' if v['pin'] else 'main23',v['id'],0,parent in v['parents'])
        if guard[i,'large',0][0]:
            for side in ['panel16','directb']:add('large9030',side,9030,0,parent==9030)
        assert preferred and alltokens
        op=float(r['us']);fly=float(h['flydsl_us'])
        rows.append({'M':m,'N':n,'K':k,'selected_backend':h['selected_backend'],'selected_kid':h['selected_kid'],'selected_us':h['selected_us'],
                     'best_legal_opus_parent':parent,'best_legal_opus_actual':actual,'best_legal_opus_us':op,
                     'opus_profile_line':r['line'],'opus_profile_splitK':r['splitK'],'opus_profile_errRatio':r['errRatio'],
                     'opus_profile_kernel_name':r['kernelName'],'opus_profile_legal_candidates':sum(key(t)==(m,n,k) for t in legal),
                     'opus_actual_dispatch_condition':chosen_condition,'opus_actual_family':reg[actual].name_tag,
                     'opus_actual_tile_M':reg[actual].B_M,'opus_actual_tile_N':reg[actual].B_N,
                     'opus_actual_global_splitK':reg[actual].bpreshuffle_split_k,'opus_fixed_K':reg[actual].bpreshuffle_fixed_k or (k if parent==9060 and k in [3072,7168] else 0),
                     'flydsl_kind':h['flydsl_kind'],'flydsl_kid':h['flydsl_kid'],'flydsl_name':h['flydsl_name'],'flydsl_splitK':h['flydsl_splitK'],'flydsl_us':fly,
                     'opus_over_flydsl':op/fly,'slow_pct':100*(op/fly-1),'needed_latency_reduction_pct':100*(1-fly/op),'gap_us':op-fly,
                     'preferred_candidate_tokens':';'.join(preferred),'preferred_candidate_count':len(preferred),
                     'all_contract_legal_candidate_tokens':';'.join(alltokens),'all_contract_legal_candidate_count':len(alltokens),
                     'hybrid_resolved_launch_actual':same_hybrid if same_hybrid else '',
                     'mapping_status':'existing compiled candidates; CPU contracts passed; numerical/timing pending GPU stop'})
    for name,data in [('additional40.csv',rows),(str(out.relative_to(HERE)/'candidate_cases.csv'),case_rows)]:
        with (HERE/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    assert len(rows)==40 and len({key(r) for r in rows})==40
    main576=next(r for r in rows if key(r)==(576,7168,384));assert main576['preferred_candidate_tokens']==';'.join(f'main_variants/main23:{v}' for v in [92000,92001,92002,92003])
    assert before==snapshot(),'Existing package files changed during read-only audit'
    report={'status':'cpu_historical_comparison_and_existing_candidate_contract_audit_passed','cpu_only':True,'gpu_queries':0,'gpu_kernels':0,
            'experiment_libraries_loaded':False,'registered':False,'new_kernels':0,'existing_packages_unchanged':True,
            'historical_shapes':745,'profile_rows':len(profile),'legal_opus_rows':len(legal),'profile_rejections':dict(counts),
            'pure_opus_losers':334,'existing_selected_opus_losers':294,'additional_non_opus_selected_losers':40,
            'exact_difference':'additional40 keys = fastest legal OPUS historical losers334 minus existing losers294; also exactly non-OPUS-selected losers',
            'selected_backend_counts':dict(collections.Counter(r['selected_backend'] for r in rows)),
            'best_opus_parent_counts':dict(sorted(collections.Counter(r['best_legal_opus_parent'] for r in rows).items())),
            'best_opus_actual_counts':dict(sorted(collections.Counter(r['best_legal_opus_actual'] for r in rows).items())),
            'preferred_candidate_cases':sum(preferred_count.values()),'preferred_case_counts_by_package':dict(preferred_count),
            'all_contract_legal_candidate_cases':len(case_rows),'all_legal_case_counts_by_package':dict(all_count),
            'every_shape_has_preferred_compiled_candidate':True,'candidate_catalog':catalog,
            'mapping_method':'preferred candidates retain the best OPUS family; other candidate tokens are explicit cross-family remaps accepted by actual source guards; hybrid public9044 dispatch resolved before private explicit-actual launch',
            'profile_method':'minimum (us,kernelId,CSV line) over positive finite errRatio0, splitK0 OPUS rows accepted by AST-extracted scalar registry supports_shape; all13027 records legal and unique',
            'dispatch_method':'ordered scalar dispatch AST extracted from registry source whose SHA matches frozen hybrid source_manifest control; positive C++ / uses Python //; no repository module import',
            'host_contract_audit':{'compile_argv':argv,'binary_sha256':sha(binary),'source_sha256':sha(source),'needed_libraries':needed,'no_hip_hsa_dependencies':True,'contract_queries':len(guard),
                                   'headers':['hybrid_small/contract.h','small_split/contract.h','main_variants/contract.h','large9030/contract.h'],
                                   'additional_launcher_guards':'main pin92100..92114 requires M%256=0; fixedK/tileN from variants and actual launch source'},
            'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [HIST,PROFILE,OLD,COMMON,CODEGEN]},
            'package_evidence':{pkg:{name:sha(HERE/pkg/name) for name in ['source_manifest.json','build_receipt.json','cpu_audit.json','final_manifest.json']} for pkg in ['hybrid_small','small_split','main_variants','large9030']},
            'package_contract_and_launch_sources':{str(p.relative_to(HERE)):sha(p) for pkg in ['hybrid_small','small_split','main_variants','large9030'] for p in [HERE/pkg/'contract.h',HERE/pkg/'launch.hip']},
            'output_directory':str(out.relative_to(HERE)),
            'outputs':{'additional40.csv':sha(HERE/'additional40.csv'),str(out.relative_to(HERE)/'candidate_cases.csv'):sha(out/'candidate_cases.csv'),str(out.relative_to(HERE)/'contract_results.txt'):sha(out/'contract_results.txt')},
            'source_sha256':sha(HERE/'additional40.py'),'numerical_validation':'not_run_gpu_stopped','performance_validation':'historical_saved_CSV_only_no_new_measurements',
            'limitation':'Recorded timings come from saved runs, not a clean same-GPU comparison; source/CPU legality does not establish GPU numerical correctness or candidate speed.',
            'existing_package_file_count_checked':len(before)}
    (HERE/'additional40_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['status','pure_opus_losers','additional_non_opus_selected_losers','preferred_candidate_cases','preferred_case_counts_by_package','all_contract_legal_candidate_cases','all_legal_case_counts_by_package','existing_packages_unchanged']}))
if __name__=='__main__':main()
