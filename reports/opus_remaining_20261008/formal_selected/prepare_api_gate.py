#!/usr/bin/env python3
"""Prepare finite signed8 official API gates after the complete CPU identity audit.

No build or GPU execution. Covers all26 public parents, changed-entry winner
K groups and selected mechanisms' necessary boundary/control cases. Resolves
every target's exact emitted branch and linked kernel using retained CPU trait
inventory, then includes private candidates only when device identities match.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from audit_integration import (BASE, COMMON, HERE, ROOT, digest, git, module,
                               new_write, only_rows, read, require, sha)

INVENTORY = ROOT/'reports/opus_bound_analysis_20261007/shape_inventory.json'
TRAITS = ROOT/'reports/opus_bound_analysis_20261007/shape_inventory_evidence/traits_probe.json'
REMAINING = HERE.parent/'inventory.json'
RUNNER = ROOT/'reports/opus_bound_analysis_20261007/official_smoke.py'
EXPERIMENT = ROOT/'reports/opus_bound_analysis_20261007/experiment_runner.py'
LAUNCHER = ROOT/'reports/opus_bound_analysis_20261007/owned_python_launch.py'


def canonical(s):
    s = re.sub(r'\s+','',s).replace('true','1').replace('false','0')
    return re.sub(r'u(?=[,>])','',s)


def registry(source_root):
    return module('remaining_api_scalar_registry',source_root/'csrc/opus_gemm/opus_gemm_common.py')


def branch_identity(kid,shape,inventory,traits,public,eval_module):
    parent = inventory['parents'][str(kid)]
    matches = [b for b in parent['branches'] if all(eval_module.cpp_eval(c,*shape) for c in b['conditions'])]
    require(len(matches)==1,'Ambiguous generated branch for API target')
    branch = matches[0]
    trait = traits['traits'][branch['traits_cpp']]
    canon = canonical(trait['canonical_type'])
    rows = [v['candidate'] for v in public[kid]['variants'] if branch['kernel_function']+'<' in v['candidate']['demangled']
            and canon in canonical(v['candidate']['demangled'])]
    require(len(rows)==1,'Exact branch trait not unique in parent linked/device inventory')
    return branch,trait,rows[0]


def current_traits(inventory,retained,source_root):
    """Resolve simple aliases from current frozen source, preserving Oct7 reuse."""
    result = {'traits':{name:dict(value) for name,value in retained['traits'].items()}}
    headers = {h for p in inventory['parents'].values() for h in p['traits_headers']}
    aliases = {}
    for header in headers:
        source = (source_root/'csrc/opus_gemm/include'/header).read_text()
        aliases.update({m[1]:m[2].strip() for m in re.finditer(r'using\s+(\w+)\s*=\s*([^;]+);',source)})
    corrections = []
    for name,value in result['traits'].items():
        if name not in aliases:
            continue
        actual = aliases[name]
        # A direct fully numeric base template needs no host compiler probe.
        # All other aliases retain the exact saved canonical CPU type.
        if not re.fullmatch(r'\w+<\s*(?:-?\d+|true|false)(?:\s*,\s*(?:-?\d+|true|false))*\s*>',actual):
            continue
        # Canonical saved types expand default template arguments. Only a
        # direct alias with the same complete arity can replace that type.
        actual_match = re.fullmatch(r'(\w+)<(.*)>',actual)
        saved_match = re.fullmatch(r'(\w+)<(.*)>',value['canonical_type'])
        if not saved_match or actual_match[1]!=saved_match[1] or len(actual_match[2].split(','))!=len(saved_match[2].split(',')):
            continue
        if canonical(actual)!=canonical(value['canonical_type']):
            corrections.append({'alias':name,'retained_canonical_type':value['canonical_type'],
                                'current_source_canonical_type':actual,
                                'source_resolved_without_compiler':True})
            value['canonical_type'] = actual
    return result,corrections


def candidates(selected):
    result = {}
    for name,item in selected.items():
        kid = item['selection']['parent_id']
        result.setdefault(kid,{})[name] = item['selection']
    return result


def prepare(identity_path,output_dir):
    identity = read(identity_path)
    require(identity['status']=='passed_pending_official_API_no_adoption' and not identity['failures'],
            'Complete CPU integration identity must pass before API queue preparation')
    source_root = Path(identity['candidate_source_root'])
    binary = Path(identity['candidate_binary'])
    require(sha(binary)==identity['candidate_binary_sha256'],'Official module changed after CPU identity')
    selection = identity['selection']
    inventory,retained_traits,remaining = read(INVENTORY),read(TRAITS),read(REMAINING)
    traits,trait_corrections = current_traits(inventory,retained_traits,source_root)
    registry_path = source_root/'csrc/opus_gemm/opus_gemm_common.py'
    codegen_path = source_root/'csrc/opus_gemm/codegen/gen_instances_gfx950.py'
    require(inventory['registry_sha256']==sha(registry_path) and inventory['codegen_sha256']==sha(codegen_path),
            'Retained scalar dispatch inventory no longer matches candidate')
    # The retained GPU runner explicitly adds the production checkout to
    # sys.path. Its Python API/JIT/registry/reference inputs must therefore be
    # byte-identical to the candidate worktree; kernels load the recorded SO.
    python_import_inputs = ['aiter/jit/core.py','aiter/test_common.py',
                           'aiter/ops/shuffle.py','csrc/opus_gemm/opus_gemm_common.py',
                           'csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py']
    python_import_inputs += [str(p.relative_to(ROOT)) for p in (ROOT/'aiter/ops/opus').glob('*.py')]
    for rel in python_import_inputs:
        require(sha(ROOT/rel)==sha(source_root/rel),'GPU runner Python input differs from candidate: '+rel)
    scalar = registry(source_root)
    require(len(scalar.A8W8_BPRESHUFFLE_TUNING_KIDS)==26,'Public parent pool differs')
    eval_module = module('remaining_api_cpp_eval',ROOT/'reports/opus_bound_analysis_20261007/shape_inventory.py')
    public = {p['parent_id']:p for p in identity['parents']}
    selected = identity['selected_devices']
    selected_bykid = candidates(selected)
    rows_byname = {e['symbol']:e for e in remaining['entries']}
    requested = []
    seen = {}

    def add(kid,shape,purpose,expected=None):
        key = (kid,*shape)
        if key in seen:
            old = seen[key]
            old['purposes'].append(purpose)
            if expected:
                old['expected_symbols'].add(expected)
            return
        item = {'kid':kid,'shape':shape,'purposes':[purpose],'expected_symbols':{expected} if expected else set()}
        seen[key] = item
        requested.append(item)

    # One supported representative per public API entry; all reducers are
    # checked through their parent split-K call and complete CPU identity.
    for kid in scalar.A8W8_BPRESHUFFLE_TUNING_KIDS:
        r = next((r for r in inventory['rows'] if int(r['parent_id'])==kid),None)
        require(r is not None,'No retained public-parent representative')
        add(kid,[int(r[k]) for k in ('M','N','K')],'one supported representative for each26-parent official API')
    for name,item in selected.items():
        kid = item['selection']['parent_id']
        entry = rows_byname.get(name)
        require(entry,'Changed symbol has no exact remaining actual-winner inventory')
        shapes = entry['winner_shapes']
        require(shapes,'Changed symbol has no retained winner shape')
        add(kid,entry['representative'],'selected exact-entry representative',name)
        for k in sorted({shape[2] for shape in shapes}):
            add(kid,next(shape for shape in shapes if shape[2]==k),'one actual winner for each selected-entry K group',name)
    changed_parents = set(selected_bykid)
    extras = {
        9020:[([1536,7168,384],'fixed384 actual winner'),([16,256,384],'fixed384 minimum aligned M tail'),
              ([1536,7168,16384],'unchanged9020 runtime long-K control'),
              ([8192,768,7168],'unchanged9020 specialization control')],
        9023:[([544,7168,7168],'actual winner with32-row tail'),([544,7168,16384],'accepted long-K winner finite cost'),
              ([576,7168,7168],'full64-row actual winner'),([576,7168,16384],'full-row long-K actual winner'),
              ([16,128,384],'accepted M16 nonwinner default/partial boundary'),
              ([17,128,384],'partial byte helper rows17'),([64,128,4096],'exact32-panel boundary'),
              ([64,128,4224],'33-tile runtime scale refill'),([1024,128,7168],'unchanged9023 fixed entry control')],
        9024:[([1024,768,7168],'fixed7168 first actual winner'),([1344,768,7168],'grouped-M final1-tile tail'),
              ([1408,768,7168],'grouped-M final2-tile tail'),([1536,768,7168],'fixed7168 last actual winner'),
              ([64,128,16384],'unchanged9024 runtime long-K control')],
        9022:[([16,128,128],'minimum aligned M one tile/default high scales'),
              ([16,128,256],'aligned M16 tail/two tiles'),
              ([32,128,384],'aligned M32 tail/three tiles'),([160,128,4096],'exact32-tile panel'),
              ([160,128,4224],'33-tile scale refill'),([800,7168,16384],'long-K actual winner boundary')],
        9030:[([65536,16384,384],'legal greater-than2GB output odd tile terminal guard'),
              ([65536,16384,256],'legal greater-than2GB output even tile terminal guard'),
              ([65536,16384,1536],'actual winner large-output API/aliasC guard'),
              ([65536,16384,16384],'maximum K large-output bounded reference')],
        9051:[([16,128,128],'minimum one-tile register queue'),([16,128,256],'two-tile register queue'),
              ([16,128,384],'three-tile register queue'),([16,128,4096],'interior register queue'),
              ([16,128,16384],'maxK register queue')],
    }
    for kid in sorted(changed_parents):
        for shape,purpose in extras.get(kid,[]):
            add(kid,shape,purpose)
    for item in selection.get('gate_targets',[]):
        add(item['kid'],item['shape'],item['purpose'],item.get('symbol'))
    private_cache = {}
    targets = []
    checks = []
    for index,item in enumerate(requested):
        kid,shape = item['kid'],item['shape']
        instance = scalar.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        require(scalar.a8w8_mxscale_bpreshuffle_supports_shape(instance,*shape),
                'Official scalar support rejects API target: '+str((kid,shape)))
        branch,trait,row = branch_identity(kid,shape,inventory,traits,public,eval_module)
        require(not item['expected_symbols'] or item['expected_symbols']=={row['name']},
                'Required target resolves to another device entry')
        privates = {}
        private_checks = []
        if row['name'] in selected:
            sel = selected[row['name']]['selection']
            path = Path(sel['private_library'])
            require(sha(path)==sel['private_library_sha256'],'Selected private candidate changed')
            if str(path) not in private_cache:
                private_cache[str(path)] = {r['name']:r for r in only_rows(path)}
            require(row['name'] in private_cache[str(path)] and
                    COMMON.identity(row,private_cache[str(path)][row['name']])['matches'],
                    'API private label lacks exact official candidate identity')
            label = 'remaining_kid'+str(kid)
            privates[label] = str(path)
            private_checks.append({'label':label,'path':str(path),'sha256':sha(path),'exact_device_identity':True})
        purpose = '; '.join(dict.fromkeys(item['purposes']))
        targets.append({'kid':kid,'shape':shape,'seed':17,'signed':True,'purpose':purpose,
                        'private_baselines':privates})
        split = instance.bpreshuffle_split_k
        checks.append({'target_index':index,'parent_id':kid,'shape':shape,'support_contract_passed':True,
                       'actual_branch_index':branch['index'],'actual_conditions':branch['conditions'],
                       'actual_traits':branch['traits_cpp'],'canonical_traits':trait['canonical_type'],
                       'actual_candidate_symbol':row['name'],'changed':row['name'] in selected,
                       'split_k':split,'workspace_elements':split*shape[0]*shape[1] if split>1 else 0,
                       'private_checks':private_checks,'reference_chunk_rows':256 if shape[0]*shape[1]>32*1024*1024 else None})
    require({c['parent_id'] for c in checks}==set(scalar.A8W8_BPRESHUFFLE_TUNING_KIDS),
            'API gate does not cover26 public parents')
    require({c['actual_candidate_symbol'] for c in checks if c['changed']}==set(selected),
            'API gate misses a changed exact entry')
    out = Path(output_dir).resolve()
    require(out.is_relative_to(HERE),'API gate artifacts must use new remaining/formal_selected path')
    out.mkdir(parents=True,exist_ok=True)
    plan_path,queue_path = out/'official_api_plan.json',out/'official_api_queue.json'
    plan = {'status':'prepared_CPU_identity_passed_pending_official_API','source_head':identity['source_head'],
            'official_binary':str(binary),'official_binary_sha256':sha(binary),'current_parent_count':26,
            'identity_audit_path':str(identity_path),'identity_audit_sha256':sha(identity_path),
            'registry_sha256':sha(registry_path),'codegen_sha256':sha(codegen_path),
            'runner_sha256':sha(RUNNER),'experiment_runner_sha256':sha(EXPERIMENT),
            'owner_launcher_sha256':sha(LAUNCHER),'check_only':True,'repetitions_per_label_and_target':8,
            'offset_contract':COMMON.offset_contract(),'targets':targets}
    env = {'AITER_JIT_DIR':str(binary.parent),'AITER_AOT_IMPORT':'1','GPU_ARCHS':'gfx950','CU_NUM':'256',
           'HIP_CLANG_PATH':'/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin',
           'OPUS_HIP_CLANG_PATH':'/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin',
           'LD_LIBRARY_PATH':'/opt/rocm/lib:/usr/local/lib'}
    commands = []
    for index,target in enumerate(targets):
        name = f'official_remaining_{index}_kid{target["kid"]}'
        output = out/'api_results'/(name+'.json')
        commands.append({'name':name,'argv':['/opt/venv/bin/python3',str(RUNNER),'--plan',str(plan_path),
                            '--output',str(output),'--target-index',str(index),'--check-only','--repetitions','8'],
                         'log':str(output.with_suffix('.log'))})
    queue = {'status':'prepared_CPU_identity_passed_pending_root_owned_GPU',
             'ready_for_gpu_execution':True,'env':env,'commands':commands,
             'claim_log':str(out/'official_api_claim.jsonl'),'queue_log':str(out/'official_api_queue.log'),
             'execution_prerequisites':['Root physical idle-device lock with unique owner handshake.',
                                        'Retained runner Python inputs match candidate; loaded official SO SHA checked at runtime.']}
    new_write(plan_path,plan)
    new_write(queue_path,queue)
    report = {'status':'passed_CPU_API_plan_identity_pending_GPU','generated_utc':datetime.now(timezone.utc).isoformat(),
              'cpu_only':True,'new_GPU_execution':False,'new_build_execution':False,'production_modified':False,
              'identity_audit_sha256':sha(identity_path),'plan_path':str(plan_path),'plan_sha256':sha(plan_path),
              'queue_path':str(queue_path),'queue_sha256':sha(queue_path),'target_count':len(targets),
              'public_parent_count':26,'changed_entries_covered':len(selected),'repetitions_per_label_and_target':8,
              'rows':checks,'evidence':{str(p):sha(p) for p in (INVENTORY,TRAITS,REMAINING,RUNNER,EXPERIMENT,LAUNCHER)},
              'current_source_trait_alias_corrections':trait_corrections,
              'retained_runner_python_inputs_exact_candidate':{rel:sha(source_root/rel) for rel in python_import_inputs},
              'limits':['This prepares correctness/API gates; no timing claim or support-domain exhaustive test.',
                        'Reference uses independent FP32 bounds; large output reference is bounded to256-row chunks.',
                        'Output view offset256B and guards are retained; no input-offset sweep.',
                        'Every selected private label is attached only after exact emitted branch and device identity checks.']}
    new_write(out/'api_plan_cpu_audit.json',report)
    require('torch' not in sys.modules,'API preparation imported torch')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity',type=Path,default=HERE/'identity_audit.json')
    parser.add_argument('--output-dir',type=Path,default=HERE/'api_gate')
    args = parser.parse_args()
    report = prepare(args.identity,args.output_dir)
    print(json.dumps({'status':report['status'],'target_count':report['target_count'],
                      'plan':report['plan_path'],'queue':report['queue_path']}))


if __name__ == '__main__':
    main()
