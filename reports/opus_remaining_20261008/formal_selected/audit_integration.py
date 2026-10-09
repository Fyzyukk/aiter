#!/usr/bin/env python3
"""CPU-only audit of an explicitly selected Oct8 remaining-kernel integration.

Reads the existing Oct8 baseline and a root-built official worktree. Compares
all public parents, all compilation objects and every linked gfx950 bundle.
Only selected symbols may differ and must match their exact tested private
candidate. Writes new records only; never merges, builds, or runs HIP.
"""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/'reports/opus_resume_20261008/formal_selected'
MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
TARGET = 'hip-amdgcn-amd-amdhsa--gfx950'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    require('torch' not in sys.modules, 'CPU audit imported torch')
    return result


COMMON = module('remaining_formal_ELF',BASE/'smoke_common.py')


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root)


def new_write(path,value):
    path = Path(path).resolve()
    require(path.is_relative_to(HERE) and not path.exists(), 'Output must be a new file under remaining/formal_selected')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def bundles(path):
    outer = COMMON.Elf(Path(path).read_bytes())
    if '.hip_fatbin' not in outer.sections:
        return []
    data = outer.bytes(outer.sections['.hip_fatbin'])
    result = []
    cursor = 0
    while True:
        start = data.find(MAGIC,cursor)
        if start < 0:
            break
        require(start+len(MAGIC)+8 <= len(data), 'Truncated HIP bundle')
        count = struct.unpack_from('<Q',data,start+len(MAGIC))[0]
        require(0<count<1024,'Invalid HIP target count')
        offset = start+len(MAGIC)+8
        end = offset
        targets = []
        images = []
        for _ in range(count):
            require(offset+24<=len(data),'Truncated HIP target descriptor')
            begin,size,idlen = struct.unpack_from('<QQQ',data,offset)
            offset += 24
            require(offset+idlen<=len(data) and start+begin+size<=len(data),'Truncated HIP image')
            target = data[offset:offset+idlen].decode()
            offset += idlen
            targets.append(target)
            end = max(end,start+begin+size,offset)
            if target == TARGET:
                image = data[start+begin:start+begin+size]
                rows = COMMON.summarize_image(image)
                require(len({r['name'] for r in rows})==len(rows),'Duplicate device symbols in bundle')
                images.append({'image_sha256':digest(image),'rows':rows})
        require(len(images)==1,'Expected exactly one gfx950 image per bundle')
        result.append({'ordinal':len(result),'section_offset':start,'targets':targets,**images[0]})
        cursor = end
    require(result,'No parsed gfx950 bundles')
    return result


def only_rows(path):
    images = bundles(path)
    require(len(images)==1,'Expected one compilation/private bundle: '+str(path))
    return images[0]['rows']


def verify_sources(selection,baseline_source):
    candidate = Path(selection['source_root']).resolve()
    baseline = Path(baseline_source['candidate_worktree']).resolve()
    require(candidate not in (ROOT,baseline),'Official candidate must be a separate worktree')
    head = git(candidate,'rev-parse','HEAD').decode().strip()
    require(head==baseline_source['source_head']==git(baseline,'rev-parse','HEAD').decode().strip(),
            'Candidate/baseline worktree HEAD differs')
    source_files = selection['source_files']
    selected = {f['path']:f for f in source_files}
    require(len(selected)==len(source_files)>0,'Empty or duplicate source selection')
    inventory = []
    changed = []
    for row in git(candidate,'ls-files','--stage','-z').decode().split('\0'):
        if not row:
            continue
        modeblob,rel = row.split('\t',1)
        if modeblob.split()[0]=='160000' or not rel.startswith(('aiter/','csrc/','3rdparty/')):
            continue
        a,b = baseline/rel,candidate/rel
        require(a.is_file() and b.is_file(),'Missing tracked build input: '+rel)
        ah,bh = sha(a),sha(b)
        if ah!=bh:
            changed.append(rel)
            require(rel in selected,'Unselected build source changed: '+rel)
        else:
            require(rel not in selected,'Selected source did not change: '+rel)
        if rel in selected:
            item = selected[rel]
            tested = Path(item['tested_source'])
            require(bh==item['tested_sha256']==sha(tested),'Candidate source differs from reviewed tested snapshot: '+rel)
            if 'baseline_sha256' in item:
                require(ah==item['baseline_sha256'],'Selected baseline source SHA differs: '+rel)
        inventory.append({'path':rel,'baseline_sha256':ah,'candidate_sha256':bh,'changed':ah!=bh})
    require(set(changed)==set(selected),'Source selection includes missing/non-build files')
    git(candidate,'diff','--check')
    source_diff = git(candidate,'diff','--binary')
    selected_diff = ''.join(''.join(difflib.unified_diff((baseline/rel).read_text().splitlines(True),
                              (candidate/rel).read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
                           for rel in sorted(changed)).encode()
    return candidate,baseline,{'source_head':head,'changed_files_vs_current_Oct8':sorted(changed),
        'all_other_tracked_build_inputs_byte_equal':True,'tracked_build_inputs_checked':len(inventory),
        'build_input_inventory_sha256':digest(json.dumps(inventory,sort_keys=True).encode()),
        'source_diff_sha256':digest(source_diff),'selected_diff_sha256':digest(selected_diff),
        'source_diff':source_diff.decode(),'selected_diff':selected_diff.decode(),
        'source_file_snapshots':source_files,'baseline_source_root':str(baseline),
        'candidate_source_root':str(candidate)}


def selected_devices(selection,public_rows):
    selected = {}
    library_cache = {}
    for item in selection['changed_entries']:
        name = item['symbol']
        require(name in public_rows and public_rows[name]['parent_id']==item['parent_id'],
                'Selected symbol is outside recorded public parent: '+name)
        require(name not in selected and item['parent_id'] not in (9000,9010,9021),
                'Duplicate or frozen selected entry')
        candidates = []
        for side,key in (('baseline','private_baseline_library'),('candidate','private_library')):
            path = Path(item[key]).resolve()
            require(sha(path)==item[key+'_sha256'],'Private selected library SHA differs')
            if str(path) not in library_cache:
                library_cache[str(path)] = {r['name']:r for r in only_rows(path)}
            require(name in library_cache[str(path)],'Selected private symbol missing')
            candidates.append(library_cache[str(path)][name])
        before,after = candidates
        require(COMMON.identity(before,public_rows[name]['row'])['matches'],
                'Selected private baseline is not exact current Oct8: '+name)
        require(not COMMON.identity(before,after)['matches'],'Selected candidate is unchanged: '+name)
        selected[name] = {'selection':item,'baseline':before,'candidate':after}
    require(selected,'No selected device entries')
    return selected


def compare_rows(before,after,selected):
    a,b = {r['name']:r for r in before},{r['name']:r for r in after}
    require(set(a)==set(b),'Device symbol set changed')
    result = []
    for name,old in a.items():
        expected = selected[name]['candidate'] if name in selected else old
        if name in selected:
            require(COMMON.identity(old,selected[name]['baseline'])['matches'],
                    'Selected baseline differs in compilation/linked occurrence')
        check = COMMON.identity(b[name],expected)
        require(check['matches'],'Unexpected device instruction/metadata/descriptor: '+name)
        result.append({'name':name,'changed':name in selected,'identity_checks':check,
                       'baseline':old,'candidate':b[name]})
    return result


def generated_checks(base,candidate,base_root,candidate_root):
    rows = []
    for directory in ('impl','instances'):
        a,b = base/'blob.staging'/directory,candidate/'blob.staging'/directory
        names = {str(p.relative_to(a)) for p in a.rglob('*') if p.is_file()}
        require(names=={str(p.relative_to(b)) for p in b.rglob('*') if p.is_file()},'Generated file set changed')
        for name in sorted(names):
            require((a/name).read_bytes()==(b/name).read_bytes(),'Generated TU/host launch changed: '+name)
            rows.append({'path':directory+'/'+name,'sha256':sha(b/name),'byte_equal':True})
    normalize = lambda path,sroot: path.read_text().replace(str(path.parents[1]),'<JIT_MODULE>').replace(str(sroot),'<SOURCE_ROOT>')
    require(normalize(base/'build/build.ninja',base_root)==normalize(candidate/'build/build.ninja',candidate_root),
            'Build flags/input/link ordering differs beyond source/JIT paths')
    return {'files':rows,'generated_file_count':len(rows),'all_byte_equal':True,
            'normalized_build_ninja_equal':True,'baseline_ninja_sha256':sha(base/'build/build.ninja'),
            'candidate_ninja_sha256':sha(candidate/'build/build.ninja')}


def audit(selection_path,build_path):
    selection = read(selection_path)
    old_source,old_build,old_audit = (read(BASE/p) for p in ('source_manifest.json','build_manifest.json','identity_audit.json'))
    require(old_audit['summary']['public_parents']==26 and old_audit['summary']['actual_device_variants']==56
            and not old_audit['failures'],'Current Oct8 identity baseline is incomplete')
    require(sha(old_build['binary'])==old_build['binary_sha256']==old_audit['candidate_binary_sha256'],
            'Current Oct8 official binary changed')
    candidate_root,base_root,source_checks = verify_sources(selection,old_source)
    build = read(build_path)
    require(build['status']=='passed' and build['rebuilt'] and build['pending_candidate_no_adoption']
            and not build['gpu_execution_requested'] and build['imported_tune'] is None,'Root official CPU build incomplete')
    require(build['visibility']=={k:'' for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')},
            'Build GPUs were not hidden')
    require(Path(build['source_root']).resolve()==candidate_root and build['source_head']==source_checks['source_head'],
            'Build/source selection differs')
    require(Path(build['imported_core']).resolve()==candidate_root/'aiter/jit/core.py','Wrong official JIT core')
    require(digest(build['source_diff'].encode())==source_checks['source_diff_sha256'],'Build source diff changed')
    for key in ('kids','compiler','filtered_backend_option','resource_directory','hip_declaration_header',
                'hip_declaration_header_sha256','build_entry'):
        require(build[key]==old_build[key],'Official build setting differs: '+key)
    binary = Path(build['binary']).resolve()
    require(binary.parent==Path(build['jit_directory']).resolve() and sha(binary)==build['binary_sha256'],
            'Official candidate binary path/SHA differs')
    require(sha(build['hip_declaration_header'])==build['hip_declaration_header_sha256'],'HIP declarations changed')
    base = Path(old_build['jit_directory'])/'build/module_deepgemm_opus'
    candidate = Path(build['jit_directory'])/'build/module_deepgemm_opus'
    generated = generated_checks(base,candidate,base_root,candidate_root)
    public = {v['name']:{'parent_id':p['parent_id'],'row':v['candidate']}
              for p in old_audit['parents'] for v in p['variants']}
    require(len(public)==56,'Public symbols not unique')
    selected = selected_devices(selection,public)
    parents = []
    changed_names = set()
    for old_parent in old_audit['parents']:
        kid = old_parent['parent_id']
        bobj = Path(old_parent['candidate_object'])
        cobj = candidate/'build'/bobj.name
        require(sha(bobj)==old_parent['candidate_object_sha256'],'Baseline public object changed')
        before,after = only_rows(bobj),only_rows(cobj)
        recorded = {v['name']:v['candidate'] for v in old_parent['variants']}
        require(set(recorded)=={r['name'] for r in before},'Baseline parent symbol set changed')
        for row in before:
            require(COMMON.identity(row,recorded[row['name']])['matches'],'Baseline recorded parent identity changed')
            expect = 20 if 'opus_gemm_mxscale_bpreshuffle_reduce_kernel' in row['name'] else 96
            require(row['metadata']['.kernarg_segment_size']==expect,'Parent ABI kernarg changed')
        variants = compare_rows(before,after,selected)
        changed_names.update(v['name'] for v in variants if v['changed'])
        prefix = bobj.name.removesuffix('_Cbf16_t.device.cuda.o')
        parents.append({'parent_id':kid,'baseline_object':str(bobj),'baseline_object_sha256':sha(bobj),
                        'candidate_object':str(cobj),'candidate_object_sha256':sha(cobj),
                        'generated_tu_sha256':sha(candidate/'blob.staging/instances'/(prefix+'_Cbf16_t.device.cu')),
                        'generated_impl_sha256':sha(candidate/'blob.staging/impl'/(prefix+'.cuh')),'variants':variants})
    require(len(parents)==26 and sum(len(p['variants']) for p in parents)==56 and changed_names==set(selected),
            'Public26/56 selection coverage mismatch')
    bobjects = {p.name:p for p in (base/'build').glob('*.cuda.o')}
    cobjects = {p.name:p for p in (candidate/'build').glob('*.cuda.o')}
    recorded = {r['object']:r for r in old_audit['all_build_object_checks']}
    require(set(bobjects)==set(cobjects)==set(recorded),'Complete build object set changed')
    objects = []
    for name,bobj in sorted(bobjects.items()):
        cobj = cobjects[name]
        require(sha(bobj)==recorded[name]['candidate_sha256'],'Baseline build object changed: '+name)
        before,after = bundles(bobj),bundles(cobj)
        require(len(before)==len(after),'Build object bundle count changed')
        entries = []
        for a,b in zip(before,after):
            require(a['targets']==b['targets'],'Build object target list differs')
            entries.extend(compare_rows(a['rows'],b['rows'],selected))
        objects.append({'object':name,'baseline_sha256':sha(bobj),'candidate_sha256':sha(cobj),
                        'device_bundle_present':bool(before),'gfx950_bundle_count':len(before),
                        'device_entries_checked':len(entries),'changed_entries':sum(r['changed'] for r in entries),
                        'all_expected_identities_equal':True})
    linked_before,linked_after = bundles(old_build['binary']),bundles(binary)
    require(len(linked_before)==len(linked_after)==len(old_audit['linked_module_bundle_checks']),
            'Linked bundle count changed')
    linked = []
    seen = []
    for a,b,old in zip(linked_before,linked_after,old_audit['linked_module_bundle_checks']):
        require(a['image_sha256']==old['candidate_image_sha256'] and a['targets']==b['targets']==old['targets'],
                'Baseline image or linked target ordering changed')
        entries = compare_rows(a['rows'],b['rows'],selected)
        changed = [r['name'] for r in entries if r['changed']]
        seen.extend(changed)
        linked.append({'ordinal':a['ordinal'],'baseline_image_sha256':a['image_sha256'],
                       'candidate_image_sha256':b['image_sha256'],'targets':a['targets'],
                       'entries_checked':len(entries),'changed_entries':len(changed),'changed_symbols':changed,
                       'all_expected_identities_equal':True})
    require(len(seen)==len(selected) and set(seen)==set(selected),'Selected linked occurrences are missing/duplicate')
    require(sha(binary)==build['binary_sha256'] and sha(old_build['binary'])==old_build['binary_sha256'],
            'Binary changed during audit')
    summary = {'public_parents':26,'actual_device_variants':56,'changed_device_entries':len(selected),
               'unchanged_device_entries':56-len(selected),'changed_entries_exact_tested_candidate':True,
               'all_unaffected_instructions_metadata_normalized_descriptors_exact_Oct8':True,
               'frozen9000_9010_9021_unchanged':True,
               'generated_files_checked':generated['generated_file_count'],'normalized_ninja_equal':True,
               'build_objects_checked':len(objects),'device_objects_checked':sum(r['device_bundle_present'] for r in objects),
               'linked_gfx950_bundles_checked':len(linked),
               'linked_device_entries_checked':sum(r['entries_checked'] for r in linked),'failure_count':0}
    return {'status':'passed_pending_official_API_no_adoption','generated_utc':datetime.now(timezone.utc).isoformat(),
            'cpu_only':True,'new_GPU_execution':False,'new_build_execution':False,'production_modified':False,
            'source_head':build['source_head'],'candidate_source_root':str(candidate_root),
            'candidate_binary':str(binary),'candidate_binary_sha256':build['binary_sha256'],
            'baseline_binary':old_build['binary'],'baseline_binary_sha256':old_build['binary_sha256'],
            'selection':selection,'selection_path':str(selection_path),'selection_sha256':sha(selection_path),
            'build_manifest_path':str(build_path),'build_manifest_sha256':sha(build_path),
            'audit_script_sha256':sha(__file__),
            'baseline_evidence':{str(BASE/p):sha(BASE/p) for p in ('source_manifest.json','build_manifest.json','identity_audit.json')},
            'source_checks':source_checks,'generated_checks':generated,'selected_devices':selected,
            'parents':parents,'all_build_object_checks':objects,'linked_module_bundle_checks':linked,
            'summary':summary,'failures':[],
            'limits':['CPU audit never imports or runs the official candidate HIP module.',
                      'Only descriptor entry-offset bytes16..23 are normalized; full metadata and all instruction bytes compared.',
                      'All current linked entries, including nonpublic kernels, are checked against current Oct8.',
                      'Performance comes from sealed private evidence, not this identity audit.',
                      'Official API/reference/repeatability/guards and actual loaded-module SHA remain required.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--build',type=Path,default=HERE/'build_manifest.json')
    parser.add_argument('--output',type=Path,default=HERE/'identity_audit.json')
    parser.add_argument('--source-only',action='store_true')
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(HERE) and not args.output.exists(),'Choose a new output record')
    try:
        if args.source_only:
            candidate,baseline,checks = verify_sources(read(args.selection),read(BASE/'source_manifest.json'))
            report = {'status':'passed_CPU_source_chain_no_build','cpu_only':True,'new_GPU_execution':False,
                      'new_build_execution':False,'production_modified':False,'source_checks':checks,
                      'selection_sha256':sha(args.selection),'audit_script_sha256':sha(__file__),'failures':[]}
        else:
            report = audit(args.selection,args.build)
    except Exception as error:
        report = {'status':'failed_CPU_integration_requires_review','cpu_only':True,'new_GPU_execution':False,
                  'new_build_execution':False,'production_modified':False,'audit_script_sha256':sha(__file__),
                  'failures':[{'type':type(error).__name__,'reason':str(error)}]}
        new_write(args.output,report)
        print(json.dumps({'status':report['status'],'output':str(args.output),'failures':report['failures']}))
        raise SystemExit(1)
    new_write(args.output,report)
    require('torch' not in sys.modules,'CPU audit imported torch')
    print(json.dumps({'status':report['status'],'output':str(args.output),'summary':report.get('summary'),
                      'failures':report['failures']}))


if __name__ == '__main__':
    main()
