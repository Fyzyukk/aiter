#!/usr/bin/env python3
"""CPU-only identity audit of Oct8 official selected against Oct7 selected.

Reads existing sources, generated TUs, ELF objects and linked HIP bundles.
Only the 9021 entry may change, and it must exactly match the Event-tested
scale_issue_publish candidate. No build, HIP import or GPU call is performed.
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

HERE = Path(__file__).resolve().parent
RESUME = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/opus_bound_analysis_20261007'
OLD_SELECTED = OLD / 'formal_selected'
PRIVATE = RESUME / 'scale_issue_publish'
HEADER = 'csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh'
SMALL = 'csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh'
MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
TARGET = 'hip-amdgcn-amd-amdhsa--gfx950'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def load_extractor():
    path = OLD / 'audit_current_compute_metadata.py'
    # The historical helper creates its existing evidence directory at import.
    # Require it to exist so this audit does not create historical artifacts.
    require((OLD / 'official_compute_metadata').is_dir(), 'Historical helper evidence directory missing')
    spec = importlib.util.spec_from_file_location('oct8_elf_audit_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(actual, expected):
    checks = {
        'symbol_equal': actual['name'] == expected['name'],
        'instruction_bytes_equal': actual['instruction_bytes'] == expected['instruction_bytes'],
        'instruction_equal': actual['instruction_sha256'] == expected['instruction_sha256'],
        'metadata_equal': actual['metadata'] == expected['metadata'],
        'descriptor_normalized_equal': actual['descriptor_normalized_sha256'] == expected['descriptor_normalized_sha256'],
    }
    checks['matches_expected'] = all(checks.values())
    return checks


def bundles(path, extract):
    """Parse every concatenated offload bundle, including linked module bundles."""
    outer = extract.Elf(path.read_bytes())
    if '.hip_fatbin' not in outer.sections:
        return []
    data = outer.bytes(outer.sections['.hip_fatbin'])
    result = []
    cursor = 0
    while True:
        begin = data.find(MAGIC, cursor)
        if begin < 0:
            break
        require(begin + len(MAGIC) + 8 <= len(data), f'Truncated bundle in {path}')
        count = struct.unpack_from('<Q', data, begin + len(MAGIC))[0]
        require(0 < count < 1024, f'Invalid bundle target count in {path}')
        offset = begin + len(MAGIC) + 8
        targets = []
        images = []
        bundle_end = offset
        for _ in range(count):
            require(offset + 24 <= len(data), f'Truncated bundle descriptor in {path}')
            payload, size, idlen = struct.unpack_from('<QQQ', data, offset)
            offset += 24
            require(offset + idlen <= len(data), f'Truncated target ID in {path}')
            target = data[offset:offset + idlen].decode()
            offset += idlen
            require(begin + payload + size <= len(data), f'Truncated target image in {path}')
            targets.append(target)
            bundle_end = max(bundle_end, begin + payload + size, offset)
            if target == TARGET:
                image = data[begin + payload:begin + payload + size]
                _, rows = extract.summarize_image(image)
                require(len({r['name'] for r in rows}) == len(rows), f'Duplicate kernel symbol in {path}')
                images.append({'image_sha256': digest(image), 'rows': rows})
        require(len(images) == 1, f'Expected one gfx950 image per bundle in {path}')
        result.append({'ordinal': len(result), 'section_offset': begin, 'targets': targets, **images[0]})
        cursor = bundle_end
    require(result, f'No valid gfx950 offload bundle in {path}')
    return result


def object_rows(path, extract):
    images = bundles(path, extract)
    require(len(images) == 1, f'Expected one bundle in compilation object {path}')
    return images[0]['rows']


def verify_sources(sources, old_source, old_integration):
    candidate = Path(sources['candidate_worktree']).resolve()
    old_root = Path(old_source['candidate_worktree']).resolve()
    require(Path(sources['production_checkout']).resolve() == ROOT, 'Unexpected production checkout')
    require(candidate != ROOT and candidate != old_root, 'Candidate must use a separate worktree')
    require(sources['pending_no_adoption'] and not sources['production_sources_modified'], 'Unexpected source adoption state')
    require(git(candidate, 'rev-parse', 'HEAD').decode().strip() == sources['source_head'] == old_source['source_head'], 'Source HEAD mismatch')
    require(git(old_root, 'rev-parse', 'HEAD').decode().strip() == sources['source_head'], 'Oct7 worktree HEAD mismatch')
    old_files = {f['path']: f for f in old_source['files']}
    applied = {f['path']: f for f in old_integration['source_files']}
    files = {f['path']: f for f in sources['files']}
    require(set(files) == set(old_files) == {HEADER, SMALL}, 'Unexpected selected source scope')
    for path, item in files.items():
        require(digest(git(candidate, 'show', 'HEAD:' + path)) == item['original_sha256'] == old_files[path]['original_sha256'], f'HEAD source mismatch: {path}')
        require(sha(old_root / path) == old_files[path]['candidate_sha256'] == applied[path]['applied_sha256'] == item['checkout_current_sha256'], f'Oct7 selected source mismatch: {path}')
        require(sha(ROOT / path) == item['checkout_current_sha256'], f'Production source changed during preparation: {path}')
        require(sha(candidate / path) == item['candidate_sha256'] == sha(Path(item['private_candidate'])), f'Tested private source mismatch: {path}')
        require(item['production_unchanged'], f'Unexpected production modification: {path}')
    require(files[SMALL]['candidate_sha256'] == files[SMALL]['checkout_current_sha256'], 'Oct7 register aliases changed')
    changed_head = git(candidate, 'diff', '--name-only').decode().splitlines()
    require(set(changed_head) == {HEADER, SMALL} == set(sources['changed_files_vs_head']), 'Unexpected changes relative to HEAD')
    changed_selected = [path for path, item in files.items() if item['candidate_sha256'] != item['checkout_current_sha256']]
    require(changed_selected == [HEADER] == sources['changed_files_vs_current'], 'Only9021 may change relative to Oct7 selected')
    joint = git(candidate, 'diff', '--binary')
    require(digest(joint) == sources['source_diff_sha256'] == sha(Path(sources['source_diff_path'])), 'Joint source diff SHA mismatch')
    require(joint == Path(sources['source_diff_path']).read_bytes(), 'Joint source diff bytes mismatch')
    require(digest(git(old_root, 'diff', '--binary')) == old_source['source_diff_sha256'], 'Historical selected worktree changed')
    selected = ''.join(difflib.unified_diff((ROOT / HEADER).read_text().splitlines(keepends=True),
                                          (candidate / HEADER).read_text().splitlines(keepends=True),
                                          fromfile='a/' + HEADER, tofile='b/' + HEADER)).encode()
    require(digest(selected) == sources['selected_diff_sha256'] == sources['production_diff_sha256'], 'Diff relative to selected SHA mismatch')
    require(selected == Path(sources['selected_diff_path']).read_bytes() == Path(sources['production_diff_path']).read_bytes(), 'Diff relative to selected bytes mismatch')
    git(candidate, 'diff', '--check')
    # The unchanged build inputs must match the retained selected checkout.
    # Ignore non-build documentation and gitlink directories, whose state is
    # separately represented in the source preparation manifest.
    tracked = git(candidate, 'ls-files', '--stage', '-z').decode().split('\0')
    checked = []
    for row in tracked:
        if not row:
            continue
        mode_and_blob, path = row.split('\t', 1)
        mode = mode_and_blob.split()[0]
        if mode == '160000' or not path.startswith(('aiter/', 'csrc/', '3rdparty/')):
            continue
        old = old_root / path
        new = candidate / path
        production = ROOT / path
        require(old.is_file() and new.is_file() and production.is_file(), f'Build source missing: {path}')
        old_hash, new_hash, production_hash = sha(old), sha(new), sha(production)
        require(production_hash == old_hash, f'Production build source differs from selected: {path}')
        if path != HEADER:
            require(new_hash == old_hash, f'Unintended build source change: {path}')
        checked.append({'path': path, 'selected_sha256': old_hash, 'candidate_sha256': new_hash})
    control = sources['unchanged_control']
    require(sha(candidate / control['path']) == sha(ROOT / control['path']) == control['candidate_sha256'] == control['checkout_current_sha256'] == control['head_sha256'], '9022 source control changed')
    for evidence in sources['source_evidence'].values():
        require(sha(Path(evidence['path'])) == evidence['sha256'], 'Source evidence changed: ' + evidence['path'])
    preparation = sources['build_preparation']
    require(sha(Path(preparation['script'])) == preparation['script_sha256'], 'Official build script changed')
    return candidate, old_root, {'changed_files_vs_head': changed_head, 'changed_files_vs_selected': changed_selected,
                                'source_diff_sha256': digest(joint), 'selected_diff_sha256': digest(selected),
                                'unchanged_build_input_count': len(checked) - 1,
                                'build_source_inventory_sha256': digest(json.dumps(checked, sort_keys=True).encode()),
                                'production_sources_unchanged': True, 'old_selected_sources_unchanged': True}


def verify_build(manifest, sources, old_build, candidate):
    require(manifest['status'] == 'passed' and manifest['pending_candidate_no_adoption'] and manifest['rebuilt'], 'Official build not complete/pending')
    require(manifest['gpu_execution_requested'] is False and manifest['imported_tune'] is None, 'Build must be CPU-only without tuner import')
    require(manifest['visibility'] == {k: '' for k in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}, 'GPUs were not hidden during build')
    require(Path(manifest['source_root']).resolve() == candidate and manifest['source_head'] == sources['source_head'], 'Official build source mismatch')
    require(Path(manifest['imported_core']).resolve() == candidate / 'aiter/jit/core.py', 'Wrong imported JIT core')
    preparation = sources['build_preparation']
    require(Path(manifest['jit_directory']).resolve() == Path(preparation['jit_directory']).resolve(), 'Wrong candidate JIT directory')
    require(Path(manifest['output_directory']).resolve() == HERE and Path(manifest['manifest_path']).resolve() == HERE / 'build_manifest.json', 'Wrong official output path')
    require(Path(manifest['binary']).resolve() == Path(manifest['jit_directory']).resolve() / 'module_deepgemm_opus.so', 'Wrong official binary path')
    require(sha(Path(manifest['binary'])) == manifest['binary_sha256'], 'Official binary SHA mismatch')
    require(digest(manifest['source_diff'].encode()) == sources['source_diff_sha256'], 'Build source diff differs from preparation')
    for key in ('kids', 'compiler', 'filtered_backend_option', 'resource_directory', 'hip_declaration_header',
                'hip_declaration_header_sha256', 'build_entry'):
        require(manifest[key] == old_build[key], 'Official build setting changed: ' + key)
    require(sha(Path(manifest['hip_declaration_header'])) == manifest['hip_declaration_header_sha256'], 'HIP declaration header changed')
    return Path(manifest['jit_directory']) / 'build/module_deepgemm_opus'


def verify_private(extract):
    source = read(PRIVATE / 'source_manifest.json')
    build = read(PRIVATE / 'build_manifest.json')
    audit = read(PRIVATE / 'device_audit.json')
    require(build['status'] == audit['status'] == 'passed', 'Private build/device audit failed')
    built = {r['side']: r for r in build['builds']}
    audited = {r['label']: r for r in audit['libraries']}
    result = {}
    for label in ('baseline', 'candidate'):
        library = PRIVATE / label / 'experiments.so'
        require(sha(library) == built[label]['library_sha256'] == audited[label]['input_sha256'], 'Private library chain mismatch: ' + label)
        require(built[label]['returncode'] == 0, 'Private build failed: ' + label)
        for item in source['source_files']:
            require(sha(PRIVATE / label / item['path']) == item[label + '_sha256'], 'Private source chain mismatch: ' + item['path'])
        images = bundles(library, extract)
        require(len(images) == 1 and images[0]['image_sha256'] == audited[label]['device_sha256'], 'Private device image SHA mismatch')
        rows = images[0]['rows']
        require(len(rows) == len(audited[label]['kernels']) == 2, 'Private kernel count mismatch')
        by_name = {r['name']: r for r in rows}
        result[label] = {}
        for recorded in audited[label]['kernels']:
            require(identity(by_name[recorded['name']], recorded)['matches_expected'], 'Private exact kernel identity mismatch')
            result[label][recorded['kid']] = by_name[recorded['name']]
    require(identity(result['baseline'][9022], result['candidate'][9022])['matches_expected'], 'Private9022 changed')
    return result


def generated_checks(base, candidate):
    checks = []
    for directory in ('impl', 'instances'):
        old = base / 'blob.staging' / directory
        new = candidate / 'blob.staging' / directory
        names = {str(p.relative_to(old)) for p in old.rglob('*') if p.is_file()}
        new_names = {str(p.relative_to(new)) for p in new.rglob('*') if p.is_file()}
        require(names == new_names, 'Generated ' + directory + ' file set changed')
        for name in sorted(names):
            a, b = old / name, new / name
            require(a.read_bytes() == b.read_bytes(), 'Generated TU/host implementation changed: ' + name)
            checks.append({'path': directory + '/' + name, 'sha256': sha(b), 'byte_equal': True})
    def normalized_ninja(root):
        data = (root / 'build/build.ninja').read_text()
        return data.replace(str(root), '<JIT_MODULE>').replace(str(Path(read(OLD_SELECTED / 'source_manifest.json')['candidate_worktree'])), '<SOURCE_ROOT>')
    old_ninja = normalized_ninja(base)
    new_ninja = (candidate / 'build/build.ninja').read_text().replace(str(candidate), '<JIT_MODULE>')
    source_root = Path(read(HERE / 'source_manifest.json')['candidate_worktree'])
    new_ninja = new_ninja.replace(str(source_root), '<SOURCE_ROOT>')
    require(old_ninja == new_ninja, 'Build ninja flags/input/link ordering changed beyond source/JIT roots')
    return {'files': checks, 'generated_file_count': len(checks), 'all_byte_equal': True,
            'baseline_build_ninja_sha256': sha(base / 'build/build.ninja'),
            'candidate_build_ninja_sha256': sha(candidate / 'build/build.ninja'),
            'normalized_build_ninja_equal': True}


def compare_images(before, after, changed_name, private_candidate):
    bmap, cmap = {r['name']: r for r in before}, {r['name']: r for r in after}
    require(set(bmap) == set(cmap), 'Device kernel symbol set changed')
    entries = []
    for name, old in bmap.items():
        if name == changed_name:
            require(not identity(old, private_candidate)['matches_expected'], 'Expected9021 candidate is identical to selected baseline')
        expected = private_candidate if name == changed_name else old
        check = identity(cmap[name], expected)
        require(check['matches_expected'], 'Unexpected device instruction/metadata/descriptor identity: ' + name)
        entries.append({'name': name, 'changed': name == changed_name, 'identity_checks': check,
                        'baseline': old, 'candidate': cmap[name], 'expected_private_or_selected': expected})
    return entries


def run(source_only=False):
    sources = read(HERE / 'source_manifest.json')
    old_source = read(OLD_SELECTED / 'source_manifest.json')
    old_build = read(OLD_SELECTED / 'build_manifest.json')
    old_audit = read(OLD_SELECTED / 'identity_audit.json')
    old_integration = read(OLD_SELECTED / 'integration_manifest.json')
    require(old_integration['adopted'] and old_integration['status'] == 'applied_verified_not_committed', 'Baseline is not Oct7 adopted selected')
    require(old_audit['status'] == 'passed_pending_cpu_identity_no_adoption' and not old_audit['failures'], 'Historical selected identity failed')
    require(old_audit['summary']['public_parents'] == 26 and old_audit['summary']['actual_device_variants'] == 56, 'Historical public entry inventory mismatch')
    for evidence in old_integration['evidence']:
        if evidence['path'] in ('formal_selected/source_manifest.json', 'formal_selected/build_manifest.json', 'formal_selected/identity_audit.json'):
            require(sha(OLD / evidence['path']) == evidence['sha256'], 'Historical selected manifest changed')
    require(sha(Path(old_build['binary'])) == old_build['binary_sha256'] == old_audit['candidate_binary_sha256'] == old_integration['official_module']['sha256'], 'Oct7 selected official binary changed')
    candidate_root, old_root, source_checks = verify_sources(sources, old_source, old_integration)
    if source_only:
        return {'status': 'passed_cpu_source_chain_only', 'cpu_only': True, 'gpu_executed': False,
                'source_checks': source_checks, 'official_device_audit_performed': False}
    manifest = read(HERE / 'build_manifest.json')
    candidate = verify_build(manifest, sources, old_build, candidate_root)
    base = Path(old_build['jit_directory']) / 'build/module_deepgemm_opus'
    extract = load_extractor()
    private = verify_private(extract)
    generated = generated_checks(base, candidate)
    changed_name = private['candidate'][9021]['name']
    parents = []
    changed_count = 0
    for old_parent in old_audit['parents']:
        kid = old_parent['parent_id']
        bobj = Path(old_parent['candidate_object'])
        cobj = candidate / 'build' / bobj.name
        require(sha(bobj) == old_parent['candidate_object_sha256'], 'Historical selected public object changed')
        before, after = object_rows(bobj, extract), object_rows(cobj, extract)
        require(len(before) == len(after) == len(old_parent['variants']), 'Public parent entry count changed')
        recorded = {v['candidate']['name']: v['candidate'] for v in old_parent['variants']}
        require(set(recorded) == {r['name'] for r in before}, 'Historical selected public symbols changed')
        for row in before:
            require(identity(row, recorded[row['name']])['matches_expected'], 'Historical selected public identity changed')
            expected_kargs = 20 if 'opus_gemm_mxscale_bpreshuffle_reduce_kernel' in row['name'] else 96
            require(row['metadata']['.kernarg_segment_size'] == expected_kargs, 'Public kernel ABI size changed')
        if kid == 9021:
            require(len(before) == 1 and identity(before[0], private['baseline'][9021])['matches_expected'], 'Private baseline is not Oct7 selected9021')
        if kid == 9022:
            require(len(before) == 1 and identity(before[0], private['baseline'][9022])['matches_expected'], 'Private9022 control mismatch')
        entries = compare_images(before, after, changed_name, private['candidate'][9021])
        require(all(v['changed'] == (kid == 9021) for v in entries), 'Changed entry outside9021 parent')
        changed_count += sum(v['changed'] for v in entries)
        prefix = bobj.name.removesuffix('_Cbf16_t.device.cuda.o')
        parents.append({'parent_id': kid, 'baseline_object': str(bobj), 'baseline_object_sha256': sha(bobj),
                        'candidate_object': str(cobj), 'candidate_object_sha256': sha(cobj),
                        'exact_object_sha_equal': sha(bobj) == sha(cobj), 'generated_tu_equal': True,
                        'generated_impl_host_launch_equal': True,
                        'generated_tu_sha256': sha(candidate / 'blob.staging/instances' / (prefix + '_Cbf16_t.device.cu')),
                        'generated_impl_sha256': sha(candidate / 'blob.staging/impl' / (prefix + '.cuh')),
                        'variants': entries})
    require(len(parents) == 26 and sum(len(p['variants']) for p in parents) == 56 and changed_count == 1, 'Expected26parents/56entries/one changed9021')
    baseline_objects = {p.name: p for p in (base / 'build').glob('*.cuda.o')}
    candidate_objects = {p.name: p for p in (candidate / 'build').glob('*.cuda.o')}
    historical_objects = {c['object']: c for c in old_audit['all_build_object_checks']}
    require(set(baseline_objects) == set(candidate_objects) == set(historical_objects), 'Complete build object set changed')
    object_checks = []
    for name, bobj in sorted(baseline_objects.items()):
        cobj = candidate_objects[name]
        require(sha(bobj) == historical_objects[name]['candidate_sha256'], 'Historical build object changed: ' + name)
        before, after = bundles(bobj, extract), bundles(cobj, extract)
        require(len(before) == len(after), 'Build object bundle count changed: ' + name)
        count = 0
        for bimage, cimage in zip(before, after):
            require(bimage['targets'] == cimage['targets'], 'Build object target set changed: ' + name)
            count += len(compare_images(bimage['rows'], cimage['rows'], changed_name, private['candidate'][9021]))
        object_checks.append({'object': name, 'baseline_sha256': sha(bobj), 'candidate_sha256': sha(cobj),
                              'exact_object_sha_equal': sha(bobj) == sha(cobj), 'device_bundle_present': bool(before),
                              'gfx950_bundle_count': len(before), 'device_entries_checked': count,
                              'all_expected_identities_equal': True})
    linked_before = bundles(Path(old_build['binary']), extract)
    linked_after = bundles(Path(manifest['binary']), extract)
    require(len(linked_before) == len(linked_after), 'Linked module bundle count changed')
    linked_checks = []
    linked_changed = 0
    for before, after in zip(linked_before, linked_after):
        require(before['targets'] == after['targets'], 'Linked target ordering changed')
        entries = compare_images(before['rows'], after['rows'], changed_name, private['candidate'][9021])
        linked_changed += sum(v['changed'] for v in entries)
        linked_checks.append({'ordinal': before['ordinal'], 'baseline_image_sha256': before['image_sha256'],
                              'candidate_image_sha256': after['image_sha256'], 'targets': before['targets'],
                              'entries_checked': len(entries), 'changed_entries': sum(v['changed'] for v in entries),
                              'all_expected_identities_equal': True})
    require(linked_changed == 1, 'Linked official module must contain exactly one changed9021')
    require(sha(Path(manifest['binary'])) == manifest['binary_sha256'] and sha(Path(old_build['binary'])) == old_build['binary_sha256'], 'Binary changed during audit')
    summary = {'public_parents': len(parents), 'actual_device_variants': 56, 'changed_device_entries': 1,
               'unchanged_device_entries': 55, 'changed_entries_match_tested_private_candidate': True,
               'unchanged_entries_match_Oct7_official_selected': True, '9022_unchanged': True,
               'Oct7_register_alias_entries_unchanged': True, 'generated_host_launch_and_tus_equal': True,
               'generated_files_checked': generated['generated_file_count'], 'normalized_build_ninja_equal': True,
               'build_objects_checked': len(object_checks), 'device_objects_checked': sum(c['device_bundle_present'] for c in object_checks),
               'linked_gfx950_bundles_checked': len(linked_checks), 'linked_device_entries_checked': sum(c['entries_checked'] for c in linked_checks),
               'failure_count': 0}
    evidence_paths = [OLD_SELECTED / 'source_manifest.json', OLD_SELECTED / 'build_manifest.json',
                      OLD_SELECTED / 'identity_audit.json', OLD_SELECTED / 'integration_manifest.json',
                      PRIVATE / 'source_manifest.json', PRIVATE / 'build_manifest.json', PRIVATE / 'device_audit.json',
                      OLD / 'audit_current_compute_metadata.py']
    return {'status': 'passed_pending_cpu_identity_no_adoption', 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'cpu_only': True, 'gpu_executed': False, 'build_executed': False, 'production_checkout_modified': False,
            'source_head': sources['source_head'], 'candidate_source_root': str(candidate_root),
            'candidate_binary': manifest['binary'], 'candidate_binary_sha256': manifest['binary_sha256'],
            'baseline_kind': 'Oct7 adopted official selected, including retained9021 prologue and9042/9053/9054 aliases',
            'baseline_binary': old_build['binary'], 'baseline_binary_sha256_unchanged': old_build['binary_sha256'],
            'build_manifest_sha256': sha(HERE / 'build_manifest.json'), 'source_manifest_sha256': sha(HERE / 'source_manifest.json'),
            'audit_script_sha256': sha(Path(__file__)), 'evidence_sha256': {str(p): sha(p) for p in evidence_paths},
            'source_checks': source_checks, 'generated_checks': generated, 'parents': parents,
            'all_build_object_checks': object_checks, 'linked_module_bundle_checks': linked_checks,
            'summary': summary, 'failures': [],
            'limits': ['No official candidate HIP module was imported or run; formal API smoke and loaded-module SHA remain required before application.',
                       'The baseline is the retained Oct7 final selected module, not original HEAD or the earlier five-entry candidate.',
                       'Only descriptor kernel_code_entry_byte_offset bytes16..23 are normalized; full metadata and all instruction bytes are compared.',
                       'Whole host/object/bundle SHA may differ with source paths and code placement; exact kernel identities and generated host launch/TU bytes are separately checked.',
                       'CPU identity reuses Event evidence for the exact tested9021 implementation; it does not extend performance coverage to unmeasured supported shapes.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true', help='read-only source chain check, no ELF audit or output file')
    parser.add_argument('--output', type=Path, default=HERE / 'identity_audit.json')
    args = parser.parse_args()
    if args.source_only:
        print(json.dumps(run(source_only=True), ensure_ascii=False))
        return
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and not output.exists(), 'Audit output must be a new file within Oct8 formal_selected')
    try:
        report = run()
    except Exception as error:
        report = {'status': 'identity_failed_requires_review', 'cpu_only': True, 'gpu_executed': False,
                  'build_executed': False, 'production_checkout_modified': False,
                  'audit_script_sha256': sha(Path(__file__)),
                  'failures': [{'type': type(error).__name__, 'reason': str(error)}]}
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'output': str(output), 'failures': report['failures']}))
        raise SystemExit(1)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'summary': report['summary'], 'failures': []}))


if __name__ == '__main__':
    main()
