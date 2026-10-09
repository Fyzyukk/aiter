#!/usr/bin/env python3
"""Read-only CPU review of sealed remaining-Oct8 identity/API/application evidence.

Uses the retained ELF section/symbol reader, then compares actual FUNC bytes,
whole metadata dictionaries and descriptors independently of audit_integration.
Never builds, imports torch, executes HIP or applies sources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/opus_resume_20261008/formal_selected'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


spec = importlib.util.spec_from_file_location('independent_oct8_elf_reader', OLD / 'smoke_common.py')
ELF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ELF)
require('torch' not in sys.modules, 'CPU review imported torch')


def raw_images(path):
    result = []
    for image in ELF.device_bundles(path):
        elf = ELF.Elf(image)
        require(elf.machine == 224, 'Expected AMDGPU device ELF')
        symbols = elf.symbols()
        rows = {}
        for metadata in elf.metadata()['amdhsa.kernels']:
            name = metadata['.name']
            code = symbols[name]
            descriptor = symbols[metadata['.symbol']]['bytes']
            require(name not in rows and code['type'] == 2 and len(descriptor) == 64,
                    'Duplicate or invalid raw kernel: ' + name)
            require(len(code['bytes']) == code['size'], 'Truncated FUNC bytes')
            # Entry offset changes with link position. No other byte is ignored.
            normalized = descriptor[:16] + bytes(8) + descriptor[24:]
            rows[name] = {'code': code['bytes'], 'metadata': metadata,
                          'descriptor': normalized}
        result.append({'image_sha256': hashlib.sha256(image).hexdigest(), 'rows': rows})
    return result


def same(a, b):
    return a['code'] == b['code'] and a['metadata'] == b['metadata'] and a['descriptor'] == b['descriptor']


def verify_hash(path, expected, report):
    path = Path(path).resolve()
    actual = sha(path)
    require(actual == expected, 'Sealed SHA changed: ' + str(path))
    report[str(path)] = actual


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def build_sources(candidate, baseline, selected):
    rows = []
    changed = []
    for item in git(candidate, 'ls-files', '--stage', '-z').decode().split('\0'):
        if not item:
            continue
        mode, rel = item.split('\t', 1)
        if mode.split()[0] == '160000' or not rel.startswith(('aiter/', 'csrc/', '3rdparty/')):
            continue
        require((candidate / rel).is_file() and (baseline / rel).is_file(), 'Missing build source: ' + rel)
        old, new = sha(baseline / rel), sha(candidate / rel)
        if old != new:
            changed.append(rel)
            require(rel in selected, 'Unselected source changed: ' + rel)
            require(old == selected[rel]['baseline_sha256'] and new == selected[rel]['tested_sha256'],
                    'Selected source hash mismatch: ' + rel)
        else:
            require(rel not in selected, 'Selected source unchanged: ' + rel)
        rows.append({'path': rel, 'baseline_sha256': old, 'candidate_sha256': new, 'changed': old != new})
    require(set(changed) == set(selected), 'Selected build source set differs')
    return rows


def review_identity_plan():
    audit_path = HERE / 'identity_audit.json'
    cpu_path = HERE / 'api_gate/api_plan_cpu_audit.json'
    audit, cpu = read(audit_path), read(cpu_path)
    plan_path, queue_path = HERE / 'api_gate/official_api_plan.json', HERE / 'api_gate/official_api_queue.json'
    plan, queue = read(plan_path), read(queue_path)
    require(audit['status'] == 'passed_pending_official_API_no_adoption' and not audit['failures'],
            'Official identity did not pass')
    require(cpu['status'] == 'passed_CPU_API_plan_identity_pending_GPU', 'CPU API plan did not pass')
    hashes = {}
    for path, expected in [(audit['selection_path'], audit['selection_sha256']),
                           (audit['build_manifest_path'], audit['build_manifest_sha256']),
                           (HERE / 'audit_integration.py', audit['audit_script_sha256']),
                           (audit_path, cpu['identity_audit_sha256']),
                           (audit_path, plan['identity_audit_sha256']),
                           (plan_path, cpu['plan_sha256']), (queue_path, cpu['queue_sha256']),
                           (audit['candidate_binary'], audit['candidate_binary_sha256']),
                           (audit['baseline_binary'], audit['baseline_binary_sha256']),
                           (ROOT / 'csrc/opus_gemm/opus_gemm_common.py', plan['registry_sha256']),
                           (ROOT / 'csrc/opus_gemm/codegen/gen_instances_gfx950.py', plan['codegen_sha256']),
                           (ROOT / 'reports/opus_bound_analysis_20261007/official_smoke.py', plan['runner_sha256']),
                           (ROOT / 'reports/opus_bound_analysis_20261007/experiment_runner.py', plan['experiment_runner_sha256']),
                           (ROOT / 'reports/opus_bound_analysis_20261007/owned_python_launch.py', plan['owner_launcher_sha256'])]:
        verify_hash(path, expected, hashes)
    for source in (audit['baseline_evidence'], cpu['evidence']):
        for path, expected in source.items():
            verify_hash(path, expected, hashes)
    require(plan['official_binary'] == audit['candidate_binary'] and
            plan['official_binary_sha256'] == audit['candidate_binary_sha256'], 'Plan official library differs')
    selection = read(audit['selection_path'])
    require(selection == audit['selection'], 'Selection contents differ from sealed audit')
    candidate = Path(selection['source_root'])
    baseline = Path(audit['source_checks']['baseline_source_root'])
    selected_sources = {r['path']: r for r in selection['source_files']}
    for item in selected_sources.values():
        verify_hash(item['tested_source'], item['tested_sha256'], hashes)
    require(git(candidate, 'rev-parse', 'HEAD').decode().strip() == audit['source_head'] ==
            git(baseline, 'rev-parse', 'HEAD').decode().strip(), 'Source HEAD differs')
    source_rows = build_sources(candidate, baseline, selected_sources)
    require(len(source_rows) == audit['source_checks']['tracked_build_inputs_checked'], 'Build source count differs')
    require(hashlib.sha256(json.dumps(source_rows, sort_keys=True).encode()).hexdigest() ==
            audit['source_checks']['build_input_inventory_sha256'], 'Source inventory differs')
    private = {}
    selected = {}
    for item in selection['changed_entries']:
        name = item['symbol']
        require(name not in selected and item['parent_id'] in (9020, 9023, 9024), 'Unexpected selected entry')
        side_rows = []
        for key in ('private_baseline_library', 'private_library'):
            verify_hash(item[key], item[key + '_sha256'], hashes)
            if item[key] not in private:
                images = raw_images(item[key])
                require(len(images) == 1, 'Expected one private device image')
                private[item[key]] = images[0]['rows']
            require(name in private[item[key]], 'Selected private entry missing')
            side_rows.append(private[item[key]][name])
        require(not same(*side_rows), 'Selected private candidate has no change')
        selected[name] = {'parent_id': item['parent_id'], 'before': side_rows[0], 'after': side_rows[1]}
    require(len(selected) == 3, 'Expected exact final three-entry selection')
    before, after = raw_images(audit['baseline_binary']), raw_images(audit['candidate_binary'])
    require(len(before) == len(after) == 202, 'Linked bundle count differs')
    entries, changed = 0, []
    for index, (old_image, new_image) in enumerate(zip(before, after)):
        old, new = old_image['rows'], new_image['rows']
        require(set(old) == set(new), 'Linked kernel set differs at bundle ' + str(index))
        recorded = audit['linked_module_bundle_checks'][index]
        require(recorded['baseline_image_sha256'] == old_image['image_sha256'] and
                recorded['candidate_image_sha256'] == new_image['image_sha256'], 'Recorded linked image SHA differs')
        for name, old_row in old.items():
            entries += 1
            expected = old_row
            if name in selected:
                require(same(old_row, selected[name]['before']), 'Linked private baseline mismatch')
                expected = selected[name]['after']
                changed.append(name)
            require(same(new[name], expected), 'Actual linked FUNC/metadata/descriptor differs: ' + name)
    require(entries == 263 and len(changed) == 3 and set(changed) == set(selected), 'Linked entry coverage differs')
    old_objects = Path(audit['parents'][0]['baseline_object']).parent
    new_objects = Path(audit['parents'][0]['candidate_object']).parent
    device_objects = 0
    require(len(audit['all_build_object_checks']) == 206, 'Build object count differs')
    for recorded in audit['all_build_object_checks']:
        verify_hash(old_objects / recorded['object'], recorded['baseline_sha256'], hashes)
        verify_hash(new_objects / recorded['object'], recorded['candidate_sha256'], hashes)
        if not recorded['device_bundle_present']:
            continue
        device_objects += 1
        old, new = raw_images(old_objects / recorded['object']), raw_images(new_objects / recorded['object'])
        require(len(old) == len(new) == recorded['gfx950_bundle_count'], 'Build object image count differs')
        for a_image, b_image in zip(old, new):
            a, b = a_image['rows'], b_image['rows']
            require(set(a) == set(b), 'Build object symbol set differs')
            for name in a:
                expected = selected[name]['after'] if name in selected else a[name]
                require(same(b[name], expected), 'Actual compilation object identity differs: ' + name)
    require(device_objects == 202, 'Device build object count differs')
    old_module, new_module = old_objects.parent, new_objects.parent
    generated = audit['generated_checks']
    require(len(generated['files']) == 313, 'Generated file count differs')
    for row in generated['files']:
        verify_hash(old_module / 'blob.staging' / row['path'], row['sha256'], hashes)
        verify_hash(new_module / 'blob.staging' / row['path'], row['sha256'], hashes)
    for directory in ('impl', 'instances'):
        old_names = {str(p.relative_to(old_module / 'blob.staging'))
                     for p in (old_module / 'blob.staging' / directory).rglob('*') if p.is_file()}
        new_names = {str(p.relative_to(new_module / 'blob.staging'))
                     for p in (new_module / 'blob.staging' / directory).rglob('*') if p.is_file()}
        require(old_names == new_names == {r['path'] for r in generated['files'] if r['path'].startswith(directory + '/')},
                'Generated file set differs')
    old_ninja, new_ninja = old_objects / 'build.ninja', new_objects / 'build.ninja'
    verify_hash(old_ninja, generated['baseline_ninja_sha256'], hashes)
    verify_hash(new_ninja, generated['candidate_ninja_sha256'], hashes)
    normalize = lambda p, source: p.read_text().replace(str(p.parents[1]), '<JIT_MODULE>').replace(str(source), '<SOURCE_ROOT>')
    require(normalize(old_ninja, baseline) == normalize(new_ninja, candidate), 'Normalized ninja input/flag/link order differs')
    public = []
    scoped9020 = []
    for parent in audit['parents']:
        kid = parent['parent_id']
        verify_hash(parent['baseline_object'], parent['baseline_object_sha256'], hashes)
        verify_hash(parent['candidate_object'], parent['candidate_object_sha256'], hashes)
        old, new = raw_images(parent['baseline_object']), raw_images(parent['candidate_object'])
        require(len(old) == len(new) == 1, 'Public object bundle count differs')
        a, b = old[0]['rows'], new[0]['rows']
        require(set(a) == set(b), 'Public object symbol set differs')
        for name in a:
            expected = selected[name]['after'] if name in selected else a[name]
            require(same(b[name], expected), 'Actual public object identity differs')
            public.append({'parent_id': kid, 'symbol': name, 'changed': name in selected})
            if kid == 9020:
                scoped9020.append({'symbol': name, 'changed': name in selected,
                                   'exact_tested_candidate' if name in selected else 'exact_baseline': True,
                                   'instruction_bytes': len(b[name]['code']),
                                   'instruction_sha256': hashlib.sha256(b[name]['code']).hexdigest()})
    require(len(public) == 56 and len({r['parent_id'] for r in public}) == 26, 'Public coverage differs')
    require(len(scoped9020) == 7 and sum(r['changed'] for r in scoped9020) == 1, '9020 scoped coverage differs')
    targets = plan['targets']
    require(len(targets) == cpu['target_count'] == len(queue['commands']) == len(cpu['rows']) == 44,
            'API target count differs')
    require(len({t['kid'] for t in targets}) == 26 and all(t['signed'] and t['seed'] == 17 for t in targets),
            'API parent/signed seed coverage differs')
    require(plan['check_only'] and plan['repetitions_per_label_and_target'] == cpu['repetitions_per_label_and_target'] == 8,
            'API check-only/signed8 contract differs')
    require(queue['env']['AITER_JIT_DIR'] == str(Path(audit['candidate_binary']).parent) and
            queue['env']['AITER_AOT_IMPORT'] == '1' and queue['env']['GPU_ARCHS'] == 'gfx950' and
            queue['env']['CU_NUM'] == '256', 'API runtime environment differs')
    target_rows = []
    for index, (target, command, mapped) in enumerate(zip(targets, queue['commands'], cpu['rows'])):
        require(mapped['target_index'] == index and mapped['parent_id'] == target['kid'] and
                mapped['shape'] == target['shape'] and mapped['support_contract_passed'], 'Mapped target differs')
        name = f'official_remaining_{index}_kid{target["kid"]}'
        output = HERE / 'api_gate/api_results' / (name + '.json')
        expected = ['/opt/venv/bin/python3', str(ROOT / 'reports/opus_bound_analysis_20261007/official_smoke.py'),
                    '--plan', str(plan_path), '--output', str(output), '--target-index', str(index),
                    '--check-only', '--repetitions', '8']
        require(command == {'name': name, 'argv': expected, 'log': str(output.with_suffix('.log'))},
                'API command differs at index ' + str(index))
        actual = mapped['actual_candidate_symbol']
        require(mapped['changed'] == (actual in selected), 'Mapped changed flag differs')
        expected_labels = {f'remaining_kid{target["kid"]}': selected[actual]['parent_id']} if actual in selected else {}
        require(set(target['private_baselines']) == set(expected_labels), 'Private target labels differ')
        for label, path in target['private_baselines'].items():
            item = selected[actual]
            require(item['parent_id'] == target['kid'], 'Selected target parent differs')
            require(same(private[path][actual], item['after']), 'Private target identity differs')
        target_rows.append({'target_index': index, 'kid': target['kid'], 'shape': target['shape'],
                            'changed': actual in selected, 'symbol': actual,
                            'private_labels': list(target['private_baselines'])})
    covered = {row['symbol'] for row in target_rows if row['changed']}
    require(covered == set(selected), 'API selected-entry coverage differs')
    require('torch' not in sys.modules, 'CPU review imported torch')
    return {'status': 'passed_independent_actual_FUNC_metadata_descriptor_SHA_and_API_plan_review',
            'sealed_hashes': hashes, 'source_inputs_checked': len(source_rows),
            'source_changes_vs_baseline': sorted(selected_sources),
            'linked_bundles': len(after), 'linked_entries': entries,
            'build_objects': 206, 'device_build_objects': device_objects,
            'generated_files': 313, 'normalized_ninja_equal': True,
            'public_parents': 26, 'public_variants': len(public),
            'selected_entries': len(selected), 'unchanged_public_variants': 53,
            'scoped9020_entries': scoped9020, 'API_target_count': len(targets), 'API_targets': target_rows,
            'application_ready_checks': ['all tracked aiter/csrc/3rdparty inputs exact selected worktree',
                                         'all three applied source hashes exact reviewed snapshots',
                                         'registry/codegen hashes unchanged', 'sealed official module unchanged'],
            'limits': ['Uses the retained CPU ELF reader; raw FUNC bytes compared directly.',
                       'Branch legality is sealed to CPU plan audit; independent branch review is separate.',
                       'No GPU result or timing claim; official API result acceptance remains pending.']}


def review_application(application_root):
    audit = read(HERE / 'identity_audit.json')
    selection = read(audit['selection_path'])
    api = read(HERE / 'api_gate/gpu_api_analysis.json')
    require(api['status'] == 'passed_official_API_signed8_loaded_identity_all26parents'
            and api['target_count'] == 44 and api['public_parent_count'] == 26
            and api['changed_entries_checked'] == 3, 'Official API gate incomplete')
    for path, expected in api['evidence'].items():
        require(sha(path) == expected, 'Official API accepted evidence changed: ' + path)
    require(api['official_module'] == {'path': audit['candidate_binary'], 'sha256': audit['candidate_binary_sha256']},
            'Accepted API library differs from formal identity')
    candidate = Path(selection['source_root']).resolve()
    application_root = Path(application_root).resolve()
    preapply_path = HERE / 'source_preapply_snapshot.json'
    preapply = read(preapply_path)
    require(preapply['status'] == 'passed_preapply_production_exact_Oct8_baseline', 'Preapply baseline audit missing')
    require(preapply['identity_audit_sha256'] == sha(HERE / 'identity_audit.json') and
            preapply['selection_sha256'] == sha(audit['selection_path']), 'Preapply identity/selection changed')
    require(application_root == ROOT and application_root != candidate, 'Application must be production root')
    require(git(application_root, 'rev-parse', 'HEAD').decode().strip() == audit['source_head'], 'Application HEAD differs')
    rows = []
    for item in git(candidate, 'ls-files', '--stage', '-z').decode().split('\0'):
        if not item:
            continue
        mode, rel = item.split('\t', 1)
        if mode.split()[0] == '160000' or not rel.startswith(('aiter/', 'csrc/', '3rdparty/')):
            continue
        require((application_root / rel).is_file() and (candidate / rel).is_file(), 'Missing applied build source: ' + rel)
        require((application_root / rel).read_bytes() == (candidate / rel).read_bytes(), 'Applied build input differs: ' + rel)
        rows.append({'path': rel, 'sha256': sha(candidate / rel)})
    require(len(rows) == audit['source_checks']['tracked_build_inputs_checked'], 'Applied source inventory differs')
    applied = []
    for source in selection['source_files']:
        require(sha(application_root / source['path']) == source['tested_sha256'] == sha(source['tested_source']),
                'Applied selected source differs')
        applied.append({'path': source['path'], 'sha256': source['tested_sha256']})
    require(sha(audit['candidate_binary']) == audit['candidate_binary_sha256'], 'Sealed official module changed')
    # Root relayed the user's explicit instruction to append material findings
    # to these two original reports. Preserve their entire old prefix while
    # allowing appended bytes; every other historical artifact remains exact.
    authorized_docs = {str(ROOT / 'reports/opus_bound_analysis_20261007' / name)
                       for name in ('COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md',
                                    'MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md')}
    document_checks = []
    unchanged_artifacts = 0
    for recorded in preapply['historical_artifacts']:
        if recorded['path'] in authorized_docs:
            content = Path(recorded['path']).read_bytes()
            require(len(content) >= recorded['bytes'] and
                    hashlib.sha256(content[:recorded['bytes']]).hexdigest() == recorded['sha256'],
                    'Authorized append changed historical document prefix: ' + recorded['path'])
            document_checks.append({'path': recorded['path'], 'old_prefix_bytes': recorded['bytes'],
                                    'old_prefix_sha256': recorded['sha256'], 'old_prefix_byte_equal': True,
                                    'appended_bytes': len(content) - recorded['bytes'],
                                    'current_sha256': hashlib.sha256(content).hexdigest()})
        else:
            require(sha(recorded['path']) == recorded['sha256'], 'Historical artifact changed: ' + recorded['path'])
            unchanged_artifacts += 1
    require(len(document_checks) == 2, 'Authorized report append coverage differs')
    for directory, recorded_names in preapply['historical_artifact_file_sets'].items():
        current_names = [str(p.relative_to(directory)) for p in Path(directory).rglob('*') if p.is_file()]
        # pathlib sorts by path components, while strings sort punctuation in
        # the whole path. File-set preservation must not depend on that order.
        require(len(current_names) == len(recorded_names) == len(set(recorded_names))
                and set(current_names) == set(recorded_names),
                'Historical artifact file set changed: ' + directory)
    git(application_root, 'diff', '--check')
    return {'status': 'passed_independent_application_all_build_inputs_exact_formal_selected_worktree',
            'application_root': str(application_root), 'selected_worktree': str(candidate),
            'source_head': audit['source_head'], 'build_inputs_checked': len(rows), 'applied_sources': applied,
            'official_binary_sha256': audit['candidate_binary_sha256'],
            'official_API_analysis_sha256': sha(HERE / 'api_gate/gpu_api_analysis.json'),
            'preapply_snapshot_sha256': sha(preapply_path),
            'historical_artifacts_checked': len(preapply['historical_artifacts']),
            'historical_unchanged_artifacts_checked': unchanged_artifacts,
            'historical_artifacts_except_authorized_document_appends_unchanged': True,
            'authorized_document_append_checks': document_checks,
            'document_append_authorization': {
                'source': 'Parent /root relayed explicit user instruction in this session.',
                'instruction': 'Important conclusions must be appended to the original Compute and Memory reports.',
                'interpretation': 'Only appended bytes are allowed; all preapply bytes remain exact.'},
            'input_inventory_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            'limits': ['CPU source application review; no new build or GPU execution.',
                       'GPU API acceptance is recorded separately.']}


def review_preapply(application_root):
    audit = read(HERE / 'identity_audit.json')
    selection = read(audit['selection_path'])
    application_root = Path(application_root).resolve()
    baseline = Path(audit['source_checks']['baseline_source_root']).resolve()
    candidate = Path(selection['source_root']).resolve()
    require(application_root == ROOT and application_root not in (baseline, candidate), 'Preapply root differs')
    require(git(application_root, 'rev-parse', 'HEAD').decode().strip() == audit['source_head'], 'Preapply HEAD differs')
    rows = []
    for item in git(baseline, 'ls-files', '--stage', '-z').decode().split('\0'):
        if not item:
            continue
        mode, rel = item.split('\t', 1)
        if mode.split()[0] == '160000' or not rel.startswith(('aiter/', 'csrc/', '3rdparty/')):
            continue
        require((application_root / rel).is_file() and (baseline / rel).is_file(), 'Missing preapply input: ' + rel)
        require((application_root / rel).read_bytes() == (baseline / rel).read_bytes(),
                'Preapply production differs from old Oct8: ' + rel)
        rows.append({'path': rel, 'sha256': sha(baseline / rel)})
    require(len(rows) == audit['source_checks']['tracked_build_inputs_checked'], 'Preapply build input count differs')
    selected_before = []
    for source in selection['source_files']:
        require(sha(application_root / source['path']) == source['baseline_sha256'], 'Preapply selected baseline differs')
        require(sha(candidate / source['path']) == source['tested_sha256'] == sha(source['tested_source']),
                'Selected candidate snapshot changed')
        selected_before.append({'path': source['path'], 'before_sha256': source['baseline_sha256'],
                                'expected_after_sha256': source['tested_sha256']})
    directories = [ROOT / 'reports/opus_bound_analysis_20261007', ROOT / 'reports/opus_resume_20261008']
    artifacts = []
    file_sets = {}
    for directory in directories:
        files = sorted(p for p in directory.rglob('*') if p.is_file())
        file_sets[str(directory)] = [str(p.relative_to(directory)) for p in files]
        artifacts.extend({'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files)
    require(sha(audit['baseline_binary']) == audit['baseline_binary_sha256'] and
            sha(audit['candidate_binary']) == audit['candidate_binary_sha256'], 'Preapply official libraries changed')
    git(application_root, 'diff', '--check')
    return {'status': 'passed_preapply_production_exact_Oct8_baseline',
            'production_root': str(application_root), 'baseline_source_root': str(baseline),
            'selected_worktree': str(candidate), 'source_head': audit['source_head'],
            'identity_audit_sha256': sha(HERE / 'identity_audit.json'),
            'selection_sha256': sha(audit['selection_path']), 'build_inputs_checked': len(rows),
            'production_build_input_inventory': rows, 'selected_source_before_after': selected_before,
            'historical_artifacts': artifacts, 'historical_artifact_file_sets': file_sets,
            'historical_artifact_count': len(artifacts),
            'baseline_binary_sha256': audit['baseline_binary_sha256'],
            'selected_binary_sha256': audit['candidate_binary_sha256'],
            'limits': ['CPU read-only snapshot; no source application, build or GPU execution.',
                       'Historical directories include hidden files and Python bytecode; application audit requires exact preservation.']}


def review_documents():
    snapshot_path = HERE / 'authorized_documents_snapshot.json'
    snapshot = read(snapshot_path)
    preapply = read(HERE / 'source_preapply_snapshot.json')
    require(snapshot['status'] == 'passed_authorized_section12_document_snapshot_old_prefixes_preserved',
            'Authorized document snapshot incomplete')
    for path, expected in snapshot['evidence'].items():
        require(sha(path) == expected, 'Authorized document snapshot evidence changed')
    authorized = {row['path'] for row in snapshot['documents']}
    rows = []
    for row in snapshot['documents']:
        content = Path(row['path']).read_bytes()
        marker = row['section12_marker'].encode()
        require(content.count(marker) == 1 and content.index(marker) == row['protected_prefix_bytes'],
                'Authorized section12 boundary moved')
        require(hashlib.sha256(content[:row['protected_prefix_bytes']]).hexdigest() == row['protected_prefix_sha256'],
                'Protected pre-section12 bytes changed: ' + row['path'])
        rows.append({'path': row['path'], 'protected_prefix_bytes': row['protected_prefix_bytes'],
                     'protected_prefix_sha256': row['protected_prefix_sha256'], 'old_prefix_exact': True,
                     'section12_bytes': len(content) - row['protected_prefix_bytes'],
                     'section12_sha256': hashlib.sha256(content[row['protected_prefix_bytes']:]).hexdigest(),
                     'final_bytes': len(content), 'final_sha256': hashlib.sha256(content).hexdigest()})
    unchanged = 0
    for row in preapply['historical_artifacts']:
        if row['path'] not in authorized:
            require(sha(row['path']) == row['sha256'], 'Frozen artifact changed: ' + row['path'])
            unchanged += 1
    for directory, old in preapply['historical_artifact_file_sets'].items():
        now = [str(p.relative_to(directory)) for p in Path(directory).rglob('*') if p.is_file()]
        require(len(now) == len(old) == len(set(old)) and set(now) == set(old), 'Historical file set changed')
    require(unchanged == 4972 and len(rows) == 2, 'Final document artifact count differs')
    return {'status': 'passed_final_authorized_section12_documents_all_old_prefixes_and4972_artifacts_preserved',
            'documents': rows, 'frozen_artifacts_checked': unchanged,
            'historical_file_sets_exact': True, 'authorization': snapshot['authorization'],
            'snapshot_sha256': sha(snapshot_path),
            'limitations': ['Checks document byte preservation and final hashes; substantive final prose is reviewed separately.',
                            'No linked binary, source application or GPU rerun.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--application-root', type=Path)
    parser.add_argument('--preapply-root', type=Path)
    parser.add_argument('--documents-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and not output.exists(), 'Output must be a new formal_selected file')
    require(sum(bool(v) for v in (args.application_root, args.preapply_root, args.documents_only)) <= 1,
            'Choose one review mode')
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
              'new_GPU_execution': False, 'new_build_execution': False, 'production_modified': False,
              'review_script_sha256': sha(__file__)}
    try:
        result.update(review_documents() if args.documents_only else review_preapply(args.preapply_root) if args.preapply_root else
                      review_application(args.application_root) if args.application_root else review_identity_plan())
    except Exception as exc:
        result.update(status='failed_independent_review', error=type(exc).__name__ + ': ' + str(exc))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'error') if k in result}))
    return 0 if result['status'].startswith('passed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
