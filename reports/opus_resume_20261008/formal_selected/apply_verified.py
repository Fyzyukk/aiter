#!/usr/bin/env python3
"""Apply only the verified Oct8 9021 header and preserve the evidence chain.

Run after the official identity and eight-target API audits pass. This script
does not build, run GPU work, edit documentation, commit, or push.
"""
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RESUME = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/opus_bound_analysis_20261007'
OLD_SELECTED = OLD / 'formal_selected'
HEADER = 'csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh'
SMALL = 'csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh'
SNAPSHOT = HERE / 'pre_apply_evidence_snapshot.json'
INTEGRATION = HERE / 'integration_manifest.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_text())


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def collect_refs(value, evidence):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = Path(value['path'])
            # The formal identity catalogue uses paths below blob.staging.
            if not path.is_absolute() and path.parts[0] in ('impl', 'instances'):
                require('..' not in path.parts, 'Invalid relative generated path')
                path = RESUME / 'jit_formal_selected/build/module_deepgemm_opus/blob.staging' / path
            if path.is_absolute():
                require(sha(path) == value['sha256'], 'Evidence SHA differs: ' + str(path))
                evidence[str(path)] = value['sha256']
        for item in value.values():
            collect_refs(item, evidence)
    elif isinstance(value, list):
        for item in value:
            collect_refs(item, evidence)


def main():
    require(not SNAPSHOT.exists() and not INTEGRATION.exists(), 'Application evidence already exists; review before rerunning')
    source, build = read(HERE / 'source_manifest.json'), read(HERE / 'build_manifest.json')
    identity, smoke = read(HERE / 'identity_audit.json'), read(HERE / 'gpu_smoke_analysis.json')
    old_source, old_integration = read(OLD_SELECTED / 'source_manifest.json'), read(OLD_SELECTED / 'integration_manifest.json')
    require('passed' in identity['status'] and 'passed' in smoke['status'], 'Official identity and API smoke must pass')
    require(not identity['failures'] and identity['summary']['failure_count'] == 0, 'Official identity failures remain')
    summary = identity['summary']
    for key, expected in {'public_parents': 26, 'actual_device_variants': 56,
                          'changed_device_entries': 1, 'unchanged_device_entries': 55,
                          'build_objects_checked': 206, 'device_objects_checked': 202,
                          'linked_gfx950_bundles_checked': 202}.items():
        require(summary[key] == expected, 'Official identity scope differs: ' + key)
    for key in ('changed_entries_match_tested_private_candidate', 'unchanged_entries_match_Oct7_official_selected',
                '9022_unchanged', 'Oct7_register_alias_entries_unchanged', 'generated_host_launch_and_tus_equal',
                'normalized_build_ninja_equal'):
        require(summary[key] is True, 'Official identity check failed: ' + key)
    entries = [(p['parent_id'], v) for p in identity['parents'] for v in p['variants']]
    require(len(entries) == 56 and [kid for kid, v in entries if v['changed']] == [9021]
            and all(v['identity_checks']['matches_expected'] is True for _, v in entries), 'Expected one changed9021 and55 unchanged entries')
    api = smoke['summary']
    require(api['planned_and_completed_targets'] == 8 and api['repetitions_per_label_and_target'] == 8
            and api['checked_numerical_calls_per_label'] == {'official': 64, 'scale_issue_publish': 64}
            and api['total_checked_numerical_calls'] == 128 and len(smoke['rows']) == 8
            and api['changed_device_entries_covered'] == api['unchanged_control_device_entries_covered'] == 1,
            'Expected eight API targets and128 numerical calls')
    for key in ('all_own_claim_epochs_clean', 'all_unique_owner_handshakes_matched',
                'all_actual_loaded_module_identities_matched', 'all_private_device_identities_matched',
                'all_numerical_repeatability_guards_passed'):
        require(api[key] is True, 'API smoke check failed: ' + key)
    require(api['performance_measured'] is False, 'API smoke cannot supply performance evidence')

    candidate = Path(source['candidate_worktree']).resolve()
    old_tree = Path(old_source['candidate_worktree']).resolve()
    head = source['source_head']
    require(Path(source['production_checkout']).resolve() == ROOT and candidate != ROOT and candidate != old_tree,
            'Unexpected production or candidate checkout')
    require(all(git(tree, 'rev-parse', 'HEAD').decode().strip() == head for tree in (ROOT, candidate, old_tree)), 'Source HEAD changed')
    require(old_integration['status'] == 'applied_verified_not_committed' and old_integration['adopted'] is True,
            'Baseline must remain Oct7 adopted selected')
    files = {item['path']: item for item in source['files']}
    require(set(files) == {HEADER, SMALL} and source['changed_files_vs_current'] == [HEADER], 'Application source scope differs')
    require(build['status'] == 'passed' and build['source_head'] == identity['source_head'] == smoke['source_head'] == head
            and Path(build['source_root']).resolve() == candidate and build['pending_candidate_no_adoption'] is True,
            'Official build/source identity differs')
    module = {'path': build['binary'], 'sha256': build['binary_sha256']}
    require(sha(module['path']) == module['sha256'] == identity['candidate_binary_sha256']
            and identity['candidate_binary'] == module['path'] and smoke['official_module'] == module
            and all(row['actual_official_module'] == module for row in smoke['rows']), 'Tested official module identity differs')
    require(identity['source_manifest_sha256'] == smoke['source_manifest_sha256'] == sha(HERE / 'source_manifest.json')
            and identity['build_manifest_sha256'] == smoke['build_manifest_sha256'] == sha(HERE / 'build_manifest.json')
            and smoke['formal_identity_audit_sha256'] == sha(HERE / 'identity_audit.json'), 'Official audit chain changed')
    require(identity['audit_script_sha256'] == sha(HERE / 'audit_formal_selected.py')
            and smoke['analysis_script_sha256'] == sha(HERE / 'analyze_official_smoke.py'), 'Audit script changed')
    old_files = {item['path']: item for item in old_source['files']}
    applied = {item['path']: item for item in old_integration['source_files']}
    before = (ROOT / HEADER).read_bytes()
    replacement = (candidate / HEADER).read_bytes()
    for path, item in files.items():
        require(sha(ROOT / path) == item['checkout_current_sha256'] == old_files[path]['candidate_sha256']
                == applied[path]['applied_sha256'] == sha(old_tree / path), 'Current Oct7 selected source differs: ' + path)
        require(sha(candidate / path) == item['candidate_sha256'] == sha(item['private_candidate']), 'Tested candidate source differs: ' + path)
        require(digest(git(ROOT, 'show', head + ':' + path)) == item['original_sha256'], 'HEAD source differs: ' + path)
    require(files[SMALL]['candidate_sha256'] == files[SMALL]['checkout_current_sha256'], 'Retained register traits changed')
    joint = git(candidate, 'diff', '--binary')
    require(joint == Path(source['source_diff_path']).read_bytes() == build['source_diff'].encode()
            and digest(joint) == source['source_diff_sha256'] == identity['source_checks']['source_diff_sha256'], 'Tested joint source diff differs')
    selected = ''.join(difflib.unified_diff(before.decode().splitlines(keepends=True), replacement.decode().splitlines(keepends=True),
                                          fromfile='a/' + HEADER, tofile='b/' + HEADER)).encode()
    require(selected == Path(source['selected_diff_path']).read_bytes() == Path(source['production_diff_path']).read_bytes()
            and digest(selected) == source['selected_diff_sha256'] == source['production_diff_sha256']
            == identity['source_checks']['selected_diff_sha256'], 'Exact change from Oct7 selected differs')
    require(set(git(candidate, 'diff', '--name-only').decode().splitlines()) == {HEADER, SMALL}, 'Unexpected candidate HEAD diff')
    require(digest(git(old_tree, 'diff', '--binary')) == old_source['source_diff_sha256'], 'Oct7 selected worktree changed')
    require(set(git(ROOT, 'diff', '--name-only', '--', 'aiter', 'csrc', '3rdparty').decode().splitlines()) == {HEADER, SMALL},
            'Unexpected production build-source diff')
    git(candidate, 'diff', '--check')
    git(ROOT, 'apply', '--check', str(Path(source['selected_diff_path'])))

    # Protect all tracked build inputs, including traits,9022,9000,9020 and helpers.
    protected, inventory = {}, []
    for row in git(candidate, 'ls-files', '--stage', '-z').decode().split('\0'):
        if not row:
            continue
        mode_and_blob, path = row.split('\t', 1)
        if mode_and_blob.split()[0] == '160000' or not path.startswith(('aiter/', 'csrc/', '3rdparty/')):
            continue
        old_hash, new_hash = sha(old_tree / path), sha(candidate / path)
        require(sha(ROOT / path) == old_hash and (path == HEADER or new_hash == old_hash), 'Protected build input differs: ' + path)
        protected[path] = old_hash
        inventory.append({'path': path, 'selected_sha256': old_hash, 'candidate_sha256': new_hash})
    require(digest(json.dumps(inventory, sort_keys=True).encode()) == identity['source_checks']['build_source_inventory_sha256'],
            'Official source inventory changed since identity audit')

    evidence = {}
    required_paths = [HERE / name for name in ('source_manifest.json', 'source.diff', 'changes_vs_selected.diff',
                     'build_manifest.json', 'identity_audit.json', 'gpu_smoke_analysis.json', 'apply_verified.py')]
    confirmation_path = RESUME / 'results/scale_issue_publish_confirmation_analysis.json'
    screen_path = RESUME / 'results/scale_issue_publish_screen_analysis.json'
    confirmation, screen = read(confirmation_path), read(screen_path)
    required_paths += [confirmation_path, screen_path, OLD_SELECTED / 'integration_manifest.json']
    for path in required_paths:
        evidence[str(path)] = sha(path)
    for report in (source, identity, smoke, confirmation, screen):
        collect_refs(report, evidence)
    for path, expected in identity['evidence_sha256'].items():
        require(sha(path) == expected, 'Historical identity evidence changed: ' + path)
        evidence[path] = expected
    for path, expected in confirmation['winner_coverage']['frozen_inputs'].items():
        require(sha(path) == expected, 'Frozen winner plan changed: ' + path)
        evidence[path] = expected
    require('passed' in confirmation['status'] and 'passed' in screen['status']
            and confirmation['winner_coverage']['complete_exact42'] is True
            and confirmation['winner_coverage']['covered'] == confirmation['actual_winner_summary']['shape_count'] == 42
            and confirmation['winner_coverage']['remaining'] == []
            and all(row['all_checks_passed'] is True for report in (confirmation, screen) for row in report['rows']),
            'Complete winner or boundary Event evidence missing')
    costs = []
    for shape in ([1, 128, 128], [15, 128, 256]):
        windows = {}
        for label, report in (('screen', screen), ('confirmation', confirmation)):
            row = next(row for row in report['rows'] if row['shape'] == shape)
            require(row['current745_same_parent_winner'] is False, 'Boundary cost must remain a nonwinner observation')
            windows[label] = {key: row[key] for key in ('event_median_us', 'candidate_time_change_percent',
                                                       'speedup_percent', 'candidate_faster_rounds', 'candidate_slower_rounds')}
        costs.append({'shape': shape, 'accepted_finite_nonwinner_cost': True, 'windows': windows})
    evidence_rows = [{'path': path, 'sha256': expected} for path, expected in sorted(evidence.items())]
    production_diff = git(ROOT, 'diff', '--binary')
    source_diff_before = git(ROOT, 'diff', '--binary', '--', HEADER, SMALL)
    snapshot = {'status': 'verified_before_production_source_application', 'timestamp_utc': utc(), 'source_head': head,
                'production_checkout': str(ROOT), 'candidate_worktree': str(candidate),
                'tracked_diff_before': production_diff.decode(), 'tracked_diff_before_sha256': digest(production_diff),
                'selected_sources_diff_before_sha256': digest(source_diff_before), 'source_files': source['files'],
                'protected_build_inputs_count': len(protected),
                'protected_build_inputs_sha256': digest(json.dumps(protected, sort_keys=True).encode()),
                'protected_sources': {path: value for path, value in protected.items()
                                      if path.startswith('csrc/opus_gemm/include/')},
                'official_module': module, 'formal_smoke_summary': api, 'evidence': evidence_rows,
                'historical_evidence_policy': 'Earlier adopted=false and not-built fields retain their collection-time meanings.'}
    write_new(SNAPSHOT, snapshot)
    require((ROOT / HEADER).read_bytes() == before and (candidate / HEADER).read_bytes() == replacement
            and all(sha(path) == expected for path, expected in evidence.items()), 'Source or evidence changed before application')
    target = ROOT / HEADER
    require(target.is_file() and not target.is_symlink(), 'Target must be a regular source file')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + target.name + '.', dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(replacement)
            os.fchmod(handle.fileno(), stat.S_IMODE(target.stat().st_mode))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        require(sha(target) == files[HEADER]['candidate_sha256'] and target.read_bytes() == replacement, 'Applied source differs from tested candidate')
        require(all(sha(ROOT / path) == expected for path, expected in protected.items() if path != HEADER), 'Protected source changed during application')
        require(git(ROOT, 'diff', '--binary', '--', HEADER, SMALL) == joint, 'Applied HEAD diff differs from tested worktree')
        require(git(candidate, 'diff', '--binary') == joint and all(sha(path) == expected for path, expected in evidence.items()), 'Tested source or evidence changed during application')
        git(ROOT, 'diff', '--check', '--', HEADER, SMALL)
    except Exception:
        if target.read_bytes() == replacement:
            target.write_bytes(before)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    integration = {'status': 'applied_verified_not_committed', 'timestamp_utc': utc(), 'source_head': head,
                   'production_checkout': str(ROOT), 'candidate_worktree': str(candidate), 'adopted': True,
                   'production_sources_modified': True, 'accepted_parent_ids_this_change': [9021],
                   'retained_Oct7_parent_ids': [9021, 9042, 9053, 9054],
                   'source_files': [{'path': HEADER, 'baseline_sha256': files[HEADER]['checkout_current_sha256'],
                                     'applied_sha256': sha(target), 'matches_tested_selected_source': True}],
                   'changed_source_files_this_application': 1, 'changed_source_files_relative_to_HEAD': 2,
                   'changed_device_entries': 1, 'unchanged_device_entries': 55,
                   'device_identity_baseline': 'Oct7 adopted official selected',
                   'source_diff_sha256': source['source_diff_sha256'], 'source_diff_path': source['source_diff_path'],
                   'selected_diff_sha256': source['selected_diff_sha256'], 'selected_diff_path': source['selected_diff_path'],
                   'official_module': module, 'formal_identity_summary': summary, 'formal_smoke_summary': api,
                   'actual_winner_summary': confirmation['actual_winner_summary'],
                   'actual_winner_coverage': confirmation['winner_coverage'], 'accepted_nonwinner_costs': costs,
                   'adoption_decision': 'Retain global9021 issue/publish for the validated actual-winner improvement and accept the two independently repeated finite nonwinner boundary costs.',
                   'support_domain_performance_coverage': 'partial; no claim that every supported shape improves and no new K selection rule',
                   'protected_build_inputs_verified_unchanged_except_9021': True,
                   'pre_apply_snapshot': {'path': str(SNAPSHOT), 'sha256': sha(SNAPSHOT)}, 'evidence': evidence_rows,
                   'validation_reuse': 'The applied header exactly matches the Event-tested private source and the formally built/tested isolated source; source application needs no new GPU run.',
                   'default_jit_cache_note': 'The verified module resides in the recorded isolated JIT directory; future default builds must use the applied source and recorded compiler/flags.',
                   'git_diff_check_passed': True, 'commit_created': False, 'pushed': False,
                   'documents_finalized': False, 'task_complete': False,
                   'final_document_audit_pending': True}
    write_new(INTEGRATION, integration)
    print(json.dumps({'status': integration['status'], 'integration_manifest': str(INTEGRATION),
                      'applied_sha256': sha(target), 'changed_device_entries': 1, 'unchanged_device_entries': 55,
                      'API_targets': 8, 'numerical_calls': 128, 'commit_created': False, 'pushed': False}))


if __name__ == '__main__':
    main()
