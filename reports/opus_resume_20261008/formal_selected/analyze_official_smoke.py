#!/usr/bin/env python3
"""CPU audit of eight completed Oct8 API smokes and their physical claim epochs.

Reads exact plans, immutable builds, loaded-module identities, private FUNC
identities, numerical repetitions and owner/monitor records. Runs no GPU work.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from audit_smoke_plan import audit_plan
from smoke_common import (BUILD, CLAIMS, CPU_AUDIT, EXPERIMENT, HERE, LABEL, LAUNCHER,
                          OFFICIAL, PLAN, PRIVATE, PRIVATE_AUDIT, QUEUE, QUEUE_LOG,
                          REPETITIONS, RUNNER, SOURCE, identity, offset_contract,
                          read, require, sha, write)

FORMAL_IDENTITY = HERE / 'identity_audit.json'
OUTPUT = HERE / 'gpu_smoke_analysis.json'


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def evidence(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def formal_gate(plan, cpu):
    formal = read(FORMAL_IDENTITY)
    require(formal['status'] == 'passed_pending_cpu_identity_no_adoption'
            and formal['cpu_only'] is True and formal['gpu_executed'] is False
            and formal['production_checkout_modified'] is False and formal['failures'] == [],
            'Formal full-device identity audit not passed')
    require(formal['candidate_binary'] == str(OFFICIAL)
            and formal['candidate_binary_sha256'] == plan['official_binary_sha256'] == sha(OFFICIAL)
            and formal['source_head'] == plan['source_head'], 'Formal candidate module/HEAD differs')
    require(formal['build_manifest_sha256'] == sha(BUILD)
            and formal['source_manifest_sha256'] == sha(SOURCE), 'Formal source/build manifests changed')
    summary = formal['summary']
    require(summary['public_parents'] == 26 and summary['actual_device_variants'] == 56
            and summary['changed_device_entries'] == 1 and summary['unchanged_device_entries'] == 55
            and summary['failure_count'] == 0, 'Formal full-parent coverage differs')
    for key in ['changed_entries_match_tested_private_candidate', 'unchanged_entries_match_Oct7_official_selected',
                '9022_unchanged', 'Oct7_register_alias_entries_unchanged', 'generated_host_launch_and_tus_equal']:
        require(summary[key] is True, f'Formal gate incomplete: {key}')
    require(formal['audit_script_sha256'] == sha(HERE / 'audit_formal_selected.py'), 'Formal audit script changed')
    for path, digest in formal['evidence_sha256'].items():
        require(sha(path) == digest, f'Formal referenced evidence changed: {path}')
    parents = {parent['parent_id']: parent for parent in formal['parents']}
    for entry in cpu['device_entries']:
        kid, symbol = entry['parent_id'], entry['official']['name']
        variants = [v for v in parents[kid]['variants'] if v['candidate']['name'] == symbol]
        require(len(variants) == 1 and variants[0]['changed'] == (kid == 9021)
                and variants[0]['identity_checks']['matches_expected'] is True
                and identity(variants[0]['candidate'], entry['official'])['matches'],
                f'Formal and smoke linked identity differ for {kid}')
    return formal


def claim_epoch(command, result, claims):
    starts = [record for record in claims if record.get('event') == 'start'
              and record.get('command', {}).get('name') == command['name']
              and record['time'] <= result['started']]
    require(starts, f"Own root command start missing: {command['name']}")
    start = max(starts, key=lambda record: record['time'])
    ends = [record for record in claims if record.get('event') == 'end'
            and record.get('name') == command['name'] and record['time'] >= result['finished']]
    require(ends, f"Own clean command end missing: {command['name']}")
    end = min(ends, key=lambda record: record['time'])
    require(start['command'] == command and end['returncode'] == 0 and end['contamination'] is False,
            'Own executed command or clean end differs')
    require(not any(record.get('event') == 'start' and start['time'] < record['time'] < end['time']
                    for record in claims), 'Another root command overlaps smoke execution')
    claimed = [record for record in claims if record.get('event') == 'claimed' and record['time'] <= start['time']]
    require(claimed, 'Preceding physical GPU claim missing')
    claim = max(claimed, key=lambda record: record['time'])
    require(not any(record.get('event') in ['claimed', 'released']
                    and start['time'] < record['time'] < end['time'] for record in claims),
            'Physical claim changed during command')
    owners = [record for record in claims if record.get('event') == 'owner_identity'
              and record.get('name') == command['name'] and start['time'] <= record['time'] <= end['time']]
    require(len(owners) == 1, 'One unique owner identity required per smoke command')
    owner = owners[0]
    require(owner['new_host_pids'] == [owner['host_pid']]
            and owner['host_pid'] not in owner['baseline_host_pids']
            and owner['launcher_sha256'] == sha(LAUNCHER)
            and owner['method'] == 'KFD open + unique sysfs registration + same-mm runpy handshake',
            'Owner handshake does not identify one unique host PID')
    monitors = [record for record in claims if record.get('event') == 'monitor'
                and owner['time'] <= record['time'] <= end['time']]
    require(monitors and all(record['child_pid'] == owner['inner_pid'] for record in monitors),
            'Owner monitor inner PID differs')
    require(all(all(process['pid'] == owner['host_pid'] for process in record['processes'])
                for record in monitors), 'Unexpected physical-GPU process observed during own claim')
    gpu = result['gpu']
    require(gpu['pci_bdf'] == gpu['expected_pci_bdf'] == claim['gpu']['bdf']
            and gpu['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index']),
            'Actual/expected/claimed physical GPU or visible ordinal differs')
    epoch = {'claim': claim, 'start': start, 'end': end}
    return epoch, owner, monitors


def analyze():
    plan, queue, cpu = read(PLAN), read(QUEUE), read(CPU_AUDIT)
    require(cpu['status'] == 'passed_cpu_smoke_plan_identity_no_gpu_execution'
            and cpu['target_count'] == 8, 'Expected passed eight-target CPU plan audit')
    require(cpu == audit_plan(), 'Fresh CPU plan/device audit differs from sealed CPU evidence')
    formal = formal_gate(plan, cpu)
    claims = [json.loads(line) for line in CLAIMS.read_text().splitlines() if line.strip()]
    require(claims and all(isinstance(record['time'], (int, float)) for record in claims)
            and all(a['time'] <= b['time'] for a, b in zip(claims, claims[1:])), 'Invalid claim-log time order')
    official = {'path': str(OFFICIAL), 'sha256': sha(OFFICIAL)}
    private = {'path': str(PRIVATE), 'sha256': sha(PRIVATE)}
    expected_libraries = {'official': official, LABEL: private}
    rows, label_calls = [], Counter()
    changed, controls = set(), set()
    for index, (target, command, cpu_row) in enumerate(zip(plan['targets'], queue['commands'], cpu['rows'])):
        argv = command['argv']
        path, log = Path(argv[argv.index('--output') + 1]), Path(command['log'])
        result = read(path)
        require(path.stem == command['name'] and result['status'] == 'passed'
                and not result.get('contamination'), f'{path.name}: incomplete or contaminated')
        require(result['plan'] == plan and result['source_head'] == plan['source_head']
                and result['official_binary_sha256'] == official['sha256'], 'Executed plan/source/module SHA differs')
        require(result['runner_sha256'] == sha(RUNNER)
                and result['experiment_runner_sha256'] == sha(EXPERIMENT), 'Executed runner source changed')
        require(result['profiling_only'] is False and result['finished'] > result['started']
                and len(result['rows']) == 1, 'Invalid check-only epoch or target count')
        row = result['rows'][0]
        require(row['target_index'] == index == cpu_row['target_index']
                and row['kid'] == target['kid'] == cpu_row['parent_id']
                and row['shape'] == target['shape'] == cpu_row['shape'] and row['split'] == 1
                and row['purpose'] == target['purpose'], 'Executed row/plan/scalar branch differs')
        require(target['signed'] is True and target['seed'] == 17 and row['timings'] == []
                and row['actual_official_module'] == official, 'Signed inputs/check-only/loaded module differs')
        require(row['libraries'] == expected_libraries
                and set(row['correctness']) == set(expected_libraries), 'Executed library set differs')
        for label, check in row['correctness'].items():
            require(check['repetitions'] == REPETITIONS and check['errRatio'] == 0
                    and check['repeatable'] is True and check['output_guards'] is True
                    and check['workspace_guards'] is True, 'Eight-repeat numerical/repeatability/guard check incomplete')
            label_calls[label] += REPETITIONS
        require('--check-only' in argv and '--profile' not in argv and '--event-confirm' not in argv
                and argv[argv.index('--repetitions') + 1] == '8'
                and argv[argv.index('--target-index') + 1] == str(index), 'Execution protocol differs')
        epoch, owner, monitors = claim_epoch(command, result, claims)
        (changed if cpu_row['changed'] else controls).add(cpu_row['actual_candidate_symbol'])
        rows.append({'target_index': index, 'parent_id': row['kid'], 'shape': row['shape'],
                     'changed_entry': cpu_row['changed'], 'actual_branch_index': cpu_row['actual_branch_index'],
                     'actual_candidate_symbol': cpu_row['actual_candidate_symbol'],
                     'signed': True, 'seed': 17, 'split': 1,
                     'result': str(path), 'result_sha256': sha(path), 'log': str(log), 'log_sha256': sha(log),
                     'evidence': {'result': evidence(path), 'log': evidence(log)},
                     'actual_official_module': row['actual_official_module'], 'correctness': row['correctness'],
                     'private_checks': cpu_row['private_checks'], 'offset_contract': offset_contract(),
                     'gpu': result['gpu'], 'claim_epoch': epoch, 'owner_identity': owner,
                     'monitor_count': len(monitors), 'all_monitored_gpu_processes_owned': True,
                     'started_utc': utc(result['started']), 'finished_utc': utc(result['finished']),
                     'performance_measured': False})
    require(len(rows) == 8 and len(changed) == len(controls) == 1
            and label_calls == Counter(official=64, scale_issue_publish=64), 'Eight-target128-call coverage differs')
    require('torch' not in sys.modules, 'CPU smoke analysis imported torch')
    summary = {'planned_and_completed_targets': 8, 'changed_device_entries_covered': 1,
               'unchanged_control_device_entries_covered': 1,
               'repetitions_per_label_and_target': REPETITIONS,
               'checked_numerical_calls_per_label': dict(label_calls),
               'total_checked_numerical_calls': sum(label_calls.values()),
               'physical_gpus': sorted({row['gpu']['pci_bdf'] for row in rows}),
               'HIP_visible_ordinals': sorted({row['gpu']['HIP_VISIBLE_DEVICES'] for row in rows}),
               'all_own_claim_epochs_clean': True, 'all_unique_owner_handshakes_matched': True,
               'all_actual_loaded_module_identities_matched': True,
               'all_private_device_identities_matched': True,
               'all_numerical_repeatability_guards_passed': True,
               'guarded_output_offset_bytes': 256, 'additional_input_offset_sweep': False,
               'started_utc': min(row['started_utc'] for row in rows),
               'finished_utc': max(row['finished_utc'] for row in rows), 'performance_measured': False}
    return {'status': 'passed_formal_selected_gpu_api_smoke_cpu_audited',
            'cpu_only_analysis': True, 'new_gpu_execution': False,
            'production_sources_modified': False, 'adopted': False,
            'source_head': plan['source_head'], 'official_module': official, 'private_module': private,
            'formal_identity_audit_sha256': sha(FORMAL_IDENTITY),
            'build_manifest_sha256': sha(BUILD), 'source_manifest_sha256': sha(SOURCE),
            'plan_sha256': sha(PLAN), 'queue_sha256': sha(QUEUE), 'cpu_plan_audit_sha256': sha(CPU_AUDIT),
            'private_device_audit_sha256': sha(PRIVATE_AUDIT), 'runner_sha256': sha(RUNNER),
            'experiment_runner_sha256': sha(EXPERIMENT), 'owner_launcher_sha256': sha(LAUNCHER),
            'claim_log_sha256': sha(CLAIMS), 'queue_log_sha256': sha(QUEUE_LOG),
            'analysis_script_sha256': sha(Path(__file__)),
            'analysis_helpers_sha256': {'audit_smoke_plan.py': sha(HERE / 'audit_smoke_plan.py'),
                                        'smoke_common.py': sha(HERE / 'smoke_common.py')},
            'evidence': {label: evidence(path) for label, path in {
                'plan': PLAN, 'queue': QUEUE, 'cpu_plan_audit': CPU_AUDIT, 'formal_identity_audit': FORMAL_IDENTITY,
                'build_manifest': BUILD, 'source_manifest': SOURCE, 'private_device_audit': PRIVATE_AUDIT,
                'runner': RUNNER, 'experiment_runner': EXPERIMENT, 'owner_launcher': LAUNCHER,
                'claim_log': CLAIMS, 'queue_log': QUEUE_LOG}.items()},
            'formal_identity_summary': formal['summary'], 'summary': summary,
            'offset_contract': offset_contract(), 'rows': rows,
            'limits': ['Check-only signed numerical/API identity evidence; no Event timing or performance claim.',
                       'Rank2 public GEMM/native E8M0 inputs; no public rank3 support claim.',
                       'Seven9021 shapes and one unchanged9022 control cover selected boundaries, not the complete support domain.',
                       'Output storage offset256 bytes is inferred from exact unchanged runner SHA and backing guards; no input-offset sweep.',
                       'Numerical API claims may use any clean physical GPU; Event comparison retains its own same-card requirement.']}


def main():
    report = analyze()
    write(OUTPUT, report)
    print(json.dumps({'status': report['status'], 'output': str(OUTPUT), **report['summary']}))


if __name__ == '__main__':
    main()
