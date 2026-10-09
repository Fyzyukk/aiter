#!/usr/bin/env python3
"""CPU-only validation of the twelve final selected API smokes.

Reads each result's own physical-device claim epoch and checks the loaded
module, exact plan/command, signed numerical repetitions, guards and private
device identity. Does not rewrite plans, queues or GPU result files.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def main():
    plan_path = OUT / 'plans/formal_selected_smoke.json'
    queue_path = OUT / 'formal_selected_smoke_queue.json'
    cpu_path = OUT / 'formal_selected/smoke_plan_cpu_audit.json'
    identity_path = OUT / 'formal_selected/identity_audit.json'
    plan, queue, cpu, identity = map(read, [plan_path, queue_path, cpu_path, identity_path])
    require(len(plan['targets']) == len(queue['commands']) == cpu['target_count'] == 12,
            'expected exactly twelve planned smoke cases')
    require(cpu['status'] == 'passed_cpu_smoke_plan_identity_no_gpu_execution', 'CPU plan audit not passed')
    require(cpu['plan_sha256'] == sha(plan_path) and cpu['queue_sha256'] == sha(queue_path), 'CPU plan or queue changed')
    require(cpu['formal_identity_audit_sha256'] == sha(identity_path), 'formal identity audit changed')
    official = {'path': str(Path(plan['official_binary']).resolve()), 'sha256': plan['official_binary_sha256']}
    require(sha(Path(official['path'])) == official['sha256'] == identity['candidate_binary_sha256'],
            'current formal binary differs from audited binary')
    require(plan['source_head'] == identity['source_head'] and plan['current_parent_count'] == 26,
            'formal source HEAD or parent pool differs')
    require(Path(queue['env']['AITER_JIT_DIR']).resolve() == Path(official['path']).parent,
            'queue JIT selection differs')
    spec = importlib.util.spec_from_file_location('formal_audit_cpu', OUT / 'audit_formal_candidate.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    formal_entries = {v['candidate']['name']: v['candidate']
                      for p in identity['parents'] for v in p['variants']}
    private_entries = {}
    claims = [json.loads(line) for line in (OUT / 'gpu_claim_log.jsonl').read_text().splitlines()]
    runner_sha = sha(OUT / 'official_smoke.py')
    experiment_sha = sha(OUT / 'experiment_runner.py')
    rows, label_calls, changed_entries, unchanged_entries = [], Counter(), set(), set()
    for index, (target, command, cpu_row) in enumerate(zip(plan['targets'], queue['commands'], cpu['rows'])):
        path = Path(command['argv'][command['argv'].index('--output') + 1])
        result = read(path)
        require(path.stem == command['name'], 'result name/command mismatch')
        require(result['status'] == 'passed' and not result.get('contamination'), f'{path.name}: incomplete or contaminated')
        require(result['plan'] == plan and result['source_head'] == plan['source_head'], 'recorded plan/HEAD differs')
        require(result['official_binary_sha256'] == official['sha256'], 'recorded official SHA differs')
        require(result['runner_sha256'] == runner_sha and result['experiment_runner_sha256'] == experiment_sha,
                'runner files changed since execution')
        require(result['profiling_only'] is False and result['finished'] > result['started'], 'invalid execution epoch')
        require(len(result['rows']) == 1, 'expected one executed target per command')
        row = result['rows'][0]
        require(row['target_index'] == index == cpu_row['target_index'] and row['kid'] == target['kid'] == cpu_row['parent_id']
                and row['shape'] == target['shape'] == cpu_row['shape'] and row['split'] == 1,
                'actual row/plan/CPU branch mismatch')
        require(target['signed'] is True and target['seed'] == 17 and row['purpose'] == target['purpose'],
                'input contract differs')
        require(row['timings'] == [] and row['actual_official_module'] == official, 'check-only or actual loaded module differs')
        expected_libraries = {'official': official}
        private_checks = []
        for check in cpu_row['private_checks']:
            label, private_path = check['label'], Path(check['path'])
            require(label in target['private_baselines'] and target['private_baselines'][label] == str(private_path),
                    'private plan identity differs')
            require(sha(private_path) == check['sha256'], 'private binary changed since CPU plan audit')
            expected_libraries[label] = {'path': str(private_path), 'sha256': check['sha256']}
            if private_path not in private_entries:
                private_entries[private_path] = {r['name']: r for r in audit.rows(private_path)}
            symbol = cpu_row['actual_candidate_symbol']
            require(symbol in private_entries[private_path] and symbol in formal_entries, 'private branch absent')
            exact = audit.identity(formal_entries[symbol], private_entries[private_path][symbol])
            require(audit.all_matched(exact) and exact == check['device_identity'], 'private instruction/metadata/descriptor identity differs')
            private_checks.append({'label': label, 'path': str(private_path), 'sha256': check['sha256'],
                                   'exact_device_identity': exact})
        require(row['libraries'] == expected_libraries and set(row['correctness']) == set(expected_libraries),
                'executed library set differs')
        for label, check in row['correctness'].items():
            require(check['repetitions'] == 8 and check['errRatio'] == 0 and check['repeatable'] is True
                    and check['output_guards'] is True and check['workspace_guards'] is True,
                    'eight-repeat numerical/repeatability/guard checks incomplete')
            label_calls[label] += 8
        starts = [c for c in claims if c.get('event') == 'start' and c.get('command', {}).get('name') == command['name']
                  and c['time'] <= result['started']]
        require(starts, 'own root command start missing')
        start = max(starts, key=lambda c: c['time'])
        ends = [c for c in claims if c.get('event') == 'end' and c.get('name') == command['name']
                and c['time'] >= result['finished']]
        require(ends, 'own root command end missing')
        end = min(ends, key=lambda c: c['time'])
        require(start['command'] == command and end['returncode'] == 0 and end['contamination'] is False,
                'own executed command or clean end differs')
        claim = max((c for c in claims if c.get('event') == 'claimed' and c['time'] <= start['time']), key=lambda c: c['time'])
        require(not any(c.get('event') == 'claimed' and start['time'] < c['time'] < end['time'] for c in claims),
                'physical claim changed during command')
        owners = [c for c in claims if c.get('event') == 'owner_identity' and c.get('name') == command['name']
                  and start['time'] <= c['time'] <= end['time']]
        require(len(owners) == 1, 'final selected smoke requires one recorded owner identity')
        owner = owners[0]
        require(owner['new_host_pids'] == [owner['host_pid']] and owner['host_pid'] not in owner['baseline_host_pids']
                and owner['launcher_sha256'] == sha(OUT / 'owned_python_launch.py')
                and owner['method'] == 'KFD open + unique sysfs registration + same-mm runpy handshake',
                'final selected owner marker is not one unique host PID')
        monitors = [c for c in claims if c.get('event') == 'monitor' and owner['time'] <= c['time'] <= end['time']]
        require(monitors and all(c['child_pid'] == owner['inner_pid'] for c in monitors), 'final selected owner inner PID differs')
        gpu = result['gpu']
        require(gpu['pci_bdf'] == gpu['expected_pci_bdf'] == claim['gpu']['bdf']
                and gpu['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index']), 'physical GPU/visible ordinal mismatch')
        argv = command['argv']
        require('--check-only' in argv and '--profile' not in argv and '--event-confirm' not in argv
                and argv[argv.index('--repetitions') + 1] == '8' and argv[argv.index('--target-index') + 1] == str(index)
                and Path(argv[argv.index('--plan') + 1]) == plan_path, 'executed smoke protocol differs')
        symbol = cpu_row['actual_candidate_symbol']
        (changed_entries if cpu_row['changed'] else unchanged_entries).add(symbol)
        rows.append({'target_index': index, 'parent_id': row['kid'], 'shape': row['shape'],
                     'changed_entry': cpu_row['changed'], 'actual_branch_index': cpu_row['actual_branch_index'],
                     'actual_candidate_symbol': symbol, 'signed': True, 'seed': 17, 'split': 1,
                     'result': str(path.relative_to(ROOT)), 'result_sha256': sha(path),
                     'log_sha256': sha(Path(command['log'])), 'actual_official_module': row['actual_official_module'],
                     'correctness': row['correctness'], 'private_checks': private_checks,
                     'gpu': gpu, 'claim_epoch': {'claim': claim, 'start': start, 'end': end},
                     'owner_identity': owner,
                     'started_utc': utc(result['started']), 'finished_utc': utc(result['finished']),
                     'performance_measured': False})
    require(len(changed_entries) == 4 and len(unchanged_entries) == 3, 'changed/control device coverage differs')
    require(label_calls == Counter(official=96, compute_prologue=24, register_scoped=48, compute_prologue_baseline=8), 'numerical call counts differ')
    require('torch' not in sys.modules, 'CPU summary unexpectedly imported torch')
    report = {'status': 'passed_formal_candidate_gpu_api_smoke_cpu_audited',
              'new_gpu_execution': False, 'production_sources_modified': False, 'adopted': False,
              'plan_sha256': sha(plan_path), 'queue_sha256': sha(queue_path), 'cpu_plan_audit_sha256': sha(cpu_path),
              'formal_identity_audit_sha256': sha(identity_path), 'official_module': official,
              'runner_sha256': runner_sha, 'experiment_runner_sha256': experiment_sha,
              'summary': {'planned_and_completed_targets': 12, 'changed_device_entries_covered': len(changed_entries),
                          'unchanged_control_device_entries_covered': len(unchanged_entries),
                          'repetitions_per_label_and_target': 8, 'checked_numerical_calls_per_label': dict(label_calls),
                          'total_checked_numerical_calls': sum(label_calls.values()),
                          'physical_gpus': sorted({r['gpu']['pci_bdf'] for r in rows}),
                          'HIP_visible_ordinals': sorted({r['gpu']['HIP_VISIBLE_DEVICES'] for r in rows}),
                          'all_own_claim_epochs_clean': True, 'all_actual_loaded_module_identities_matched': True,
                          'all_private_device_identities_matched': True, 'all_numerical_repeatability_guards_passed': True,
                          'started_utc': min(r['started_utc'] for r in rows), 'finished_utc': max(r['finished_utc'] for r in rows),
                          'performance_measured': False},
              'rows': rows,
              'limits': ['check-only numerical/API identity evidence; no Event timings or performance claim.',
                         'Public GEMM-only rank2/native E8M0 API; no public rank3 support claim.',
                         'Four changed entries and three unchanged controls are covered; full support-domain sweeps remain partial.',
                         'Complete Event decisions have selected 9021 and three register aliases; production application still requires matching these retained source/binary identities.']}
    output = OUT / 'formal_selected/gpu_smoke_analysis.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(output), **report['summary']}))


if __name__ == '__main__':
    main()
