#!/usr/bin/env python3
"""CPU-only preparation of the exact 42-winner plus two-boundary Event plan."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESUME = ROOT.parent
OLD = RESUME.parent / 'opus_bound_analysis_20261007'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing to replace existing differing plan: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    parts = sorted((OLD / 'plans/compute_prologue_full_scope').glob('compute_prologue_winners_9021_part*.json'))
    targets = []
    for path in parts:
        targets.extend(json.loads(path.read_text())['targets'])
    expected = {tuple(t['shape']) for t in targets}
    assert len(parts) == 3 and len(targets) == len(expected) == 42
    assert all(t['kid'] == 9021 and t['signed'] and t['same_parent_current_winner'] for t in targets)
    inventory = json.loads((OLD / 'shape_inventory.json').read_text())
    assert expected == {(r['M'], r['N'], r['K']) for r in inventory['rows'] if r['parent_id'] == 9021}
    libraries = {side: str((ROOT / side / 'experiments.so').resolve()) for side in ('baseline', 'candidate')}
    audit = json.loads((ROOT / 'device_audit.json').read_text())
    assert audit['status'] == 'passed'
    assert {lib['label']: lib['input_sha256'] for lib in audit['libraries']} == {
        side: sha(Path(path)) for side, path in libraries.items()}
    new_targets = [{'kid': 9021, 'shape': list(shape), 'seed': 17, 'signed': True,
                    'same_parent_current_winner': True, 'coverage': ['all42_current9021_winners'],
                    'baseline_version': 'Oct7 selected'}
                   for shape in sorted(expected, key=lambda s: (s[2], s[0], s[1]))]
    boundaries = [(1, 128, 128), (15, 128, 256)]
    assert not expected.intersection(boundaries)
    new_targets.extend({'kid': 9021, 'shape': list(shape), 'seed': 17, 'signed': True,
                        'same_parent_current_winner': False,
                        'coverage': ['independent_boundary_repeat_after_screen'],
                        'baseline_version': 'Oct7 selected'} for shape in boundaries)
    assert len(new_targets) == len({tuple(t['shape']) for t in new_targets}) == 44
    plan_path = ROOT / 'confirmation_plan_20261008.json'
    output = RESUME / 'results/scale_issue_publish_confirmation_20261008.json'
    queue_path = ROOT / 'confirmation_queue_20261008.json'
    plan = {'workspace': False, 'libraries': libraries, 'targets': new_targets,
            'purpose': 'Exact42 current9021 winners plus two predeclared independent boundary repeats, current selected baseline versus scale issue-before-publish candidate'}
    command = {'name': 'scale_issue_publish_confirmation_20261008',
               'argv': ['/opt/venv/bin/python3', str(OLD / 'experiment_runner.py'),
                        '--plan', str(plan_path), '--output', str(output),
                        '--rounds', '5', '--iters', '51', '--repetitions', '8', '--event-confirm'],
               'log': str(output.with_suffix('.log'))}
    env = json.loads((RESUME / 'scale_issue_publish_queue.json').read_text())['env']
    write_new(plan_path, plan)
    write_new(queue_path, {'env': env, 'commands': [command]})
    manifest_path = ROOT / 'confirmation_manifest_20261008.json'
    manifest = {'status': 'cpu_prepared_not_gpu_executed', 'gpu_access': False,
                'baseline': 'Oct7 selected', 'source_head': inventory['source_head'],
                'winner_count': 42, 'boundary_repeat_count': 2, 'total_targets': 44,
                'winner_shapes': [list(s) for s in sorted(expected)],
                'independent_boundary_repeat_shapes': [list(s) for s in boundaries],
                'inputs': {str(p.resolve()): sha(p) for p in parts + [OLD / 'shape_inventory.json',
                           ROOT / 'device_audit.json', ROOT / 'build_manifest.json', ROOT / 'isa_order_audit.json']},
                'libraries': {side: {'path': path, 'sha256': sha(Path(path))} for side, path in libraries.items()},
                'plan': {'path': str(plan_path), 'sha256': sha(plan_path)},
                'queue': {'path': str(queue_path), 'sha256': sha(queue_path)},
                'runner_sha256': sha(OLD / 'experiment_runner.py'),
                'protocol': {'trace_rounds': 5, 'event_rounds': 5, 'iterations': 51,
                             'numerical_repetitions_per_side': 8, 'shared_pool_count': 51,
                             'event_order': 'AB/BA/AB/BA/AB',
                             'event_pool_output_checks_per_target': 510},
                'coverage_note': 'Three frozen Oct7 winner plans exactly agree with9021 actual winner inventory. Boundary repeats are separate nonwinner mechanism targets. Screen records remain independent and are not overwritten.',
                'prepare_script_sha256': sha(Path(__file__))}
    write_new(manifest_path, manifest)
    print(json.dumps({'status': manifest['status'], 'targets': 44, 'plan': str(plan_path),
                      'queue': str(queue_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
