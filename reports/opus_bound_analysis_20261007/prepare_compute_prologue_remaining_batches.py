#!/usr/bin/env python3
"""CPU-create immutable <=8-target same-parent remaining winner Event batches.

Copies exact original targets/libraries/workspace from the 201-winner queue,
deduplicates a strict coverage snapshot, and writes new uniquely named plans
and queue. Never modifies existing plans/queues or runs GPU.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
from shape_inventory import cpp_eval, load_cpu_registry


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def key(target):
    return (target['kid'], *target['shape'])


def write_new(path, value):
    require(not path.exists(), f'new artifact already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--max-targets', type=int, default=8)
    args = parser.parse_args()
    require(1 <= args.max_targets <= 8, 'batch must contain one to eight targets')
    require(args.tag.replace('_', '').replace('-', '').isalnum(), 'tag must be filename-safe')
    coverage_path = OUT / 'results/compute_prologue_winner_event_coverage.json'
    source_path = OUT / 'compute_prologue_all_winners_event_short_k_queue.json'
    coverage, source = read(coverage_path), read(source_path)
    coverage_bytes = coverage_path.read_bytes()
    require(coverage['remaining_queue_written'] is False, 'coverage must have been refreshed summary-only')
    covered = {(r['parent_id'], *r['shape']) for r in coverage['covered_winners']}
    remaining = {(r['parent_id'], *r['shape']) for r in coverage['remaining_winners']}
    require(len(covered) + len(remaining) == 201 and not (covered & remaining), 'coverage partition differs')
    registry, _ = load_cpu_registry()
    inventory = read(OUT / 'shape_inventory.json')
    require(inventory['source_head'] == coverage['source_head'], 'inventory HEAD mismatch')
    all_original = {}
    original_files = {source_path: sha(source_path)}
    libraries = None
    selected = {9021: [], 9022: []}
    for command in source['commands']:
        argv = command['argv']
        plan_path = Path(argv[argv.index('--plan') + 1])
        plan = read(plan_path)
        original_files[plan_path] = sha(plan_path)
        target_index = int(argv[argv.index('--target-index') + 1])
        target = plan['targets'][target_index]
        shape_key = key(target)
        require(shape_key not in all_original and target['kid'] in selected, 'original target duplicate or parent mismatch')
        require(plan['workspace'] is False, 'original workspace changed')
        if libraries is None:
            libraries = plan['libraries']
        require(plan['libraries'] == libraries, 'original library pair differs')
        require(target['signed'] is True and target['seed'] == 17, 'remaining target contract differs')
        require('--event-confirm' in argv and argv[argv.index('--iters') + 1] == '51'
                and argv[argv.index('--repetitions') + 1] == '8' and argv[argv.index('--rounds') + 1] == '1',
                'original Event protocol differs')
        all_original[shape_key] = {'target': target, 'source_plan': str(plan_path), 'source_plan_sha256': sha(plan_path),
                                   'source_target_index': target_index, 'source_command': command['name']}
        if shape_key in remaining:
            selected[target['kid']].append(all_original[shape_key])
    require(set(all_original) == covered | remaining, 'source 201 domain differs from coverage')
    require(sum(map(len, selected.values())) == len(remaining), 'remaining selection differs')
    for label, path in libraries.items():
        require(coverage['libraries'][label] == {'path': path, 'sha256': sha(Path(path))}, 'library SHA mismatch')
    batches, commands = [], []
    plan_dir = OUT / 'plans' / ('compute_prologue_remaining_batches_' + args.tag)
    queue_path = OUT / ('compute_prologue_remaining_batches_' + args.tag + '_queue.json')
    manifest_path = OUT / ('compute_prologue_remaining_batches_' + args.tag + '_manifest.json')
    snapshot_path = OUT / ('compute_prologue_remaining_batches_' + args.tag + '_coverage_snapshot.json')
    require(not plan_dir.exists() and not queue_path.exists() and not manifest_path.exists() and not snapshot_path.exists(),
            'batch output tag already exists')
    for kid, records in selected.items():
        for offset in range(0, len(records), args.max_targets):
            entries = records[offset:offset + args.max_targets]
            name = f'compute_prologue_batch_{args.tag}_{kid}_{offset // args.max_targets:02d}'
            path = plan_dir / (name + '.json')
            plan = {'libraries': libraries, 'workspace': False, 'targets': [r['target'] for r in entries]}
            write_new(path, plan)
            result_path = OUT / 'results' / (name + '.json')
            command = {'name': name, 'argv': [source['commands'][0]['argv'][0], str(OUT / 'experiment_runner.py'),
                '--plan', str(path), '--output', str(result_path), '--rounds', '1', '--iters', '51',
                '--repetitions', '8', '--event-confirm'], 'log': str(result_path.with_suffix('.log'))}
            require('--target-index' not in command['argv'], 'batch must execute every target')
            audits = []
            for index, entry in enumerate(entries):
                target = plan['targets'][index]
                require(target == all_original[key(target)]['target'], 'copied target changed')
                m, n, k = target['shape']
                require(registry.a8w8_mxscale_bpreshuffle_supports_shape(
                    registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid], m, n, k), 'current support rejects target')
                inv = [r for r in inventory['rows'] if (r['parent_id'], r['M'], r['N'], r['K']) == key(target)]
                require(len(inv) == 1, 'target is not exactly one current winner')
                branches = [b for b in inventory['parents'][str(kid)]['branches']
                            if all(cpp_eval(c, m, n, k) for c in b['conditions'])]
                require(len(branches) == 1, 'generated host branch ambiguous')
                audits.append({'target_index': index, 'parent_id': kid, 'shape': target['shape'],
                               'target_exactly_equal_original': True, 'support_contract_passed': True,
                               'actual_branch_index': branches[0]['index'], 'actual_traits': branches[0]['traits_cpp'],
                               **{k: entry[k] for k in ['source_plan', 'source_plan_sha256', 'source_target_index', 'source_command']}})
            batches.append({'name': name, 'parent_id': kid, 'target_count': len(entries), 'plan': str(path),
                            'plan_sha256': sha(path), 'result': str(result_path), 'targets': audits})
            commands.append(command)
    queue = {'env': source['env'], 'commands': commands}
    write_new(queue_path, queue)
    snapshot_path.write_bytes(coverage_bytes)
    expected = {key(r['target']) for rows in selected.values() for r in rows}
    actual = {(t['parent_id'], *t['shape']) for b in batches for t in b['targets']}
    require(expected == actual == remaining and sum(b['target_count'] for b in batches) == len(remaining),
            'batched domain changed or duplicated')
    for path, digest in original_files.items():
        require(sha(path) == digest, 'original source plan or queue changed')
    confirmation = read(OUT / 'plans/compute_prologue_confirmation_priority.json')['targets'][6]
    confirmation_key = key(confirmation)
    require(confirmation_key in all_original, 'confirmation6 is not a current winner')
    confirmation_resolution = ('already_covered_clean' if confirmation_key in covered else 'included_once_in_remaining_winner_batch')
    require('torch' not in sys.modules, 'CPU plan generation imported torch')
    manifest = {'status': 'passed_cpu_remaining_batch_scope_identity_no_gpu_execution',
        'new_gpu_execution': False, 'original_plans_and_queues_modified': False,
        'source_head': coverage['source_head'], 'tag': args.tag, 'max_targets_per_batch': args.max_targets,
        'coverage_snapshot': str(snapshot_path), 'coverage_snapshot_sha256': sha(snapshot_path),
        'source_queue': str(source_path), 'source_queue_sha256': sha(source_path),
        'queue': str(queue_path), 'queue_sha256': sha(queue_path),
        'runner_sha256': sha(OUT / 'experiment_runner.py'), 'libraries': coverage['libraries'],
        'summary': {'expected_winner_domain': 201, 'covered_at_snapshot': len(covered), 'remaining_winner_shapes': len(remaining),
                    'remaining_parent_counts': dict(Counter(k[0] for k in remaining)), 'batch_commands': len(commands),
                    'batch_size_distribution': dict(Counter(b['target_count'] for b in batches)),
                    'one_target_command_starts_avoided': len(remaining) - len(commands),
                    'settle_time_avoided_if_3seconds_each': 3 * (len(remaining) - len(commands)),
                    'all_targets_exact_original': True, 'all_library_workspace_contracts_exact_original': True,
                    'all_current_support_and_generated_branch_checks_passed': True},
        'confirmation6': {'parent_id': confirmation['kid'], 'shape': confirmation['shape'], 'resolution': confirmation_resolution,
                          'extra_duplicate_command_added': False},
        'batches': batches,
        'protocol': 'Each target retains signed8 numerical/repeatability/guards and shared-pool51 calls × 5 AB/BA Event; original runner unchanged. One Python/HIP context per <=8 same-parent targets.',
        'limits': ['Snapshot may include newly completed targets if another queue executes after generation; coverage deduplicates parent+shape. Filter only before starting, not by rewriting active plans.',
                   'If a batch is interrupted or contaminated, its entire result remains excluded from final Event evidence; <=8 bounds lost work.',
                   'All shape data/reference/pools are cleaned between targets by the unchanged runner; Event warmup/order/barriers are unmodified.',
                   'CPU plan identity is not GPU correctness or speed evidence.']}
    write_new(manifest_path, manifest)
    print(json.dumps({'manifest': str(manifest_path), 'queue': str(queue_path), **manifest['summary'],
                      'confirmation6': manifest['confirmation6']}))


if __name__ == '__main__':
    main()
