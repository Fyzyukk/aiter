#!/usr/bin/env python3
"""Prepare CPU-validated 16-shape screens and all current-winner Event jobs."""
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
from shape_inventory import cpp_eval, load_cpu_registry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    registry, _ = load_cpu_registry()
    inventory_path = OUT / 'shape_inventory.json'
    scope_path = OUT / 'compute_adoption_scope.json'
    inventory = json.loads(inventory_path.read_text())
    scope = json.loads(scope_path.read_text())
    rows = inventory['rows']
    assert len(rows) == 745 and len({(r['M'], r['N'], r['K']) for r in rows}) == 745
    env = json.loads((OUT / 'confirmation_idle_queue.json').read_text())['env']
    libraries = {label: str(OUT / 'compute_prologue' / label / 'experiments.so')
                 for label in ['baseline', 'candidate']}
    screen_queue = {'env': env, 'commands': []}
    event_queue = {'env': env, 'commands': []}
    summary = []
    for kid, expected_support, expected_winners in [(9021, 735, 42), (9022, 691, 159)]:
        parent = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        domain = next(d for d in scope['parent_domains'] if d['parent_id'] == kid)
        declared = {tuple(s['shape']): s for s in domain['shapes']}
        support = [r for r in rows if registry.a8w8_mxscale_bpreshuffle_supports_shape(parent, r['M'], r['N'], r['K'])]
        support_shapes = {(r['M'], r['N'], r['K']) for r in support}
        assert support_shapes == set(declared) and len(support) == expected_support
        branches = inventory['parents'][str(kid)]['branches']
        targets = []
        for row in support:
            shape = (row['M'], row['N'], row['K'])
            actual = [b for b in branches if all(cpp_eval(c, *shape) for c in b['conditions'])]
            assert len(actual) == 1 and actual[0]['index'] == declared[shape]['branch_index']
            assert declared[shape]['changes']
            targets.append({'kid': kid, 'shape': list(shape), 'seed': 17, 'signed': True,
                            'actual_candidate_branch': actual[0]['branch_label'],
                            'current745_winner_parent': row['parent_id'],
                            'same_parent_current_winner': row['parent_id'] == kid,
                            'coverage': ['all supported current745 candidate domain']})
        screen_parts = []
        for part, first in enumerate(range(0, len(targets), 16)):
            name = f'compute_prologue_support_{kid}_part{part:02d}'
            path = OUT / 'plans' / 'compute_prologue_full_scope' / (name + '.json')
            write(path, {'libraries': libraries, 'workspace': False, 'targets': targets[first:first + 16]})
            result = OUT / 'results' / (name + '.json')
            command = {'name': name, 'argv': ['/opt/venv/bin/python3', str(OUT / 'experiment_runner.py'),
                       '--plan', str(path), '--output', str(result), '--rounds', '1', '--iters', '11',
                       '--repetitions', '8'], 'log': str(result.with_suffix('.log'))}
            screen_queue['commands'].append(command)
            screen_parts.append({'plan': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                 'targets': len(targets[first:first + 16]), 'first_target': first,
                                 'numerical_repetitions_per_side': 8, 'trace_rounds': 1,
                                 'trace_iterations': 11, 'event_confirmation': False})
        winners = [t for t in targets if t['same_parent_current_winner']]
        assert len(winners) == expected_winners
        event_parts = []
        for part, first in enumerate(range(0, len(winners), 16)):
            name = f'compute_prologue_winners_{kid}_part{part:02d}'
            path = OUT / 'plans' / 'compute_prologue_full_scope' / (name + '.json')
            subset = winners[first:first + 16]
            write(path, {'libraries': libraries, 'workspace': False, 'targets': subset})
            event_parts.append({'plan': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                'targets': len(subset), 'first_winner': first})
            # Each Event command executes one shape, allowing idle windows and resumptions.
            for local, target in enumerate(subset):
                job = f'compute_prologue_winner_{kid}_{first + local:03d}'
                result = OUT / 'results' / (job + '.json')
                event_queue['commands'].append({'name': job, 'argv': ['/opt/venv/bin/python3',
                    str(OUT / 'experiment_runner.py'), '--plan', str(path), '--output', str(result),
                    '--rounds', '1', '--iters', '51', '--repetitions', '8', '--target-index', str(local),
                    '--event-confirm'], 'log': str(result.with_suffix('.log'))})
        summary.append({'parent_id': kid, 'supported_shapes': len(targets), 'support_parts': screen_parts,
                        'current_winner_shapes': len(winners), 'winner_parts': event_parts,
                        'winner_commands': len(winners),
                        'all_support_and_branch_expressions_cpu_validated': True})
    assert len(screen_queue['commands']) == 90 and len(event_queue['commands']) == 201
    assert 'torch' not in sys.modules
    screen_path = OUT / 'compute_prologue_full_support_queue.json'
    event_path = OUT / 'compute_prologue_all_winners_event_queue.json'
    write(screen_path, screen_queue)
    write(event_path, event_queue)
    manifest = {'status': 'cpu_validated_plans_not_executed', 'gpu_imported_or_initialized': False,
                'source_head': inventory['source_head'], 'production_sources_modified': False,
                'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [inventory_path, scope_path]},
                'library_sha256': {label: sha(Path(path)) for label, path in libraries.items()},
                'part_size': 16, 'parents': summary,
                'support_screen_queue': str(screen_path.relative_to(ROOT)),
                'all_winners_event_queue': str(event_path.relative_to(ROOT)),
                'support_screen_commands': 90, 'support_target_parent_pairs': 1426,
                'winner_event_commands': 201,
                'protocols': {'support_screen': {'repetitions_per_side': 8, 'trace_rounds': 1,
                    'trace_iterations': 11, 'timing_use': 'screen obvious regressions only; no adoption gain claim'},
                    'winner_confirmation': {'repetitions_per_side': 8, 'trace_rounds': 1,
                    'trace_iterations': 51, 'event_rounds': 5, 'event_iters_per_graph': 51,
                    'address_pool': 'shared automatically sized rotation', 'order': 'alternating AB/BA',
                    'checks': 'all original and Event pool outputs numerical/repeatability/guards'}},
                'execution_note': 'Root serializes these jobs on idle physical GPUs; existing prior confirmations are not skipped by these plans.',
                'decision_note': 'Decide each parent from observed complete scope results. Screen regressions receive controlled Event confirmation; do not treat the screen as final performance evidence.'}
    manifest_path = OUT / 'compute_prologue_full_scope_manifest.json'
    write(manifest_path, manifest)
    print(json.dumps({'manifest': str(manifest_path), 'support_counts': [735, 691],
                      'screen_parts': [46, 44], 'winner_counts': [42, 159], 'winner_jobs': 201,
                      'GPU_run': False}))


if __name__ == '__main__':
    main()
