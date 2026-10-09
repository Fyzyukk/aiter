#!/usr/bin/env python3
"""CPU preparation of controlled Event checks for clean support trace signals.

Current-winner shapes remain in the complete 201-winner queue; completed
clean Event representatives are referenced instead of scheduling duplicate
work. Every other clean support-screen signal remains eligible, even when
the parent is only a fallback candidate for that shape.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
from shape_inventory import load_cpu_registry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def key(kid, shape):
    return (kid, *shape)


def verified_events(libraries):
    claims = []
    for line in (OUT / 'gpu_claim_log.jsonl').read_text().splitlines():
        try:
            claims.append(json.loads(line))
        except ValueError:
            pass
    found = {}
    for path in sorted((OUT / 'results').glob('*.json')):
        if '.attempt_' in path.name or path.name.endswith('_analysis.json'):
            continue
        try:
            result = read(path)
        except (OSError, ValueError):
            continue
        if (result.get('status') != 'passed' or result.get('contamination')
                or result.get('libraries') != libraries or not result.get('finished')):
            continue
        starts = [c for c in claims if c.get('event') == 'start'
                  and c.get('command', {}).get('name') == path.stem and c['time'] <= result['started']]
        ends = [c for c in claims if c.get('event') == 'end'
                and c.get('name') == path.stem and c['time'] >= result['finished']]
        if not starts or not ends:
            continue
        end = min(ends, key=lambda c: c['time'])
        if end.get('returncode') != 0 or end.get('contamination') is not False:
            continue
        for row in result.get('rows', []):
            event = row.get('event_confirmation', {})
            if (event.get('status') != 'passed' or event.get('rounds') != 5
                    or event.get('iters_per_graph') != 51 or not event.get('shared_pool')
                    or len(event.get('measurements', [])) != 10):
                continue
            checks_ok = all(c.get('errRatio') == 0 and c.get('output_guards') and c.get('workspace_guards')
                            and c.get('output_repeatable') for m in event['measurements'] for c in m['all_pool_checks'])
            if not checks_ok:
                continue
            found.setdefault(key(row['kid'], row['shape']), []).append({
                'result': str(path.relative_to(ROOT)), 'result_sha256': sha(path),
                'claim_end': end, 'event_median_us': event['median_us'],
                'event_speedup': event['median_speedup'], 'current_library_hashes_match': True})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-time-increase-percent', type=float, default=5.0)
    args = parser.parse_args()
    assert args.screen_time_increase_percent >= 0
    support_path = OUT / 'results/compute_prologue_support_analysis.json'
    support = read(support_path)
    assert support['status'] in ['partial_support_coverage_no_full_pass_claim', 'complete_clean_support_numerical_audit']
    libraries = support['library_identity']
    for lib in libraries.values():
        assert sha(Path(lib['path'])) == lib['sha256']
    winners_path = OUT / 'compute_prologue_all_winners_event_queue.json'
    winners = read(winners_path)
    winner_map = {}
    for c in winners['commands']:
        argv = c['argv']
        plan = read(Path(argv[argv.index('--plan') + 1]))
        target = plan['targets'][int(argv[argv.index('--target-index') + 1])]
        winner_map[key(target['kid'], target['shape'])] = c['name']
    assert len(winner_map) == 201
    completed = verified_events(libraries)
    parts = {p['name']: p for p in support['parts'] if p['status'] == 'clean_complete_slice'}
    selected = [r for r in support['clean_screen_rows']
                if r['trace_candidate_time_increase_percent'] >= args.screen_time_increase_percent]
    registry, _ = load_cpu_registry()
    targets = []
    dispositions = []
    for row in selected:
        part = parts[row['part']]
        source = ROOT / part['result']
        assert sha(source) == part['result_sha256']
        result = read(source)
        source_row = result['rows'][row['target_index']]
        assert source_row['kid'] == row['parent_id'] and source_row['shape'] == row['shape']
        identity = key(row['parent_id'], row['shape'])
        assert registry.a8w8_mxscale_bpreshuffle_supports_shape(
            registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[row['parent_id']], *row['shape'])
        evidence = {'part': row['part'], 'target_index': row['target_index'],
                    'result': part['result'], 'result_sha256': part['result_sha256'],
                    'plan': part['plan'], 'plan_sha256': part['plan_sha256'],
                    'clean_claim_end': part['claim_epoch']['end'],
                    'trace_baseline_us': row['trace_baseline_us'], 'trace_candidate_us': row['trace_candidate_us'],
                    'candidate_time_increase_percent': row['trace_candidate_time_increase_percent']}
        disposition = {'parent_id': row['parent_id'], 'shape': row['shape'],
                       'current745_winner_parent': row['current745_winner_parent'], 'screen_evidence': evidence}
        if identity in completed:
            disposition.update(action='already_clean_Event_measured', completed_Event=completed[identity])
        elif identity in winner_map:
            disposition.update(action='covered_by_required_201_winner_Event_queue', winner_command=winner_map[identity])
        else:
            target = {'kid': row['parent_id'], 'shape': row['shape'], 'seed': 17, 'signed': True,
                      'current745_winner_parent': row['current745_winner_parent'],
                      'same_parent_current_winner': row['same_parent_current_winner'],
                      'screen_evidence': evidence,
                      'purpose': 'controlled Event check of supported fallback regression risk; single trace does not prove regression'}
            targets.append(target)
            disposition.update(action='new_support_Event_followup', plan_target_index=len(targets) - 1)
        dispositions.append(disposition)
    targets.sort(key=lambda t: (t['shape'][2], -t['screen_evidence']['candidate_time_increase_percent'], t['kid'], t['shape']))
    indexes = {key(t['kid'], t['shape']): i for i, t in enumerate(targets)}
    for d in dispositions:
        if d['action'] == 'new_support_Event_followup':
            d['plan_target_index'] = indexes[key(d['parent_id'], d['shape'])]
    assert len(indexes) == len(targets)
    plan_path = OUT / 'plans/compute_support_event_followups.json'
    plan = {'libraries': {label: lib['path'] for label, lib in libraries.items()},
            'workspace': False, 'targets': targets,
            'support_full_numerical_coverage_complete': support['summary']['full_support_numerical_regression_complete'],
            'selection_is_adoption_gate': False}
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    queue = {'env': winners['env'], 'commands': []}
    for index, target in enumerate(targets):
        kid = target['kid']; m, n, k = target['shape']
        name = f'compute_support_event_{kid}_m{m}_n{n}_k{k}'
        result = OUT / 'results' / (name + '.json')
        queue['commands'].append({'name': name, 'argv': ['/opt/venv/bin/python3', str(OUT / 'experiment_runner.py'),
            '--plan', str(plan_path), '--output', str(result), '--target-index', str(index),
            '--rounds', '1', '--iters', '51', '--repetitions', '8', '--event-confirm'],
            'log': str(result.with_suffix('.log'))})
    queue_path = OUT / 'compute_support_event_followups_queue.json'
    queue_path.write_text(json.dumps(queue, indent=2) + '\n')
    output = {'status': 'partial_support_followups_cpu_prepared' if not plan['support_full_numerical_coverage_complete'] else 'support_followups_cpu_prepared',
              'new_gpu_execution': False, 'production_sources_modified': False,
              'support_analysis': str(support_path.relative_to(ROOT)), 'support_analysis_sha256': sha(support_path),
              'support_summary': support['summary'], 'library_identity': libraries,
              'threshold_candidate_time_increase_percent': args.screen_time_increase_percent,
              'threshold_is_adoption_gate': False, 'screen_signal_count': len(selected),
              'new_followup_count': len(targets), 'dispositions': dispositions,
              'plan': str(plan_path.relative_to(ROOT)), 'plan_sha256': sha(plan_path),
              'queue': str(queue_path.relative_to(ROOT)), 'queue_sha256': sha(queue_path),
              'protocol': '8 numerical repetitions per side, then shared rotation 51-call graphs x 5 AB/BA Event rounds',
              'limits': ['Candidate supported paths are relevant even when a different parent is the historical shape winner.',
                         'Single trace screening cannot justify rejection; existing positive Event can disagree with later trace.',
                         'Current-winner screens are covered by the complete 201-event queue, not dropped from required validation.',
                         'Only clean complete support slices are considered. Partial coverage does not imply the full 735/691 domains passed.',
                         'Regenerate after each root batch completion; do not regenerate a mutable plan during its active execution.']}
    output_path = OUT / 'compute_support_event_followups_manifest.json'
    output_path.write_text(json.dumps(output, indent=2) + '\n')
    assert 'torch' not in sys.modules
    print(json.dumps({'manifest': str(output_path), 'signals': len(selected), 'new_followups': len(targets),
                      'dispositions': [{'shape': d['shape'], 'parent_id': d['parent_id'], 'action': d['action']} for d in dispositions]}))


if __name__ == '__main__':
    main()
