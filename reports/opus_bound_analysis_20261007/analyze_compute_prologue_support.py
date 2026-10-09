#!/usr/bin/env python3
"""Audit complete prologue support slices on CPU; partial/polluted runs excluded.

The 1x11 profiler times are screens, never final adoption evidence. This tool
reads the recorded root queue and does not launch HIP, profiling or GPU work.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def log_rows(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            result.append(row)
    return result


def require(condition, description):
    if not condition:
        raise ValueError(description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'results' / 'compute_prologue_support_analysis.json')
    parser.add_argument('--screen-slowdown-percent', type=float, default=5.0,
                        help='descriptive screen list threshold only; not an adoption gate')
    args = parser.parse_args()
    require(args.screen_slowdown_percent >= 0, 'screen slowdown listing threshold must be nonnegative')
    queue_path = OUT / 'compute_prologue_full_support_queue.json'
    manifest_path = OUT / 'compute_prologue_full_scope_manifest.json'
    queue = read(queue_path)
    manifest = read(manifest_path)
    inventory = read(OUT / 'shape_inventory.json')
    claims = log_rows(OUT / 'gpu_claim_log.jsonl')
    plan_records = {str(ROOT / p['plan']): p
                    for parent in manifest['parents'] for p in parent['support_parts']}
    inventory_shapes = {(r['M'], r['N'], r['K']): r for r in inventory['rows']}
    current_libs = {label: {'path': str(OUT / 'compute_prologue' / label / 'experiments.so'),
                           'sha256': sha(OUT / 'compute_prologue' / label / 'experiments.so')}
                    for label in ['baseline', 'candidate']}
    require({k: v['sha256'] for k, v in current_libs.items()} == manifest['library_sha256'],
            'current .so hash changed since scope manifest; do not mix binaries')
    parts = []
    valid_rows = []
    archived = []
    expected_pairs = []
    for command in queue['commands']:
        argv = command['argv']
        plan_path = Path(argv[argv.index('--plan') + 1])
        path = Path(argv[argv.index('--output') + 1])
        plan = read(plan_path)
        plan_sha = sha(plan_path)
        require(plan_sha == plan_records[str(plan_path)]['sha256'], 'plan changed since scope manifest')
        require(plan['workspace'] is False and len(plan['targets']) == plan_records[str(plan_path)]['targets'],
                'plan workspace or target count changed')
        require(1 <= len(plan['targets']) <= 16, 'unexpected slice size')
        kid = plan['targets'][0]['kid']
        expected_pairs.extend((t['kid'], *t['shape']) for t in plan['targets'])
        part = {'name': command['name'], 'parent_id': kid, 'expected_rows': len(plan['targets']),
                'plan': str(plan_path.relative_to(ROOT)), 'plan_sha256': plan_sha,
                'result': str(path.relative_to(ROOT)), 'status': 'not_run'}
        parts.append(part)
        for attempt in sorted(path.parent.glob(path.stem + '.attempt_*.json')):
            try:
                old = read(attempt)
                archived.append({'result': str(attempt.relative_to(ROOT)), 'sha256': sha(attempt),
                                 'status': old.get('status'), 'contamination': old.get('contamination'),
                                 'included_in_support_coverage': False,
                                 'reason': 'archived attempt; current primary result determines coverage'})
            except (OSError, ValueError) as exc:
                archived.append({'result': str(attempt.relative_to(ROOT)), 'included_in_support_coverage': False,
                                 'reason': 'unreadable archived attempt: ' + str(exc)})
        if not path.is_file():
            continue
        try:
            result = read(path)
        except (OSError, ValueError) as exc:
            part.update(status='incomplete_or_unreadable', reason=str(exc))
            continue
        part.update(result_sha256=sha(path), application_status=result.get('status'),
                    observed_rows=len(result.get('rows', [])), started=result.get('started'),
                    finished=result.get('finished'))
        if result.get('contamination') or result.get('status') == 'interrupted_external_gpu_work':
            part.update(status='contaminated_excluded', contamination=result.get('contamination'),
                        included_in_support_coverage=False, timings_usable_for_final_decision=False)
            continue
        if result.get('status') != 'passed' or not result.get('finished'):
            part.update(status='incomplete_or_failed', included_in_support_coverage=False)
            continue
        starts = [c for c in claims if c.get('event') == 'start'
                  and c.get('command', {}).get('name') == command['name']
                  and c['time'] <= result['started']]
        ends = [c for c in claims if c.get('event') == 'end'
                and c.get('name') == command['name'] and c['time'] >= result['finished']]
        if not starts or not ends:
            part.update(status='awaiting_claim_completion', included_in_support_coverage=False)
            continue
        start = max(starts, key=lambda c: c['time'])
        end = min(ends, key=lambda c: c['time'])
        if end.get('contamination'):
            part.update(status='contaminated_excluded', claim_end=end, included_in_support_coverage=False)
            continue
        try:
            require(end['returncode'] == 0 and end['contamination'] is False,
                    'root claim did not finish cleanly')
            require(start['command']['argv'] == argv, 'executed command differs from recorded full support command')
            claim = max((c for c in claims if c.get('event') == 'claimed' and c['time'] <= start['time']),
                        key=lambda c: c['time'])
            require(result['gpu']['pci_bdf'] == result['gpu']['expected_pci_bdf'] == claim['gpu']['bdf'],
                    'GPU physical identity differs from claim')
            require(result['gpu']['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index']),
                    'visible GPU identity differs from claim')
            require(result['source_head'] == manifest['source_head'] == inventory['source_head'], 'source HEAD mismatch')
            require(result['plan_sha256'] == plan_sha and result['plan'] == plan, 'recorded plan mismatch')
            require(result['libraries'] == current_libs, 'result library path/SHA differs from current binaries')
            require(result['rounds'] == 1 and result['iters'] == 11 and result['warmup'] == 5,
                    'expected 1-round/11-iteration screen protocol')
            require(not result['event_confirmation_requested'] and not result['profiling_only']
                    and result['target_index'] is None and result['label_filter'] is None,
                    'support command is not a full numerical+trace slice')
            require(len(result['rows']) == len(plan['targets']), 'slice does not contain all expected rows')
            require([r['target_index'] for r in result['rows']] == list(range(len(plan['targets']))),
                    'slice indexes are missing, duplicated or reordered')
            expected_trace_logs = []
            local_rows = []
            for index, (row, target) in enumerate(zip(result['rows'], plan['targets'])):
                require(row['kid'] == target['kid'] == kid and row['shape'] == target['shape'], 'row shape/kid mismatch')
                require(target['signed'] is True and target['seed'] == 17, 'screen input contract mismatch')
                require(row['split'] == 1 and 'event_confirmation' not in row, 'unexpected split/Event support row')
                for label in ['baseline', 'candidate']:
                    check = row['correctness'][label]
                    require(check['repetitions'] == 8 and check['errRatio'] == 0 and check['repeatable']
                            and check['output_guards'] and check['workspace_guards'], 'original numerical/guard/repeatability failure')
                require(len(row['timings']) == 2 and [t['label'] for t in row['timings']] == ['baseline', 'candidate'],
                        'expected exactly one AB trace pair')
                for t in row['timings']:
                    require(t['round'] == 0 and t['errRatio'] == 0 and t['output_guards'] and t['workspace_guards']
                            and math.isfinite(t['us']) and t['us'] > 0, 'trace row invalid')
                    require(row['median_us'][t['label']] == t['us'], 'single-round median mismatch')
                    expected_trace_logs.append({'kid': kid, 'shape': row['shape'],
                                                **{key: t[key] for key in ['round', 'label', 'us']}})
                baseline, candidate = row['median_us']['baseline'], row['median_us']['candidate']
                require(row['median_speedup']['candidate'] == baseline / candidate, 'trace ratio mismatch')
                inv = inventory_shapes[tuple(row['shape'])]
                require(target['current745_winner_parent'] == inv['parent_id'], 'historical winner identity mismatch')
                local_rows.append({'parent_id': kid, 'shape': row['shape'], 'part': command['name'],
                                   'target_index': index, 'current745_winner_parent': inv['parent_id'],
                                   'same_parent_current_winner': kid == inv['parent_id'],
                                   'trace_baseline_us': baseline, 'trace_candidate_us': candidate,
                                   'trace_throughput_change_percent': 100 * (baseline / candidate - 1),
                                   'trace_candidate_time_increase_percent': 100 * (candidate / baseline - 1),
                                   'timing_interpretation': 'single profiler screen only; not final Event evidence'})
            log_path = Path(command['log'])
            actual_trace_logs = [r for r in log_rows(log_path) if 'us' in r and r.get('method') is None]
            require(actual_trace_logs == expected_trace_logs, 'complete trace logs do not match result rows')
            part.update(status='clean_complete_slice', included_in_support_coverage=True,
                        claim_epoch={'claim': claim, 'start': start, 'end': end},
                        original_numerical_checks=len(plan['targets']) * 2 * 8,
                        rows_numerical_repeatability_guards_passed=True,
                        complete_trace_log_matches=True, log_sha256=sha(log_path),
                        gpu=result['gpu'], runner_sha256=result['runner_sha256'])
            valid_rows.extend(local_rows)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            part.update(status='audit_failed_excluded', included_in_support_coverage=False, reason=str(exc))
    require(len(expected_pairs) == 1426 and len(set(expected_pairs)) == 1426, 'expected full support coverage mismatch')
    observed_pairs = {(r['parent_id'], *r['shape']) for r in valid_rows}
    require(len(observed_pairs) == len(valid_rows), 'clean primary support shapes overlap')
    require(observed_pairs <= set(expected_pairs), 'clean result outside planned support domain')
    clean_parts = [p for p in parts if p['status'] == 'clean_complete_slice']
    parent_summaries = []
    for kid, expected in [(9021, 735), (9022, 691)]:
        parent_rows = [r for r in valid_rows if r['parent_id'] == kid]
        parent_parts = [p for p in parts if p['parent_id'] == kid]
        parent_summaries.append({'parent_id': kid, 'expected_supported_shapes': expected,
                                 'clean_complete_shapes': len(parent_rows),
                                 'clean_complete_slices': sum(p['status'] == 'clean_complete_slice' for p in parent_parts),
                                 'expected_slices': len(parent_parts),
                                 'full_support_numerical_regression_complete': len(parent_rows) == expected,
                                 'current_winner_shapes_in_clean_support_screens': sum(r['same_parent_current_winner'] for r in parent_rows)})
    screens = sorted((r for r in valid_rows if r['trace_candidate_time_increase_percent'] >= args.screen_slowdown_percent),
                     key=lambda r: (-r['trace_candidate_time_increase_percent'], r['parent_id'], r['shape']))
    all_complete = len(valid_rows) == 1426 and len(clean_parts) == 90
    report = {'status': 'complete_clean_support_numerical_audit' if all_complete else 'partial_support_coverage_no_full_pass_claim',
              'new_gpu_execution': False, 'gpu_imported_or_initialized': False, 'production_sources_modified': False,
              'source_head': manifest['source_head'], 'library_identity': current_libs,
              'queue_sha256': sha(queue_path), 'scope_manifest_sha256': sha(manifest_path),
              'summary': {'expected_slices': 90, 'expected_parent_shape_pairs': 1426,
                          'clean_complete_slices': len(clean_parts), 'clean_complete_parent_shape_pairs': len(valid_rows),
                          'full_support_numerical_regression_complete': all_complete,
                          'original_numerical_output_checks': len(valid_rows) * 2 * 8,
                          'slice_status_counts': dict(Counter(p['status'] for p in parts)),
                          'global_adoption': False, 'full_support_Event_confirmation': False},
              'parents': parent_summaries, 'parts': parts, 'clean_screen_rows': valid_rows,
              'archived_attempts_excluded': archived,
              'profiler_screen_followups': {'descriptive_candidate_time_increase_threshold_percent': args.screen_slowdown_percent,
                  'threshold_is_adoption_gate': False, 'count': len(screens), 'rows': screens,
                  'action': 'confirm these screens with shared-pool AB/BA Event; single trace differences cannot prove regression'},
              'limits': ['Only complete primary slices with matching plans, binaries, logs, 8rep numerical checks and clean root claim epochs count.',
                         'Incomplete and contaminated slices provide no support-pass coverage, even if some rows already exist.',
                         'Numerical screen coverage is separate from all 201 current winners requiring controlled Event.',
                         'One AB profiler round/11 calls has address/profiling variability; no stable speedup or rejection claim.',
                         'Root may still be writing results or claim ends; rerun the CPU audit after the batch finishes.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'status': report['status'], 'summary': report['summary'],
                      'parents': parent_summaries, 'screen_followups': len(screens)}))
    require('torch' not in sys.modules, 'CPU support audit must not import torch')


if __name__ == '__main__':
    main()
