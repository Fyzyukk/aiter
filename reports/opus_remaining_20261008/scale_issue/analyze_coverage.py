#!/usr/bin/env python3
"""Independent CPU audit of frozen166 scale winners; no GPU/build/plan edits."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/opus_bound_analysis_20261007'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, text):
    if not ok:
        raise ValueError(text)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def main():
    plan_path = HERE / 'coverage_plan.json'
    result_path = HERE / 'coverage_results.json'
    queue_path = HERE / 'coverage_queue.json'
    claim_path = HERE / 'coverage_claim.jsonl'
    plan = read(plan_path); queue = read(queue_path)
    inventory = read(HERE.parent / 'inventory.json')
    source_review = read(HERE / 'scale_source_review.json')
    require(source_review['status'] == 'ready_for_root_controlled_numerical_guard_and_Event_screen', 'Final scale CPU source review differs')
    require(len(plan['targets']) == 166, 'Not the frozen166-target pool')
    require({t['kid'] for t in plan['targets']} == {9020, 9022}, 'Unexpected parent')
    expected9020 = next(e for e in inventory['entries'] if e['parent_id'] == 9020 and 'Li8ELi384' in e['symbol'])
    expected9022 = next(e for e in inventory['entries'] if e['parent_id'] == 9022)
    planned9020 = [t['shape'] for t in plan['targets'] if t['kid'] == 9020]
    planned9022 = [t['shape'] for t in plan['targets'] if t['kid'] == 9022]
    require(len(planned9020) == 7 and {tuple(s) for s in planned9020} == {tuple(s) for s in expected9020['winner_shapes']}, 'Not complete existing9020 fixed384 winners')
    require(len(planned9022) == 159 and {tuple(s) for s in planned9022} == {tuple(s) for s in expected9022['winner_shapes']}, 'Not complete159 actual9022 winners')
    require(len({(t['kid'], *t['shape']) for t in plan['targets']}) == 166, 'Duplicate target')
    require(sha(plan['libraries']['baseline']) == inventory['official_module']['sha256'], 'Current official baseline changed')
    require(sha(plan['libraries']['candidate']) == source_review['candidate_library_sha256'], 'Candidate library changed from final waitpack review')
    wrapper_review = source_review['baseline_runner_review']
    require(sha(HERE / 'event_runner.py') == wrapper_review['wrapper_sha256'], 'Official baseline wrapper changed')
    require(sha(OLD / 'experiment_runner.py') == wrapper_review['retained_experiment_runner_sha256'], 'Event/check helper changed')
    require(sha(OLD / 'official_smoke.py') == wrapper_review['retained_official_smoke_sha256'], 'Official API wrapper changed')
    sys.path.insert(0, str(ROOT / 'csrc/opus_gemm'))
    import opus_gemm_common as registry
    require(all(registry.a8w8_mxscale_bpreshuffle_supports_shape(registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[t['kid']], *t['shape']) for t in plan['targets']), 'CPU scalar support rejects planned target')
    if not result_path.exists() or not claim_path.exists():
        print(json.dumps({'status': 'CPU_prepared_awaiting166_root_results'})); return
    app = read(result_path)
    claims = [json.loads(l) for l in claim_path.read_text().splitlines()]
    command, = queue['commands']
    completed = [c for c in claims if c.get('event') == 'end' and c.get('name') == command['name']]
    if app.get('status') == 'running' or not completed:
        print(json.dumps({'status': 'awaiting_complete166_root_GPU_claim', 'rows_started': len(app.get('rows', [])),
                          'rows_Event_passed': sum(r.get('event_confirmation', {}).get('status') == 'passed' for r in app.get('rows', []))})); return
    require(app['status'] == 'passed' and app['plan'] == plan and app['plan_sha256'] == sha(plan_path), 'Completed app/plan mismatch')
    require(len(app['rows']) == 166 and app['event_confirmation_requested'] and not app['profiling_only'], 'Incomplete166 event coverage')
    require(app['rounds'] == 5 and app['iters'] == 51 and app['runner_sha256'] == sha(OLD / 'experiment_runner.py'), 'Timing runner contract mismatch')
    require(app['source_head'] == inventory['source_head'], 'Source HEAD changed')
    require(app['libraries']['baseline'] == inventory['official_module'], 'Recorded official baseline identity differs')
    require(app['libraries']['candidate']['sha256'] == source_review['candidate_library_sha256'], 'Executed candidate not final waitpack build')
    require(all(sha(r['path']) == r['sha256'] for r in app['libraries'].values()), 'Actual library mutated')
    # Reuse the strict current ATT owner validator: no additional-owner exception.
    claim_helper = module('scale166_claim_cpu', HERE.parent / 'diagnostics/merged_att_analysis.py')
    claim = claim_helper.clean_claim(command['name'], app, command, queue, claims)
    summary_helper = module('scale166_summary_cpu', HERE.parent / 'narrow_candidate/analyze_retention.py')
    rows = []
    for index, (row, target) in enumerate(zip(app['rows'], plan['targets'])):
        require(row['target_index'] == index and row['kid'] == target['kid'] and row['shape'] == target['shape'] and row['split'] == 1, 'Target identity/split mismatch')
        require(target['signed'] and target['seed'] == 17, 'Input contract changed')
        require(set(row['correctness']) == {'baseline', 'candidate'}, 'Missing correctness label')
        for correct in row['correctness'].values():
            require(correct['repetitions'] == 8 and correct['errRatio'] == 0 and correct['repeatable'] and correct['output_guards'] and correct['workspace_guards'], 'Numerical/guard/repeatability gate failed')
        timings = row['timings']; require(len(timings) == 10, 'Missing regular timing gate')
        for round_index in range(5):
            order = ['baseline', 'candidate'] if round_index % 2 == 0 else ['candidate', 'baseline']
            selected = timings[round_index * 2:round_index * 2 + 2]
            require([t['label'] for t in selected] == order and all(t['round'] == round_index for t in selected), 'Regular timing order incomplete')
            require(all(t['errRatio'] == 0 and t['output_guards'] and t['workspace_guards'] and math.isfinite(t['us']) and t['us'] > 0 for t in selected), 'Regular timing guard failed')
        event = row['event_confirmation']
        require(event['status'] == 'passed' and event['rounds'] == 5 and event['iters_per_graph'] == 51 and event['shared_pool'], 'Event incomplete')
        count = event['rotation']['count']
        require(event['rotation']['mode'] == 'aiter_automatic' and 1 <= count <= 51 and len(event['pool_pointers']) == count, 'Automatic pool contract differs')
        for ptr in event['pool_pointers']:
            require(set(ptr) == {'a', 'b', 'c', 'sfa', 'sfb', 'workspace'} and ptr['workspace'] == 0, 'Pool ABI identity mismatch')
        require(all(len({p[name] for p in event['pool_pointers']}) == count for name in ['a', 'b', 'c', 'sfa', 'sfb']), 'Pool address identity incomplete')
        measurements = event['measurements']; require(len(measurements) == 10, 'Event measurement count differs')
        byround = {}
        for round_index in range(5):
            order = ['baseline', 'candidate'] if round_index % 2 == 0 else ['candidate', 'baseline']
            selected = measurements[round_index * 2:round_index * 2 + 2]
            require([m['label'] for m in selected] == order, 'Event AB/BA order mismatch')
            for m in selected:
                require(m['round'] == round_index and m['order'] == order and m['iters'] == 51 and m['rotation_count'] == count, 'Event round/pool contract mismatch')
                require(math.isfinite(m['us_per_call']) and m['us_per_call'] > 0 and m['us_per_call'] == m['event_total_ms'] * 1000 / 51, 'Event timing arithmetic differs')
                checks = m['all_pool_checks']
                require(len(checks) == count and [c['pool_index'] for c in checks] == list(range(count)), 'Incomplete Event address guards')
                require(all(c['output_repeatable'] and c['errRatio'] == 0 and c['output_guards'] and c['workspace_guards'] and c['workspace_repeatable'] is False for c in checks), 'Event pool numerical/guards failed')
                byround.setdefault(round_index, {})[m['label']] = m['us_per_call']
        medians = {label: statistics.median(byround[i][label] for i in range(5)) for label in ['baseline', 'candidate']}
        speedup = medians['baseline'] / medians['candidate']
        paired = [byround[i]['baseline'] / byround[i]['candidate'] for i in range(5)]
        require(medians == event['median_us'] and speedup == event['median_speedup']['candidate'], 'Saved Event statistics differ')
        m, n, k = row['shape']; bm, bn = (192, 256) if row['kid'] == 9020 else (160, 128)
        rows.append({'target_index': index, 'kid': row['kid'], 'shape': row['shape'], 'purpose': target['purpose'],
                     'actual_winner': True, 'existing_variant': '9020_fixed384' if row['kid'] == 9020 else '9022_runtime',
                     'baseline_us': medians['baseline'], 'candidate_us': medians['candidate'], 'speedup': speedup,
                     'speedup_percent': 100 * (speedup - 1), 'time_change_percent': 100 * (medians['candidate'] / medians['baseline'] - 1),
                     'paired_speedups': paired, 'paired_median_speedup': statistics.median(paired),
                     'faster_rounds': sum(v > 1 for v in paired),
                     'order_paired_geomeans': {order: math.exp(statistics.mean(math.log(paired[i]) for i in range(5) if (i % 2 == 0) == (order == 'AB'))) for order in ['AB', 'BA']},
                     'regular_perf_speedup': row['median_speedup']['candidate'], 'automatic_rotation_count': count,
                     'grid_workgroups': ((m + bm - 1) // bm) * (n // bn), 'M_tile_tail': m % bm != 0})
    group9020 = [r for r in rows if r['kid'] == 9020]
    group9022 = [r for r in rows if r['kid'] == 9022]
    summaries = {'9020_existing_fixed384_all7': summary_helper.summary(group9020),
                 '9022_existing_global_all159': summary_helper.summary(group9022),
                 'all166_affected_winners': summary_helper.summary(rows)}
    for k in sorted({r['shape'][2] for r in group9022}):
        summaries[f'9022_K{k}'] = summary_helper.summary([r for r in group9022 if r['shape'][2] == k])
    for key, selected in [('9022_M_tile_tail', [r for r in group9022 if r['M_tile_tail']]),
                          ('9022_M_tile_aligned', [r for r in group9022 if not r['M_tile_tail']])]:
        if selected:
            summaries[key] = summary_helper.summary(selected)
    cohort_diagnostics = {}
    for k in sorted({r['shape'][2] for r in group9022}):
        selected = [r for r in group9022 if r['shape'][2] == k]
        cohort_diagnostics[f'9022_K{k}'] = {
            'shape_count': len(selected),
            'paired_median_negative_shapes': sum(r['paired_median_speedup'] < 1 for r in selected),
            'ratio_median_negative_shapes': sum(r['speedup'] < 1 for r in selected),
            'five_of_five_slower_shapes': sum(r['faster_rounds'] == 0 for r in selected),
            'paired_wins': sum(r['faster_rounds'] for r in selected),
            'paired_observations': len(selected) * 5,
            'AB_shape_geomean': math.exp(statistics.mean(math.log(r['order_paired_geomeans']['AB']) for r in selected)),
            'BA_shape_geomean': math.exp(statistics.mean(math.log(r['order_paired_geomeans']['BA']) for r in selected)),
            'negative_rows': [r for r in selected if r['speedup'] < 1],
            'five_of_five_slower_rows': [r for r in selected if r['faster_rounds'] == 0]}
    finite_costs = [r for r in rows if r['speedup'] < 1]
    uniform_slow = [r for r in rows if r['faster_rounds'] == 0]
    candidate9022 = summaries['9022_existing_global_all159']
    recommendation9022 = ('existing_global9022_positive_aggregate_requires_explicit_finite_cost_review'
                          if candidate9022['geomean_speedup'] > 1 and candidate9022['all_five_paired_round_shape_geomeans_positive']
                          else 'reject_global9022_no_consistent_fullwinner_aggregate_gain')
    candidate9020 = summaries['9020_existing_fixed384_all7']
    recommendation9020 = ('retain_existing9020_fixed384_variant_pending_root_exact_scope_official_gates'
                          if candidate9020['geomean_speedup'] > 1 and candidate9020['all_five_paired_round_shape_geomeans_positive']
                          else 'reject_existing9020_fixed384_variant_no_consistent_gain')
    result = {'status': 'passed_independent_strict166_coverage_audit', 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'cpu_only': True, 'new_GPU_execution': False, 'new_build_execution': False, 'plan_modified': False, 'library_modified': False,
              'exact_scope': {'9020_existing_fixed384_winners': 7, '9022_existing_global_winners': 159,
                              'no_new_K_threshold': True, 'posthoc_winner_subset': False},
              'executed_libraries': app['libraries'], 'candidate_CO_sha256': source_review['candidate_CO_sha256'],
              'clean_claim': claim, 'rows': rows, 'summaries': summaries, 'finite_costs': finite_costs,
              'K_cohort_diagnostics': cohort_diagnostics,
              'resources_9022': next(v['resource_delta'] for v in source_review['variants'] if v['parent_id'] == 9022),
              'five_of_five_slower_winners': uniform_slow,
              'recommendation': {'9020': recommendation9020, '9022': recommendation9022},
              'validation': {'all166_signed8_reference_guards_repeatability': True, 'all166_shared_pool5ABBA51call_Event': True,
                             'exact166_historical_affected_winner_set': True, 'strict_same_mm_owner_clean_epoch_fingerprint': True,
                             'official_loaded_module_check': 'Successful unchanged OfficialRunner verifies actual loaded path/SHA on first baseline call. Result does not separately serialize actual_module; this review does not invent that missing field.',
                             'official_runner_source_and_helpers_SHA_match_final_review': True},
              'evidence': {n: sha(HERE / n) for n in ['coverage_plan.json', 'coverage_results.json', 'coverage_queue.json', 'coverage_claim.jsonl',
                                                   'scale_source_review.json', 'event_runner.py', 'device_audit.json', 'build_manifest.json']},
              'analysis_script_sha256': sha(__file__),
              'statistics_definition': 'Speedup is median(Eventbaseline)/median(Eventcandidate); per-shape summary geomean weights each existingwinner equally. Five paired-round geomeans are descriptive correlated samples, not confidence intervals.',
              'limits': ['9020 global scope previously rejected by mixed seven-variant screen; this finite coverage can only support the existing fixed384 variant.',
                         'All1599022 winners include every observed K cohort and geometry. Cohort regressions and every5/5 loser remain visible in a global decision.',
                         'Regular run_perftest timings and Event timings are kept distinct; retention uses the shared-pool complete-call Event.',
                         'No candidate ATT gain or new physical-flow claim follows from this Event audit.']}
    require('torch' not in sys.modules, 'CPU analysis imported torch')
    (HERE / 'coverage_independent_review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Independent166 scale-winner coverage', '',
             'Strict current owner epoch and source/command fingerprint, official baseline wrapper identity, finalwaitpack candidate SHA, signed eight-repeat guards and five alternating shared-pool Events passed.', '',
             '| Scope | Shapes | Event geometric-mean gain | Positive medians | Five-of-five faster | Five-of-five slower |',
             '|---|---:|---:|---:|---:|---:|']
    for key, s in summaries.items():
        lines.append(f"| {key} | {s['shape_count']} | {s['geomean_speedup_percent']:+.4f}% | {s['median_faster_shapes']} | {s['five_of_five_faster_shapes']} | {s['five_of_five_slower_shapes']} |")
    lines += ['', f"9020: {recommendation9020}.", f"9022: {recommendation9022}.", '',
              f"Retained negative medians: {len(finite_costs)}; retained five-of-five slower winners: {len(uniform_slow)}. No rows or K groups are removed.", '',
              'The actual official module check executes in the unchanged OfficialRunner on its first successful call. The original result has no separately serialized actual_module field; this limitation is preserved.', '',
              'Complete paired data, per-K summaries and all finite costs: [coverage_independent_review.json](coverage_independent_review.json).']
    (HERE / 'coverage_independent_review.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'recommendation': result['recommendation'],
                      'summaries': {k: v for k, v in summaries.items() if k in ['9020_existing_fixed384_all7', '9022_existing_global_all159']},
                      'finite_cost_count': len(finite_costs), 'all5_slower_count': len(uniform_slow),
                      'review_sha256': sha(HERE / 'coverage_independent_review.json')}))


if __name__ == '__main__':
    main()
