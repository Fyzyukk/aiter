#!/usr/bin/env python3
"""CPU audit of one complete9030 winner Event coverage; retains support costs."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    preparation = read(HERE / 'coverage_preparation.json')
    plan = read(HERE / 'coverage_plan.json')
    queue = read(HERE / 'coverage_queue.json')
    assert sha(HERE / 'coverage_plan.json') == preparation['plan_sha256']
    assert sha(HERE / 'coverage_queue.json') == preparation['queue_sha256']
    assert sha(HERE / 'bounded_runner.py') == plan['bounded_runner_sha256']
    inventory = read(HERE.parent / 'inventory.json')
    assert sha(HERE.parent / 'inventory.json') == plan['source_inventory_sha256']
    entry = next(e for e in inventory['entries'] if e['parent_id'] == 9030)
    assert [t['shape'] for t in plan['targets']] == entry['winner_shapes'] and len(plan['targets']) == 10
    assert {t['shape'][2] for t in plan['targets']} == {1536}
    review_screen = load_module('large_coverage_screen_helpers', HERE / 'review_screen.py')
    claim_helper = load_module('large_coverage_idle_helpers', OLD / 'run_when_idle.py')
    sys.path.insert(0, str(ROOT / 'csrc/opus_gemm'))
    import opus_gemm_common as registry
    instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[9030]
    assert all(registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *t['shape']) for t in plan['targets'])
    result_path = HERE / 'coverage_results.json'
    claim_path = HERE / 'coverage_claim.jsonl'
    if not result_path.exists() or not claim_path.exists():
        print(json.dumps({'status': 'awaiting_root_GPU_results'})); return
    result = read(result_path)
    claims = [json.loads(l) for l in claim_path.read_text().splitlines()]
    if result.get('status') == 'running' or not any(c.get('event') == 'end' for c in claims):
        print(json.dumps({'status': 'awaiting_complete_root_GPU_claim'})); return
    assert result['status'] == 'passed' and result['plan'] == plan
    assert result['plan_sha256'] == sha(HERE / 'coverage_plan.json')
    assert result['script_sha256'] == plan['bounded_runner_sha256']
    assert result['helper_sha256'] == {n: sha(OLD / n) for n in ['official_smoke.py', 'experiment_runner.py']}
    assert result['libraries']['baseline'] == inventory['official_module']
    assert result['libraries']['candidate']['sha256'] == plan['candidate_library_sha256']
    assert all(sha(i['path']) == i['sha256'] for i in result['libraries'].values())
    assert len(result['rows']) == 10
    epoch = review_screen.claim_review('large_complete10_winner_coverage_once', result, queue, claims, claim_helper)
    rows = []
    for index, (row, target) in enumerate(zip(result['rows'], plan['targets'])):
        assert row['shape'] == target['shape'] and row['kid'] == target['kid'] == 9030 and row['target_index'] == index
        assert row['actual_official_module'] == inventory['official_module']
        assert set(row['correctness']) == {'baseline', 'candidate'}
        for c in row['correctness'].values():
            assert c['repeatable'] and c['repetitions'] == 2 and c['errRatio'] == 0
            assert c['output_guards'] and c['workspace_guards'] and c['reference_chunk_rows'] == 256
        assert row['shared_pool'] and row['pool_count'] == 8 and row['iters'] == 51
        pointers = row['pool_pointers']
        assert len(pointers) == 8 and all(len(p) == 6 for p in pointers)
        assert all(len({p[i] for p in pointers}) == 8 for i in range(5)) and all(p[5] == 0 for p in pointers)
        measurements = row['measurements']; assert len(measurements) == 10
        baseline = {}; candidate = {}
        for r in range(5):
            selected = measurements[r * 2:r * 2 + 2]
            order = ['baseline', 'candidate'] if r % 2 == 0 else ['candidate', 'baseline']
            assert [m['label'] for m in selected] == order
            for m in selected:
                assert m['round'] == r and m['order'] == order and math.isfinite(m['us_per_call']) and m['us_per_call'] > 0
                assert m['checked_addresses'] == 8 and m['signed_reference_verified_expected'] and m['guards_repeatability']
                (baseline if m['label'] == 'baseline' else candidate)[r] = m['us_per_call']
        pairs = [{'round': i, 'order': 'AB' if i % 2 == 0 else 'BA', 'baseline_us': baseline[i], 'candidate_us': candidate[i],
                  'speedup': baseline[i] / candidate[i], 'baseline_minus_candidate_us': baseline[i] - candidate[i]} for i in range(5)]
        medians = {label: statistics.median(d.values()) for label, d in [('baseline', baseline), ('candidate', candidate)]}
        assert medians == row['median_us'] and medians['baseline'] / medians['candidate'] == row['median_speedup']
        order_medians = {order: statistics.median(p['speedup'] for p in pairs if p['order'] == order) for order in ['AB', 'BA']}
        rows.append({'shape': row['shape'], 'historical_actual_winner': True, 'numerical_guards_repeatability': True,
                     'paired_measurements': pairs, 'paired_wins': sum(p['speedup'] > 1 for p in pairs),
                     'paired_median_speedup': statistics.median(p['speedup'] for p in pairs),
                     'ratio_of_label_medians': row['median_speedup'], 'order_median_speedups': order_medians, 'median_us': medians,
                     'shared_pool_count': 8, 'calls_per_graph': 51})
    ratio_geomean = math.exp(statistics.mean(math.log(r['ratio_of_label_medians']) for r in rows))
    pair_geomean = math.exp(statistics.mean(math.log(r['paired_median_speedup']) for r in rows))
    aggregate = {'winner_count': 10, 'numerical_pass_count': 10, 'paired_win_count': sum(r['paired_wins'] for r in rows), 'paired_observation_count': 50,
                 'ratio_median_positive_shape_count': sum(r['ratio_of_label_medians'] > 1 for r in rows),
                 'paired_median_positive_shape_count': sum(r['paired_median_speedup'] > 1 for r in rows),
                 'positive_in_both_order_shape_count': sum(all(v > 1 for v in r['order_median_speedups'].values()) for r in rows),
                 'equal_shape_geomean_ratio_of_medians': ratio_geomean,
                 'equal_shape_geomean_paired_medians': pair_geomean,
                 'median_shape_paired_speedup': statistics.median(r['paired_median_speedup'] for r in rows),
                 'min_shape_paired_speedup': min(r['paired_median_speedup'] for r in rows),
                 'max_shape_paired_speedup': max(r['paired_median_speedup'] for r in rows),
                 'equal_once_shape_total_baseline_median_us': sum(r['median_us']['baseline'] for r in rows),
                 'equal_once_shape_total_candidate_median_us': sum(r['median_us']['candidate'] for r in rows)}
    # A mixed complete pool is a direct rejection signal. Positive pools still
    # require explicit global tradeoff review against preserved support costs.
    stable = (aggregate['paired_median_positive_shape_count'] == 10 and aggregate['positive_in_both_order_shape_count'] == 10
              and ratio_geomean > 1 and pair_geomean > 1)
    result_review = {'status': 'passed_strict_complete10_coverage_audit', 'cpu_only': True, 'new_GPU_execution': False,
                     'official_module': inventory['official_module'], 'candidate_library_sha256': plan['candidate_library_sha256'],
                     'candidate_CO_sha256': plan['candidate_CO_sha256'], 'plan_sha256': sha(HERE / 'coverage_plan.json'),
                     'result_sha256': sha(result_path), 'queue_sha256': sha(HERE / 'coverage_queue.json'), 'claim_sha256': sha(claim_path),
                     'review_script_sha256': sha(__file__), 'strict_clean_epoch': epoch, 'rows': rows, 'aggregate': aggregate,
                     'support_costs_retained': plan['support_costs_retained_from_screen'],
                     'decision_status': 'stable_positive_pool_requires_global_support_cost_judgment' if stable else 'reject_global9030_midpoint_mixed_or_order_sensitive_fullwinner_pool',
                     'scope': 'Whole current9030 body; no newK threshold or posthoc winner subset; exactly one complete10 winner run.',
                     'limits': ['Equal-shape geomeans are stated explicitly and do not invent production workload weights.',
                                'Five paired rounds and one GPU are correlated, no confidence interval is claimed.',
                                'Nonwinner K384 and K16384 support regressions remain part of global decision regardless of winner aggregates.']}
    assert 'torch' not in sys.modules
    (HERE / 'coverage_independent_review.json').write_text(json.dumps(result_review, indent=2) + '\n')
    lines = ['# 9030 complete10 winner coverage review', '', 'Strict clean epoch, immutable plan/library identity, signed references, guards and complete paired Events passed for all10 frozen actual winners.', '',
             '| Shape | Paired wins | Paired median gain | Ratio-of-medians gain | AB gain | BA gain |', '|---|---:|---:|---:|---:|---:|']
    gain = lambda v: f'{(v-1)*100:+.4f}%'
    for r in rows:
        lines.append(f"| {'×'.join(map(str,r['shape']))} | {r['paired_wins']}/5 | {gain(r['paired_median_speedup'])} | {gain(r['ratio_of_label_medians'])} | {gain(r['order_median_speedups']['AB'])} | {gain(r['order_median_speedups']['BA'])} |")
    lines += ['', f"Equal-shape geomean paired gain {gain(pair_geomean)}; ratio-of-medians gain {gain(ratio_geomean)}. {aggregate['paired_win_count']}/50 paired wins; {aggregate['positive_in_both_order_shape_count']}/10 shapes positive in both orders.", '',
              result_review['decision_status'], '', 'Preserved support controls:', '']
    for c in result_review['support_costs_retained']:
        lines.append(f"- {'×'.join(map(str,c['shape']))}: paired gain {gain(c['paired_median_speedup'])}, ratio-of-medians gain {gain(c['ratio_of_label_medians'])}, {c['paired_wins']}/5 paired wins.")
    lines += ['', 'Full rows and audit identity: [coverage_independent_review.json](coverage_independent_review.json).']
    (HERE / 'coverage_independent_review.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result_review['status'], 'aggregate': aggregate, 'decision_status': result_review['decision_status'],
                      'review_sha256': sha(HERE / 'coverage_independent_review.json')}))


if __name__ == '__main__':
    main()
