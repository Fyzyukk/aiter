#!/usr/bin/env python3
"""Freeze one full9030 historical-winner coverage; CPU only, root owns GPU."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (HERE / 'coverage_plan.json').exists(), 'Preserve prepared finite coverage'
    inventory = json.loads((HERE.parent / 'inventory.json').read_text())
    entry = next(e for e in inventory['entries'] if e['parent_id'] == 9030)
    isa = json.loads((HERE / 'isa_source_review.json').read_text())
    screen = json.loads((HERE / 'screen_independent_review.json').read_text())
    assert screen['status'] == 'passed_strict_clean_epoch_identity_numerical_guards_and_paired_Event'
    assert sha(HERE / 'candidate/experiments.so') == isa['candidate_library_sha256'] == screen['candidate_library_sha256']
    assert sha(inventory['official_module']['path']) == inventory['official_module']['sha256']
    shapes = entry['winner_shapes']
    assert len(shapes) == len({tuple(s) for s in shapes}) == entry['winner_count'] == 10
    assert {s[2] for s in shapes} == {1536}
    sys.path.insert(0, str(ROOT / 'csrc/opus_gemm'))
    import opus_gemm_common as registry
    instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[9030]
    assert all(registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *s) for s in shapes)
    assert 'torch' not in sys.modules
    target_base = {'kid': 9030, 'seed': 17, 'signed': True, 'private_baselines': {},
                   **{k: entry[k] for k in ['symbol', 'kernel_function', 'traits', 'instruction_sha256', 'actual_configuration_ids']}}
    plan = {
        'name': 'large_midpoint_complete10_historical_winner_coverage_once',
        'baseline': inventory['baseline'], 'official_module': inventory['official_module'],
        'libraries': {'baseline': inventory['official_module']['path'], 'candidate': str(HERE / 'candidate/experiments.so')},
        'targets': [{**target_base, 'shape': shape, 'purpose': 'Exact historical actual9030 winner, complete10 pool frozen before coverage.'} for shape in shapes],
        'predeclared_once': True, 'historical_winner_count': 10, 'all_K_values': [1536],
        'candidate_library_sha256': isa['candidate_library_sha256'],
        'candidate_CO_sha256': isa['candidate_code_object_sha256'],
        'identity_review_sha256': sha(HERE / 'isa_source_review.json'),
        'screen_review_sha256': sha(HERE / 'screen_independent_review.json'),
        'bounded_runner_sha256': sha(HERE / 'bounded_runner.py'),
        'source_inventory_sha256': sha(HERE.parent / 'inventory.json'),
        'timing_contract': {'pool_count_cap': 8, 'shared_physical_pool': True,
                            'iters_per_graph': 51, 'rounds': 5, 'round_order': ['AB', 'BA', 'AB', 'BA', 'AB'],
                            'reference_chunk_rows': 256, 'reference_repetitions_each_label': 2},
        'support_costs_retained_from_screen': [
            {'shape': r['shape'], 'paired_wins': r['paired_wins'], 'paired_median_speedup': r['paired_median_speedup'],
             'ratio_of_label_medians': r['ratio_of_label_medians'], 'order_statistics': r['order_statistics']}
            for run in screen['results'] for r in run['rows'] if 'paired_wins' in r and not r['historical_actual_winner']],
        'decision_contract': [
            'Review strict clean epochs, exact modules/library, signed numerical references, guards and all5 pairs for all10 frozen winners.',
            'Evaluate whole current9030 candidate globally with the existing two legal nonwinner support costs; no new K threshold or post-hoc winner subset.',
            'Do not repeat the3 mechanism screen points or fullcoverage until positive. One complete10 coverage may include the screened winner as part of the fixed pool.',
            'Reject if fullwinner evidence is mixed or too weak to justify the preserved support regressions. Root makes final production integration decision.'
        ]
    }
    (HERE / 'coverage_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    previous_queue = json.loads((HERE / 'screen_queue_corrected.json').read_text())
    queue = {'env': previous_queue['env'], 'commands': [{
                'name': 'large_complete10_winner_coverage_once',
                'argv': ['/opt/venv/bin/python3', str(HERE / 'bounded_runner.py'), '--plan', str(HERE / 'coverage_plan.json'),
                         '--output', str(HERE / 'coverage_results.json'), '--repetitions', '2', '--pool-count', '8', '--iters', '51'],
                'log': str(HERE / 'coverage_runner.log')}],
             'method': 'Current immutable9030 candidate; full10 predeclared pool; bounded8address5ABBA51graphEvent',
             'required_claim_log': str(HERE / 'coverage_claim.jsonl'),
             'root_only_GPU_execution': True}
    (HERE / 'coverage_queue.json').write_text(json.dumps(queue, indent=2) + '\n')
    command = ['/opt/venv/bin/python3', str(ROOT / 'reports/opus_bound_analysis_20261007/run_when_idle.py'),
               '--queue', str(HERE / 'coverage_queue.json'), '--log', str(HERE / 'coverage_claim.jsonl')]
    preparation = {'status': 'CPU_prepared_full10_winner_coverage_root_GPU_pending',
                   'generated_utc': datetime.now(timezone.utc).isoformat(), 'new_GPU_execution': False,
                   'root_command': command, 'plan_sha256': sha(HERE / 'coverage_plan.json'),
                   'queue_sha256': sha(HERE / 'coverage_queue.json'), 'script_sha256': sha(__file__),
                   'historical_exact_winner_count': 10, 'all_shapes_CPU_support': True,
                   'candidate_library_SHA_matches_final_ISA_review': True,
                   'runner_current_SHA_frozen': True, 'official_module_SHA_checked': True}
    (HERE / 'coverage_preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    print(json.dumps(preparation))


if __name__ == '__main__':
    main()
