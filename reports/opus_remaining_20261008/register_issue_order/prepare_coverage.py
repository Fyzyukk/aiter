#!/usr/bin/env python3
"""Freeze full17 runtime9051 winner coverage; CPU preparation, root owns GPU."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def main():
    assert not (HERE / 'coverage_queue.json').exists(), 'Preserve frozen once coverage queue'
    assert not (HERE / 'coverage_results.json').exists(), 'Do not mutate an executed plan'
    plan_path = HERE / 'coverage_event_plan.json'
    plan = json.loads(plan_path.read_text())
    review = json.loads((HERE.parent / 'small_family_review.json').read_text())
    config = next(c for c in review['configurations'] if c['actual_configuration_id'] == 9051)
    shapes = [row['shape'] for row in config['winner_shapes']]
    assert config['winner_count'] == len(shapes) == len({tuple(s) for s in shapes}) == 17
    assert [t['shape'] for t in plan['targets']] == shapes
    assert all(t['kid'] == 9051 and t['symbol'] == config['symbol']
               and t['instruction_sha256'] == config['instruction_sha256']
               and t['signed'] and t['seed'] == 17 for t in plan['targets'])
    assert plan['selected_symbol'] == config['symbol'] and plan['workspace'] is False
    audit = json.loads((HERE / 'device_audit.json').read_text())
    isa = json.loads((HERE / 'isa_review.json').read_text())
    screen = json.loads((HERE / 'results_analysis.json').read_text())
    assert audit['status'] == 'passed_exact_CPU_build_and_scope_audit'
    assert len(audit['checks']) == 40 and audit['unselected_device_entries_unchanged'] == 39
    assert audit['all_baseline_FUNC_fullmetadata_descriptors_exact_official']
    assert isa['status'] == 'passed_exact9051_issue_order_ISA_pending_GPU'
    assert isa['other39deviceentries_exactequal']
    assert screen['status'] == 'positive_two_representative_Event_requires17winner_coverage'
    assert screen['adopted'] is False and len(screen['winner_decisions']) == 2
    for side in ['baseline', 'candidate']:
        assert plan['sealed_libraries'][side] == ref(HERE / side / 'experiments.so')
        assert plan['libraries'][side] == plan['sealed_libraries'][side]['path']
    assert plan['device_audit'] == ref(HERE / 'device_audit.json')
    assert plan['ISA_review'] == ref(HERE / 'isa_review.json')
    assert sha(review['baseline']['official_module']['path']) == review['baseline']['official_module']['sha256']
    sys.path.insert(0, str(ROOT / 'csrc/opus_gemm'))
    import opus_gemm_common as registry
    instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[9051]
    assert all(registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape) for shape in shapes)
    assert 'torch' not in sys.modules
    plan.update(predeclared_once=True, historical_actual_winner_count=17,
        decision_contract='Run the immutable exactruntime9051 candidate once on all17 current actualwinners. '
            'Audit own clean physicalcard claim, exact libraries, unchanged signed8repeat/reference/guards, '
            'and same51address sharedpool5ABBA complete-call Event per winner. '
            'Judge this exacttype uniformly using all17 results; mixed or weak coverage retains currentselected. '
            'No post-hoc shape subset, new threshold, or repeated coverage to seek a positive result. '
            'Root owns GPU execution and any final production integration/formalAPI gate.')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    prior = json.loads((HERE / 'screen_queue.json').read_text())
    queue = {'env': prior['env'], 'commands': [{
        'name': 'register_runtime9051_complete17_winner_coverage_once',
        'argv': ['/opt/venv/bin/python3', str(ROOT / 'reports/opus_bound_analysis_20261007/experiment_runner.py'),
                 '--plan', str(plan_path), '--output', str(HERE / 'coverage_results.json'),
                 '--rounds', '1', '--iters', '51', '--repetitions', '8', '--event-confirm'],
        'log': str(HERE / 'coverage_runner.log')}],
        'method': 'One independent timing round plus fixed5ABBA51sharedpool completeEvent; full17 predeclared pool',
        'required_claim_log': str(HERE / 'coverage_claim.jsonl'), 'root_only_GPU_execution': True}
    queue_path = HERE / 'coverage_queue.json'
    queue_path.write_text(json.dumps(queue, indent=2) + '\n')
    preparation = {'status': 'CPU_prepared_full17_winner_coverage_root_GPU_pending',
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'new_GPU_execution': False,
        'root_command': ['/opt/venv/bin/python3', str(ROOT / 'reports/opus_bound_analysis_20261007/run_when_idle.py'),
                         '--queue', str(queue_path), '--log', str(HERE / 'coverage_claim.jsonl')],
        'plan': ref(plan_path), 'queue': ref(queue_path), 'script': ref(Path(__file__)),
        'runner': ref(ROOT / 'reports/opus_bound_analysis_20261007/experiment_runner.py'),
        'winner_shapes': shapes, 'historical_exact_winner_count': 17,
        'all17_shapes_CPU_support': True, 'all40_baseline_entries_exact_official': True,
        'other39_candidate_entries_equal': True, 'candidate_library_SHA_matches_ISA_review': True,
        'single_independent_round_reason': 'Runner rejects rounds0; completeEvent fixes5pairedrounds independently.',
        'uniform_exacttype_decision_required': True, 'posthoc_shape_scope_forbidden': True}
    (HERE / 'coverage_preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    print(json.dumps(preparation))


if __name__ == '__main__':
    main()
