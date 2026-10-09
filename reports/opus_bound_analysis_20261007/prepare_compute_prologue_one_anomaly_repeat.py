#!/usr/bin/env python3
"""CPU-create one fixed repeat for the observed large mixed-direction signal.

Reuses the exact original plan and target index with a new output/command
name. The repeat is finite; does not expand other near-zero mixed signals.
"""
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


def main():
    coverage_path = OUT / 'results/compute_prologue_winner_event_coverage.json'
    coverage = json.loads(coverage_path.read_text())
    winner = next(r for r in coverage['covered_winners'] if r['parent_id'] == 9022 and r['shape'] == [1216, 7168, 384])
    original = winner['summary_record']
    assert original['event_ratio_of_medians_change_percent'] <= -5
    assert 0 < original['paired_positive_rounds'] < 5
    command = json.loads(json.dumps(original['claim_epoch']['start']['command']))
    argv = command['argv']
    plan_path = Path(argv[argv.index('--plan') + 1])
    plan = json.loads(plan_path.read_text())
    index = int(argv[argv.index('--target-index') + 1])
    target = plan['targets'][index]
    assert target['kid'] == 9022 and target['shape'] == [1216, 7168, 384] and target['signed'] and target['seed'] == 17
    assert sha(plan_path) == original['plan_sha256'] and not plan['workspace']
    assert '--event-confirm' in argv and argv[argv.index('--repetitions') + 1] == '8'
    assert argv[argv.index('--iters') + 1] == '51' and argv[argv.index('--rounds') + 1] == '1'
    registry, _ = load_cpu_registry()
    assert registry.a8w8_mxscale_bpreshuffle_supports_shape(
        registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[9022], *target['shape'])
    inventory = json.loads((OUT / 'shape_inventory.json').read_text())
    branches = [b for b in inventory['parents']['9022']['branches']
                if all(cpp_eval(c, *target['shape']) for c in b['conditions'])]
    assert len(branches) == 1
    libraries = {label: {'path': path, 'sha256': sha(Path(path))} for label, path in plan['libraries'].items()}
    assert libraries == coverage['libraries']
    assert original['runner_sha256'] == sha(OUT / 'experiment_runner.py')
    name = 'compute_prologue_anomaly_9022_m1216_n7168_k384_fixed_repeat1'
    output = OUT / 'results' / (name + '.json')
    command['name'] = name
    command['log'] = str(output.with_suffix('.log'))
    argv[argv.index('--output') + 1] = str(output)
    source_queue = json.loads((OUT / 'compute_prologue_all_winners_event_short_k_queue.json').read_text())
    queue = {'env': source_queue['env'], 'commands': [command]}
    queue_path = OUT / 'compute_prologue_one_anomaly_repeat_queue.json'
    audit_path = OUT / 'compute_prologue_one_anomaly_repeat_cpu_audit.json'
    assert not queue_path.exists() and not audit_path.exists() and not output.exists()
    queue_path.write_text(json.dumps(queue, indent=2) + '\n')
    audit = {'status': 'passed_cpu_one_fixed_anomaly_repeat_plan_identity_no_gpu_execution',
             'new_gpu_execution': False, 'original_plans_queues_modified': False,
             'source_head': coverage['source_head'], 'coverage_sha256': sha(coverage_path),
             'queue': str(queue_path), 'queue_sha256': sha(queue_path),
             'reused_exact_original_plan': str(plan_path), 'plan_sha256': sha(plan_path), 'target_index': index,
             'target': target, 'libraries': libraries, 'runner_sha256': original['runner_sha256'],
             'actual_branch_index': branches[0]['index'], 'actual_traits': branches[0]['traits_cpp'],
             'prior_result': original['result'], 'prior_result_sha256': original['result_sha256'],
             'prior_ratio_of_medians_speedup_percent': original['event_ratio_of_medians_change_percent'],
             'prior_paired_rounds': original['paired_rounds'],
             'prior_baseline_range_over_median_percent': original['baseline_range_over_median_percent'],
             'prior_candidate_range_over_median_percent': original['candidate_range_over_median_percent'],
             'protocol': 'one predeclared independent repeat only; signed8 per side, shared rotation pool,51calls ×5AB/BA Event; original runner/plan/target/libraries unchanged',
             'decision_rule': 'Retain both windows and decide after this one repeat; no rerun-until-positive. Other mixed near-zero signals are not added.',
             'limits': ['CPU readiness only; no GPU execution or performance outcome.',
                        'This repeats one already-covered current winner, adding a window rather than a unique-shape count.']}
    assert 'torch' not in sys.modules
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'queue': str(queue_path), 'cpu_audit': str(audit_path), 'target': target['shape'],
                      'parent_id': 9022, 'original_plan_index': index, 'extra_targets': 1}))


if __name__ == '__main__':
    main()
