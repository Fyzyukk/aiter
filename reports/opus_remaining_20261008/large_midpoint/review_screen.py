#!/usr/bin/env python3
"""Independent CPU audit of corrected 9030 numerical gates and paired Events."""
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


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def claim_review(name, result, queue, claims, fingerprint_helper):
    command = next(c for c in queue['commands'] if c['name'] == name)
    starts = [c for c in claims if c.get('event') == 'start' and c['command']['name'] == name
              and c['time'] <= result['started']]
    assert len(starts) == 1
    start = starts[0]
    ends = [c for c in claims if c.get('event') == 'end' and c['name'] == name
            and c['time'] >= result['finished']]
    assert len(ends) == 1
    end = ends[0]
    assert start['command'] == command
    assert end['returncode'] == 0 and end['contamination'] is False
    epoch = [c for c in claims if start['time'] <= c['time'] <= end['time']]
    assert not any(c['event'] in ['claimed', 'external_work_started', 'owner_identity_unresolved'] for c in epoch)
    owner, = [c for c in epoch if c['event'] == 'owner_identity']
    assert owner['new_host_pids'] == [owner['host_pid']] and owner['host_pid'] not in owner['baseline_host_pids']
    assert owner['launcher_sha256'] == sha(OLD / 'owned_python_launch.py')
    monitors = [c for c in epoch if c['event'] == 'monitor']
    assert monitors and all(c['child_pid'] == owner['inner_pid'] for c in monitors)
    assert all(all(p['pid'] == owner['host_pid'] for p in c['processes']) for c in monitors)
    claim = max((c for c in claims if c['event'] == 'claimed' and c['time'] < start['time']), key=lambda c: c['time'])
    gpu = result['gpu']
    assert gpu['pci_bdf'] == gpu['expected_pci_bdf'] == claim['gpu']['bdf']
    assert gpu['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index'])
    expected_fp = fingerprint_helper.command_fingerprint(command, {**queue.get('env', {}), **command.get('env', {})})
    assert expected_fp == start['fingerprint'], 'Current command/file/source fingerprint mismatch'
    return {'status': 'passed_strict_clean_epoch', 'owner_host_pid': owner['host_pid'],
            'owner_inner_pid': owner['inner_pid'], 'physical_PCI_BDF': gpu['pci_bdf'],
            'monitor_count': len(monitors), 'monitors_with_owner': sum(bool(c['processes']) for c in monitors),
            'empty_pre_initialization_monitors': sum(not c['processes'] for c in monitors),
            'start': start['time'], 'end': end['time'], 'result_start': result['started'],
            'result_end': result['finished'], 'command_fingerprint_recomputed_equal': True,
            'command_fingerprint': expected_fp}


def main():
    queue_path = HERE / 'screen_queue_corrected.json'
    claim_path = HERE / 'screen_corrected_claim.jsonl'
    queue = read(queue_path)
    claims = [json.loads(l) for l in claim_path.read_text().splitlines()]
    inventory = read(HERE.parent / 'inventory.json')
    entry = next(e for e in inventory['entries'] if e['parent_id'] == 9030)
    winner_shapes = {tuple(s) for s in entry['winner_shapes']}
    isa = read(HERE / 'isa_source_review.json')
    assert isa['status'] == 'ready_for_root_controlled_numerical_guard_and_Event_screen'
    official = inventory['official_module']
    assert isa['official_module'] == official and sha(official['path']) == official['sha256']
    assert sha(HERE / 'candidate/experiments.so') == isa['candidate_library_sha256']
    cpu = helper('large_screen_idle_cpu', OLD / 'run_when_idle.py')
    sys.path.insert(0, str(ROOT / 'csrc/opus_gemm'))
    import opus_gemm_common as registry
    instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[9030]
    result_records = []
    for result_file, plan_file, command_name, event in [
            ('tail_guard_corrected_results.json', 'tail_guard_plan_corrected.json', 'large_terminal_guards', False),
            ('screen_results.json', 'screen_plan.json', 'large_mechanism_screen', True)]:
        result = read(HERE / result_file)
        plan = read(HERE / plan_file)
        assert result['status'] == 'passed' and result['plan'] == plan
        assert result['plan_sha256'] == sha(HERE / plan_file)
        assert result['script_sha256'] == sha(HERE / 'bounded_runner.py')
        assert result['helper_sha256'] == {n: sha(OLD / n) for n in ['official_smoke.py', 'experiment_runner.py']}
        assert len(result['rows']) == len(plan['targets'])
        libraries = result['libraries']
        assert libraries['baseline'] == official
        assert libraries['candidate']['sha256'] == isa['candidate_library_sha256']
        assert all(sha(i['path']) == i['sha256'] for i in libraries.values())
        identity = claim_review(command_name, result, queue, claims, cpu)
        rows = []
        for index, (row, target) in enumerate(zip(result['rows'], plan['targets'])):
            assert row['target_index'] == index and row['shape'] == target['shape'] and row['kid'] == target['kid'] == 9030
            assert registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *row['shape'])
            assert row['actual_official_module'] == official
            assert set(row['correctness']) == {'baseline', 'candidate'}
            for c in row['correctness'].values():
                assert c['repeatable'] and c['repetitions'] == 2 and c['errRatio'] == 0
                assert c['output_guards'] and c['workspace_guards'] and c['reference_chunk_rows'] == 256
            out = {'shape': row['shape'], 'historical_actual_winner': tuple(row['shape']) in winner_shapes,
                   'actual_module_SHA_verified': True, 'official_CPU_shape_support': True,
                   'both_labels_signed_chunked_reference_guards_repeatability': True}
            measurements = row['measurements']
            if event:
                assert len(measurements) == 10 and row['shared_pool'] and row['iters'] == 51
                assert row['pool_count'] == 8 and len(row['pool_pointers']) == 8
                pointers = row['pool_pointers']
                assert all(len(p) == 6 for p in pointers)
                assert all(len({p[i] for p in pointers}) == 8 for i in range(5))
                assert all(p[5] == 0 for p in pointers)
                baseline = {}; candidate = {}
                for round_index in range(5):
                    selected = measurements[round_index * 2:round_index * 2 + 2]
                    order = ['baseline', 'candidate'] if round_index % 2 == 0 else ['candidate', 'baseline']
                    assert [m['label'] for m in selected] == order
                    for m in selected:
                        assert m['round'] == round_index and m['order'] == order
                        assert math.isfinite(m['us_per_call']) and m['us_per_call'] > 0
                        assert m['checked_addresses'] == 8 and m['signed_reference_verified_expected'] and m['guards_repeatability']
                        (baseline if m['label'] == 'baseline' else candidate)[round_index] = m['us_per_call']
                pairs = [{'round': i, 'order': 'AB' if i % 2 == 0 else 'BA',
                          'baseline_us': baseline[i], 'candidate_us': candidate[i],
                          'speedup': baseline[i] / candidate[i],
                          'baseline_minus_candidate_us': baseline[i] - candidate[i]} for i in range(5)]
                medians = {label: statistics.median(d.values()) for label, d in [('baseline', baseline), ('candidate', candidate)]}
                assert medians == row['median_us'] and medians['baseline'] / medians['candidate'] == row['median_speedup']
                order_stats = {order: {'paired_median_speedup': statistics.median(p['speedup'] for p in pairs if p['order'] == order),
                                      'samples': sum(p['order'] == order for p in pairs)} for order in ['AB', 'BA']}
                out.update(paired_measurements=pairs, paired_wins=sum(p['speedup'] > 1 for p in pairs),
                           paired_median_speedup=statistics.median(p['speedup'] for p in pairs),
                           ratio_of_label_medians=row['median_speedup'], order_statistics=order_stats,
                           pool_count=8, iters_per_graph=51, shared_pool_verified=True,
                           median_us=medians)
            else:
                assert not measurements
            rows.append(out)
        result_records.append({'result': result_file, 'result_sha256': sha(HERE / result_file),
                               'plan': plan_file, 'plan_sha256': sha(HERE / plan_file),
                               'claim_review': identity, 'rows': rows})
    assert 'torch' not in sys.modules
    winner = next(r for rr in result_records for r in rr['rows'] if r['historical_actual_winner'])
    assert winner['paired_wins'] == 5
    result = {
        'status': 'passed_strict_clean_epoch_identity_numerical_guards_and_paired_Event',
        'cpu_only': True, 'new_GPU_execution': False, 'new_build_execution': False,
        'review_script_sha256': sha(__file__), 'queue_sha256': sha(queue_path),
        'claim_sha256': sha(claim_path), 'candidate_library_sha256': isa['candidate_library_sha256'],
        'candidate_CO_sha256': isa['candidate_code_object_sha256'], 'official_module': official,
        'source_ISA_review_sha256': sha(HERE / 'isa_source_review.json'), 'results': result_records,
        'recommendation': 'A single predeclared finite full10 historical-winner coverage is justified by the weak but5/5 paired K1536 signal. Do not repeat the3 points until positive. Do not select globally from this screen.',
        'reasoning': [
            'Only one of10 actual winners was screened. It shows paired median+0.226% with all5 pairs positive in both AB and BA order.',
            'Both support controls are nonwinners; K384 loses5/5 and K16384 is order-sensitive. They prevent a global favorable interpretation.',
            'All10 actual historical winners use K1536, so finite fullwinner coverage directly addresses the existing winner pool without inventing a new runtime threshold.',
            'Any production selection still changes the single9030 body for legal support controls; global regressions must remain visible and cannot be bypassed by post-hoc K gating.',
            'If fullwinner coverage is mixed or too small to justify the verified support regressions, reject this candidate and retain the current official body.'
        ],
        'limits': ['Five correlated replay pairs on one GPU do not establish a statistical confidence interval.',
                   'The bounded8-address pool is explicit and identical for both versions; it is not called automatic rotation.',
                   'Clean ownership does not remove GPU clock/order variability; K16384 shows strong AB/BA separation.',
                   'Preserved original invalid-M preparation failure occurred before candidate dispatch and is not a candidate numerical failure.']
    }
    (HERE / 'screen_independent_review.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Independent 9030 screen review', '',
             'Corrected five terminal guards and three mechanism targets passed strict clean owner epochs, actual official module identity, signed chunked references, output guards and repeatability.', '',
             '| Shape | Role | Paired wins | Paired median gain | Ratio-of-medians gain | AB paired gain | BA paired gain |',
             '|---|---|---:|---:|---:|---:|---:|']
    for rr in result_records:
        for r in rr['rows']:
            if 'paired_wins' in r:
                gain = lambda x: f'{(x - 1) * 100:+.4f}%'
                lines.append(f"| {'×'.join(map(str,r['shape']))} | {'Actual winner' if r['historical_actual_winner'] else 'Legal nonwinner control'} | {r['paired_wins']}/5 | {gain(r['paired_median_speedup'])} | {gain(r['ratio_of_label_medians'])} | {gain(r['order_statistics']['AB']['paired_median_speedup'])} | {gain(r['order_statistics']['BA']['paired_median_speedup'])} |")
    lines += ['', result['recommendation'], '', *result['reasoning'], '',
              'Full identity, epoch, numerical and pair data: [screen_independent_review.json](screen_independent_review.json).']
    (HERE / 'screen_independent_review.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'review_sha256': sha(HERE / 'screen_independent_review.json'),
                      'winner_paired_median': winner['paired_median_speedup'], 'winner_paired_wins': winner['paired_wins']}))


if __name__ == '__main__':
    main()
