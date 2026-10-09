#!/usr/bin/env python3
"""CPU refresh of9030 final summary and single44-configuration state entry."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inventory = json.loads((HERE.parent / 'inventory.json').read_text())
    entry = next(e for e in inventory['entries'] if e['parent_id'] == 9030)
    source = json.loads((HERE / 'isa_source_review.json').read_text())
    screen = json.loads((HERE / 'screen_independent_review.json').read_text())
    coverage_path = HERE / 'coverage_independent_review.json'
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    final_path = HERE / 'global_decision.json'
    decision = json.loads(final_path.read_text()) if final_path.exists() else None
    status = (decision['decision'] if decision else (coverage['decision_status'] if coverage else 'full10_finite_coverage_prepared_pending_root_GPU'))
    refs = {n: {'path': str(HERE / n), 'sha256': sha(HERE / n)} for n in
            ['source_manifest.json', 'isa_source_review.json', 'independent_synchronization_review.json',
             'cohort_analysis.json', 'bounded_runner_cpu_review.json', 'screen_independent_review.json',
             'coverage_plan.json', 'coverage_queue.json', 'coverage_preparation.json']}
    if coverage:
        refs['coverage_independent_review.json'] = {'path': str(coverage_path), 'sha256': sha(coverage_path)}
    if decision:
        refs['global_decision.json'] = {'path': str(final_path), 'sha256': sha(final_path)}
    controls = [r for run in screen['results'] for r in run['rows'] if 'paired_wins' in r and not r['historical_actual_winner']]
    winner_screen = next(r for run in screen['results'] for r in run['rows'] if r['historical_actual_winner'])
    result = {'status': status, 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'parent_id': 9030, 'actual_configuration_ids': entry['actual_configuration_ids'],
              'actual_configuration_count': 1, 'whole_inventory_configuration_count': inventory['actual_device_config_count'],
              'symbol': entry['symbol'], 'traits': entry['traits'], 'historical_actual_winner_count': 10,
              'historical_actual_winner_K_values': [1536], 'official_module': inventory['official_module'],
              'candidate_library_sha256': source['candidate_library_sha256'],
              'candidate_CO_sha256': source['candidate_code_object_sha256'],
              'mechanism': '9030-only midpoint publication/reader retirement, K+2 issue after midpoint, terminal C-alias barrier retained.',
              'source_and_ISA_review': 'passed_exact_official_baseline_and_single_header_scope',
              'clean_ATT': {'waves': 688, 'publication_release_groups': 86, 'MFMA_per_wave': 288,
                            'interior_cohort_count': 774, 'barrier_arrival_spread_median_shader_clocks': 908,
                            'latest_arrival_to_release_median_shader_clocks': 4,
                            'interpretation': 'Barrier share correlates with wave arrival skew; no22% removable whole-call claim.'},
              'numerical_tail_guards': {'status': 'passed', 'count': 5, 'M': 16448, 'N': 65536,
                                       'K_values': [128, 256, 384, 512, 640], 'both_labels': True},
              'mechanism_screen_actual_winner': winner_screen, 'support_nonwinner_costs_retained': controls,
              'complete_winner_coverage': coverage['aggregate'] if coverage else {'status': 'root_GPU_pending', 'predeclared_shape_count': 10, 'repetition_campaigns': 1},
              'global_decision': decision,
              'production_status': 'root_controls_integration; no9030 production header modified by this agent',
              'new_K_threshold': False, 'posthoc_winner_subset': False, 'GPU_execution_by_this_agent': False,
              'rejected_preparation_revision': 'rejected_prepare_revision_misaligned_M; originalM16385 rejected by baseline API before candidate dispatch.',
              'evidence': refs}
    (HERE / '9030_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    state = {k: result[k] for k in ['parent_id', 'actual_configuration_ids', 'actual_configuration_count',
                                  'whole_inventory_configuration_count', 'symbol', 'traits', 'status',
                                  'historical_actual_winner_count', 'historical_actual_winner_K_values',
                                  'mechanism', 'production_status', 'new_K_threshold', 'posthoc_winner_subset']}
    state.update(summary_path=str(HERE / '9030_summary.json'), summary_sha256=sha(HERE / '9030_summary.json'),
                 evidence_root=str(HERE), complete_winner_coverage=result['complete_winner_coverage'],
                 retained_support_nonwinner_costs=[{k: r[k] for k in ['shape', 'paired_wins', 'paired_median_speedup', 'ratio_of_label_medians']} for r in controls])
    (HERE / '9030_44config_status_entry.json').write_text(json.dumps(state, indent=2) + '\n')
    lines = ['# 9030 optimization record', '', f'Status: {status}. This entry represents one of the current44 actual configurations and all10 historical9030 winners, each K1536.', '',
             'The isolated candidate moves publication/reader retirement after the first8 current MFMAs, then issues K+2. It retains the scale-first prologue, all layouts/guards/helper/loop contracts and terminal barrier before C aliases matrix LDS.', '',
             'CPU source and built ISA reviews passed, including exact official FUNC/full metadata/normalized descriptor baseline alignment. Candidate CO5cee4b4… and library2bf61c8…. VGPR203/LDS143360 unchanged; SGPR50→53 is recorded without a bottleneck inference.', '',
             'Clean ATT recollect contains688 complete waves in86 publication-release cohorts. Interior arrival skew median908 shader clocks while latest arrival-to-release is4. The22% barrier wave share is not removable whole-call wall time.', '',
             'Corrected five terminal guards passed both labels at M16448/N65536 and K128–640. OriginalM16385 preparation failed the baseline API before candidate dispatch and remains preserved.', '',
             f"The mechanism screen actual winner paired gain is {(winner_screen['paired_median_speedup']-1)*100:+.4f}% ({winner_screen['paired_wins']}/5), ratio-of-medians gain {(winner_screen['ratio_of_label_medians']-1)*100:+.4f}%.", '',
             'Preserved legal nonwinner support costs:', '']
    for c in controls:
        lines.append(f"- {'×'.join(map(str,c['shape']))}: paired gain {(c['paired_median_speedup']-1)*100:+.4f}%, ratio-of-medians gain {(c['ratio_of_label_medians']-1)*100:+.4f}%, {c['paired_wins']}/5 paired wins.")
    lines += ['', 'Complete10 winner coverage is predeclared once with the current candidate, bounded8-address shared pool, five AB/BA rounds and51 calls per graph. No new K threshold or posthoc shape subset is introduced.', '']
    if coverage:
        a = coverage['aggregate']
        lines.append(f"Coverage: {a['paired_win_count']}/50 paired wins; {a['paired_median_positive_shape_count']}/10 positive paired medians; {a['positive_in_both_order_shape_count']}/10 positive in both orders. Equal-shape paired geomean gain {(a['equal_shape_geomean_paired_medians']-1)*100:+.4f}%; ratio-of-medians geomean gain {(a['equal_shape_geomean_ratio_of_medians']-1)*100:+.4f}%.")
    else:
        lines.append('Coverage GPU result and final global decision are pending.')
    if decision:
        lines += ['', decision['decision'], '', decision['reason']]
    lines += ['', 'Machine-readable summary: [9030_summary.json](9030_summary.json). Inventory entry: [9030_44config_status_entry.json](9030_44config_status_entry.json).']
    (HERE / '9030_summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': status, 'summary_sha256': sha(HERE / '9030_summary.json'),
                      '44config_entry_sha256': sha(HERE / '9030_44config_status_entry.json')}))


if __name__ == '__main__':
    main()
