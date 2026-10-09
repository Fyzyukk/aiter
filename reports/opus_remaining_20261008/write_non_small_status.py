#!/usr/bin/env python3
"""CPU-only eleven actual-config status fragment; root merges family_progress."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inventory = read(HERE / 'inventory.json')
    selected = [e for e in inventory['entries'] if e['parent_id'] in [9020, 9022, 9023, 9024, 9030]]
    assert len(selected) == 11
    scale_screen_plan = read(HERE / 'scale_issue/screen_plan.json')
    scale_screen = read(HERE / 'scale_issue/screen_analysis.json')['rows']
    scale_cov = read(HERE / 'scale_issue/coverage_independent_review.json')
    scale_dec = read(HERE / 'scale_issue/retention_decision.json')
    narrow = read(HERE / 'narrow_candidate/retention_recommendation.json')
    large = read(HERE / 'large_midpoint/9030_summary.json')
    rows = []
    for e in selected:
        parent = e['parent_id']
        row = {'parent_id': parent, 'actual_configuration_ids': e['actual_configuration_ids'],
               'symbol': e['symbol'], 'traits': e['traits'], 'actual_configuration_count': 1,
               'historical_winner_count': e['winner_count'], 'historical_winner_K_values': e['K_values'],
               'baseline_FUNC_sha256': e['instruction_sha256'], 'official_baseline': inventory['official_module'],
               'production_applied_by_this_status_writer': False}
        if parent == 9020:
            matches = [(t, r) for t, r in zip(scale_screen_plan['targets'], scale_screen) if t.get('symbol') == e['symbol']]
            assert len(matches) == 1
            target, measured = matches[0]
            is384 = 'Li8ELi384' in e['symbol']
            row.update(decision='adopt_existing_fixed384_pending_root_official_gates' if is384 else 'keep_current_reject_candidate_outside_existing_fixed384',
                       mechanism='RawSFB requests before original SFA panels; explicit wait+pack in one asm block, matrix schedule unchanged.',
                       screen_measured=[{'shape': measured['shape'], 'Event_speedup': measured['speedup'], 'paired_faster_rounds': measured['faster_rounds']}],
                       full_winner_coverage=[r for r in scale_cov['rows'] if r['kid'] == 9020] if is384 else [],
                       evidence=['scale_issue/scale_source_review.json', 'scale_issue/screen_analysis.json', 'scale_issue/retention_decision.json'],
                       remaining_root_gates=['Integrate exactly tested fixed384-only mechanism with allother9020FUNC identities unchanged.', 'Official linked identity and API/reference/guard gate.'] if is384 else [],
                       limits=['Global9020 candidate rejected: representative fixed3072 regresses2.7005% and0/5 pairs faster; other bodies mixed.',
                               'Screen covers one shape per existingbody, not complete winner coverage for retained-current bodies.'])
            if is384:
                row['coverage_summary'] = scale_dec['9020_summary']
                row['retained_negative_round'] = scale_dec['9020_retained_negative_round']
                row['limits'] = ['All7 historicalfixed384winner medians/pairedmedians positive;34/35 pairs faster; no stablewinner regression.',
                                 '12288round2speedup0.834618 retained: all7pooledroundGM−0.493%, affectedshapeABGM−4.174%; cause unverified and no remeasurement.',
                                 'Existing fixed384 scope only; no newK threshold and no global9020 change.']
        elif parent == 9022:
            cov = [r for r in scale_cov['rows'] if r['kid'] == 9022]
            assert len(cov) == 159
            row.update(decision='keep_current_reject_global_rawscale_candidate',
                       mechanism='Raw SFA passes and SFB requests before dependent publish; same scale-first matrix schedule/refill guards.',
                       measured_scope={'historical_winner_count': 159, 'all_observed_K_values': e['K_values'],
                                       'signed_reference_repetitions': 8, 'paired_Event_rounds': 5, 'calls_per_graph': 51,
                                       'automatic_same_address_pool': True},
                       coverage_summary=scale_dec['9022_summary'],
                       K_cohort_diagnostics=scale_cov['K_cohort_diagnostics'],
                       stable_five_of_five_slower_winners=scale_dec['9022_five_of_five_slower_winners'],
                       evidence=['scale_issue/scale_source_review.json', 'scale_issue/coverage_independent_review.json', 'scale_issue/retention_decision.json'],
                       limits=['Positive all159GM+0.8208% does not justify six stableactualwinner regressions and32negative medians.',
                               'K16384contains12/20negative pairedmedians andthree stablelosers; no newK threshold allowed.',
                               'OfficialRunner validates actualmodule on firstsuccessfulcall, but original result does not separately serialize actualmodule identity.'])
        elif parent in [9023, 9024]:
            subset = [r for r in narrow['rows'] if r['kid'] == parent and r['actual_winner']]
            summary = narrow['summaries']['9023_runtime_winners' if parent == 9023 else '9024_fixed_winners']
            row.update(decision='adopt_existing_runtime_pending_root_official_gates' if parent == 9023 else 'adopt_existing_fixed7168_pending_root_official_gates',
                       mechanism='Rawscale issue-before-publish prologue/refill in existing runtime entry.' if parent == 9023 else 'Rawscale issue-before-publish existingfixed7168 prologue, compile-time fixedloops removes refill.',
                       full_winner_coverage=subset, coverage_summary=summary,
                       support_controls=[r for r in narrow['rows'] if r['kid'] == parent and not r['actual_winner']],
                       evidence=['narrow_candidate/retention_recommendation.json', 'narrow_candidate/formal_merge/manifest.json'],
                       exact_source_patch='narrow_candidate/formal_merge/kid' + str(parent) + '.patch',
                       resources=narrow['resources']['9023_runtime' if parent == 9023 else '9024_fixed'],
                       remaining_root_gates=narrow['remaining_required_checks'],
                       limits=narrow['limits'])
        else:
            assert parent == 9030
            row.update(decision='keep_current_reject_global_midpoint_candidate',
                       mechanism=large['mechanism'], measured_scope={'clean_ATT_waves': 688, 'clean_ATT_WG_release_groups': 86,
                                                                   'terminal_legal_large_output_guards': 5, 'mechanism_screen_targets': 3,
                                                                   'complete_actual_winner_coverage': 10, 'all_actual_winner_K': 1536},
                       coverage_summary=large['complete_winner_coverage'], support_nonwinner_costs=large['support_nonwinner_costs_retained'],
                       evidence=['large_midpoint/coverage_independent_review.json', 'large_midpoint/global_decision.json', 'large_midpoint/9030_summary.json'],
                       limits=['Complete10ratio medians allnegative,9/10pairedmedians negative,4/50pairs faster; screenweakgain not reproduced.',
                               'Bothnonwinnercontrolnegative values retained; no newK gate/posthocshape subset/retest-to-positive.',
                               'ATTbarrier22% is correlated arrival skew, not removable wholecall time.'])
        rows.append(row)
    result = {'status': 'final_non_small11_configuration_fragment_for_root_merge', 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'CPU_only': True, 'new_GPU_execution': False, 'production_modified': False, 'root_family_progress_modified': False,
              'actual_configuration_count': 11, 'whole_inventory_configuration_count': 44,
              'count_explanation': '7×9020 +1×9022 +1×9023runtime +1×9024fixed +1×9030 =11 actualwinner configurations. Legal9023fixed/9024runtime controls are not historicalwinner inventory entries.',
              'rows': rows,
              'support_only_body_statuses': [{'parent_id': 9023, 'body': 'fixed7168', 'decision': 'keep_current', 'counted_in44_actualwinner_inventory': False},
                                           {'parent_id': 9024, 'body': 'runtime', 'decision': 'keep_current', 'counted_in44_actualwinner_inventory': False,
                                            'measured_control': next(r for r in narrow['rows'] if r['kid'] == 9024 and not r['actual_winner'])}],
              'evidence_SHA256': {p: sha(HERE / p) for p in ['inventory.json', 'scale_issue/coverage_independent_review.json',
                                                          'scale_issue/retention_decision.json', 'narrow_candidate/retention_recommendation.json',
                                                          'large_midpoint/9030_summary.json']},
              'script_SHA256': sha(__file__),
              'root_next_step': 'Merge these11 statuses with small-family results, apply retained exactexistingvariants and complete coordinated official identity/API gates.'}
    (HERE / 'non_small_11config_status_fragment.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Non-small eleven configuration status fragment', '', result['count_explanation'], '',
             '| Parent | Existing traits | Historical winners | Decision | Measured scope |', '|---|---|---:|---|---|']
    for r in rows:
        measured = 'all7 winner coverage' if r['parent_id'] == 9020 and 'Li8ELi384' in r['symbol'] else 'one representative screen' if r['parent_id'] == 9020 else 'all159 winners' if r['parent_id'] == 9022 else 'all4 runtime winners+boundary' if r['parent_id'] == 9023 else 'all9 fixedwinners+runtimecontrol' if r['parent_id'] == 9024 else 'all10 winners+5guards+2nonwinnercontrols'
        lines.append(f"| {r['parent_id']} | {r['traits']} | {r['historical_winner_count']} | {r['decision']} | {measured} |")
    lines += ['', 'Retained entries require root official linked identity/API gates. Other9020 bodies,9022 and9030 remain current. No new dispatch threshold is introduced.', '',
              'The9020fixed38412288round2negative Event remains part of every aggregate and final risk.9022sixstablewinnerlosses and9030all10negative ratio medians justify their rejections.', '',
              'Fullper-entry measuredranges, resources, costs and limits: [non_small_11config_status_fragment.json](non_small_11config_status_fragment.json). The rootfamily_progress file is untouched.']
    (HERE / 'non_small_11config_status_fragment.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'entry_count': len(rows), 'fragment_SHA256': sha(HERE / 'non_small_11config_status_fragment.json')}))


if __name__ == '__main__':
    main()
