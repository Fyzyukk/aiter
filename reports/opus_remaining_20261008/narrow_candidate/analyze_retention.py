#!/usr/bin/env python3
"""Independent CPU Event/claim audit and reviewable exact merge preparation."""
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FILES = {9023:'opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh',
         9024:'opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def summary(rows):
    value = gm([r['speedup'] for r in rows])
    a = sum(r['baseline_us'] for r in rows)
    b = sum(r['candidate_us'] for r in rows)
    paired = [gm([r['paired_speedups'][i] for r in rows]) for i in range(5)]
    return {'shape_count':len(rows), 'geomean_speedup':value, 'geomean_speedup_percent':100*(value-1),
            'sum_of_shape_medians_us':{'baseline':a,'candidate':b},
            'sum_time_speedup_percent':100*(a/b-1),
            'median_faster_shapes':sum(r['speedup']>1 for r in rows),
            'median_slower_shapes':sum(r['speedup']<1 for r in rows),
            'five_of_five_faster_shapes':sum(r['faster_rounds']==5 for r in rows),
            'five_of_five_slower_shapes':sum(r['faster_rounds']==0 for r in rows),
            'paired_round_shape_geomeans':paired,
            'all_five_paired_round_shape_geomeans_positive':all(v>1 for v in paired)}


def main():
    app = read(HERE/'screen_results.json')
    plan = read(HERE/'screen_plan.json')
    queue = read(HERE/'screen_queue.json')
    claim_rows = [json.loads(x) for x in (HERE/'screen_claim.jsonl').read_text().splitlines()]
    spec = importlib.util.spec_from_file_location('narrow_retention_claim', HERE.parent/'diagnostics/merged_att_analysis.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    command = queue['commands'][0]
    claim = helper.clean_claim(command['name'],app,command,queue,claim_rows)
    require(app['status']=='passed' and app['plan']==plan and app['plan_sha256']==sha(HERE/'screen_plan.json'),
            'Screen result/plan mismatch')
    require(len(app['rows'])==len(plan['targets'])==15, 'Not the complete finite15-target screen')
    build = read(HERE/'build_manifest.json')
    for b in build['builds']:
        require(app['libraries'][b['side']]['sha256']==b['library_sha256']==sha(app['libraries'][b['side']]['path']),
                'Actual private screen library differs from audited build')
    expected_analysis = read(HERE/'screen_analysis.json')['rows']
    rows = []
    for index,(row,target) in enumerate(zip(app['rows'],plan['targets'])):
        require(row['target_index']==index and row['kid']==target['kid'] and row['shape']==target['shape'],
                'Screen target mismatch')
        require(target['signed'] and target['seed']==17, 'Signed input contract differs')
        for correct in row['correctness'].values():
            require(correct['repetitions']==8 and correct['errRatio']==0 and correct['repeatable']
                    and correct['output_guards'] and correct['workspace_guards'], 'Numerical/guard repeat gate failed')
        event = row['event_confirmation']
        require(event['status']=='passed' and event['rounds']==5 and event['iters_per_graph']==51
                and event['shared_pool'] and event['rotation']['count']==51, 'Event/shared51-pool contract differs')
        measurements = event['measurements']
        require(len(measurements)==10, 'Event measurement count differs')
        byround = {}
        for m in measurements:
            require(m['iters']==m['rotation_count']==51 and len(m['all_pool_checks'])==51, 'Event pool incomplete')
            for check in m['all_pool_checks']:
                require(check['output_repeatable'] and check['errRatio']==0 and check['output_guards']
                        and check['workspace_guards'], 'Event pool guard/numerical failed')
            byround.setdefault(m['round'],{})[m['label']] = m['us_per_call']
        require(set(byround)==set(range(5)) and all(set(x)=={'baseline','candidate'} for x in byround.values()),
                'Incomplete paired Event rounds')
        for i in range(5):
            order = ['baseline','candidate'] if i%2==0 else ['candidate','baseline']
            require([m['label'] for m in measurements if m['round']==i]==order, 'Event AB/BA order differs')
        med = {label:statistics.median(x[label] for x in byround.values()) for label in ('baseline','candidate')}
        paired = [byround[i]['baseline']/byround[i]['candidate'] for i in range(5)]
        speedup = med['baseline']/med['candidate']
        e = expected_analysis[index]
        require(e['speedup']==speedup and e['paired']==paired and e['faster_rounds']==sum(v>1 for v in paired),
                'Independent Event statistics differ from root screen analysis')
        rows.append({'target_index':index,'kid':row['kid'],'shape':row['shape'],
                     'actual_winner':index<13,'purpose':target['purpose'],
                     'baseline_us':med['baseline'],'candidate_us':med['candidate'],
                     'speedup':speedup,'speedup_percent':100*(speedup-1),
                     'time_change_percent':100*(med['candidate']/med['baseline']-1),
                     'paired_speedups':paired,'faster_rounds':sum(v>1 for v in paired)})
    actual_9023 = rows[:4]
    actual_9024 = rows[4:13]
    summaries = {'9023_runtime_winners':summary(actual_9023),
                 '9023_runtime_winners_plus_M16_boundary':summary(actual_9023+[rows[13]]),
                 '9024_fixed_winners':summary(actual_9024), 'all13_narrow_winners':summary(rows[:13]),
                 '9023_winner_K7168':summary([r for r in actual_9023 if r['shape'][2]==7168]),
                 '9023_winner_K16384':summary([r for r in actual_9023 if r['shape'][2]==16384])}
    require(all(s['geomean_speedup']>1 and s['all_five_paired_round_shape_geomeans_positive']
                for k,s in summaries.items() if not k.startswith('9023_winner_K')), 'Aggregate retention evidence failed')
    prior = ROOT/'reports/opus_resume_20261008/scale_issue_publish/retention_decision.json'
    require(prior.exists(), 'Prior finite-boundary retention reference missing')
    device = read(HERE/'device_audit.json')
    check_byvariant = {c['variant']:c for c in device['checks']}
    resources = {}
    for variant in ('9023_runtime','9024_fixed'):
        c = check_byvariant[variant]
        keys = ('.vgpr_count','.sgpr_count','.sgpr_spill_count','.vgpr_spill_count','.agpr_count',
                '.group_segment_fixed_size','.private_segment_fixed_size')
        resources[variant] = {side:{**{k:c[side]['metadata'][k] for k in keys},
                                    'instruction_bytes':c[side]['instruction_bytes']} for side in ('baseline','candidate')}
    prepare = HERE/'formal_merge'
    prepare.mkdir(exist_ok=True)
    merge_files = []
    patches = []
    for kid,name in FILES.items():
        original = HERE/'baseline/include/gfx950'/name
        tested = HERE/'candidate/include/gfx950'/name
        production = ROOT/'csrc/opus_gemm/include/gfx950'/name
        require(original.read_bytes()==production.read_bytes(), 'Production narrow source changed before merge preparation')
        rel = str(production.relative_to(ROOT))
        patch = ''.join(difflib.unified_diff(original.read_text().splitlines(True),tested.read_text().splitlines(True),
                                            fromfile='a/'+rel,tofile='b/'+rel))
        require(patch, 'No tested source patch')
        ppath = prepare/f'kid{kid}.patch'
        ppath.write_text(patch)
        snapshot = prepare/name
        snapshot.write_bytes(tested.read_bytes())
        patches.append(patch)
        merge_files.append({'kid':kid,'production_path':str(production),
                            'production_expected_before_sha256':sha(original),
                            'tested_candidate_sha256':sha(tested), 'reviewable_snapshot':str(snapshot),
                            'patch_path':str(ppath),'patch_sha256':sha(ppath),
                            'scope':'T::FIXED_K == 0 prologue/refill' if kid==9023 else 'T::FIXED_K ==7168 prologue; fixed loops56 removes refills'})
    (prepare/'narrow.patch').write_text(''.join(patches))
    manifest = {'status':'exact_tested_source_merge_prepared','production_applied':False,
                'no_new_K_threshold':True,'files':merge_files,
                'combined_patch_path':str(prepare/'narrow.patch'),'combined_patch_sha256':sha(prepare/'narrow.patch'),
                'shared_helper_modified':False,'kernel9000_modified':False,'kernel9021_modified':False,
                'remaining':'Root applies exact patches after coordinating other candidates and runs official identity/API gates.'}
    (prepare/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    evidence_paths = ['screen_plan.json','screen_results.json','screen_analysis.json','screen_claim.jsonl','screen_queue.json',
                      'device_audit.json','isa_order_audit.json','source_review.json','build_manifest.json',
                      'formal_merge/manifest.json','formal_merge/narrow.patch']
    decision = {'status':'recommend_full_variant_retention_pending_root_official_gates',
                'generated_utc':datetime.now(timezone.utc).isoformat(),'cpu_only_analysis':True,
                'new_GPU_execution':False,'new_build_execution':False,'production_applied':False,
                'baseline':read(HERE.parent/'narrow_review.json')['baseline'],
                'recommendation':{'9023_runtime':'retain throughout existing runtime entry; no new K/shape threshold',
                                  '9024_fixed':'retain existing fixed7168 entry',
                                  '9023_fixed':'unchanged','9024_runtime':'unchanged'},
                'reason':'9023 runtime actual-winner aggregate is positive in both measured K groups and all five pooled '
                    'paired rounds, with no winner5/5 slower. Accept measured544-longK winner andM16 boundary costs '
                    'under the previous finite-cost retention standard.9024 fixed has nine positive medians, all5/5 faster, '
                    'and unchanged VGPR/LDS/spill (SGPR58→60). No repeated test-to-positive or invented K threshold is used.',
                'prior_retention_reference':{'path':str(prior),'sha256':sha(prior)},
                'statistics_definition':'Speedup=median(Event baseline us)/median(Event candidate us); '
                    'time_change=candidate/baseline-1. Shape geomean weights each target equally. '
                    'Paired-round geomeans are descriptive finite evidence, not independent cohorts or a significance test.',
                'summaries':summaries,'rows':rows,'resources':resources,
                'finite_costs':[r for r in rows[:14] if r['speedup']<1],
                'clean_claim':claim,'validation':{'all15_signed8_repetitions_passed':True,
                    'all15_reference_errRatio_zero':True,'all15_output_and_workspace_guards_passed':True,
                    'five_ABBA_Event_rounds_shared51_address_pool':True,
                    'same_private_libraries_as_CPU_audit':True,
                    'unchanged9024runtime_control_median_speedup':rows[14]['speedup'],
                    'unchanged_control_interpretation':'The0.49% control speedup is noise/process evidence and not attributed to source.'},
                'evidence':{p:sha(HERE/p) for p in evidence_paths},
                'analysis_script_sha256':sha(__file__),
                'remaining_required_checks':['Root coordinated apply of exact tested narrow sources.',
                    'Official generated TU/host launch and complete linked device identity against current Oct8 baseline.',
                    'Official API signed8/reference/repeatability/guards and loaded module identity after merge.'],
                'limits':['9023 runtime VGPR232→251(+8.19%), SGPR lane spill46→48, ISA25976→26328(+1.36%); '
                    'no scratch/VGPR spill/AGPR/LDS increase. The extra live registers reduce future compiler headroom; '
                    'this experiment does not establish occupancy or full support-domain performance.',
                    'The544 longK winner regresses0.542% in median call time,2/5 paired rounds faster; '
                    'theM16 boundary regresses0.417%,1/5 faster. These costs are accepted explicitly, not dismissed as zero.',
                    '9023 longK remains mixed per shape; aggregate positivity is not a guarantee for every shape.',
                    'Only one clean finite process window with five paired rounds is available; no confirmation-to-positive is requested.',
                    '9024 fixed SGPR58→60 and ISA7276→7408(+1.81%); VGPR110/LDS/spill remain unchanged.',
                    'The unchanged9024runtime control varies by0.49%; sub-percent per-shape results require restraint.',
                    'Baseline ATT identifies serialization; candidate ISA confirms order. No candidate ATT execution/latency claim.',
                    'ATT target12 is excluded for unowned GPU process; no clean252/264WG paired timing claim.']}
    (HERE/'retention_recommendation.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
    lines = ['# Narrow candidate retention recommendation', '',
             'Recommend keeping the tested issue-before-publish change for the existing9023 runtime and9024 fixed7168 '
             'entries. The merge patch is prepared and production remains untouched by this script.', '',
             '| Cohort | Shapes | Event geometric-mean speedup | Positive medians | Five-of-five faster |',
             '| --- | --- | --- | --- | --- |']
    for label in ('9023_runtime_winners','9024_fixed_winners','all13_narrow_winners'):
        s = summaries[label]
        lines.append(f"| {label} | {s['shape_count']} | +{s['geomean_speedup_percent']:.4f}% | {s['median_faster_shapes']} | {s['five_of_five_faster_shapes']} |")
    lines += ['', f"9023 measured K7168/K16384 cohort speedups are +{summaries['9023_winner_K7168']['geomean_speedup_percent']:.4f}% "
                  f"/+{summaries['9023_winner_K16384']['geomean_speedup_percent']:.4f}%. All five pooled paired rounds are positive "
                  'for the four winners and also after adding theM16 boundary. No actual winner is5/5 slower. '
                  'This supports retaining the existing runtime entry under the same finite-cost standard used for9021; '
                  'it does not justify a new K threshold.', '',
              '| Explicit finite cost | Baseline / candidate Event us | Call-time increase | Paired faster rounds |',
              '| --- | --- | --- | --- |']
    for r in decision['finite_costs']:
        lines.append(f"| {r['kid']} / {'×'.join(map(str,r['shape']))} | {r['baseline_us']:.4f} / {r['candidate_us']:.4f} | +{r['time_change_percent']:.4f}% | {r['faster_rounds']}/5 |")
    lines += ['', '9023VGPR increases232→251 and SGPR lane spill46→48; its ISA grows1.36%. LDS and private/VGPR spill/AGPR '
              'remain unchanged. This is a long-term compiler headroom risk and an unmeasured support-domain risk. '
              'The nine9024 winners keep110VGPR and zero spill while SGPR rises58→60 and ISA grows1.81%. '
              'The unchanged runtime control itself moves0.49%, '
              'so small individual differences should not be overinterpreted.', '',
              'Independent CPU replay of the result analysis passed the strict owner/claim/fingerprint check, all15 signed '
              'eight-repeat reference/guard checks, and all five alternating Event rounds on the same51-address pools. '
              'No new GPU test or build was run by this analysis.', '',
              'Exact tested production-target patches: [formal_merge/narrow.patch](formal_merge/narrow.patch). '
              'Before/after hashes and scoped snapshots: [formal_merge/manifest.json](formal_merge/manifest.json). '
              'Full statistics, finite costs, resources and remaining official gates: '
              '[retention_recommendation.json](retention_recommendation.json).', '',
              'Root should coordinate the source application with the other candidates and complete the official linked '
              'identity/API gates before final adoption. Shared helpers,9000,9021 and dispatch thresholds are not part of this patch.', '']
    (HERE/'retention_recommendation.md').write_text('\n'.join(lines))
    print(json.dumps({'status':decision['status'],'summaries':{k:v['geomean_speedup_percent'] for k,v in summaries.items()},
                      'costs':[(r['shape'],r['time_change_percent']) for r in decision['finite_costs']],
                      'merge_patch':manifest['combined_patch_path']}))


if __name__ == '__main__':
    main()
