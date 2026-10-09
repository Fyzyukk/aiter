#!/usr/bin/env python3
"""Audit root-owned finite9062 numerical and paired complete Event decision."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('fine_claim_cpu', HERE.parent / 'small_att_analysis.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return {'path': str(path), 'sha256': sha(path)}


def main():
    queue = json.loads((HERE / 'screen_queue.json').read_text())
    claim_path = HERE / 'screen_claim.jsonl'
    if not claim_path.exists():
        candidates = list(HERE.glob('*claim*.jsonl'))
        assert len(candidates) == 1
        claim_path = candidates[0]
    claims = [json.loads(line) for line in claim_path.read_text().splitlines() if line.strip()]
    results = []
    winner_rows = []
    for command in queue['commands']:
        argv = command['argv']; path = Path(argv[argv.index('--output') + 1])
        result = json.loads(path.read_text()); plan_path = Path(argv[argv.index('--plan') + 1])
        assert result['status'] == 'passed' and result['plan'] == json.loads(plan_path.read_text())
        assert result['runner_sha256'] == sha(ROOT / 'reports/opus_bound_analysis_20261007/experiment_runner.py')
        assert result['plan_sha256'] == sha(plan_path)
        assert result['libraries'] == result['plan']['sealed_libraries']
        assert result['rounds'] == 5 and result['iters'] == 51 and not result['profiling_only']
        own = common.clean_claim(command, result, claims)
        rows = []
        for row in result['rows']:
            for c in row['correctness'].values():
                assert c == {'repetitions': 8, 'errRatio': 0, 'output_guards': True, 'workspace_guards': True, 'repeatable': True}
            item = {'shape': row['shape'], 'target_index': row['target_index'], 'all_signed8repeat_reference_guards_passed': True,
                    'independent_timing_median_us': row['median_us']}
            if 'event_confirmation' in row:
                event = row['event_confirmation']
                assert event['status'] == 'passed' and event['rounds'] == 5 and event['iters_per_graph'] == 51
                assert event['shared_pool'] is True and event['rotation']['count'] == len(event['pool_pointers']) == 51
                assert all(set(p)=={'a','b','c','sfa','sfb','workspace'} and all(p[n]>0 and p[n]%16==0 for n in ['a','b','c','sfa']) for p in event['pool_pointers'])
                assert all((p['workspace']>0 and p['workspace']%16==0) if result['plan']['workspace'] else p['workspace']==0 for p in event['pool_pointers'])
                assert len({p['b'] for p in event['pool_pointers']}) == len({p['c'] for p in event['pool_pointers']}) == 51
                assert event['includes'] == 'complete private launch; split-K producer and reducer'
                measurements = event['measurements']
                assert len(measurements) == 10
                rounds = []
                for i in range(5):
                    pair = {m['label']: m for m in measurements if m['round'] == i}
                    assert set(pair) == {'baseline','candidate'}
                    expected_order = ['baseline','candidate'] if i % 2 == 0 else ['candidate','baseline']
                    for m in pair.values():
                        assert m['order'] == expected_order and m['iters'] == m['rotation_count'] == 51
                        assert len(m['all_pool_checks']) == 51 and all(
                            c == {'pool_index': n, 'output_repeatable': True, 'workspace_repeatable': True,
                                  'errRatio':0,'output_guards':True,'workspace_guards':True}
                            for n,c in enumerate(m['all_pool_checks']))
                    a,b = pair['baseline']['us_per_call'], pair['candidate']['us_per_call']
                    rounds.append({'round':i,'order':expected_order,'baseline_us':a,'candidate_us':b,
                                   'speedup':a/b,'candidate_time_change_percent':100*(b/a-1)})
                a = statistics.median(r['baseline_us'] for r in rounds)
                b = statistics.median(r['candidate_us'] for r in rounds)
                item.update({'complete_shared_pool_ABBA_Event': rounds,
                             'median_us': {'baseline':a,'candidate':b}, 'median_speedup':a/b,
                             'median_time_change_percent':100*(b/a-1),
                             'faster_rounds':sum(r['speedup']>1 for r in rounds),
                             'all_shared_pool_repeat_reference_output_workspace_guards_passed':True})
                winner_rows.append(item)
            rows.append(item)
        results.append({'name':command['name'],'result':ref(path),'plan':ref(plan_path),'clean_claim':own,'rows':rows})
    assert len(winner_rows) == 2 and all(r['median_speedup'] <= 1 for r in winner_rows)
    report = {'status':'rejected_no_positive_complete_Event_gain','adopted':False,
              'production_modified':False,'cpu_postprocessing_only':True,
              'reason':'Both exact fixed9062 current winner shapes have median complete-call Event speedup below1; all signed numerical/repeatability/guards pass. The initial matrix-first handoff is not adopted.',
              'mechanism_limit':'Scale vmcnt(0) includes earlier matrix requests and one extra generated lgkmcnt(0) remains in ISA. Without candidateATT, small Event losses do not prove which wait caused the loss.',
              'retry_policy':'Retain current selected. Do not repeat this same initial matrix-first mechanism without new counter/PC evidence and a distinct safe hypothesis.',
              'results':results,'winner_decisions':winner_rows,'queue':ref(HERE/'screen_queue.json'),
              'claim_log':ref(claim_path),'script':ref(Path(__file__))}
    (HERE/'results_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    md = (HERE/'review.md').read_text().split('\n<!-- EVENT_DECISION -->',1)[0]
    md += '\n<!-- EVENT_DECISION -->\n\n完整调用Event决定：拒绝此候选，保留原fixed9062 selected。两个实际赢家的signed8repeat/reference/output+workspace guards和51池地址的逐轮repeat/guard全部通过。clean owner/物理PCI65/HIP2、同一共享池5轮AB/BA完整producer+reducer Event均验证。\n\n'
    md += '| shape M,N,K | baseline median us | candidate median us | speedup | 更快轮数 |\n| --- | ---: | ---: | ---: | ---: |\n'
    for r in winner_rows:
        md += f"| {','.join(map(str,r['shape']))} | {r['median_us']['baseline']:.6f} | {r['median_us']['candidate']:.6f} | {r['median_speedup']:.6f} | {r['faster_rounds']}/5 |\n"
    md += '\n两项median均无正改善。原scale vmcnt0可能full-drain早先matrix和新增generated LGKM wait仍是风险线索；没有candidateATT不能把微小退步归因到某条wait。此方向关闭，逐轮原值见results_analysis.json；后续ring consumer请求顺序为独立机制。\n'
    (HERE/'review.md').write_text(md)
    print(json.dumps({'status':report['status'],'winners':[(r['shape'],r['median_speedup'],r['faster_rounds']) for r in winner_rows]}))


if __name__=='__main__':main()
