#!/usr/bin/env python3
"""Finalize stage6 fresh rounds, integrate stable scoped COs, publish a CSV."""
import argparse
import copy
import dataclasses
import hashlib
import json
import shutil
import shlex
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from prepare_opus_repeat import csv_rows, key, shape_row, write_csv, registry_metadata, pure_common
from prepare_co_experiment import load_common, registry

LOCAL=Path(__file__).resolve().parent
ROOT=LOCAL.parents[2]
REL=Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(shape,candidate,control,raw,prior):
    winner=raw[(shape,candidate['kid'])];base=raw[(shape,control)]
    assert set(winner)==set(base)==set(range(6))
    assert all(row['correct'] for row in winner.values()) and all(row['correct'] for row in base.values())
    base_us=statistics.median(row['us'] for row in base.values())
    ratios=[base[i]['us']/winner[i]['us'] for i in range(6)]
    faster=sum(ratio>1 for ratio in ratios)
    if candidate['kid']!=control:
        assert all((winner[i]['position']<base[i]['position'])!=(winner[i+1]['position']<base[i+1]['position']) for i in (0,2,4))
    stable=candidate['kid']!=control and candidate['us']<base_us and faster>=4
    return {**shape_row(shape),'winner_kid':candidate['kid'],'winner_us':candidate['us'],
      'winner_pipeline':candidate['pipeline'],'control_kid':control,'control_us':base_us,
      'previous_us':float(prior[shape]['us']),'baseline_us':candidate['baseline'],
      'wins_csv':candidate['us']<candidate['baseline'],'control_wins_csv':base_us<candidate['baseline'],
      'wins_csv_every_round':max(row['us'] for row in winner.values())<candidate['baseline'],
      'winner_gain_vs_control_pct':(1-candidate['us']/base_us)*100,
      'paired_median_speedup_pct':(statistics.median(ratios)-1)*100,
      'paired_faster_rounds':faster,'stable_candidate_improvement':stable,
      'winner_slower_than_csv_pct':(candidate['us']/candidate['baseline']-1)*100,
      'new_co':candidate['kid']>=100071}


def csv_candidate(shape,row,evidence):
    return {**shape_row(shape),'libtype':'opus','kernelId':row['kid'],'splitK':1,'us':row['us'],
      'kernelName':row['kernelName'],'errRatio':row['error'],'status':'wins_csv' if row['us']<row['baseline'] else 'slower_than_csv',
      'csv_baseline_us':row['baseline'],'wins_csv':row['us']<row['baseline'],
      'evidence_kind':evidence,'wins_csv_every_round':row['every']}


def integrate(campaign,comparisons,repo):
    out=campaign/'integration';out.mkdir()
    selected=defaultdict(list)
    for row in comparisons:
        if row['new_co'] and row['stable_candidate_improvement'] and row['paired_median_speedup_pct']>=.5:
            selected[row['winner_kid']].append(list(key(row)[2:5]))
    receipt={'selected_ids':sorted(selected),'shape_scopes':dict(selected),
      'selection':'Fresh final winner, all six correct, >=4/6 faster than stage5 control, paired median speedup>=0.5%',
      'applied':False,'previous_ids_verified':48240}
    if not selected:
        dump(out/'receipt.json',receipt);return receipt
    for relative in ('csrc/opus_gemm','csrc/include/opus'):
        for source in (ROOT/relative).rglob('*'):
            if source.is_file() and source.suffix in ('.py','.cuh','.h','.hpp','.cc','.cpp','.hip','.cu'):
                copied=repo/source.relative_to(ROOT)
                assert copied.is_file() and sha(source)==sha(copied),str(source)
    before_module=load_common(ROOT,label='stage6_integrate_before')
    experiment_module=load_common(repo,label='stage6_integrate_experiment')
    before,experiment=registry(before_module),registry(experiment_module)
    assert len(before)==48240
    assert all(dataclasses.asdict(instance)==dataclasses.asdict(experiment[kid]) for kid,instance in before.items())
    source=ROOT/REL;original_bytes=source.read_bytes();document=json.loads(original_bytes)
    extended=json.loads((repo/REL).read_text());entries={row['kid']:row for row in extended['gfx950']}
    images=[];headers={}
    for kid,shapes in sorted(selected.items()):
        entry=copy.deepcopy(entries[kid]);entry['supported_shapes']=sorted(shapes)
        if entry.get('static_m') or entry.get('static_n'):
            assert all((not entry.get('static_m') or shape[0]==entry['static_m'])
                       and shape[1]==entry['static_n'] for shape in shapes)
        flags=entry['device_flags']
        if flags not in document['device_flag_sets']:
            document['device_flag_sets'][flags]=copy.deepcopy(extended['device_flag_sets'][flags])
        else:assert document['device_flag_sets'][flags]==extended['device_flag_sets'][flags]
        document['gfx950'].append(entry);document['image_suffixes'][str(kid)]=extended['image_suffixes'][str(kid)]
        image=repo/REL.parent/'gfx950'/(experiment[kid].name+'.co')
        data=image.read_bytes();assert data[:6]==b'\x7fELF\x02\x01' and int.from_bytes(data[18:20],'little')==224
        assert int.from_bytes(data[48:52],'little')&255==79
        images.append({'source':str(image),'destination':str(ROOT/REL.parent/'gfx950'/image.name),'sha256':sha(image)})
        if entry.get('traits_header')=='opus_gemm_traits_stage6_gfx950.cuh':
            header=repo/'csrc/opus_gemm/include/gfx950'/entry['traits_header']
            headers[str(ROOT/'csrc/opus_gemm/include/gfx950'/entry['traits_header'])]=header
    document['_integrated_stage6']={**receipt,'source_campaign':str(campaign)}
    proposal=out/'proposed_manifest.json';dump(proposal,document)
    proposed=registry(load_common(ROOT,manifest=proposal,label='stage6_integrate_proposed'))
    assert set(proposed)==set(before)|set(selected)
    assert all(dataclasses.asdict(instance)==dataclasses.asdict(proposed[kid]) for kid,instance in before.items())
    assert len({instance.name for instance in proposed.values()})==len(proposed)
    for kid in selected:
        lhs,rhs=dataclasses.asdict(proposed[kid]),dataclasses.asdict(experiment[kid])
        lhs['bpreshuffle_co'].pop('supported_shapes',None);rhs['bpreshuffle_co'].pop('supported_shapes',None)
        assert lhs==rhs
    (out/'public_manifest_before.json').write_bytes(original_bytes)
    created=[]
    try:
        assert source.read_bytes()==original_bytes
        for destination,header in headers.items():
            target=Path(destination);assert not target.exists()
            shutil.copy2(header,target);created.append(target);assert sha(target)==sha(header)
        for image in images:
            target=Path(image['destination']);assert not target.exists()
            shutil.copy2(image['source'],target);created.append(target);assert sha(target)==image['sha256']
        source.write_bytes(proposal.read_bytes())
        applied=registry(load_common(ROOT,label='stage6_integrate_applied'))
        assert all(dataclasses.asdict(instance)==dataclasses.asdict(applied[kid]) for kid,instance in proposed.items())
        receipt.update(applied=True,images=images,headers={path:sha(header) for path,header in headers.items()},
                       manifest_before_sha256=hashlib.sha256(original_bytes).hexdigest(),manifest_after_sha256=sha(source))
    except BaseException:
        source.write_bytes(original_bytes)
        for path in created:path.unlink()
        raise
    dump(out/'receipt.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve();final=campaign/'final/confirm'
    audit=json.loads((final/'audit/summary.json').read_text())
    assert audit['completed_shapes']==117 and not audit['pending_shapes']
    prior={key(row):row for row in csv_rows(LOCAL/'stage5_optimize160_20261030/result/public_best_opus421.csv')}
    loss_shapes={shape for shape,row in prior.items() if row['status']=='slower_than_csv'}
    raw=defaultdict(dict);plan=json.loads((final/'repeat_plan.json').read_text())
    metadata={(key(item['shape']),row['kid']):row for item in plan['shapes'] for row in item['candidates']}
    for path in (final/'repeats').glob('gpu*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);shape=key(row['shape']);identity=shape,row['kid']
            assert row['round'] not in raw[identity] and row['kernelName']==metadata[identity]['kernelName']
            assert row['gpu_pci_bus']=={0:'0000:05:00.0',3:'0000:75:00.0',6:'0000:e5:00.0',5:'0000:95:00.0'}[row['gpu_physical_id']]
            raw[identity][row['round']]=row
    measured=defaultdict(list)
    for (shape,kid),rounds in raw.items():
        assert set(rounds)==set(range(6))
        if not all(row['correct'] for row in rounds.values()):continue
        measured[shape].append({'kid':kid,'kernelName':metadata[(shape,kid)]['kernelName'],
          'pipeline':metadata[(shape,kid)]['geometry'][0],'us':statistics.median(row['us'] for row in rounds.values()),
          'baseline':rounds[0]['csv_baseline_us'],'error':max(row['errRatio'] for row in rounds.values()),
          'every':all(row['us']<row['csv_baseline_us'] for row in rounds.values())})
    assert set(measured)==loss_shapes
    experimental={shape:min(rows,key=lambda row:row['us']) for shape,rows in measured.items()}
    experimental_comparisons=[compare(shape,row,int(prior[shape]['kernelId']),raw,prior) for shape,row in experimental.items()]
    out=campaign/'result';out.mkdir()
    write_csv(out/'experimental_final117.csv',experimental_comparisons)
    repo=campaign/'co_experiment/.cache/experiment_repo'
    receipt=integrate(campaign,experimental_comparisons,repo)
    # Reload the public registry after integration; weak experimental winners
    # are excluded using the exact same final batch, without more measurements.
    pure_common.cache_clear();registry_metadata.cache_clear() if hasattr(registry_metadata,'cache_clear') else None
    public_metadata,pairs=registry_metadata();common=pure_common()
    public_registry=common.get_registry('a8w8_blockscale_bpreshuffle','gfx950',256)
    best={shape:min([row for row in rows if row['kid'] in public_metadata
                    and common.a8w8_mxscale_bpreshuffle_supports_shape(public_registry[row['kid']],*shape[2:5])],
                   key=lambda row:row['us']) for shape,rows in measured.items()}
    comparisons=[compare(shape,row,int(prior[shape]['kernelId']),raw,prior) for shape,row in best.items()]
    write_csv(out/'public_final117_comparisons.csv',comparisons)
    current=[csv_candidate(shape,row,'stage6_final_fresh_six_round_public') for shape,row in best.items()]
    kept=[row for shape,row in prior.items() if shape not in loss_shapes];assert len(kept)==304
    rows=[{field:row[field] for field in current[0]} for row in kept]+current;rows.sort(key=key)
    write_csv(out/'public_best_opus421.csv',rows)
    losses=sorted([row for row in comparisons if not row['wins_csv']],key=lambda row:row['winner_slower_than_csv_pct'],reverse=True)
    if losses:write_csv(out/'public_remaining_losses.csv',losses)
    groups=defaultdict(list)
    for row in comparisons:groups[(row['N'],row['K'])].append(row)
    group_rows=[{'N':n,'K':k,'original_losses':len(rs),'new_wins':sum(r['wins_csv'] for r in rs),
      'remaining_losses':sum(not r['wins_csv'] for r in rs),'stable_candidate_gains':sum(r['stable_candidate_improvement'] for r in rs)}
      for (n,k),rs in sorted(groups.items())]
    write_csv(out/'groups.csv',group_rows)
    gains=[row['winner_gain_vs_control_pct'] for row in comparisons if row['stable_candidate_improvement']]
    wins=sum(row['wins_csv'] for row in comparisons)
    metrics={'reference_wins':304,'reference_losses':117,'retained_previous_wins':304,'fresh_final_retested_shapes':117,
      'fresh_final_wins':wins,'fresh_final_losses':117-wins,'combined421_wins':304+wins,'combined421_losses':117-wins,
      'stable_candidate_improved_shapes':len(gains),'new_win_with_control_still_slower':sum(row['wins_csv'] and not row['control_wins_csv'] for row in comparisons),
      'wins_where_old_control_also_wins_now':sum(row['wins_csv'] and row['control_wins_csv'] for row in comparisons),
      'max_stable_candidate_time_reduction_pct':max(gains,default=0),'median_stable_candidate_time_reduction_pct':statistics.median(gains) if gains else 0,
      'remaining_losses_median_pct':statistics.median(row['winner_slower_than_csv_pct'] for row in losses) if losses else 0,
      'remaining_losses_max_pct':max((row['winner_slower_than_csv_pct'] for row in losses),default=0),
      'remaining_loss_below_pct':{str(limit):sum(row['winner_slower_than_csv_pct']<limit for row in losses) for limit in (1,3,5,10)},
      'new_co_integrated_ids':receipt['selected_ids'],'new_co_best_shapes':sum(row['new_co'] for row in comparisons),
      'public_registry_count':len(public_metadata),'final_audit':audit,'csv':str(out/'public_best_opus421.csv'),
      'definition':'117 original losses use this final fresh-six-round batch; prior304 wins retained; candidate gains and control drift separated.'}
    dump(out/'public_metrics.json',metrics)
    items=[]
    for row in rows:
        shape,kid=key(row),int(row['kernelId']);assert row['kernelName']==public_metadata[kid]['kernelName']
        assert common.a8w8_mxscale_bpreshuffle_supports_shape(public_registry[kid],*shape[2:5])
        items.append({'shape':shape_row(shape),'csv_baseline_us':float(row['csv_baseline_us']),'remote_winner':None,
          'candidates':[{**public_metadata[kid],'roles':['current_step4_best_control'],'exact_opposite_kids':pairs[kid],
                        'us':float(row['us']),'errRatio':float(row['errRatio']),'correct':True}]})
    dump(out/'public421_plan.json',{'version':2,'family':'a8w8_blockscale_bpreshuffle','metadata_module':'prepare_opus_repeat',
      'correctness_limit':.05,'selection':'Current public421 identities; precompile only, no additional benchmark','shapes':items})
    (campaign/'public_env.sh').write_text('source '+shlex.quote(str(LOCAL/'env.sh'))+'\n'
       +'export ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0\n'
       +'export AITER_JIT_DIR='+shlex.quote(str(campaign/'.cache/public_jit'))+'\n')
    # All clearly slow CO trial combinations remain reviewable, including
    # screened-out candidates which have no final six-round result.
    slow=[]
    for path in (campaign/'co_experiment/run/screen/repeats').glob('gpu*.summary.json'):
        for row in json.loads(path.read_text())['candidates']:
            if row['kid']>=100071 and row['all_rounds_correct'] and row['median_us']>=1.2*row['csv_baseline_us']:
                slow.append({**shape_row(key(row['shape'])),'kid':row['kid'],'kernelName':row['kernelName'],
                 'two_round_median_us':row['median_us'],'baseline_us':row['csv_baseline_us'],
                 'slower_than_csv_pct':(row['median_us']/row['csv_baseline_us']-1)*100,'evidence':'two_round_screen_only'})
    if slow:write_csv(out/'new_co_screen_outliers20pct.csv',slow)
    dump(out/'new_co_screen_outlier_counts.json',{'candidate_shape_pairs':len(slow),'unique_shapes':len({key(row) for row in slow}),
                                               'unique_kids':len({row['kid'] for row in slow})})
    lines=[f"本轮结果：{304+wins}领先 / {117-wins}落后。",'',
      f"原304项领先保留；原117项落后全部做新的最终六轮，{wins}项越线。",
      f"其中{metrics['new_win_with_control_still_slower']}项原最优仍落后，靠新候选越线；{metrics['wins_where_old_control_also_wins_now']}项原control也越线。",
      f"{len(gains)}项有稳定成对候选收益（至少4/6轮更快），耗时下降中位数{metrics['median_stable_candidate_time_reduction_pct']:.2f}%，最大{metrics['max_stable_candidate_time_reduction_pct']:.2f}%。",'',
      f"集成新CO：{receipt['selected_ids']}，成为{metrics['new_co_best_shapes']}个shape的当前最优。旧48240配置完整保留。",
      '输出：public_best_opus421.csv、public_final117_comparisons.csv、public_remaining_losses.csv。运行环境../public_env.sh。',
      '最终使用物理0/3/5/6；只tune OPUS，FlyDSL固定六字段CSV。A端2GiB范围没有改动。','',
      '| N | K | 原落后 | 新越线 | 仍落后 | 稳定候选收益 |','|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {row['N']} | {row['K']} | {row['original_losses']} | {row['new_wins']} | {row['remaining_losses']} | {row['stable_candidate_gains']} |" for row in group_rows]
    (out/'public_result_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:metrics[key] for key in ('combined421_wins','combined421_losses','stable_candidate_improved_shapes','new_co_integrated_ids')},ensure_ascii=False))


if __name__=='__main__':main()
