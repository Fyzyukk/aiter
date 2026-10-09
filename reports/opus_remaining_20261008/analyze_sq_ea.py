#!/usr/bin/env python3
"""Summarize independent SQ/EA passes; no uncalibrated GRBM utilization."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
BASE=HERE/'diagnostics'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan=json.loads((BASE/'plan.json').read_text())
    claims=[json.loads(line) for line in (BASE/'counter_claim.jsonl').read_text().splitlines()]
    rows=[]
    for index,target in enumerate(plan['targets']):
        folder=BASE/f'target{index}_kid{target["kid"]}_sq_ea'
        parsed=folder/'parsed.json'
        if not parsed.exists():continue
        name=folder.name;epochs=[];current=None;claimed=None
        for record in claims:
            if record['event']=='claimed':claimed=record
            if record['event']=='start' and record['command']['name']==name:
                current={'claimed':claimed,'start':record,'monitors':[]}
            elif current and record['event']=='owner_identity' and record['name']==name:current['owner']=record
            elif current and record['event']=='monitor':current['monitors'].append(record)
            elif current and record['event']=='end' and record['name']==name:
                current['end']=record;epochs.append(current);current=None
        epoch=epochs[-1];owner=epoch.get('owner',{}).get('host_pid')
        unknown=[{'time':r['time'],'process':p} for r in epoch['monitors'] for p in r['processes'] if p['pid']!=owner]
        clean=epoch['end']['returncode']==0 and not epoch['end']['contamination'] and not unknown
        data=json.loads(parsed.read_text());groups=[]
        for group in data['groups']:
            samples=group['samples'];metrics=group['median_metrics'];counts=group['median_counters']
            allowed=['wave_any_wait_pct_estimate','wave_issue_wait_pct_estimate','wave_execute_pct_estimate',
                     'dram_read_bytes','dram_write_bytes','dram_total_bytes','profiled_dispatch_ns',
                     'tcp_ta_addr_stall_pct_of_interface_clock_estimate','tcp_tcr_stall_pct_of_interface_clock_estimate',
                     'tcp_lfifo_stall_pct_of_interface_clock_estimate','issued_f8_operations']
            groups.append({'kernel_name':group['kernel_name'],'selected_dispatches':len(samples),
                           'median_metrics':{k:metrics[k] for k in allowed if k in metrics},
                           'median_counters':counts,'source_file':group['source_file']})
        rows.append({'target_index':index,'kid':target['kid'],'shape':target['shape'],
                     'profile_rotation':1,'application_sha256':sha(folder/'application.json'),
                     'parsed_sha256':sha(parsed),'strict_clean_epoch':clean,'unknown_monitor_processes':unknown,
                     'gpu_pci_bdf':epoch['claimed']['gpu']['bdf'],'groups':groups})
    result={'status':'finite_SQ_EA_passes_reviewed','rows':rows,'plan_sha256':sha(BASE/'plan.json'),
            'claim_sha256':sha(BASE/'counter_claim.jsonl'),'script_sha256':sha(Path(__file__)),
            'limits':['Separate independent passes; no duration used to claim speedup',
                      'Warm repeated buffer rotation1; physical traffic does not describe cold streaming Event pool',
                      'Nonowner monitor entries remain ambiguous and excluded from clean mechanistic evidence',
                      'GRBM CSV counter reductions do not support absolute MFMA utilization or occupancy',
                      'Wave waits are cumulative quad-cycle ratios; interface stalls use nonwindowed contextual denominators']}
    (BASE/'sq_ea_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'strict_clean':sum(r['strict_clean_epoch'] for r in rows),
                      'excluded_indices':[r['target_index'] for r in rows if not r['strict_clean_epoch']]}))


if __name__=='__main__':main()
