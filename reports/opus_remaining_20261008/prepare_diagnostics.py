#!/usr/bin/env python3
"""Prepare finite, identity-bound diagnostics; GPU execution is root-owned."""
import copy
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/opus_bound_analysis_20261007'
RESUME = ROOT / 'reports/opus_resume_20261008'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    inventory = json.loads((OUT / 'inventory.json').read_text())
    entries = inventory['entries']
    selected = []

    def add(parent, shape, purpose):
        matches = [e for e in entries if e['parent_id'] == parent and shape in e['winner_shapes']]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one actual winner configuration: {parent} {shape}')
        entry = matches[0]
        selected.append({'kid': parent, 'shape': shape, 'seed': 17, 'signed': True,
                         'purpose': purpose, 'private_baselines': {},
                         'symbol': entry['symbol'], 'kernel_function': entry['kernel_function'], 'traits': entry['traits'],
                         'instruction_sha256': entry['instruction_sha256'],
                         'actual_configuration_ids': entry['actual_configuration_ids']})

    add(9020, [1536,7168,384], 'same-grid fixed short K, existing K0-first prologue')
    add(9020, [1536,7168,16384], 'same-grid runtime long K and scale panel')
    for e in entries:
        if e['parent_id'] == 9020 and e['symbol'] not in [t['symbol'] for t in selected]:
            add(9020, e['representative'], 'remaining exact 9020 branch')
    add(9022, [800,7168,384], 'same-grid short K, scale-first prologue')
    add(9022, [800,7168,16384], 'same-grid long K, panel32 and consumer retirement')
    add(9023, [544,7168,16384], 'runtime narrow long K with M tail')
    add(9023, [576,7168,7168], 'runtime narrow full M tile control')
    add(9024, [1344,768,7168], 'fixed K7168, 252 WG, GroupM4 tail group')
    add(9024, [1408,768,7168], 'fixed K7168, 264 WG, same swizzle control')
    add(9030, [65536,16384,1536], 'actual large-output winner, 2 GiB C physical writes')
    small = [(9046,[256,7168,16384]), (9055,[64,7168,7168]),
             (9042,[192,768,7168]), (9062,[144,7168,16384]),
             (9040,[8,7168,384]), (9047,[64,7168,384]), (9051,[1,65536,1536])]
    for parent, shape in small:
        add(parent, shape, 'finite small-family mechanism representative')

    plan = {'status': 'prepared_current_selected_diagnostic_plan',
            'source_head': inventory['source_head'], 'current_parent_count': 26,
            'official_binary': inventory['official_module']['path'],
            'official_binary_sha256': inventory['official_module']['sha256'],
            'baseline': inventory['baseline'], 'targets': selected}
    plan_path = OUT / 'diagnostics/plan.json'
    write(plan_path, plan)
    att_template = json.loads((RESUME / 'diagnostics/att_queue.json').read_text())
    global_env = copy.deepcopy(att_template['env'])
    global_env['AITER_JIT_DIR'] = str(Path(plan['official_binary']).parent)
    runner = str(OLD / 'official_smoke.py')
    prefix = ['/opt/venv/bin/python3', runner, '--plan', str(plan_path)]
    check = {'name': 'current_remaining_signed_repeat_guards',
             'argv': prefix + ['--output',str(OUT/'diagnostics/correctness.json'),
                               '--check-only','--repetitions','2'],
             'log': str(OUT/'diagnostics/correctness.log')}
    commands = [check]
    for index, target in enumerate(selected):
        name = f'target{index}_kid{target["kid"]}_att'
        directory = OUT / 'diagnostics' / name
        env = copy.deepcopy(att_template['commands'][0]['env'])
        env.update(ROCPROF_OUTPUT_FILE_NAME=name, ROCPROF_OUTPUT_PATH=str(directory),
                   ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=target['kernel_function'])
        commands.append({'name': name,
            'argv': prefix + ['--output',str(directory/'application.json'), '--profile',
                             '--target-index',str(index),'--iters','11','--profile-rotation','1'],
            'env': env, 'log':str(directory/'rocprof.log')})
    write(OUT/'diagnostics/att_queue.json', {'env':global_env,'commands':commands,
        'purpose':'Correctness then actual current-selected PC timelines, finite configurations',
        'limits':['ATT only SE0/CU0, complete waves checked before phase attribution',
                  'Profiler durations explain mechanisms; no speedup decisions',
                  'GRBM window uncalibrated; no absolute utilization/occupancy'],
        'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest()})

    # First collect one compatible SQ/EA pass for each mechanism representative.
    # Counter collection uses the direct owned Python process, avoiding nested PID ambiguity.
    evidence = json.loads((OLD/'counter_evidence.json').read_text())
    extra = (OLD/'gfx950_extra_counters.yaml').read_text()
    counter_template = json.loads((RESUME/'diagnostics/queue.json').read_text())['commands'][0]['env']
    counter_commands = []
    for index, target in enumerate(selected):
        if target['kid'] == 9051:  # exact unchanged four-pass historical evidence already exists
            continue
        name = f'target{index}_kid{target["kid"]}_sq_ea'
        directory = OUT/'diagnostics'/name
        env = copy.deepcopy(counter_template)
        env.update(ROCPROF_OUTPUT_FILE_NAME=name, ROCPROF_OUTPUT_PATH=str(directory),
            ROCPROF_COUNTERS='pmc: '+' '.join(evidence['groups']['sq_ea']['aggregate_counters']),
            ROCPROF_EXTRA_COUNTERS_CONTENTS=extra,
            ROCPROF_KERNEL_FILTER_INCLUDE_REGEX='gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel',
            ROCPROF_COUNTER_COLLECTION='1')
        env.pop('ROCPROF_KERNEL_FILTER_RANGE',None)
        counter_commands.append({'name':name,'argv':prefix + [
            '--output',str(directory/'application.json'),'--profile','--target-index',str(index),
            '--iters','51','--profile-rotation','1'], 'env':env,'log':str(directory/'rocprof.log')})
    write(OUT/'diagnostics/counter_queue.json',{'env':global_env,'commands':counter_commands,
        'purpose':'Matched SQ work and EA physical reads/writes, one pass per finite representative',
        'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest()})
    print(json.dumps({'targets':len(selected),'att_commands':len(commands),
                      'counter_commands':len(counter_commands), 'plan':str(plan_path)}))


if __name__ == '__main__':
    main()
