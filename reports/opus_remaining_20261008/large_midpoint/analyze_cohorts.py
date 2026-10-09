#!/usr/bin/env python3
"""CPU-only 9030 clean ATT cohort arrival evidence for midpoint hypothesis."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / 'diagnostics'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    return {'samples': len(values), 'min': values[0], 'median': statistics.median(values),
            'mean': statistics.mean(values), 'p90': values[int(.9 * (len(values) - 1))], 'max': values[-1]}


def main():
    analysis_path = DIAG / 'merged_att_analysis.json'
    accepted = json.loads(analysis_path.read_text())
    assert accepted['status'] == 'passed'
    capture = next(c for c in accepted['captures'] if c['parent_id'] == 9030)
    assert capture['capture_name'].endswith('_recollect') and capture['status'] == 'passed'
    folder = DIAG / capture['capture_name']
    ui = next(folder.glob('ui_output*'))
    assert sha(ui / 'code.json') == capture['identity']['code_json_sha256']
    code = {r[2]: r for r in json.loads((ui / 'code.json').read_text())['code']}
    refs = {r['wave']: r for r in capture['waves']}
    waves = {}
    for name, ref in refs.items():
        path = ui / name
        assert sha(path) == ref['wave_sha256']
        w = json.loads(path.read_text())['wave']
        x = w['instructions']
        mfma = [e for e in x if code[e[4]][0].startswith('v_mfma_scale')]
        bars = [e for e in x if code[e[4]][0] == 's_barrier']
        assert len(mfma) == 288 and len(bars) == 14
        assert [sum(m[0] < e[0] for m in mfma) for e in bars] == [0, 0] + list(range(24, 265, 24)) + [288]
        transitions = []
        for tile in range(11):
            bar = bars[tile + 2]
            previous = bars[tile + 1]
            begin = previous[0] + previous[3]
            region = [e for e in x if begin <= e[0] < bar[0]]
            wait = next(e for e in reversed(region) if code[e[4]][0] == 's_waitcnt vmcnt(0) lgkmcnt(0)')
            matrix = [e for e in region if code[e[4]][0].startswith('buffer_load') and ' lds' in code[e[4]][0]]
            reads = [e for e in region if code[e[4]][0].startswith('ds_read_b128')]
            assert len(matrix) == (7 if tile + 2 < 12 else 0)
            assert len(reads) == 22
            tile_mfma = mfma[tile * 24:(tile + 1) * 24]
            item = {
                'wave': name, 'tile': tile, 'scope': 'interior' if 1 <= tile <= 9 else 'startup_or_penultimate',
                'first_MFMA_issue': tile_mfma[0][0] + tile_mfma[0][2],
                'eighth_MFMA_issue': tile_mfma[7][0] + tile_mfma[7][2],
                'last_MFMA_issue': tile_mfma[-1][0] + tile_mfma[-1][2],
                'first_to_last_MFMA_issue_clocks': tile_mfma[-1][0] + tile_mfma[-1][2] - tile_mfma[0][0] - tile_mfma[0][2],
                'barrier_attempt': bar[0], 'barrier_release': bar[0] + bar[3],
                'barrier_duration': bar[3], 'barrier_stall': bar[2], 'barrier_pc': hex(code[bar[4]][5]),
                'pre_barrier_wait_duration': wait[3], 'pre_barrier_wait_stall': wait[2],
                'pre_barrier_wait_pc': hex(code[wait[4]][5]),
                'matrix_issue_event_duration_sum': sum(e[3] for e in matrix),
                'matrix_issue_stall_sum': sum(e[2] for e in matrix),
                'matrix_last_issue_success': matrix[-1][0] + matrix[-1][2] if matrix else None,
                'next_operand_first_read_attempt': reads[0][0],
                'next_operand_last_read_attempt': reads[-1][0],
                'last_MFMA_to_barrier_attempt': bar[0] - tile_mfma[-1][0] - tile_mfma[-1][2],
                'barrier_release_to_next_MFMA_issue': mfma[(tile + 1) * 24][0] + mfma[(tile + 1) * 24][2] - bar[0] - bar[3],
            }
            transitions.append(item)
        waves[name] = transitions
    cohorts = []
    wave_interiors = []
    for group_index, group in enumerate(capture['inferred_workgroups_from_publication_release']):
        assert len(group['waves']) == 8
        for tile in range(11):
            members = [waves[name][tile] for name in group['waves']]
            arrival_min = min(m['barrier_attempt'] for m in members)
            arrival_max = max(m['barrier_attempt'] for m in members)
            release_min = min(m['barrier_release'] for m in members)
            release_max = max(m['barrier_release'] for m in members)
            assert release_max - release_min <= 4
            assert release_min >= arrival_max - 4
            last_mfmas = [m['last_MFMA_issue'] for m in members]
            first_mfmas = [m['first_MFMA_issue'] for m in members]
            first_eight = [m['eighth_MFMA_issue'] for m in members]
            latest = [m for m in members if m['barrier_attempt'] >= arrival_max - 4]
            item = {'group_index': group_index, 'tile': tile, 'scope': members[0]['scope'],
                    'arrival_spread_clocks': arrival_max - arrival_min,
                    'release_spread_clocks': release_max - release_min,
                    'latest_arrival_to_release_min_clocks': release_min - arrival_max,
                    'last_MFMA_issue_spread_clocks': max(last_mfmas) - min(last_mfmas),
                    'first_MFMA_issue_spread_clocks': max(first_mfmas) - min(first_mfmas),
                    'eighth_MFMA_issue_spread_clocks': max(first_eight) - min(first_eight),
                    'barrier_duration_sum': sum(m['barrier_duration'] for m in members),
                    'pre_barrier_wait_duration_sum': sum(m['pre_barrier_wait_duration'] for m in members),
                    'latest_arrivals': latest, 'members': members}
            cohorts.append(item)
            if item['scope'] == 'interior':
                wave_interiors.extend(members)
    interiors = [c for c in cohorts if c['scope'] == 'interior']
    latest = [m for c in interiors for m in c['latest_arrivals']]
    anomalies = [c for c in interiors if c['last_MFMA_issue_spread_clocks'] > 1000]
    result = {
        'status': 'passed', 'cpu_only': True, 'GPU_execution': False, 'build_execution': False,
        'accepted_analysis_sha256': sha(analysis_path), 'capture_identity': capture['identity'],
        'script_sha256': sha(__file__), 'cohort_count': 86,
        'interior_cohort_tile_count': len(interiors), 'interior_wave_tile_count': len(wave_interiors),
        'interior_cohort_statistics': {key: stats([c[key] for c in interiors]) for key in
            ['arrival_spread_clocks', 'latest_arrival_to_release_min_clocks', 'last_MFMA_issue_spread_clocks',
             'first_MFMA_issue_spread_clocks', 'eighth_MFMA_issue_spread_clocks', 'release_spread_clocks']},
        'interior_wave_statistics': {key: stats([m[key] for m in wave_interiors]) for key in
            ['barrier_duration', 'pre_barrier_wait_duration', 'matrix_issue_event_duration_sum',
             'matrix_issue_stall_sum', 'first_to_last_MFMA_issue_clocks',
             'last_MFMA_to_barrier_attempt', 'barrier_release_to_next_MFMA_issue']},
        'latest_arrival_wave_statistics': {key: stats([m[key] for m in latest]) for key in
            ['barrier_duration', 'pre_barrier_wait_duration', 'matrix_issue_event_duration_sum',
             'matrix_issue_stall_sum', 'last_MFMA_to_barrier_attempt']},
        'retained_large_spread_cohorts': [
            {'group_index': c['group_index'], 'tile': c['tile'],
             'first_MFMA_issue_spread_clocks': c['first_MFMA_issue_spread_clocks'],
             'last_MFMA_issue_spread_clocks': c['last_MFMA_issue_spread_clocks'],
             'arrival_spread_clocks': c['arrival_spread_clocks'], 'members': c['members']}
            for c in anomalies],
        'interpretation': [
            'Interior end-barrier durations correlate with inter-wave MFMA/arrival skew; the latest arrival-to-release residual is reported separately.',
            'The barrier publishes K+2 and retires all K+1 operand readers. Its wave share cannot be deleted as whole-kernel loss.',
            'Midpoint moves publication of K+1 before its reads and delays K+2 overwrite until all K readers retire, permitting the first eight current MFMAs to overlap pending K+1 supply.',
            'Final advance must retain a terminal C-alias retirement barrier; loops=1 retains the original post-operand prologue barrier.',
            'Wait duration combines VMEM and LDS queue dependencies; no individual memory return time is inferred.',
            'Cohorts come only from synchronized initial publication release and are verified at every interior release; no physical SIMD mapping is claimed.'
        ],
        'cohorts': cohorts,
        'limits': ['One CU correlated ATT waves do not predict full-kernel Event benefit.',
                   'No clean SQ/EA/GRBM throughput inference is made here.',
                   'EXEC masks and lane-active physical bytes are not exported.']
    }
    (HERE / 'cohort_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    s = result['interior_cohort_statistics']; w = result['interior_wave_statistics']; l = result['latest_arrival_wave_statistics']
    lines = ['# 9030 midpoint scheduling evidence', '',
             'Clean Oct8 recollect: 688 complete waves, 86 publication-release cohorts, 288 MFMA per wave.', '',
             f"Interior: {len(interiors)} cohort/tile observations, {len(wave_interiors)} correlated wave/tile observations. Units are shader clocks.", '',
             '| Metric | Median | p90 |', '|---|---:|---:|']
    for label, row in [
        ('Cohort end-barrier arrival spread', s['arrival_spread_clocks']),
        ('Cohort last-MFMA issue spread', s['last_MFMA_issue_spread_clocks']),
        ('Latest arrival to earliest release', s['latest_arrival_to_release_min_clocks']),
        ('All-wave barrier event duration', w['barrier_duration']),
        ('Latest-arrival barrier event duration', l['barrier_duration']),
        ('All-wave combined VMEM/LDS wait event duration', w['pre_barrier_wait_duration']),
        ('Latest-arrival combined VMEM/LDS wait event duration', l['pre_barrier_wait_duration']),
        ('All-wave last MFMA to barrier attempt', w['last_MFMA_to_barrier_attempt']),
        ('Release to next first MFMA', w['barrier_release_to_next_MFMA_issue'])]:
        lines.append(f"| {label} | {row['median']} | {row['p90']} |")
    lines += ['', *result['interpretation'], '',
              f"Retained large-spread observations: {len(anomalies)}. Group8 tile6 has first/last-MFMA spreads 5388/5396 clocks while end-barrier arrival spread is680; four earlier waves spend about4830 clocks in the combined wait and the later waves spend about5500 clocks in async issue events. No samples are discarded or reclassified as ownership contamination.", '',
              'The isolated candidate changes only the 9030 advance schedule. The scale-first prologue, loop predicates, matrix/scale layouts, C64 base, row guards, and output code are preserved.', '',
              'The existing end barrier is moved after the first M repeat for ordinary transitions. Matrix K+2 is issued after this barrier. The last transition retains a second terminal barrier before C staging.', '',
              'Candidate ISA review, numerical guards and no-profiler Event measurements are still required. Full cohort and wave data: [cohort_analysis.json](cohort_analysis.json).']
    (HERE / 'cohort_analysis.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'cohort_stats': s, 'wave_stats': w, 'latest_stats': l}))


if __name__ == '__main__':
    main()
