#!/usr/bin/env python3
"""CPU-only exact selected-device analysis of seven small-family ATT captures.

Reads corrected queue/claims, official profile applications, decoder output
and embedded ELF images. Never builds, loads HIP or executes a GPU command.
Instruction classifications follow current opcode/dependency evidence; no
9021-specific PC map is reused.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from urllib.parse import unquote, urlparse, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIAG = HERE / 'diagnostics'
OLD = ROOT / 'reports/opus_bound_analysis_20261007'
OCT8 = ROOT / 'reports/opus_resume_20261008'
TARGETS = range(14, 21)
PLAN = DIAG / 'plan.json'
QUEUE = DIAG / 'att_queue.json'
CLAIM = DIAG / 'att_claim_corrected.jsonl'
RECOLLECT_QUEUE = DIAG / 'recollect_queue.json'
RECOLLECT_CLAIM = DIAG / 'recollect_claim.jsonl'
ORIGINAL_CLAIM = DIAG / 'att_claim.jsonl'
CORRECTNESS = DIAG / 'correctness.json'
REVIEW = HERE / 'small_family_review.json'
FORMAL = OCT8 / 'formal_selected/identity_audit.json'
OFFICIAL = OCT8 / 'jit_formal_selected/module_deepgemm_opus.so'
TOOL = '/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


elf = load_module('small_att_elf_cpu', OCT8 / 'formal_selected/smoke_common.py')
old_events = load_module('small_att_event_cpu', OCT8 / 'diagnostics/analyze_att.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def describe(values):
    values = sorted(values)
    if not values:
        return {'samples': 0}
    return {'samples': len(values), 'min': values[0], 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p90': values[int(.9 * (len(values) - 1))],
            'max': values[-1]}


def kind(text):
    if text.startswith(';'):
        return 'symbol_marker'
    op = text.split()[0]
    if op.startswith('v_mfma_scale'):
        return 'scaled_MFMA'
    if op == 's_barrier':
        return 'barrier'
    if op.startswith('s_waitcnt'):
        return 'waitcnt'
    if op.startswith('buffer_load'):
        if ' lds' in text:
            return 'matrix_async_VMEM_issue'
        if 'ubyte' in op or 'ushort' in op:
            return 'scale_VMEM_issue'
        return 'matrix_register_VMEM_issue'
    if op.startswith('s_load'):
        return 'kernarg_SMEM_issue'
    if op.startswith('ds_read'):
        return 'LDS_byte_read' if 'u8' in op else 'LDS_vector_read'
    if op.startswith('ds_write'):
        return 'LDS_byte_write' if op.endswith('b8') else 'LDS_vector_write'
    if op.startswith('buffer_store'):
        return 'global_output_store'
    if 'cvt_pk_bf16' in op:
        return 'BF16_conversion'
    if op.startswith('s_nop'):
        return 'explicit_nop'
    if op == 's_endpgm':
        return 'endpgm'
    if op.startswith('s_cbranch') or op == 's_branch':
        return 'scalar_branch'
    if op.startswith('s_'):
        return 'scalar_other'
    if op.startswith('v_'):
        return 'vector_other'
    return 'other'


def event(e, code, origin=0):
    row = code[e[4]]
    return {'pc': hex(row[5]), 'code_line': e[4], 'instruction': row[0],
            'kind': kind(row[0]), 'time': e[0] - origin, 'stall': e[2],
            'duration': e[3], 'successful_issue': e[0] + e[2] - origin}


def breakdown(insts, code, begin, end):
    counts = Counter()
    cursor = begin
    for e in insts:
        # Rare adjacent decoder events overlap by one4-clock quantum. Give
        # their shared interval to the earlier event once; retain raw events.
        overlap = max(0, min(end, e[0] + e[3]) - max(begin, e[0], cursor))
        if overlap:
            counts[kind(code[e[4]][0])] += overlap
            cursor = max(cursor, min(end, e[0] + e[3]))
    require(sum(counts.values()) <= end - begin, 'Overlapping decoder events')
    counts['unattributed_decoder_gap'] += end - begin - sum(counts.values())
    return dict(sorted(counts.items()))


def clean_claim(command, app, claims):
    candidates = []
    for index, start in enumerate(claims):
        if start.get('event') != 'start' or start.get('command', {}).get('name') != command['name']:
            continue
        next_start = next((i for i in range(index + 1, len(claims)) if claims[i].get('event') == 'start'), len(claims))
        window = claims[index + 1:next_start]
        ends = [r for r in window if r.get('event') == 'end' and r.get('name') == command['name']
                and r['time'] >= app['finished']]
        if len(ends) == 1 and start['time'] <= app['started'] and ends[0]['returncode'] == 0 \
                and ends[0]['contamination'] is False:
            candidates.append((index, start, ends[0], [r for r in window if r['time'] <= ends[0]['time']]))
    require(len(candidates) == 1, f'{command["name"]}: one clean enclosing corrected claim required')
    index, start, end, window = candidates[0]
    require(start['command'] == command, 'Executed corrected ATT command differs')
    require(not any(r.get('event') in ['claimed', 'released', 'external_work_started', 'interrupted', 'contamination']
                    for r in window), 'Physical claim epoch changed or contaminated')
    owners = [r for r in window if r.get('event') == 'owner_identity' and r.get('name') == command['name']]
    require(len(owners) == 1, 'One unique owner identity required')
    owner = owners[0]
    require(owner['new_host_pids'] == [owner['host_pid']] and owner['host_pid'] not in owner['baseline_host_pids']
            and owner['launcher_sha256'] == sha(OLD / 'owned_python_launch.py')
            and owner['method'] == 'KFD open + unique sysfs registration + same-mm runpy handshake',
            'Owner handshake differs')
    monitors = [r for r in window if r.get('event') == 'monitor']
    require(monitors and all(r['child_pid'] == owner['inner_pid'] for r in monitors)
            and all(all(p['pid'] == owner['host_pid'] for p in r['processes']) for r in monitors),
            'Unexpected monitored GPU process or owner inner PID differs')
    claim = next((r for r in reversed(claims[:index]) if r.get('event') == 'claimed'), None)
    require(claim and app['gpu']['pci_bdf'] == app['gpu']['expected_pci_bdf'] == claim['gpu']['bdf']
            and app['gpu']['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index']), 'Physical GPU identity differs')
    return {'clean': True, 'claimed': claim, 'start': start, 'end': end,
            'owner_identity': owner, 'monitor_count': len(monitors)}


def parse_traits(target, kernel):
    match = re.search(r'traits_gfx950<([^>]+)>', kernel['demangled'])
    require(match, 'Missing actual scalar trait type')
    vals = [v.strip() for v in match[1].split(',')]
    bm, bn = int(vals[0]), 128 if 'fine_traits' in kernel['demangled'] else int(vals[1])
    if 'fine_traits' in kernel['demangled']:
        tm, tn, split, cluster = int(vals[1]), int(vals[2]), int(vals[5]), int(vals[4])
        wave_k = 1
    elif 'small_register_traits' in kernel['demangled']:
        tm, tn, wave_k, split, cluster = int(vals[2]), int(vals[3]), int(vals[5]), 1, None
    else:
        tm, tn, wave_k, split, cluster = int(vals[2]), int(vals[3]), 1, int(vals[12]), int(vals[5])
    em, en = bm // (tm * 16), bn // (tn * 16)
    ktiles = target['shape'][2] // 128
    loops = [(ktiles // split + (sk < ktiles % split)) for sk in range(split)]
    wave_loops = sorted({n // wave_k + (wk < n % wave_k) for n in loops for wk in range(wave_k)})
    return {'BM': bm, 'BN': bn, 'T_M': tm, 'T_N': tn, 'wave_k': wave_k, 'split': split,
            'waves_per_WG': tm * tn * wave_k, 'MFMA_per_wave_K128_tile': em * en,
            'expected_MFMA_counts_for_full_wave': sorted({n * em * en for n in wave_loops}),
            'cluster': cluster}


def exact_code(folder, ui, name, kernel):
    doc = read(ui / 'code.json')
    require(doc['header'] == 'ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle',
            'Decoder code schema differs')
    require(doc['code'] is not None, 'Decoder code=null; application passed but capture is unusable for instruction analysis')
    code = {r[2]: r for r in doc['code']}
    require(len(code) == len(doc['code']), 'Duplicate decoder code line')
    ids = {r[4] for r in code.values()}
    require(len(ids) == 1, 'ATT code spans multiple code objects')
    object_id = next(iter(ids))
    image_files = list(folder.glob(f'*_code_object_id_{object_id}.out'))
    require(len(image_files) == 1, 'One extracted current device image required')
    image = image_files[0]
    start, func = old_events.symbol_bytes(image, kernel['name'])
    require(digest(func) == kernel['instruction_sha256'] and len(func) == kernel['instruction_bytes'],
            'Captured exact FUNC bytes differ from official selected')
    image_rows = {r['name']: r for r in elf.summarize_image(image.read_bytes())}
    require(kernel['name'] in image_rows and elf.identity(image_rows[kernel['name']], kernel)['matches'],
            'Captured instruction/fullmetadata/descriptor differs')
    raw = subprocess.check_output([TOOL, '-d', '--mcpu=gfx950', str(image)], text=True)
    isa = {}
    for line in raw.splitlines():
        match = re.match(r'^\s*(.*?)\s*// ([0-9A-Fa-f]+):', line)
        if match:
            pc = int(match[2], 16)
            if start <= pc < start + len(func):
                isa[pc] = re.sub(r'\s+', ' ', match[1].strip())
    markers = 0
    for row in code.values():
        if row[0].startswith(';'):
            markers += 1
            require(kernel['name'] in row[0], 'Decoder marker symbol differs')
        else:
            require(start <= row[5] < start + len(func) and row[5] in isa
                    and re.sub(r'\s+', ' ', row[0].strip()) == isa[row[5]], 'Decoder PC instruction differs')
    require(markers == 1 and len(isa) == len(code) - 1, 'Complete static FUNC disassembly not covered')
    stats_files = list(folder.glob('stats_ui_output_*.csv'))
    require(len(stats_files) == 1, 'One decoder stats CSV required')
    stats = list(csv.DictReader(stats_files[0].open()))
    require(len(stats) == len(doc['code']), 'Stats/code row count differs')
    for a, b in zip(stats, doc['code']):
        require(int(a['CodeObj']) == b[4] and int(a['Vaddr']) == b[5] and a['Instruction'] == b[0]
                and [int(a[x]) for x in ['Hitcount', 'Latency', 'Stall', 'Idle']] == b[6:10],
                'Decoder CSV stats differ from code JSON')
    results = read(folder / f'{name}_results.json')['rocprofiler-sdk-tool']
    require(len(results) == 1, 'Expected one profiler process record')
    tool = results[0]
    objects = [o for o in tool['code_objects'] if o['code_object_id'] == object_id]
    require(len(objects) == 1, 'Code-object callback missing')
    parsed = urlparse(objects[0]['uri'])
    qs = parse_qs(parsed.fragment)
    path = Path(unquote(parsed.path))
    require(parsed.scheme == 'file' and path.resolve() == OFFICIAL.resolve(), 'Captured code-object URI not official module')
    offset, size = int(qs['offset'][0]), int(qs['size'][0])
    require(OFFICIAL.read_bytes()[offset:offset + size] == image.read_bytes(), 'Exact loaded module device URI slice differs')
    syms = [s for s in tool['kernel_symbols'] if s['kernel_name'] == kernel['name'] + '.kd'
            and s['code_object_id'] == object_id]
    require(len(syms) == 1, 'Exact traced kernel callback missing')
    dispatches = [d for d in tool['buffer_records']['kernel_dispatch']
                  if d['dispatch_info']['kernel_id'] == syms[0]['kernel_id']]
    require(dispatches, 'Exact traced kernel dispatches missing')
    ui_match = re.search(r'ui_output_agent_(\d+)_dispatch_(\d+)', ui.name)
    require(ui_match, 'Decoder UI directory lacks agent/dispatch identity')
    agent_id, dispatch_id = map(int, ui_match.groups())
    traced = [d for d in dispatches if d['dispatch_info']['dispatch_id'] == dispatch_id
              and d['dispatch_info']['agent_id']['handle'] == agent_id]
    require(len(traced) == 1, 'Decoder agent/dispatch does not match exact symbol trace')
    return code, {'image': ref(image), 'FUNC_sha256': digest(func), 'FUNC_bytes': len(func),
                  'full_metadata_normalized_descriptor_equal': True, 'static_PC_instruction_matches': len(isa),
                  'code_json': ref(ui / 'code.json'), 'stats_csv': ref(stats_files[0]),
                  'code_object_callback': objects[0], 'kernel_symbol_callback': syms[0],
                  'captured_dispatch': traced[0], 'matched_kernel_dispatch_count': len(dispatches)}, tool


def capture(index, command, target, plan, claims, kernel):
    folder = Path(command['argv'][command['argv'].index('--output') + 1]).parent
    ui_files = list(folder.glob('ui_output_*'))
    require(len(ui_files) == 1, f'target{index}: one completed corrected UI capture required')
    ui = ui_files[0]
    app = read(folder / 'application.json')
    require(app['status'] == 'passed' and app['plan'] == plan and app['profiling_only'] is True
            and app['source_head'] == plan['source_head'] and app['official_binary_sha256'] == sha(OFFICIAL)
            and app['runner_sha256'] == sha(OLD / 'official_smoke.py')
            and app['experiment_runner_sha256'] == sha(OLD / 'experiment_runner.py')
            and len(app['rows']) == 1, 'Official profile application/identity differs')
    row = app['rows'][0]
    require(row['target_index'] == index and row['kid'] == target['kid'] and row['shape'] == target['shape']
            and row['profile_iterations'] == 11 and row['profile_rotation'] == 1
            and row['profile_label'] == 'official' and row['timings'] == [] and row['correctness'] == {}
            and row['actual_official_module'] == {'path': str(OFFICIAL), 'sha256': sha(OFFICIAL)},
            'Profile target/rotation/loaded module differs')
    target_cu = int(command['env']['ROCPROF_ATT_PARAM_TARGET_CU'])
    require(command['env']['ROCPROF_KERNEL_FILTER_INCLUDE_REGEX'] in kernel['demangled']
            and command['env']['ROCPROF_KERNEL_FILTER_RANGE'] == '6'
            and target_cu == (1 if index == 15 and command['name'].endswith('_recollect') else 0)
            and command['env']['ROCPROF_ATT_PARAM_SHADER_ENGINE_MASK'] == '1', 'Corrected filter/ATT scope differs')
    claim = clean_claim(command, app, claims)
    code, exact, tool = exact_code(folder, ui, command['name'], kernel)
    require(tool['metadata']['pid'] == claim['owner_identity']['inner_pid'], 'Profiler process differs from own handshake')
    geometry = parse_traits(target, kernel)
    expected = geometry['expected_MFMA_counts_for_full_wave']
    waves = sorted(ui.glob('se*_sm*_sl*_wv*.json'))
    require(waves, 'No stitched wave instruction files')
    aggregate, waits, barriers = defaultdict(lambda: [0, 0, 0]), defaultdict(list), defaultdict(list)
    gaps, gap_examples, records = defaultdict(list), defaultdict(list), []
    idle = 0
    overlap_clocks = 0
    for path in waves:
        obj = read(path)
        w, insts = obj['wave'], obj['wave']['instructions']
        require(w['cu'] == target_cu and obj['num_insts'] == obj['num_stitched'] == len(insts)
                and all(len(e) == 5 and e[3] >= e[2] >= 0 for e in insts)
                and all(a[0] <= b[0] and a[0] + a[3] <= b[0] + 4 for a, b in zip(insts, insts[1:])),
                'Partial wave or decoder overlap beyond one quantum')
        timeline_delta = sum(t[1] for t in w['timeline']) - (w['end'] - w['begin'])
        require(timeline_delta in [0, 4], 'Decoder timeline endpoints differ beyond one quantum')
        mfma_pos = [i for i, e in enumerate(insts) if kind(code[e[4]][0]) == 'scaled_MFMA']
        require(len(mfma_pos) in expected, f'Full-wave MFMA count differs: target{index}, {len(mfma_pos)}, {expected}')
        issues = [insts[p][0] + insts[p][2] for p in mfma_pos]
        overlaps = [(i, a[0] + a[3] - b[0]) for i, (a, b) in enumerate(zip(insts, insts[1:]))
                    if a[0] + a[3] > b[0]]
        overlap_clocks += sum(v for _, v in overlaps)
        idle += insts[0][0] - w['begin'] + sum(max(0, b[0] - a[0] - a[3]) for a, b in zip(insts, insts[1:]))
        for e in insts:
            a = aggregate[e[4]]
            a[0] += 1; a[1] += e[3]; a[2] += e[2]
            if kind(code[e[4]][0]) == 'waitcnt':
                waits[e[4]].append({**event(e, code, w['begin']), 'wave': path.name, 'simd': w['simd']})
        for line in {e[4] for e in insts if kind(code[e[4]][0]) == 'barrier'}:
            for ordinal, e in enumerate([e for e in insts if e[4] == line]):
                barriers[(line, ordinal)].append({'wave': path.name, 'attempt': e[0],
                                                   'release': e[0] + e[3], 'duration': e[3]})
        tiles, per_tile = len(mfma_pos) // geometry['MFMA_per_wave_K128_tile'], geometry['MFMA_per_wave_K128_tile']
        for ordinal, (p, q) in enumerate(zip(mfma_pos, mfma_pos[1:])):
            tile, mfma = divmod(ordinal, per_tile)
            group_edge = mfma == per_tile - 1
            phase = ('startup_tile' if tile == 0 else 'final_tile' if tile == tiles - 1
                     else 'penultimate_tile' if tile == tiles - 2 else 'interior_tile')
            name = phase + ('/tile_boundary' if group_edge else '/inside_tile')
            if geometry['cluster'] and group_edge:
                name += '/cluster_boundary' if (tile + 1) % geometry['cluster'] == 0 else '/within_cluster'
            begin, end = issues[ordinal], issues[ordinal + 1]
            b = breakdown(insts[p:q + 1], code, begin, end)
            gaps[name].append(end - begin)
            gap_examples[name].append({'wave': path.name, 'K128_tile_ordinal_within_wave_partition': tile,
                                      'MFMA_ordinal_within_tile': mfma, 'gap_clocks': end - begin,
                                      'begin_PC': hex(code[insts[p][4]][5]), 'end_PC': hex(code[insts[q][4]][5]),
                                      'event_interval_breakdown_clocks': b,
                                      'instructions': [event(e, code, begin) for e in insts[p:q + 1]]})
        deps = defaultdict(dict)
        for line, entries in w['waitcnt']:
            d = [{'pc': hex(code[e[0]][5]), 'instruction': code[e[0]][0], 'kind': kind(code[e[0]][0])} for e in entries]
            deps[line][json.dumps(d, sort_keys=True)] = d
        stores = [e for e in insts if kind(code[e[4]][0]) == 'global_output_store']
        post = breakdown(insts, code, issues[-1], w['end'])
        record = {'wave': path.name, 'cu': w['cu'], 'simd': w['simd'], 'slot': w['slot'],
                  'begin': w['begin'], 'end': w['end'], 'duration_clocks': w['end'] - w['begin'],
                  'scaled_MFMA_count': len(mfma_pos), 'K128_tiles_in_wave_partition': tiles,
                  'prologue_to_first_MFMA_clocks': issues[0] - w['begin'],
                  'first_to_last_MFMA_clocks': issues[-1] - issues[0],
                  'last_MFMA_to_wave_end_clocks': w['end'] - issues[-1],
                  'startup_breakdown_clocks': breakdown(insts, code, w['begin'], issues[0]),
                  'post_last_MFMA_breakdown_clocks': post,
                  'whole_wave_breakdown_clocks': breakdown(insts, code, w['begin'], w['end']),
                  'first_store_after_last_MFMA_clocks': (stores[0][0] + stores[0][2] - issues[-1]) if stores else None,
                  'timeline_sum_minus_wave_duration': timeline_delta,
                  'adjacent_decoder_overlap_clocks': sum(v for _, v in overlaps),
                  'adjacent_decoder_overlap_events': [{'prior': event(insts[i], code, w['begin']),
                                                       'next': event(insts[i + 1], code, w['begin']),
                                                       'overlap_clocks': n} for i, n in overlaps],
                  'wait_dependencies': [{'pc': hex(code[line][5]), 'instruction': code[line][0],
                                         'distinct_nonempty_dependency_sets': list(values.values())}
                                        for line, values in sorted(deps.items())],
                  'prologue_request_wait_publish_barrier_events': [event(e, code, w['begin']) for e in insts[:mfma_pos[0]]
                       if kind(code[e[4]][0]) in ['kernarg_SMEM_issue', 'matrix_async_VMEM_issue',
                           'matrix_register_VMEM_issue', 'scale_VMEM_issue', 'LDS_vector_write', 'LDS_byte_write',
                           'waitcnt', 'barrier']],
                  'evidence': ref(path)}
        records.append(record)
    require(all(aggregate[row[2]] == row[6:9] for row in code.values()), 'Wave aggregate hits/duration/stall differ from code stats')
    require(sum(row[9] for row in code.values()) == idle, 'Code idle differs from wave lead/interevent gaps')
    wait_rows = [{'pc': hex(code[line][5]), 'instruction': code[line][0],
                  'event_duration_clocks': describe([v['duration'] for v in vals]),
                  'event_duration_sum_clocks': sum(v['duration'] for v in vals),
                  'stall_sum_clocks': sum(v['stall'] for v in vals), 'events': vals}
                 for line, vals in sorted(waits.items(), key=lambda item: -sum(v['duration'] for v in item[1]))]
    multiple_WGs = len(records) > geometry['waves_per_WG']
    barrier_rows = [{'pc': hex(code[line][5]), 'occurrence': ordinal, 'captured_waves': len(vals),
                     'aggregation_scope': 'cross_WG_same_PC_occurrence' if multiple_WGs else 'one_captured_wave_cohort',
                     'valid_for_intra_WG_arrival_imbalance': not multiple_WGs,
                     'attempt_spread_clocks': max(v['attempt'] for v in vals) - min(v['attempt'] for v in vals),
                     'release_spread_clocks': max(v['release'] for v in vals) - min(v['release'] for v in vals),
                     'duration_clocks': describe([v['duration'] for v in vals]), 'waves': vals}
                    for (line, ordinal), vals in sorted(barriers.items())]
    examples = {name: min(vals, key=lambda v: abs(v['gap_clocks'] - statistics.median(gaps[name])))
                for name, vals in gap_examples.items()}
    report = {'target_index': index, 'parent_id': target['kid'], 'shape': target['shape'],
              'symbol': kernel['name'], 'demangled': kernel['demangled'], 'geometry': geometry,
              'capture': str(folder), 'ui': str(ui), 'captured_complete_waves': len(records),
              'target_CU': target_cu, 'multiple_workgroups_captured': multiple_WGs,
              'GPU': app['gpu'], 'clean_claim': claim,
              'identity': {**exact, 'loaded_official_module': row['actual_official_module'],
                           'all_wave_code_stats_match': True, 'all_code_idle_matches': True,
                           'adjacent_decoder_overlap_clocks': overlap_clocks,
                           'overlap_accounting': 'At most one4-clock quantum per adjacent event; raw stats retained, interval breakdown counts shared interval once.'},
              'summary': {field: describe([r[field] for r in records]) for field in
                          ['duration_clocks', 'prologue_to_first_MFMA_clocks', 'first_to_last_MFMA_clocks',
                           'last_MFMA_to_wave_end_clocks']},
              'phase_mean_share_percent': {field: statistics.mean(100 * r[field] / r['duration_clocks'] for r in records)
                                          for field in ['prologue_to_first_MFMA_clocks', 'first_to_last_MFMA_clocks',
                                                        'last_MFMA_to_wave_end_clocks']},
              'waitcnt_by_PC': wait_rows, 'barrier_occurrences': barrier_rows,
              'MFMA_gap_groups': {name: describe(vals) for name, vals in sorted(gaps.items())},
              'representative_MFMA_windows': examples, 'waves': records,
              'evidence': {label: ref(path) for label, path in {
                  'application': folder / 'application.json', 'results': folder / f'{command["name"]}_results.json',
                  'kernel_trace': folder / f'{command["name"]}_kernel_trace.csv', 'rocprof_log': Path(command['log'])}.items()},
              'performance_measured': False,
              'limits': [f'One SE0/CU{target_cu} capture and correlated waves/tiles; no full-GPU percentage gain.',
                         ('Multiple workgroups are captured. Same-PC/occurrence barrier spans aggregate different workgroups and are excluded from intra-WG imbalance conclusions.'
                          if multiple_WGs else 'Barrier rows correlate one finite captured wave cohort; no whole-GPU imbalance claim.'),
                         'Opcode labels distinguish vector/byte requests but do not identify one memory return timestamp.',
                         'Waitcnt nonempty dependency sets establish queue membership; empty dynamic waits can be absent.',
                         'Group-boundary classification is ordinal/source geometry; exact PC windows retained for verification.',
                         'Profiling application does not run independent reference/guard checks; correctness is separate.']}
    return report


def original_null_exclusion(queue, plan, kernels, claims):
    commands = [c for c in queue['commands'] if c['name'] == 'target15_kid9055_att']
    require(len(commands) == 1, 'Original9055 command missing')
    command = commands[0]
    folder = Path(command['argv'][command['argv'].index('--output') + 1]).parent
    ui = next(folder.glob('ui_output_*'))
    code = read(ui / 'code.json')
    require(code['code'] is None and not list(ui.glob('se*_sm*_sl*_wv*.json')), 'Original9055 exclusion state changed')
    app = read(folder / 'application.json')
    require(app['status'] == 'passed' and app['plan'] == plan, 'Original9055 application failed')
    return {'target_index': 15, 'reason': 'CU0 decoder code=null and zero stitched wave files',
            'application_passed': True, 'usable_for_instruction_analysis': False,
            'application': ref(folder / 'application.json'), 'code_json': ref(ui / 'code.json'),
            'clean_claim': clean_claim(command, app, claims),
            'replacement': 'Separate CU1 recollection, independently checked against its own queue/claim/owner/device.'}


def mechanism_findings(captures):
    findings = []
    for c in captures:
        index = c['target_index']
        item = {'target_index': index, 'parent_id': c['parent_id'], 'performance_measured': False,
                'complete_call_Event_required_for_adoption': True}
        if index == 17:
            first = min(c['barrier_occurrences'], key=lambda b: min(w['attempt'] for w in b['waves']))
            prologues = []
            for w in c['waves']:
                events = w['prologue_request_wait_publish_barrier_events']
                matrix = [e for e in events if e['kind'] == 'matrix_async_VMEM_issue']
                publications = [e for e in events if e['kind'] in ['LDS_vector_write', 'LDS_byte_write']]
                prologues.append({'wave': w['wave'], 'first_initial_matrix_issue_clocks': matrix[0]['time'],
                                  'scale_LDS_publication_events': publications,
                                  'initial_barrier_event': next(e for e in events if e['kind'] == 'barrier')})
            item.update({'status': 'isolated_exact_fixed9062_candidate_CPU_build_prepared',
                         'evidence': {'first_publication_barrier': first, 'wave_initial_handoff': prologues},
                         'observed_mechanism': 'One producer wave executes a second SFA pass and SFB publish before any initial matrix request; other waves arrive at the first publication barrier earlier.',
                         'candidate': {'directory': str(HERE / 'fine_startup'),
                                       'scope': c['demangled'], 'single_change': 'Move the existing three initial matrix tiles before the unchanged scale load/publish body.',
                                       'preserved': ['All A/B/SFA/SFB byte requests and guards', '4-stage ring, all tile retirement and consumer code',
                                                     'Runtime waits/barriers, split2 producer and matching reducer'],
                                       'risk_to_resolve': 'Compiler generated scale-dependent vmcnt waits include earlier matrix requests; early full-drain and changed register allocation may negate overlap.'}})
            decision_path = HERE / 'fine_startup/results_analysis.json'
            if decision_path.exists():
                decision = read(decision_path)
                require(decision['status'] == 'rejected_no_positive_complete_Event_gain'
                        and decision['adopted'] is False, 'Finite9062 decision state differs')
                item.update(status='finite_fixed9062_initial_handoff_rejected_by_complete_Event',
                            complete_call_decision=ref(decision_path),
                            decision_reason=decision['reason'],
                            winner_Event_decisions=[{k:r[k] for k in ['shape','median_us','median_speedup','faster_rounds']}
                                                    for r in decision['winner_decisions']])
                item['candidate']['adopted'] = False
        elif index == 14:
            item.update({'status': 'diagnosed_finite_ring_read_order_followup',
                         'observed_mechanism': 'Cluster boundaries include LDS reads, barrier and matrix issue time; first consumer LDS reads after group visibility exceed many within-group reads.',
                         'finite_followup': 'Exact9046 ring LDS-read/future-prefetch ordering only, after fixed9062 candidate; preserve cluster publication and slot retirement.'})
        elif index == 16:
            item.update({'status': 'keep_fixed9071_N48_after_own_ATT_history_and_numerical_review',
                         'observed_mechanism': 'Initial request/setup, partial queue waits and final wave-K reduction contribute finite gaps.',
                         'finite_followup': 'Finite series closed with currentN48/Q4/OUTPUT4 and falseBscaleReuse preserved. Own oldN32 completeEvent failed; reuse128%BN assertion prevents B-scale reuse. No new9071 candidate is built or measured; other9051 issue-order results are not9071 performance evidence.'})
        elif index == 20:
            item.update({'status': 'diagnosed_runtime9051_issue_capacity_followup',
                         'observed_mechanism': 'Many captured wave prologues include long matrix issue and scale issue/wait events; historical exact9051 LFIFO/translation counter evidence remains reusable.',
                         'finite_followup': 'Runtime9051 matrix/scale request order or pacing, preserving request counts and geometry; keep global reuse rejected.',
                         'barrier_exclusion': '32 waves span multiple workgroups. Aggregate barrier spans cannot measure intra-WG imbalance.'})
            register_path = HERE / 'register_issue_order/isa_review.json'
            if register_path.exists():
                register = read(register_path)
                require(register['status'] == 'passed_exact9051_issue_order_ISA_pending_GPU', 'Finite9051 ISA state differs')
                item.update(status='finite_runtime9051_scale_first_CPU_exact_passed_pending_GPU',
                            isolated_register_candidate={'directory':str(HERE/'register_issue_order'),
                                'ISA_review':ref(register_path), '39_other_device_entries_unchanged':True,
                                'same_request_compute_LDS_store_counts':True, 'adopted':False,
                                'performance_measured':False})
                decision_path = HERE / 'register_issue_order/results_analysis.json'
                if decision_path.exists():
                    decision = read(decision_path)
                    require(decision['status']=='positive_two_representative_Event_requires17winner_coverage'
                            and decision['adopted'] is False, 'Finite9051 positive representative state differs')
                    item.update(status='finite_runtime9051_scale_first_two_positive_Event_full17scope_pending',
                        complete_call_decision=ref(decision_path),decision_reason=decision['reason'],
                        representative_Event_decisions=[{k:r[k] for k in ['shape','median_us','median_speedup','faster_rounds']}
                                                       for r in decision['winner_decisions']])
                    item['isolated_register_candidate'].update(performance_measured=True,
                        adopted=False,coverage_Event_plan=ref(HERE/'register_issue_order/coverage_event_plan.json'))
                coverage_path = HERE / 'register_issue_order/coverage_analysis.json'
                if coverage_path.exists():
                    coverage = read(coverage_path)
                    require(coverage['audit_status'] == 'passed_CPU_full17_exact_identity_claim_correctness_Event_review'
                            and coverage['status'] == 'rejected_runtime9051_full17_mixed_or_weak_Event'
                            and coverage['adopted'] is False, 'Finite9051 full17 state differs')
                    item.update(status='finite_runtime9051_scale_first_rejected_by_full17_complete_Event',
                        complete_call_decision=ref(coverage_path), decision_reason=coverage['reason'],
                        full17_Event_summary={k:coverage[k] for k in ['full_actual_winner_count',
                            'median_positive_winners','median_regressed_winners','all5rounds_faster_winners',
                            'both_order_groups_positive_winners','uniform_exacttype_decision']},
                        finite_followup=coverage['retry_policy'])
                    item['isolated_register_candidate'].update(performance_measured=True, adopted=False,
                        decision='reject_uniform_exacttype_keep_current', full17coverage=ref(coverage_path))
        elif index in [18, 19]:
            item.update({'status': 'keep_shortK_current_after_own_ATT_and_numerical_review',
                         'observed_mechanism': 'ShortK prologue includes long kernarg waits and substantial setup relative to six MFMA issues per complete wave.',
                         'finite_followup': 'Finite series closed with currentselected preserved. Own shortK capture does not justify queue-depth or scale-reuse changes; no shortK candidate performance was measured this round.'})
        else:
            first = min(c['barrier_occurrences'], key=lambda b: min(w['attempt'] for w in b['waves']))
            prologues = []
            for w in c['waves']:
                events = w['prologue_request_wait_publish_barrier_events']
                prologues.append({'wave': w['wave'],
                    'first_matrix_issue_clocks': next(e['time'] for e in events if e['kind'] == 'matrix_async_VMEM_issue'),
                    'scale_publish_events': [e for e in events if e['kind'] in ['matrix_register_VMEM_issue', 'scale_VMEM_issue', 'LDS_vector_write', 'LDS_byte_write']],
                    'long_startup_waits': [e for e in events if e['kind'] == 'waitcnt' and e['duration'] >= 100]})
            item.update({'status': 'valid_CU1_capture_scale_publication_imbalance_diagnosed',
                         'evidence': {'first_publication_barrier': first, 'wave_initial_handoff': prologues},
                         'observed_mechanism': 'Two waves publish SFA, one also publishes SFB before matrix issue. The SFB producer reaches the first barrier2300 clocks after the earliest wave; its SFA/SFB vmcnt(0) take892/944clocks in this capture.',
                         'finite_followup': 'Exact9055 initial matrix/scale handoff is a separate followup after fixed9062 Event, preservingS12/C4 group visibility and ring retirement. Do not extend the current9062 type predicate without independent build/correctness/fullEvent.'})
        findings.append(item)
        if index in [14,15]:
            ring_path = HERE / 'ring_overlap/isa_review.json'
            if ring_path.exists():
                ring = read(ring_path)
                require(ring['status'] == 'passed_exact_ring_scope_ISA_pending_GPU', 'Finite ring ISA review state differs')
                item.update(status='finite_ring_issue_read_order_CPU_exact_passed_pending_GPU',
                            isolated_ring_candidate={'directory':str(HERE/'ring_overlap'),
                                'ISA_review':ref(ring_path),
                                'source_diff':ref(HERE/'ring_overlap/candidate.diff'),
                                'same_request_read_write_MFMA_counts':True,
                                '38_other_device_entries_unchanged':True,
                                'adopted':False,
                                'performance_measured':False})
                decision_path = HERE / 'ring_overlap/results_analysis.json'
                if decision_path.exists():
                    decision = read(decision_path)
                    require(decision['status'] == 'rejected_both_ring_types_no_positive_complete_Event_gain'
                            and decision['adopted'] is False, 'Finite ring decision state differs')
                    selected = [r for r in decision['winner_decisions'] if r['shape'] == c['shape']]
                    require(len(selected)==1, 'Finite ring representative decision missing')
                    item.update(status='finite_ring_issue_read_order_rejected_by_complete_Event',
                        complete_call_decision=ref(decision_path), decision_reason=decision['reason'],
                        representative_Event_decision={k:selected[0][k] for k in ['shape','median_us','median_speedup','faster_rounds']})
                    item['isolated_ring_candidate'].update(performance_measured=True,
                                                          decision='rejected_for_this_exact_type')
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', type=int, default=list(TARGETS))
    parser.add_argument('--output', type=Path, default=HERE / 'small_att_analysis.json')
    args = parser.parse_args()
    require(set(args.targets) <= set(TARGETS), 'Small-family targets must be14..20')
    plan, queue, formal = read(PLAN), read(QUEUE), read(FORMAL)
    require(plan['official_binary'] == str(OFFICIAL) and plan['official_binary_sha256'] == sha(OFFICIAL)
            and formal['candidate_binary_sha256'] == sha(OFFICIAL), 'Selected official binary identity differs')
    correctness = read(CORRECTNESS)
    require(correctness['status'] == 'passed' and correctness['plan'] == plan
            and correctness['profiling_only'] is False
            and correctness['official_binary_sha256'] == sha(OFFICIAL)
            and correctness['runner_sha256'] == sha(OLD / 'official_smoke.py')
            and correctness['experiment_runner_sha256'] == sha(OLD / 'experiment_runner.py'),
            'Separate signed-repeat correctness gate differs')
    correctness_commands = [c for c in queue['commands'] if '--check-only' in c['argv']]
    require(len(correctness_commands) == 1, 'One complete-cases correctness command required')
    correctness_claims = [json.loads(line) for line in CLAIM.read_text().splitlines() if line.strip()]
    correctness_epoch = clean_claim(correctness_commands[0], correctness, correctness_claims)
    correctness_rows = {r['target_index']: r for r in correctness['rows']}
    require(set(correctness_rows) == set(range(len(plan['targets']))), 'Separate numerical gate target coverage differs')
    for index in args.targets:
        row = correctness_rows[index]
        require(row['kid'] == plan['targets'][index]['kid'] and row['shape'] == plan['targets'][index]['shape']
                and row['timings'] == [] and row['actual_official_module'] == {'path': str(OFFICIAL), 'sha256': sha(OFFICIAL)},
                'Separate numerical row/loaded module differs')
        check = row['correctness']['official']
        require(check['repetitions'] == 2 and check['repeatable'] is True and check['errRatio'] == 0
                and check['output_guards'] is True and check['workspace_guards'] is True,
                'Separate numerical/repeatability/guards gate incomplete')
    claims = [json.loads(line) for line in CLAIM.read_text().splitlines() if line.strip()]
    kernels = {v['candidate']['name']: v['candidate'] for p in formal['parents'] for v in p['variants']}
    captures = []
    exclusions = []
    recollect_queue = read(RECOLLECT_QUEUE) if RECOLLECT_QUEUE.exists() else None
    recollect_claims = [json.loads(line) for line in RECOLLECT_CLAIM.read_text().splitlines() if line.strip()] if RECOLLECT_CLAIM.exists() else None
    for index in args.targets:
        commands = [c for c in queue['commands'] if '--target-index' in c['argv']
                    and c['argv'][c['argv'].index('--target-index') + 1] == str(index)]
        require(len(commands) == 1, f'One corrected command for target{index} required')
        target = plan['targets'][index]
        require(target['symbol'] in kernels and target['instruction_sha256'] == kernels[target['symbol']]['instruction_sha256'],
                'Plan exact selected FUNC differs')
        current_claims = claims
        if index == 15 and recollect_queue and recollect_claims:
            exclusions.append(original_null_exclusion(queue, plan, kernels, claims))
            commands = [c for c in recollect_queue['commands'] if c['name'] == 'target15_kid9055_att_recollect']
            require(len(commands) == 1, 'One CU1 recollection9055 command required')
            current_claims = recollect_claims
        captures.append(capture(index, commands[0], target, plan, current_claims, kernels[target['symbol']]))
    require('torch' not in sys.modules, 'CPU ATT analyzer imported torch')
    report = {'status': 'passed_cpu_exact_small_family_ATT_analysis',
              'generated_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
              'new_GPU_execution': False, 'build_executed': False, 'kernel_sources_modified': False,
              'analyzed_target_indices': args.targets, 'capture_count': len(captures),
              'official_module': ref(OFFICIAL), 'formal_identity_audit': ref(FORMAL),
              'plan': ref(PLAN), 'corrected_queue': ref(QUEUE), 'corrected_claim_log': ref(CLAIM),
              'review': {'path': str(REVIEW), 'scope': 'Mutable33config processing registry; exact kernel identities come from sealedformal identity and diagnostic plan.'},
              'analysis_script': ref(Path(__file__)),
              'separate_correctness_gate': {'result': ref(CORRECTNESS), 'claim_log': ref(CLAIM),
                                           'repetitions_per_target': 2, 'checked_small_targets': args.targets,
                                           'clean_claim': correctness_epoch,
                                           'all_signed_reference_repeatability_guards_passed': True},
              'timing_contract': 'Decoder event=[time,type,stall,duration,code_line]; successful MFMA issue=time+stall; duration includes stall. All times shader clocks. No counter-cycle to ns conversion.',
              'captures': captures,
              'excluded_captures': exclusions, 'mechanism_findings': mechanism_findings(captures),
              'recollect_queue': ref(RECOLLECT_QUEUE) if exclusions else None,
              'recollect_claim_log': ref(RECOLLECT_CLAIM) if exclusions else None,
              'limits': ['Finite actual-config representatives; no complete18-parent/33-config performance claim.',
                         'Current numerical correctness and complete-call Event decisions are separate from ATT.',
                         'Rejected global reuse/fine_wait/N64/N32 changes remain rejected; these traces diagnose selected baseline.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'targets': args.targets, 'output': str(args.output),
                      'captures': [{'target_index': c['target_index'], 'summary': c['summary'],
                                    'phase_mean_share_percent': c['phase_mean_share_percent']} for c in captures]}))


if __name__ == '__main__':
    main()
