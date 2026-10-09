#!/usr/bin/env python3
"""CPU-only audit of rocprofv3 JSON raw GRBM records and CSV reductions.

The long CSV can collapse raw counter instances by summation. The JSON callback
records retain values but this SDK may omit instance IDs. Never assign XCC IDs
from list order; preserve the catalogue and every raw record separately.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def handle(value):
    return value.get('handle') if isinstance(value, dict) else value


def summary(values):
    return {'minimum': min(values), 'median': statistics.median(values),
            'maximum': max(values)}


def canonical(name):
    for prefix in ('OPUS_GFX950_', 'OCT8_'):
        if name.startswith(prefix):
            return name[len(prefix):].rsplit('_', 1)[0]
    return name


def claim_for(log, name):
    epochs = []
    active = None
    for row in log:
        if row['event'] == 'start':
            if row['command']['name'] == name:
                if active is not None:
                    raise ValueError(f'{name}: overlapping claim starts')
                active = {'start': row['time'], 'fingerprint': row['fingerprint'],
                          'owners': [], 'monitors': []}
        elif active and row['event'] == 'owner_identity' and row['name'] == name:
            active['owners'].append(row)
        elif active and row['event'] == 'monitor':
            active['monitors'].append(row)
        elif active and row['event'] == 'end' and row['name'] == name:
            active['end'] = row['time']
            active['returncode'] = row['returncode']
            active['contamination'] = row['contamination']
            epochs.append(active)
            active = None
    if not epochs:
        raise ValueError(f'{name}: no completed claim')
    return epochs


def parse_pass(root, command, log, last):
    name = command['name']
    directory = root / name
    paths = {'application': directory / 'application.json',
             'csv': directory / f'{name}_counter_collection.csv',
             'sdk_json': directory / f'{name}_results.json'}
    app = json.loads(paths['application'].read_text())
    if app['status'] != 'passed' or not app['profiling_only']:
        raise ValueError(f'{name}: not a passed profiling-only application')
    claims = claim_for(log, name)
    accepted = [c for c in claims if c['start'] <= app['started']
                < app['finished'] <= c['end']]
    if len(accepted) != 1:
        raise ValueError(f'{name}: ambiguous application/claim epoch')
    claim = accepted[0]
    if claim['returncode'] != 0 or claim['contamination'] or len(claim['owners']) != 1:
        raise ValueError(f'{name}: dirty claim or unresolved owner')
    owner = claim['owners'][0]
    if owner['new_host_pids'] != [owner['host_pid']]:
        raise ValueError(f'{name}: invalid host identity')
    if any(m['child_pid'] != owner['inner_pid'] or m['processes'] for m in claim['monitors']):
        raise ValueError(f'{name}: dirty monitor epoch')
    sdk_sessions = json.loads(paths['sdk_json'].read_text())['rocprofiler-sdk-tool']
    if len(sdk_sessions) != 1:
        raise ValueError(f'{name}: expected one SDK process session')
    sdk = sdk_sessions[0]
    if sdk['metadata']['pid'] != owner['inner_pid']:
        raise ValueError(f'{name}: SDK and claim process mismatch')
    catalogue = defaultdict(list)
    for counter in sdk['counters']:
        catalogue[handle(counter['id'])].append(counter)
    by_kernel = {k['kernel_id']: k for k in sdk['kernel_symbols']}
    agents = {handle(a['id']): a for a in sdk['agents']}
    callbacks = [r for r in sdk['callback_records']['counter_collection']
                 if 'gemm_a8w8_mxfp8_scale_4wave_128x128_kernel' in
                 by_kernel[r['dispatch_data']['dispatch_info']['kernel_id']]['kernel_name']]
    callbacks.sort(key=lambda r: (r['dispatch_data']['start_timestamp'],
                                  r['dispatch_data']['dispatch_info']['dispatch_id']))
    if len(callbacks) < last:
        raise ValueError(f'{name}: fewer than {last} complete dispatches')
    selected = callbacks[-last:]
    csv_rows = defaultdict(list)
    with paths['csv'].open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for line, row in enumerate(reader, 2):
            csv_rows[(int(row['Dispatch_Id']), int(row['Kernel_Id']))].append((line, row))
    samples = []
    used_catalogue = {}
    target = app['plan']['targets'][app['target_index']]
    m, n, k = target['shape']
    expected_waves = math.ceil(m / 128) * math.ceil(n / 128) * 4
    expected_f8_counter = 2 * math.ceil(m / 128) * 128 * math.ceil(n / 128) * 128 * k // 512
    for callback in selected:
        data = callback['dispatch_data']
        info = data['dispatch_info']
        start, end = data['start_timestamp'], data['end_timestamp']
        if end <= start:
            raise ValueError(f'{name}: invalid dispatch nanosecond interval')
        rows = csv_rows[(info['dispatch_id'], info['kernel_id'])]
        if any(int(row['Start_Timestamp']) != start or int(row['End_Timestamp']) != end
               for _, row in rows):
            raise ValueError(f'{name}: CSV/JSON timestamp mismatch')
        grouped = defaultdict(list)
        for ordinal, record in enumerate(callback['records']):
            counter_id = handle(record['counter_id'])
            candidates = catalogue[counter_id]
            names = {c['name'] for c in candidates}
            if len(names) != 1:
                raise ValueError(f'{name}: ambiguous counter definition {counter_id}')
            counter_name = next(iter(names))
            used_catalogue[str(counter_id)] = candidates
            value = record['value']
            if not math.isfinite(value):
                raise ValueError(f'{name}: nonfinite raw counter')
            explicit = {key: val for key, val in record.items()
                        if key not in ('counter_id', 'value')}
            coordinate_fields = {key: val for key, val in explicit.items()
                                 if 'dimension' in key.lower() or 'instance' in key.lower()}
            grouped[counter_name].append({'ordinal_in_callback': ordinal,
                                         'value': value, 'raw_record': record,
                                         'explicit_instance_fields': explicit,
                                         'coordinate_fields': coordinate_fields,
                                         'coordinate_mapping': 'explicit_fields' if coordinate_fields else 'unknown'})
        clock = {}
        for counter_name in ('GRBM_COUNT', 'GRBM_GUI_ACTIVE'):
            clock_names = [c for c in grouped if canonical(c) == counter_name]
            if len(clock_names) != 1:
                raise ValueError(f'{name}: missing or ambiguous {counter_name}')
            collected_name = clock_names[0]
            records = grouped[collected_name]
            values = [r['value'] for r in records]
            matching_csv = [(line, row) for line, row in rows if row['Counter_Name'] == collected_name]
            if len(matching_csv) != 1:
                raise ValueError(f'{name}: expected one flattened CSV {counter_name} row')
            line, row = matching_csv[0]
            value_csv = float(row['Counter_Value'])
            clock[counter_name] = {'collected_name': collected_name,
                                   'raw_instance_values_available': collected_name == counter_name,
                                   'records': records, 'record_count': len(values),
                                   'sum': sum(values), 'minimum': min(values), 'maximum': max(values),
                                   'csv_line': line, 'csv_value': value_csv,
                                   'csv_equals_json_sum': value_csv == sum(values),
                                   'csv_equals_json_maximum': value_csv == max(values),
                                   'maximum_divided_by_duration_GHz': max(values) / (end - start),
                                   'csv_divided_by_duration_GHz': value_csv / (end - start)}
        scalars = {canonical(c): sum(r['value'] for r in records)
                   for c, records in grouped.items() if canonical(c) not in clock}
        if 'SQ_WAVES' in scalars and scalars['SQ_WAVES'] != expected_waves:
            raise ValueError(f'{name}: unexpected issued waves')
        if ('SQ_INSTS_VALU_MFMA_MOPS_F8' in scalars and
                scalars['SQ_INSTS_VALU_MFMA_MOPS_F8'] != expected_f8_counter):
            raise ValueError(f'{name}: unexpected padded F8 operation count')
        samples.append({'dispatch_id': info['dispatch_id'], 'kernel_id': info['kernel_id'],
                        'agent_id': handle(info['agent_id']), 'queue_id': handle(info['queue_id']),
                        'correlation_id': data['correlation_id'], 'start_ns': start, 'end_ns': end,
                        'duration_ns': end - start, 'clock': clock,
                        'non_clock_counter_sums': scalars})
    agent_ids = {s['agent_id'] for s in samples}
    if len(agent_ids) != 1:
        raise ValueError(f'{name}: multiple target agents')
    agent = agents[next(iter(agent_ids))]
    bdf = f"{agent['domain']:04x}:{agent['location_id'] >> 8:02x}:{(agent['location_id'] >> 3) & 31:02x}.{agent['location_id'] & 7}"
    if bdf != app['gpu']['pci_bdf'] or bdf != app['gpu']['expected_pci_bdf']:
        raise ValueError(f'{name}: physical GPU identity mismatch')
    clock_summaries = {}
    for counter_name in ('GRBM_COUNT', 'GRBM_GUI_ACTIVE'):
        entries = [s['clock'][counter_name] for s in samples]
        clock_summaries[counter_name] = {
            'collected_names': sorted({e['collected_name'] for e in entries}),
            'raw_instance_values_available': all(e['raw_instance_values_available'] for e in entries),
            'record_counts': sorted({e['record_count'] for e in entries}),
            'all_csv_equal_json_sum': all(e['csv_equals_json_sum'] for e in entries),
            'all_csv_equal_json_maximum': all(e['csv_equals_json_maximum'] for e in entries),
            'json_maximum': summary([e['maximum'] for e in entries]),
            'csv_sum': summary([e['csv_value'] for e in entries]),
            'maximum_over_timestamp_GHz': summary([e['maximum_divided_by_duration_GHz'] for e in entries]),
            'csv_sum_over_timestamp_GHz': summary([e['csv_divided_by_duration_GHz'] for e in entries]),
            'above_max_engine_clock_dispatches': sum(e['maximum_divided_by_duration_GHz'] >
                agent['max_engine_clk_fcompute'] / 1000 for e in entries)}
    return {'name': name, 'group': command['group'], 'shape': target['shape'],
            'sources': {key: {'path': str(path.resolve()), 'sha256': sha(path)} for key, path in paths.items()},
            'application_plan_sha256': app['plan_sha256'], 'runner_sha256': app['runner_sha256'],
            'libraries': app['libraries'], 'claim': claim,
            'agent': {key: agent[key] for key in ('node_id', 'logical_node_id', 'num_xcc', 'cu_count',
                                                'simd_count', 'max_engine_clk_fcompute')},
            'physical_pci_bdf': bdf, 'timestamp_unit': 'ns',
            'dispatches_before_selection': len(callbacks), 'selected_dispatches': len(samples),
            'csv_fields': fields, 'counter_catalogue': used_catalogue,
            'duration_ns': summary([s['duration_ns'] for s in samples]),
            'clock_summary': clock_summaries, 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--last', type=int, default=51)
    args = parser.parse_args()
    if args.last < 1:
        parser.error('--last must be positive')
    root = args.root.resolve()
    queue_path = root / 'queue.json'
    log_path = root / 'gpu_claim_log.jsonl'
    commands = json.loads(queue_path.read_text())['commands']
    log = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    collections = [parse_pass(root, command, log, args.last) for command in commands]
    raw = [c for c in collections if c['group'] == 'clock_raw']
    comparison = []
    for collection in sorted(raw, key=lambda c: c['shape'][2]):
        counter = collection['clock_summary']['GRBM_COUNT']
        comparison.append({'name': collection['name'], 'K': collection['shape'][2],
                           'median_duration_ns': collection['duration_ns']['median'],
                           'median_raw_maximum': counter['json_maximum']['median'],
                           'median_raw_maximum_over_timestamp_GHz': counter['maximum_over_timestamp_GHz']['median'],
                           'median_flattened_csv_sum_over_timestamp_GHz': counter['csv_sum_over_timestamp_GHz']['median'],
                           'above_max_engine_clock_dispatches': counter['above_max_engine_clock_dispatches']})
    result = {'status': 'complete_cpu_raw_clock_audit',
              'generated_utc': datetime.now(timezone.utc).isoformat(), 'gpu_access': False,
              'queue_sha256': sha(queue_path), 'claim_log_sha256': sha(log_path),
              'selected_last_dispatches_per_independent_pass': args.last,
              'completed_passes': len(collections), 'collections': collections,
              'short_long_clock_raw_comparison': comparison,
              'findings': [
                  'Raw GRBM CSV rows equal the sum of eight JSON callback values; CSV does not retain instance dimensions. Treating that scalar CSV as a max is incorrect.',
                  'SDK counter catalogue declares DIMENSION_INSTANCE=1 and DIMENSION_XCC=8. Callback records omit instance_id/dimension fields, so values are preserved by ordinal without assigning an XCC coordinate.',
                  'JSON raw maxima remove the demonstrated CSV aggregation problem but still exceed the recorded 2.4 GHz engine-clock limit, particularly for short K. The clock/window correspondence remains unresolved.',
                  'Short/long counter passes are independent source/process/dispatch sets. Their maxima, durations and implied rates are compared descriptively; no paired samples or fixed-factor correction are created.'
              ],
              'absolute_MFMA_utilization_and_occupancy_allowed': False,
              'limits': [
                  'No SDK packet or independently calibrated clock window proves that GRBM raw maxima and kernel timestamps cover the same interval.',
                  'The sum-to-max CSV issue is established for these raw-counter collections; earlier OPUS custom reduce(max) scalar results are different evidence and are not retroactively changed.',
                  'Raw record order does not prove which XCC produced a value. Catalogue coordinates describe possible instances, not a callback-to-instance mapping.',
                  'Profiler duration is instrumented. This clock audit contains no optimization speedup or bottleneck classification.'
              ]}
    output = root / 'clock_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'completed_passes': len(collections),
                      'short_long': comparison, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
