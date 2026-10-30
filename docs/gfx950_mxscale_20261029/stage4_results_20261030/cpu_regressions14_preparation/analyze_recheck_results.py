"""Validate paired Opus rechecks and compare with the fixed saved gfx950 CSV.

Uses only the Python standard library. No device access or timing is performed.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCOPE_FIELDS = ('gfx', 'cu_num', 'M', 'N', 'K', 'w_scale_block')
EXPECTED_SCOPE = ('gfx950', 256, '128x128')
BATCHES = (1, 2, 3)
ERR_LIMIT = 0.05
ROLES = {'historical_winner', 'current_winner'}


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not re.fullmatch(r'[0-9]+', str(value)):
        raise ValueError(f'{label}: expected an integer, got {value!r}')
    result = int(value)
    if result < minimum:
        raise ValueError(f'{label}: expected >= {minimum}, got {result}')
    return result


def number(value, label, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{label}: expected a finite number, got {value!r}')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label}: expected a finite number, got {value!r}') from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f'{label}: invalid value {value!r}')
    return result


def scope(row, label):
    try:
        result = (row['gfx'], integer(row['cu_num'], f'{label} cu_num', 1),
                  *(integer(row[name], f'{label} {name}', 1) for name in ('M', 'N', 'K')),
                  row['w_scale_block'])
    except KeyError as error:
        raise ValueError(f'{label}: missing field {error.args[0]}') from error
    if (result[0], result[1], result[5]) != EXPECTED_SCOPE:
        raise ValueError(f'{label}: scope must be gfx950/256/128x128, got {result}')
    return result


def scope_dict(shape):
    return dict(zip(SCOPE_FIELDS, shape))


def read_pairs(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    pairs, by_shape, baselines = {}, defaultdict(list), {}
    for line, row in enumerate(rows, 2):
        label = f'{path}:{line}'
        shape = scope(row, label)
        try:
            kid = integer(row['kernelId'], f'{label} kernelId')
            split = integer(row['splitK'], f'{label} splitK', 1)
            name, role = row['kernelName'], row['role']
            baseline = number(row['fixed_gfx950_csv_us'], f'{label} fixed CSV us', True)
        except KeyError as error:
            raise ValueError(f'{label}: missing field {error.args[0]}') from error
        if row.get('libtype') != 'opus' or not name.startswith('opus_gemm_'):
            raise ValueError(f'{label}: expected a complete Opus configuration')
        roles = role.split(';')
        if not roles or len(roles) != len(set(roles)) or not set(roles) <= ROLES:
            raise ValueError(f'{label}: invalid role {role!r}')
        pair_key = (shape, kid)
        if pair_key in pairs:
            raise ValueError(f'{label}: duplicate shape/kernelId pair {pair_key}')
        if shape in baselines and baselines[shape] != baseline:
            raise ValueError(f'{label}: inconsistent fixed CSV timing for {shape}')
        pairs[pair_key] = {**scope_dict(shape), 'libtype': 'opus', 'kernelId': kid,
                           'splitK': split, 'kernelName': name, 'role': role}
        by_shape[shape].append(pair_key)
        baselines[shape] = baseline
    if not pairs:
        raise ValueError('Paired input must contain at least one complete shape pair')
    for shape, keys in by_shape.items():
        if len(keys) not in (1, 2):
            raise ValueError(f'{shape}: expected one or two distinct configurations')
        roles = [role for key in keys for role in pairs[key]['role'].split(';')]
        if Counter(roles) != Counter(ROLES):
            raise ValueError(f'{shape}: expected exactly one historical and one current role')
    return pairs, by_shape, baselines


def read_results(paths, pairs):
    results = {}
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                label = f'{path}:{line_number}'
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{label}: invalid JSON: {error.msg}') from error
                if not isinstance(row, dict):
                    raise ValueError(f'{label}: expected a JSON object')
                shape = scope(row, label)
                try:
                    kid = integer(row['kernelId'], f'{label} kernelId')
                    batch = integer(row['batch'], f'{label} batch', 1)
                    split = integer(row['splitK'], f'{label} splitK', 1)
                    order = integer(row['order'], f'{label} order')
                    seed = integer(row['seed'], f'{label} seed')
                    status, latency, error_ratio = row['status'], row['us'], row['errRatio']
                except KeyError as error:
                    raise ValueError(f'{label}: missing field {error.args[0]}') from error
                if batch not in BATCHES:
                    raise ValueError(f'{label}: batch must be 1, 2, or 3')
                pair_key = (shape, kid)
                if pair_key not in pairs:
                    raise ValueError(f'{label}: extra or unrecognized pair {pair_key}')
                pair = pairs[pair_key]
                for field in ('libtype', 'kernelName', 'role'):
                    if row.get(field) != pair[field]:
                        raise ValueError(f'{label}: {field} does not match the paired input')
                if split != pair['splitK']:
                    raise ValueError(f'{label}: splitK does not match the paired input')
                result_key = (batch, pair_key)
                if result_key in results:
                    raise ValueError(f'{label}: duplicate batch/shape/kernelId result, including across files')
                if status not in ('ok', 'error'):
                    raise ValueError(f'{label}: status must be ok or error')
                latency = number(latency, f'{label} us', True) if latency is not None else None
                error_ratio = number(error_ratio, f'{label} errRatio') if error_ratio is not None else None
                if status == 'ok' and (latency is None or error_ratio is None):
                    raise ValueError(f'{label}: ok result requires positive us and numeric errRatio')
                valid = status == 'ok' and error_ratio <= ERR_LIMIT
                results[result_key] = {**pair, 'batch': batch, 'us': latency,
                                       'errRatio': error_ratio, 'status': status,
                                       'valid': valid, 'order': order, 'seed': seed}
    # A pair comparison is meaningful only when both configurations used the same seed.
    seeds = defaultdict(set)
    for (batch, (shape, _kid)), row in results.items():
        seeds[(batch, shape)].add(row['seed'])
    for batch_shape, values in seeds.items():
        if len(values) != 1:
            raise ValueError(f'{batch_shape}: paired configurations used different input seeds')
    return results


def timing_statistics(values):
    if not values:
        return {'median_us': None, 'min_us': None, 'max_us': None,
                'spread_us': None, 'spread_pct': None}
    median = statistics.median(values)
    spread = max(values) - min(values)
    return {'median_us': median, 'min_us': min(values), 'max_us': max(values),
            'spread_us': spread, 'spread_pct': 100 * spread / median}


def analyze(pairs, by_shape, baselines, results):
    pair_stats = []
    for pair_key, pair in pairs.items():
        batches = []
        valid_times = []
        for batch in BATCHES:
            row = results.get((batch, pair_key))
            if row is None:
                batches.append({'batch': batch, 'status': 'missing', 'valid': False,
                                'us': None, 'errRatio': None})
            else:
                batches.append({field: row[field] for field in (
                    'batch', 'status', 'valid', 'us', 'errRatio', 'order', 'seed',
                )})
                if row['valid']:
                    valid_times.append(row['us'])
        pair_stats.append({**pair, 'fixed_gfx950_csv_us': baselines[pair_key[0]],
                           'complete': len(valid_times) == len(BATCHES),
                           'valid_batches': len(valid_times),
                           **timing_statistics(valid_times), 'batches': batches})
    comparisons, batch_counts = [], {batch: Counter() for batch in BATCHES}
    for shape, keys in sorted(by_shape.items()):
        baseline = baselines[shape]
        record = {**scope_dict(shape), 'fixed_gfx950_csv_us': baseline,
                  'pair_count': len(keys)}
        classifications, best_times = [], []
        for batch in BATCHES:
            candidates = [results.get((batch, pair_key)) for pair_key in keys]
            complete = all(row is not None and row['valid'] for row in candidates)
            valid_candidates = [row for row in candidates if row is not None and row['valid']]
            best = min(valid_candidates, key=lambda row: (row['us'], row['kernelId'])) if valid_candidates else None
            classification = ('win' if best['us'] < baseline else 'loss') if complete else 'incomplete'
            classifications.append(classification)
            batch_counts[batch][classification] += 1
            # Partial best measurements are retained but never classify a shape as a loss.
            prefix = f'batch{batch}_'
            record.update({prefix + 'status': classification,
                           prefix + 'best_us': best['us'] if best else None,
                           prefix + 'best_kernelId': best['kernelId'] if best else None,
                           prefix + 'best_kernelName': best['kernelName'] if best else None,
                           prefix + 'reduction_vs_csv_pct': (
                               100 * (1 - best['us'] / baseline) if complete else None),
                           prefix + 'valid_pairs': len(valid_candidates)})
            if complete:
                best_times.append(best['us'])
        if 'incomplete' in classifications:
            classification = 'incomplete'
        elif all(value == 'win' for value in classifications):
            classification = 'consistent_win'
        elif all(value == 'loss' for value in classifications):
            classification = 'consistent_loss'
        else:
            classification = 'mixed'
        record['classification'] = classification
        stats = timing_statistics(best_times if classification != 'incomplete' else [])
        record.update({'median_batch_best_us': stats['median_us'],
                       'spread_batch_best_us': stats['spread_us'],
                       'spread_batch_best_pct': stats['spread_pct']})
        comparisons.append(record)
    counts = Counter(row['classification'] for row in comparisons)
    required = len(pairs) * len(BATCHES)
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'comparison_scope': 'fixed gfx950/256/128x128 CSV timings from paired input; no FlyDSL retiming',
        'rule': 'Every expected pair must be present and correct in all three batches. '
                'Each batch selects its fastest valid Opus pair. Win means us < fixed CSV us; ties are losses. '
                'Missing, failed, or incorrect measurements classify the shape as incomplete.',
        'correctness_errRatio_limit': ERR_LIMIT, 'expected_batches': list(BATCHES),
        'expected_shapes': len(by_shape), 'expected_pairs': len(pairs),
        'expected_result_rows': required, 'observed_result_rows': len(results),
        'missing_result_rows': required - len(results),
        'error_result_rows': sum(row['status'] == 'error' for row in results.values()),
        'incorrect_result_rows': sum(row['status'] == 'ok' and not row['valid'] for row in results.values()),
        'valid_result_rows': sum(row['valid'] for row in results.values()),
        'classification_counts': {name: counts[name] for name in (
            'consistent_win', 'consistent_loss', 'mixed', 'incomplete',
        )},
        'complete': counts['incomplete'] == 0,
        'per_batch': {str(batch): {'wins': batch_counts[batch]['win'],
                                  'losses': batch_counts[batch]['loss'],
                                  'incomplete': batch_counts[batch]['incomplete']}
                      for batch in BATCHES},
        'pair_statistics': pair_stats,
    }
    return comparisons, summary


def write_outputs(output_dir, comparisons, summary, results_paths, pairs_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary['results_files'] = [str(path.resolve()) for path in results_paths]
    summary['results_file'] = summary['results_files'][0] if len(results_paths) == 1 else None
    summary['pairs_file'] = str(pairs_path.resolve())
    comparison_path = output_dir / 'comparison.csv'
    with comparison_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(comparisons[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparisons)
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return comparison_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--pairs', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'regressions14_pair_candidates.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    try:
        pairs, by_shape, baselines = read_pairs(args.pairs)
        results = read_results(args.results, pairs)
        comparisons, summary = analyze(pairs, by_shape, baselines, results)
        csv_path, json_path = write_outputs(args.output_dir, comparisons, summary,
                                            args.results, args.pairs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({'complete': summary['complete'],
                      **summary['classification_counts'],
                      'comparison_csv': str(csv_path), 'summary_json': str(json_path)}))
    if args.require_complete and not summary['complete']:
        print('Recheck incomplete: missing, failed, or incorrect pair measurements; see summary.json.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
