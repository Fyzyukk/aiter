"""Standard-library-only validation of the frozen 14-shape recheck inputs."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
STAGE4 = BUNDLE.parent
ROOT = STAGE4.parent
REPO = Path('/home/fyz/aiter')
JIT_ROOT = STAGE4 / 'aiter_jit'
CO_ROOT = REPO / 'csrc/opus_gemm/gen_co/gfx950'
PAIRS = STAGE4 / 'regressions14_pair_candidates.csv'
FROZEN = BUNDLE / 'frozen_inputs.json'
GPU_UUIDS = {7: 'GPU-fc54ff82fc812ffd', 6: 'GPU-a8da4c5b961f1b75',
             5: 'GPU-70a3d8a095a5ac63', 4: 'GPU-43837dc1dbc12da4'}
GPU_BUSES = {7: 0xF5, 6: 0xE5, 5: 0x95, 4: 0x85}
PRIORITY = [(3072, 7168, 21504), (14336, 7168, 16384)]
SHAPE_FIELDS = ('gfx', 'cu_num', 'M', 'N', 'K', 'w_scale_block')
ROLES = {'historical_winner', 'current_winner'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(int(row[field]) for field in ('M', 'N', 'K'))


def finite_positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'Expected a finite positive value: {value!r}')
    return number


def co_symbol(document, entry):
    family = {'pin_agpr': '4wave_pin_agpr_co',
              'prefetch_b_vgpr': 'b_vgpr_co',
              'prefetch_b_vgpr_overlap': 'b_vgpr_overlap_co'}[entry['pipeline']]
    fields = ((entry['block_size'], *entry['tile']), entry['wave_layout'],
              (16, 16, 128), (1, 128, 128))
    name = '_'.join(('opus_gemm', 'gfx950', family)
                    + tuple('x'.join(map(str, values)) for values in fields))
    suffix = document['image_suffixes'][str(entry['kid'])]
    return name + (f'_{suffix}' if suffix else '')


def collect_inputs():
    expected_head = json.loads((ROOT / 'preparation.json').read_text())['git_head']
    actual_head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if actual_head != expected_head:
        raise ValueError(f'Git HEAD changed: expected {expected_head}, got {actual_head}')
    target_rows = csv_rows(ROOT / 'stage4_preparation/shapes_421.csv')
    target = {key(row) for row in target_rows}
    baseline_path = ROOT / 'csv_baseline.csv'
    baseline_rows = [row for row in csv_rows(baseline_path)
                     if row['gfx'] == 'gfx950' and int(row['cu_num']) == 256
                     and row['w_scale_block'] == '128x128' and key(row) in target]
    baseline = {key(row): row for row in baseline_rows}
    current_path = STAGE4 / 'full421_tuned.csv'
    history_path = REPO / 'docs/gfx950_mxscale_20261029/all_shapes_421.csv'
    current_rows, history_rows = csv_rows(current_path), csv_rows(history_path)
    current, history = ({key(row): row for row in rows}
                        for rows in (current_rows, history_rows))
    if (len(target_rows) != 421 or len(target) != 421
            or len(baseline_rows) != len(baseline) or len(baseline) != 421
            or len(current_rows) != len(current) or set(current) != target
            or len(history_rows) != len(history) or set(history) != target):
        raise ValueError('Expected unique, matching 421-shape target/baseline/history/results')
    regressions = {shape for shape in target
                   if finite_positive(history[shape]['opus_us'])
                   < finite_positive(baseline[shape]['us'])
                   <= finite_positive(current[shape]['us'])}
    pair_rows = csv_rows(PAIRS)
    groups, identities = defaultdict(list), set()
    co_path = CO_ROOT.parent / 'co_kernels_gfx950.json'
    document = json.loads(co_path.read_text())
    co_entries = {row['kid']: {**document['defaults'], **row}
                  for row in document['gfx950']}
    build_path = CO_ROOT / 'build_info.json'
    ready = {row['kid']: row for row in json.loads(build_path.read_text())['kernels']}
    sidecar = JIT_ROOT / 'build/compiled_kids_opus.json'
    receipt = sidecar.with_name(sidecar.name + '.receipt')
    compiled = set(json.loads(sidecar.read_text()))
    opus_so = JIT_ROOT / 'module_deepgemm_opus.so'
    stat = opus_so.stat()
    receipt_data = json.loads(receipt.read_text())
    if (receipt_data['sha256'] != digest(sidecar)
            or receipt_data['artifact'] != [stat.st_dev, stat.st_ino, stat.st_size,
                                            stat.st_mtime_ns, stat.st_ctime_ns]):
        raise ValueError('Compiled-kid sidecar/receipt does not bind to the current Opus SO')
    resource_rows = {}
    for row in pair_rows:
        shape, kid = key(row), int(row['kernelId'])
        if (row['gfx'] != 'gfx950' or int(row['cu_num']) != 256
                or row['w_scale_block'] != '128x128' or row['libtype'] != 'opus'
                or int(row['splitK']) != 1 or shape not in regressions):
            raise ValueError(f'Invalid paired configuration scope: {row}')
        identity = (shape, kid)
        if identity in identities:
            raise ValueError(f'Duplicate paired configuration: {identity}')
        identities.add(identity)
        roles = row['role'].split(';')
        if not roles or len(roles) != len(set(roles)) or not set(roles) <= ROLES:
            raise ValueError(f'Invalid role: {row["role"]}')
        if float(row['fixed_gfx950_csv_us']) != float(baseline[shape]['us']):
            raise ValueError(f'Fixed baseline reference differs for {shape}')
        if float(row['historical_opus_us']) != float(history[shape]['opus_us']):
            raise ValueError(f'Historical timing reference differs for {shape}')
        if 'historical_winner' in roles and row['kernelName'] != history[shape]['opus_kernel']:
            raise ValueError(f'Historical winner name differs for {shape}')
        if 'current_winner' in roles and any(row[field] != current[shape][field]
                for field in ('kernelId', 'splitK', 'kernelName')):
            raise ValueError(f'Current winner identity differs for {shape}')
        entry = co_entries[kid]
        symbol = co_symbol(document, entry)
        if row['kernelName'] != symbol or ready[kid]['status'] != 'ready':
            raise ValueError(f'CO name/readiness mismatch for {identity}')
        if ready[kid]['symbol'] != symbol or ready[kid]['image'] != symbol + '.co':
            raise ValueError(f'CO build metadata mismatch for {identity}')
        if f'a8w8_blockscale_bpreshuffle/gfx950/256/{kid}' not in compiled:
            raise ValueError(f'Pair {identity} is absent from compiled membership')
        image = CO_ROOT / (symbol + '.co')
        with image.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError(f'CO image is not ELF: {image}')
        m, n, k = shape
        bm, bn, bk = entry['tile']
        if (m % entry.get('m_align', 16) or n % bn or k % bk
                or (entry.get('fixed_k', 0) and k != entry['fixed_k'])
                or (entry.get('max_k') is not None and k > entry['max_k'])
                or max(m * k, n * k) > 2**31 - 1
                or 2 * m * n > 2**31 - 1
                or 2 * ((min(m, bm) - 1) * n + bn) > 2**31 - 1):
            raise ValueError(f'Alignment, fixed K, or buffer addressing limit fails: {identity}')
        groups[shape].append(row)
        resource_rows[kid] = {field: ready[kid].get(field) for field in (
            'vgpr_count', 'sgpr_count', 'agpr_count', 'group_segment_fixed_size',
            'private_segment_fixed_size', 'vgpr_spill_count', 'sgpr_spill_count')}
    if len(regressions) != 14 or set(groups) != regressions or len(pair_rows) != 26:
        raise ValueError('The recheck must cover exactly the 14 regressions and 26 pairs')
    for shape, rows in groups.items():
        if len(rows) not in (1, 2) or Counter(
                role for row in rows for role in row['role'].split(';')) != Counter(ROLES):
            raise ValueError(f'Missing or duplicated historical/current role: {shape}')
        rows.sort(key=lambda row: ('historical_winner' not in row['role'], int(row['kernelId'])))
    sources = [
        REPO / 'csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py',
        REPO / 'aiter/test_common.py', REPO / 'aiter/utility/mp_tuner.py',
        REPO / 'aiter/jit/core.py', REPO / 'csrc/opus_gemm/opus_gemm_common.py',
        REPO / 'csrc/opus_gemm/include/gfx950/opus_co_launch_gfx950.cuh',
    ]
    files = [PAIRS, baseline_path, current_path, history_path, co_path, build_path,
             sidecar, receipt, opus_so, JIT_ROOT / 'module_aiter_core.so', *sources]
    for kid in resource_rows:
        symbol = co_symbol(document, co_entries[kid])
        files.extend([CO_ROOT / (symbol + '.co'),
                      JIT_ROOT / 'build/module_deepgemm_opus/blob/impl' / (symbol + '.cuh')])
    fingerprints = {str(path): digest(path) for path in files}
    order = PRIORITY + sorted(regressions - set(PRIORITY), key=lambda shape:
        -(float(current[shape]['us']) / float(baseline[shape]['us']) - 1))
    return {'git_head': actual_head, 'groups': groups, 'shape_order': order,
            'fingerprints': fingerprints, 'resources': resource_rows,
            'maximum_tensor_bytes': {'A': max(m*k for m,n,k in regressions),
                'B': max(n*k for m,n,k in regressions),
                'C_bf16': max(2*m*n for m,n,k in regressions),
                'reference_C_fp32': max(4*m*n for m,n,k in regressions)}}


def validate_bundle(scope='all'):
    if scope not in ('all', 'priority', 'remaining'):
        raise ValueError(f'Unknown scope: {scope}')
    inputs = collect_inputs()
    frozen = json.loads(FROZEN.read_text())
    if frozen['git_head'] != inputs['git_head'] or frozen['fingerprints'] != inputs['fingerprints']:
        raise ValueError('A frozen input, binary, CO, launcher or measurement source changed')
    shapes = (inputs['shape_order'] if scope == 'all' else list(PRIORITY)
              if scope == 'priority' else inputs['shape_order'][len(PRIORITY):])
    selected = [row for shape in shapes for row in inputs['groups'][shape]]
    plan = {'scope': scope, 'shapes': len(shapes), 'pairs': len(selected),
            'unique_kernel_ids': sorted({int(row['kernelId']) for row in selected}),
            'iters': 101, 'warmup': 5, 'batches': 3, 'seed': 0,
            'rotation_policy': 'Resolve once per shape, then share across both candidates and all batches',
            'candidate_order': 'historical/current; current/historical; historical/current',
            'expected_results': len(selected) * 3, 'gpu_count_if_executed': 1,
            'flydsl_gpu_calls': 0, 'baseline': 'Read-only matching fixed gfx950 CSV',
            'gpu_executed_by_this_check': False,
            'maximum_tensor_bytes': inputs['maximum_tensor_bytes']}
    return {'scope': scope, 'selected_pairs': selected,
            'shape_order': [list(shape) for shape in shapes], 'plan': plan,
            'fingerprints': inputs['fingerprints'], 'git_head': inputs['git_head'],
            'resources': inputs['resources']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Freeze CPU-validated inputs; never starts GPU work')
    parser.add_argument('--scope', choices=('all', 'priority', 'remaining'), default='all')
    args = parser.parse_args()
    if args.freeze:
        inputs = collect_inputs()
        FROZEN.write_text(json.dumps({'schema_version': 1,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'git_head': inputs['git_head'], 'fingerprints': inputs['fingerprints'],
            'gpu_used': False}, indent=2) + '\n')
    checked = validate_bundle(args.scope)
    print(json.dumps({'status': 'CPU checks passed', 'git_head': checked['git_head'],
                      'frozen_file_count': len(checked['fingerprints']), 'plan': checked['plan']}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'CPU preflight failed: {error}', file=sys.stderr)
        raise SystemExit(2)
