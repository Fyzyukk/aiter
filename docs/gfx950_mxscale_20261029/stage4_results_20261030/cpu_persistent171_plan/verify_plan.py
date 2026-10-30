#!/usr/bin/env python3
"""Validate the CPU design package and existing candidates; never launch kernels."""
import csv
import hashlib
import importlib.abc
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

PLAN = Path(__file__).resolve().parent
REPO = Path('/home/fyz/aiter')
FORBIDDEN = {'torch', 'aiter', 'csrc', 'ctypes', 'subprocess'}


class CPUImportsOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError(f'CPU plan validation rejects import: {fullname}')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    sys.meta_path.insert(0, CPUImportsOnly())
    manifest = json.loads((PLAN / 'plan_manifest.json').read_text())
    require(manifest['gpu_executed'] is False, 'This package describes CPU work only')
    for item in manifest['inputs']:
        require(sha256(item['path']) == item['sha256'], f"Input changed: {item['path']}")

    spec = importlib.util.spec_from_file_location(
        'cpu_design_opus_common', REPO / 'csrc/opus_gemm/opus_gemm_common.py')
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    registry = common.get_registry('a8w8_blockscale_bpreshuffle', 'gfx950', 256)

    losses = read_csv(PLAN.parent / 'persistent171_analysis.csv')
    inventory = read_csv(PLAN / 'current_winner_inventory.csv')
    shape = lambda row: tuple(int(row[key]) for key in ('M', 'N', 'K'))
    loss_by_shape = {shape(row): row for row in losses}
    require(len(losses) == len(loss_by_shape) == 171, 'Persistent loss set must have 171 unique shapes')
    require(len(inventory) == 171 and {shape(row) for row in inventory} == set(loss_by_shape),
            'Winner inventory does not match persistent loss set')
    for row in inventory:
        original = loss_by_shape[shape(row)]
        require(float(original['current_us']) > float(original['csv_us']) and
                float(original['historical_opus_us']) > float(original['csv_us']),
                f'Unexpected loss classification: {shape(row)}')
        require(row['kernelId'] == original['kernelId'] and row['kernelName'] == original['kernelName'],
                f'Inventory identity changed: {shape(row)}')
        instance = registry[int(row['kernelId'])]
        require(instance.name == row['kernelName'] and
                common.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape(row)),
                f'Current winner unavailable or illegal: {shape(row)}')

    candidates = read_csv(PLAN / 'narrow_existing_candidates.csv')
    keys = [(shape(row), int(row['kernelId'])) for row in candidates]
    counts = Counter(key[0] for key in keys)
    require(len(keys) == len(set(keys)) and len(counts) == 4 and len(keys) <= 16 and
            max(counts.values()) <= 4, 'Narrow matrix must contain four shapes and at most four exact candidates each')
    for row in candidates:
        require(shape(row) in loss_by_shape, f'Narrow representative is outside 171: {shape(row)}')
        require((row['gfx'], row['cu_num'], row['w_scale_block'], row['libtype']) ==
                ('gfx950', '256', '128x128', 'opus'), 'Unexpected candidate comparison scope')
        instance = registry[int(row['kernelId'])]
        require(instance.name == row['kernelName'] and
                common.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape(row)),
                f'Planned existing candidate unavailable or illegal: {shape(row)}, {row["kernelId"]}')
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        actual_traits = entry['traits']
        if isinstance(actual_traits, dict):
            actual_traits = actual_traits['name'] + '<' + ', '.join(actual_traits['args']) + '>'
        declared_traits = row['traits']
        if declared_traits.lstrip().startswith('{'):
            encoded = json.loads(declared_traits)
            declared_traits = encoded['name'] + '<' + ', '.join(encoded['args']) + '>'
        require(re.sub(r'\s+', '', actual_traits) == re.sub(r'\s+', '', declared_traits),
                f'Candidate traits mismatch: {shape(row)}, {row["kernelId"]}')
    for selected_shape in counts:
        require((selected_shape, int(loss_by_shape[selected_shape]['kernelId'])) in set(keys),
                f'Current winner missing from narrow controls: {selected_shape}')

    designs = {name: json.loads((PLAN / f'{name}_plan.json').read_text())
               for name in ('4wave', 'interleave', 'narrow')}
    narrow_variants = designs['narrow']['variants']
    variant_by_key = {
        (tuple(item['shape'][key] for key in ('M', 'N', 'K')), item['kernelId']): item
        for item in narrow_variants
    }
    require(len(variant_by_key) == len(narrow_variants) == len(keys) and set(variant_by_key) == set(keys),
            'Narrow JSON and CSV candidates differ')
    single_factor_checks = 0
    for key, variant in variant_by_key.items():
        compare_id = variant['compare_to_kernelId']
        if compare_id is None:
            continue
        other = variant_by_key[(key[0], compare_id)]
        parameters = variant['parameters']
        previous = other['parameters']
        differences = [name for name in set(parameters) | set(previous)
                       if parameters.get(name) != previous.get(name)]
        require(len(differences) == 1 and differences[0] == variant['changed_parameters'][0]['parameter'],
                f'Narrow control changes multiple input factors: {key}')
        single_factor_checks += 1
    forbidden_loaded = [name for name in sys.modules if name.split('.')[0] in FORBIDDEN]
    require(not forbidden_loaded, f'Unexpected imports: {forbidden_loaded}')
    print(json.dumps({
        'status': 'cpu_validation_passed', 'gpu_executed': False,
        'new_kernels_compiled': False, 'frozen_inputs_checked': len(manifest['inputs']),
        'persistent_winner_identities_and_shape_legality_checked': len(inventory),
        'narrow_existing_shapes': len(counts), 'narrow_existing_pairs': len(keys),
        'narrow_single_input_parameter_controls_checked': single_factor_checks,
        'design_documents_read': list(designs), 'forbidden_modules_loaded': forbidden_loaded,
        'limits': 'Design validation only; no device correctness or performance verification.'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
