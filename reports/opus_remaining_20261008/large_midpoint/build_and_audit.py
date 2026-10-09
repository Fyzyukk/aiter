#!/usr/bin/env python3
"""Root-invoked CPU build; strict exact official 9030 alignment before GPU gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inventory = json.loads((HERE.parent / 'inventory.json').read_text())
    entry = next(e for e in inventory['entries'] if e['parent_id'] == 9030)
    official = inventory['official_module']
    assert sha(official['path']) == official['sha256']
    source = json.loads((HERE / 'source_manifest.json').read_text())
    for f in source['source_files']:
        for side in ['baseline', 'candidate']:
            assert sha(HERE / side / f['path']) == f[side + '_sha256']
    template_dir = HERE.parent / 'scale_issue'
    command = json.loads((template_dir / 'build_manifest.json').read_text())['builds'][0]['command']
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    builds = []
    for side in ['baseline', 'candidate']:
        argv = [arg.replace(str(template_dir / 'baseline'), str(HERE / side)) for arg in command]
        started = time.monotonic()
        with (HERE / (side + '_build.log')).open('w') as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'side': side, 'command': argv, 'returncode': result.returncode,
               'seconds': time.monotonic() - started}
        builds.append(row)
        assert result.returncode == 0, side + ' build failed; see log'
        row['library_sha256'] = sha(HERE / side / 'experiments.so')
    manifest = {'status': 'passed', 'cpu_only': True, 'builds': builds,
                'source_manifest_sha256': sha(HERE / 'source_manifest.json')}
    (HERE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    helper = ROOT / 'reports/opus_bound_analysis_20261007/audit_current_compute_metadata.py'
    spec = importlib.util.spec_from_file_location('large_midpoint_elf_helpers', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    formal = json.loads((ROOT / 'reports/opus_resume_20261008/formal_selected/identity_audit.json').read_text())
    linked = {c['candidate_image_sha256'] for c in formal['linked_module_bundle_checks']}
    att = json.loads((HERE.parent / 'diagnostics/merged_att_analysis.json').read_text())
    capture = next(c for c in att['captures'] if c['parent_id'] == 9030)
    assert capture['status'] == 'passed' and capture['identity']['official_module_sha256'] == official['sha256']
    reference_path = Path(capture['identity']['code_object'])
    assert sha(reference_path) == capture['identity']['code_object_sha256'] in linked
    _, references = module.summarize_image(reference_path.read_bytes())
    reference = next(r for r in references if r['name'] == entry['symbol'])
    assert reference['instruction_sha256'] == entry['instruction_sha256']
    rows = {}; images = {}
    for side in ['baseline', 'candidate']:
        data = module.device_bundle(HERE / side / 'experiments.so')
        path = HERE / side / 'device.co'
        path.write_bytes(data)
        _, kernels = module.summarize_image(data)
        assert [r['name'] for r in kernels] == [entry['symbol']]
        rows[side] = kernels[0]
        images[side] = {'path': str(path), 'sha256': sha(path), 'kernels': kernels}
        with (HERE / side / 'device.s').open('w') as stream:
            subprocess.run(['/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump',
                            '-d', '--mcpu=gfx950', str(path)], stdout=stream, check=True)
    base = rows['baseline']; candidate = rows['candidate']
    for key in ['instruction_sha256', 'instruction_bytes', 'metadata', 'descriptor_normalized_sha256']:
        assert base[key] == reference[key], 'Strict official baseline mismatch: ' + key
    for row in rows.values():
        for field in ['.private_segment_fixed_size', '.vgpr_spill_count', '.sgpr_spill_count']:
            assert row['metadata'][field] == 0
    assert base['metadata']['.group_segment_fixed_size'] == candidate['metadata']['.group_segment_fixed_size'] == 143360
    result = {'status': 'passed', 'cpu_only': True, 'official_module': official, 'images': images,
              'checks': [{'parent_id': 9030, 'name': entry['symbol'], 'baseline': base, 'candidate': candidate,
                          'baseline_matches_oct8_FUNC_full_metadata_normalized_descriptor': True, 'LDS_unchanged': True}],
              'build_manifest_sha256': sha(HERE / 'build_manifest.json'),
              'remaining': 'Independent candidate ISA synchronization review, numerical guards and Event screen.'}
    (HERE / 'device_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'baseline_exact': True,
                      'resources': {s: {f: r['metadata'][f] for f in ['.vgpr_count', '.sgpr_count', '.group_segment_fixed_size']} for s, r in rows.items()}}))


if __name__ == '__main__':
    main()
