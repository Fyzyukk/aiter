#!/usr/bin/env python3
"""Build isolated narrow scale issue candidate and audit exact baseline; CPU only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/opus_bound_analysis_20261007/narrow_unroll'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    template = json.loads((OLD / 'build_manifest.json').read_text())['builds'][0]['command']
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    builds = []
    for side in ['baseline', 'candidate']:
        command = [arg.replace(str(OLD / 'baseline'), str(HERE / side)) for arg in template]
        start = time.monotonic()
        with (HERE / f'{side}_build.log').open('w') as log:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        builds.append({'side': side, 'command': command, 'returncode': run.returncode, 'seconds': time.monotonic() - start})
        (HERE / 'build_manifest.json').write_text(json.dumps({'status': 'building', 'cpu_only': True, 'gpu_executed': False, 'builds': builds}, indent=2) + '\n')
        if run.returncode: raise RuntimeError(f'{side} CPU build failed; inspect log')
        builds[-1]['library_sha256'] = sha(HERE / side / 'experiments.so')
    (HERE / 'build_manifest.json').write_text(json.dumps({'status': 'passed', 'cpu_only': True, 'gpu_executed': False, 'builds': builds}, indent=2) + '\n')
    helper = ROOT / 'reports/opus_bound_analysis_20261007/audit_current_compute_metadata.py'
    spec = importlib.util.spec_from_file_location('cpu_elf', helper)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    review = json.loads((HERE.parent / 'narrow_review.json').read_text())
    expected = {v['symbol']: v for v in review['variants']}
    images = {}; rows = {}
    for side in ['baseline', 'candidate']:
        data = mod.device_bundle(HERE / side / 'experiments.so')
        image = HERE / side / 'device.co'; image.write_bytes(data)
        _, kernels = mod.summarize_image(data)
        rows[side] = {r['name']: r for r in kernels}
        images[side] = {'path': str(image), 'sha256': sha(image), 'kernels': kernels}
        with (HERE / side / 'device.s').open('w') as out:
            subprocess.run(['/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump', '-d', '--mcpu=gfx950', str(image)], stdout=out, check=True)
    assert set(rows['baseline']) == set(rows['candidate']) == set(expected)
    checks = []
    for name, ref in expected.items():
        base, candidate = rows['baseline'][name], rows['candidate'][name]
        exact = (base['instruction_sha256'] == ref['instruction_sha256'] and base['metadata'] == ref['metadata'] and base['descriptor_normalized_sha256'] == ref['descriptor_normalized_sha256'])
        changed_expected = ref['variant'] in ('9023_runtime', '9024_fixed')
        assert exact, f'Baseline differs from current official: {ref["variant"]}'
        same = all(base[k] == candidate[k] for k in ('instruction_sha256', 'metadata', 'descriptor_normalized_sha256'))
        assert not changed_expected or not same, f'Candidate did not change affected entry: {ref["variant"]}'
        assert changed_expected or same, f'Candidate changed frozen control: {ref["variant"]}'
        for r in (base, candidate):
            assert r['metadata']['.private_segment_fixed_size'] == r['metadata']['.vgpr_spill_count'] == 0
        checks.append({'variant': ref['variant'], 'parent_id': ref['parent_id'], 'name': name, 'baseline_exact_current_official': exact, 'candidate_changed_expected': changed_expected, 'candidate_same_as_baseline': same, 'baseline': base, 'candidate': candidate})
    result = {'status': 'passed', 'cpu_only': True, 'gpu_executed': False, 'production_modified': False, 'images': images, 'checks': checks, 'build_manifest_sha256': sha(HERE / 'build_manifest.json')}
    (HERE / 'device_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'entries': [(r['variant'], r['baseline']['metadata']['.vgpr_count'], r['candidate']['metadata']['.vgpr_count'], r['baseline']['metadata']['.sgpr_spill_count'], r['candidate']['metadata']['.sgpr_spill_count']) for r in checks]}))

if __name__ == '__main__': main()
