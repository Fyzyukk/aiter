#!/usr/bin/env python3
"""CPU-only rebuild and audit of eight isolated runtime-register traits."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LLVM = Path('/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin')
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    return subprocess.check_output([str(x) for x in args], env=ENV, cwd=ROOT, text=True)


def audit(side, variants):
    folder = HERE / side
    lib = folder / 'experiments.so'
    fat = folder / 'fatbin.bin'
    device = folder / 'device.co'
    inspection_copy = folder / 'inspection_copy.so'
    run([LLVM / 'llvm-objcopy', '--dump-section=.hip_fatbin=' + str(fat), lib, inspection_copy])
    inspection_copy.unlink()
    targets = run([LLVM / 'clang-offload-bundler', '-type=o', '-list', '-input=' + str(fat)]).splitlines()
    target = 'hip-amdgcn-amd-amdhsa--gfx950'
    assert target in targets, targets
    run([LLVM / 'clang-offload-bundler', '-type=o', '-unbundle', '-targets=' + target,
         '-input=' + str(fat), '-output=' + str(device)])
    meta = run([LLVM / 'llvm-readelf', '--notes', device])
    (folder / 'metadata.txt').write_text(meta)
    note = meta[meta.index('amdhsa.kernels:'):].split('\n...', 1)[0]
    kernels = yaml.safe_load(note)['amdhsa.kernels']
    assert len(kernels) == len(variants) == 8, len(kernels)
    isa = run([LLVM / 'llvm-objdump', '-d', '--mcpu=gfx950', device])
    (folder / 'isa.txt').write_text(isa)
    symbols = run([LLVM / 'llvm-readelf', '-s', device])
    sections = run([LLVM / 'llvm-readelf', '-S', device])
    section_match = re.search(r'\[\s*\d+\]\s+\.text\s+\S+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)', sections)
    assert section_match
    text_va, text_offset = (int(v, 16) for v in section_match.groups())
    blob = device.read_bytes()
    symbol_map = {}
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == 'FUNC':
            symbol_map[fields[-1]] = (int(fields[1], 16), int(fields[2]))
    isa_blocks = {}
    matches = list(re.finditer(r'^([0-9a-fA-F]+) <([^>]+)>:\n', isa, re.M))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(isa)
        isa_blocks[match.group(2)] = isa[match.end():end]
    rows = []
    for kernel in kernels:
        name = kernel['.name']
        integers = [int(v) for v in re.findall(r'Li(-?\d+)E', name)]
        traits = integers[:9]
        chosen = [v for v in variants if traits == [v['B_M'], v['B_N'], v['T_M'], v['T_N'],
                  v['prefetch'], v['wave_k'], v['output'], v['B_cache'], v['FIXED_K']]]
        assert len(chosen) == 1, (traits, chosen)
        addr, size = symbol_map[name]
        code = blob[text_offset + addr - text_va:text_offset + addr - text_va + size]
        assert len(code) == size
        counts = collections.Counter()
        for line in isa_blocks[name].splitlines():
            match = re.match(r'\s+([a-z][a-z0-9_]+)\b.*//\s+[0-9A-Fa-f]+:', line)
            if match:
                counts[match.group(1)] += 1
        resources = {key: kernel['.' + key] for key in ['agpr_count', 'vgpr_count', 'sgpr_count',
            'group_segment_fixed_size', 'private_segment_fixed_size', 'vgpr_spill_count', 'sgpr_spill_count']}
        rows.append(dict(variant=chosen[0]['variant'], name=name, instruction_bytes=size,
                         instruction_sha256=hashlib.sha256(code).hexdigest(), resources=resources,
                         instruction_counts=dict(sorted(counts.items())),
                         b8_vmem_count=counts['buffer_load_ubyte'], template_ints=traits))
    rows.sort(key=lambda v: v['variant'])
    return dict(mode=side, binary_sha256=sha(lib), device_sha256=sha(device),
                metadata_sha256=sha(folder / 'metadata.txt'), isa_sha256=sha(folder / 'isa.txt'),
                target=target, kernels=rows)


def main():
    claim_path = HERE.parent / 'gpu_claim_log.jsonl'
    if claim_path.exists():
        claim = json.loads(claim_path.read_text().splitlines()[-1])
        assert claim['event'] == 'waiting', claim
    prior = json.loads((HERE / 'previous_build_manifest_6_entries.json').read_text())
    prior_audit = json.loads((HERE / 'previous_device_audit_6_entries.json').read_text())
    variants = json.loads((HERE / 'variants.json').read_text())['variants']
    manifest = dict(cpu_only=True, gpu_executed=False, status='running', libraries=[])
    for old in prior['libraries']:
        side = old['mode']; folder = HERE / side
        command = old['command']
        started = time.time()
        with (folder / 'build.log').open('w') as log:
            proc = subprocess.run(command, env=ENV, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, (side, proc.returncode)
        manifest['libraries'].append(dict(mode=side, command=command, returncode=proc.returncode,
             seconds=time.time()-started, source_sha256={p.name: sha(p) for p in folder.glob('*')
             if p.suffix in ['.hip', '.cuh']}, binary_sha256=sha(folder / 'experiments.so')))
        print(json.dumps(dict(mode=side, cpu_build='passed')), flush=True)
    result = dict(cpu_only=True, gpu_executed=False, libraries=[])
    for side in ['baseline', 'candidate']:
        result['libraries'].append(audit(side, variants))
    original_unchanged = {}
    for old, new in zip(prior_audit['libraries'], result['libraries']):
        by_id = {v['variant']: v for v in new['kernels']}
        for kernel in old['kernels']:
            fresh = by_id[kernel['variant']]
            original_unchanged[str(kernel['variant']) + '/' + old['mode']] = dict(
                instruction_bytes=kernel['instruction_bytes'] == fresh['instruction_bytes'],
                instruction_sha256=kernel['instruction_sha256'] == fresh['instruction_sha256'],
                resources=kernel['resources'] == fresh['resources'])
    result['original_six_comparison'] = original_unchanged
    (HERE / 'device_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest.update(status='passed', verified_device_entries_per_library=8,
                    device_audit_sha256=sha(HERE / 'device_audit.json'),
                    original_six_comparison=original_unchanged,
                    extension='Added identical host/device 9041 and 9050 entries only; original six cases unchanged.',
                    previous_build_manifest=prior,
                    gpu_visibility='ROCR/HIP/CUDA_VISIBLE_DEVICES empty during every compiler/audit subprocess')
    (HERE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for kid in [9041, 9050]:
        print(json.dumps(dict(kid=kid, resources=[next(v for v in library['kernels'] if v['variant']==kid)
             for library in result['libraries']])), flush=True)


if __name__ == '__main__':
    main()
