#!/usr/bin/env python3
"""CPU-only BN48 tail versus BN32 fixed K7168 experiment and code audit."""
import collections
import hashlib
import importlib.util
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


def audit_code(device, folder):
    meta_text = run([LLVM / 'llvm-readelf', '--notes', device])
    (folder / 'metadata.txt').write_text(meta_text)
    meta = yaml.safe_load(meta_text[meta_text.index('amdhsa.kernels:'):].split('\n...', 1)[0])
    isa = run([LLVM / 'llvm-objdump', '-d', '--mcpu=gfx950', device])
    (folder / 'isa.txt').write_text(isa)
    sections = run([LLVM / 'llvm-readelf', '-S', device])
    section = re.search(r'\[\s*\d+\]\s+\.text\s+\S+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)', sections)
    text_va, text_offset = (int(v, 16) for v in section.groups())
    symbols = {}
    for line in run([LLVM / 'llvm-readelf', '-s', device]).splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == 'FUNC':
            symbols[fields[-1]] = (int(fields[1], 16), int(fields[2]))
    blocks = {}
    matches = list(re.finditer(r'^([0-9a-fA-F]+) <([^>]+)>:\n', isa, re.M))
    for index, match in enumerate(matches):
        end = matches[index+1].start() if index+1 < len(matches) else len(isa)
        blocks[match.group(2)] = isa[match.end():end]
    blob = device.read_bytes()
    rows = []
    for kernel in meta['amdhsa.kernels']:
        name = kernel['.name']; addr, size = symbols[name]
        code = blob[text_offset+addr-text_va:text_offset+addr-text_va+size]
        counts = collections.Counter()
        for line in blocks[name].splitlines():
            match = re.match(r'\s+([a-z][a-z0-9_]+)\b.*//\s+[0-9A-Fa-f]+:', line)
            if match: counts[match.group(1)] += 1
        rows.append(dict(name=name, template_ints=[int(v) for v in re.findall(r'Li(-?\d+)E', name)][:9],
                         metadata=kernel, instruction_bytes=size,
                         instruction_sha256=hashlib.sha256(code).hexdigest(),
                         instruction_counts=dict(sorted(counts.items()))))
    return dict(device_sha256=sha(device), metadata_sha256=sha(folder/'metadata.txt'),
                isa_sha256=sha(folder/'isa.txt'), kernels=rows)


def extract_code(source, folder):
    fat = folder/'fatbin.bin'; device = folder/'device.co'
    run([LLVM/'llvm-objcopy', '--dump-section=.hip_fatbin='+str(fat), source])
    targets = run([LLVM/'clang-offload-bundler','-type=o','-list','-input='+str(fat)]).splitlines()
    target = 'hip-amdgcn-amd-amdhsa--gfx950'; assert target in targets
    run([LLVM/'clang-offload-bundler','-type=o','-unbundle','-targets='+target,
         '-input='+str(fat),'-output='+str(device)])
    return device


def main():
    claim = json.loads((HERE.parent/'gpu_claim_log.jsonl').read_text().splitlines()[-1])
    assert claim['event']=='waiting', claim
    prior = json.loads((HERE.parent/'register_reuse/previous_build_manifest_6_entries.json').read_text())
    variants = json.loads((HERE/'variants.json').read_text())['variants']
    manifest = dict(cpu_only=True, gpu_executed=False, production_modified=False, builds=[])
    audit = dict(cpu_only=True, gpu_executed=False, production_modified=False, libraries=[])
    for side in ['baseline','candidate']:
        folder=HERE/side
        command=list(next(v for v in prior['libraries'] if v['mode']==side)['command'])
        command=[v.replace(str(HERE.parent/'register_reuse'/side),str(folder)) for v in command]
        started=time.time()
        with (folder/'build.log').open('w') as log:
            proc=subprocess.run(command,env=ENV,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert proc.returncode==0,(side,proc.returncode)
        manifest['builds'].append(dict(side=side,command=command,exit_code=0,seconds=time.time()-started,
            source_sha256={p.name:sha(p) for p in folder.glob('*') if p.suffix in ['.hip','.cuh']},
            binary_sha256=sha(folder/'experiments.so')))
        result=audit_code(extract_code(folder/'experiments.so',folder),folder)
        assert len(result['kernels'])==2
        for kernel in result['kernels']:
            matches=[v for v in variants if v['side']==side and kernel['template_ints']==
                [v[k] for k in ['B_M','B_N','T_M','T_N','prefetch','wave_k','output','B_cache','FIXED_K']]]
            assert len(matches)==1; kernel['variant']=matches[0]['variant']
            kernel['N_TAIL']=matches[0]['N_TAIL'];kernel['ReuseBScale']=False
        result.update(side=side,binary_sha256=sha(folder/'experiments.so'))
        audit['libraries'].append(result)
        print(json.dumps(dict(side=side,status='cpu_build_passed',device_entries=2)),flush=True)
    official_folder=HERE/'official';official_folder.mkdir(exist_ok=True)
    official_object=ROOT/'reports/opus_bound_analysis_20261007/jit_baseline/build/module_deepgemm_opus/build/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_Cbf16_t.device.cuda.o'
    official=audit_code(extract_code(official_object,official_folder),official_folder)
    audit['official_object']=dict(path=str(official_object),sha256=sha(official_object))
    for kernel in audit['libraries'][0]['kernels']:
        match=next(v for v in official['kernels'] if v['name']==kernel['name'])
        kernel['official_comparison']=dict(instruction_bytes=kernel['instruction_bytes']==match['instruction_bytes'],
            instruction_sha256=kernel['instruction_sha256']==match['instruction_sha256'],
            metadata=kernel['metadata']==match['metadata'])
    (HERE/'device_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest.update(status='passed',verified_device_entries_per_library=2,
        device_audit_sha256=sha(HERE/'device_audit.json'),
        scope='BN48/PadN true baseline -> BN32/PadN false candidate; ReuseBScale false both; FixedK7168 and Output4 fixed',
        gpu_visibility='ROCR/HIP/CUDA_VISIBLE_DEVICES empty for compilation/audit')
    (HERE/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for library in audit['libraries']:
        for kernel in library['kernels']:
            print(json.dumps(dict(side=library['side'],kid=kernel['variant'],
                 resource={k:v for k,v in kernel['metadata'].items() if any(s in k for s in ['count','size']) and k!='.args'},
                 instruction_bytes=kernel['instruction_bytes'],
                 official=kernel.get('official_comparison'))),flush=True)


if __name__=='__main__':
    main()
