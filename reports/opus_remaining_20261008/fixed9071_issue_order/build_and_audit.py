#!/usr/bin/env python3
"""Compile exact formal small-family TUs with isolated headers; CPU only."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLANG = '/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++'
OBJDUMP = '/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump'
FORMAL_BUILD = ROOT / 'reports/opus_resume_20261008/jit_formal_selected/build/module_deepgemm_opus/build'
SPEC = importlib.util.spec_from_file_location('fine_cpu_elf', ROOT / 'reports/opus_resume_20261008/formal_selected/smoke_common.py')
ELF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ELF)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_one(task):
    side, tu, flags = task
    out = HERE / side / (Path(tu['official_object']).name)
    argv = [CLANG, '-x', 'hip', '-I' + str(HERE / side / 'include')] + flags + ['-c', tu['source'], '-o', str(out)]
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    started = time.monotonic()
    log = HERE / side / ('kid' + str(tu['parent_id']) + '_build.log')
    if REUSE_OBJECTS:
        assert out.exists() and log.exists() and 'error:' not in log.read_text()
        assert out.stat().st_mtime >= max(p.stat().st_mtime for p in (HERE / side / 'include').rglob('*') if p.is_file())
    else:
        with log.open('w') as handle:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{side} kid{tu["parent_id"]} failed; see {log}')
    images = ELF.device_bundles(out)
    assert len(images) == 1
    image = HERE / side / ('kid' + str(tu['parent_id']) + '.co')
    image.write_bytes(images[0])
    rows = {r['name']: r for r in ELF.summarize_image(images[0])}
    expected = {r['name']: r for r in tu['variants']}
    assert set(rows) == set(expected), f'TU symbols differ {side} {tu["parent_id"]}'
    if side == 'baseline':
        for name, row in rows.items():
            assert ELF.identity(row, expected[name])['matches'], f'Exact Oct8 baseline differs {name}'
    for row in rows.values():
        assert row['metadata']['.private_segment_fixed_size'] == 0
        assert row['metadata']['.vgpr_spill_count'] == row['metadata']['.sgpr_spill_count'] == 0
    with image.with_suffix('.s').open('w') as handle:
        subprocess.run([OBJDUMP, '-d', '--mcpu=gfx950', str(image)], stdout=handle, check=True)
    print(json.dumps({'side': side, 'parent': tu['parent_id'], 'exact_baseline': side == 'baseline', 'kernels': len(rows)}), flush=True)
    return {'side': side, 'parent_id': tu['parent_id'], 'command': argv, 'object': str(out),
            'object_sha256': sha(out), 'image': str(image), 'image_sha256': sha(image),
            'seconds': time.monotonic() - started, 'returncode': 0, 'kernels': list(rows.values())}


def host_source():
    return '#define __HIPCC_RTC__ 1\n#include <opus/hip_minimal.hpp>\n#include <cstdint>\n#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"\ntemplate<class T> __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);\nusing F9071=opus_gemm_small_register_traits_gfx950<16,48,1,1,4,4,4,3,7168,true,false>;\n#if !defined(__HIP_DEVICE_COMPILE__)\nextern "C" __attribute__((visibility("default"))) int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream){\n constexpr int64_t limit=INT32_MAX;\n if(kid!=9071||!a||!b||!sfa||!sfb||!c||m<1||m>512||n<128||n%128||k!=7168||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*2>limit||reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||reinterpret_cast<uintptr_t>(c)%16)return 1;\n opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};\n args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;args.m=m;args.n=n;args.k=k;args.batch=1;\n args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;\n args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);\n gemm_a8w8_mxfp8_scale_small_register_kernel<F9071><<<dim3((n+47)/48,(m+15)/16),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args);\n return hipGetLastError();\n}\n#endif\n'


def main():
    global REUSE_OBJECTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-completed-device-objects', action='store_true',
                        help='Resume after host-only failure using unchanged, already compiled and audited TU objects.')
    args = parser.parse_args()
    REUSE_OBJECTS = args.reuse_completed_device_objects
    source = json.loads((HERE / 'source_manifest.json').read_text())
    flags = next(line.split(' = ', 1)[1] for line in (FORMAL_BUILD / 'build.ninja').read_text().splitlines() if line.startswith('cuda_cflags = '))
    flags = shlex.split(flags)
    tuples = sorted(source['device_tus'], key=lambda t: (t['parent_id'] != 9042, t['parent_id']))
    tasks = [(side, t, flags) for t in tuples for side in ['baseline', 'candidate']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds = list(pool.map(compile_one, tasks))
    (HERE / 'object_build_records.json').write_text(json.dumps({'builds': builds,
           'reused_completed_device_objects_after_host_failure': REUSE_OBJECTS,
           'source_manifest_sha256': sha(HERE / 'source_manifest.json')}, indent=2) + '\n')
    by_key = {(b['side'], b['parent_id']): b for b in builds}
    selected_names = set(source['selected_symbols'])
    checks = []
    for tu in tuples:
        baseline = {r['name']: r for r in by_key['baseline', tu['parent_id']]['kernels']}
        candidate = {r['name']: r for r in by_key['candidate', tu['parent_id']]['kernels']}
        for name, base in baseline.items():
            cand = candidate[name]
            match = ELF.identity(base, cand)
            selected = name in selected_names
            assert selected or match['matches'], f'Unselected kernel changed {tu["parent_id"]} {name}'
            assert base['metadata']['.group_segment_fixed_size'] == cand['metadata']['.group_segment_fixed_size']
            checks.append({'parent_id': tu['parent_id'], 'name': name, 'selected': selected,
                           'baseline_exact_official': True, 'baseline': base, 'candidate': cand,
                           'identity': match})
    assert sum(c['selected'] for c in checks) == 1
    libs = []
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    for side in ['baseline', 'candidate']:
        src = HERE / side / 'host.hip'
        src.write_text(host_source())
        obj = HERE / side / 'host.o'
        command = [CLANG, '-x', 'hip', '-I' + str(HERE / side / 'include')] + flags + ['--offload-host-only', '-c', str(src), '-o', str(obj)]
        with (HERE / side / 'host_build.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        lib = HERE / side / 'experiments.so'
        link = [CLANG, '-shared', '-mcmodel=large', '-Wl,--gc-sections', str(obj)] + [b['object'] for b in builds if b['side'] == side] + ['-L/opt/rocm/lib', '-lamdhip64', '-o', str(lib)]
        with (HERE / side / 'link.log').open('w') as log:
            subprocess.run(link, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        _, linked = ELF.device_rows(lib)
        assert selected_names <= {r['name'] for r in linked}
        expected = {c['name']: c[side] for c in checks}
        assert all(ELF.identity(row, expected[row['name']])['matches'] for row in linked)
        libs.append({'side': side, 'host_command': command, 'link_command': link, 'path': str(lib),
                     'sha256': sha(lib), 'linked_rows': len(linked)})
    result = {'status': 'passed_exact_CPU_build_and_scope_audit', 'cpu_only': True,
              'GPU_executed': False, 'production_modified': False,
              'selected_symbols': sorted(selected_names), 'source_manifest_sha256': sha(HERE / 'source_manifest.json'),
              'reused_completed_device_objects_after_host_failure': REUSE_OBJECTS,
              'script_sha256': sha(__file__), 'builds': builds, 'libraries': libs,
              'checks': checks, 'unselected_device_entries_unchanged': len(checks) - 1,
              'all_baseline_FUNC_fullmetadata_descriptors_exact_official': True}
    (HERE / 'device_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'build_manifest.json').write_text(json.dumps({'status': result['status'], 'cpu_only': True,
                                                        'builds': builds, 'libraries': libs}, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'entries': len(checks), 'unselected_unchanged': len(checks)-1,
                      'selected_resources': [(c['baseline']['metadata']['.vgpr_count'], c['candidate']['metadata']['.vgpr_count'],
                                              c['baseline']['metadata']['.sgpr_count'], c['candidate']['metadata']['.sgpr_count'])
                                             for c in checks if c['selected']]}), flush=True)


if __name__ == '__main__':
    main()
