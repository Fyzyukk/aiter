#!/usr/bin/env python3
"""Read compiled HIP/ELF bytes offline; never load or execute a shared library."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GENERATED = HERE.parent / 'codegen'
LLVM = Path('/opt/rocm-llvm23-46fcb339/bin')
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
PARSER = ROOT / 'reports/opus_flydsl_all_20261009/hybrid_small/cpu_audit_v4/elf_parser.py'
spec = importlib.util.spec_from_file_location('offline_elf_parser', PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(argv):
    p = subprocess.run([str(x) for x in argv], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout

def symbols(path, flag):
    text = command([LLVM / 'llvm-nm', flag, path])
    return {line.split()[-1] for line in text.splitlines() if line.strip()}, text

def images(data):
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    offset = 0
    while True:
        begin = data.find(magic, offset)
        if begin < 0:
            return
        count, = struct.unpack_from('<Q', data, begin + len(magic))
        cursor = begin + len(magic) + 8
        records = []
        for _ in range(count):
            start, size, length = struct.unpack_from('<QQQ', data, cursor)
            cursor += 24
            target = data[cursor:cursor + length].decode()
            cursor += length
            records.append((target, start, size))
        for target, start, size in records:
            if 'amdgcn' in target and 'gfx950' in target:
                yield target, data[begin + start:begin + start + size]
        offset = begin + max(start + size for _, start, size in records)

def run():
    output = HERE / 'final_receipt.json'
    if output.exists():
        raise SystemExit('Refusing to overwrite final inspection')
    build = json.loads((HERE / 'build_receipt.json').read_text())
    assert build['status'] == 'full105_generated_hip_object_compilation_passed'
    meta = {row['kid']: row for row in json.loads((GENERATED / 'all105_metadata.json').read_text())}
    host_refs, host_text = symbols(HERE / 'objects/fused_host.o', '--undefined-only')
    stubs = {name for name in host_refs if '__device_stub__' in name}
    (HERE / 'host_undefined_symbols.txt').write_text(host_text)
    definitions, object_rows, runtime_rows = set(), [], []
    resources_names = ('sgpr_count', 'vgpr_count', 'agpr_count', 'group_segment_fixed_size',
                       'private_segment_fixed_size', 'vgpr_spill_count', 'sgpr_spill_count')
    old_build = {row['kid']: row for row in json.loads((ROOT / 'reports/opus_register92_20261009/linked_codegen_final/build_receipt.json').read_text())['builds']}
    old_bytes_equal = []
    for row in build['builds']:
        path = ROOT / row['object']
        assert sha(path) == row['object_sha256']
        defined, _ = symbols(path, '--defined-only')
        definitions |= defined
        elf = parser.Elf(path.read_bytes())
        bundles = list(images(elf.bytes(elf.sections['.hip_fatbin'])))
        assert len(bundles) == 1
        target, data = bundles[0]
        device = parser.Elf(data)
        kernels = device.metadata()['amdhsa.kernels']
        funcs = device.symbols()
        resources = []
        for kernel in kernels:
            name = kernel['.name']
            resource = {k: kernel.get('.' + k, 0) for k in resources_names}
            if meta[row['kid']]['runtime_split_k']:
                assert all(resource[k] == 0 for k in ('private_segment_fixed_size', 'vgpr_spill_count', 'sgpr_spill_count')), (row['kid'], name, resource)
            assert funcs[name]['size'] > 0
            resources.append(dict(kernel=name, resources=resource, kernarg_size=kernel.get('.kernarg_segment_size'), instruction_bytes=funcs[name]['size'], instruction_sha256=hashlib.sha256(funcs[name]['bytes']).hexdigest()))
        obj = dict(kid=row['kid'], compiler=row['side'], target=target, kernel_count=len(kernels), kernels=resources, bundled_device_sha256=hashlib.sha256(data).hexdigest())
        object_rows.append(obj)
        if meta[row['kid']]['runtime_split_k']:
            runtime_rows.append(obj)
        else:
            prior = parser.Elf((ROOT / old_build[row['kid']]['object']).read_bytes())
            prior_images = list(images(prior.bytes(prior.sections['.hip_fatbin'])))
            old_bytes_equal.append(dict(kid=row['kid'], device_image_byte_identical=(data == prior_images[0][1])))
    assert stubs <= definitions, sorted(stubs - definitions)
    linked = HERE / 'all105_runtime_registered.so'
    linked_defs, linked_text = symbols(linked, '--defined-only')
    linked_undef, undefined_text = symbols(linked, '--undefined-only')
    assert stubs <= linked_defs
    assert not {s for s in linked_undef if '__device_stub__' in s}
    (HERE / 'linked_defined_symbols.txt').write_text(linked_text)
    (HERE / 'linked_undefined_symbols.txt').write_text(undefined_text)
    (HERE / 'linked_dynamic.txt').write_text(command([LLVM / 'llvm-readelf', '-d', linked]))
    linked_elf = parser.Elf(linked.read_bytes())
    linked_kernel_names = set()
    for _, data in images(linked_elf.bytes(linked_elf.sections['.hip_fatbin'])):
        linked_kernel_names.update(k['.name'] for k in parser.Elf(data).metadata()['amdhsa.kernels'])
    runtime_names = {k['kernel'] for row in runtime_rows for k in row['kernels']}
    assert runtime_names <= linked_kernel_names
    assert len(runtime_rows) == 9
    runtime_main_names = {name for name in runtime_names if 'reduce_runtime_kernel' not in name}
    assert len(runtime_main_names) == 12
    runtime_reduce_names = {name for name in runtime_names if 'reduce_runtime_kernel' in name}
    assert len(runtime_reduce_names) == 1
    assert all(k['kernarg_size'] == (24 if 'reduce_runtime_kernel' in k['kernel'] else 104) for r in runtime_rows for k in r['kernels'])
    result = dict(status='all105_fresh_hip_compile_and_router_no_undefined_link_verified_gpu_pending',
                  registered_ids=105, public_tuning_candidates=89, runtime_configurations=9,
                  fresh_compile_count=105, reused_object_count=0,
                  compiler_counts={side: sum(r['side'] == side for r in build['builds']) for side in ('main23', 'pin24')},
                  fused_host_stub_reference_count=len(stubs), all_host_stub_references_resolved=True,
                  runtime_main_device_specializations=len(runtime_main_names), runtime_reducer_specializations=len(runtime_reduce_names),
                  runtime_device_symbols_in_linked_fatbin=True, all_emitted_device_instances=sum(r['kernel_count'] for r in object_rows),
                  runtime_device_instances_zero_scratch_and_spills=True,
                  nonruntime96_device_image_comparison=old_bytes_equal,
                  nonruntime96_device_images_byte_identical=all(r['device_image_byte_identical'] for r in old_bytes_equal),
                  runtime_resources=runtime_rows, all_objects=object_rows,
                  library=str(linked.relative_to(ROOT)), library_sha256=sha(linked),
                  receipt_inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in (GENERATED/'all105_metadata.json', HERE/'build_receipt.json', HERE/'host_build_receipt.json', HERE/'link_receipt.json', Path(__file__), PARSER)},
                  gpu_queries=0, torch_or_aiter_module_imports=0, libraries_loaded=0, kernel_launches=0,
                  numerical_validation='not_run_gpu_stopped', performance_validation='not_run_gpu_stopped')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_resources','all_objects','receipt_inputs_sha256','nonruntime96_device_image_comparison')}, indent=2))

if __name__ == '__main__':
    run()
