#!/usr/bin/env python3
"""CPU-only helpers for the Oct8 selected official API smoke evidence."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import msgpack

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / 'reports/opus_bound_analysis_20261007'
PLAN = HERE / 'official_smoke_plan.json'
QUEUE = HERE / 'official_smoke_queue.json'
BUILD = HERE / 'build_manifest.json'
SOURCE = HERE / 'source_manifest.json'
CPU_AUDIT = HERE / 'smoke_plan_cpu_audit.json'
PREPARATION = HERE / 'smoke_preparation.json'
CLAIMS = HERE / 'official_smoke_claim_log.jsonl'
QUEUE_LOG = HERE / 'official_smoke_queue.log'
OFFICIAL = OUT / 'jit_formal_selected/module_deepgemm_opus.so'
PRIVATE = OUT / 'scale_issue_publish/candidate/experiments.so'
PRIVATE_AUDIT = OUT / 'scale_issue_publish/device_audit.json'
RUNNER = OLD / 'official_smoke.py'
EXPERIMENT = OLD / 'experiment_runner.py'
LAUNCHER = OLD / 'owned_python_launch.py'
INVENTORY = OLD / 'shape_inventory.json'
REGISTRY = ROOT / 'csrc/opus_gemm/opus_gemm_common.py'
CODEGEN = ROOT / 'csrc/opus_gemm/codegen/gen_instances_gfx950.py'
REPETITIONS = 8
LABEL = 'scale_issue_publish'
TARGET_CASES = [
    (9021, [480, 7168, 384], 'K384 representative winner; selected9021 API identity'),
    (9021, [1, 128, 128], 'minimum positive M and one K128 tile'),
    (9021, [15, 128, 256], 'M below16 and two K128 tiles'),
    (9021, [17, 128, 384], 'M above16 and three K128 tiles'),
    (9021, [127, 128, 4096], 'M below128 and exact32-tile scale panel'),
    (9021, [129, 128, 4224], 'M above128 and33-tile scale panel refill'),
    (9021, [512, 7168, 16384], 'maximum supported K16384 and128-tile scale panels'),
    (9022, [544, 7168, 384], 'unchanged9022 control; rejected candidate remains absent'),
]
ENV = {
    'AITER_JIT_DIR': str(OFFICIAL.parent), 'AITER_AOT_IMPORT': '1',
    'GPU_ARCHS': 'gfx950', 'CU_NUM': '256',
    'HIP_CLANG_PATH': '/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin',
    'OPUS_HIP_CLANG_PATH': '/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin',
}
OFFSET_CONTRACT = {
    'output_dtype': 'bfloat16', 'element_bytes': 2,
    'output_storage_offset_elements': 128, 'output_storage_offset_bytes': 256,
    'output_offset_alignment_bytes': 16,
    'output_guard_elements_before': 128, 'output_guard_elements_after': 128,
    'split': 1, 'workspace_elements': 0,
    'scope': 'Existing runner guarded output view, inferred from exact runner source SHA; no additional input-offset sweep.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def expected_targets():
    return [{'kid': kid, 'shape': shape, 'seed': 17, 'signed': True,
             'purpose': purpose, 'private_baselines': {LABEL: str(PRIVATE)}}
            for kid, shape, purpose in TARGET_CASES]


def expected_commands():
    result = []
    for index, (kid, _, _) in enumerate(TARGET_CASES):
        name = f'official_smoke_oct8_{index}_kid{kid}'
        output = HERE / 'smoke_results' / (name + '.json')
        result.append({'name': name,
                       'argv': ['/opt/venv/bin/python3', str(RUNNER), '--plan', str(PLAN),
                                '--output', str(output), '--target-index', str(index),
                                '--check-only', '--repetitions', str(REPETITIONS)],
                       'log': str(output.with_suffix('.log'))})
    return result


@lru_cache(maxsize=1)
def inventory_module():
    spec = importlib.util.spec_from_file_location('oct8_shape_inventory_cpu', OLD / 'shape_inventory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_registry():
    module = inventory_module()
    registry, _ = module.load_cpu_registry()
    require('torch' not in sys.modules, 'CPU preparation imported torch')
    return registry


class Elf:
    def __init__(self, data):
        self.data = data
        hdr = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
        require(hdr[0][:6] == b'\x7fELF\x02\x01', 'Expected little-endian ELF64')
        self.machine = hdr[2]
        off, ents, count, strindex = hdr[6], hdr[11], hdr[12], hdr[13]
        self.rows = [struct.unpack_from('<IIQQQQIIQQ', data, off + i * ents) for i in range(count)]
        self.strings = self.bytes(self.rows[strindex])
        self.sections = {self.string(self.strings, r[0]): r for r in self.rows}

    @staticmethod
    def string(table, offset):
        return table[offset:].split(b'\0', 1)[0].decode()

    def bytes(self, row):
        return self.data[row[4]:row[4] + row[5]]

    def symbols(self):
        result = {}
        for row in self.rows:
            if row[1] not in (2, 11):
                continue
            strings = self.bytes(self.rows[row[6]])
            for offset in range(row[4], row[4] + row[5], row[9]):
                name, info, _, section, value, size = struct.unpack_from('<IBBHQQ', self.data, offset)
                if name and section and section < len(self.rows):
                    sec = self.rows[section]
                    begin = sec[4] + value - sec[3]
                    result[self.string(strings, name)] = {'size': size, 'bytes': self.data[begin:begin + size],
                                                          'type': info & 15}
        return result

    def metadata(self):
        for row in self.rows:
            if row[1] != 7:
                continue
            content, offset = self.bytes(row), 0
            while offset + 12 <= len(content):
                namesize, descsize, kind = struct.unpack_from('<III', content, offset)
                offset += 12
                owner = content[offset:offset + namesize].rstrip(b'\0')
                offset += (namesize + 3) & ~3
                desc = content[offset:offset + descsize]
                offset += (descsize + 3) & ~3
                if owner == b'AMDGPU' and kind == 32:
                    return msgpack.unpackb(desc, raw=False)
        raise ValueError('No AMDGPU metadata note')


def device_bundles(path):
    outer = Elf(Path(path).read_bytes())
    bundle = outer.bytes(outer.sections['.hip_fatbin'])
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    cursor, result = 0, []
    while True:
        start = bundle.find(magic, cursor)
        if start < 0:
            break
        count = struct.unpack_from('<Q', bundle, start + len(magic))[0]
        require(0 < count < 1024, 'Invalid HIP bundle target count')
        offset, matches, end = start + len(magic) + 8, [], start
        for _ in range(count):
            begin, size, idlen = struct.unpack_from('<QQQ', bundle, offset)
            offset += 24
            target = bundle[offset:offset + idlen].decode()
            offset += idlen
            require(start + begin + size <= len(bundle), 'Truncated HIP bundle image')
            end = max(end, start + begin + size, offset)
            if target == 'hip-amdgcn-amd-amdhsa--gfx950':
                matches.append(bundle[start + begin:start + begin + size])
        require(len(matches) == 1, f'Expected one gfx950 image per bundle in {path}')
        result.extend(matches)
        cursor = end
    require(result, f'No gfx950 bundle in {path}')
    return result


def device_rows(path):
    bundles = device_bundles(path)
    rows = [row for data in bundles for row in summarize_image(data)]
    # One-image private libraries retain the exact device image SHA. The
    # official linked image set hashes concatenated images in bundle order.
    return hashlib.sha256(b''.join(bundles)).hexdigest(), rows


def summarize_image(data):
    elf = Elf(data)
    require(elf.machine == 224, 'Expected AMDGPU ELF')
    metadata, symbols = elf.metadata(), elf.symbols()
    names = [k['.name'] for k in metadata['amdhsa.kernels']]
    demangles = subprocess.check_output(['c++filt', *names], text=True).splitlines()
    require(len(names) == len(set(names)) == len(demangles), 'Duplicate device symbols or demangle failure')
    rows = []
    for kernel, demangle in zip(metadata['amdhsa.kernels'], demangles):
        code, descriptor = symbols[kernel['.name']], symbols[kernel['.symbol']]['bytes']
        require(code['type'] == 2 and len(descriptor) == 64, 'Invalid FUNC symbol or kernel descriptor')
        normalized = bytearray(descriptor)
        normalized[16:24] = b'\0' * 8
        rows.append({'name': kernel['.name'], 'demangled': demangle, 'metadata': kernel,
                     'instruction_bytes': code['size'],
                     'instruction_sha256': hashlib.sha256(code['bytes']).hexdigest(),
                     'descriptor_sha256': hashlib.sha256(descriptor).hexdigest(),
                     'descriptor_normalized_sha256': hashlib.sha256(normalized).hexdigest()})
    return rows


def identity(a, b):
    checks = {'symbol_equal': a['name'] == b['name'],
              'instruction_equal': a['instruction_sha256'] == b['instruction_sha256'],
              'instruction_bytes_equal': a['instruction_bytes'] == b['instruction_bytes'],
              'metadata_equal': a['metadata'] == b['metadata'],
              'descriptor_normalized_equal': a['descriptor_normalized_sha256'] == b['descriptor_normalized_sha256']}
    checks['matches'] = all(checks.values())
    return checks


def offset_contract():
    # Verify exact unchanged source expressions without importing the GPU runner.
    source = EXPERIMENT.read_text()
    require('guard = torch.full((m*n+256,),42,device="cuda",dtype=torch.bfloat16)' in source
            and 'data["out"] = guard[128:-128].view(m,n)' in source,
            'guarded output offset implementation differs')
    require('offset=tensor.storage_offset()' in source
            and 'backing[offset-128:offset]==value' in source
            and 'backing[offset+tensor.numel():offset+tensor.numel()+128]==value' in source,
            'actual backing-storage guard checks differ')
    require(OFFSET_CONTRACT['output_storage_offset_bytes'] % 16 == 0, 'output offset alignment differs')
    return OFFSET_CONTRACT


def validate_prepared(plan, queue, require_build=True):
    require(plan['targets'] == expected_targets(), 'Expected exactly eight ordered signed seed17 targets')
    require(queue['commands'] == expected_commands() and queue['env'] == ENV, 'Queue command/environment differs')
    require(queue['claim_log'] == str(CLAIMS) and queue['queue_log'] == str(QUEUE_LOG), 'Claim/log isolation differs')
    require(plan['official_binary'] == str(OFFICIAL), 'Official binary path differs')
    require(plan['source_head'] == head() and plan['current_parent_count'] == 26, 'Source HEAD or parent pool differs')
    require(plan['offset_contract'] == offset_contract(), 'Offset contract differs')
    require(plan['repetitions_per_label_and_target'] == REPETITIONS and plan['check_only'] is True,
            'Check-only numerical protocol differs')
    registry = cpu_registry()
    require(len(registry.A8W8_BPRESHUFFLE_TUNING_KIDS) == 26, 'Current registry pool differs')
    for target in plan['targets']:
        kid, (m, n, k) = target['kid'], target['shape']
        instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        require(registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k)
                and instance.bpreshuffle_split_k == 1, f'Unsupported shape or split: {target}')
        require(m > 0 and n % 128 == 0 and k % 128 == 0 and 128 <= k <= 16384
                and (kid == 9021 or m % 16 == 0), 'Private scalar launch contract differs')
    if require_build:
        build, source = read(BUILD), read(SOURCE)
        require(build['status'] == 'passed' and build['pending_candidate_no_adoption'] is True, 'Formal build not passed')
        require(build['binary'] == str(OFFICIAL) and build['binary_sha256'] == plan['official_binary_sha256'] == sha(OFFICIAL),
                'Official built/current/plan SHA differs')
        require(build['source_head'] == source['source_head'] == plan['source_head'], 'Formal build/source HEAD differs')
        require(build['source_root'] == source['candidate_worktree'], 'Formal source root differs')
        require(set(build['kids']) == set(registry.A8W8_BPRESHUFFLE_TUNING_KIDS), 'Formal build parents differ')
        require(build['visibility'] == {'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '', 'CUDA_VISIBLE_DEVICES': ''}
                and build['gpu_execution_requested'] is False and build['imported_tune'] is None,
                'Formal build CPU-only contract differs')
        require(plan['status'] == queue['status'] == 'prepared_sealed_pending_cpu_identity_audit'
                and queue['ready_for_gpu_execution'] is False, 'Preparation gate status differs')
        require(plan['build_manifest_sha256'] == sha(BUILD) and plan['source_manifest_sha256'] == sha(SOURCE),
                'Sealed build or source manifest changed')
    return registry
