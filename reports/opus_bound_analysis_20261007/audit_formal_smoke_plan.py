#!/usr/bin/env python3
"""CPU audit of pending 15-target official API smoke branch/private identities."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
from shape_inventory import cpp_eval, load_cpu_registry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    spec = importlib.util.spec_from_file_location('formal_audit_cpu', OUT / 'audit_formal_candidate.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    registry, _ = load_cpu_registry()
    plan_path = OUT / 'plans/formal_candidate_smoke.json'
    queue_path = OUT / 'formal_candidate_smoke_queue.json'
    plan, queue = read(plan_path), read(queue_path)
    inventory = read(OUT / 'shape_inventory.json')
    formal = read(OUT / 'formal_candidate/identity_audit.json')
    parents = {p['parent_id']: p for p in formal['parents']}
    assert len(plan['targets']) == len(queue['commands']) == 15
    assert plan['current_parent_count'] == len(registry.A8W8_BPRESHUFFLE_TUNING_KIDS) == 26
    assert plan['source_head'] == formal['source_head']
    assert plan['official_binary'] == formal['candidate_binary']
    assert sha(Path(plan['official_binary'])) == plan['official_binary_sha256'] == formal['candidate_binary_sha256']
    assert Path(queue['env']['AITER_JIT_DIR']).resolve() == Path(plan['official_binary']).parent.resolve()
    rows = []
    changed_seen = set()
    for index, (target, command) in enumerate(zip(plan['targets'], queue['commands'])):
        kid = target['kid']; m, n, k = target['shape']
        instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        assert registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k)
        assert target['signed'] and target['seed'] == 17
        matches = [b for b in inventory['parents'][str(kid)]['branches']
                   if all(cpp_eval(c, m, n, k) for c in b['conditions'])]
        assert len(matches) == 1
        branch = matches[0]
        # Device kernel order in the official TU is not necessarily host leaf
        # order. Match the baseline demangled trait type retained by inventory.
        traits = branch['traits_cpp'].replace(' ', '')
        if '<' not in traits and 'small_register' in traits:
            # Named aliases are expanded in the ELF symbol. Resolve the
            # retained baseline source, including omitted default arguments.
            header = (OUT / 'register_reuse_scoped/baseline/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh').read_text()
            match = re.search(r'using\s+' + re.escape(traits) + r'\s*=\s*opus_gemm_small_register_traits_gfx950<([^;]+)>;', header)
            assert match
            values = [v.strip() for v in match.group(1).split(',')]
            values += ['0'] * (9 - len(values)) + ['false', 'false']
            traits = 'opus_gemm_small_register_traits_gfx950<' + ','.join(values) + '>'
        candidates = [v for v in parents[kid]['variants']
                      if traits in v['baseline']['demangled'].replace(' ', '')]
        assert len(candidates) == 1, (kid, traits, [v['baseline']['demangled'] for v in parents[kid]['variants']])
        actual = candidates[0]
        if actual['changed']:
            changed_seen.add(actual['candidate']['name'])
        private_checks = []
        for label, path in target.get('private_baselines', {}).items():
            private = Path(path)
            private_rows = audit.rows(private)
            expected = [r for r in private_rows if r['name'] == actual['candidate']['name']]
            assert len(expected) == 1, (kid, target['shape'], label, 'wrong private branch')
            checks = audit.identity(actual['candidate'], expected[0])
            assert audit.all_matched(checks)
            if label == 'compute_prologue':
                assert n % 128 == 0 and k % 128 == 0 and 128 <= k <= 16384
                assert m > 0 and (kid == 9021 or m % 16 == 0)
            elif label == 'register_scoped':
                assert m <= 512 and n % 128 == 0 and k % 128 == 0 and k <= 16384
                assert not (kid in [9042, 9053] and k == 7168)
            else:
                raise AssertionError('unexpected private label')
            private_checks.append({'label': label, 'path': str(private), 'sha256': sha(private),
                                   'device_identity': checks, 'private_scalar_launch_contract_supported': True})
        argv = command['argv']
        assert Path(argv[1]) == OUT / 'official_smoke.py'
        assert argv[argv.index('--plan') + 1] == str(plan_path)
        assert argv[argv.index('--target-index') + 1] == str(index)
        assert argv[argv.index('--repetitions') + 1] == '8' and '--check-only' in argv
        rows.append({'target_index': index, 'parent_id': kid, 'shape': target['shape'],
                     'support_contract_passed': True, 'actual_branch_index': branch['index'],
                     'actual_conditions': branch['conditions'], 'actual_baseline_traits': branch['traits_cpp'],
                     'actual_candidate_symbol': actual['candidate']['name'], 'changed': actual['changed'],
                     'private_checks': private_checks, 'command_protocol': 'check-only 8rep'})
    assert len(changed_seen) == 5
    assert 'torch' not in sys.modules
    report = {'status': 'passed_cpu_smoke_plan_identity_no_gpu_execution', 'new_gpu_execution': False,
              'production_sources_modified': False, 'plan_sha256': sha(plan_path), 'queue_sha256': sha(queue_path),
              'formal_identity_audit_sha256': sha(OUT / 'formal_candidate/identity_audit.json'),
              'target_count': 15, 'changed_entries_covered': 5, 'rows': rows,
              'limits': ['CPU support and exact private branch identity only; real official API correctness remains unexecuted.',
                         'Current unchanged K7168 control branches intentionally have no mismatching runtime private launcher.',
                         'The existing official smoke script verifies actual loaded module path and SHA at first launch.',
                         'These rank2 calls match the public GEMM-only family; public opus_bmm rejects this family, so rank3 is not a required positive API gate.',
                         'Explicit 16B storage offsets preserve unchanged data_ptr alignment and relative addressing; the three-file device changes do not require an additional offset gate.']}
    path = OUT / 'formal_candidate/smoke_plan_cpu_audit.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(path), 'targets': 15, 'changed_entries': 5}))


if __name__ == '__main__':
    main()
