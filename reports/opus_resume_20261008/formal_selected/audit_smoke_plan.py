#!/usr/bin/env python3
"""CPU-only audit of eight planned API calls and exact linked private identities."""
import json
import sys

from smoke_common import (BUILD, CODEGEN, CPU_AUDIT, EXPERIMENT, INVENTORY, LABEL, LAUNCHER,
                          OFFICIAL, PLAN, PRIVATE, PRIVATE_AUDIT, QUEUE, REGISTRY, RUNNER,
                          SOURCE, device_rows, identity, inventory_module, offset_contract,
                          read, require, sha, validate_prepared, write)


def audit_plan():
    plan, queue = read(PLAN), read(QUEUE)
    validate_prepared(plan, queue)
    sealed_files = {'registry_sha256': REGISTRY, 'codegen_sha256': CODEGEN,
                    'inventory_sha256': INVENTORY, 'runner_sha256': RUNNER,
                    'experiment_runner_sha256': EXPERIMENT, 'owner_launcher_sha256': LAUNCHER,
                    'private_device_audit_sha256': PRIVATE_AUDIT, 'private_binary_sha256': PRIVATE}
    for key, path in sealed_files.items():
        require(plan[key] == sha(path), f'Sealed file changed: {path}')
    private_audit = read(PRIVATE_AUDIT)
    require(private_audit['status'] == 'passed' and private_audit['cpu_only'] is True
            and private_audit['sources_unchanged'] is True, 'Private device audit not passed')
    recorded = [lib for lib in private_audit['libraries'] if lib['label'] == 'candidate']
    require(len(recorded) == 1 and recorded[0]['input'] == str(PRIVATE)
            and recorded[0]['input_sha256'] == sha(PRIVATE), 'Private audited input differs')
    official_device_sha, official_rows = device_rows(OFFICIAL)
    private_device_sha, private_rows = device_rows(PRIVATE)
    require(private_device_sha == recorded[0]['device_sha256'], 'Private device bundle changed')
    private_by_name = {row['name']: row for row in private_rows}
    recorded_by_kid = {row['kid']: row for row in recorded[0]['kernels']}
    require(set(recorded_by_kid) == {9021, 9022} and len(private_rows) == 2,
            'Private library must contain exactly9021 and9022')
    device_entries = {}
    for kid, recorded_kernel in recorded_by_kid.items():
        name = recorded_kernel['name']
        official_matches = [row for row in official_rows if row['name'] == name]
        require(name in private_by_name and len(official_matches) == 1, f'Exact selected kernel missing/ambiguous: {kid}')
        private = private_by_name[name]
        require(identity(private, recorded_kernel)['matches'], f'Private recorded kernel identity differs: {kid}')
        official = official_matches[0]
        checks = identity(official, private)
        require(checks['matches'], f'Official linked kernel differs from private candidate: {kid}')
        require(official['metadata']['.kernarg_segment_size'] == 96
                and official['metadata']['.wavefront_size'] == 64
                and official['metadata']['.max_flat_workgroup_size'] == 256,
                f'Expected scalar native E8M0 ABI differs: {kid}')
        device_entries[kid] = {'parent_id': kid, 'changed': kid == 9021,
                               'official': official, 'private': private, 'identity': checks}
    control = [check for check in private_audit['required_identity_comparisons']
               if check['label'] == 'candidate' and check['kid'] == 9022]
    require(len(control) == 1 and control[0]['identity']['matches'] is True,
            'Unchanged9022 private candidate control identity missing')
    inventory = read(INVENTORY)
    require(inventory['registry_sha256'] == sha(REGISTRY) and inventory['codegen_sha256'] == sha(CODEGEN),
            'Retained inventory no longer matches current scalar codegen')
    rows = []
    for index, target in enumerate(plan['targets']):
        kid, (m, n, k) = target['kid'], target['shape']
        parent = inventory['parents'][str(kid)]
        matches = [branch for branch in parent['branches']
                   if all(inventory_module().cpp_eval(condition, m, n, k) for condition in branch['conditions'])]
        require(len(matches) == 1 and parent['codegen_launch_body_identical'] is True,
                f'Ambiguous/unverified scalar branch for target{index}')
        branch, actual = matches[0], device_entries[kid]
        require(branch['kernel_function'] in actual['official']['demangled']
                and branch['traits_cpp'].replace(' ', '') in actual['official']['demangled'].replace(' ', ''),
                f'Plan branch differs from exact linked trait type for target{index}')
        rows.append({'target_index': index, 'parent_id': kid, 'shape': target['shape'],
                     'signed': True, 'seed': 17, 'split': 1, 'changed': kid == 9021,
                     'support_contract_passed': True, 'actual_branch_index': branch['index'],
                     'actual_conditions': branch['conditions'], 'actual_traits': branch['traits_cpp'],
                     'actual_candidate_symbol': actual['official']['name'],
                     'private_checks': [{'label': LABEL, 'path': str(PRIVATE), 'sha256': sha(PRIVATE),
                                         'device_identity': actual['identity'],
                                         'private_scalar_launch_contract_supported': True}],
                     'command_protocol': 'check-only eight repetitions',
                     'offset_contract': offset_contract()})
    require('torch' not in sys.modules, 'CPU plan audit imported torch')
    return {'status': 'passed_cpu_smoke_plan_identity_no_gpu_execution',
            'cpu_only': True, 'new_gpu_execution': False, 'production_sources_modified': False,
            'plan_sha256': sha(PLAN), 'queue_sha256': sha(QUEUE),
            'build_manifest_sha256': sha(BUILD), 'source_manifest_sha256': sha(SOURCE),
            'private_device_audit_sha256': sha(PRIVATE_AUDIT),
            'runner_sha256': sha(RUNNER), 'experiment_runner_sha256': sha(EXPERIMENT),
            'owner_launcher_sha256': sha(LAUNCHER), 'registry_sha256': sha(REGISTRY),
            'codegen_sha256': sha(CODEGEN), 'inventory_sha256': sha(INVENTORY),
            'official_module': {'path': str(OFFICIAL), 'sha256': sha(OFFICIAL)},
            'official_device_sha256': official_device_sha,
            'private_module': {'path': str(PRIVATE), 'sha256': sha(PRIVATE)},
            'private_device_sha256': private_device_sha,
            'target_count': 8, 'changed_entries_covered': 1, 'unchanged_entries_covered': 1,
            'device_entries': [device_entries[kid] for kid in [9021, 9022]], 'rows': rows,
            'offset_contract': offset_contract(),
            'limits': ['CPU support and linked/private device identity only; API correctness remains unexecuted.',
                       'Descriptor normalization clears only entry-offset bytes16..23.',
                       'Signed seed17 rank2/native E8M0 API with guarded output offset256 bytes; no input-offset sweep.',
                       'Seven9021 shapes and one9022 control give partial support-domain coverage.']}


def main():
    report = audit_plan()
    write(CPU_AUDIT, report)
    print(json.dumps({'status': report['status'], 'output': str(CPU_AUDIT),
                      'targets': report['target_count'], 'official_sha256': report['official_module']['sha256']}))


if __name__ == '__main__':
    main()
