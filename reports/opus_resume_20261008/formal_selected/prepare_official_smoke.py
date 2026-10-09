#!/usr/bin/env python3
"""Prepare and seal eight API correctness commands without GPU execution."""
from datetime import datetime, timezone
import sys

from smoke_common import (BUILD, CODEGEN, CPU_AUDIT, ENV, EXPERIMENT, HERE, INVENTORY,
                          LAUNCHER, OFFICIAL, PLAN, PREPARATION, PRIVATE, PRIVATE_AUDIT,
                          QUEUE, QUEUE_LOG, CLAIMS, REGISTRY, REPETITIONS, RUNNER, SOURCE,
                          expected_commands, expected_targets, head, offset_contract,
                          read, require, sha, validate_prepared, write)


def main():
    private = read(PRIVATE_AUDIT)
    require(private['status'] == 'passed' and private['cpu_only'] is True
            and private['sources_unchanged'] is True, 'Private device audit not passed')
    candidates = [row for row in private['libraries'] if row['label'] == 'candidate']
    require(len(candidates) == 1 and candidates[0]['input'] == str(PRIVATE)
            and candidates[0]['input_sha256'] == sha(PRIVATE), 'Private candidate SHA differs')
    complete_build = BUILD.exists() and read(BUILD).get('status') == 'passed'
    build = read(BUILD) if complete_build else {}
    status = ('prepared_sealed_pending_cpu_identity_audit' if complete_build
              else 'prepared_pending_official_build')
    plan = {'status': status, 'source_head': head(), 'official_binary': str(OFFICIAL),
            'official_binary_sha256': build.get('binary_sha256'), 'current_parent_count': 26,
            'build_manifest_sha256': sha(BUILD) if complete_build else None,
            'source_manifest_sha256': sha(SOURCE) if SOURCE.exists() else None,
            'private_device_audit_sha256': sha(PRIVATE_AUDIT), 'private_binary_sha256': sha(PRIVATE),
            'registry_sha256': sha(REGISTRY), 'codegen_sha256': sha(CODEGEN),
            'inventory_sha256': sha(INVENTORY), 'runner_sha256': sha(RUNNER),
            'experiment_runner_sha256': sha(EXPERIMENT), 'owner_launcher_sha256': sha(LAUNCHER),
            'check_only': True, 'repetitions_per_label_and_target': REPETITIONS,
            'offset_contract': offset_contract(),
            'selection': 'Seven selected9021 boundary/representative shapes plus one unchanged9022 control.',
            'targets': expected_targets()}
    queue = {'status': status, 'ready_for_gpu_execution': False,
             'execution_prerequisites': ['Passed formal linked-device identity audit.',
                                         f'Passed CPU smoke plan audit: {CPU_AUDIT}',
                                         'Root physical idle-device lock with unique owner handshake.'],
             'claim_log': str(CLAIMS), 'queue_log': str(QUEUE_LOG),
             'env': ENV, 'commands': expected_commands()}
    validate_prepared(plan, queue, require_build=complete_build)
    require('torch' not in sys.modules, 'CPU preparation imported torch')
    write(PLAN, plan)
    write(QUEUE, queue)
    (HERE / 'smoke_results').mkdir(exist_ok=True)
    report = {'status': status, 'prepared_utc': datetime.now(timezone.utc).isoformat(),
              'cpu_only': True, 'new_gpu_execution': False, 'production_sources_modified': False,
              'plan': str(PLAN), 'plan_sha256': sha(PLAN), 'queue': str(QUEUE), 'queue_sha256': sha(QUEUE),
              'official_module': {'path': str(OFFICIAL), 'sha256': plan['official_binary_sha256']},
              'build_manifest_sha256': plan['build_manifest_sha256'],
              'source_manifest_sha256': plan['source_manifest_sha256'],
              'private_binary_sha256': plan['private_binary_sha256'],
              'target_count': 8, 'repetitions_per_label_and_target': REPETITIONS,
              'expected_numerical_calls_per_label': {'official': 64, 'scale_issue_publish': 64},
              'expected_total_numerical_calls': 128, 'offset_contract': plan['offset_contract'],
              'claim_log': str(CLAIMS), 'queue_log': str(QUEUE_LOG),
              'limits': ['Preparation does not execute the queue; formal CPU identity and physical claims remain required.',
                         'Check-only API/numerical evidence has no timing or performance claim.',
                         'The exact unchanged runner uses a guarded BF16 output view at256 bytes; input offsets are not swept.']}
    write(PREPARATION, report)
    print({'status': status, 'plan': str(PLAN), 'queue': str(QUEUE), 'targets': 8})


if __name__ == '__main__':
    main()
