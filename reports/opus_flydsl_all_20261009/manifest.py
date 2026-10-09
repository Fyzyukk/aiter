#!/usr/bin/env python3
"""Bind offline artifacts or verify their hashes without loading any libraries."""
import argparse
import json
from pathlib import Path

from aggregate import FORMAL, HERE, ROOT, sha, write_json


def files():
    return {str(p.relative_to(HERE)): sha(p)
            for p in sorted(HERE.rglob('*'))
            if p.is_file() and p != HERE / 'artifact_manifest.json'
            and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads((HERE / 'artifact_manifest.json').read_text())
        for path, digest in manifest['files'].items():
            assert sha(HERE / path) == digest, path
        assert set(files()) == set(manifest['files']), 'New or removed unbound artifact'
        for path, digest in manifest['external_inputs'].items():
            assert sha(ROOT / path) == digest, path
        print(json.dumps(dict(status='artifact_hashes_verified', files=len(manifest['files']))))
        return
    inputs = dict(FORMAL)
    for path in ['HANDOFF_MXFP8.md', 'reports/opus_flydsl_gap_20261009/losers294.csv',
                 'reports/opus_clang23_mixed_retune_20261008/profile.csv',
                 'reports/opus_flydsl_resume_20261009/historical_745_best_per_shape.csv',
                 'csrc/opus_gemm/opus_gemm_common.py',
                 'csrc/opus_gemm/codegen/gen_instances_gfx950.py']:
        inputs[path] = sha(ROOT / path)
    for path, digest in FORMAL.items():
        assert sha(ROOT / path) == digest, path
    write_json(HERE / 'artifact_manifest.json', dict(
        status='all_families_offline_artifacts_bound_gpu_validation_pending',
        gpu_tests='stopped_by_user', gpu_operations=0, registered=False,
        numerical_validation='not_run_gpu_stopped', performance_validation='not_run_gpu_stopped',
        scope='All report files including failed attempts; success is determined only by latest receipts and reviews.',
        external_inputs=inputs, files=files()))
    print(json.dumps(dict(status='artifact_manifest_written',
                          sha256=sha(HERE / 'artifact_manifest.json'))))


if __name__ == '__main__':
    main()
