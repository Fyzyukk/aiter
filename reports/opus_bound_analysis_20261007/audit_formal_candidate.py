#!/usr/bin/env python3
"""CPU identity audit of a pending isolated official JIT against baseline/private.

Checks all 26 public Opus parents, unchanged device instructions/metadata,
the five changed entries against the tested private candidates, and all ELF
objects from the official build. Does not load a HIP module or call a GPU.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HERE = OUT / 'formal_candidate'
BASE = OUT / 'jit_baseline/build/module_deepgemm_opus'
CAND = OUT / 'jit_formal_candidate/build/module_deepgemm_opus'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location('elf_audit_cpu', OUT / 'audit_current_compute_metadata.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def image(path):
    return extract.device_bundle(path)


def rows(path):
    return extract.summarize_image(image(path))[1]


def canonical_metadata(metadata):
    return {k: v for k, v in metadata.items() if k not in ('.name', '.symbol')}


def identity(a, b, allow_symbol=False):
    return {'instruction_equal': a['instruction_sha256'] == b['instruction_sha256'],
            'metadata_equal': (canonical_metadata(a['metadata']) == canonical_metadata(b['metadata'])) if allow_symbol else a['metadata'] == b['metadata'],
            'descriptor_normalized_equal': a['descriptor_normalized_sha256'] == b['descriptor_normalized_sha256'],
            'symbol_equal': a['name'] == b['name']}


def all_matched(check):
    return check['instruction_equal'] and check['metadata_equal'] and check['descriptor_normalized_equal']


def main():
    build_path = HERE / 'build_manifest.json'
    manifest = json.loads(build_path.read_text())
    source_path = HERE / 'source_manifest.json'
    sources = json.loads(source_path.read_text())
    assert manifest['status'] == 'passed' and manifest['pending_candidate_no_adoption']
    assert manifest['visibility'] == {'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '', 'CUDA_VISIBLE_DEVICES': ''}
    candidate_root = Path(sources['candidate_worktree'])
    assert Path(manifest['source_root']) == candidate_root
    assert manifest['source_head'] == sources['source_head']
    assert Path(manifest['imported_core']).is_relative_to(candidate_root) and manifest['imported_tune'] is None
    assert sha(Path(manifest['binary'])) == manifest['binary_sha256']
    baseline_manifest = json.loads((OUT / 'current_build.json').read_text())
    assert sha(Path(baseline_manifest['binary'])) == baseline_manifest['binary_sha256']
    for f in sources['files']:
        assert sha(ROOT / f['path']) == f['original_sha256']
        assert sha(candidate_root / f['path']) == f['candidate_sha256'] == sha(Path(f['private_candidate']))
    actual_diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=candidate_root, text=True)
    assert hashlib.sha256(actual_diff.encode()).hexdigest() == sources['source_diff_sha256']
    changed_files = subprocess.check_output(['git', 'diff', '--name-only'], cwd=candidate_root, text=True).splitlines()
    assert set(changed_files) == {f['path'] for f in sources['files']}
    resources = json.loads((OUT / 'shape_inventory_evidence/official_resources.json').read_text())
    prologue = {side: rows(OUT / 'compute_prologue' / side / 'experiments.so') for side in ['baseline', 'candidate']}
    register = {side: rows(OUT / 'register_reuse_scoped' / side / 'experiments.so') for side in ['baseline', 'candidate']}
    prologue_names = {r['name'] for r in prologue['baseline']}
    register_base = {r['name']: r for r in register['baseline']}
    register_candidate = {r['name']: r for r in register['candidate']}
    changed_register_names = set(register_base) - set(register_candidate)
    assert len(changed_register_names) == 3
    parents = []
    failures = []
    changed_pairs = set()
    # Formal baseline/current inventory already maps each public parent object.
    for kid_text, original in resources.items():
        kid = int(kid_text)
        base_object = Path(original['object'])
        candidate_object = CAND / 'build' / base_object.name
        assert base_object.is_file() and candidate_object.is_file()
        base_rows = rows(base_object)
        cand_rows = rows(candidate_object)
        base_by_name = {r['name']: r for r in base_rows}
        cand_by_name = {r['name']: r for r in cand_rows}
        assert len(base_rows) == len(original['variants']) == len(cand_rows)
        prefix = base_object.name.removesuffix('_Cbf16_t.device.cuda.o')
        base_impl = BASE / 'blob.staging/impl' / (prefix + '.cuh')
        cand_impl = CAND / 'blob.staging/impl' / (prefix + '.cuh')
        base_tu = BASE / 'blob.staging/instances' / (prefix + '_Cbf16_t.device.cu')
        cand_tu = CAND / 'blob.staging/instances' / (prefix + '_Cbf16_t.device.cu')
        assert base_impl.read_bytes() == cand_impl.read_bytes()
        assert base_tu.read_bytes() == cand_tu.read_bytes()
        variants = []
        for base_name, base in base_by_name.items():
            expected_kargs = 20 if 'opus_gemm_mxscale_bpreshuffle_reduce_kernel' in base_name else 96
            assert base['metadata']['.kernarg_segment_size'] == expected_kargs
            if base_name in prologue_names:
                changed = True
                expected = next(r for r in prologue['candidate'] if r['name'] == base_name)
                candidate = cand_by_name[base_name]
                private_baseline = next(r for r in prologue['baseline'] if r['name'] == base_name)
                assert all_matched(identity(base, private_baseline))
                mechanism = 'prologue'
            elif base_name in changed_register_names:
                changed = True
                # Only the explicit ReuseBScale=false final bool becomes true.
                candidate_name = base_name.replace('ELb0ELb0EEEv', 'ELb0ELb1EEEv')
                assert candidate_name != base_name and candidate_name in cand_by_name
                expected = register_candidate[candidate_name]
                candidate = cand_by_name[candidate_name]
                assert all_matched(identity(base, register_base[base_name]))
                mechanism = 'scoped register ReuseBScale'
            else:
                changed = False
                expected = base
                candidate = cand_by_name[base_name]
                mechanism = 'unchanged'
            check = identity(candidate, expected)
            check['matches_expected'] = all_matched(check)
            if not check['matches_expected']:
                failures.append({'parent_id': kid, 'name': candidate['name'], 'mechanism': mechanism, 'checks': check})
            if changed:
                changed_pairs.add((base_name, candidate['name']))
            variants.append({'changed': changed, 'mechanism': mechanism,
                             'baseline': base, 'candidate': candidate, 'expected_private_or_baseline': expected,
                             'identity_checks': check})
        parents.append({'parent_id': kid, 'baseline_object': str(base_object), 'baseline_object_sha256': sha(base_object),
                        'candidate_object': str(candidate_object), 'candidate_object_sha256': sha(candidate_object),
                        'exact_object_sha_equal': sha(base_object) == sha(candidate_object),
                        'generated_tu_equal': True, 'generated_impl_host_launch_equal': True,
                        'generated_tu_sha256': sha(cand_tu), 'generated_impl_sha256': sha(cand_impl),
                        'variants': variants})
    assert len(parents) == 26 and len(changed_pairs) == 5
    # Complete object audit also covers shared reducer and non-9000 default
    # compiled kernels. Host objects can include source paths/symbol changes.
    object_checks = []
    for base in sorted((BASE / 'build').glob('*.cuda.o')):
        candidate = CAND / 'build' / base.name
        if not candidate.is_file():
            failures.append({'object': base.name, 'reason': 'candidate object missing'})
            continue
        check = {'object': base.name, 'baseline_sha256': sha(base), 'candidate_sha256': sha(candidate),
                 'exact_object_sha_equal': sha(base) == sha(candidate)}
        try:
            b_rows, c_rows = rows(base), rows(candidate)
        except (KeyError, ValueError, AssertionError):
            check['device_bundle_present'] = False
            object_checks.append(check)
            continue
        check['device_bundle_present'] = True
        check['baseline_kernels'] = len(b_rows)
        check['candidate_kernels'] = len(c_rows)
        bmap = {r['name']: r for r in b_rows}
        cmap = {r['name']: r for r in c_rows}
        ignored = {a for a, b in changed_pairs}
        introduced = {b for a, b in changed_pairs}
        unchanged_names = set(bmap) - ignored
        check['unchanged_kernel_count'] = len(unchanged_names)
        check['unexpected_removed_symbols'] = sorted(set(bmap) - set(cmap) - ignored)
        check['unexpected_added_symbols'] = sorted(set(cmap) - set(bmap) - introduced)
        check['unchanged_identity_equal'] = all(name in cmap and all_matched(identity(cmap[name], bmap[name])) for name in unchanged_names)
        if check['unexpected_removed_symbols'] or check['unexpected_added_symbols'] or not check['unchanged_identity_equal']:
            failures.append({'object': base.name, 'reason': 'unexpected device identity change', 'checks': check})
        object_checks.append(check)
    report = {'status': 'passed_pending_cpu_identity_no_adoption' if not failures else 'identity_failed_requires_review',
              'cpu_only': True, 'gpu_executed': False, 'production_checkout_modified': False,
              'source_head': sources['source_head'], 'candidate_source_root': str(candidate_root),
              'candidate_binary': manifest['binary'], 'candidate_binary_sha256': manifest['binary_sha256'],
              'baseline_binary_sha256_unchanged': baseline_manifest['binary_sha256'],
              'build_manifest_sha256': sha(build_path), 'source_manifest_sha256': sha(source_path),
              'changed_source_files': changed_files, 'source_diff_sha256': sources['source_diff_sha256'],
              'parents': parents, 'all_build_object_checks': object_checks, 'failures': failures,
              'summary': {'public_parents': len(parents), 'actual_device_variants': sum(len(p['variants']) for p in parents),
                          'changed_device_entries': len(changed_pairs),
                          'changed_entries_match_tested_private_candidate': all(v['identity_checks']['matches_expected'] for p in parents for v in p['variants'] if v['changed']),
                          'unchanged_entries_match_official_baseline': all(v['identity_checks']['matches_expected'] for p in parents for v in p['variants'] if not v['changed']),
                          'generated_host_launch_and_tus_equal': all(p['generated_tu_equal'] and p['generated_impl_host_launch_equal'] for p in parents),
                          'build_objects_checked': len(object_checks),
                          'exact_object_sha_equal_count': sum(c['exact_object_sha_equal'] for c in object_checks),
                          'device_objects_checked': sum(c['device_bundle_present'] for c in object_checks),
                          'failure_count': len(failures)},
              'limits': ['No official candidate HIP module has been loaded or run on GPU; candidate API smoke and actual loaded module SHA remain required.',
                         'Whole host/device object SHA may differ due to worktree source paths, host symbols and TU placement; instruction/metadata/normalized descriptor comparisons separately establish unchanged device identity.',
                         'Normalization clears only kernel_code_entry_byte_offset bytes16..23.',
                         'CPU compilation prepares a reviewable pending change. Numerical and Event support outcomes decide adoption.']}
    path = HERE / 'identity_audit.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(path), 'summary': report['summary'], 'failures': failures}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
