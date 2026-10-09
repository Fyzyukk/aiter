#!/usr/bin/env python3
"""Aggregate existing offline receipts. Standard library only; no GPU operations."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGES = ('main_variants', 'hybrid_small', 'small_split', 'large9030')
FORMAL = {
    'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv':
        '3c54663129807c1ff264b4be3c1265b75bc603b85f87b252897759a84b3f0f01',
    'reports/opus_clang23_mixed_retune_20261008/jit/module_deepgemm_opus.so':
        '08201bc0f5c76b6d0449492661479bfae5e5d456bc967d2f09abcb1c0a6dc7c0',
    'reports/opus_flydsl_comparison_20261008/upstream_mxscale_main.csv':
        '97974cf93a11b71834770acd44a2a7a3ab1342e7572f4ca45c2da989b0d883cb',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    assert rows
    fields = list(rows[0])
    assert all(set(x) == set(fields) for x in rows)
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shape(row):
    return tuple(int(row[k]) for k in ('M', 'N', 'K'))


def main():
    for path, digest in FORMAL.items():
        assert sha(ROOT / path) == digest, path
    original_path = ROOT / 'reports/opus_flydsl_gap_20261009/losers294.csv'
    original = read_csv(original_path)
    by_shape = {shape(x): x for x in original}
    assert len(original) == len(by_shape) == 294
    assert len({x['opus_parent_kid'] for x in original}) == 26
    final_review = read_json(HERE / 'total_read_only_review.json')
    assert final_review['status'] == 'read_only_final_package_verification_passed'
    assert not final_review['issues'] and final_review['current_kernel_count'] == 114
    for path, digest in final_review['review_input_sha256'].items():
        p = Path(path)
        if not p.is_absolute():
            p = HERE / p
        assert sha(p) == digest, p

    manifests = {}
    for package in PACKAGES:
        directory = HERE / package
        manifest = read_json(directory / 'final_manifest.json')
        for path, digest in manifest['files'].items():
            assert sha(directory / path) == digest, (package, path)
        manifests[package] = {'sha256': sha(directory / 'final_manifest.json'),
                              'bound_files': len(manifest['files'])}

    cases = []
    ownership = {}

    def add(package, row, candidate, details):
        key = shape(row)
        assert key in by_shape
        historical = by_shape[key]
        parent = int(row.get('parent', row.get('opus_parent_kid')))
        actual = int(row.get('actual_kid', historical['actual_kid']))
        assert parent == int(historical['opus_parent_kid'])
        assert actual == int(historical['actual_kid'])
        assert key not in ownership or ownership[key] == package
        ownership[key] = package
        cases.append(dict(M=key[0], N=key[1], K=key[2], package=package,
                          opus_parent_kid=parent, actual_kid=actual,
                          candidate_key=f'{package}:{candidate}',
                          offline_compile_link='passed', cpu_layout_contract_audit='passed',
                          numerical_validation='not_run_gpu_stopped',
                          performance_validation='not_run_gpu_stopped', registered=False,
                          details_json=json.dumps(details, sort_keys=True)))

    main_variants = read_json(HERE / 'main_variants/variants.json')
    variants_by_id = {x['id']: x for x in main_variants}
    for row in read_csv(HERE / 'main_variants/coverage.csv'):
        for candidate in row['candidate_ids'].split(';'):
            trait = variants_by_id[int(candidate)]
            assert int(candidate) >= 92000
            assert int(row['parent']) in trait['parents']
            assert not trait['fixed_K'] or int(row['K']) == trait['fixed_K']
            add('main_variants', row, candidate, trait)
    for row in read_json(HERE / 'hybrid_small/candidate_coverage.json')['rows']:
        assert row['legal'] and row['offline_compile_link'] == 'passed'
        assert row['scratch_bytes'] == row['VGPR_spills'] == 0
        add('hybrid_small', row, f"{row['actual_kid']}@{row['candidate']}", row)
    for row in read_csv(HERE / 'small_split/candidate_cases.csv'):
        row['actual_kid'] = row['actual']
        add('small_split', row, row['variant'], row)
    for row in read_json(HERE / 'large9030/candidate_coverage.json')['rows']:
        assert row['legal'] and row['offline_compile_link'] == 'passed'
        assert row['scratch_bytes'] == row['VGPR_spills'] == 0
        add('large9030', row, row['candidate'], row)
    assert set(ownership) == set(by_shape)
    assert len(cases) == len({(shape(x), x['candidate_key']) for x in cases}) == 1280
    grouped = defaultdict(list)
    for case in cases:
        grouped[shape(case)].append(case)
    coverage = []
    for key in sorted(by_shape):
        row = dict(by_shape[key])
        row.update(package=ownership[key], candidate_count=len(grouped[key]),
                   candidate_keys=';'.join(sorted(x['candidate_key'] for x in grouped[key])),
                   offline_compile_link='passed', cpu_layout_contract_audit='passed',
                   numerical_validation='not_run_gpu_stopped',
                   performance_validation='not_run_gpu_stopped', registered=False)
        coverage.append(row)
    cases.sort(key=lambda x: (shape(x), x['candidate_key']))
    write_csv(HERE / 'coverage294.csv', coverage)
    write_csv(HERE / 'candidate_cases.csv', cases)

    catalog = []
    for row in main_variants:
        if row['id'] >= 92000:
            catalog.append(dict(package='main_variants',
                                candidate_key=f"main_variants:{row['id']}", details=row))
    hybrid = read_json(HERE / 'hybrid_small/source_manifest.json')
    for actual in hybrid['actual_ids']:
        for ahead in hybrid['b_ahead']:
            catalog.append(dict(package='hybrid_small',
                                candidate_key=f'hybrid_small:{actual}@direct_b_ahead{ahead}',
                                details=dict(actual_kid=actual, b_ahead=ahead,
                                             library=f'direct_b_ahead{ahead}',
                                             launch_argument=actual)))
    for candidate in read_json(HERE / 'small_split/source_manifest.json')['variants']:
        catalog.append(dict(package='small_split', candidate_key=f'small_split:{candidate}',
                            details=dict(variant=candidate, pool='register' if candidate < 200 else 'fine')))
    for candidate in ('panel16', 'directb'):
        catalog.append(dict(package='large9030', candidate_key=f'large9030:{candidate}',
                            details=dict(actual_kid=9030, fixed_K=1536, library=candidate)))
    keys = {x['candidate_key'] for x in catalog}
    assert len(catalog) == len(keys) == 62
    assert all(x['candidate_key'] in keys for x in cases)
    write_json(HERE / 'candidate_catalog.json', catalog)

    package_stats = {}
    expected = dict(main_variants=(109, 465, 24, 32), hybrid_small=(85, 255, 24, 32),
                    small_split=(90, 540, 12, 45), large9030=(10, 20, 2, 3))
    for package, (shapes, pairs, candidates, emitted) in expected.items():
        subset = [x for x in cases if x['package'] == package]
        assert len(subset) == pairs and len({shape(x) for x in subset}) == shapes
        package_stats[package] = dict(historical_losing_shapes=shapes, primary_candidate_cases=pairs,
                                     new_candidates=candidates,
                                     used_candidate_keys=len({x['candidate_key'] for x in subset}),
                                     current_emitted_kernels_including_controls_and_reducers=emitted)

    summary = dict(status='all_families_implemented_offline_gpu_validation_pending',
                   gpu_tests='stopped_by_user', gpu_queries=0, gpu_kernel_launches=0,
                   experiment_libraries_loaded=False, registered=False,
                   historical_selected=dict(total=745, opus=693, opus_faster=399, opus_slower=294),
                   historical_pure_opus=dict(total=745, faster=411, slower=334),
                   historical_timing_limit='Saved historical CSVs; not clean same-card measurements. mxpsh caller scale reorder excluded.',
                   primary_covered_shapes=294, primary_parent_groups=26, new_candidates=62,
                   primary_candidate_cases=1280, current_emitted_kernels=114,
                   same_source_compiler_controls=2, compiler_control_shapes=3,
                   final_scratch_bytes_and_vgpr_sgpr_spills=0, packages=package_stats,
                   numerical_validation='not_run_gpu_stopped',
                   performance_validation='not_run_gpu_stopped',
                   final_CPU_audits_have_no_HIP_HSA_dependency=True,
                   historical_CPU_runtime_link_issue=dict(
                       affected_packages=['large9030', 'small_split'],
                       description='Initial host-only HIP direct links inherited libamdhip64 and host executables loaded that runtime. No HIP calls, GPU kernels, experiment library loads or timings were performed. Those receipts are superseded by object/plain-link/readelf-checked host audits.',
                       receipts=['large9030/audit_attempts.json', 'small_split/README.md']),
                   unchanged_formal_inputs=FORMAL,
                   package_manifests=manifests,
                   total_read_only_review_sha256=sha(HERE / 'total_read_only_review.json'))
    if (HERE / 'additional40.csv').exists():
        extras = read_csv(HERE / 'additional40.csv')
        extra_audit = read_json(HERE / 'additional40_audit.json')
        assert extra_audit['status'] == 'cpu_historical_comparison_and_existing_candidate_contract_audit_passed'
        assert extra_audit['every_shape_has_preferred_compiled_candidate']
        assert extra_audit['pure_opus_losers'] == 334
        assert extra_audit['preferred_candidate_cases'] == 172
        assert extra_audit['host_contract_audit']['no_hip_hsa_dependencies']
        for path, digest in extra_audit['inputs'].items():
            assert sha(ROOT / path) == digest, path
        for path, digest in extra_audit['package_contract_and_launch_sources'].items():
            assert sha(HERE / path) == digest, path
        for path, digest in extra_audit['outputs'].items():
            assert sha(HERE / path) == digest, path
        assert sha(HERE / 'additional40.py') == extra_audit['source_sha256']
        for package, evidence in extra_audit['package_evidence'].items():
            for name, digest in evidence.items():
                assert sha(HERE / package / name) == digest, (package, name)
        assert len(extras) == len({shape(x) for x in extras}) == 40
        assert not (set(by_shape) & {shape(x) for x in extras})
        assert sha(HERE / 'additional40.csv') == extra_audit['outputs']['additional40.csv']

        def canonical(token):
            location, launch_id = token.split(':')
            package, side = location.split('/')
            if package == 'hybrid_small':
                return f'{package}:{launch_id}@{side}'
            return f'{package}:{launch_id}'

        extra_cases = []
        extra_groups = {}
        for row in extras:
            tokens = row['preferred_candidate_tokens'].split(';')
            assert len(tokens) == int(row['preferred_candidate_count'])
            assert set(tokens) <= set(row['all_contract_legal_candidate_tokens'].split(';'))
            mapped = [canonical(x) for x in tokens]
            assert set(mapped) <= keys
            assert len(mapped) == len(set(mapped))
            assert float(row['best_legal_opus_us']) > float(row['flydsl_us'])
            assert row['selected_backend'] != 'opus'
            packages = {x.split(':')[0] for x in mapped}
            assert len(packages) == 1
            package = packages.pop()
            extra_groups[shape(row)] = (package, mapped)
            for token, candidate_key in zip(tokens, mapped):
                descriptor = extra_audit['candidate_catalog'][token]
                assert sha(ROOT / descriptor['library']) == descriptor['library_sha256']
                extra_cases.append(dict(M=int(row['M']), N=int(row['N']), K=int(row['K']),
                                        package=package,
                                        opus_parent_kid=int(row['best_legal_opus_parent']),
                                        actual_kid=int(row['best_legal_opus_actual']),
                                        candidate_key=candidate_key,
                                        offline_compile_link='passed', cpu_layout_contract_audit='passed',
                                        numerical_validation='not_run_gpu_stopped',
                                        performance_validation='not_run_gpu_stopped', registered=False,
                                        details_json=json.dumps(dict(source='additional40_audit.json',
                                                                     preferred_same_family=True,
                                                                     candidate_token=token,
                                                                     descriptor=descriptor), sort_keys=True)))
        assert len(extra_cases) == 172
        all_cases = sorted(cases + extra_cases, key=lambda x: (shape(x), x['candidate_key']))
        assert len(all_cases) == len({(shape(x), x['candidate_key']) for x in all_cases}) == 1452
        assert len({shape(x) for x in all_cases}) == 334
        write_csv(HERE / 'candidate_cases334.csv', all_cases)
        unified = []

        def unified_row(row, scope, package, candidate_keys):
            extra = scope == 'additional_non_opus_selected'
            return dict(M=int(row['M']), N=int(row['N']), K=int(row['K']), scope=scope,
                        selected_backend=row['selected_backend'] if extra else 'opus',
                        selected_kid=row['selected_kid'] if extra else row['opus_parent_kid'],
                        opus_parent_kid=row['best_legal_opus_parent'] if extra else row['opus_parent_kid'],
                        actual_kid=row['best_legal_opus_actual'] if extra else row['actual_kid'],
                        historical_best_opus_us=row['best_legal_opus_us'] if extra else row['opus_us'],
                        flydsl_kind=row['flydsl_kind'], flydsl_kid=row['flydsl_kid'],
                        flydsl_us=row['flydsl_us'], slow_pct=row['slow_pct'],
                        needed_latency_reduction_pct=row['needed_latency_reduction_pct'],
                        package=package, candidate_count=len(candidate_keys),
                        candidate_keys=';'.join(sorted(candidate_keys)),
                        offline_compile_link='passed', cpu_layout_contract_audit='passed',
                        numerical_validation='not_run_gpu_stopped',
                        performance_validation='not_run_gpu_stopped', registered=False)

        for row in coverage:
            unified.append(unified_row(row, 'selected_opus', row['package'], row['candidate_keys'].split(';')))
        for row in extras:
            package, mapped = extra_groups[shape(row)]
            unified.append(unified_row(row, 'additional_non_opus_selected', package, mapped))
        unified.sort(key=shape)
        assert len(unified) == len({shape(x) for x in unified}) == 334
        write_csv(HERE / 'coverage334.csv', unified)
        summary['additional_non_opus_selected_shapes'] = 40
        summary['additional_selected_backend_counts'] = dict(Counter(row['selected_backend'] for row in extras))
        summary['additional_preferred_candidate_cases'] = 172
        summary['additional_preferred_case_counts_by_package'] = extra_audit['preferred_case_counts_by_package']
        summary['additional_all_contract_legal_cases_including_cross_family_remaps'] = 1705
        summary['pure_opus_covered_shapes'] = 334
        summary['pure_opus_preferred_candidate_cases'] = 1452
        summary['pure_opus_coverage_by_package'] = dict(Counter(row['package'] for row in unified))
        summary['additional_coverage_audit_sha256'] = sha(HERE / 'additional40_audit.json')
        summary['additional_coverage_audit_status'] = extra_audit['status']
    write_json(HERE / 'summary.json', summary)
    output_names = ['coverage294.csv', 'candidate_cases.csv', 'candidate_catalog.json', 'summary.json']
    if summary.get('pure_opus_covered_shapes') == 334:
        output_names.extend(['coverage334.csv', 'candidate_cases334.csv', 'additional40.csv',
                             'additional40_audit.json'])
    verification = dict(status='offline_aggregation_verified', stdlib_only=True, gpu_operations=0,
                        exact294_partition=True, unique1280_candidate_cases=True,
                        legal_fixedK_main_mapping=True, candidate_catalog_unique62=True,
                        exact334_coverage=summary.get('pure_opus_covered_shapes') == 334,
                        unique1452_preferred_candidate_cases=summary.get('pure_opus_preferred_candidate_cases') == 1452,
                        original_csv_sha256=sha(original_path), formal_inputs_match=True,
                        all_child_manifest_bound_files_match=True, child_manifests=manifests,
                        total_read_only_review_sha256=sha(HERE / 'total_read_only_review.json'),
                        aggregate_script_sha256=sha(Path(__file__)),
                        outputs={name: sha(HERE / name) for name in output_names})
    write_json(HERE / 'verification.json', verification)
    print(json.dumps(dict(status=verification['status'], shapes=294, pairs=1280,
                          candidates=62, kernels=114)))


if __name__ == '__main__':
    main()
