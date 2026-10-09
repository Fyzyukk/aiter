"""CPU-only final decision and application of the already tested 9010 source."""
from pathlib import Path
import datetime, hashlib, json, math, statistics, subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HERE = OUT / 'formal_selected'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

audit = read(OUT / 'owner_audit.json')
def epoch(log, name):
    record = next(r for r in audit['records'] if Path(r['path']).resolve() == (OUT / log).resolve())
    item = next(e for e in record['epochs'] if e['name'] == name)
    assert not record['incomplete'] and item['strict_clean']
    return {'path': record['path'], 'sha256': record['sha256'], 'epoch': item}

def summarize(path, owner, method):
    data = read(path)
    assert data['status'] == 'passed'
    rows = []
    for r in data['rows']:
        e = r.get('event_confirmation', r)
        assert e.get('status', 'passed') == 'passed'
        for c in r['correctness'].values():
            assert c['errRatio'] == 0 and c['repeatable'] and c['output_guards'] and c['workspace_guards']
        by = {(m['round'], m['label']): m['us_per_call'] for m in e['measurements']}
        assert len(by) == 10
        pairs = [by[(i, 'baseline')] / by[(i, 'candidate')] for i in range(5)]
        ratio = e['median_speedup']
        ratio = ratio['candidate'] if isinstance(ratio, dict) else ratio
        assert math.isclose(ratio, e['median_us']['baseline'] / e['median_us']['candidate'])
        rows.append({'kid': r['kid'], 'shape': r['shape'], 'speedup': ratio,
                     'faster_rounds': sum(p > 1 for p in pairs), 'paired_speedups': pairs,
                     'median_us': e['median_us'], 'pool_count': r.get('pool_count', 51),
                     'method': method, 'source': str(path.relative_to(OUT))})
    return {'path': str(path), 'sha256': sha(path), 'owner': owner, 'rows': rows}

def aggregate(parts):
    rows = [r for p in parts for r in p['rows']]
    assert len({tuple(r['shape']) for r in rows}) == len(rows)
    return {'completed_shapes': len(rows),
            'geomean_speedup': math.exp(statistics.mean(math.log(r['speedup']) for r in rows)),
            'positive_medians': sum(r['speedup'] > 1 for r in rows),
            'negative_medians': sum(r['speedup'] < 1 for r in rows),
            'five_of_five_faster': sum(r['faster_rounds'] == 5 for r in rows),
            'stable_losers': sum(r['faster_rounds'] == 0 for r in rows),
            'minimum_speedup': min(r['speedup'] for r in rows), 'rows': rows}

small = summarize(OUT / 'scale_reset/small9000_event.json', epoch('full9000_claim.jsonl', 'small9000'), '51_addresses_51_calls_graph_Event')
large = summarize(OUT / 'scale_reset/large9000_event.clean_serial.json', epoch('final_serial_claim.jsonl', 'large9000_clean_serial'), '8_addresses_51_calls_graph_Event')
result9000 = aggregate([small, large])
assert result9000['completed_shapes'] == 129 and result9000['geomean_speedup'] < 1 and result9000['stable_losers'] > 0
result9000.update(status='reject_9000_global_scale_reset_clean_serial', adopted=False,
                  parts=[{k: v for k, v in p.items() if k != 'rows'} for p in [small, large]],
                  excluded=['scale_reset/large9000_event.json', 'scale_reset/full9000_analysis.owner_excluded.json'],
                  limits=['Two disjoint size groups use 51- and 8-address pools respectively; ratios are within each clean paired experiment.',
                          'Equal-shape geometric mean is not production workload weighting.',
                          'Old large-output attempt and its aggregate are excluded due to non-owner GPU PIDs.'])
old = OUT / 'scale_reset/full9000_analysis.json'
backup = OUT / 'scale_reset/full9000_analysis.owner_excluded.json'
if not backup.exists():
    prior = read(old)
    prior['original_status'] = prior['status']
    prior['status'] = 'owner_excluded_not_decision_evidence'
    prior['superseded_by'] = 'full9000_analysis.json (clean serial large-output repeat)'
    write(backup, prior)
write(old, result9000)

part9010 = summarize(OUT / 'loop_unroll4/full9010_event.json', epoch('full9010_claim.jsonl', 'unroll4_all38_9010'), '51_addresses_51_calls_graph_Event')
result9010 = aggregate([part9010])
assert result9010['completed_shapes'] == 38 and result9010['geomean_speedup'] > 1 and result9010['stable_losers'] == 0
result9010.update(status='retain_9010_unroll4_after_formal_API', adopted=True,
                  negative_rows=[r for r in result9010['rows'] if r['speedup'] < 1],
                  source={k: v for k, v in part9010.items() if k != 'rows'},
                  limits=['Small measured aggregate gain; four negative shape medians retained explicitly.',
                          'Finite historical winners, not every supported shape or guaranteed production gain.'])
write(OUT / 'loop_unroll4/full9010_analysis.json', result9010)

selection = read(HERE / 'selection.json')
identity = read(HERE / 'identity_audit.json')
api_path = HERE / 'api_results.clean_serial.json'
api = read(api_path)
api_owner = epoch('final_serial_claim.jsonl', 'formal_API_clean_serial')
assert identity['unchanged_count'] == 262 and len(identity['changed']) == 1 and identity['linked_bundles'] == 202
assert api['status'] == 'passed' and len(api['rows']) == 9
assert sha(identity['official_binary']) == identity['official_binary_sha256'] == api['official_binary_sha256']
for r in api['rows']:
    c = r['correctness']['official']
    assert c['repetitions'] == 8 and c['errRatio'] == 0 and c['repeatable'] and c['output_guards'] and c['workspace_guards']
    assert r['actual_official_module']['sha256'] == api['official_binary_sha256']

rel = selection['changed_files'][0]
source = Path(selection['source_root']) / rel
destination = ROOT / rel
assert sha(source) == selection['source_sha256']
before = sha(destination)
baseline_source = read(OUT / 'source_manifest.json')['changes'][1]['baseline_sha256']
assert before in [baseline_source, selection['source_sha256']]
prior_manifest = read(ROOT / 'reports/opus_remaining_20261008/formal_selected/integration_manifest.json')
protected = {p: sha(ROOT / p) for p in prior_manifest['changed_source_files_relative_to_HEAD']} if isinstance(prior_manifest['changed_source_files_relative_to_HEAD'], list) else {}
protected.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'csrc/opus_gemm/include/gfx950').glob('*') if p.is_file() and str(p.relative_to(ROOT)) != rel})
destination.write_bytes(source.read_bytes())
assert sha(destination) == selection['source_sha256']
assert all(sha(ROOT / p) == digest for p, digest in protected.items())
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)

source_diff = []
for prefix in ['aiter', 'csrc']:
    for p in subprocess.check_output(['git', 'ls-files', prefix], cwd=ROOT, text=True).splitlines():
        a, b = ROOT / p, Path(selection['source_root']) / p
        if a.is_file() and b.is_file() and a.read_bytes() != b.read_bytes():
            source_diff.append(p)
assert not source_diff
manifest = {'status': 'applied_verified_not_committed',
            'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'accepted_parent_ids_this_change': [9010], 'rejected_parent_ids_this_change': [9000],
            'changed_source': {'path': rel, 'baseline_sha256': baseline_source, 'applied_sha256': sha(destination)},
            'production_source_matches_tested_worktree': True,
            'official_module': {'path': identity['official_binary'], 'sha256': identity['official_binary_sha256']},
            'baseline_module': selection['baseline_module'],
            'identity': {'path': str(HERE / 'identity_audit.json'), 'sha256': sha(HERE / 'identity_audit.json'),
                         'public_parents': 26, 'public_entries': 56, 'changed_public_entries': 1,
                         'unchanged_public_entries': 55, 'linked_bundles': 202,
                         'linked_entries': 263, 'unchanged_linked_entries': 262,
                         'changed_entry_exact_tested_private': True},
            'formal_API': {'path': str(api_path), 'sha256': sha(api_path), 'targets': 9, 'numerical_calls': 72,
                           'changed_parent_targets': 6, 'unchanged_9000_controls': 3,
                           'status': 'signed8_reference_repeat_output_workspace_guards_passed', 'owner': api_owner,
                           'performance_timed': False},
            'performance': {'9000': {k: v for k, v in result9000.items() if k != 'rows'},
                            '9010': {k: v for k, v in result9010.items() if k != 'rows'}},
            'prior_accepted_changes_preserved': ['9021 K0-first and scale issue/publish', '9020 existing fixed384',
                                                 '9023 runtime', '9024 fixed7168', '9042/9053/9054 runtime B-scale aliases'],
            'excluded_attempts': ['formal_selected/api_results.json', 'scale_reset/large9000_event.json'],
            'pending_experiments': 0, 'commit_or_push': False,
            'limitations': ['Formal numerical gate covers changed 9010 and three unchanged controls, not all 26 parents anew.',
                            'Every other linked entry is exact unchanged from the prior formally validated module.',
                            'Bottleneck ATT for 9010 is its pre-unroll baseline; final dominant bottleneck was not reprofiled.']}
write(HERE / 'integration_manifest.json', manifest)
print(json.dumps({'9000': {k: result9000[k] for k in ['completed_shapes', 'geomean_speedup', 'positive_medians', 'stable_losers']},
                  '9010': {k: result9010[k] for k in ['completed_shapes', 'geomean_speedup', 'positive_medians', 'stable_losers']},
                  'application': manifest['status'], 'source': rel}))
