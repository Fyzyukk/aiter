"""CPU-only formal identity check for two optional tuning alternatives."""
from pathlib import Path
import collections, hashlib, importlib.util, json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HERE = OUT / 'split_formal_scale_reset'
spec = importlib.util.spec_from_file_location('elf', ROOT/'reports/opus_remaining_20261008/formal_selected/audit_integration.py')
elf = importlib.util.module_from_spec(spec); spec.loader.exec_module(elf)
build = json.loads((HERE/'build_manifest.json').read_text())
baseline = json.loads((OUT/'source_manifest.json').read_text())['baseline']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(build['binary']) == build['binary_sha256'] and sha(baseline['path']) == baseline['sha256']
old = elf.bundles(baseline['path']); new = elf.bundles(build['binary'])
keys = ['instruction_sha256','metadata','descriptor_normalized_sha256']
def signature(row):
    return json.dumps({k:row[k] for k in keys}, sort_keys=True)
old_rows = [r for b in old for r in b['rows']]
new_rows = [r for b in new for r in b['rows']]
expected = collections.Counter((r['name'], signature(r)) for r in old_rows)
actual = collections.Counter((r['name'], signature(r)) for r in new_rows)
assert not (expected-actual), 'Existing entry machine identity changed'
extras = list((actual-expected).elements())
assert len(extras)==2 and len(new_rows)==len(old_rows)+2
adopted=[]
for name,_ in extras:
    row=next(r for r in new_rows if r['name']==name)
    if row['name'].startswith('_Z28gemm_a8w8_mxfp8_scale_kernel'):
        kid,base,side=9001,9000,'scale_reset'
    else:
        assert row['name'].startswith('_Z51gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel')
        kid,base,side=9011,9010,'loop_unroll4'
    private=json.loads((OUT/side/'build_manifest.json').read_text())['rows']
    reference=next(r for r in private if r['name'].split('I')[0]==row['name'].split('I')[0])
    assert row['instruction_sha256']==reference['instruction_sha256']
    assert row['descriptor_normalized_sha256']==reference['descriptor_normalized_sha256']
    metadata={k:v for k,v in row['metadata'].items() if k not in ['.name','.symbol']}
    original={k:v for k,v in reference['metadata'].items() if k not in ['.name','.symbol']}
    assert metadata==original
    assert row['metadata']['.private_segment_fixed_size']==0 and row['metadata']['.vgpr_spill_count']==0 and row['metadata']['.sgpr_spill_count']==0
    adopted.append({'kid':kid,'base_parent_id':base,'side':side,'kernel':row,'private_reference':reference,
                    'identity_note':'FUNC, normalized descriptor and full metadata match tested private candidate; only .name/.symbol follow new traits identity.'})
result={'status':'passed_formal_identity_pending_API','baseline':baseline,
        'official_module':{'path':build['binary'],'sha256':build['binary_sha256']},
        'original_public_parents':26,'original_public_entries':56,'final_public_parents':28,'final_public_entries':58,
        'original_linked_bundles':len(old),'final_linked_bundles':len(new),'original_linked_entries':len(old_rows),
        'final_linked_entries':len(new_rows),'unchanged_existing_entries':len(old_rows),
        'new_candidates':adopted,'source_root':build['source_root'],
        'retention_policy':'9000 and 9010 original machine entries remain; at most one optional alternative per geometry; tuner measures each.'}
(HERE/'identity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['new_candidates','baseline','official_module']}))
