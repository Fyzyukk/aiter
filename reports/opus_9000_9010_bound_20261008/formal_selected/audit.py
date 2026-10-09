from pathlib import Path
import json,importlib.util,hashlib,subprocess
ROOT=Path.cwd();OUT=ROOT/'reports/opus_9000_9010_bound_20261008';HERE=OUT/'formal_selected'
sp=importlib.util.spec_from_file_location('elf',ROOT/'reports/opus_remaining_20261008/formal_selected/audit_integration.py');elf=importlib.util.module_from_spec(sp);sp.loader.exec_module(elf)
sel=json.loads((HERE/'selection.json').read_text());build=json.loads((HERE/'build_manifest.json').read_text());base=elf.bundles(sel['baseline_module']['path']);new=elf.bundles(build['binary']);assert len(base)==len(new)
private={r['name']:r for b in elf.bundles(sel['candidate_private']) for r in b['rows']};changed=[];unchanged=[]
for b,c in zip(base,new):
 assert {r['name']for r in b['rows']}=={r['name']for r in c['rows']}
 for old,row in zip(sorted(b['rows'],key=lambda r:r['name']),sorted(c['rows'],key=lambda r:r['name'])):
  keys=['instruction_sha256','metadata','descriptor_normalized_sha256']
  different=any(old[k]!=row[k] for k in keys)
  if different:
   assert row['name'].startswith('_Z51gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel')
   assert all(row[k]==private[row['name']][k]for k in keys)
   changed.append({'baseline':old,'candidate':row})
  else:unchanged.append(row['name'])
assert len(changed)==1
work=Path(sel['source_root']);diff=[]
for prefix in ['aiter','csrc']:
 tracked=subprocess.check_output(['git','ls-files',prefix],cwd=ROOT,text=True).splitlines()
 for rel in tracked:
  a=ROOT/rel;b=work/rel
  if a.is_file() and b.is_file() and a.read_bytes()!=b.read_bytes():diff.append(rel)
assert diff==sel['changed_files'];source=work/diff[0];assert hashlib.sha256(source.read_bytes()).hexdigest()==sel['source_sha256']
manifest={'status':'passed_formal_identity_pending_GPU','official_binary':build['binary'],'official_binary_sha256':build['binary_sha256'],'baseline':sel['baseline_module'],'changed':changed,'unchanged_count':len(unchanged),'linked_bundles':len(base),'source_diff_files':diff,'source_root':str(work),'source_sha256':sel['source_sha256']}
(HERE/'identity_audit.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items()if k not in ['changed','baseline']}))
