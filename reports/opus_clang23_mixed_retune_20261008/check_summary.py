"""Execute new summary on frozen raw data without changing frozen outputs."""
from pathlib import Path
import ast,collections,hashlib,json,math,shutil,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
namespace=dict(Path=Path,collections=collections,hashlib=hashlib,json=json,math=math,pd=pd,ROOT=ROOT,KEYS=['gfx','cu_num','M','N','K'],CALL=['gfx','cu_num','M','N','K','libtype','kernelName','splitK'])
nodes=[node for node in ast.parse((OUT/'summarize.py').read_text()).body if isinstance(node,ast.FunctionDef)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'summarize.py'),'exec'),namespace)
raw=pd.read_csv(ROOT/'reports/opus_retune28_20261008/profile.csv')
shapes=pd.read_csv(OUT/'shapes_745.csv');plan=json.loads((OUT/'plan.json').read_text())
with tempfile.TemporaryDirectory(prefix='mixed-summary-test-') as tmp:
 directory=Path(tmp);namespace['OUT']=directory
 (directory/'plan.json').write_text(json.dumps(plan))
 for name in ['shapes_745.csv','compiler_manifest.json']:shutil.copy2(OUT/name,directory/name)
 commands=[];claims=[]
 for index,indices in enumerate(np.array_split(range(745),8)):
  data=raw.merge(shapes.iloc[indices][['M','N','K']],on=['M','N','K'],validate='many_to_one').drop(columns='checkpoint')
  profile=directory/f'profile{index}.csv';data.to_csv(profile,index=False)
  meta=directory/f'run{index}.json'
  data[data.libtype.eq('opus')][['M','N','K','kernelId']].to_csv(directory/f'run{index}_expected_opus_candidates.csv',index=False)
  meta.write_text(json.dumps({'status':'completed','libraries':{n:{'sha256':sha} for n,sha in plan['library_sha256'].items()},'gpu_bdf':str(index)}))
  name=f'fake{index}';commands.append({'name':name,'argv':['--metadata',str(meta),'-o2',str(profile)]})
  claims.extend([{'event':'owner_identity','name':name,'host_pid':index},{'event':'monitor','name':name,'processes':[{'pid':index}]},{'event':'end','name':name,'returncode':0,'contamination':False}])
 (directory/'queue_eight.json').write_text(json.dumps({'commands':commands}))
 (directory/'claim_eight.jsonl').write_text('\n'.join(json.dumps(row) for row in claims))
 namespace['main']()
 receipt=json.loads((directory/'summary.json').read_text())
 (OUT/'summary_frozen_data_integration_test.json').write_text(json.dumps({'status':'passed','fixture':'frozen retune28 raw, temporary output only','shapes':receipt['shapes'],'opus_tasks':receipt['valid_opus_tasks'],'profile_rows':receipt['profile_rows'],'upstream_same_execution':receipt['upstream_same_execution_descriptive_only']},indent=2)+'\n')
 # Exercise publication and documentation in an isolated mock checkout.
 fake_root=directory/'checkout';fake_root.mkdir()
 hashes=json.loads((OUT/'old_artifact_hashes.json').read_text())
 for relative in list(hashes)+['HANDOFF_MXFP8.md','csrc/opus_gemm/README.md']:
  target=fake_root/relative;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(ROOT/relative,target)
 (directory/'old_artifact_hashes.json').write_text(json.dumps(hashes))
 tree=ast.parse((OUT/'write_report.py').read_text())
 tree.body=[node for node in tree.body if not isinstance(node,ast.Assign) or not any(isinstance(target,ast.Name) and target.id in {'ROOT','OUT'} for target in node.targets)]
 exec(compile(tree,str(OUT/'write_report.py'),'exec'),{'ROOT':fake_root,'OUT':directory})
 published=pd.read_csv(fake_root/'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv')
 assert len(published)==745
 assert np.allclose(published.bw,pd.read_csv(directory/'tuned_all.csv').bw/1000)
 (OUT/'publication_integration_test.json').write_text(json.dumps({'status':'passed','fixture':'temporary mock checkout only','rows':745,'bandwidth_GBps_to_TBps':True,'docs_rewritten':True},indent=2)+'\n')
print('FULL SUMMARY INTEGRATION PASSED')
