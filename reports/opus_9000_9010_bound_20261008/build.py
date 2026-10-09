from pathlib import Path
import json,os,subprocess,time,importlib.util,sys,hashlib
ROOT=Path.cwd();OUT=ROOT/'reports/opus_9000_9010_bound_20261008'
old=ROOT/'reports/opus_resume_20261008/scale_issue_publish'
command=json.loads((old/'build_manifest.json').read_text())['builds'][0]['command']
spec=importlib.util.spec_from_file_location('elf',ROOT/'reports/opus_remaining_20261008/formal_selected/audit_integration.py');elf=importlib.util.module_from_spec(spec);spec.loader.exec_module(elf)
official=json.loads((OUT/'source_manifest.json').read_text())['baseline'];allrows=[r for b in elf.bundles(official['path']) for r in b['rows']]
expected={r['name']:r for r in allrows if r['name'].startswith(('_Z28gemm_a8w8_mxfp8_scale_kernel','_Z51gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel'))}
records=[]
for side in sys.argv[1:] or ['baseline','matrix_first']:
 d=OUT/side;argv=[a.replace(str(old/'baseline'),str(d)) for a in command]
 argv.insert(argv.index('-shared'),'-resource-dir=/opt/rocm/lib/llvm/lib/clang/20')
 env=dict(os.environ,HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
 started=time.monotonic()
 with (d/'build.log').open('w') as log:ret=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if ret.returncode:raise RuntimeError(str(d/'build.log'))
 bundles=elf.bundles(d/'experiments.so');assert len(bundles)==1
 rows=bundles[0]['rows'];assert {r['name'] for r in rows}==set(expected)
 for row in rows:
  if side=='baseline':
   for key in ['instruction_sha256','metadata','descriptor_normalized_sha256']:assert row[key]==expected[row['name']][key],(row['name'],key)
  row['resource_screen_passed']=row['metadata']['.private_segment_fixed_size']==0 and row['metadata']['.vgpr_spill_count']==0 and row['metadata']['.sgpr_spill_count']==0
 image=elf.COMMON.device_bundle(d/'experiments.so') if hasattr(elf.COMMON,'device_bundle') else None
 # Recover the single gfx950 image using the retained reader.
 helper=ROOT/'reports/opus_bound_analysis_20261007/audit_current_compute_metadata.py'
 sp=importlib.util.spec_from_file_location('images',helper);im=importlib.util.module_from_spec(sp);sp.loader.exec_module(im)
 (d/'device.co').write_bytes(im.device_bundle(d/'experiments.so'))
 with (d/'device.s').open('w') as f:subprocess.run(['/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump','-d','--mcpu=gfx950',str(d/'device.co')],stdout=f,check=True)
 record={'side':side,'command':argv,'seconds':time.monotonic()-started,'library_sha256':hashlib.sha256((d/'experiments.so').read_bytes()).hexdigest(),'rows':rows,'baseline_exact_official':side=='baseline'}
 (d/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
 print(json.dumps({'side':side,'seconds':record['seconds'],'resources':[(r['demangled'].split('<')[0],r['metadata']['.vgpr_count'],r['metadata']['.sgpr_count']) for r in rows]}),flush=True)
