#!/usr/bin/env python3
"""CPU-only ELF, shape and queue audit. Never load a HIP library."""
import csv,collections,hashlib,importlib.util,json,os,re,subprocess,difflib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=H/'cpu_audit'
 if out.exists():raise SystemExit('Refusing existing audit')
 out.mkdir()
 path=R/'reports/opus_resume_20261008/sfa_packed/audit_device.py';spec=importlib.util.spec_from_file_location('cpu_elf_reader',path);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
 receipt=json.loads((H/'build_receipt.json').read_text()); records=[]
 for b in receipt['builds']:
  assert sha(b['library'])==b['library_sha256']
  # Detect old hip vs hipv4 bundle without loading any runtime.
  elf=parser.Elf(Path(b['library']).read_bytes());bundle=elf.bytes(elf.sections['.hip_fatbin'])
  parser.TARGET='hipv4-amdgcn-amd-amdhsa--gfx950' if b'hipv4-amdgcn' in bundle else 'hip-amdgcn-amd-amdhsa--gfx950'
  device=parser.bundled_image(Path(b['library']));co=out/(b['side']+'.co');co.write_bytes(device)
  elf=parser.Elf(device);meta=elf.metadata();symbols=elf.symbols()
  (out/(b['side']+'_metadata.json')).write_text(json.dumps(meta,indent=2)+'\n')
  isa=subprocess.check_output(['/opt/rocm-llvm23-46fcb339/bin/llvm-objdump','-d','--mcpu=gfx950',str(co)],env=ENV,text=True)
  (out/(b['side']+'_isa.txt')).write_text(isa)
  kernels=[]
  for k in meta['amdhsa.kernels']:
   s=symbols[k['.name']];assert s['type']==2 and len(s['bytes'])==s['size']>0
   name=subprocess.check_output(['c++filt',k['.name']],env=ENV,text=True).strip()
   resources={f:k.get('.'+f) for f in ['vgpr_count','sgpr_count','agpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']}
   kernels.append({'name':name,'resources':resources,'ISA_bytes':s['size'],'ISA_sha256':hashlib.sha256(s['bytes']).hexdigest(),'static_status':'spill_free' if resources['private_segment_fixed_size']==0 and resources['vgpr_spill_count']==0 and resources['sgpr_spill_count']==0 else 'has_spill'})
  records.append({'side':b['side'],'device_sha256':sha(co),'kernels':kernels})
 # Host adapter affects pure layout helpers only. HIP kernels built from unchanged inputs.
 helper_source=(H/'frozen/gemm_include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh').read_text()
 start=helper_source.index('template<class T, int Pass>\n');end=helper_source.index('} // namespace opus_gemm_4wave_128x128_layout')
 helpers=helper_source[start:end].replace('__device__','__host__')
 (out/'helpers.h').write_text('#pragma once\n#include <opus/hip_minimal.hpp>\n#include <opus/opus.hpp>\nusing opus::operator""_I;\nnamespace checked {\n'+helpers+'}\n')
 original=(H/'frozen/opus/opus.hpp').read_text();adapter=original.replace('template<index_t Vec, typename Layout> OPUS_D constexpr auto layout_to_offsets','template<index_t Vec, typename Layout> OPUS_H_D constexpr auto layout_to_offsets')
 # Exact OPUS_D spellings in upstream differ; use narrow pure-layout attributes.
 adapter=re.sub(r'OPUS_D([^\n]*(?:layout_to_offsets|unfold_x_stride|unfold_p_coord|make_layout|dim_offset_sum|dim_axis_at|dim_prefix_prod|y_prefix_prod|p_prefix_count|axis_prefix_prod)[^\n]*)',r'OPUS_H_D\1',adapter)
 begin=adapter.rfind('namespace impl {',0,adapter.index('constexpr index_t dim_offset_sum'));end=adapter.index('/////////////////////////////////////////////////////////////////////////////////////////////////////////',begin) if '/////////////////////////////////////////////////////////////////////////////////////////////////////////' in adapter[begin:] else begin+7000
 adapter=adapter[:begin]+adapter[begin:end].replace('OPUS_D','OPUS_H_D')+adapter[end:]
 (out/'host_include/opus').mkdir(parents=True)
 (out/'host_include/opus/opus.hpp').write_text(adapter)
 (out/'host_attribute.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),adapter.splitlines(True))))
 argv=['/opt/rocm-llvm23-46fcb339/bin/clang++','-x','hip','--offload-host-only','--rocm-path=/opt/rocm','--hip-path=/opt/rocm','-std=c++20','-O2','-D__HIPCC_RTC__=1','-I'+str(out),'-I'+str(H),'-I'+str(out/'host_include'),'-I'+str(H/'frozen'),'-I'+str(H/'frozen/opus'),'-I'+str(H/'frozen/gemm_include'),'-I'+str(H/'frozen/gemm_include/gfx950'),'-c',str(H/'layout_check.cpp'),'-o',str(out/'layout_check.o')]
 done=subprocess.run(argv,cwd=H,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'compile.log').write_text(done.stdout)
 if done.returncode:raise RuntimeError(done.stdout[-5000:])
 link_argv=['/opt/rocm-llvm23-46fcb339/bin/clang++',str(out/'layout_check.o'),'-o',str(out/'layout_check')]
 subprocess.run(link_argv,check=True,env=ENV)
 needed=subprocess.check_output(['readelf','-d',str(out/'layout_check')],env=ENV,text=True)
 assert 'amdhip' not in needed.lower() and 'hsa-runtime' not in needed.lower(),needed
 stdout=subprocess.check_output([str(out/'layout_check')],env=ENV,text=True)
 rows=list(csv.DictReader((R/'reports/opus_flydsl_gap_20261009/losers294.csv').open()))
 pool=[x for x in rows if int(x['opus_parent_kid']) in [9000,9001,9011,9020,9021,9022,9024]]
 variants=json.loads((H/'variants.json').read_text());coverage=[]
 for x in pool:
  parent,m,n,k=map(int,[x['opus_parent_kid'],x['M'],x['N'],x['K']]);vs=[v for v in variants if v['id']>=92000 and parent in v['parents'] and n%v['tile_N']==0 and (not v['fixed_K'] or k==v['fixed_K'])]
  assert vs,(parent,m,n,k)
  coverage.append({'M':m,'N':n,'K':k,'parent':parent,'candidate_ids':';'.join(str(v['id']) for v in vs)})
 with (H/'coverage.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
 assert len(coverage)==109
 audit={'status':'cpu_audit_passed_gpu_stopped','gpu_operations':0,'numerical_validation':'not_run_gpu_stopped','performance_validation':'not_run_gpu_stopped','libraries':records,'coverage_shapes':109,'coverage_parent_counts':dict(collections.Counter(x['parent'] for x in coverage)),'host_check':{'compile_argv':argv,'link_argv':link_argv,'dynamic_dependencies':needed,'no_HIP_HSA_dependencies':True,'stdout':stdout,'binary_sha256':sha(out/'layout_check')},'elf_parser_sha256':sha(path)}
 (H/'cpu_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('CPU layout and coverage PASS; ELF kernels:',[(b['side'],len(b['kernels'])) for b in records]);print('spills',[(k['name'],k['resources']) for b in records for k in b['kernels'] if k['static_status']!='spill_free'])
if __name__=='__main__':main()
