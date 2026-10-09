from pathlib import Path
import json,shutil,hashlib,difflib
ROOT=Path.cwd();OUT=ROOT/'reports/opus_9000_9010_bound_20261008';OLD=ROOT/'reports/opus_bound_analysis_20261007'
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
module=json.loads((ROOT/'reports/opus_remaining_20261008/formal_selected/integration_manifest.json').read_text())['official_module']
print('module',module)
if 'path' not in module:raise ValueError(module)
inv=json.loads((OLD/'shape_inventory.json').read_text())
rows=[r for r in inv['rows'] if r['parent_id'] in [9000,9010]]
dump(OUT/'winner_inventory.json',{'baseline_module':module,'rows':rows})
selected=[]
for kid,desired in [(9000,[[1792,7168,384],[8192,8192,8192],[1024,7168,16384]]),(9010,[[1856,7168,384],[1856,7168,16384]])]:
 own=[r for r in rows if r['parent_id']==kid]
 for shape in desired:
  matched=[r for r in own if [r['M'],r['N'],r['K']]==shape]
  if not matched:
   k=shape[2];matched=sorted([r for r in own if r['K']==k],key=lambda r:abs(r['M']-shape[0])+abs(r['N']-shape[1]))[:1]
  r=matched[0] if matched else own[0]
  selected.append({'kid':kid,'shape':[r['M'],r['N'],r['K']],'seed':17,'signed':True,'purpose':'current winner short/long K diagnostic','private_baselines':{},'symbol':r['official_kernel_symbol'],'kernel_function':r['kernel_function'],'instruction_sha256':r['official_instruction_sha256']})
# Extra bounded-tail and full-M controls belong to the same 9010 body.
for shape in [[64,256,128],[320,256,256],[512,512,8320]]:
 selected.append({'kid':9010,'shape':shape,'seed':17,'signed':True,'purpose':'tail or panel-refill correctness control','private_baselines':{},'kernel_function':'gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel'})
plan={'source_head':inv['source_head'],'current_parent_count':26,'official_binary':module['path'],'official_binary_sha256':module['sha256'],'targets':selected}
dump(OUT/'baseline_diagnostics/plan.json',plan)
template=json.loads((ROOT/'reports/opus_remaining_20261008/diagnostics/att_queue.json').read_text())
base_env=template['env'].copy();base_env['AITER_JIT_DIR']=str(Path(module['path']).parent)
prefix=['/opt/venv/bin/python3',str(OLD/'official_smoke.py'),'--plan',str(OUT/'baseline_diagnostics/plan.json')]
commands=[{'name':'baseline_correctness','argv':prefix+['--output',str(OUT/'baseline_diagnostics/correctness.json'),'--check-only','--repetitions','2'],'log':str(OUT/'baseline_diagnostics/correctness.log')}]
for i,t in enumerate(selected[:5]):
 name=f'target{i}_kid{t["kid"]}_att';folder=OUT/'baseline_diagnostics'/name
 env=template['commands'][1]['env'].copy();env.update(ROCPROF_OUTPUT_FILE_NAME=name,ROCPROF_OUTPUT_PATH=str(folder),ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=t['kernel_function'])
 commands.append({'name':name,'argv':prefix+['--output',str(folder/'application.json'),'--profile','--target-index',str(i),'--iters','11','--profile-rotation','1'],'env':env,'log':str(folder/'rocprof.log')})
dump(OUT/'baseline_diagnostics/att_queue.json',{'env':base_env,'commands':commands})
# Fresh include snapshots, changing the two kernel bodies only.
files=['opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh','opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh']
for side in ['baseline','matrix_first']:
 d=OUT/side
 if d.exists():raise RuntimeError('existing experiment '+str(d))
 shutil.copytree(ROOT/'csrc/opus_gemm/include',d/'include')
launcher='''#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 if((kid!=9000&&kid!=9010)||m<=0||n<=0||k<=0||k>16384||k%128||n%256||m%(kid==9000?256:64)||int64_t(m)*k>INT32_MAX||int64_t(n)*k>INT32_MAX||int64_t(m)*n*2>INT32_MAX) return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;
 args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 auto hs=reinterpret_cast<hipStream_t>(stream);
 if(kid==9000) gemm_a8w8_mxfp8_scale_kernel<opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950><<<dim3(n/256,m/256),256,0,hs>>>(args);
 else gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel<opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950><<<dim3(n/256,(m+255)/256),256,0,hs>>>(args);
 return int(hipGetLastError());
}
'''
changes=[]
for filename in files:
 path=OUT/'matrix_first/include/gfx950'/filename;old=path.read_text()
 start=old.index('    // ===== Prologue =====');end=old.index('    publish_sfa_panel(0);',start)
 block=old[start:end]
 blines=[line for line in block.splitlines(True) if 'async_load<T::VEC_B>' in line]
 assert len(blines)==2
 block=''.join(line for line in block.splitlines(True) if 'async_load<T::VEC_B>' not in line)
 block=block.replace('    load_sfa_panel(0);',''.join(blines)+'    __builtin_amdgcn_sched_barrier(0);\n    load_sfa_panel(0);')
 new=old[:start]+block+old[end:];path.write_text(new)
 (OUT/f'{filename}.matrix_first.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='baseline/'+filename,tofile='matrix_first/'+filename)))
 changes.append({'path':'csrc/opus_gemm/include/gfx950/'+filename,'baseline_sha256':hashlib.sha256(old.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(new.encode()).hexdigest()})
for side in ['baseline','matrix_first']:(OUT/side/'launch.hip').write_text(launcher)
dump(OUT/'source_manifest.json',{'status':'prepared_private','baseline':module,'changes':changes,'mechanism':'Issue both K0 matrix halves before scale requests; preserve scale format, all hardware waits/barriers, K1 and steady loop.'})
print('targets',[(t['kid'],t['shape']) for t in selected])
