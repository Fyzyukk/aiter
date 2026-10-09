#!/usr/bin/env python3
"""Prepare isolated main-family candidates without importing HIP or aiter."""
import csv,difflib,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
INC=ROOT/'csrc/opus_gemm/include'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_variant(filename, source, private):
 (HERE/'candidate'/filename).write_text(private)
 (HERE/'candidate'/(filename+'.diff')).write_text(''.join(difflib.unified_diff(source.splitlines(True),private.splitlines(True),fromfile='current frozen source',tofile='candidate/'+filename)))
def main():
 if (HERE/'frozen').exists(): raise SystemExit('Refusing to replace frozen inputs')
 (HERE/'candidate').mkdir(exist_ok=True)
 shutil.copytree(INC,HERE/'frozen/gemm_include')
 shutil.copytree(ROOT/'csrc/include/opus',HERE/'frozen/opus')
 # Generalize the actual 160-row pipeline only through template geometry and fixed K.
 source=(INC/'gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh').read_text()
 private=source.replace('gemm_a8w8_mxfp8_scale_4wave_160x128_kernel','opus_private_main_geometry_kernel').replace('#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"','#include "traits.cuh"')
 private=private.replace('const int loops = kargs.k / T::B_K;','const int loops = (T::FIXED_K ? T::FIXED_K : kargs.k) / T::B_K;')
 private=''.join(line for line in private.splitlines(True) if not any(s in line for s in ('const auto u_gsfa_0 =','const auto u_ssfa_0 =','const auto u_gsfa_1 =','const auto u_ssfa_1 =')))
 private=private.replace('const auto sfa_gmem_offsets = opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_0), layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_1));','const auto sfa_gmem_offsets = opus::transform_tuple([&](auto pass) { return layout_to_offsets<T::VEC_SCALE_A>(layout_9021::make_layout_gsfa_scale<T, decltype(pass)::value>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa)); }, opus::to_tuple(opus::make_index_seq<T::SFA_PASSES>{}));')
 private=private.replace('const auto sfa_smem_offsets = opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_0), layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_1));','const auto sfa_smem_offsets = opus::transform_tuple([&](auto pass) { return layout_to_offsets<T::VEC_SCALE_A>(layout_9021::make_layout_ssfa_scale<T, decltype(pass)::value>(lane_id, wave_id_m, wave_id_n)); }, opus::to_tuple(opus::make_index_seq<T::SFA_PASSES>{}));')
 assert 'u_gsfa_0' not in private and 'u_gsfa_1' not in private
 write_variant('geometry.cuh',source,private)
 # Explicit fixed-K pin prototypes keep all existing pin locations and scheduling.
 for stem,symbol,out in [('4wave','gemm_a8w8_mxfp8_scale_kernel','pin.cuh'),('4wave_256x256_padded_m','gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel','pad_pin.cuh')]:
  src=(INC/f'gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{stem}_gfx950.cuh').read_text()
  p=src.replace(symbol,symbol+'_private_fixed')
  p=p.replace('#if !defined(__HIP_DEVICE_COMPILE__)', 'namespace opus_private_pin {\n#if !defined(__HIP_DEVICE_COMPILE__)', 1) + '\n} // namespace opus_private_pin\n'
  if out=='pin.cuh': p=p.replace('if (wave_id < T::SCALE_N_HALVES) {', 'if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {')
  p=p.replace('static_cast<unsigned int>(kargs.k) / T::GROUP_K','T::FIXED_K / T::GROUP_K').replace('static_cast<unsigned int>(kargs.k) / T::B_K','T::FIXED_K / T::B_K')
  if out=='pad_pin.cuh': p=p.replace('#pragma clang loop unroll_count(T::LOOP_UNROLL)', '#pragma clang loop unroll(disable)')
  if out=='pad_pin.cuh': p=p.replace('#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"','#include "pin.cuh"')
  assert 'const int loops = T::FIXED_K / T::B_K;' in p
  write_variant(out,src,p)
 source_files=[]
 for prefix,destination in [(INC,HERE/'frozen/gemm_include'),(ROOT/'csrc/include/opus',HERE/'frozen/opus')]:
  for p in sorted(prefix.rglob('*')):
   if p.is_file():
    frozen=destination/p.relative_to(prefix);assert sha(p)==sha(frozen)
    source_files.append({'original':str(p.relative_to(ROOT)),'frozen':str(frozen.relative_to(HERE)),'sha256':sha(p)})
 manifest={'status':'prepared_gpu_stopped','cpu_only':True,'registered':False,'frozen_files':source_files,'formal_inputs':{p:sha(ROOT/p) for p in ['aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv','reports/opus_clang23_mixed_retune_20261008/jit/module_deepgemm_opus.so']}}
 (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Frozen',len(source_files),'inputs; generated geometry and fixed-K pin pipelines')
if __name__=='__main__':main()
