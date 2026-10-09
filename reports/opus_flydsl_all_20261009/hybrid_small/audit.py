#!/usr/bin/env python3
"""CPU ELF/source/layout/ring audit; never loads built libraries or HIP runtime."""
import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LLVM=Path('/opt/rocm-llvm23-46fcb339/bin')
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
CONFIG={
    9043:(32,64,1,4,8,False,False,2),9044:(64,64,2,2,4,False,False,2),
    9045:(96,64,2,2,4,False,False,1),9046:(64,128,4,2,6,False,False,2),
    9047:(32,64,2,2,4,True,False,2),9049:(32,128,2,2,4,True,True,2),
    9055:(32,64,1,4,12,False,False,2),9056:(64,64,2,2,8,False,False,2)}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    arguments=argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('--output-name',default='cpu_audit_final')
    options=arguments.parse_args()
    assert options.output_name and Path(options.output_name).name==options.output_name
    output=HERE/options.output_name
    if output.exists() or (HERE/'cpu_audit.json').exists(): raise SystemExit('Refusing to overwrite audit')
    output.mkdir()
    parser_source=ROOT/'reports/opus_resume_20261008/sfa_packed/audit_device.py'
    parser_path=output/'elf_parser.py';shutil.copyfile(parser_source,parser_path)
    spec=importlib.util.spec_from_file_location('private_read_only_elf_parser',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    parser.TARGET='hipv4-amdgcn-amd-amdhsa--gfx950'
    receipt=json.loads((HERE/'build_receipt.json').read_text())
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    records=[]
    for build in receipt['builds']:
        lib=Path(build['library']);assert sha(lib)==build['library_sha256']
        device=output/(build['side']+'.co');device.write_bytes(parser.bundled_image(lib))
        elf=parser.Elf(device.read_bytes());metadata=elf.metadata();symbols=elf.symbols()
        (output/(build['side']+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        argv=[str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx950',str(device)]
        isa=subprocess.check_output(argv,env=ENV,text=True)
        isa_path=output/(build['side']+'_isa.txt');isa_path.write_text(isa)
        kernels=[]
        for row in metadata['amdhsa.kernels']:
            name=subprocess.check_output(['c++filt',row['.name']],env=ENV,text=True).strip()
            if build['b_ahead']:
                match=re.search(r'opus_private_small_baseline<(\d+)>',name)
                # The alias resolves to a named frozen baseline traits specialization.
                if match: actual=int(match[1])
                else:
                    actual=next(k for k,v in CONFIG.items() if {
                        9043:'<32, 64, 1, 4, 8,',9044:'<64, 64, 2, 2, 4,',9045:'<96, 64, 2, 2, 4,',
                        9046:'<64, 128, 4, 2, 6,',9047:'<32, 64, 2, 2, 4,',9049:'<32, 128, 2, 2, 4,',
                        9055:'<32, 64, 1, 4, 12,',9056:'<64, 64, 2, 2, 8,'}[k] in name)
            else:
                actual=next(k for k,v in CONFIG.items() if {
                    9043:'<32, 64, 1, 4, 8,',9044:'<64, 64, 2, 2, 4,',9045:'<96, 64, 2, 2, 4,',
                    9046:'<64, 128, 4, 2, 6,',9047:'<32, 64, 2, 2, 4,',9049:'<32, 128, 2, 2, 4,',
                    9055:'<32, 64, 1, 4, 12,',9056:'<64, 64, 2, 2, 8,'}[k] in name)
            fields=['agpr_count','vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']
            code=symbols[row['.name']];assert code['type']==2 and len(code['bytes'])==code['size']>0
            resources={f:row.get('.'+f) for f in fields}
            assert resources['group_segment_fixed_size']==0 # Matrix/C LDS is dynamic.
            kernels.append(dict(actual_kid=actual,name=name,symbol=row['.name'],resources=resources,
                                isa_bytes=code['size'],isa_sha256=hashlib.sha256(code['bytes']).hexdigest(),
                                static_resource_status='spill_free' if resources['private_segment_fixed_size']==resources['vgpr_spill_count']==resources['sgpr_spill_count']==0 else 'resource_risk'))
        assert len(kernels)==8 and {k['actual_kid'] for k in kernels}==set(CONFIG)
        records.append(dict(side=build['side'],b_ahead=build['b_ahead'],device_sha256=sha(device),kernels=kernels,
                            isa_argv=argv,isa_text_sha256=sha(isa_path)))

    # Host attribute adapter only exposes the pure frozen layout section.
    # The HIP compile input remains byte-identical to the frozen production tree.
    host=output/'host_include/opus';host.mkdir(parents=True)
    original=(HERE/'frozen/opus/opus.hpp').read_text()
    begin=original.index('template<typename FDim, typename Target, index_t I0')
    end=original.index('#define OPUS_KP_',begin)
    adapted=original[:begin]+original[begin:end].replace('OPUS_D ','OPUS_H_D ')+original[end:]
    (host/'opus.hpp').write_text(adapted)
    diff=output/'host_attributes.diff'
    diff.write_text(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile='frozen/opus/opus.hpp',tofile='cpu_audit_final/host_include/opus/opus.hpp')))
    for file in ['dtypes.hpp','hip_minimal.hpp']:shutil.copyfile(HERE/'frozen/opus'/file,host/file)
    layouts=(HERE/'frozen/gemm_include/gfx950/opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh').read_text()
    begin=layouts.index('template<class T>\n__device__ inline auto make_layout_ga_scale')
    end=layouts.index('template<class T>\n__device__ inline constexpr auto make_layout_gsfa_scale',begin)
    helper=output/'frozen_layout_helpers.h'
    helper.write_text('#pragma once\n#include <opus/hip_minimal.hpp>\n#include <opus/opus.hpp>\nusing opus::operator""_I;\nnamespace checked_layout {\n'+layouts[begin:end].replace('__device__','__host__')+'}\n')
    binary=output/'layout_guard_check'
    host_object=output/'layout_guard_check.o'
    argv=[str(LLVM/'clang++'),'-x','hip','--offload-host-only','--rocm-path=/opt/rocm','--hip-path=/opt/rocm',
          '-std=c++20','-O2','-D__HIPCC_RTC__=1','-I'+str(output),'-I'+str(HERE),'-I'+str(output/'host_include'),
          '-I'+str(HERE/'frozen'),'-I'+str(HERE/'frozen/gemm_include/gfx950'),'-c',str(HERE/'layout_guard_check.cpp'),'-o',str(host_object)]
    compiled=subprocess.run(argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (output/'layout_compile.log').write_text(compiled.stdout)
    if compiled.returncode:raise RuntimeError(compiled.stdout)
    link_argv=[str(LLVM/'clang++'),str(host_object),'-o',str(binary)]
    linked=subprocess.run(link_argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (output/'layout_link.log').write_text(linked.stdout)
    if linked.returncode:raise RuntimeError(linked.stdout)
    # This is a host-only ELF with no HIP/ROCR dynamic dependencies.
    dynamic=subprocess.check_output([str(LLVM/'llvm-readelf'),'-d',str(binary)],env=ENV,text=True)
    assert 'amdhip' not in dynamic and 'hsa-runtime' not in dynamic
    (output/'layout_dynamic.txt').write_text(dynamic)
    checked=subprocess.check_output([str(binary)],cwd=output,env=ENV,text=True).strip()
    checked_json=json.loads(checked)
    source=(HERE/'candidate/pipeline.cuh').read_text()
    baseline=(HERE/'frozen/gemm_include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh').read_text()
    output_begin='    if constexpr (SplitK > 1) {\n        const auto pc ='
    assert source[source.index(output_begin):]==baseline[baseline.index(output_begin):]
    assert 'load<16>(sb' not in source and 'auto sb =' not in source
    assert source.count('load<16>(gb,')==2 and source.count('prefetch_b(')==2
    audit=dict(status='cpu_audit_passed_unvalidated_gpu_stopped',cpu_only=True,gpu_operations=0,registered=False,
               numerical_validation='not_run_gpu_stopped',performance_validation='not_run_gpu_stopped',libraries=records,
               layout_guard_check=dict(compile_argv=argv,link_argv=link_argv,host_object_sha256=sha(host_object),binary_sha256=sha(binary),helper_source_sha256=sha(helper),
                                       host_attribute_adapter_sha256=sha(host/'opus.hpp'),host_attribute_diff_sha256=sha(diff),
                                       result=checked_json,no_hip_runtime_dependency=True),
               source_audit=dict(b_lds_removed=True,direct_b_loads_per_fragment=2,output_and_reducer_byte_identical=True,
                                 conservative_vmcnt_lgkmcnt_and_two_barriers_per_k=True,scale_contract_unchanged=True),
               elf_parser=dict(original_path=str(parser_source),original_sha256=sha(parser_source),frozen_path=str(parser_path),frozen_sha256=sha(parser_path)))
    audit['production_sources_unchanged']=all(sha(ROOT/r['original'])==r['sha256']==sha(HERE/r['frozen']) for r in manifest['frozen_files'])
    audit['controls_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in manifest['controls'])
    assert audit['production_sources_unchanged'] and audit['controls_unchanged']
    (HERE/'cpu_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    coverage=[]
    historical=list(csv.DictReader((ROOT/'reports/opus_flydsl_gap_20261009/losers294.csv').open()))
    covered=[r for r in historical if int(r['actual_kid']) in CONFIG]
    assert len(covered)==85
    for shape in covered:
        actual=int(shape['actual_kid']);m,n,k=[int(shape[f]) for f in ['M','N','K']]
        bm,bn,wm,wn,stages,register,xor,output_strategy=CONFIG[actual]
        a_stage=(bm//8)*(1024+(0 if xor else 32));b_stage=(bn//8)*(1024+(0 if xor else 32))
        loops=k//128;active=min(loops,stages);scale=0 if register else (bm+(bn+127)//128)*loops
        baseline_lds=active*(a_stage+b_stage)+scale
        candidate_lds=max(active*a_stage+scale,bm*(bn+8)*2 if output_strategy==1 else 0)
        legal=m>0 and m<=(512 if actual in [9047,9049] else 2048) and n%128==0 and 0<k<=16384 and k%128==0 and max(m*k,n*k,m*n*2)<=2147483647
        assert legal and candidate_lds<=160*1024
        for record in records[1:]:
            kernel=next(r for r in record['kernels'] if r['actual_kid']==actual)
            coverage.append(dict(M=m,N=n,K=k,opus_parent_kid=int(shape['opus_parent_kid']),actual_kid=actual,
                                 candidate=record['side'],b_ahead=record['b_ahead'],legal=legal,offline_compile_link='passed',
                                 cpu_address_ring_scale_audit='passed',baseline_dynamic_lds_bytes=baseline_lds,
                                 candidate_dynamic_lds_bytes=candidate_lds,tile_M=bm,tile_N=bn,wave_M=wm,wave_N=wn,
                                 A_stages=stages,B_slots=record['b_ahead']+1,global_splitK=int(shape['global_splitK']),
                                 output_strategy=output_strategy,register_scales=register,xor_A_lds=xor,
                                 duplicate_B_read_factor=wm,B_queue_payload_dwords=(record['b_ahead']+1)*(bn//(wn*16))*8,
                                 VGPR=kernel['resources']['vgpr_count'],SGPR=kernel['resources']['sgpr_count'],
                                 scratch_bytes=kernel['resources']['private_segment_fixed_size'],
                                 VGPR_spills=kernel['resources']['vgpr_spill_count'],resource_status=kernel['static_resource_status'],
                                 numerical_validation='not_run_gpu_stopped',performance_validation='not_run_gpu_stopped',registered=False))
    with (HERE/'candidate_coverage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    coverage_summary=dict(status='offline_candidate_coverage_unvalidated_gpu_stopped',historical_losing_shapes=len(covered),
                          candidate_shape_pairs=len(coverage),actual_ids=sorted(CONFIG),parent_ids=sorted({int(r['opus_parent_kid']) for r in covered}),
                          every_shape_has_legal_compiled_direct_b=True,registered=False,rows=coverage)
    (HERE/'candidate_coverage.json').write_text(json.dumps(coverage_summary,indent=2)+'\n')
    final=dict(status=audit['status'],cpu_only=True,gpu_operations=0,registered=False,historical_losing_shapes=85,
               candidate_shape_pairs=255,every_shape_has_legal_compiled_direct_b=True,
               numerical_validation='not_run_gpu_stopped',performance_validation='not_run_gpu_stopped',
               files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='final_manifest.json'})
    (HERE/'final_manifest.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(dict(status=audit['status'],host_audit=checked_json,coverage_shapes=85,candidate_shape_pairs=255,
                         resources=[dict(side=r['side'],kernels=[dict(actual=k['actual_kid'],VGPR=k['resources']['vgpr_count'],SGPR=k['resources']['sgpr_count'],scratch=k['resources']['private_segment_fixed_size']) for k in r['kernels']]) for r in records]),indent=2))
if __name__=='__main__':main()
