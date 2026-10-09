#!/usr/bin/env python3
"""Offline ELF inspection and host-only layout/contract checks; no GPU use."""
import argparse, difflib, hashlib, importlib.util, json, os, re, subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LLVM=Path('/opt/rocm-llvm23-46fcb339/bin')
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-name',default='cpu_audit_no_runtime')
    arg=ap.parse_args();assert Path(arg.output_name).name==arg.output_name
    out=HERE/arg.output_name
    if out.exists(): raise SystemExit('Refusing to overwrite audit')
    out.mkdir()
    parser_path=ROOT/'reports/opus_resume_20261008/sfa_packed/audit_device.py'
    spec=importlib.util.spec_from_file_location('cpu_elf',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    parser.TARGET='hipv4-amdgcn-amd-amdhsa--gfx950'
    receipt=json.loads((HERE/'build_receipt.json').read_text())
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    libs=[]
    for build in receipt['builds']:
        lib=Path(build['library']);assert sha(lib)==build['library_sha256']
        image=parser.bundled_image(lib);device=out/(build['side']+'.co');device.write_bytes(image)
        elf=parser.Elf(image);metadata=elf.metadata();symbols=elf.symbols()
        (out/(build['side']+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        isa_argv=[str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx950',str(device)]
        (out/(build['side']+'_isa.txt')).write_text(subprocess.check_output(isa_argv,env=ENV,text=True))
        kernels=[]
        for row in metadata['amdhsa.kernels']:
            code=symbols[row['.name']];assert code['type']==2 and code['size']>0
            name=subprocess.check_output(['c++filt',row['.name']],env=ENV,text=True).strip()
            keys=['agpr_count','vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']
            kernels.append({'name':name,'symbol':row['.name'],'resources':{k:row.get('.'+k) for k in keys},
                            'isa_bytes':code['size'],'isa_sha256':hashlib.sha256(code['bytes']).hexdigest()})
        assert len(kernels)==(31 if build['side']=='baseline' else 14)
        assert all(k['resources']['private_segment_fixed_size']==0 and k['resources']['vgpr_spill_count']==0 and k['resources']['sgpr_spill_count']==0 for k in kernels)
        libs.append({'side':build['side'],'device_sha256':sha(device),'kernels':kernels,'isa_argv':isa_argv})
    # Host attributes change only the pure layout/adaptor section in an audit copy.
    # The compiled device sources and frozen originals remain unchanged.
    host=out/'host_include/opus';host.mkdir(parents=True)
    original=(HERE/'frozen/opus/opus.hpp').read_text()
    begin=original.index('template<typename FDim, typename Target, index_t I0')
    end=original.index('#undef OPUS_KP_',begin)
    adapted=original[:begin]+original[begin:end].replace('OPUS_D ','OPUS_H_D ')+original[end:]
    (host/'opus.hpp').write_text(adapted)
    (out/'host_attributes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile='frozen/opus/opus.hpp',tofile='host_include/opus/opus.hpp')))
    for name in ['dtypes.hpp','hip_minimal.hpp']:(host/name).write_bytes((HERE/'frozen/opus'/name).read_bytes())
    layout=(HERE/'frozen/gemm_include/gfx950/opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh').read_text()
    begin=layout.index('template<class T>\n__device__ inline constexpr auto make_layout_gb_scale')
    end=layout.index('template<class T>\n__device__ inline constexpr auto make_layout_gsfa_scale',begin)
    helper=out/'frozen_layout_helpers.h'
    helper.write_text('#pragma once\nusing opus::operator""_I;\nnamespace checked_layout {\n'+layout[begin:end].replace('__device__','__host__')+'}\n')
    binary=out/'layout_guard_check'
    host_object=out/'layout_guard_check.o'
    argv=[str(LLVM/'clang++'),'-x','hip','--offload-host-only','--rocm-path=/opt/rocm','--hip-path=/opt/rocm','-std=c++20','-O2','-D__HIPCC_RTC__=1',
          '-I'+str(out),'-I'+str(HERE),'-I'+str(out/'host_include'),'-I'+str(HERE/'frozen'),'-I'+str(HERE/'frozen/gemm_include/gfx950'),'-c',str(HERE/'layout_guard_check.cpp'),'-o',str(host_object)]
    compiled=subprocess.run(argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'layout_compile.log').write_text(compiled.stdout)
    if compiled.returncode: raise RuntimeError(compiled.stdout)
    link_argv=[str(LLVM/'clang++'),str(host_object),'-o',str(binary)]
    linked=subprocess.run(link_argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'layout_link.log').write_text(linked.stdout)
    if linked.returncode: raise RuntimeError(linked.stdout)
    needed=subprocess.check_output(['readelf','-d',str(binary)],env=ENV,text=True)
    (out/'layout_dynamic.txt').write_text(needed)
    needed_names=re.findall(r'Shared library: \[([^\]]+)\]',needed)
    assert all('hip' not in name.lower() and 'hsa' not in name.lower() for name in needed_names),needed_names
    checked=subprocess.check_output([str(binary)],env=ENV,text=True).strip()
    unchanged=all(sha(ROOT/r['original'])==r['sha256'] and sha(HERE/r['frozen'])==r['sha256'] for r in manifest['frozen_files'])
    assert unchanged
    assert all(sha(HERE/p)==h for p,h in receipt['sources'].items())
    report={'status':'cpu_audit_passed_unvalidated_numerics','cpu_only':True,'gpu_operations':0,'registered':False,
            'numerical_validation':'not_run_gpu_stopped','performance_validation':'not_run_gpu_stopped',
            'libraries':libs,'layout_guard_check':{'compile_argv':argv,'link_argv':link_argv,'needed_libraries':needed_names,
            'no_hip_hsa_dependencies':True,'binary_sha256':sha(binary),'helper_sha256':sha(helper),'stdout':checked,
            'host_attribute_diff_sha256':sha(out/'host_attributes.diff')},'production_sources_unchanged':unchanged,
            'build_sources_unchanged':True,'elf_parser_reference':{'path':str(parser_path),'sha256':sha(parser_path)}}
    (HERE/'cpu_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'kernel_counts':{x['side']:len(x['kernels']) for x in libs},'host_check':checked}))
if __name__=='__main__':main()
