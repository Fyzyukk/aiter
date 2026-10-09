#!/usr/bin/env python3
"""Inspect existing same-source compiler9070 libraries on CPU only."""
import csv,hashlib,importlib.util,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LLVM=Path('/opt/rocm-llvm23-46fcb339/bin')
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    folder=HERE/'compiler9070_v2';out=folder/'cpu_audit'
    if out.exists():raise SystemExit('Refusing to overwrite compiler audit')
    out.mkdir();receipt=json.loads((folder/'build_receipt.json').read_text())
    parser_path=ROOT/'reports/opus_resume_20261008/sfa_packed/audit_device.py'
    spec=importlib.util.spec_from_file_location('cpu_elf',parser_path);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    libs=[]
    for build in receipt['builds']:
        lib=Path(build['library']);assert sha(lib)==build['library_sha256']
        image=None
        for spelling in ['hipv4-amdgcn-amd-amdhsa--gfx950','hip-amdgcn-amd-amdhsa--gfx950']:
            parser.TARGET=spelling
            try:image=parser.bundled_image(lib);break
            except ValueError:pass
        assert image is not None
        device=out/(build['side']+'.co');device.write_bytes(image)
        elf=parser.Elf(image);meta=elf.metadata();symbols=elf.symbols()
        assert len(meta['amdhsa.kernels'])==1
        row=meta['amdhsa.kernels'][0];symbol=symbols[row['.name']]
        name=subprocess.check_output(['c++filt',row['.name']],env=ENV,text=True).strip()
        assert '<16, 32, 1, 1, 3, 8, 4, 3, 7168, false, true>' in name
        keys=['vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']
        resource={key:row.get('.'+key) for key in keys}
        assert resource['private_segment_fixed_size']==resource['vgpr_spill_count']==resource['sgpr_spill_count']==0
        (out/(build['side']+'_metadata.json')).write_text(json.dumps(meta,indent=2)+'\n')
        isa_argv=[str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx950',str(device)]
        (out/(build['side']+'_isa.txt')).write_text(subprocess.check_output(isa_argv,env=ENV,text=True))
        libs.append({'side':build['side'],'bundle_target':parser.TARGET,'symbol':row['.name'],'name':name,'resources':resource,
                     'device_sha256':sha(device),'isa_bytes':symbol['size'],'isa_sha256':hashlib.sha256(symbol['bytes']).hexdigest(),'isa_argv':isa_argv})
    shapes=[r for r in csv.DictReader((HERE/'frozen/losers294.csv').open()) if r['actual_kid']=='9070' and r['clang24_local_gain_verified']=='True']
    assert len(shapes)==3
    with (folder/'coverage.csv').open('w') as f:
        data=[{'M':r['M'],'N':r['N'],'K':r['K'],'parent':r['opus_parent_kid'],'actual':r['actual_kid'],'control_id':9052,
               'clang23_library':receipt['builds'][0]['library'],'clang24_library':receipt['builds'][1]['library'],
               'workspace_bytes':0,'kernel_calls':1,'numerical_validation':'pending_gpu_stop','performance_validation':'pending_gpu_stop'} for r in shapes]
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    same_symbols=libs[0]['symbol']==libs[1]['symbol'];assert same_symbols
    final={'status':'cpu_elf_audit_passed_unvalidated_numerics','cpu_only':True,'gpu_operations':0,'registered':False,
           'shapes':3,'same_actual9070_source':True,'same_kernel_symbol':same_symbols,
           'code_hashes_equal':libs[0]['isa_sha256']==libs[1]['isa_sha256'],'libraries':libs,
           'source_sha256':receipt['source_sha256'],'coverage_sha256':sha(folder/'coverage.csv'),
           'numerical_validation':'not_run_gpu_stopped','performance_validation':'not_run_gpu_stopped'}
    (folder/'cpu_audit.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({'status':final['status'],'shapes':3,'libraries':[{k:r[k] for k in ['side','resources','isa_sha256']} for r in libs]}))
if __name__=='__main__':main()
