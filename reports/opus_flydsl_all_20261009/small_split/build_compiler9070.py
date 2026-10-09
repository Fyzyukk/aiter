#!/usr/bin/env python3
"""Same-source actual9070 Clang23/24 offline compiler comparison only."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'reports/opus_compiler_ab_baseline_review_20261008/build_manifest.json'
ENV=dict(os.environ,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    output=HERE/'compiler9070_v2'
    if output.exists():raise SystemExit('Refusing to overwrite compiler pair')
    output.mkdir()
    old=json.loads(OLD.read_text());(output/'old_build_manifest.json').write_bytes(OLD.read_bytes())
    reference=next(r for r in old['objects_changed'] if r['kid']==9052)
    options=reference['argv'][6:]
    # Preserve production defines and codegen flags. Redirect include paths to
    # the frozen private source tree; replace only TU/object output. The old
    # irrelevant blob-staging include remains and contains no used headers.
    rewrite={'-I'+str(ROOT/'csrc/include'):'-I'+str(HERE/'frozen'),
             '-I'+str(ROOT/'csrc/opus_gemm/include'):'-I'+str(HERE/'frozen/gemm_include')}
    options=[rewrite.get(x,x) for x in options]
    options.insert(0,'-I'+str(HERE/'frozen/gemm_include/gfx950'))
    options.insert(0,'-I'+str(HERE))
    options[options.index('-c')+1]=str(HERE/'compiler9070.hip')
    compilers={'clang23':Path('/opt/rocm-llvm23-46fcb339/bin/clang++'),
               'clang24':Path(old['compiler24']['path'])}
    resource=old['compiler24']['resource_dir'];builds=[]
    report={'status':'building','cpu_only':True,'gpu_operations':0,'registered':False,
            'source':str(HERE/'compiler9070.hip'),'source_sha256':sha(HERE/'compiler9070.hip'),
            'old_build_manifest':{'path':str(OLD),'sha256':sha(OLD),'kid':9052},
            'method':'same private actual9070 TU, same production defines/codegen flags from old kid9052 manifest; private frozen include redirection; compiler24 pinned resource20; no production library byte-identity claim',
            'numerical_validation':'not_run_gpu_stopped','performance_validation':'not_run_gpu_stopped','builds':builds}
    for label,compiler in compilers.items():
        folder=output/label;folder.mkdir();obj=folder/'launch.o';lib=folder/'experiments.so'
        local=list(options);local[local.index('-o')+1]=str(obj)
        argv=[str(compiler),'-x','hip','--rocm-path=/opt/rocm','--hip-path=/opt/rocm']
        if label=='clang24':argv+=['-resource-dir='+resource]
        argv+=local
        link=[str(compiler),'-shared',str(obj),'-L/opt/rocm/lib','-lamdhip64','-Wl,--build-id=sha1','-o',str(lib)]
        row={'side':label,'compiler':str(compiler),'compiler_sha256':sha(compiler),
             'version':subprocess.check_output([str(compiler),'--version'],env=ENV,text=True),
             'resource_dir':resource if label=='clang24' else subprocess.check_output([str(compiler),'-print-resource-dir'],env=ENV,text=True).strip(),
             'compile_argv':argv,'link_argv':link,'object':str(obj),'library':str(lib)}
        builds.append(row)
        for phase in ['compile','link']:
            start=time.monotonic();done=subprocess.run(row[phase+'_argv'],cwd=HERE,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            logfile=folder/(phase+'.log');logfile.write_text(done.stdout)
            row[phase+'_returncode']=done.returncode;row[phase+'_seconds']=time.monotonic()-start;row[phase+'_log_sha256']=sha(logfile)
            if done.returncode:
                report['status']='failed_'+phase;(output/'build_receipt.json').write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError(done.stdout)
        row['object_sha256']=sha(obj);row['library_sha256']=sha(lib)
        print(json.dumps({'side':label,'offline_build':'passed'}),flush=True)
    assert sha(HERE/'compiler9070.hip')==report['source_sha256']
    report['status']='offline_build_passed_unvalidated_numerics'
    (output/'build_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
