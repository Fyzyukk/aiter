#!/usr/bin/env python3
"""Relink current OPUS with only the 24 unpinned public BF16 device TUs changed."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, shlex, shutil, subprocess, time
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/'reports/opus_clang23_mixed_retune_20261008'
SOURCE=RUN/'jit/build/module_deepgemm_opus/build'
BUILD=OUT/'clang24_device_build'
PIN='/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++'
RESOURCE='/opt/rocm/lib/llvm/lib/clang/20'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((RUN/'opus_per_object_compiler_audit.json').read_text())
    selected=[x for x in inventory if x['compiler_major']==23]
    assert len(selected)==24
    all_objects=sorted(SOURCE.glob('*.o'))
    assert len(all_objects)==206
    for obj in all_objects: shutil.copy2(obj,BUILD/obj.name)
    jobs=[]
    for item in selected:
        old=Path(item['object']); argv=shlex.split(item['command'])[1:]
        argv[argv.index('-o')+1]=str(BUILD/old.name)
        command=[PIN,'-x','hip','--rocm-path=/opt/rocm','--hip-path=/opt/rocm',f'-resource-dir={RESOURCE}',*argv]
        source=Path(argv[argv.index('-c')+1])
        jobs.append({'kid':item['kid'],'source':str(source),'source_sha256':sha(source),'old_object_sha256':sha(old),'object':str(BUILD/old.name),'argv':command})
    def compile_one(job):
        start=time.monotonic()
        result=subprocess.run(job['argv'],cwd=BUILD,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (BUILD/f"kid{job['kid']}.log").write_text(result.stdout)
        if result.returncode: raise RuntimeError(f"kid{job['kid']}: {result.stdout[-4000:]}")
        job.update(seconds=time.monotonic()-start,sha256=sha(job['object']),elf_comment=subprocess.check_output(['readelf','-p','.comment',job['object']],text=True))
        assert 'clang version 24.' in job['elf_comment']
        print(json.dumps({'kid':job['kid'],'seconds':job['seconds']}),flush=True)
        return job
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool: receipts=list(pool.map(compile_one,jobs))
    ninja=(SOURCE/'build.ninja').read_text()
    ldflags=next(line.split(' = ',1)[1] for line in ninja.splitlines() if line.startswith('ldflags = '))
    link_objects=next(line.split(': link ',1)[1].split() for line in ninja.splitlines() if line.startswith('build module_deepgemm_opus.so: link '))
    response=BUILD/'link.rsp'; response.write_text('\n'.join(shlex.quote(str(BUILD/name)) for name in link_objects)+'\n')
    # CPython caches single-phase extension modules by their init name. Give
    # the alternate pybind entry point a distinct name, using the SAME Clang23
    # wrapper flags and source, changing only its two module-name definitions.
    wrapper_name='opus_gemm_pybind.cuda.o'
    commands=(RUN/'module_deepgemm_opus_commands.txt').read_text().splitlines()
    wrapper_command=next(line for line in commands if f'-o {wrapper_name} ' in line)
    wrapper_argv=shlex.split(wrapper_command)[1:]
    wrapper_argv=[v.replace('NAME=module_deepgemm_opus','NAME=module_deepgemm_opus_ab24') for v in wrapper_argv]
    wrapper_argv[wrapper_argv.index('-o')+1]=str(BUILD/wrapper_name)
    wrapper_build=['/opt/rocm-llvm23-46fcb339/bin/clang++','-x','hip','--rocm-path=/opt/rocm','--hip-path=/opt/rocm',*wrapper_argv]
    subprocess.run(wrapper_build,check=True,cwd=BUILD)
    destination=OUT/'clang24/module_deepgemm_opus_ab24.so';destination.parent.mkdir(exist_ok=True)
    command=['c++',f'@{response}',*shlex.split(ldflags),'-o',str(destination)]
    subprocess.run(command,check=True,cwd=BUILD)
    changed={Path(x['object']).name for x in receipts}|{wrapper_name}
    unchanged=[{'object':x.name,'sha256':sha(x),'copied_sha256':sha(BUILD/x.name)} for x in all_objects if x.name not in changed]
    assert all(x['sha256']==x['copied_sha256'] for x in unchanged)
    baseline=RUN/'jit/module_deepgemm_opus.so'
    manifest={'status':'completed','method':'same 206 objects and link flags; recompile 24 unpinned public BF16 device TUs; 181 objects byte-identical including dispatch host and all 4 pin TUs; pybind wrapper same Clang23/source/flags except two module-name macros to avoid CPython single-phase module cache reuse','wrapper_argv':wrapper_build,'compiler24':{'path':PIN,'sha256':sha(PIN),'version':subprocess.check_output([PIN,'--version'],text=True),'resource_dir':RESOURCE},'compiler23':json.loads((RUN/'compiler_manifest.json').read_text())['compilers']['baseline'],'objects_changed':receipts,'objects_unchanged':unchanged,'link_argv':command,'libraries':{'clang23':str(baseline),'clang24':str(destination)},'library_sha256':{'clang23':sha(baseline),'clang24':sha(destination)},'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('BUILD_COMPLETED',flush=True)
if __name__=='__main__':main()
