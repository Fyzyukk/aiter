#!/usr/bin/env python3
"""CPU-build private register local-wave plus last-arrival global split-K."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config, config_identity, config_traits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination=HERE/'fused_wavek'
    destination.mkdir(exist_ok=False)
    lock=(HERE/'cache_build.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    plan=json.loads((HERE/'plan.json').read_text())
    for path,expected in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha(ROOT/path)==expected,path
    parent=HERE/'fused_register/pipeline.cuh'
    body=parent.read_text().replace('opus745_fused_register_detail','opus745_fused_wavek_detail')
    body=body.replace('opus745_fused_kargs','opus745_fused_wavek_kargs')
    old='''            if (wk > 0) return;
            static_for<T::WAVE_K - 1>([&](auto sk) {
                static_for<T::E_M * T::E_N>([&](auto i) {
                    c[decltype(i)::value] += load<elem_c>(shared,
                        decltype(sk)::value * elems + offset + decltype(i)::value * 64 * elem_c);
                });
            });'''
    new='''            // All K waves stay live for the later group arrival barriers.
            // Only K-wave zero owns the local FP32 sum and global fragment.
            if (wk == 0) {
                static_for<T::WAVE_K - 1>([&](auto sk) {
                    static_for<T::E_M * T::E_N>([&](auto i) {
                        c[decltype(i)::value] += load<elem_c>(shared,
                            decltype(sk)::value * elems + offset + decltype(i)::value * 64 * elem_c);
                    });
                });
            }'''
    assert body.count(old)==1
    body=body.replace(old,new)
    old='''                opus_gemm_small_output::store_mfma16_packed<T>(c, gc, lane, wm, wn, args.stride_c);
                return;'''
    new='''                if (wk == 0)
                    opus_gemm_small_output::store_mfma16_packed<T>(c, gc, lane, wm, wn, args.stride_c);
                return;'''
    assert body.count(old)==1
    body=body.replace(old,new)
    body=body.replace('            static_assert(T::WAVE_K == 1);','            static_assert(T::WAVE_K > 1);')
    body=body.replace('(fragment * T::BLOCK_SIZE + thread_id_x())',
                      '(fragment * (T::BLOCK_SIZE / T::WAVE_K) + thread_id_x())')
    old='''            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int fragment = decltype(fi)::value;
                const int offset = split * tile_elements +
                                   (fragment * (T::BLOCK_SIZE / T::WAVE_K) + thread_id_x()) * elem_c;
                store<elem_c>(slots, c[fragment], offset);
            });'''
    assert body.count(old)==1
    body=body.replace(old,'''            if (wk == 0) {
'''+old+'''
            }''')
    body=body.replace('''            if (last_arrival) {
                array''','''            if (last_arrival) {
                if (wk == 0) {
                array''')
    old='''                opus_gemm_small_output::store_mfma16_packed<T>(sum, gc, lane, wm, wn, args.stride_c);
                // Reset'''
    assert body.count(old)==1
    body=body.replace(old,'''                opus_gemm_small_output::store_mfma16_packed<T>(sum, gc, lane, wm, wn, args.stride_c);
                }
                // Reset''')
    (destination/'pipeline.cuh').write_text(body)
    kargs=(HERE/'fused_register/fused_kargs.cuh').read_text().replace('opus745_fused_kargs','opus745_fused_wavek_kargs')
    (destination/'fused_kargs.cuh').write_text(kargs)
    configs=[]
    for tm,tn,wk in ((16,16,2),(16,16,4),(16,32,2),(16,32,4),(32,32,2),(32,32,4)):
        c=construct_config('register',tile_m=tm,tile_n=tn,wave_m=1,wave_n=1,wave_k=wk,
                           prefetch=3,runtime_split_k=True,b_cache=2,dispatch=(),specializations=())
        configs.append({'id':len(configs),'identity':config_identity(c),'pipeline':c.pipeline,
                        'compile_params':json.loads(c.to_json())})
    lines=['// SPDX-License-Identifier: Apache-2.0','#include <cstdint>',
           '#include <opus/hip_minimal.hpp>','#include "fused_kargs.cuh"','#include "pipeline.cuh"']
    for row in configs:
        i=row['id'];c=construct_config(row['pipeline'],config=row['compile_params'])
        lines.append(f'''using FusedTraits{i} = opus_gemm_mxscale_bpreshuffle_pipeline_traits<{config_traits(c)},1>;
__global__ __launch_bounds__(FusedTraits{i}::BLOCK_SIZE,1)
void fused_wavek_kernel{i}(opus745_fused_wavek_kargs args) {{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_wavek_detail::run<FusedTraits{i}>(args);
#endif
}}''')
    launch=(HERE/'fused_register/launch.hip').read_text()
    host=launch[launch.index('#if !defined(__HIP_DEVICE_COMPILE__)\ntemplate<class T>'):]
    host=host.replace('opus745_fused_kargs','opus745_fused_wavek_kargs')
    host=host[:host.index('    switch (variant) {')]+'    switch (variant) {\n'
    # This private branch keeps the same <=512 support range as its controls.
    host=host.replace('m <= 0 || n <= 0','m <= 0 || m > 512 || n <= 0')
    for row in configs:
        i=row['id'];host+=f'    case {i}:return launch_one<FusedTraits{i}>(args,stream,reinterpret_cast<const void*>(fused_wavek_kernel{i}));\n'
    host+='    default:return 1;\n    }\n}\n#endif\n'
    source=destination/'launch.hip';source.write_text('\n'.join(lines)+'\n'+host)
    reference=json.loads((HERE/'fused_register/build.json').read_text())
    argv=[v for v in reference['argv'][:reference['argv'].index('-shared')] if not v.startswith('-I')]
    argv+=['-I'+str(ROOT/'csrc/include'),'-I'+str(ROOT/'csrc/opus_gemm/include'),
           '-I'+str(destination),'-shared',str(source),'-o',str(destination/'experiments.so')]
    started=time.monotonic();env={**os.environ,'HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':''}
    proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (destination/'build.log').write_text(proc.stdout)
    report={'returncode':proc.returncode,'argv':argv,'seconds':time.monotonic()-started,
            'parent_source':str(parent),'parent_source_sha256':sha(parent),'configurations':configs,
            'kernel_launches':0,'config_library_loads':0,'plan_sha256':sha(HERE/'plan.json'),
            'sources':{p.name:sha(p) for p in destination.iterdir() if p.suffix in ('.hip','.cuh')},
            'numerical_validation':'pending_idle_gpu','performance_validation':'pending_idle_gpu'}
    if proc.returncode==0:
        artifact=destination/'experiments.so';report['library_sha256']=sha(artifact)
        def module(name,path):
            spec=importlib.util.spec_from_file_location(name,path)
            result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
            return result
        audit=module('fused_wavek745_audit',HERE/'audit_cache.py')
        parser=module('fused_wavek745_elf',ROOT/'reports/opus_resume_20261008/sfa_packed/audit_device.py')
        image,targets=audit.device_image(parser,artifact.read_bytes())
        (destination/'device.co').write_bytes(image)
        report['targets']=targets;report['device_sha256']=hashlib.sha256(image).hexdigest()
        metadata=parser.Elf(image).metadata()['amdhsa.kernels']
        report['kernels']=[{'name':r['.name'],'resources':{key:r.get('.'+key,0) for key in
          ('vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count')}} for r in metadata]
        tool=Path(argv[0]).with_name('llvm-objdump')
        proc=subprocess.run([str(tool),'-d','--mcpu=gfx950',str(destination/'device.co')],capture_output=True,text=True,check=True)
        (destination/'device.s').write_text(proc.stdout)
    (destination/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    lock.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ('configurations','argv','sources')}))
    return report['returncode']


if __name__=='__main__':
    raise SystemExit(main())
