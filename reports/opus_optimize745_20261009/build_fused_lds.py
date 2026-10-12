#!/usr/bin/env python3
"""CPU-build private LDS last-arrival epilogues against existing public controls."""
import ast
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
sys.path.insert(0, str(ROOT))
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config, config_traits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = HERE / "fused_lds"
    destination.mkdir(exist_ok=False)
    lock = (HERE / "cache_build.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    plan = json.loads((HERE / "plan.json").read_text())
    for path, expected in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        assert sha(ROOT / path) == expected, path
    geometries = {(48,64,1,4),(64,64,2,2),(64,128,2,2)}
    configs = []
    for row in plan["configs"].values():
        c = row["compile_params"]
        geometry = tuple(c[k] for k in ("tile_m","tile_n","wave_m","wave_n"))
        if row["pipeline"] == "lds" and c["runtime_split_k"] and geometry in geometries and c["stages"] in (3,4):
            configs.append({**row, "id":len(configs)})
    assert len(configs) == 6
    source_path = ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_lds_gfx950.cuh"
    original = source_path.read_text()
    body = original[:original.index("\n#if !defined(__HIP_DEVICE_COMPILE__)\ntemplate<class T>")]
    for name in ("opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh",
                 "opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh",
                 "opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh",
                 "opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh"):
        body = body.replace('"'+name+'"', '"gfx950/'+name+'"')
    body = body.replace('"opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh"', '"fused_kargs.cuh"')
    body = body.replace("opus_gemm_bpreshuffle_lds_detail", "opus745_fused_lds_detail")
    body = body.replace("std::is_same_v<Kargs, opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950>",
                        "std::is_same_v<Kargs, opus745_fused_lds_kargs>")
    body = body.replace("    const int split = Partial ? block_id_z() : 0;", """    const int n_tiles = args.n / T::B_N;
    const int m_tiles = (args.m + T::B_M - 1) / T::B_M;
    const int bid = block_id_x();
    const int tile = bid / (8 * args.split_k) * 8 + bid % 8;
    if (tile >= m_tiles * n_tiles) return;
    const int split = bid / 8 % args.split_k;""")
    old = "    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;"
    assert body.count(old) == 1
    body = body.replace(old, "    const int row = tile / n_tiles * T::B_M, col = tile % n_tiles * T::B_N;")
    body = body.replace(" + static_cast<int64_t>(split) * args.m * args.stride_c", "")
    body = body.replace("    using Base = typename decltype(mma)::MMA;",
                        "    using Base = typename decltype(mma)::MMA;\n    constexpr int elem_c = 4;")
    register_source = HERE / "build_fused_register.py"
    tree = ast.parse(register_source.read_text())
    value = next(n.value for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == "FUSED_OUTPUT" for t in n.targets))
    epilogue = ast.literal_eval(value)
    epilogue = epilogue[:epilogue.index("        } else if constexpr (T::OUTPUT == 3) {")] + "        }\n"
    epilogue = epilogue.replace("T::WAVE_K == 1", "T::T_K == 1")
    begin = body.index("    if constexpr (Partial) {\n")
    end = body.rindex("\n}\n} // namespace")
    body = body[:begin] + "    static_assert(!Partial);\n" + epilogue + body[end:]
    assert "block_id_y()" not in body and "block_id_z()" not in body
    (destination / "pipeline.cuh").write_text(body)
    (destination / "fused_kargs.cuh").write_text('''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gfx950/opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh"
struct opus745_fused_lds_kargs : opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950 {
    float* ptr_workspace;
    int* ptr_counters;
};
''')
    lines = ['// SPDX-License-Identifier: Apache-2.0','#include <cstdint>',
             '#include <opus/hip_minimal.hpp>','#include "fused_kargs.cuh"','#include "pipeline.cuh"']
    for row in configs:
        config = construct_config("lds",config=row["compile_params"])
        i = row["id"]
        lines.append(f'''using FusedTraits{i} = opus_gemm_mxscale_bpreshuffle_pipeline_traits<{config_traits(config)},1>;
__global__ __launch_bounds__(FusedTraits{i}::BLOCK_SIZE,1)
void fused_lds_kernel{i}(opus745_fused_lds_kargs args) {{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits{i},false>(args);
#endif
}}''')
    lines.append('''#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> int launch_one(opus745_fused_lds_kargs args, void* stream, const void* kernel) {
    const int tiles = ((args.m + T::B_M - 1) / T::B_M) * (args.n / T::B_N);
    const dim3 grid(((tiles + 7) / 8) * 8 * args.split_k);
    void* argv[] = {&args};
    return hipLaunchKernel(kernel,grid,dim3(T::BLOCK_SIZE),argv,T::lds_bytes(args.k,args.split_k),stream);
}
extern "C" __attribute__((visibility("default")))
int launch(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
           void* c,float* workspace,int* counters,int m,int n,int k,int split_k,void* stream) {
    if (m<=0 || m>2048 || n<=0 || k<128 || k>16384 || n%128 || k%128 ||
        split_k<1 || split_k>16 || split_k>k/128 || int64_t(m)*k>INT32_MAX ||
        int64_t(n)*k>INT32_MAX || int64_t(m)*n*2>INT32_MAX ||
        !a || !b || !sfa || !sfb || !c || (split_k>1 && (!workspace || !counters)) ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(sfa)%16 || reinterpret_cast<uintptr_t>(sfb)%4 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(workspace)%16 ||
        reinterpret_cast<uintptr_t>(counters)%4) return 1;
    opus745_fused_lds_kargs args{};
    args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
    args.ptr_workspace=workspace;args.ptr_counters=counters;args.split_k=split_k;
    args.m=m;args.n=n;args.k=k;args.batch=1;
    args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
    switch (variant) {
''')
    for row in configs:
        i=row["id"]
        lines.append(f'    case {i}:return launch_one<FusedTraits{i}>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel{i}));')
    lines += ['    default:return 1;','    }','}','#endif']
    source = destination / "launch.hip"
    source.write_text("\n".join(lines)+"\n")
    reference = json.loads((HERE / "fused_register/build.json").read_text())
    argv = reference["argv"][:reference["argv"].index("-shared")]
    argv = [v for v in argv if not v.startswith("-I")]
    argv += ['-I'+str(ROOT/'csrc/include'),'-I'+str(ROOT/'csrc/opus_gemm/include'),
             '-I'+str(destination),'-shared',str(source),'-o',str(destination/'experiments.so')]
    started=time.monotonic()
    env={**os.environ,'HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':''}
    proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (destination/'build.log').write_text(proc.stdout)
    report={'returncode':proc.returncode,'argv':argv,'seconds':time.monotonic()-started,
            'parent_source':str(source_path),'parent_source_sha256':sha(source_path),
            'epilogue_parent_sha256':sha(register_source),'configurations':configs,
            'kernel_launches':0,'config_library_loads':0,'plan_sha256':sha(HERE/'plan.json'),
            'sources':{p.name:sha(p) for p in destination.iterdir() if p.suffix in ('.hip','.cuh')},
            'numerical_validation':'pending_idle_gpu','performance_validation':'pending_idle_gpu'}
    if proc.returncode==0:
        artifact=destination/'experiments.so';report['library_sha256']=sha(artifact)
        def module(name,path):
            spec=importlib.util.spec_from_file_location(name,path)
            result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
            return result
        audit=module('fused_lds745_audit',HERE/'audit_cache.py')
        parser=module('fused_lds745_elf',ROOT/'reports/opus_resume_20261008/sfa_packed/audit_device.py')
        image,targets=audit.device_image(parser,artifact.read_bytes())
        (destination/'device.co').write_bytes(image)
        report['targets']=targets;report['device_sha256']=hashlib.sha256(image).hexdigest()
        metadata=parser.Elf(image).metadata()['amdhsa.kernels']
        report['kernels']=[{'name':r['.name'],'resources':{key:r.get('.'+key,0) for key in
          ('vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count')}} for r in metadata]
    (destination/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    lock.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ('configurations','argv','sources')}))
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
