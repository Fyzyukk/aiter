#!/usr/bin/env python3
"""Build a private last-arrival split-K experiment without editing production."""
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
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config, config_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FUSED_OUTPUT = r'''
        if constexpr (Runtime) {
            if (args.split_k == 1) {
                opus_gemm_small_output::store_mfma16_packed<T>(c, gc, lane, wm, wn, args.stride_c);
                return;
            }
            static_assert(T::WAVE_K == 1);
            constexpr int tile_elements = T::B_M * T::B_N;
            auto slots = make_gmem(args.ptr_workspace + static_cast<int64_t>(tile) *
                                    args.split_k * tile_elements,
                                    static_cast<unsigned>(args.split_k * tile_elements * sizeof(float)));
            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int fragment = decltype(fi)::value;
                const int offset = split * tile_elements +
                                   (fragment * T::BLOCK_SIZE + thread_id_x()) * elem_c;
                store<elem_c>(slots, c[fragment], offset);
            });
            // Every producer wave releases its own partial stores before the
            // workgroup publishes one arrival. Agent scope remains valid even
            // when the runtime schedules a tile's splits on different XCDs.
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __builtin_amdgcn_s_barrier();
            __shared__ int last_arrival;
            if (thread_id_x() == 0) {
                const int old = __atomic_fetch_add(args.ptr_counters + tile, 1, __ATOMIC_ACQ_REL);
                last_arrival = old == args.split_k - 1;
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            if (last_arrival) {
                array<typename Base::vtype_c, T::E_M * T::E_N> sum{};
                // sc0/sc1 bypass private caches for other producers' partials.
                #pragma clang loop unroll(disable)
                for (int part = 0; part < args.split_k; ++part) {
                    static_for<T::E_M * T::E_N>([&](auto fi) {
                        constexpr int fragment = decltype(fi)::value;
                        const int offset = part * tile_elements +
                                           (fragment * T::BLOCK_SIZE + thread_id_x()) * elem_c;
                        sum[fragment] += load<elem_c>(slots, offset, 0, number<17>{});
                    });
                }
                opus_gemm_small_output::store_mfma16_packed<T>(sum, gc, lane, wm, wn, args.stride_c);
                // Reset only after every output store has completed. The next
                // launch on the same stream starts from zero counters again.
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
                __builtin_amdgcn_s_barrier();
                if (thread_id_x() == 0)
                    __atomic_store_n(args.ptr_counters + tile, 0, __ATOMIC_RELEASE);
            }
        } else if constexpr (T::OUTPUT == 3) {
'''


def main():
    destination = HERE / "fused_register"
    destination.mkdir(exist_ok=False)
    source_path = ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_gfx950.cuh"
    original = source_path.read_text()
    body = original[:original.index('\n#if !defined(__HIP_DEVICE_COMPILE__)\ntemplate<class T>')]
    body = body.replace('"opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"',
                        '"gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"')
    body = body.replace('"opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh"', '"fused_kargs.cuh"')
    body = body.replace('"opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh"',
                        '"gfx950/opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh"')
    body = body.replace('"opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh"',
                        '"gfx950/opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh"')
    body = body.replace("opus_gemm_bpreshuffle_register_detail", "opus745_fused_register_detail")
    body = body.replace("std::is_same_v<Kargs, opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950>",
                        "std::is_same_v<Kargs, opus745_fused_kargs>")
    old = "    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;"
    assert body.count(old) == 1
    body = body.replace(old, '''    const int n_tiles = args.n / T::B_N;
    const int m_tiles = (args.m + T::B_M - 1) / T::B_M;
    const int bid = block_id_x();
    const int tile = bid / (8 * args.split_k) * 8 + bid % 8;
    if (tile >= m_tiles * n_tiles) return;
    const int row = tile / n_tiles * T::B_M, col = tile % n_tiles * T::B_N;''')
    assert body.count("split = block_id_z();") == 1
    body = body.replace("split = block_id_z();", "split = bid / 8 % args.split_k;")
    start = body.index("        if constexpr (Runtime) {\n            if (args.split_k == 1)")
    end = body.index("        } else if constexpr (T::OUTPUT == 3) {", start)
    end += len("        } else if constexpr (T::OUTPUT == 3) {\n")
    body = body[:start] + FUSED_OUTPUT + body[end:]
    (destination / "pipeline.cuh").write_text(body)
    (destination / "fused_kargs.cuh").write_text('''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gfx950/opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh"
struct opus745_fused_kargs : opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950 {
    float* ptr_workspace;
    int* ptr_counters;
};
''')
    configs = []
    for tm, tn, wm, wn in ((16,16,1,1), (16,32,1,1), (32,32,1,1),
                           (32,64,1,1), (64,64,2,2)):
        for depth in (3,6):
            c = construct_config("register", tile_m=tm, tile_n=tn, wave_m=wm, wave_n=wn,
                                 wave_k=1, prefetch=depth, runtime_split_k=True,
                                 b_cache=2, dispatch=(), specializations=())
            configs.append({"id":len(configs), "identity":config_identity(c),
                            "pipeline":c.pipeline, "compile_params":json.loads(c.to_json())})
    lines = ['// SPDX-License-Identifier: Apache-2.0', '#include <cstdint>',
             '#include <opus/hip_minimal.hpp>', '#include "fused_kargs.cuh"', '#include "pipeline.cuh"']
    for row in configs:
        c = row['compile_params']
        i = row['id']
        lines.append(f'''using FusedTraits{i} = opus_gemm_mxscale_bpreshuffle_pipeline_traits<
    opus_gemm_mxscale_bpreshuffle_runtime_config_traits<
        opus_gemm_small_register_traits_gfx950<{c['tile_m']},{c['tile_n']},{c['wave_m']},{c['wave_n']},{c['prefetch']},1,3,2,0,false,{str(c['reuse_b_scale']).lower()}>,16,128>,1>;
__global__ __launch_bounds__(FusedTraits{i}::BLOCK_SIZE,1)
void fused_kernel{i}(opus745_fused_kargs args) {{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_register_detail::run<FusedTraits{i}>(args);
#endif
}}''')
    lines.append('''#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> int launch_one(opus745_fused_kargs args, void* stream, const void* kernel) {
    const int tiles = ((args.m + T::B_M - 1) / T::B_M) * (args.n / T::B_N);
    const dim3 grid(((tiles + 7) / 8) * 8 * args.split_k);
    void* argv[] = {&args};
    return hipLaunchKernel(kernel, grid, dim3(T::BLOCK_SIZE), argv, 0, stream);
}
extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, float* workspace, int* counters, int m, int n, int k,
           int split_k, void* stream) {
    if (m <= 0 || m > 512 || n <= 0 || k < 128 || k > 16384 || n % 128 || k % 128 ||
        split_k < 1 || split_k > 16 || split_k > k / 128 ||
        int64_t(m)*k > INT32_MAX || int64_t(n)*k > INT32_MAX ||
        int64_t(m)*n*2 > INT32_MAX || !a || !b || !sfa || !sfb || !c ||
        (split_k > 1 && (!workspace || !counters)) ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(workspace)%16 ||
        reinterpret_cast<uintptr_t>(counters)%4)
        return 1;
    opus745_fused_kargs args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_c=c; args.ptr_sfa=sfa; args.ptr_sfb=sfb;
    args.ptr_workspace=workspace; args.ptr_counters=counters; args.split_k=split_k;
    args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    switch (variant) {
''')
    for row in configs:
        i=row['id']
        lines.append(f'    case {i}: return launch_one<FusedTraits{i}>(args,stream,reinterpret_cast<const void*>(fused_kernel{i}));')
    lines.extend(['    default: return 1;', '    }', '}', '#endif'])
    source = destination / "launch.hip"
    source.write_text('\n'.join(lines)+'\n')
    reference = json.loads((HERE / "extra_device_build/receipt.json").read_text())
    argv = next(r['argv'] for r in reference['rows'] if r['pipeline']=='register')
    argv = [v for v in argv[:argv.index('-c')] if not v.startswith('-I')]
    argv += ['-I'+str(ROOT/'csrc/include'), '-I'+str(ROOT/'csrc/opus_gemm/include'),
             '-I'+str(destination), '-shared', str(source), '-o', str(destination/'experiments.so')]
    started=time.monotonic()
    env={**os.environ,'HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':''}
    proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (destination/'build.log').write_text(proc.stdout)
    report={'returncode':proc.returncode,'argv':argv,'seconds':time.monotonic()-started,
            'parent_source':str(source_path),'parent_source_sha256':sha(source_path),
            'configurations':configs,'kernel_launches':0,'config_library_loads':0,
            'sources':{p.name:sha(p) for p in destination.iterdir() if p.suffix in ('.hip','.cuh')},
            'numerical_validation':'pending_idle_gpu','performance_validation':'pending_idle_gpu'}
    if proc.returncode==0:
        artifact=destination/'experiments.so'
        report['library_sha256']=sha(artifact)
        parser_path=HERE/'audit_cache.py'
        spec=importlib.util.spec_from_file_location('fused745_audit',parser_path)
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        parser_path=ROOT/'reports/opus_resume_20261008/sfa_packed/audit_device.py'
        spec=importlib.util.spec_from_file_location('fused745_elf',parser_path)
        parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
        image,targets=audit.device_image(parser,artifact.read_bytes())
        report['targets']=targets;report['device_sha256']=hashlib.sha256(image).hexdigest()
        metadata=parser.Elf(image).metadata()['amdhsa.kernels']
        report['kernels']=[{'name':r['.name'],'resources':{key:r.get('.'+key,0) for key in
          ('vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count')}} for r in metadata]
    (destination/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('configurations','argv','sources')}))
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
