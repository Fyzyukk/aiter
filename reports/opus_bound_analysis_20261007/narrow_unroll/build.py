#!/usr/bin/env python3
"""Build a private runtime-only narrow loop-unroll experiment; CPU only."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
HEADERS = ROOT / "csrc/opus_gemm/include"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = [
    "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh",
    "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh",
    "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh",
    "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh",
    "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh",
    "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh",
    "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh",
    "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh",
    "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh",
    "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh",
    "opus_gemm_utils.cuh",
]
initial = {f: sha(HEADERS / f) for f in files}
launcher = r'''#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"

extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if ((variant != 9023 && variant != 9024) || m <= 0 || n <= 0 || k <= 0 ||
        k > 16384 || k % 128 || n % 128 ||
        int64_t(m)*k > limit || int64_t(n)*k > limit || int64_t(m)*n*2 > limit ||
        !a || !b || !sfa || !sfb || !c ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(sfa)%16)
        return static_cast<int>(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_sfa=sfa; args.ptr_sfb=sfb; args.ptr_c=c;
    args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    args.stride_a_batch=m*k; args.stride_b_batch=n*k; args.stride_c_batch=m*n;
    args.stride_sfa_batch=m*(k/128); args.stride_sfb_batch=(n/128)*(k/128);
    const auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    if (variant==9023) {
        if (m>=1024 && n<=1024 && k==7168 && int64_t((m+63)/64)*(n/128)<=256) {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<4,64,7168>;
            gemm_a8w8_mxfp8_scale_4wave_64x128_kernel<T><<<dim3(n/128,(m+63)/64),256,0,hip_stream>>>(args);
        } else {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950;
            gemm_a8w8_mxfp8_scale_4wave_64x128_kernel<T><<<dim3(n/128,(m+63)/64),256,0,hip_stream>>>(args);
        }
    } else {
        if (m>=1024 && m<=2048 && n<=1024 && k==7168) {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,4>;
            gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/64,(m+63)/64),256,0,hip_stream>>>(args);
        } else {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950;
            gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/64,(m+63)/64),256,0,hip_stream>>>(args);
        }
    }
    return static_cast<int>(hipGetLastError());
}
'''
for mode in ("baseline", "candidate"):
    directory = OUT / mode
    directory.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = directory / "include" / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HEADERS / f, target)
    (directory / "launch.hip").write_text(launcher)

changes = {
    files[0]: ("#pragma clang loop unroll_count(T::NUM_STAGES)",
               "#pragma clang loop unroll_count(T::FIXED_K ? T::NUM_STAGES : 1)"),
    files[1]: ("#pragma unroll 4", "#pragma clang loop unroll_count(T::FIXED_K ? 4 : 1)"),
}
patch = []
for f, (before, after) in changes.items():
    target = OUT / "candidate/include" / f
    text = target.read_text()
    assert text.count(before) == 1
    target.write_text(text.replace(before, after))
    patch.extend(difflib.unified_diff((OUT / "baseline/include" / f).read_text().splitlines(True),
                                    target.read_text().splitlines(True),
                                    fromfile="baseline/include/"+f, tofile="candidate/include/"+f))
(OUT / "changes.diff").write_text("".join(patch))

template = json.loads((OUT.parent / "compute_prologue/build_manifest.json").read_text())["builds"][0]["command"]
manifest = {"cpu_only": True, "gpu_executed": False, "production_modified": False,
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str((HEADERS/f).relative_to(ROOT)): digest for f, digest in initial.items()},
            "change": "Only runtime narrow main loop unroll count becomes 1; fixed-K pragma expression keeps original count",
            "builds": []}
for mode in ("baseline", "candidate"):
    directory = OUT / mode
    cmd = []
    for arg in template:
        if "/compute_prologue/baseline" in arg:
            arg = arg.replace("/compute_prologue/baseline", "/narrow_unroll/"+mode)
        cmd.append(arg)
    cmd.insert(cmd.index("-shared"), "-resource-dir=/opt/rocm/lib/llvm/lib/clang/20")
    start = time.time()
    with (directory / "build.log").open("w") as log:
        run = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    manifest["builds"].append({"mode": mode, "command": cmd, "exit_code": run.returncode,
                               "seconds": time.time()-start, "binary_sha256": sha(directory/"experiments.so") if run.returncode==0 else None})
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if run.returncode:
        raise SystemExit(f"{mode} build failed; see {directory/'build.log'}")
assert all(sha(HEADERS/f)==digest for f,digest in initial.items())
manifest["status"] = "passed"
(OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps({"status": manifest["status"], "builds": [{"mode": b["mode"], "seconds": b["seconds"]} for b in manifest["builds"]]}))
