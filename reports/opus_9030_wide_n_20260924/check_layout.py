"""Run the exact layout helpers on CPU; do not initialize a GPU context."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LLVM='/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin'
source=ROOT/'csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh'
text=source.read_text()
start=text.index('template<class T>\n__device__ inline auto make_layout_ga_scale')
end=text.index('template<class T>\n__device__ inline constexpr auto make_layout_gsfa_scale',start)
helpers=text[start:end].replace('__device__','__host__')
(HERE/'host_layout_helpers.inc').write_text(helpers)
shadow=HERE/'host_include/opus'
shadow.mkdir(parents=True,exist_ok=True)
header=(ROOT/'csrc/include/opus/opus.hpp').read_text()
marker=header.index('// adaptor\n')
(shadow/'opus.hpp').write_text(header[:marker]+header[marker:].replace('OPUS_D ', 'OPUS_H_D '))
(shadow/'dtypes.hpp').write_bytes((ROOT/'csrc/include/opus/dtypes.hpp').read_bytes())
command=['/opt/rocm/bin/hipcc','-x','hip','--offload-host-only','--offload-arch=gfx950','-std=c++20','-O2',
         '-I'+str(HERE/'host_include'),
         '-I'+str(ROOT/'csrc/include'),'-I'+str(ROOT/'csrc/opus_gemm/include'),
         '-I'+str(ROOT/'csrc/opus_gemm/include/gfx950'),str(HERE/'layout_check.cpp'),'-o',str(HERE/'layout_check')]
subprocess.run(command,env=dict(os.environ,HIP_CLANG_PATH=LLVM),check=True)
result=subprocess.run([str(HERE/'layout_check')],text=True,capture_output=True)
record=dict(status='passed' if result.returncode==0 else 'failed',cpu_only=True,
            helper_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [source,HERE/'layout_check.cpp',Path(__file__),ROOT/'csrc/include/opus/opus.hpp',
                            ROOT/'csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh']},
            host_adaptation='Only host/device qualifiers changed in the copied layout/adaptor definitions; no GPU instructions executed',
            stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
(HERE/'layout_check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
