"""Build isolated wide-N candidates with the accepted compiler options."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path('/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin')
saved = json.loads((ROOT/'reports/opus_9010_9011_opt_20260924/agent_9010/final/build_command.json').read_text())
command = [*saved[:saved.index('-shared')], '-shared',
           '-I'+str(ROOT/'csrc/opus_gemm/include/gfx950'),
           '-I'+str(ROOT/'csrc/opus_gemm/include'), '-I'+str(ROOT/'csrc/include'),
           '-mllvm', '-verify-machineinstrs', str(HERE/'launch.hip'), '-o', str(HERE/'experiments.so')]
sources = [HERE/'launch.hip', Path(__file__), *sorted((ROOT/'csrc/opus_gemm/include/gfx950').glob('*.cuh')),
           ROOT/'csrc/include/opus/opus.hpp', ROOT/'aiter/jit/optCompilerConfig.json']
hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(HERE/'build_command.json').write_text(json.dumps(command, indent=2)+'\n')
start = time.time()
with (HERE/'build.log').open('w') as output:
    result = subprocess.call(command, cwd=ROOT, env=dict(os.environ, HIP_CLANG_PATH=str(LLVM)),
                             stdout=output, stderr=subprocess.STDOUT)
manifest = dict(status='passed' if result == 0 else 'failed', returncode=result, cpu_only=True,
                elapsed_seconds=time.time()-start, command=command, source_sha256=hashes)
if result == 0:
    manifest['binary_sha256'] = hashlib.sha256((HERE/'experiments.so').read_bytes()).hexdigest()
(HERE/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print((HERE/'build.log').read_text())
print(json.dumps({k:manifest[k] for k in ['status','returncode','elapsed_seconds']}), flush=True)
raise SystemExit(result)
