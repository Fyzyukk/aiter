"""Compile and run CPU-only validation without HIP runtime calls or GPU discovery.

The matrix/output checks execute the established production layout helpers. The
scale producer/consumer and the fixed-K schedule are extracted from the current
short-K pipeline and executed with bounded CPU memory and event instrumentation.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LLVM = Path('/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin')
GFX = ROOT / 'csrc/opus_gemm/include/gfx950'
LAYOUT_SOURCE = GFX / 'opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh'
PIPELINE_SOURCE = GFX / 'opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh'
TRAITS_SOURCE = GFX / 'opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh'


def extract(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


layout_text = LAYOUT_SOURCE.read_text()
helpers = extract(layout_text,
                  'template<class T>\n__device__ inline auto make_layout_ga_scale',
                  'template<class T>\n__device__ inline constexpr auto make_layout_gsfa_scale')
(HERE / 'host_layout_helpers.inc').write_text(helpers.replace('__device__', '__host__'))

pipeline = PIPELINE_SOURCE.read_text()
scale_load = extract(pipeline, '    auto load_scale_panel = ', '    auto read_scales = ')
scale_read = extract(pipeline, '    auto read_scales = ', '    auto load_a = ')
for name, body in [('host_scale_load.inc', scale_load), ('host_scale_read.inc', scale_read)]:
    # Memory wrappers change, while the production indexing/branching stays exact.
    body = body.replace('thread_id_x()', 'cpu_thread')
    body = re.sub(r'\b(load|store)<', r'cpu_check::\1<', body)
    (HERE / name).write_text(body)
schedule = extract(pipeline, '    // Seed K0/K1 once.', '    const auto p_coord = ')
(HERE / 'host_schedule.inc').write_text(schedule.replace('__builtin_amdgcn_s_barrier()', 'schedule_barrier()'))

shadow = HERE / 'host_include/opus'
shadow.mkdir(parents=True, exist_ok=True)
header_source = ROOT / 'csrc/include/opus/opus.hpp'
header = header_source.read_text()
marker = header.index('// adaptor\n')
(shadow / 'opus.hpp').write_text(header[:marker] + header[marker:].replace('OPUS_D ', 'OPUS_H_D '))
(shadow / 'dtypes.hpp').write_bytes((ROOT / 'csrc/include/opus/dtypes.hpp').read_bytes())

command = [str(LLVM / 'clang++'), '-x', 'hip', '--offload-host-only', '--offload-arch=gfx950',
           '--hip-path=/opt/rocm', '-nogpulib', '-std=c++20', '-O2',
           '-DOPUS_ENABLE_RUNTIME_QUERY=0', '-I' + str(HERE / 'host_include'),
           '-I' + str(ROOT / 'csrc/include'), '-I' + str(ROOT / 'csrc/opus_gemm/include'),
           '-I' + str(GFX), str(HERE / 'layout_check.cpp'), '-o', str(HERE / 'layout_check')]
# Direct clang host-only compilation avoids hipcc's runtime library linkage.
compile_result = subprocess.run(command, text=True, capture_output=True)
(HERE / 'build.log').write_text(compile_result.stdout + compile_result.stderr)
if compile_result.returncode:
    print(compile_result.stdout + compile_result.stderr)
    raise SystemExit(compile_result.returncode)

dynamic = subprocess.run(['readelf', '-d', str(HERE / 'layout_check')], text=True, capture_output=True, check=True)
undefined = subprocess.run(['nm', '-u', str(HERE / 'layout_check')], text=True, capture_output=True, check=True)
(HERE / 'elf_dynamic.txt').write_text(dynamic.stdout)
(HERE / 'undefined_symbols.txt').write_text(undefined.stdout)
assert not re.search(r'lib(?:amdhip|hip|hsa)', dynamic.stdout, re.IGNORECASE), dynamic.stdout
assert not re.search(r'\b(?:hip|hsa|__hip)[A-Za-z_0-9]*', undefined.stdout), undefined.stdout

result = subprocess.run([str(HERE / 'layout_check')], text=True, capture_output=True)
sources = [LAYOUT_SOURCE, PIPELINE_SOURCE, TRAITS_SOURCE, header_source,
           ROOT / 'csrc/include/opus/dtypes.hpp', HERE / 'layout_check.cpp',
           HERE / 'scale_schedule_check.inc', Path(__file__)]
generated = [HERE / name for name in ['host_layout_helpers.inc', 'host_scale_load.inc',
                                     'host_scale_read.inc', 'host_schedule.inc']]
record = {
    'status': 'passed' if result.returncode == 0 else 'failed',
    'cpu_only': True,
    'fixed_k': [384, 768, 1024],
    'compile_command': command,
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'generated_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in generated},
    'host_adaptation': {
        'layouts': 'Only host/device qualifiers changed in copies of layout helpers and adaptor definitions.',
        'scales': 'Exact production lambdas; thread_id_x becomes an enumerated CPU thread; load/store become bounds-checked CPU memory operations.',
        'schedule': 'Exact production prologue, static_for dispatch, and final compute; operations instrumented and barrier intrinsic redirected.',
    },
    'runtime_library_linkage': 'No HIP/HSA dynamic library dependencies or undefined HIP/HSA symbols.',
    'limits': [
        'No GPU discovery, device calls, GPU execution, timing, or runtime numerical-correctness testing.',
        'Schedule instrumentation checks logical operand lifetimes, exact tile counts, and barriers; it does not simulate hardware latency or prove ISA hazard behavior.',
        'Buffer-bound tests validate coordinate contracts; hardware OOB zero-fill and store discard semantics are assumed.',
    ],
    'stdout': result.stdout,
    'stderr': result.stderr,
    'returncode': result.returncode,
}
(HERE / 'layout_check.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
raise SystemExit(result.returncode)
