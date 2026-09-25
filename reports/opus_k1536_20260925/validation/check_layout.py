"""Compile and run CPU-only validation without HIP runtime calls or GPU discovery.

The matrix/output checks execute the established production layout helpers and
the current K=1536 matrix lambdas. Scale producers/consumers, MFMA operand selection,
and both fixed-K schedules are extracted and executed with CPU instrumentation.
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
PIPELINE_SOURCE = GFX / 'opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_k1536_gfx950.cuh'
TRAITS_SOURCE = GFX / 'opus_gemm_traits_a8w8_mxscale_bpreshuffle_k1536_gfx950.cuh'
header_source = ROOT / 'csrc/include/opus/opus.hpp'
sources = [LAYOUT_SOURCE, PIPELINE_SOURCE, TRAITS_SOURCE, header_source,
           GFX / 'opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh',
           ROOT / 'csrc/include/opus/dtypes.hpp', HERE / 'layout_check.cpp',
           HERE / 'scale_schedule_check.inc', HERE / 'cpu_memory.inc', Path(__file__)]
source_snapshots = {p: p.read_bytes() for p in sources}
# A failed rerun must not leave a stale passing record behind.
(HERE / 'layout_check.json').unlink(missing_ok=True)


def extract(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


layout_text = source_snapshots[LAYOUT_SOURCE].decode()
helpers = extract(layout_text,
                  'template<class T>\n__device__ inline auto make_layout_ga_scale',
                  'template<class T>\n__device__ inline constexpr auto make_layout_gsfa_scale')
(HERE / 'host_layout_helpers.inc').write_text(helpers.replace('__device__', '__host__'))

pipeline = source_snapshots[PIPELINE_SOURCE].decode()
extracts = {
    'host_matrix_layouts.inc': extract(pipeline, '    const auto u_ga = ', '    auto mma = '),
    'host_matrix_prefetch.inc': extract(pipeline, '    auto issue_matrix_prefetch = ', '    auto load_scale_panel = '),
    'host_scale_load.inc': extract(pipeline, '    auto load_scale_panel = ', '    auto read_scales = '),
    'host_scale_read.inc': extract(pipeline, '    auto read_scales = ', '    auto load_a = '),
    'host_matrix_read.inc': extract(pipeline, '    auto load_a = ', '    auto compute = '),
    'host_compute.inc': extract(pipeline, '    auto compute = ', '    // Seed K0/K1 once.'),
    'host_output_layout.inc': extract(pipeline, '    const auto p_coord = ', '    static_for<T::E_M * T::E_N>'),
}
for name, body in extracts.items():
    # Memory wrappers change, while the production indexing/branching stays exact.
    body = body.replace('thread_id_x()', 'cpu_thread')
    body = re.sub(r'\b(async_load|load|store)<', r'cpu_check::\1<', body)
    (HERE / name).write_text(body)
schedule = extract(pipeline, '    // Seed K0/K1 once.', '    const auto p_coord = ')
(HERE / 'host_schedule.inc').write_text(schedule.replace('__builtin_amdgcn_s_barrier()', 'schedule_barrier()'))

shadow = HERE / 'host_include/opus'
shadow.mkdir(parents=True, exist_ok=True)
header = source_snapshots[header_source].decode()
marker = header.index('// adaptor\n')
(shadow / 'opus.hpp').write_text(header[:marker] + header[marker:].replace('OPUS_D ', 'OPUS_H_D '))
(shadow / 'dtypes.hpp').write_bytes(source_snapshots[ROOT / 'csrc/include/opus/dtypes.hpp'])

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
assert all(p.read_bytes() == snapshot for p, snapshot in source_snapshots.items()), \
    'A source changed during validation; rerun to validate one consistent snapshot.'
generated = [HERE / name for name in ['host_layout_helpers.inc', 'host_schedule.inc', *extracts,
                                     'host_include/opus/opus.hpp', 'host_include/opus/dtypes.hpp']]
record = {
    'status': 'passed' if result.returncode == 0 else 'failed',
    'cpu_only': True,
    'fixed_k': 1536,
    'loop_unroll': [12, 2],
    'n_values': [256, 512, 6144, 7168, 8192, 16384],
    'valid_m_rows_per_tile': [64, 128, 192],
    'm_tile_origins': [0, 192],
    'n_tile_origins': ['first', 'last'],
    'compile_command': command,
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(data).hexdigest() for p, data in source_snapshots.items()},
    'generated_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in generated},
    'host_adaptation': {
        'layouts': 'Only host/device qualifiers changed in copies of layout helpers and adaptor definitions; production matrix-layout declarations and output partition declarations execute unchanged.',
        'matrices': 'Exact production prefetch and LDS-reader lambdas; async_load/load use bounded CPU coordinate storage. Async writes model Wave64 lane*16 LDS placement and check both slots.',
        'scales': 'Exact production lambdas; thread_id_x becomes an enumerated CPU thread; load/store become bounds-checked CPU memory operations.',
        'schedule': 'Exact production prologue, advance lambda, both LoopUnroll branches, and final compute; operations instrumented and barrier intrinsic redirected. Full event traces must match between specializations.',
        'compute': 'Exact production compute lambda; BaseMMA is replaced with a CPU oracle checking A/B group and indices, packed scale selection, accumulator indexing, and logical LDS readiness.',
    },
    'runtime_library_linkage': 'No HIP/HSA dynamic library dependencies or undefined HIP/HSA symbols.',
    'limits': [
        'No GPU discovery, device calls, GPU execution, timing, or runtime numerical-correctness testing.',
        'Schedule instrumentation checks logical operand lifetimes, exact tile counts, and barriers; it does not simulate hardware latency or prove ISA hazard behavior.',
        'Buffer-bound tests validate coordinate contracts; hardware OOB zero-fill, store discard, and async Wave64 lane placement semantics are assumed.',
    ],
    'stdout': result.stdout,
    'stderr': result.stderr,
    'returncode': result.returncode,
}
(HERE / 'layout_check.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
raise SystemExit(result.returncode)
