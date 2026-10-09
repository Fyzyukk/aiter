#!/usr/bin/env python3
"""Prepare an isolated exact9062 initial matrix/scale handoff experiment."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'fine_startup'
SOURCE = ROOT / 'csrc/opus_gemm/include'
FORMAL = ROOT / 'reports/opus_resume_20261008/formal_selected/identity_audit.json'
TRAIT = 'opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 2, 4, 128, 0, 16384>'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError('Refuse to overwrite an existing experiment')
    OUT.mkdir()
    for side in ['baseline', 'candidate']:
        shutil.copytree(SOURCE, OUT / side / 'include')
    rel = Path('include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh')
    baseline = (OUT / 'baseline' / rel).read_text()
    old = '''    if constexpr (!T::REGISTER_SCALES && T::EARLY_SCALE_LOADS) {
        load_scales();
        __builtin_amdgcn_sched_barrier(0);
    }
    static_for<T::NUM_STAGES>([&](auto i) {
        if (decltype(i)::value < initial_tiles) prefetch(i, decltype(i)::value);
    });
'''
    new = '''    // Isolated fixed9062 prologue: issue its existing initial matrix tiles
    // before the unequal SFA producer paths. All consumers and retirement stay
    // below the original waits/barriers; no scale helper or ring step changes.
    constexpr bool fine9062_initial_matrix_first = std::is_same_v<T,
        opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 2, 4, 128, 0, 16384>>;
    if constexpr (fine9062_initial_matrix_first) {
        static_for<T::NUM_STAGES>([&](auto i) {
            if (decltype(i)::value < initial_tiles) prefetch(i, decltype(i)::value);
        });
        __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (!T::REGISTER_SCALES && T::EARLY_SCALE_LOADS) {
        load_scales();
        __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (!fine9062_initial_matrix_first) {
        static_for<T::NUM_STAGES>([&](auto i) {
            if (decltype(i)::value < initial_tiles) prefetch(i, decltype(i)::value);
        });
    }
'''
    assert baseline.count(old) == 1
    candidate = baseline.replace(old, new)
    # The common LDS header is also parsed by nonfine TUs. A declaration only
    # makes the exact type predicate well formed without changing any helper.
    declaration = '''template<int BlockM, int WaveM, int WaveN, int Stages, int Cluster, int SplitK,
         int ReduceVec, int ReduceBlock, int StoreCache, int FixedK>
struct opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950;

'''
    candidate = candidate.replace('#if !defined(__HIP_DEVICE_COMPILE__)\n', declaration + '#if !defined(__HIP_DEVICE_COMPILE__)\n', 1)
    sync = r'(?:s_waitcnt_vmcnt|s_waitcnt_lgkmcnt|__builtin_amdgcn_s_barrier)\([^;]*?\);'
    assert re.findall(sync, baseline) == re.findall(sync, candidate)
    assert baseline.split('    auto step =', 1)[1] == candidate.split('    auto step =', 1)[1]
    assert baseline.split('    auto load_scales =', 1)[1].split(old, 1)[0] == candidate.split('    auto load_scales =', 1)[1].split('    // Isolated fixed9062', 1)[0]
    (OUT / 'candidate' / rel).write_text(candidate)
    (OUT / 'candidate.diff').write_text(''.join(difflib.unified_diff(
        baseline.splitlines(True), candidate.splitlines(True),
        fromfile='baseline/' + str(rel), tofile='candidate/' + str(rel))))
    source_rows = []
    for path in sorted((OUT / 'baseline/include').rglob('*')):
        if path.is_file():
            relative = path.relative_to(OUT / 'baseline')
            other = OUT / 'candidate' / relative
            source_rows.append({'path': str(relative), 'baseline_sha256': sha(path.read_bytes()),
                                'candidate_sha256': sha(other.read_bytes())})
    changed = [r for r in source_rows if r['baseline_sha256'] != r['candidate_sha256']]
    assert [r['path'] for r in changed] == [str(rel)]
    formal = json.loads(FORMAL.read_text())
    selected = [p for p in formal['parents'] if p['parent_id'] >= 9040]
    device_tus = []
    for p in selected:
        object_path = Path(p['candidate_object'])
        staging = object_path.parent.parent / 'blob.staging'
        source = staging / 'instances' / object_path.name.replace('.cuda.o', '.cu')
        device_tus.append({'parent_id': p['parent_id'], 'official_object': str(object_path),
                           'source': str(source), 'source_sha256': sha(source.read_bytes()),
                           'staging': str(staging), 'variants': [v['candidate'] for v in p['variants']]})
    manifest = {'status': 'isolated_prepared_not_built', 'production_modified': False,
                'mechanism': 'fixed9062 initial matrix tiles before unchanged scale producer paths',
                'selected_trait': TRAIT, 'initial_tiles': 3, 'active_ring_slots': 4,
                'formal_identity': {'path': str(FORMAL), 'sha256': sha(FORMAL.read_bytes())},
                'device_tus': device_tus, 'source_files': source_rows, 'changes': changed,
                'runtime_wait_barrier_sequence_unchanged': True, 'all_step_and_output_source_unchanged': True,
                'scale_helper_and_scale_body_unchanged': True,
                'scope': 'Exact fixed9062 type only; original prefetch calls and all byte/layout/tail guards preserved. Compiler generated scale waits may now include earlier matrix requests; ISA and Event must decide.',
                'script_sha256': sha(Path(__file__).read_bytes())}
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'out': str(OUT), 'selected_TUs': len(device_tus)}))


if __name__ == '__main__':
    main()
