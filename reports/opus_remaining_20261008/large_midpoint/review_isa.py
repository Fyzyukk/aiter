#!/usr/bin/env python3
"""CPU audit of built isolated 9030 midpoint synchronization and source scope."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEADER = 'include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    audit = json.loads((HERE / 'device_audit.json').read_text())
    build = json.loads((HERE / 'build_manifest.json').read_text())
    assert audit['status'] == build['status'] == 'passed'
    assert build['source_manifest_sha256'] == sha(HERE / 'source_manifest.json')
    assert audit['checks'][0]['baseline_matches_oct8_FUNC_full_metadata_normalized_descriptor']
    assert sha(audit['official_module']['path']) == audit['official_module']['sha256']
    changed = []
    for f in manifest['source_files']:
        for side in ['baseline', 'candidate']:
            assert sha(HERE / side / f['path']) == f[side + '_sha256']
        if f['baseline_sha256'] != f['candidate_sha256']:
            changed.append(f['path'])
    assert changed == [HEADER]
    source = (HERE / 'baseline' / HEADER).read_text()
    candidate = (HERE / 'candidate' / HEADER).read_text()
    assert source[:source.index('    auto advance_tile = ')] == candidate[:candidate.index('    auto advance_tile = ')]
    assert source[source.index('    // Prologue'):] == candidate[candidate.index('    // Prologue'):]
    assert sha(HERE / 'baseline/launch.hip') == sha(HERE / 'candidate/launch.hip') == manifest['launcher_sha256']
    assert 'k%128||m%64||n%256||' in (HERE / 'candidate/launch.hip').read_text()
    for row in build['builds']:
        assert sha(HERE / row['side'] / 'experiments.so') == row['library_sha256']
    helper = ROOT / 'reports/opus_bound_analysis_20261007/audit_current_compute_metadata.py'
    spec = importlib.util.spec_from_file_location('large_review_elf', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    name = audit['checks'][0]['name']
    for side in ['baseline', 'candidate']:
        co = HERE / side / 'device.co'
        assert sha(co) == audit['images'][side]['sha256']
        assert module.device_bundle(HERE / side / 'experiments.so') == co.read_bytes()
        function = module.Elf(co.read_bytes()).symbols()[name]
        assert hashlib.sha256(function['bytes']).hexdigest() == audit['checks'][0][side]['instruction_sha256']
    isa = {}
    for line in (HERE / 'candidate/device.s').read_text().splitlines():
        match = re.match(r'^\s*(.*?)\s*// ([0-9A-Fa-f]+):', line)
        if match:
            op = re.sub(r'\s+', ' ', match[1].strip())
            pc = int(match[2], 16)
            if 0x1b00 <= pc < 0x1b00 + audit['checks'][0]['candidate']['instruction_bytes']:
                isa[pc] = op
    assert len(isa) == 1279
    bar_pcs = [pc for pc, op in isa.items() if op == 's_barrier']
    assert bar_pcs == [0x24f4, 0x2634, 0x2918, 0x2b7c, 0x2c5c, 0x2ec8, 0x34e8]
    assert isa[0x24f0] == 's_waitcnt vmcnt(0) lgkmcnt(0)'
    assert isa[0x2624] == 's_waitcnt lgkmcnt(0)'
    assert isa[0x1d10] == 's_ashr_i32 s33, s0, 7'
    assert isa[0x26a4] == 's_mov_b32 s42, 3'
    assert isa[0x2850] == 's_add_i32 s42, s42, 2'
    assert isa[0x291c] == 's_add_i32 s45, s42, -1'
    records = []
    for first, mid, first_issue, last_issue, terminal_wait, terminal_bar, last in [
            (0x2870, 0x2918, 0x2944, 0x29b0, 0x2b74, 0x2b7c, 0x2bac),
            (0x2bb4, 0x2c5c, 0x2c78, 0x2cdc, 0x2ec0, 0x2ec8, 0x2ecc)]:
        region = [(pc, op) for pc, op in isa.items() if first <= pc <= last]
        before = [(pc, op) for pc, op in region if pc < mid]
        assert sum(op.startswith('v_mfma_scale') for pc, op in before) == 8
        assert not any(op.startswith('ds_read_b128') or (op.startswith('buffer_load') and ' lds' in op) for pc, op in before)
        assert isa[mid - 4] == 's_waitcnt vmcnt(0) lgkmcnt(0)'
        matrix = [(pc, op) for pc, op in region if op.startswith('buffer_load') and ' lds' in op]
        assert len(matrix) == 7 and matrix[0][0] == first_issue and matrix[-1][0] == last_issue
        assert all(mid < pc for pc, _ in matrix)
        reads = [(pc, op) for pc, op in region if op.startswith('ds_read_b128')]
        assert len(reads) == 22 and all(mid < pc < terminal_wait for pc, _ in reads)
        assert sum(op.startswith('v_mfma_scale') for pc, op in region) == 24
        assert isa[terminal_wait] == 's_waitcnt lgkmcnt(0)'
        assert isa[terminal_wait + 4].startswith('s_cbranch_scc1')
        assert isa[terminal_bar] == 's_barrier'
        records.append({'first_MFMA_pc': hex(first), 'midpoint_pc': hex(mid),
                        'current_MFMA_count_before_midpoint': 8,
                        'midpoint_wait0_lgkm0_before_barrier': True,
                        'next_matrix_reads_before_midpoint': 0,
                        'Kplus2_requests_after_midpoint': [hex(pc) for pc, _ in matrix],
                        'next_matrix_read_count': 22,
                        'last_next_matrix_read_pc': hex(reads[-1][0]),
                        'end_read_wait_pc': hex(terminal_wait), 'guarded_terminal_barrier_pc': hex(terminal_bar)})
    assert isa[0x2928] == 's_cmp_ge_i32 s45, s33'
    assert isa[0x2b70] == 's_cmp_lg_u32 s12, 1'
    assert isa[0x2c60] == 's_cmp_ge_i32 s42, s33'
    assert isa[0x2ebc] == 's_cmp_lg_u32 s10, 1'
    first_c = min(pc for pc, op in isa.items() if op.startswith('v_cvt_pk_bf16'))
    c_write = min(pc for pc, op in isa.items() if pc > first_c and op.startswith('ds_write'))
    assert first_c == 0x3104 and c_write > first_c > 0x2ec8
    counts = dict(collections.Counter(op.split()[0] for op in isa.values()))
    result = {
        'status': 'ready_for_root_controlled_numerical_guard_and_Event_screen',
        'cpu_only': True, 'new_GPU_execution': False, 'new_build_execution': False,
        'production_changed_by_this_agent': False, 'changed_headers': changed,
        'official_module': audit['official_module'],
        'source_manifest_sha256': sha(HERE / 'source_manifest.json'),
        'candidate_diff_sha256': sha(HERE / 'candidate.diff'),
        'build_manifest_sha256': sha(HERE / 'build_manifest.json'),
        'device_audit_sha256': sha(HERE / 'device_audit.json'),
        'review_script_sha256': sha(__file__),
        'candidate_source_sha256': sha(HERE / 'candidate' / HEADER),
        'candidate_library_sha256': sha(HERE / 'candidate/experiments.so'),
        'candidate_code_object_sha256': sha(HERE / 'candidate/device.co'),
        'candidate_FUNC_sha256': audit['checks'][0]['candidate']['instruction_sha256'],
        'baseline_exact_FUNC_full_metadata_normalized_descriptor': True,
        'prologue_loop_epilogue_output_source_exact': True,
        'all_shared_helpers_and_traits_source_exact': True,
        'ISA_barrier_pcs': [hex(p) for p in bar_pcs], 'advance_bodies': records,
        'terminal_guard_derivation': 's33=K/128; s42=tile_k+3 before stage0. Stage0 terminal tests s45=s42-1 >= loops; stage1 tests s42 >= loops. Each scalar condition is workgroup-uniform.',
        'first_C_conversion_pc': hex(first_c), 'first_C_LDS_write_pc': hex(c_write),
        'instruction_counts': counts, 'resources': {s: audit['checks'][0][s]['metadata'] for s in ['baseline', 'candidate']},
        'proof': [
            'Current operands are all in VGPRs before first advance: original prologue operand reads then lgkm0 and workgroup barrier are retained.',
            'For later advances, all current matrix reads completed at previous advance tail lgkm0. Midpoint lgkm0 and cross-wave barrier retire all current stage readers before K+2 overwrite.',
            'Midpoint vmcnt0/lgkm0 and barrier publish K+1 direct-LDS writes before any K+1 ds_read_b128. K+2 writes target the opposite current stage.',
            'Each A fragment is replaced after its eight N uses; B fragments are replaced only during the last M repeat. Built ISA has no premature next-matrix read crossing midpoint.',
            'Terminal lgkm0 follows all 22 final-tile operand reads. Both odd/even terminal paths retain a workgroup barrier before C aliases matrix LDS.',
            'loops1 skips advance and retains original second prologue barrier, protecting final C alias.',
            'Scale reads are in a separate invariant LDS region and are consumed only after matching waits; scale panel loading/publishing bytes and guards are unchanged.'
        ],
        'limits': ['ISA synchronization review is not a numerical test or speedup claim.',
                   'ATT barrier share is correlated arrival skew, not removable whole-call wall time.',
                   'Original unbounded event_runner.py is prepared for traceability; root uses bounded_runner.py for giant-output guards/Event.']
    }
    (HERE / 'isa_source_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'review_sha256': sha(HERE / 'isa_source_review.json'),
                      'candidate_CO': result['candidate_code_object_sha256'], 'candidate_library': result['candidate_library_sha256']}))


if __name__ == '__main__':
    main()
