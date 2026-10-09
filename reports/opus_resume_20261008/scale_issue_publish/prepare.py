#!/usr/bin/env python3
"""Prepare an isolated 9021 scale issue/publish experiment, without GPU/build.

Source inputs are the Oct8 SFA experiment's unchanged Oct7-selected baseline.
Only launch.hip and include sources are copied. Existing different destination
files are rejected so rerunning cannot overwrite a later experimental edit.
"""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / 'sfa_packed' / 'baseline'
KERNEL = Path('include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh')
EXPECTED_BASELINE_SHA256 = 'f42695b106bbaaa9564cdd2cdaf128a3218c653dd78e213a26d4361b7d79e836'

OLD_HELPERS = '''    // Scale A global memory -> VGPR -> LDS, retaining raw E8M0 bytes.
    auto load_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = sfa_smem_offsets[pass];
            const int local_k_group = smem_offset / T::B_M;
            const int local_row = smem_offset % T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                const auto raw = layout_9021::load_sfa_vector<T>(
                    g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin),
                    kargs.m - row - local_row);
                store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
            }
        });
    };
    // Scale B global memory -> VGPR -> LDS, one byte per K128 group.
    auto load_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)),
                     sfb_smem_offsets[0]);
        }
    };
'''

NEW_HELPERS = '''    // Issue both raw scale reads before publishing either panel to LDS.
    // Unissued lanes are never published: issue and publish use identical guards.
    array<vector_t<D_SF, T::VEC_SCALE_A>, T::SFA_PASSES> raw_sfa_panel;
    vector_t<D_SF, 1> raw_sfb_panel;
    auto issue_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = sfa_smem_offsets[pass];
            const int local_k_group = smem_offset / T::B_M;
            const int local_row = smem_offset % T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                raw_sfa_panel[pass] = layout_9021::load_sfa_vector<T>(
                    g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin),
                    kargs.m - row - local_row);
            }
        });
    };
    auto issue_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            raw_sfb_panel = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin));
        }
    };
    auto publish_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = sfa_smem_offsets[pass];
            const int local_k_group = smem_offset / T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                store<T::VEC_SCALE_A>(s_sfa, raw_sfa_panel[pass], smem_offset);
            }
        });
    };
    auto publish_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, raw_sfb_panel, sfb_smem_offsets[0]);
        }
    };
'''

OLD_PROLOGUE = '''    load_sfa_panel(0);
    load_sfb_panel(0);
'''
NEW_PROLOGUE = '''    issue_sfa_panel(0);
    issue_sfb_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    publish_sfa_panel(0);
    publish_sfb_panel(0);
'''
OLD_REFILL = '''            load_sfa_panel(tile_k + 1);
            load_sfb_panel(tile_k + 1);
'''
NEW_REFILL = '''            __builtin_amdgcn_sched_barrier(0);
            issue_sfa_panel(tile_k + 1);
            issue_sfb_panel(tile_k + 1);
            __builtin_amdgcn_sched_barrier(0);
            publish_sfa_panel(tile_k + 1);
            publish_sfb_panel(tile_k + 1);
            __builtin_amdgcn_sched_barrier(0);
'''
CHANGES = [(OLD_HELPERS, NEW_HELPERS), (OLD_PROLOGUE, NEW_PROLOGUE),
           (OLD_REFILL, NEW_REFILL)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def guarded_write(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'Refusing to overwrite a different prepared source: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def synchronization(text):
    pattern = r'(?:s_waitcnt_vmcnt|s_waitcnt_lgkmcnt|__builtin_amdgcn_s_barrier)\([^;]*?\);'
    return re.findall(pattern, text)


def main():
    files = sorted([INPUT / 'launch.hip', *[p for p in (INPUT / 'include').rglob('*') if p.is_file()]])
    if not files or not (INPUT / KERNEL).is_file():
        raise ValueError('Missing frozen source baseline')
    baseline_bytes = (INPUT / KERNEL).read_bytes()
    if sha(baseline_bytes) != EXPECTED_BASELINE_SHA256:
        raise ValueError('9021 input is not the verified Oct7 selected baseline')
    baseline = baseline_bytes.decode()
    candidate = baseline
    for old, new in CHANGES:
        if candidate.count(old) != 1:
            raise ValueError('Expected exactly one frozen source block to replace')
        candidate = candidate.replace(old, new, 1)
    reverted = candidate
    for old, new in reversed(CHANGES):
        if reverted.count(new) != 1:
            raise ValueError('Candidate reverse audit failed')
        reverted = reverted.replace(new, old, 1)
    assert reverted == baseline
    assert synchronization(candidate) == synchronization(baseline)
    assert baseline.split('// Main loop\n', 1)[1].replace(OLD_REFILL, NEW_REFILL, 1) == candidate.split('// Main loop\n', 1)[1]
    assert baseline.split('// Epilogue\n', 1)[1] == candidate.split('// Epilogue\n', 1)[1]
    assert baseline.split('// Scale A global memory -> VGPR -> LDS', 1)[0] == candidate.split('// Issue both raw scale reads', 1)[0]
    assert candidate.count('issue_sfa_panel(') == candidate.count('issue_sfb_panel(') == 2
    assert candidate.count('publish_sfa_panel(') == candidate.count('publish_sfb_panel(') == 2
    assert candidate.count('__builtin_amdgcn_sched_barrier(0);') == baseline.count('__builtin_amdgcn_sched_barrier(0);') + 4
    traits = (INPUT / 'include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh').read_text()
    assert 'VEC_SCALE_A = 16' in traits and 'static_assert(SFA_PASSES == 1' in traits and 'SCALE_PANEL = 32' in traits
    rows = []
    for path in files:
        relative = path.relative_to(INPUT)
        content = path.read_bytes()
        changed = relative == KERNEL
        candidate_content = candidate.encode() if changed else content
        guarded_write(ROOT / 'baseline' / relative, content)
        guarded_write(ROOT / 'candidate' / relative, candidate_content)
        rows.append({'path': str(relative), 'source_path': str(path.resolve()),
                     'baseline_sha256': sha(content), 'candidate_sha256': sha(candidate_content),
                     'changed': changed})
    changed_files = [r['path'] for r in rows if r['changed']]
    assert changed_files == [str(KERNEL)]
    diff = ''.join(difflib.unified_diff(baseline.splitlines(keepends=True),
                                       candidate.splitlines(keepends=True),
                                       fromfile='baseline/' + str(KERNEL),
                                       tofile='candidate/' + str(KERNEL)))
    (ROOT / 'source_changes.diff').write_text(diff)
    manifest = {
        'status': 'cpu_source_prepared_not_built_not_gpu_tested',
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': '9021 raw SFA/SFB issue before LDS publish',
        'evidence_trigger': 'ATT short-K scale producer serialized SFA load/wait/store then SFB load/wait/store while other three waves wait at the scale publication barrier',
        'baseline_source': str(INPUT.resolve()), 'baseline': 'Oct7 selected unchanged9021',
        'baseline_9021_sha256': EXPECTED_BASELINE_SHA256,
        'gpu_access': False, 'build_executed': False, 'production_source_modified': False,
        'historical_records_modified': False, 'changed_source_files': changed_files,
        'source_files': rows, 'prepare_script_sha256': sha(Path(__file__).read_bytes()),
        'source_diff_sha256': sha(diff.encode()),
        'source_audits': {
            'candidate_reverse_transform_exactly_matches_baseline': True,
            'only_9021_header_changed': True,
            'launch_and_traits_and_helpers_and_other_kernels_identical': True,
            'all_existing_vmcnt_lgkmcnt_and_runtime_barrier_calls_identical': True,
            'all_existing_sched_barriers_retained': True,
            'added_sched_barrier_zero_fences': 4,
            'main_loop_identical_except_scale32_refill_call_block': True,
            'epilogue_identical': True,
            'SFA_tail_load_helper_and_0x7f_default_identical': True,
            'issue_publish_guards_identical_to_original_scale_guards': True,
            'raw_SFA_type': 'array<vector_t<unsigned char,16>,1>',
            'raw_SFB_type': 'vector_t<unsigned char,1>',
            'SCALE_PANEL': 32
        },
        'scope': 'Prologue and SCALE_PANEL32 refill only; matrix issue order, main-loop MFMA, addresses, LDS layout, scale consumer, output and traits stay identical',
        'mechanism_hypothesis': 'Issue both independent scale reads before either dependent LDS publication, reducing serialization in the producer wave and other waves barrier wait',
        'required_next_checks': [
            'Compile both sides with identical flags; verify candidate raw SFA and raw SFB VMEM reads precede the first dependent publish in actual prologue and scale32 refill ISA.',
            'Compiler-inserted waits inside load_sfa_vector, especially byte tail paths, may retain serialization; sched_barrier is a compiler constraint, not a runtime completion barrier.',
            'Check full metadata and descriptor, added VGPR/live ranges, spills and any residency threshold change.',
            'Validate loops1/2, M tails and nonaligned SFA, K4096/4224 and subsequent scale panel refills before performance adoption.',
            'Use unchanged selected baseline versus candidate shared-address Event timing and output/guard checks; ATT/profile changes alone do not establish speedup.'
        ],
        'risks': [
            'Retained raw SFA vector16 and raw SFB byte overlap in lifetime and can raise VGPR demand or change scalar/vector scheduling.',
            'The frontend may insert a wait while issuing the conditional vector/tail SFA path; source issue order does not prove actual VMEM overlap.',
            'Changing the scale producer scheduling can move the dominant wait to matrix supply, issue, or output. No new performance claim is made.'
        ]
    }
    (ROOT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'source_files': len(rows),
                      'changed_source_files': changed_files, 'manifest': str(ROOT / 'source_manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
