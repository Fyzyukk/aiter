#!/usr/bin/env python3
"""Independent read-only small_split peer review, with a no-runtime CPU executable."""
import collections, csv, hashlib, json, os, re, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
PEER = HERE.parent / "small_split"
ROOT = HERE.parents[2]
LLVM = Path('/opt/rocm-llvm23-46fcb339/bin')
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out = HERE / 'small_split_peer_review'
    if out.exists(): raise SystemExit('Refusing to overwrite peer review')
    out.mkdir()
    checked_paths = ['candidate/register_global_split.cuh','candidate/traits.cuh','launch.hip','contract.h',
        'frozen/gemm_include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh',
        'cpu_audit.json','build_receipt.json','coverage.csv','candidate_cases.csv','coverage_audit.json','layout_guard_check.cpp','audit.py']
    before = {p: sha(PEER / p) for p in checked_paths}
    reg = (PEER / checked_paths[0]).read_text()
    fine = (PEER / checked_paths[4]).read_text()
    launcher = (PEER / 'launch.hip').read_text()
    # These are copied literal arithmetic statements from the actual device inputs.
    a = reg.index('    const int total_tiles =')
    b = reg.index('    const int scale_groups =', a)
    reg_part = reg[a:b].replace('    const int split = block_id_z();\n', '')
    a = fine.index('    const int total_loops =')
    b = fine.index('    const int active_stages =', a)
    fine_part = fine[a:b].replace('SplitK', 'T::SPLIT_K')
    rows = list(csv.DictReader((PEER / 'coverage.csv').open()))
    assert len(rows) == len({(r['M'],r['N'],r['K']) for r in rows}) == 90
    frozen = list(csv.DictReader((PEER / 'frozen/losers294.csv').open()))
    parents = {9040,9041,9042,9051,9052,9053,9054,9060,9061,9062,9063}
    assigned = [r for r in frozen if int(r['opus_parent_kid']) in parents]
    current = list(csv.DictReader((ROOT / 'reports/opus_flydsl_gap_20261009/losers294.csv').open()))
    assert {(r['M'],r['N'],r['K'],r['opus_parent_kid']) for r in assigned} == {(r['M'],r['N'],r['K'],r['opus_parent_kid']) for r in rows}
    assert {(r['M'],r['N'],r['K'],r['opus_parent_kid']) for r in assigned} == {(r['M'],r['N'],r['K'],r['opus_parent_kid']) for r in current if int(r['opus_parent_kid']) in parents}
    frozen_by_shape = {(r['M'],r['N'],r['K']):r for r in assigned}
    for row in rows:
        source = frozen_by_shape[row['M'],row['N'],row['K']]
        assert all(row[f] == source[f] for f in ['opus_parent_kid','actual_kid','actual_tile_M','actual_tile_N','actual_waves','global_splitK','fixed_K'])
    shapes = ',\n'.join('{'+','.join([r['M'],r['N'],r['K'],r['opus_parent_kid'],r['global_splitK'],r['actual_tile_M'],r['actual_tile_N'],'true' if r['pool']=='fine' else 'false'])+'}' for r in rows)
    header = '#pragma once\n#include "candidate/traits.cuh"\n#include "contract.h"\nstruct KernelArgs { int k; };\nstruct RegisterPartition {int global_begin,global_loops,tile_begin,loops;};\ntemplate<class T> RegisterPartition register_partition(int k,int split,int wk) { KernelArgs args{k};\n' + reg_part + 'return {global_begin,global_loops,tile_begin,loops};\n}\nstruct FinePartition {int tile_begin,loops;};\ntemplate<class T> FinePartition fine_partition(int k,int split) {KernelArgs args{k};\n' + fine_part + 'return {tile_begin,loops};\n}\nstruct ReviewedShape {int m,n,k,parent,baseline_split,baseline_bm,baseline_bn; bool fine;};\nconst ReviewedShape reviewed_shapes[] = {\n' + shapes + '\n};\n'
    (out / 'small_split_source_expressions.h').write_text(header)
    # Only OUTPUT3/runtime-K traits are instantiated by this new global split ABI.
    assert 'opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3>' in (PEER / 'candidate/traits.cuh').read_text()
    assert 'array<typename Base::vtype_c, T::E_M * T::E_N> c{};' in reg
    assert 'kt += tile_begin;' in reg and 'store<4>(gc, c[decltype(i)::value], offsets[decltype(i)::value]);' in reg
    assert reg.index('if (wk > 0) return;') > reg.index('__builtin_amdgcn_s_barrier();',reg.index('if constexpr (T::WAVE_K > 1)'))
    assert 'store<4>(gc, c[decltype(i)::value], offsets[decltype(i)::value], 0, number<T::STORE_CACHE>{});' in fine
    assert 'reinterpret_cast<const float*>(args.ptr_c), reinterpret_cast<opus::bf16_t*>(c)' in launcher
    assert 'args.ptr_c=cfg.split>1?workspace:c;' in launcher
    assert 'register_split_launch<opus_private_register_split4_traits<16,16,2>>' in launcher
    assert 'const int index = (block_id_x() * Block + thread_id_x()) * Vec;' in fine
    assert 'workspace + static_cast<int64_t>(decltype(split)::value) * elements' in fine
    cases = list(csv.DictReader((PEER / 'candidate_cases.csv').open()))
    assert len(cases) == 540
    assert len({(r['M'],r['N'],r['K'],r['variant']) for r in cases}) == 540
    counts = collections.Counter(int(r['variant']) for r in cases)
    assert all(counts[k] == (52 if k < 200 else 38) for k in counts) and len(counts) == 12
    for c in cases:
        m,n,k,v,bm,bn,split = (int(c[x]) for x in ['M','N','K','variant','tile_M','tile_N','global_splitK'])
        assert int(c['workspace_bytes']) == (split*m*n*4 if split > 1 else 0)
        assert int(c['producer_grid_x']) == n//bn and int(c['producer_grid_y']) == (m+bm-1)//bm and int(c['producer_grid_z']) == split
        assert int(c['kernel_calls']) == (2 if split>1 else 1)
        assert int(c['reducer_grid_x']) == ((m*n+2047)//2048 if split>1 else 0)
        assert m*n*(4 if split>1 else 2) <= 2147483647
        expected_ids = {'210','211','220','221','230','231'} if frozen_by_shape[c['M'],c['N'],c['K']]['opus_parent_kid'] in {'9060','9061','9062','9063'} else {'110','111','120','121','130','140'}
        assert c['variant'] in expected_ids
    audit = json.loads((PEER / 'cpu_audit.json').read_text())
    receipt = json.loads((PEER / 'build_receipt.json').read_text())
    assert all(sha(PEER / p) == h for p,h in receipt['sources'].items())
    assert all(sha(Path(b['library'])) == b['library_sha256'] for b in receipt['builds'])
    assert all(k['resources']['private_segment_fixed_size']==k['resources']['vgpr_spill_count']==k['resources']['sgpr_spill_count']==0 for lib in audit['libraries'] for k in lib['kernels'])
    peer_binary = Path(audit['layout_guard_check']['compile_argv'][-1]).with_suffix('')
    needed = subprocess.check_output([str(LLVM/'llvm-readelf'),'-d',str(peer_binary)],env=ENV,text=True)
    names = re.findall(r'Shared library: \[([^\]]+)\]',needed)
    assert not any('hip' in n.lower() or 'hsa' in n.lower() for n in names)
    assert sha(peer_binary) == audit['layout_guard_check']['binary_sha256']
    obj = out / 'small_split_review_check.o';binary = out / 'small_split_review_check'
    argv = [str(LLVM/'clang++'),'-x','hip','--offload-host-only','--rocm-path=/opt/rocm','--hip-path=/opt/rocm','-std=c++20','-O2','-D__HIPCC_RTC__=1',
        '-I'+str(out),'-I'+str(PEER),'-I'+str(PEER/'cpu_audit_no_runtime/host_include'),'-I'+str(PEER/'frozen'),'-I'+str(PEER/'frozen/gemm_include/gfx950'),'-c',str(HERE/'small_split_review_check.cpp'),'-o',str(obj)]
    compiled = subprocess.run(argv,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out / 'compile.log').write_text(compiled.stdout)
    if compiled.returncode: raise RuntimeError(compiled.stdout)
    link = [str(LLVM/'clang++'),str(obj),'-o',str(binary)]
    linked = subprocess.run(link,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out / 'link.log').write_text(linked.stdout)
    if linked.returncode: raise RuntimeError(linked.stdout)
    dynamic = subprocess.check_output([str(LLVM/'llvm-readelf'),'-d',str(binary)],env=ENV,text=True)
    (out / 'dynamic.txt').write_text(dynamic)
    own_names = re.findall(r'Shared library: \[([^\]]+)\]',dynamic)
    assert not any('hip' in n.lower() or 'hsa' in n.lower() for n in own_names)
    completed = subprocess.check_output([str(binary)],env=ENV,text=True).strip()
    assert all(sha(PEER / p) == h for p,h in before.items())
    report = {'status':'independent_read_only_source_CPU_review_passed_unvalidated_gpu_stopped','peer':'small_split','gpu_kernel_launches':0,'peer_modified':False,
        'reviewed_source_sha256':before,'assigned_shapes':90,'register_shapes':52,'fine_shapes':38,'candidate_cases':540,
        'all_partition_K128_counts':128,'actual_partition_expressions_extracted':True,'actual_traits_and_launcher_guards_used':True,
        'empty_partition_output_review':'Zero-initialized FP32 accumulators and unconditional surviving-wave partial stores cover empty global and local partitions. No input prefetch executes for zero loop count.',
        'scale_review':'Register kt always includes global and local starts. Fine GSA/GSB bases include tile_begin and per-tile offsets add local K; absolute K128 scales remain in bounds.',
        'ABI_review':'Candidate IDs match traits and config split counts. Producer receives workspace for split>1; matching FP32 reducer writes BF16 C on the same caller stream. Capacity/alignment/overlap checks cover required workspace; no allocation or host wait.',
        'unused_branch_limit':'OUTPUT4/N-tail branch of private copied register source is uninstantiated; its BF16 stores through float gc are outside this ABI and must not be enabled without a dedicated change.',
        'resource_review':'All14 main candidate kernels and31 baseline kernels report zero scratch and VGPR/SGPR spills; dynamic fine LDS is audited separately from metadata fixed LDS.',
        'peer_host_audit_needed_libraries':names,'independent_host_needed_libraries':own_names,'no_HIP_or_HSA_dependency_checked_before_independent_execution':True,
        'compile_argv':argv,'link_argv':link,'binary_sha256':sha(binary),'stdout':completed,
        'limits':'CPU mapping/source review cannot establish device synchronization behavior, signed/cancellation numerical correctness, occupancy or complete-call speed. Global split changes accumulation order and introduces workspace traffic plus reducer launch.'}
    (HERE/'small_split_read_only_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'assigned_shapes':90,'candidate_cases':540,'stdout':completed},indent=2))
if __name__=='__main__':main()
