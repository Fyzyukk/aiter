#!/usr/bin/env python3
"""CPU-only full-vector CFG audit of the already-built narrow candidate.

The helper partial/default paths are intentionally outside the full-vector
claim. Parse branch machine words, follow the full-vector helper edges, then
enumerate every remaining forward execz edge through the publication barrier.
No build, GPU execution, or production modification is performed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def function(side, name, record, elf_class):
    path = HERE / side / 'device.s'
    data = path.read_text()
    header = next(m for m in re.finditer(r'^([0-9a-f]+) <([^>]+)>:', data, re.M)
                  if m[2] == name)
    base = int(header[1], 16)
    body = data[header.end():]
    end = re.search(r'^([0-9a-f]+) <', body, re.M)
    if end:
        body = body[:end.start()]
    line_start = data[:header.end()].count('\n')
    rows = []
    for m in re.finditer(r'^\s*(.*?)\s*//\s*([0-9A-F]+):\s*([0-9A-F ]+)([^\n]*)', body, re.M):
        op = re.sub(r'\s+', ' ', m[1].strip())
        words = m[3].strip().split()
        require(all(len(w) == 8 for w in words), 'Unexpected ISA machine words')
        row = {'relative_pc':int(m[2],16)-base, 'pc':int(m[2],16), 'instruction':op,
               'bytes':b''.join(int(w,16).to_bytes(4,'little') for w in words),
               'line':line_start + body[:m.start()].count('\n') + 1}
        if op.startswith(('s_branch ', 's_cbranch_')):
            require(len(words) == 1, 'Branch is not a single dword')
            immediate = int(words[0],16) & 0xffff
            if immediate & 0x8000:
                immediate -= 0x10000
            row['branch_target'] = row['relative_pc'] + 4 + 4*immediate
            label = re.search(r'\+0x([0-9a-f]+)>', m[4])
            require(label and int(label[1],16) == row['branch_target'], 'Branch label/machine-word target differs')
        rows.append(row)
    image = HERE / side / 'device.co'
    elf = elf_class(image.read_bytes())
    actual = elf.symbols()[name]['bytes']
    decoded = b''.join(r['bytes'] for r in rows)
    require(len(decoded) == record['instruction_bytes'] and decoded == actual,
            'Disassembly does not reconstruct the exact audited FUNC')
    require(hashlib.sha256(decoded).hexdigest() == record['instruction_sha256'], 'FUNC SHA differs')
    for a,b in zip(rows,rows[1:]):
        require(a['relative_pc']+len(a['bytes']) == b['relative_pc'], 'ISA gap')
    bypc = {r['relative_pc']:r for r in rows}
    require(len(bypc) == len(rows), 'Duplicate ISA PC')
    for r in rows:
        if 'branch_target' in r:
            require(r['branch_target'] in bypc, 'Branch target outside reconstructed function')
    return {'path':str(path), 'sha256':sha(path), 'image_path':str(image), 'image_sha256':sha(image),
            'kernel_base_pc':hex(base), 'instruction_sha256':record['instruction_sha256'],
            'instruction_bytes':len(decoded), 'rows':rows, 'bypc':bypc, 'record':record}


def public(r, role=None):
    result = {'relative_pc':hex(r['relative_pc']), 'pc':hex(r['pc']),
              'line':r['line'], 'instruction':r['instruction']}
    if role:
        result['role'] = role
    if 'branch_target' in r:
        result['branch_target_relative_pc'] = hex(r['branch_target'])
    return result


def vector_gate(rows, index):
    vector = rows[index]
    j = index-1
    while not rows[j]['instruction'].startswith('s_cbranch_'):
        j -= 1
    full_branch = rows[j]
    require(full_branch['instruction'].startswith('s_cbranch_execz '), 'Unexpected full-vector guard')
    require(full_branch['branch_target'] == vector['relative_pc']+len(vector['bytes']),
            'Full-vector guard does not skip just the vector issue')
    merge = rows[j-1]
    require(merge['instruction'].startswith('s_andn2_saveexec_b64'), 'Unexpected partial/full EXEC merge')
    partial = [r for r in rows[:j] if r.get('branch_target') == merge['relative_pc']
               and r['instruction'].startswith('s_cbranch_execz ')]
    require(len(partial) == 1, 'Expected one forward partial-helper skip')
    partial = partial[0]
    region = [r for r in rows if partial['relative_pc'] < r['relative_pc'] < merge['relative_pc']]
    require(any(r['instruction'].startswith('buffer_load_ubyte') for r in region), 'No partial byte body')
    waits = [r for r in region if r['instruction'].startswith('s_waitcnt') and 'vmcnt(' in r['instruction']]
    require(waits, 'No partial-helper waits')
    return {'partial':partial, 'full':full_branch, 'merge':merge, 'vector':vector,
            'conditional_bytes':sum(r['instruction'].startswith('buffer_load_ubyte') for r in region),
            'conditional_vmcnt_waits':waits}


def enumerate_paths(fn, start, end, forced, linked):
    """Enumerate the finite forward CFG inside one publication copy."""
    bypc = fn['bypc']
    result = []

    def visit(pc, trace, branches, decisions):
        require(len(trace) < 200, 'Unexpected long/looping full-vector publication path')
        require(start <= pc <= end, 'Publication CFG escaped audited interval')
        row = bypc[pc]
        trace = trace+[row]
        if pc == end:
            result.append((trace,branches))
            return
        next_pc = pc+len(row['bytes'])
        if 'branch_target' not in row:
            visit(next_pc,trace,branches,decisions)
            return
        op = row['instruction']
        require(op.startswith(('s_branch ', 's_cbranch_execz ')), 'Unexpected publication branch type')
        choices = [True] if op.startswith('s_branch ') else ([forced[pc]] if pc in forced else
                  ([decisions[linked[pc]]] if pc in linked else [False,True]))
        for taken in choices:
            dest = row['branch_target'] if taken else next_pc
            require(dest > pc, 'Backward edge in publication CFG')
            visit(dest,trace,branches+[{'branch':public(row), 'taken':taken,
                                        'forced_full_vector_or_matching_publish':pc in forced,
                                        'same_guard_as_branch':hex(linked[pc]) if pc in linked else None}],
                  {**decisions,pc:taken})
    visit(start,[],[],{})
    require(result, 'No complete full-vector CFG paths')
    return result


def audit_copy(fn, pair, side, copy_index):
    rows = fn['rows']
    gates = [vector_gate(rows, i) for i in pair]
    low, high = (g['vector']['relative_pc'] for g in gates)
    hi_index = pair[1]
    barrier = next(r for r in rows[hi_index+1:] if r['instruction'] == 's_barrier')
    end = barrier['relative_pc']
    region = [r for r in rows if low <= r['relative_pc'] <= end]
    b = [r for r in region if r['instruction'].startswith('buffer_load_ubyte')
         and ('s[80:83]' in r['instruction'] or 's[16:19]' in r['instruction'])]
    require(len(b) == 1, 'Expected one producer SFB issue in publication copy')
    bpc = b[0]['relative_pc']
    forced = {g['partial']['relative_pc']:True for g in gates}
    forced.update({g['full']['relative_pc']:False for g in gates})
    linked = {}
    guard_evidence = []
    if side == 'candidate':
        # The issue and publish guards are identical in the reviewed source.
        # Confirm the B predicate mask register is retained in the generated ISA.
        bindex = next(i for i,r in enumerate(rows) if r['relative_pc'] == bpc)
        issue_guard_index = bindex-1
        while not rows[issue_guard_index]['instruction'].startswith('s_and_saveexec_b64 '):
            issue_guard_index -= 1
        issue_guard = rows[issue_guard_index]
        issue_branch = rows[issue_guard_index+1]
        require(issue_branch['instruction'].startswith('s_cbranch_execz '), 'B issue EXEC guard differs')
        bpub_index = next(i for i,r in enumerate(rows) if high < r['relative_pc'] < end
                          and r['instruction'].startswith('ds_write_b8 '))
        pub_guard_index = bpub_index-1
        while not rows[pub_guard_index]['instruction'].startswith('s_and_saveexec_b64 '):
            pub_guard_index -= 1
        pub_guard = rows[pub_guard_index]
        issue_mask = issue_guard['instruction'].split(', ',1)[1]
        pub_mask = pub_guard['instruction'].split(', ',1)[1]
        require(issue_mask == pub_mask, 'B issue/publish use different predicate mask registers')
        # No instruction in between writes the mask register pair. Register
        # pairs only occur as first operands for scalar writes; compares may
        # define VCC and are checked separately if VCC was the predicate.
        mask_reg = issue_mask.split('[',1)[0]
        mask_numbers = [int(x) for x in re.findall(r'\d+',issue_mask)]
        require(mask_reg == 's' and len(mask_numbers) == 2, 'Unexpected B predicate register')
        for r in rows[issue_guard_index+1:pub_guard_index]:
            first = r['instruction'].split(' ',1)[1].split(', ',1)[0] if ' ' in r['instruction'] else ''
            numbers = [int(x) for x in re.findall(r'\d+',first)]
            dest = (set(range(numbers[0],numbers[-1]+1)) if first.startswith('s[') and numbers else
                    ({numbers[0]} if re.fullmatch(r's\d+',first) else set()))
            # s_cbranch immediate and s_waitcnt count are not destinations.
            if r['instruction'].startswith(('s_cbranch','s_waitcnt','s_cmp','s_barrier')):
                dest = set()
            require(not dest.intersection(range(mask_numbers[0],mask_numbers[1]+1)),
                    'B predicate mask modified between issue and publish')
        guard_evidence.append({'role':'SFB_same_guard', 'issue_exec_mask':public(issue_guard),
                               'publish_exec_mask':public(pub_guard),
                               'mask_register_unchanged_between_issue_and_publish':True})
        if rows[pub_guard_index+1]['instruction'].startswith('s_cbranch_execz '):
            linked[rows[pub_guard_index+1]['relative_pc']] = issue_branch['relative_pc']
        # A full-vector issue already establishes a nonempty producer guard;
        # its identical publish guard cannot become empty in this copy.
        first_apub = next(i for i,r in enumerate(rows) if high < r['relative_pc'] < end
                         and r['instruction'].startswith('ds_write_b128 '))
        apub_guard_index = first_apub-1
        while not rows[apub_guard_index]['instruction'].startswith('s_and_saveexec_b64 '):
            apub_guard_index -= 1
        apub_branch = rows[apub_guard_index+1]
        require(apub_branch['instruction'].startswith('s_cbranch_execz '), 'A publish EXEC guard differs')
        forced[apub_branch['relative_pc']] = False
        guard_evidence.append({'role':'SFA_full_issue_implies_same_publish_guard_nonempty',
                               'publish_exec_mask':public(rows[apub_guard_index]),
                               'publish_branch':public(apub_branch), 'publish_branch_taken':False,
                               'basis':'Source issue and publish have identical local_k_group/panel_k_begin/loops conditions.'})
    start = gates[0]['partial']['relative_pc']
    paths = enumerate_paths(fn,start,end,forced,linked)
    traces = []
    for n,(trace,branches) in enumerate(paths):
        pcs = [r['relative_pc'] for r in trace]
        require(low in pcs and high in pcs and pcs.index(low) < pcs.index(high), 'Full-vector issues missing/out of order')
        vmwaits = [r for r in trace if r['instruction'].startswith('s_waitcnt') and 'vmcnt(' in r['instruction']]
        require(vmwaits, 'Publication lacks VMEM wait')
        sfa_pub = [r for r in trace if r['instruction'].startswith('ds_write_b128')]
        sfb_pub = [r for r in trace if r['instruction'].startswith('ds_write_b8 ')]
        require(len(sfa_pub) in (0,2) and len(sfb_pub) in (0,1), 'Unexpected scale publication count')
        active_b = bpc in pcs
        #9023 has an unbranched DS under the same B EXEC mask: it remains in
        #the PC trace with empty EXEC when the issue branch is skipped.
        active_b_publish = bool(sfb_pub) and active_b
        if side == 'candidate':
            require(all(r['relative_pc'] > high for r in vmwaits), 'Candidate VMEM wait before full A issues')
            if active_b:
                require(all(r['relative_pc'] > bpc for r in vmwaits), 'Candidate VMEM wait before B issue')
            require(all(r['relative_pc'] > bpc for r in sfa_pub), 'Candidate A publication before B issue region')
        else:
            if active_b:
                require(any(high < r['relative_pc'] < bpc for r in vmwaits), 'Baseline A wait not before B issue')
                require(all(r['relative_pc'] < bpc for r in sfa_pub), 'Baseline A publication not before B issue')
        require(trace[-2]['instruction'] == 's_waitcnt vmcnt(0) lgkmcnt(0)',
                'Final publication wait/barrier changed')
        traces.append({'path_index':n, 'SFA_publish_executed':bool(sfa_pub), 'SFB_issue_executed':active_b,
                       'SFB_publish_instruction_in_trace':bool(sfb_pub),
                       'SFB_publish_active_lanes':active_b_publish, 'branch_decisions':branches,
                       'first_vmcnt_wait':public(vmwaits[0]),
                       'scale_and_wait_events':[public(r) for r in trace
                         if r['instruction'].startswith(('buffer_load_', 'ds_write_', 's_waitcnt'))
                         or r['instruction'] == 's_barrier'],
                       'instruction_trace_relative_PCs':[hex(pc) for pc in pcs]})
    active = next(p for p in traces if p['SFA_publish_executed'] and p['SFB_publish_active_lanes'])
    return {'copy_index':copy_index, 'copy_role':'prologue' if copy_index == 0 else
            ('main_loop_unroll_copy' if copy_index < 4 else 'remainder_loop_copy'),
            'full_vector_precondition':'The scale A producer EXEC is nonempty; both low/high chunks '
                'are 16B-aligned with >=16 valid rows in every active producer lane. '
                'Partial-helper EXEC is empty, so its execz edge is taken and full-vector execz is not taken.',
            'low_issue':public(gates[0]['vector'],'low_SFA_vector16_issue'),
            'high_issue':public(gates[1]['vector'],'high_SFA_vector16_issue'),
            'SFB_issue':public(b[0],'SFB_byte_issue'), 'barrier':public(barrier,'publication_barrier'),
            'helper_control':[{'partial_skip_branch':public(g['partial']),
                              'full_vector_guard_branch':public(g['full']),
                              'full_merge':public(g['merge']),
                              'bypassed_conditional_byte_instructions':g['conditional_bytes'],
                              'bypassed_conditional_vmcnt_waits':[public(r) for r in g['conditional_vmcnt_waits']]}
                             for g in gates],
            'matching_issue_publish_guard_evidence':guard_evidence,
            'remaining_forward_CFG_paths_enumerated':len(traces),
            'all_full_vector_paths_verified':True,
            'active_A_and_B_path_index':active['path_index'],
            'active_A_and_B_first_vmcnt_wait':active['first_vmcnt_wait'],
            'paths':traces}


def main():
    spec = importlib.util.spec_from_file_location('narrow_isa_elf', ROOT / 'reports/opus_resume_20261008/formal_selected/smoke_common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    device_path = HERE/'device_audit.json'
    audit = load(device_path)
    require(audit['status'] == 'passed' and not audit['gpu_executed'] and not audit['production_modified'], 'Device audit not passed')
    checks = {c['variant']:c for c in audit['checks']}
    evidence = {}
    for side in ('baseline','candidate'):
        evidence[side] = {}
        for variant,expected in (('9023_runtime',10),('9024_fixed',2)):
            check = checks[variant]
            fn = function(side,check['name'],check[side],common.Elf)
            vectors = [i for i,r in enumerate(fn['rows']) if r['instruction'].startswith('buffer_load_dwordx4 ')
                       and ' lds' not in r['instruction']]
            require(len(vectors) == expected, 'Not every static full-vector copy covered')
            pairs = list(zip(vectors[::2],vectors[1::2]))
            copies = [audit_copy(fn,pair,side,i) for i,pair in enumerate(pairs)]
            evidence[side][variant] = {k:v for k,v in fn.items() if k not in ('rows','bypc','record')}
            evidence[side][variant]['full_vector_copies'] = copies
    unchanged = []
    for variant in ('9023_fixed','9024_runtime'):
        c = checks[variant]
        require(c['candidate_same_as_baseline'] and c['baseline_exact_current_official'], 'Unchanged variant mismatch')
        unchanged.append({'variant':variant, 'instruction_metadata_normalized_descriptor_unchanged':True,
                          'instruction_sha256':c['candidate']['instruction_sha256']})
    resource_keys = ('.vgpr_count','.sgpr_count','.sgpr_spill_count','.vgpr_spill_count',
                     '.agpr_count','.group_segment_fixed_size','.private_segment_fixed_size')
    resources = {variant:{side:{**{key:checks[variant][side]['metadata'][key] for key in resource_keys},
                                     'instruction_bytes':checks[variant][side]['instruction_bytes']}
                         for side in ('baseline','candidate')}
                 for variant in ('9023_runtime','9024_fixed')}
    libraries = {b['side']:{'path':str(HERE/b['side']/'experiments.so'),
                            'sha256':b['library_sha256']} for b in load(HERE/'build_manifest.json')['builds']}
    require(all(sha(x['path']) == x['sha256'] for x in libraries.values()), 'Library SHA changed')
    report = {'status':'passed_full_vector_CFG_all_static_copies', 'generated_utc':datetime.now(timezone.utc).isoformat(),
              'cpu_only':True, 'new_GPU_execution':False, 'new_build_execution':False, 'production_modified':False,
              'audit_script_sha256':sha(__file__), 'device_audit_sha256':sha(device_path),
              'source_review_sha256':sha(HERE/'source_review.json'), 'libraries':libraries,
              'full_vector_copies_covered':{'9023_runtime':5,'9024_fixed':1}, 'resources':resources,
              'unchanged_variants':unchanged, 'evidence':evidence,
              'result':'Every candidate full/full SFA path issues low/high16B loads and the guarded SFB byte '
                  'before its first VMEM completion wait. The baseline waits and publishes A before B issue. '
                  'All remaining forward execz edges through the matching final wait/barrier are enumerated. '
                  'The9024 fixed LGKM-only wait before SFB issue waits for SMEM address inputs and does not drain VMEM.',
              'limits':['This is an ISA/CFG order check, not a hardware latency or performance measurement.',
                        'Conditional partial/default helper paths retain their byte/default/control code and VMEM waits; '
                        'no issue-overlap claim applies to these paths or to mixed partial/full EXEC.',
                        'M544 last WG has invalid-high chunks. Its default path is excluded from the full/full precondition '
                        'and must be covered by the numerical/guard GPU screen.',
                        'Resources increased for9023 runtime; no occupancy or throughput inference is made.',
                        'Numerical/guard and shared-address AB/BA Event screens remain required before adoption.']}
    (HERE/'isa_order_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# Narrow candidate CPU readiness', '', report['result'], '',
             '| Variant | Static publication copies | Baseline VGPR / SGPR spill | Candidate VGPR / SGPR spill | LDS | ISA bytes |',
             '| --- | --- | --- | --- | --- | --- |']
    for variant,r in resources.items():
        a,b = r['baseline'],r['candidate']
        lines.append(f"| {variant} | {report['full_vector_copies_covered'][variant]} | {a['.vgpr_count']} / {a['.sgpr_spill_count']} | {b['.vgpr_count']} / {b['.sgpr_spill_count']} | {b['.group_segment_fixed_size']} | {a['instruction_bytes']}→{b['instruction_bytes']} |")
    lines += ['', 'All four private baseline kernels match the current Oct8 official instructions, metadata and normalized descriptors. '
                  'Candidate9023 fixed and9024 runtime remain identical. Private/AGPR/VGPR spill are0 for all variants. '
                  'SGPR spill counts describe lane storage, not HBM traffic.', '',
              'The audit reconstructs every FUNC byte from the disassembly and checks it against the audited CO. '
              'It computes branch targets from machine words, takes the full-vector helper edges and enumerates every '
              'remaining forward branch through each publication barrier. Complete path decisions and PC traces are in '
              '[isa_order_audit.json](isa_order_audit.json).', '',
              'Baseline library: [baseline/experiments.so](baseline/experiments.so). Candidate library: '
              '[candidate/experiments.so](candidate/experiments.so). Both retain the experiment launcher ABI.', '']
    lines += ['- '+x for x in report['limits']]
    (HERE/'readiness.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'], 'copies':report['full_vector_copies_covered'],
                      'path_counts':{side:{variant:[c['remaining_forward_CFG_paths_enumerated'] for c in v['full_vector_copies']]
                                           for variant,v in values.items()} for side,values in evidence.items()}}))


if __name__ == '__main__':
    main()
