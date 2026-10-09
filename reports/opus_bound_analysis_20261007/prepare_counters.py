#!/usr/bin/env python3
"""CPU-only verification and generation of small gfx950 PMC groups.

This reads installed YAML definitions. It does not query or initialize a GPU.
The SDK must still accept each group in a later idle-device collection.
"""
import hashlib
import json
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent
DEFS = Path('/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml')
SPEC = Path('/opt/rocm/libexec/rocprofiler-compute/utils/mi_gpu_spec.yaml')

GROUPS = {
    'sq_ea': {
        'purpose': 'Wave lifetime/waits, MFMA activity, physical DRAM bytes, TCP backpressure.',
        'counters': ['GRBM_COUNT', 'GRBM_GUI_ACTIVE', 'SQ_WAVE_CYCLES', 'SQ_WAVES',
            'SQ_WAIT_ANY', 'SQ_WAIT_INST_ANY', 'SQ_ACTIVE_INST_ANY', 'SQ_BUSY_CYCLES',
            'SQ_VALU_MFMA_BUSY_CYCLES', 'SQ_INSTS_VALU_MFMA_MOPS_F8',
            'TCC_EA0_RDREQ_DRAM_32B', 'TCC_EA0_WRREQ_WRITE_DRAM_32B',
            'TCP_TCP_TA_ADDR_STALL_CYCLES', 'TCP_TCR_TCP_STALL_CYCLES',
            'TCP_LFIFO_STALL_CYCLES', 'TCP_GATE_EN1'],
    },
    'l2_tagmap': {
        'purpose': 'L2 hit/tag pressure, four TCP tagram request shares, paper TA address rate.',
        'counters': ['GRBM_COUNT', 'GRBM_GUI_ACTIVE', 'TCC_HIT', 'TCC_MISS',
            'TCC_TAG_STALL', 'TCC_BUSY', 'TCP_TAGRAM0_REQ', 'TCP_TAGRAM1_REQ',
            'TCP_TAGRAM2_REQ', 'TCP_TAGRAM3_REQ', 'TA_BUFFER_WAVEFRONTS',
            'TA_BUFFER_TOTAL_CYCLES'],
    },
    'utcl1_credits': {
        'purpose': 'UTCL1 pressure and DRAM/GMI credits; TA downstream stalls.',
        'counters': ['GRBM_COUNT', 'GRBM_GUI_ACTIVE', 'TCP_UTCL1_SERIALIZATION_STALL',
            'TCP_UTCL1_THRASHING_STALL', 'TCP_UTCL1_STALL_INFLIGHT_MAX', 'TCP_GATE_EN1',
            'TCC_EA0_RDREQ_DRAM_CREDIT_STALL', 'TCC_EA0_RDREQ_GMI_CREDIT_STALL',
            'TCC_EA0_WRREQ_DRAM_CREDIT_STALL', 'TCC_EA0_WRREQ_GMI_CREDIT_STALL',
            'TA_ADDR_STALLED_BY_TC_CYCLES', 'TA_TA_BUSY'],
    },
    'ta_lds': {
        'purpose': 'Direct-LDS traffic, SQ address issue pressure, LDS waits/conflicts, TCP request pressure.',
        'counters': ['GRBM_COUNT', 'GRBM_GUI_ACTIVE', 'SQ_BUSY_CYCLES',
            'SQ_VMEM_TA_ADDR_FIFO_FULL', 'SQ_LDS_BANK_CONFLICT', 'SQ_WAIT_INST_LDS',
            'TCP_TCP_TA_ADDR_STALL_CYCLES', 'TCP_RFIFO_STALL_CYCLES',
            'TCP_READ_TAGCONFLICT_STALL_CYCLES', 'TCP_GATE_EN1',
            'TA_BUFFER_READ_LDS_WAVEFRONTS', 'TA_BUFFER_READ_WAVEFRONTS'],
    },
    'ea_crosscheck': {
        'purpose': 'Total EA read size mix vs DRAM reads; TA coalescable reads; translation hit/miss.',
        'optional': True,
        'counters': ['GRBM_COUNT', 'GRBM_GUI_ACTIVE', 'TCC_EA0_RDREQ_32B',
            'TCC_EA0_RDREQ_64B', 'TCC_EA0_RDREQ_128B', 'TCC_EA0_RDREQ_DRAM_32B',
            'TA_BUFFER_COALESCED_READ_CYCLES', 'TA_BUFFER_COALESCEABLE_WAVEFRONTS',
            'TCP_UTCL1_TRANSLATION_HIT', 'TCP_UTCL1_TRANSLATION_MISS',
            'TCP_TCC_READ_REQ', 'TCP_GATE_EN1'],
    },
    'dispatch': {
        'purpose': 'Optional SPI allocation investigation; CSN source defaults to CS0 and requires scope verification.',
        'optional': True,
        'counters': ['GRBM_COUNT', 'GRBM_GUI_ACTIVE', 'SQ_WAVE_CYCLES', 'SQ_WAVES',
            'SQ_BUSY_CYCLES', 'SPI_CSN_BUSY', 'SPI_CSN_NUM_THREADGROUPS',
            'SPI_CSN_WINDOW_VALID', 'SPI_RA_REQ_NO_ALLOC'],
    },
}

def main():
    definitions = yaml.safe_load(DEFS.read_text())['rocprofiler-sdk']['counters']
    available = {}
    for counter in definitions:
        for definition in counter.get('definitions', []):
            if 'gfx950' in definition.get('architectures', []):
                available[counter['name']] = {'description': counter.get('description'), **definition}
    hardware = yaml.safe_load(SPEC.read_text())['mi_gpu_spec']
    matches = [arch for series in hardware for arch in series['gpu_archs'] if arch['gpu_arch'] == 'gfx950']
    if len(matches) != 1:
        raise ValueError(f'Expected one gfx950 spec, found {len(matches)}')
    budget = matches[0]['perfmon_config']
    aliases, evidence = [], {}
    for name, group in GROUPS.items():
        use = {}
        aggregates = []
        for counter in group['counters']:
            definition = available[counter]
            block = definition['block']
            use[block] = use.get(block, 0) + 1
            operation = 'max' if counter.startswith('GRBM_') else 'sum'
            alias = f'OPUS_GFX950_{counter}_{operation.upper()}'
            aggregates.append(alias)
            if alias not in {c['name'] for c in aliases}:
                aliases.append({'name': alias, 'description': f'{operation} of gfx950 {counter}; {definition["description"]}',
                    'properties': [], 'definitions': [{'architectures': ['gfx950'],
                        'expression': f'reduce({counter},{operation})'}]})
        if any(count > budget[block] for block, count in use.items()):
            raise ValueError(f'{name} exceeds static hardware budget: {use}, {budget}')
        evidence[name] = {**group, 'aggregate_counters': aggregates, 'block_use': use,
            'static_budget_passed': True, 'runtime_sdk_group_acceptance': 'not_run'}
        (OUT / 'pmc' / f'{name}.txt').parent.mkdir(parents=True, exist_ok=True)
        (OUT / 'pmc' / f'{name}.txt').write_text('pmc: ' + ' '.join(aggregates) + '\n')
    extras = {'rocprofiler-sdk': {'counters-schema-version': 1, 'counters': aliases}}
    (OUT / 'gfx950_extra_counters.yaml').write_text(yaml.safe_dump(extras, sort_keys=False))
    selected = sorted({c for g in GROUPS.values() for c in g['counters']})
    result = {
        'status': 'cpu_definitions_verified_no_gpu_execution',
        'architecture': 'gfx950', 'counter_defs': str(DEFS),
        'counter_defs_sha256': hashlib.sha256(DEFS.read_bytes()).hexdigest(),
        'hardware_spec': str(SPEC), 'hardware_spec_sha256': hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        'perfmon_budget': budget, 'groups': evidence,
        'raw_definitions': {c: available[c] for c in selected},
        'derived_definitions': {c: available[c] for c in ['SIMD_NUM', 'CU_NUM', 'MfmaUtil', 'OccupancyPercent']},
        'normalization': {
            'dram_read_bytes': '32 * sum(TCC_EA0_RDREQ_DRAM_32B)',
            'dram_write_bytes': '32 * sum(TCC_EA0_WRREQ_WRITE_DRAM_32B)',
            'dram_tb_s': 'dram_bytes / dispatch_duration_ns / 1000',
            'mean_wave_life_cycles_estimate': '4 * sum(SQ_WAVE_CYCLES) / sum(SQ_WAVES), matched scope/window',
            'wave_wait_fraction': 'sum(SQ_WAIT_ANY or SQ_WAIT_INST_ANY) / sum(SQ_WAVE_CYCLES)',
            'issued_f8_operations': '512 * sum(SQ_INSTS_VALU_MFMA_MOPS_F8), includes padded/executed operations',
            'mfma_util_pct': '100 * sum(SQ_VALU_MFMA_BUSY_CYCLES) / (max(GRBM_GUI_ACTIVE) * simd_count)',
            'mean_resident_waves_per_cu_estimate': '4 * sum(SQ_WAVE_CYCLES) / (max(GRBM_GUI_ACTIVE) * CU_NUM)',
            'ta_paper_addr_rate': '4 * sum(TA_BUFFER_WAVEFRONTS) / sum(TA_BUFFER_TOTAL_CYCLES); mixed/direct-LDS semantics require verification',
            'tagram_share': 'sum(TCP_TAGRAMi_REQ) / sum(all four TCP_TAGRAMi_REQ); L2 request mapping, not all L1 lookups',
            'tcc_hit_fraction': 'sum(TCC_HIT) / (sum(TCC_HIT) + sum(TCC_MISS))',
        },
        'limits': [
            'Installed definitions and per-block budgets do not prove runtime SDK schedulability.',
            'GRBM/TA/TCC/TCP clock gate and selected stall counters are unwindowed; isolate GPU and inspect definitions.',
            'TCC_TAG_STALL probes and UTCL1_THRASHING_STALL can overlap; ratios are diagnostic estimates.',
            'Profiling durations are instrumented; use unprofiled run_perftest for speedup decisions.',
            'Collect one pass in one subprocess and keep source/process/agent/queue/dispatch identities separate.',
            'Use last N dispatches per kernel to exclude automatic rotation probe and warmup.',
            'For split-K report producer and reducer separately, and validated pairs for whole-call traffic/time.',
            'Logical tensor bytes and roofline logical FLOPs are separate from physical DRAM traffic and issued operations.',
        ],
    }
    (OUT / 'counter_evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'groups': {n:g['block_use'] for n,g in evidence.items()},
                      'counter_count':len(selected)}))

if __name__ == '__main__':
    main()
