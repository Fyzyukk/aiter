#!/usr/bin/env python3
"""Final common-loss campaign using the remote scale/interleave/narrow plans."""
import collections
import copy
import dataclasses
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint
from prepare_opus_repeat import csv_rows, key, registry_metadata, traits_args
from prepare_stage16_pin_compute import replace_once, verify_compute_order
from prepare_stage12_scale import verify_scale_cache

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage18_remote_final84_20261030"
ALIGN = LOCAL / "remote_alignment_54bef351_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
INCLUDE = Path("csrc/opus_gemm/include/gfx950")
PIN_HEADER = "opus_gemm_pipeline_stage18_early_scale_gfx950.cuh"
INTER_HEADER = "opus_gemm_pipeline_stage18_interleave_gfx950.cuh"
TRAITS = "opus_gemm_traits_stage18_remote_final_gfx950.cuh"
SCOPE = "a8w8_blockscale_bpreshuffle"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def early_scale_body():
    original = (ROOT / INCLUDE / "opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh").read_text()
    source = original.replace("opus_stage14_pin_detail", "opus_stage18_pin_detail").replace(
        "opus_stage14_pin_body", "opus_stage18_pin_body")
    assert source.count("        refill_scale_panel(tile + 1);") == 2
    source = source.replace("        refill_scale_panel(tile + 1);",
        "        if constexpr (!T::EARLY_SCALE_LOAD) refill_scale_panel(tile + 1);")
    anchor = "        // A half 1 x B half 0 -> C[1][0]"
    pending = '''        // Issue global scale reads after C00. These values live only
        // within this K iteration; LDS remains unchanged until its boundary.
        const bool early_scale_boundary = ((k_tile + 2) & scale_panel_mask) == 0 && k_tile + 2 < loops;
        opus::array<opus::vector_t<D_SF, T::VEC_SCALE_A>, T::SFA_PASSES_PER_CACHE_PANEL> pending_sfa_raw;
        D_SF_PACK pending_sfb_raw = 0;
        if constexpr (T::EARLY_SCALE_LOAD) {
            if (early_scale_boundary) {
                pending_sfa_raw = load_sfa_panel(k_tile + 2);
                if (wave_id < T::SCALE_N_HALVES && (!T::GUARD_SFB_LANE || lane_id < T::SCALE_PANEL_K_CAPACITY)) {
                    const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(k_tile + 2));
                    pending_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
                }
            }
        }

'''
    # The C10 comment also exists in the drain: change only the main loop.
    start = source.index("// ===== Main loop =====")
    position = source.index(anchor, start)
    source = source[:position] + pending + source[position:]
    boundary = '''        // The original handoff was at the start of the following
        // chunk. This is the same iteration boundary, after every old scale
        // consumer and before the first read of the replacement panel.
        if constexpr (T::EARLY_SCALE_LOAD) {
            if (early_scale_boundary) {
                s_waitcnt_lgkmcnt(0_I);
                opus::sync_threads();
                s_waitcnt_vmcnt(0_I);
                publish_sfa_panel(k_tile + 2, pending_sfa_raw);
                if (wave_id < T::SCALE_N_HALVES && (!T::GUARD_SFB_LANE || lane_id < T::SCALE_PANEL_K_CAPACITY)) {
                    store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, pending_sfb_raw), u_ssfb);
                }
                s_waitcnt_lgkmcnt(0_I);
                opus::sync_threads();
            }
        }
'''
    source = replace_once(source, "        stage = next_stage;\n        }\n    }\n    // ===== Epilogue =====",
        boundary + "        stage = next_stage;\n        }\n    }\n    // ===== Epilogue =====")
    begin = original.index("        // A half 0 x B half 0 -> C[0][0]", original.index("// ===== Main loop ====="))
    end = original.index("        v_sfb[1] = v_sfb_next[1];", begin)
    return source, verify_compute_order(original[begin:end])


def verify_scale_prefetch(panel, loops):
    original = [r for r in range(1, loops) if r % panel == 0]
    early = [(t, t + 2) for t in range(max(loops - 2, 0)) if (t + 2) % panel == 0 and t + 2 < loops]
    assert [r for t, r in early] == original
    # Original and new LDS publication both occur between t=R-2 and t=R-1.
    assert all(t == r - 2 and r < loops for t, r in early)
    return {"panel": panel, "loops": loops, "early_issue_and_publish": early,
        "all_boundaries_once_no_read_at_loops": True, "same_consumer_boundary": True}


def interleave_body():
    original = (ROOT / INCLUDE / "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh").read_text()
    source = original.replace("opus_gemm_bpreshuffle_wave_interleave_layout", "opus_stage18_interleave_layout").replace(
        "opus_gemm_bpreshuffle_wave_interleave_body", "opus_stage18_interleave_body").replace(
        "opus_gemm_bpreshuffle_wave_interleave_kernel", "opus_stage18_interleave_kernel")
    before = '''            async_load<16>(g_a, s_a.ptr + slot * T::A_SLOT + sa_offsets[step],
                ga_offsets[step], (slot % 2) * T::HALF_M * kargs.stride_a + k * 128, number<0>{}, number<T::CACHE_A>{});'''
    after = '''            constexpr int rows_per_wave = T::WARP_SIZE / (T::K_STAGE / T::VEC_A);
            constexpr int rows_per_block = T::BLOCK_SIZE / (T::K_STAGE / T::VEC_A);
            static_assert(T::HALF_M % rows_per_wave == 0);
            const int producer_row = wave_id * rows_per_wave + step * rows_per_block;
            // Do not skip actual tail-M zero writes: only rows with no consumer.
            if (!T::SKIP_UNUSED_A || producer_row < T::HALF_M) {
''' + before + '''
            }'''
    source = replace_once(source, before, after)
    before = '''        load_a(number<first>{}, k);
        load_a(number<first + 1>{}, k);
        load_b(number<first>{}, k);
        load_b(number<first + 1>{}, k);'''
    after = '''        if constexpr (T::B_FIRST) {
            load_b(number<first>{}, k);
            load_b(number<first + 1>{}, k);
            load_a(number<first>{}, k);
            load_a(number<first + 1>{}, k);
        } else {
''' + before + '''
        }'''
    source = replace_once(source, before, after)
    waits = r"s_waitcnt_\w+\([^;]*?\)|(?:opus::)?sync_threads\(\)"
    assert re.findall(waits, source) == re.findall(waits, original)
    checks = []
    for bm in (128, 192, 256):
        for wn in (2, 4):
            half, waves, rows_per_wave = bm // 2, 2 * wn, 8
            block_rows = waves * rows_per_wave
            steps = (half + block_rows - 1) // block_rows
            producers = {wave * 8 + step * block_rows + lane_row
                for wave in range(waves) for step in range(steps) for lane_row in range(8)
                if wave * 8 + step * block_rows < half}
            consumers = set(range(half))
            assert consumers == producers
            checks.append({"BM": bm, "WaveN": wn, "all_consumer_rows_initialized": True,
                "before_rows": steps * block_rows, "after_rows": len(producers),
                "tail_M_zero_writes_preserved": True})
    return source, checks


def verify_ring(em, ring):
    slots = list(range(ring))
    release = em - ring
    reads = set(range(ring))
    for m in range(em):
        if m == release:
            assert reads == set(range(em))
        assert slots[m % ring] == m
        if m + ring < em:
            slots[m % ring] = m + ring
            reads.add(m + ring)
    return {"E_M": em, "A_RING": ring, "release_m": release,
        "all_old_LDS_reads_before_release": True, "reuse_after_last_consumer": True}


def main():
    reference = LOCAL / "stage17_pin_invariants54_20261030/result/public_best_opus421.csv"
    rows = [r for r in csv_rows(reference) if r["status"] == "slower_than_csv"]
    assert len(rows) == 84
    alignment = json.loads((ALIGN / "set_alignment_summary.json").read_text())
    assert alignment["remote_only_wins"] == 0 and alignment["both_losses"] == 84
    before = registry(load_common(ROOT, label="stage18_before"));assert len(before) == 48261
    metadata, _ = registry_metadata();byname = {v["kernelName"]: kid for kid, v in metadata.items()}
    document = json.loads((ROOT / REL).read_text());original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    pin, pin_audit = early_scale_body();interleave, inter_audits = interleave_body()
    traits = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage15_pin_unroll_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh"

template<class Base, int Panel, bool Early>
struct opus_stage18_scale_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<Base, Panel> {
    static constexpr int LOOP_UNROLL = 2;
    static constexpr int SCHEDULE_POLICY = 0;
    static constexpr bool GUARD_SFB_LANE = true;
    static constexpr bool EARLY_SCALE_LOAD = Early;
};

template<class Base, bool SkipUnusedA, bool BFirst>
struct opus_stage18_interleave_traits : Base {
    static constexpr bool SKIP_UNUSED_A = SkipUnusedA;
    static constexpr bool B_FIRST = BFirst;
};

template<class Base, int Ring>
struct opus_stage18_m_repeat_traits : Base {
    static_assert(Ring >= 2 && Ring <= Base::E_M && Ring <= 3);
    static constexpr int A_RING = Ring;
};
'''
    points, matches = [], []
    def append(entry, parent, axis, comparison_only):
        for field in ("kid", "configKey", "supported_shapes"):
            entry.pop(field, None)
        kid = 101300 + len(points)
        entry["kid"] = kid;document["gfx950"].append(entry)
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        document["image_suffixes"][str(kid)] = f"stage18_remote_seed{parent}_{digest}"
        points.append({"new_kid": kid, "seed_kid": parent, "pipeline": entry["pipeline"],
            "axis": axis, "comparison_only": comparison_only})
        return kid
    for parent, panel in ((101011, 32), (101020, 64), (388, 32), (389, 64)):
        base = None
        for early in (False, True):
            e = copy.deepcopy(entries[parent]);e.update(traits_header=TRAITS,
                pipeline_header=PIN_HEADER if early else "opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh",
                body_func="opus_stage18_pin_body" if early else "opus_stage14_pin_body",
                body_args=[f"opus_stage18_scale_traits<{e['body_args'][0]}, {panel}, {str(early).lower()}>"])
            kid = append(e, parent, {"EarlyScaleLoadAfterC00": early, "Panel": panel, "Unroll": 2}, not early)
            if early:matches.append([base, kid])
            else:base = kid
    for parent in (47669, 47680):
        name, args = traits_args(metadata[parent]["traits"]);bm, fixed_k, gm, xcd, ca, cb, wn = args
        base = None
        for skip, b_first in ((False, False), (True, False), (False, True)):
            e = {"pipeline": "wave_interleave", "block_size": 2 * int(wn) * 64,
                "tile": [int(bm), 256, 256], "wave_layout": [2, int(wn)], "stages": 2,
                "launch_bounds": [2 * int(wn) * 64, 1], "device_flags": "flags_00",
                "pad_m": True, "wide_output": True, "max_k": 16384, "fixed_k": int(fixed_k),
                "traits": {"name": name, "args": list(args)}, "traits_header": TRAITS,
                "pipeline_header": INTER_HEADER if skip or b_first else "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh",
                "body_func": "opus_stage18_interleave_body" if skip or b_first else "opus_gemm_bpreshuffle_wave_interleave_body",
                "body_args": [f"opus_stage18_interleave_traits<CoTraits, {str(skip).lower()}, {str(b_first).lower()}>"] if skip or b_first else ["CoTraits"],
                "grid_layout": "n_m", "pin_variant": False, "pin_mask": 0}
            kid = append(e, parent, {"SkipUnconsumedARows": skip, "BFirst": b_first}, not (skip or b_first))
            if skip or b_first:matches.append([base, kid])
            else:base = kid
    for parent in (18094, 17914, 20706, 20569):
        name, args = traits_args(metadata[parent]["traits"])
        bm, bn, panel, fixed_k, wm = map(int, args[:5]);em = bm // (wm * 16);assert em >= 3
        base = None
        for ring in (3, 2):
            e = {"pipeline": "pingpong_lds", "block_size": wm * 2 * 64,
                "tile": [bm, bn, 128], "wave_layout": [wm, 2], "stages": 2,
                "launch_bounds": [wm * 2 * 64, 1], "device_flags": "flags_00",
                "pad_m": True, "wide_output": True, "max_k": panel * 128, "fixed_k": fixed_k,
                "traits": {"name": name, "args": list(args)}, "traits_header": TRAITS,
                "pipeline_header": "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_m_repeat_gfx950.cuh",
                "body_func": "opus_gemm_bpreshuffle_pingpong_lds_body",
                "body_args": [f"opus_stage18_m_repeat_traits<CoTraits, {ring}>"] if ring == 2 else ["CoTraits"],
                "grid_layout": "n_m", "pin_variant": False, "pin_mask": 0}
            kid = append(e, parent, {"A_RING": ring, "E_M": em}, ring == 3)
            if ring == 2:matches.append([base, kid])
            else:base = kid
    OUT.mkdir();experiment = OUT / "co_experiment";experiment.mkdir()
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage18_extended");after = registry(module)
    assert set(after) == set(before) | {p["new_kid"] for p in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    for p in points:
        instance = after[p["new_kid"]]
        shapes = [list(key(r)[2:5]) for r in rows if module.a8w8_mxscale_bpreshuffle_supports_shape(instance, *key(r)[2:5])]
        assert shapes;p["shapes"],p["kernelName"] = shapes,instance.name
        next(e for e in document["gfx950"] if e["kid"] == p["new_kid"])["supported_shapes"] = shapes
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    target = {key(r) for r in rows};existing = collections.defaultdict(list)
    for file, role in (("full421_tuned.csv", "remote_54bef351_best"),
        ("cpu_persistent171_plan/narrow_existing_candidates.csv", "remote_cpu_narrow_exact"),
        ("regressions14_pair_candidates.csv", "remote_regression14_exact")):
        for r in csv_rows(ALIGN / "remote_snapshot" / file):
            if key(r) not in target:continue
            kid = byname[r["kernelName"]]
            assert module.a8w8_mxscale_bpreshuffle_supports_shape(after[kid], *key(r)[2:5])
            existing[",".join(map(str,key(r)[2:5]))].append({"kid":kid,"kernelName":r["kernelName"],"role":role})
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    headers = {PIN_HEADER: pin, INTER_HEADER: interleave, TRAITS: traits}
    for name, content in headers.items():(repo / INCLUDE / name).write_text(content)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029";helpers.mkdir()
    for p in LOCAL.glob("*.py"):shutil.copy2(p, helpers / p.name)
    dump(experiment / "schedule_invariants.json", {"pin_MFMA_and_matrix_dataflow":pin_audit,
        "scale_prefetch_checks":[verify_scale_prefetch(p,k) for p in (32,64) for k in range(1,201)],
        "interleave_consumer_coverage":inter_audits,
        "ring_ownership":[verify_ring(em,r) for em in (3,4) for r in (2,3)],
        "scale_cache":[verify_scale_cache(p,k) for p in (32,64) for k in sorted({int(r['K'])//128 for r in rows})]})
    dump(experiment / "identity_and_points.json", {"previous_ids_verified":len(before),
        "previous_config_digest":digest.hexdigest(),"jit_kid_base":780,"repo":str(repo),
        "points":points,"reference":str(reference),"target_only_shapes":84,"remote_commit":"54bef35161139556602a39d0d2dcf367102193fc",
        "compile_family_only":SCOPE,"matched_controls":matches,"require_matched_control":True,
        "existing_candidates_by_shape":dict(existing),"new_source_changes":[str(INCLUDE/name) for name in headers],
        "new_source_sha256":{name:hashlib.sha256(content.encode()).hexdigest() for name,content in headers.items()},
        "preserved_leading_shapes":337,"final_optimization_round_requested_by_user":True,
        "A_input_2GiB_guard_unchanged":True,"output_64bit_base_and_tile_buffer_windows_preserved":True})
    (experiment / "experiment_env.sh").write_text("source "+shlex.quote(str(LOCAL/"env.sh"))
        +"\nexport OPUS_RUN_REPO="+shlex.quote(str(repo))+'\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        +"\nexport AITER_JIT_DIR="+shlex.quote(str(experiment/".cache/aiter_jit"))
        +"\nexport OPUS_COMPILE_FAMILY_ONLY="+SCOPE
        +'\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0 OPUS_TARGET_NEW_ONLY=0\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource "+shlex.quote(str(experiment/"experiment_env.sh"))
        +'\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        +" ".join(str(p["new_kid"]) for p in points)+' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        +shlex.quote(str(experiment/"build_info.json"))+"\n")
    print(json.dumps({"candidate_points":len(points),"new_versions":sum(not p['comparison_only'] for p in points),
        "matched_controls":len(matches),"target_shapes":84,"old_ids_verified":len(before),
        "remote_existing_shape_candidate_pairs":sum(len(v) for v in existing.values())}),flush=True)


if __name__ == "__main__":main()
