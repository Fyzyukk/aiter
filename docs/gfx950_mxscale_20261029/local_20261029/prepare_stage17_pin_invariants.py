#!/usr/bin/env python3
"""Let LLVM hoist invariant matrix offsets out of the MFMA main loop."""
import re

import prepare_stage16_pin_compute as campaign


def invariant_body():
    original = (campaign.ROOT / campaign.INCLUDE / "opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh").read_text()
    source = original
    matrix_anchor = '        asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));'
    future_anchor = '        asm volatile("" : "+s"(future_matrix_offset));'
    source = campaign.replace_once(source, matrix_anchor,
        "        // Matrix lane offsets are invariant across K tiles.\n"
        "        if constexpr (!T::STABLE_MATRIX_OFFSETS) {\n" + matrix_anchor + "\n        }")
    source = campaign.replace_once(source, future_anchor,
        "        // Expose the uniform K tile offset to scalar propagation.\n"
        "        if constexpr (!T::STABLE_FUTURE_OFFSET) {\n" + future_anchor + "\n        }")
    source = source.replace("opus_stage14_pin_detail", "opus_stage17_pin_detail").replace(
        "opus_stage14_pin_body", "opus_stage17_pin_body")
    waits = r"s_waitcnt_\w+\([^;]*?\)|opus::sync_threads\(\)|__builtin_amdgcn_sched_barrier\([^;]*?\)|__builtin_amdgcn_sched_group_barrier\([^;]*?\)"
    assert re.findall(waits, original) == re.findall(waits, source)
    begin = original.index("        // A half 0 x B half 0 -> C[0][0]", original.index("// ===== Main loop ====="))
    end = original.index("        v_sfb[1] = v_sfb_next[1];", begin)
    audit = campaign.verify_compute_order(original[begin:end])
    return source, {"MFMA_and_memory_dataflow_unchanged": audit,
        "all_hardware_and_compiler_memory_schedule_barriers_preserved": True,
        "only_empty_asm_invariance_hints_changed": ["matrix_voffsets[0..3]", "future_matrix_offset"]}


if __name__ == "__main__":
    campaign.OUT = campaign.LOCAL / "stage17_pin_invariants54_20261030"
    campaign.HEADER = "opus_gemm_pipeline_stage17_pin_invariants_gfx950.cuh"
    campaign.TRAITS = "opus_gemm_traits_stage17_pin_invariants_gfx950.cuh"
    campaign.compute_body = invariant_body
    traits = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage15_pin_unroll_gfx950.cuh"

template<class Base, bool MatrixInvariant, bool FutureScalar>
struct opus_stage17_pin_traits : Base {
    static constexpr bool STABLE_MATRIX_OFFSETS = MatrixInvariant;
    static constexpr bool STABLE_FUTURE_OFFSET = FutureScalar;
};
'''
    campaign.main(reference=campaign.LOCAL / "stage16_pin_compute54_20261030/result/public_best_opus421.csv",
        first_kid=101200, trait_type="opus_stage17_pin_traits", body_func="opus_stage17_pin_body",
        traits_source=traits, axis_names=("StableMatrixOffsets", "StableFutureOffset"), experiment_tag="stage17_invariants")
