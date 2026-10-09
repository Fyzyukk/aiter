#!/usr/bin/env python3
"""Freeze current dependencies and create unregistered A-LDS/direct-B variants.

CPU source work only. Does not import torch/aiter, load HIP, or query a device.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPELINE = "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
FAMILIES = {
    9043: "small_lds_32x64",
    9044: "small_lds_64x64",
    9045: "small_lds_96x64",
    9046: "small_lds_64x128",
    9047: "small_regscale_32x64",
    9049: "small_regscale_xor_32x128",
    9055: "small_lds_deep_32x64",
    9056: "small_lds_deep_64x64",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for name in ["frozen", "candidate", "source_manifest.json"]:
        if (HERE / name).exists():
            raise SystemExit(f"Refusing to overwrite {name}")
    shutil.copytree(ROOT / "csrc/opus_gemm/include", HERE / "frozen/gemm_include")
    shutil.copytree(ROOT / "csrc/include/opus", HERE / "frozen/opus")
    (HERE / "candidate").mkdir()
    source = (HERE / "frozen/gemm_include" / PIPELINE).read_text()
    private = source.replace("gemm_a8w8_mxfp8_scale_small_lds_kernel", "opus_private_small_a_lds_direct_b_kernel")
    private = private.replace('#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"', '#include "traits.cuh"')
    private = private.replace("active_stages * (T::A_STAGE + T::B_STAGE)", "active_stages * T::A_STAGE")
    private = private.replace("    auto sb = make_smem(reinterpret_cast<fp8_t*>(lds + active_stages * T::A_STAGE));\n", "")
    start = private.index("    const auto ugb =")
    end = private.index("    auto lds_offset =", start)
    private = private[:start] + private[end:]
    private = private.replace("    array<array<unsigned, (T::E_M + 3) / 4>, T::NUM_STAGES> qsa;", "    // B has its own compile-time VGPR ring. It never enters matrix LDS.\n    array<typename decltype(mma)::vtype_b, T::B_SLOTS> qb;\n    array<array<unsigned, (T::E_M + 3) / 4>, T::NUM_STAGES> qsa;")
    private = private.replace("    auto prefetch = [&](auto slot, int kt) {", "    auto prefetch_a = [&](auto slot, int kt) {")
    start = private.index("        if constexpr (T::XOR_LDS) {\n            // Each B copy")
    end = private.index("    auto load_scales =", start)
    direct = '''    };
    auto prefetch_b = [&](auto slot, int kt) {
        auto& target = qb[decltype(slot)::value];
        // Standard N16/K16 preshuffle: one K128 tile occupies 2048 bytes
        // per N16 group. wm is deliberately absent: M waves duplicate B reads.
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const int nr = (n * T::T_N + wn) * 16;
            const int offset = nr * args.stride_b + lane * 16;
            set_slice(target, load<16>(gb, offset, kt * 2048),
                      number<n * 32>{}, number<n * 32 + 16>{});
            set_slice(target, load<16>(gb, offset + 1024, kt * 2048),
                      number<n * 32 + 16>{}, number<(n + 1) * 32>{});
        });
    };
'''
    private = private[:start] + direct + private[end:]
    start = private.index("    static_for<T::NUM_STAGES>([&](auto i) {\n        if (decltype(i)::value < initial_tiles)")
    end = private.index("        typename decltype(mma)::vtype_a a;", start)
    schedule = '''    // Fill each independent queue only with valid K128 tiles.
    static_for<T::NUM_STAGES>([&](auto i) {
        if (decltype(i)::value < loops) prefetch_a(i, decltype(i)::value);
    });
    static_for<T::B_SLOTS>([&](auto i) {
        if (decltype(i)::value < loops) prefetch_b(i, decltype(i)::value);
    });
    if constexpr (!T::REGISTER_SCALES && !T::EARLY_SCALE_LOADS) load_scales();
    auto step = [&](auto si, auto bi, int kt) {
        constexpr int stage = decltype(si)::value;
        constexpr int bstage = decltype(bi)::value;
        // Mixed VMEM-to-LDS A and VMEM-to-VGPR B queues use a conservative
        // completion fence. Later timing work may refine the wait count.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
'''
    private = private[:start] + schedule + private[end:]
    start = private.index("        typename decltype(mma)::vtype_b b;")
    end = private.index("        array<unsigned, (T::E_M + 3) / 4> sfa{};", start)
    private = private[:start] + "        typename decltype(mma)::vtype_b b = qb[bstage];\n" + private[end:]
    start = private.index("        if constexpr (decltype(ring)::value && !T::PREFETCH_BEFORE_READ)")
    end = private.index("        static_for<T::E_M>([&](auto mi) {", start)
    advance = '''        // Complete A/scale LDS reads in every wave before reusing an A slot.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        if (kt + T::NUM_STAGES < loops) prefetch_a(si, kt + T::NUM_STAGES);
        // Copying qb into b before overwriting this B slot preserves the
        // currently consumed fragment while future direct loads are in flight.
        if (kt + T::B_SLOTS < loops) prefetch_b(bi, kt + T::B_SLOTS);
'''
    private = private[:start] + advance + private[end:]
    start = private.index("    if (loops <= T::NUM_STAGES) {\n        // Short K:")
    end = private.index("    if constexpr (SplitK > 1) {", start)
    private = private[:start] + '''    // Unroll only one common ring period. Runtime K keeps code growth bounded.
    #pragma clang loop unroll(disable)
    for (int base = 0; base < loops; base += T::RING_PERIOD) {
        static_for<T::RING_PERIOD>([&](auto i) {
            constexpr int index = decltype(i)::value;
            if (base + index < loops)
                step(number<index % T::NUM_STAGES>{}, number<index % T::B_SLOTS>{}, base + index);
        });
    }
''' + private[end:]
    # The reducer remains byte-identical and available for future split variants.
    assert "sb.ptr" not in private and "load<16>(sb" not in private
    assert "decltype(ring)" not in private and "distance" not in private
    (HERE / "candidate/pipeline.cuh").write_text(private)
    (HERE / "candidate/pipeline.diff").write_text("".join(difflib.unified_diff(source.splitlines(True), private.splitlines(True), fromfile="frozen/gemm_include/" + PIPELINE, tofile="candidate/pipeline.cuh")))
    traits = '''#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"

constexpr int opus_private_small_gcd(int a, int b) { while (b) { int r=a%b; a=b; b=r; } return a; }
template<class Baseline, int Ahead>
struct opus_private_small_direct_b_traits : Baseline {
    static constexpr int B_AHEAD = Ahead, B_SLOTS = Ahead + 1;
    static constexpr int RING_PERIOD = Baseline::NUM_STAGES * B_SLOTS / opus_private_small_gcd(Baseline::NUM_STAGES, B_SLOTS);
    static constexpr int B_STAGE = 0;
    static constexpr int C_BYTES = Baseline::OUTPUT == 1 ? Baseline::B_M * (Baseline::B_N + 8) * 2 : 0;
    static constexpr int MAX_LDS_BYTES = Baseline::NUM_STAGES * Baseline::A_STAGE +
        (Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * Baseline::MAX_LOOPS);
    static_assert(Ahead >= 1 && Ahead <= 3 && Baseline::T_K == 1);
    static_assert(MAX_LDS_BYTES <= 160 * 1024 && C_BYTES <= 160 * 1024);
    static constexpr int lds_bytes(int k) {
        const int loops = (k / Baseline::B_K + Baseline::SPLIT_K - 1) / Baseline::SPLIT_K;
        const int stages = loops < Baseline::NUM_STAGES ? loops : Baseline::NUM_STAGES;
        const int matrix_scale = stages * Baseline::A_STAGE +
            (Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * loops);
        return matrix_scale > C_BYTES ? matrix_scale : C_BYTES;
    }
};
template<int Actual> struct opus_private_small_baseline;
'''
    for kid, name in FAMILIES.items():
        traits += f"template<> struct opus_private_small_baseline<{kid}> {{ using type = opus_gemm_mxscale_bpreshuffle_{name}_traits_gfx950; }};\n"
    traits += "template<int Actual, int Ahead> using opus_private_small_traits = opus_private_small_direct_b_traits<typename opus_private_small_baseline<Actual>::type, Ahead>;\n"
    (HERE / "candidate/traits.cuh").write_text(traits)
    frozen = []
    for original_dir, frozen_dir in [(ROOT / "csrc/opus_gemm/include", HERE / "frozen/gemm_include"), (ROOT / "csrc/include/opus", HERE / "frozen/opus")]:
        for original in sorted(original_dir.rglob("*")):
            if original.is_file():
                target = frozen_dir / original.relative_to(original_dir)
                assert sha(original) == sha(target)
                frozen.append(dict(original=str(original.relative_to(ROOT)), frozen=str(target.relative_to(HERE)), sha256=sha(original)))
    controls = [ROOT / "reports/opus_clang23_mixed_retune_20261008/jit/module_deepgemm_opus.so", ROOT / "reports/opus_flydsl_gap_20261009/losers294.csv", ROOT / "csrc/opus_gemm/opus_gemm_common.py", ROOT / "csrc/opus_gemm/codegen/gen_instances_gfx950.py"]
    manifest = dict(status="prepared_unvalidated_gpu_stopped", cpu_only=True, registered=False,
                    actual_ids=list(FAMILIES), b_ahead=[1,2,3], frozen_files=frozen,
                    original_pipeline_sha256=sha(HERE / "frozen/gemm_include" / PIPELINE), private_pipeline_sha256=sha(HERE / "candidate/pipeline.cuh"),
                    controls=[dict(path=str(p),sha256=sha(p),action="read-only hash; not loaded or executed") for p in controls])
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status=manifest["status"],actual_ids=list(FAMILIES),frozen_file_count=len(frozen)),indent=2))

if __name__ == "__main__": main()
