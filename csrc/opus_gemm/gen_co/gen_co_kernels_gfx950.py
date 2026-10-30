#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate the compact unified gfx950 MXFP8 CO parameter set.

PIN_PARAMETER_GROUPS crosses pad/no-pad, scale panel, unroll, grouped M,
wide output and the required static/scale/compiler options within small
implementation groups. It yields 40 pin COs. B_PARAMETER_GROUPS crosses
fixed K and pin-A/pin-C per tile/addressing group; the four implementation
groups bring B-VGPR to 35 and overlap to 29. Interleave contributes one.

All 105 candidates are generated from parameters and numbered 0..104.
CO image names describe the generated parameters, independently of IDs.
The 69 result CO configurations retain flags and shape scopes.
JSON contains every generated candidate. build_co.py retains its normal
complete-build logic; image cleanup is a separate step after compilation.
JIT IDs remain based at 200 independently of the CO count.
"""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

LDS_LIMIT = 160 * 1024
WARP_SIZE = 64
MFMA_M = MFMA_N = 16
MFMA_K = 128
VECTOR_SIZE = 16
PIPELINE_PREFIX = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_"
TRAITS_PREFIX = "opus_gemm_blockscale_mxscale_bpreshuffle_"
DEVICE_FLAGS = [
    "-fgpu-flush-denormals-to-zero",
    "-fno-offload-uniform-block",
    "-fno-gpu-rdc",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-mllvm",
    "--amdgpu-kernarg-preload-count=32",
    "-mllvm",
    "--amdgpu-mfma-vgpr-form",
    "-mllvm",
    "--lsr-drop-solution=1",
    "-mllvm",
    "-amdgpu-early-inline-all=true",
    "-mllvm",
    "-amdgpu-function-calls=false",
    "-mllvm",
    "-enable-post-misched=0",
]


# Existing result scopes remain fixed independently of parameter generation.
FINAL_SHAPE_SCOPES = (((1088, 7168, 4096), (1152, 7168, 4096), (1216, 7168, 4096), (1280, 7168, 4096), (1344, 7168, 4096),
  (1408, 7168, 4096), (1472, 7168, 4096), (1536, 7168, 4096), (1600, 7168, 4096), (1664, 7168, 4096),
  (1728, 7168, 4096), (1792, 7168, 4096), (1856, 7168, 4096), (1920, 7168, 4096), (1984, 7168, 4096),
  (2048, 7168, 4096), (4096, 7168, 4096), (6144, 7168, 4096), (8192, 7168, 4096), (10240, 7168, 4096),
  (12288, 7168, 4096), (14336, 7168, 4096), (16384, 7168, 4096), (20480, 7168, 4096), (24576, 7168, 4096),
  (28672, 7168, 4096), (40960, 7168, 4096), (49152, 7168, 4096), (57344, 7168, 4096), (65536, 7168, 4096)),
 ((1088, 768, 7168), (1088, 1536, 7168), (1088, 2048, 7168), (1152, 768, 7168), (1152, 1536, 7168),
  (1152, 2048, 7168), (1216, 768, 7168), (1216, 1536, 7168), (1216, 2048, 7168), (1280, 768, 7168),
  (1280, 1536, 7168), (1280, 2048, 7168), (1344, 768, 7168), (1344, 1536, 7168), (1344, 2048, 7168),
  (1408, 768, 7168), (1408, 1536, 7168), (1408, 2048, 7168), (1472, 768, 7168), (1472, 1536, 7168),
  (1472, 2048, 7168), (1536, 768, 7168), (1536, 1536, 7168), (1536, 2048, 7168), (1600, 768, 7168),
  (1600, 1536, 7168), (1600, 2048, 7168), (1664, 768, 7168), (1664, 1536, 7168), (1664, 2048, 7168),
  (1728, 768, 7168), (1728, 1536, 7168), (1728, 2048, 7168), (1792, 768, 7168), (1792, 1536, 7168),
  (1792, 2048, 7168), (1856, 768, 7168), (1856, 1536, 7168), (1856, 2048, 7168), (1920, 768, 7168),
  (1920, 1536, 7168), (1920, 2048, 7168), (1984, 768, 7168), (1984, 1536, 7168), (1984, 2048, 7168),
  (2048, 768, 7168), (2048, 1536, 7168), (2048, 2048, 7168), (4096, 768, 7168), (4096, 1536, 7168),
  (4096, 2048, 7168), (6144, 768, 7168), (6144, 1536, 7168), (6144, 2048, 7168), (8192, 768, 7168),
  (8192, 1536, 7168), (8192, 2048, 7168), (10240, 768, 7168), (10240, 1536, 7168), (10240, 2048, 7168),
  (12288, 768, 7168), (12288, 1536, 7168), (12288, 2048, 7168), (14336, 768, 7168), (14336, 1536, 7168),
  (14336, 2048, 7168), (16384, 768, 7168), (16384, 1536, 7168), (16384, 2048, 7168), (20480, 1536, 7168),
  (24576, 1536, 7168), (28672, 1536, 7168), (32768, 768, 7168), (32768, 1536, 7168), (32768, 2048, 7168),
  (40960, 1536, 7168), (49152, 1536, 7168), (57344, 1536, 7168), (65536, 1536, 7168)),
 ((1280, 6144, 7168), (1280, 7168, 7168), (1280, 7168, 16384), (1536, 6144, 7168), (1536, 7168, 7168),
  (1536, 7168, 16384), (1792, 6144, 7168), (1792, 7168, 7168), (1792, 7168, 16384), (2048, 6144, 7168),
  (2048, 7168, 7168), (2048, 7168, 16384), (4096, 6144, 7168), (4096, 7168, 7168), (4096, 7168, 16384),
  (6144, 7168, 7168), (6144, 7168, 16384), (8192, 6144, 7168), (8192, 7168, 7168), (8192, 7168, 16384),
  (10240, 6144, 7168), (10240, 7168, 7168), (10240, 7168, 16384), (12288, 6144, 7168), (12288, 7168, 7168),
  (12288, 7168, 16384), (14336, 6144, 7168), (14336, 7168, 7168), (14336, 7168, 16384), (16384, 6144, 7168),
  (16384, 7168, 7168), (16384, 7168, 16384), (20480, 6144, 7168), (20480, 7168, 7168), (20480, 7168, 16384),
  (24576, 6144, 7168), (24576, 7168, 7168), (24576, 7168, 16384), (28672, 6144, 7168), (28672, 7168, 7168),
  (28672, 7168, 16384), (32768, 6144, 7168), (32768, 7168, 7168), (40960, 6144, 7168), (40960, 7168, 7168),
  (40960, 7168, 16384), (49152, 6144, 7168), (49152, 7168, 7168), (49152, 7168, 16384), (57344, 6144, 7168),
  (57344, 7168, 7168), (57344, 7168, 16384), (65536, 6144, 7168), (65536, 7168, 7168), (65536, 7168, 16384)),
 ((3072, 7168, 21504), (10240, 1536, 7168), (28672, 1536, 7168), (65536, 1536, 7168)),
 ((14336, 2048, 7168), (16384, 2048, 7168), (40960, 1536, 7168), (49152, 1536, 7168)), ((1984, 7168, 16384),),
 ((8192, 7168, 2048), (8192, 7168, 3072), (16384, 7168, 4096), (28672, 7168, 3072), (32768, 7168, 4096)),
 ((6144, 7168, 3072), (8192, 7168, 4096), (10240, 7168, 3072), (32768, 7168, 16384), (57344, 1536, 7168)),
 ((20480, 1536, 7168), (32768, 1536, 7168)), ((10240, 1536, 7168),), ((14336, 2048, 7168),),
 ((1792, 7168, 3072), (2048, 7168, 2048)),
 ((6144, 7168, 2048), (6144, 7168, 3072), (6144, 7168, 4096), (8192, 7168, 4096), (10240, 7168, 3072),
  (10240, 7168, 4096), (16384, 7168, 3072), (20480, 7168, 4096), (24576, 1536, 7168), (24576, 7168, 4096),
  (28672, 7168, 4096), (32768, 7168, 3072), (32768, 7168, 4096), (40960, 7168, 4096), (49152, 7168, 4096),
  (57344, 1536, 7168), (57344, 7168, 4096), (65536, 7168, 4096)),
 ((1920, 7168, 16384),), ((12288, 768, 7168),), ((2048, 1536, 7168),), ((1536, 768, 7168),),
 ((20480, 7168, 3072),), ((16384, 6144, 7168), (65536, 7168, 3072)),
 ((32768, 6144, 7168), (40960, 7168, 3072)), ((32768, 6144, 7168),), ((16384, 7168, 4096),),
 ((10240, 7168, 4096), (40960, 6144, 7168), (57344, 6144, 7168)), ((12288, 7168, 4096),),
 ((1920, 7168, 3072),), ((1280, 7168, 16384), (1344, 7168, 16384)))

# Shape restrictions for specialized implementations are independent of IDs.
# The hash covers build/launch settings and selects only the applicable scope;
# every candidate itself is produced by the parameter groups below.
CO_SHAPE_SCOPES = {
    '80fd0f8bc24893e62611747fd731d7d12b4fb152ffdc704b209719b031511b2c': 0,
    'fc828f37f5ef0bbcef2c352cd3d27d560179d3fc940b649cdca3b215227970f9': 1,
    '38836e0b20b8f23113fcf0cf387c7160a41b6f6cb31d9b17a812985b4f1edccf': 2,
    'bbc7e60f830095935cbb7159954b07e98ec16f53ccec528912fa93b896f2f366': 2,
    '2e0280e5b12a293370817be90fa2820f0f5f7896852fe2c266227e8cbe2a70bf': 3,
    '5b6a89dcecce3d7ea90e84f2e09deca1a3f41a671c48633fbfc8a6d1678bbf83': 4,
    '2aa4459884833cbd1ec88218b2a5c7b8324d309e20f306cb228cdeb30613b6fc': 5,
    '34caecd2f810a5657db89640e674d288541edfccd26570336d5bd8dfac259620': 6,
    '79cd545bf3bb77dcb7e25f05f4568c8d4135133d3e324957ef5710986149715d': 7,
    'aa85c2bbc9482bacb5ed0b6344306d31e221a80cc6853ea44035d8de8834102f': 8,
    'b5388dfb6a8b8341a25ba2194f92f71c235917776288238086654e71c12d8f7c': 9,
    '72f478386d0d62c845540592b85f637c15b9bb50f3e07aaa4bd2fc707ec3c0d1': 10,
    '7fed626336dc05fc1cebce150754e5ae280c5b33658d4fff55bcbf8e2117530c': 11,
    '1b9a44315f40d236ecca19083043c053f598dcad306394e68d76d4d50909703f': 12,
    'e52163c22fa37715597f94be6dc06fe3c650feeab5f7c33d668afeb799f45153': 13,
    '3894df8627a4000819578e7c77de82851c8787a3ba4ef2e74c64de554b604078': 14,
    'e7bb2030ff068d78050258b13c7129ad0a0aaa709cc5f42f30e0806c51e164e2': 15,
    '12e1e72be9ab55e1bf1c5265f29cf168b105357c739cddfe84f198f1255f6eed': 16,
    'f4dcd3c961db07a22429a402d694ef68ba2256bbe84b0be2370906e677d15944': 17,
    '62d16f26a0865ca15a026ac44e127d915efb96ce0baecca01a2a99adc3bddda7': 18,
    'e9c87b5ad3f6c4682e675f3553af2b44b61bf721245635153d58cfa3d5db9c03': 19,
    '424581897df219569c7e4022fdfbe8a56b87add4d61edc8b1b12aa32b6ad82c4': 20,
    '85de9233e4a74cb084f32320635677e5ce2a1820cb85509d0d47b8af19b4e600': 21,
    '96c4a1b8daae1428c732996bec1af406fe6c8af04b2b06d5b08d659c5d8e3218': 22,
    'd3110a8f51d25d8987784681c6cc728833e492f699e51ed059538bab54ee2934': 23,
    '1f3e14ef21ddabc20d8adfd15dd7b4e96a2494bff935cb69bfc69d840969ff65': 24,
    '43b7ec81a8d533a00fc81a75d1cedfe423c35cc60d9eee497c6626044914d902': 25,
}



def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def cpp_values(values):
    return [str(value).lower() for value in values]


def record(
    pipeline,
    tile,
    waves,
    stages,
    traits_name,
    traits_args,
    body_func,
    header,
    body_args,
    max_k,
    fixed_k=0,
    pin_mask=0,
    grid_layout="n_m",
    pad_m=True,
    wide_output=True,
):
    if not pin_mask:
        raise ValueError("CO candidates must declare their register pinning")
    threads = waves[0] * waves[1] * WARP_SIZE
    return {
        "pipeline": pipeline,
        "block_size": threads,
        "tile": list(tile),
        "stages": stages,
        "wave_layout": list(waves),
        "launch_bounds": [threads, 1],
        "device_flags": "flags_00",
        "pad_m": pad_m,
        "wide_output": wide_output,
        "max_k": max_k,
        "fixed_k": fixed_k,
        "traits": {"name": traits_name, "args": cpp_values(traits_args)},
        "body_func": body_func,
        "body_args": list(body_args),
        "pipeline_header": PIPELINE_PREFIX + header + "_gfx950.cuh",
        "grid_layout": grid_layout,
        "pin_variant": pipeline == "pin_agpr" or bool(pin_mask),
        "pin_mask": pin_mask,
    }


B_BLOCK_K = (128, 256)
B_WGS_PER_CU = (1, 2)

# One group crosses only the knobs needed by its tile/implementation family.
PIN_PARAMETER_GROUPS = (
    ('base', dict(pad_m=(True, False), panel=(16, 32, 64), wide_output=(True, False))),
    ('grouped_no_pad', dict(pad_m=(False,), panel=(64,), unroll=(2, 4), group_m=(4, 16), wide_output=(True, False))),
    ('grouped_pad', dict(pad_m=(True,), panel=(32, 64), unroll=(2, 4), group_m=(4,), wide_output=(False,))),
    ('static_grouped', dict(pad_m=(False,), panel=(64,), unroll=(4,), group_m=(4,), wide_output=(True, False), static_n=(1536, 2048))),
    ('static_7168', dict(pad_m=(False,), panel=(64,), wide_output=(False,), static_n=(7168,))),
    ('agpr_form', dict(pad_m=(False,), panel=(64,), wide_output=(True,), group_m=(16,), flags=('flags_stage6_agpr_form',))),
    ('scheduled', dict(pad_m=(True,), panel=(64,), wide_output=(False,), implementation=('scheduled',), policy=(2,))),
    ('unroll8', dict(pad_m=(False,), panel=(64,), unroll=(8,), wide_output=(True,), group_m=(16,), implementation=('unroll',), flags=('flags_00', 'flags_stage6_agpr_form'))),
    ('local_scale', dict(pad_m=(False,), panel=(32,), unroll=(8,), wide_output=(True,), group_m=(16,), implementation=('local_scale',), flags=('flags_stage6_agpr_form',))),
    ('chunk_scale', dict(pad_m=(False,), panel=(32, 64), unroll=(8,), wide_output=(True,), group_m=(16,), implementation=('chunk_scale',), flags=('flags_00', 'flags_stage6_agpr_form'))),
    ('pin_panel', dict(pad_m=(True,), panel=(64,), unroll=(2,), wide_output=(False,), implementation=('pin_panel',))),
    ('chunk_pin_panel', dict(pad_m=(False,), panel=(32,), unroll=(2,), wide_output=(True,), group_m=(16,), implementation=('chunk_pin_panel',), flags=('flags_stage6_agpr_form',))),
)

# (overlap, geometry M/N/wave-N, fixed Ks, pin-A/pin-C bundles, XCD/contiguous/BD)
B_PARAMETER_GROUPS = (
    (False, (128, 128, 4), (384,), ((True, False),), (True, True, 2)),
    (False, (64, 128, 4), (384,), ((True, False),), (False, False, 1)),
    (False, (96, 256, 4), (384, 768, 1024, 1536, 7168), ((True, False), (False, True)), (False, False, 1)),
    (False, (128, 256, 4), (768, 2048, 3072, 7168, 16384), ((True, False), (False, True), (True, True)), (False, False, 1)),
    (False, (96, 128, 4), (1536, 4096, 7168), ((True, False), (True, True)), (False, False, 1)),
    (False, (128, 128, 4), (7168,), ((True, False),), (False, False, 1)),
    (True, (96, 256, 4), (7168,), ((False, True),), (True, True, 2)),
    (True, (128, 128, 4), (384,), ((True, False),), (True, True, 2)),
    (True, (64, 128, 4), (384,), ((True, False),), (False, False, 1)),
    (True, (96, 256, 4), (384, 768, 1536, 2048), ((True, False), (False, True)), (False, False, 1)),
    (True, (128, 256, 4), (2048, 3072, 4096, 7168, 16384), ((True, False), (False, True), (True, True)), (False, False, 1)),
)

B_IMPLEMENTATION_GROUPS = (
    ('overlap_three_stage', dict(bm=(128,), bn=(128,), wn=(4,), fixed_k=(7168,), stages=(3,), bd=(2,), pin_a=(True,), pin_c=(False,), overlap=(True,))),
    ('stream_a', dict(bm=(96,), bn=(128,), wn=(4,), fixed_k=(7168,), bk=(256,), stages=(2,), bd=(1,), pin_a=(False,), pin_c=(False,), xcd=(True,), contiguous=(False,), cache_a=(17,), cache_b=(1,), implementation=('stream_ring',), stream_a=(True,))),
    ('static_b_ring', dict(bm=(64,), bn=(128,), wn=(2,), fixed_k=(7168,), bk=(256,), stages=(4,), bd=(2,), pin_a=(False,), pin_c=(False,), xcd=(True,), contiguous=(False,), wg=(2,), cache_a=(1,), cache_b=(17,), overlap=(True,), publication=(1,), implementation=('stream_ring',), static_ring=(True,))),
    ('compact_queue', dict(bm=(64,), bn=(64,), wn=(2,), fixed_k=(7168,), bk=(256,), stages=(4,), bd=(2,), pin_a=(False,), pin_c=(False,), xcd=(True,), contiguous=(False,), cache_a=(17,), cache_b=(1,), overlap=(True,), publication=(2,), implementation=('compact',), stream_a=(True,), static_ring=(True,), queue_ahead=(7,))),
)


def cpp_template(name, *args):
    values = (str(arg).lower() if isinstance(arg, bool) else str(arg) for arg in args)
    return name + "<" + ", ".join(values) + ">"


def pin_record(pad_m, panel, unroll=2, *, wide_output=False, group_m=0,
               implementation="base", static_n=0, policy=0,
               flags="flags_00"):
    """Compose one pin implementation entirely from named parameters."""
    base_panel = panel if implementation in ("base", "scheduled", "unroll") else 64
    base_unroll = unroll if implementation in ("base", "scheduled") else 2
    wrapper_name = TRAITS_PREFIX + "4wave_pin_agpr_traits"
    wrapper = cpp_template(wrapper_name, "CoTraits")
    if wide_output or group_m:
        wrapper = cpp_template(
            wrapper_name, "CoTraits",
            TRAITS_PREFIX + "pin_matrix_view::tile",
            TRAITS_PREFIX + "pin_output::lds", False, wide_output, group_m,
        )
    body = TRAITS_PREFIX + "4wave_pin_agpr"
    header = PIPELINE_PREFIX + "4wave_pin_agpr_gfx950.cuh"
    traits_header = None
    pin_mask = 3
    if implementation == "scheduled":
        wrapper = cpp_template("opus_gemm_bpreshuffle_pin_schedule_traits", wrapper, base_unroll, policy)
        body = "opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr"
        header = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
        pin_mask = 0
    elif implementation in ("unroll", "local_scale", "chunk_scale", "chunk_pin_panel"):
        wrapper = cpp_template("opus_gemm_bpreshuffle_pin_schedule_traits", wrapper, 8, policy)
        body = "opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr"
        header = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
    if implementation == "local_scale":
        wrapper = cpp_template("opus_gemm_bpreshuffle_pin_scale_traits", wrapper, panel, 1)
        body = "opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr"
        header = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
    elif implementation in ("chunk_scale", "chunk_pin_panel"):
        chunk_panel = panel if implementation == "chunk_scale" else 64
        wrapper = cpp_template("opus_gemm_bpreshuffle_pin_scale_traits", wrapper, chunk_panel, 2)
        body = "opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr"
        header = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
    if implementation in ("pin_panel", "chunk_pin_panel"):
        wrapper = cpp_template("opus_gemm_bpreshuffle_pin_scale_traits", wrapper, panel, 2, unroll)
        body = "opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr"
        header = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
    if static_n:
        wrapper = cpp_template("opus_gemm_bpreshuffle_static_shape_traits", wrapper, 0, static_n)
        if traits_header is None:
            traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
    candidate = record(
        "pin_agpr", (256, 256, 128), (2, 2), 2,
        TRAITS_PREFIX + "4wave_pin_agpr_panel_traits",
        (base_panel, base_unroll, pad_m), body, "4wave_pin_agpr", (wrapper,),
        16384 if wide_output else None, pin_mask=3, pad_m=pad_m,
        wide_output=wide_output,
    )
    candidate["pipeline_header"] = header
    if traits_header is not None:
        candidate["traits_header"] = traits_header
    candidate["device_flags"] = flags
    candidate["pin_mask"] = pin_mask
    candidate["m_align"] = 16 if pad_m else 256
    candidate["partial_pin_c"] = {"base": 148, "dwords": 64}
    if static_n:
        candidate.update(static_m=0, static_n=static_n)
    return candidate


def parameter_grid(parameters):
    """Cross the independent knobs within one tile/implementation group."""
    for values in itertools.product(*parameters.values()):
        yield dict(zip(parameters, values))


def pin_parameter_candidates():
    for group, parameters in PIN_PARAMETER_GROUPS:
        for args in parameter_grid(parameters):
            yield pin_record(**args), group, pin_image_suffix(args)


def pin_image_suffix(args):
    """Fixed order, including defaults: pad M, panel, unroll, GroupM,
    wide stores, static N, scale-load mode, scheduling policy, MFMA VGPR form.
    sl=0/1/2 selects persistent/local/chunked scales; vf=0 uses AGPR form.
    """
    mode = args.get("implementation", "base")
    scale_load = 1 if mode == "local_scale" else int(
        mode in ("chunk_scale", "pin_panel", "chunk_pin_panel")) * 2
    parts = [f"pm{int(args['pad_m'])}", f"sp{args['panel']}",
             f"u{args.get('unroll', 2)}", f"gm{args.get('group_m', 0)}",
             f"wide{int(args['wide_output'])}", f"sn{args.get('static_n', 0)}",
             f"sl{scale_load}", f"sched{args.get('policy', 0)}",
             f"vf{int(args.get('flags', 'flags_00') == 'flags_00')}"]
    return "_".join(parts)


def b_vgpr_lds(bm, bn, wn, bk, nb, fixed_k, pin_a, pin_c, *, wg=1, bd=1):
    """Mirror the B traits' compile-time constraints before CO emission."""
    threads = wn * WARP_SIZE
    if bm <= 0 or bm % MFMA_M or wn not in (1, 2, 4, 8):
        return None
    mr, nr = bm // MFMA_M, bn // (wn * MFMA_N)
    if bn <= 0 or bn > 256 or bn % (wn * MFMA_N) or mr * nr * 4 > 256:
        return None
    if bk not in B_BLOCK_K or nb < 2 or not 1 <= bd < nb or wg not in B_WGS_PER_CU:
        return None
    if bm * bk % (threads * VECTOR_SIZE) or fixed_k <= 0 or fixed_k % bk:
        return None
    if fixed_k // bk < nb - 1:
        return None
    if pin_a and bk // MFMA_K * mr * 8 > 128:
        return None
    if pin_c and mr * nr * 4 > 128:
        return None
    scales = fixed_k // MFMA_K
    sa_bytes = ceil_div(bm * scales, threads * VECTOR_SIZE) * threads * VECTOR_SIZE
    if ceil_div(mr, 4) * 64 * scales > sa_bytes:
        return None
    b_scale_rows = ceil_div(bn, 128)
    b_scale_slots = scales if scales % 4 else scales // 4
    if b_scale_rows * b_scale_slots > threads:
        return None
    lds = nb * bm * bk + sa_bytes + threads * 4
    if ceil_div(lds, 1280) * 1280 * wg > LDS_LIMIT:
        return None
    return lds


def b_record(bm, bn, wn, fixed_k, *, bk=128, stages=4, bd=2,
             pin_a=False, pin_c=True, xcd=True, contiguous=True,
             wg=1, cache_a=0, cache_b=0, overlap=False, publication=0,
             implementation="base", stream_a=False, static_ring=False,
             queue_ahead=0):
    """Shared parameter composer for normal, overlap, and optimized B bodies."""
    args = (bm, bn, stages, bd, cache_b, xcd, fixed_k, wn, wg, cache_a,
            2, 0, bk, contiguous, pin_a, pin_c, 0)
    traits_name = TRAITS_PREFIX + "prefetch_b_vgpr_traits"
    body = "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body"
    body_args = ("CoTraits",)
    traits_header = None
    header = PIPELINE_PREFIX + "b_vgpr_gfx950.cuh"
    if overlap:
        args += (publication,)
        traits_name = TRAITS_PREFIX + "prefetch_b_vgpr_overlap_traits"
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"
        if not xcd and not contiguous and bd == 1:
            body_args = ("CoTraits", "CoTraits::PUBLISH_M")
        else:
            body = "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_overlap_body"
    if implementation != "base":
        wrapper = "CoTraits"
        if queue_ahead:
            wrapper = cpp_template("opus_gemm_bpreshuffle_b_queue_traits", wrapper, queue_ahead)
        wrapper = cpp_template("opus_gemm_bpreshuffle_b_pipeline_traits", wrapper, stream_a,
                               static_ring, 0 if implementation == "compact" else -1,
                               implementation == "compact")
        body_args = (wrapper,)
        traits_header = "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"
        family = "opus_gemm_prefetch_b_vgpr_detail" if implementation == "compact" else "opus_gemm_prefetch_b_vgpr_detail"
        body = family + ("::bulk_panel_overlap_body" if overlap else "::bulk_panel_body")
        header = ("opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh" if implementation == "compact"
                  else "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh")
    mask = int(pin_a) + 4 * int(pin_c)
    candidate = record(
        "prefetch_b_vgpr_overlap" if overlap else "prefetch_b_vgpr",
        (bm, bn, bk), (1, wn), stages, traits_name, args, body, "b_vgpr",
        body_args, fixed_k, fixed_k=fixed_k, pin_mask=mask or 1,
        grid_layout="xcd" if xcd else "linear",
    )
    candidate["pipeline_header"] = header
    candidate["pin_mask"] = mask
    candidate["pin_variant"] = bool(mask) or implementation != "base"
    candidate["launch_bounds"] = [candidate["block_size"], wg]
    if traits_header is not None:
        candidate["traits_header"] = traits_header
    return candidate


def b_parameter_candidates(overlap=False):
    for is_overlap, (bm, bn, wn), ks, pins, (xcd, contiguous, bd) in B_PARAMETER_GROUPS:
        if is_overlap != overlap:
            continue
        for k, (pin_a, pin_c) in itertools.product(ks, pins):
            if b_vgpr_lds(bm, bn, wn, 128, 4, k, pin_a, pin_c, bd=bd) is None:
                continue
            args = dict(bm=bm, bn=bn, wn=wn, fixed_k=k, bd=bd,
                        pin_a=pin_a, pin_c=pin_c, xcd=xcd,
                        contiguous=contiguous, overlap=overlap)
            yield b_record(**args), f"tile{bm}x{bn}_w{wn}_bd{bd}", b_image_suffix(args)


def b_image_suffix(args):
    """Fixed order for both B families, including disabled/default settings.
    p=stages, bd=B prefetch distance, loops=FixedK/BK, pinA/C=AGPR pinning,
    xcd/contig=tile addressing, ca/cb=cache controls, wg=workgroups per CU,
    pub=M publication index, sa/sb=stream A/static B-ring traits,
    compact=compact B ring, ahead=queue override (auto uses the base traits),
    out=output mode (0 uses LDS stripes). Ordinary B kernels publish at 0.
    """
    parts = [f"p{args.get('stages', 4)}", f"bd{args.get('bd', 2)}",
             f"loops{args['fixed_k'] // args.get('bk', 128)}",
             f"pinA{int(args.get('pin_a', False))}C{int(args.get('pin_c', True))}",
             f"xcd{int(args.get('xcd', True))}",
             f"contig{int(args.get('contiguous', True))}",
             f"ca{args.get('cache_a', 0)}cb{args.get('cache_b', 0)}",
             f"wg{args.get('wg', 1)}", f"pub{args.get('publication', 0)}",
             f"sa{int(args.get('stream_a', False))}",
             f"sb{int(args.get('static_ring', False))}",
             f"compact{int(args.get('implementation') == 'compact')}",
             f"ahead{args.get('queue_ahead') or 'auto'}", "out0"]
    return "_".join(parts)


def optimized_b_candidates():
    for group, parameters in B_IMPLEMENTATION_GROUPS:
        for args in parameter_grid(parameters):
            bm, bn, wn, k = (args[key] for key in ("bm", "bn", "wn", "fixed_k"))
            bk, stages, bd, wg = (args.get(key, default) for key, default in
                                 (("bk", 128), ("stages", 4), ("bd", 2), ("wg", 1)))
            if b_vgpr_lds(bm, bn, wn, bk, stages, k, args["pin_a"],
                          args["pin_c"], wg=wg, bd=bd) is None:
                continue
            queue = args.get("queue_ahead", 0)
            if queue:
                mr, nr = bm // MFMA_M, bn // (wn * MFMA_N)
                room = (256 if wg == 2 else 512) - mr * nr * 4 - bk // MFMA_K * mr * 8 - 48
                if queue > max(room // 8 - 1, 1) or queue >= k // bk * (bk // MFMA_K) * nr:
                    continue
            yield b_record(**args), group, b_image_suffix(args)


def interleave_candidates():
    for bm, bn, bk, waves, group_m, cache_a, cache_b, wave_n in (
        (192, 256, 256, (2, 4), 8, 1, 17, 4),
    ):
        candidate = record(
            "wave_interleave", (bm, bn, bk), waves, 2,
            "opus_gemm_bpreshuffle_wave_interleave_traits",
            (bm, 0, group_m, True, cache_a, cache_b, wave_n),
            "opus_gemm_bpreshuffle_wave_interleave_body", "interleave",
            (cpp_template("opus_gemm_bpreshuffle_interleave_load_traits", "CoTraits", True, False),),
            16384, pin_mask=1,
        )
        candidate.update(pin_mask=0, pin_variant=False,
                         pipeline_header="opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh",
                         traits_header="opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh")
        # loops=FixedK/BK (auto means runtime K), gm=GroupM, xcd=tile order,
        # ca/cb=cache, skipA=skip unused A rows, bfirst=B-before-A issue order.
        suffix = (f"loopsauto_gm{group_m}_xcd1_ca{cache_a}cb{cache_b}"
                  "_skipA1_bfirst0")
        yield candidate, "interleave", suffix


def configuration_hash(entry):
    settings = {key: value for key, value in entry.items()
                if key not in ("kid", "configKey", "supported_shapes")}
    return hashlib.sha256(json.dumps(settings, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def generate_document(shapes=None):
    """Emit all compact parameter groups with consecutive CO IDs from zero."""
    if shapes is None:
        input_path = Path(__file__).resolve().parents[3] / "docs/gfx950_mxscale_20261029/shapes_421.csv"
        with input_path.open(newline="") as stream:
            shapes = [tuple(int(row[key]) for key in ("M", "N", "K"))
                      for row in csv.DictReader(stream)]
    optimized_b = tuple(optimized_b_candidates())
    candidates = itertools.chain(
        pin_parameter_candidates(),
        b_parameter_candidates(),
        (item for item in optimized_b if item[0]["pipeline"] == "prefetch_b_vgpr"),
        b_parameter_candidates(overlap=True),
        (item for item in optimized_b if item[0]["pipeline"] == "prefetch_b_vgpr_overlap"),
        interleave_candidates(),
    )
    records, suffixes, seen, counts = [], {}, set(), Counter()
    for candidate, group, suffix in candidates:
        digest = configuration_hash(candidate)
        if digest in seen:
            continue
        seen.add(digest)
        counts[(candidate["pipeline"], group)] += 1
        kid = len(records)
        candidate["kid"] = kid
        candidate["configKey"] = f"bpreshuffle_co_{candidate['pipeline']}_{kid}"
        if digest in CO_SHAPE_SCOPES:
            candidate["supported_shapes"] = [list(shape) for shape in
                FINAL_SHAPE_SCOPES[CO_SHAPE_SCOPES[digest]]]
        elif candidate.get("static_n"):
            candidate["supported_shapes"] = [list(shape) for shape in shapes
                if shape[1] == candidate["static_n"]
                and shape[0] % candidate["m_align"] == 0
                and (candidate["max_k"] is None or shape[2] <= candidate["max_k"])]
        records.append(candidate)
        suffixes[str(kid)] = suffix
    option = DEVICE_FLAGS.index("--amdgpu-mfma-vgpr-form")
    return {
        "_generation": {
            "version": 5,
            "parameter_counts": {f"{pipeline}/{group}": count
                                 for (pipeline, group), count in sorted(counts.items())},
        },
        "defaults": {
            "tag": "a8w8_blockscale_mxscale_gemm_bpreshuffle",
            "dtypes": {"a": "fp8_t", "b": "fp8_t", "c": "bf16_t", "acc": "fp32_t"},
        },
        "device_flag_sets": {
            "flags_00": list(DEVICE_FLAGS),
            "flags_stage6_agpr_form": DEVICE_FLAGS[:option - 1] + DEVICE_FLAGS[option + 1:],
        },
        "jit_kid_base": 200,
        "image_suffixes": suffixes,
        "gfx950": records,
    }


def write_document(document, path):
    path.write_text(json.dumps(document, indent=2) + "\n")


def validate_document(document, shapes):
    records = [{**document["defaults"], **entry} for entry in document["gfx950"]]
    kids = {record["kid"] for record in records}
    if len(kids) != len(records):
        raise ValueError("Duplicate local CO kid")
    if any(type(kid) is not int or kid < 0 for kid in kids):
        raise ValueError("Local CO kids must be nonnegative integers")
    if kids != set(range(len(records))) or len(records) > 200:
        raise ValueError("CO IDs must be consecutive from zero and below the JIT base")
    if document.get("jit_kid_base") != 200:
        raise ValueError("JIT IDs must remain based at 200")
    if set(document["image_suffixes"]) != {str(kid) for kid in kids}:
        raise ValueError("Image suffixes must exactly match declared CO IDs")
    symbols = set()
    for candidate in records:
        geometry = (tuple(candidate["tile"]), tuple(candidate["wave_layout"]))
        symbol = (
            candidate["pipeline"],
            geometry,
            document["image_suffixes"][str(candidate["kid"])],
        )
        if symbol in symbols:
            raise ValueError("Two generated COs share an image/symbol")
        symbols.add(symbol)
        if candidate["launch_bounds"][0] != candidate["block_size"]:
            raise ValueError("Inconsistent block size and launch bounds")
        if candidate["pipeline"] == "pin_agpr":
            if candidate["pin_mask"] not in (0, 3) or candidate.get("partial_pin_c") != {"base": 148, "dwords": 64}:
                raise ValueError("4wave pin metadata must describe A/B and the C01 AGPR range")
            pad_m = candidate["pad_m"]
            if candidate["m_align"] != (16 if pad_m else candidate["tile"][0]):
                raise ValueError("Pin M alignment differs from its PAD_M path")
            if candidate["traits"]["args"][-1] != str(pad_m).lower():
                raise ValueError("Pin PAD_M differs between metadata and traits")
        if candidate["pipeline"] == "prefetch_b_vgpr_overlap":
            publish = int(candidate["traits"]["args"][-1])
            if not 0 <= publish < candidate["tile"][0] // MFMA_M - 1:
                raise ValueError("Overlap publication must precede the final M repeat")
    uncovered = []
    for m, n, k in shapes:
        if not any(
            m % candidate.get("m_align", MFMA_M) == 0
            and (candidate["pad_m"] or m % candidate["tile"][0] == 0)
            and n % candidate["tile"][1] == 0
            and k % candidate["tile"][2] == 0
            and (candidate["max_k"] is None or k <= candidate["max_k"])
            and (not candidate["fixed_k"] or candidate["fixed_k"] == k)
            and m * k <= 2147483647
            and n * k <= 2147483647
            and (candidate["wide_output"] or 2 * m * n <= 2147483647)
            and 2 * ((min(m, candidate["tile"][0]) - 1) * n + candidate["tile"][1])
            <= 2147483647
            for candidate in records
        ):
            uncovered.append((m, n, k))
    if uncovered:
        raise ValueError(f"No declared candidate covers input shapes: {uncovered}")
    return dict(Counter(candidate["pipeline"] for candidate in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "docs/gfx950_mxscale_20261029/shapes_421.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("co_kernels_gfx950.json"),
    )
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    shapes = [(int(row["M"]), int(row["N"]), int(row["K"])) for row in rows]
    k_values = sorted({k for _, _, k in shapes})
    if any(k <= 0 or k % MFMA_K for k in k_values):
        raise ValueError("Input K values must be positive multiples of 128")
    document = generate_document(shapes)
    counts = validate_document(document, shapes)
    print(
        json.dumps(
            {
                "input_shapes": len(shapes),
                "fixed_k_values": sorted({entry["fixed_k"] for entry in document["gfx950"] if entry["fixed_k"]}),
                "input_k_values": k_values,
                "co_counts": counts,
                "total_co": sum(counts.values()),
                "jit_kid_base": 200,
                "parameter_counts": document["_generation"]["parameter_counts"],
                "local_co_range": [
                    min(candidate["kid"] for candidate in document["gfx950"]),
                    max(candidate["kid"] for candidate in document["gfx950"]),
                ],
            },
            indent=2,
        )
    )
    if not args.count_only:
        write_document(document, args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
