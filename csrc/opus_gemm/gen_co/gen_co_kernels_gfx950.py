#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate one unified gfx950 MXFP8 CO manifest from pipeline parameters.

Regular parameter candidates and the final used configurations are generated
from source, without inheriting experimental variants from an existing JSON.
IDs and image suffixes remain compatible with the verified 421-shape result.
The manifest includes all candidates; build_co.py keeps its normal build logic.

Pipeline parameter catalog (tile dimensions are M x N x K):

* pin_agpr: 12 regular + 22 final configurations. Regular candidates use
  tile 256x256x128, waves 2x2, two stages, scale panel 16/32/64,
  unroll 2, PAD_M true/false, and the two output/layout bundles below.
  Final configurations also specify group-M 4/16, unroll 4/8,
  static N 1536/2048/7168, scale chunks 32/64, or pin panel 32/64.
  These are explicit supported configurations, not a Cartesian product.
* prefetch_b_vgpr: 384 regular + 1 final configuration. The seven
  (M, N, wave-N) geometries are the union of B_GEOMETRIES and
  TILED_B_GEOMETRIES. Regular candidates use K-tile 128, four stages,
  FIXED_K_VALUES, and pin-A/pin-C (false,true)/(true,false)/(true,true).
  The BD=2/XCD/contiguous and BD=1/linear/strided bundles each yield
  192 configurations after b_vgpr_lds() checks. The final configuration
  uses tile 96x128x256, waves 1x4, two stages, BD=1, fixed K=7168.
* prefetch_b_vgpr_overlap: the same 384 regular parameters, with
  publication point 0, plus three final configurations:
  (tile, wave-N, stages, BD, publication) =
  (128x128x128, 4, 3, 2, 0), (64x128x256, 2, 4, 2, 1),
  (64x64x256, 2, 4, 2, 2); all three have fixed K=7168.
* wave_interleave: one final configuration, tile 192x256x256, waves 2x4,
  512 threads, traits (192,0,8,true,1,17,4), body options (true,false).

The complete manifest has 807 configurations. Regular B candidates must pass
tile/vector divisibility, K-depth, register-pinning, scale-load, and rounded
LDS <= 160 KiB checks before configuration deduplication. FINAL_CONFIGURATIONS
spells out the additional implementations and their supported shape scopes.
Image cleanup is a separate operation after the complete CO build.
"""

import argparse
import copy
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

LDS_LIMIT = 160 * 1024
WARP_SIZE = 64
MFMA_M = MFMA_N = 16
MFMA_K = 128
VECTOR_SIZE = 16
B_GEOMETRIES = ((64, 128, 2), (96, 128, 4), (96, 256, 4), (128, 128, 4))
# Additional tiled layouts use the same pipeline parameter generation.
# Their complete identities remain compatible with existing result CSVs.
TILED_B_GEOMETRIES = ((48, 128, 2), (64, 128, 4), (96, 256, 4), (128, 256, 4))
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


# Frozen parameter universe keeps the original CO IDs and JIT base stable.
FIXED_K_VALUES = (384, 768, 1024, 1536, 2048, 3072, 4096, 7168, 16384, 21504)
# The additional definitions are final configurations, grouped by pipeline.
# Tuple: kid, regular parameter base, overrides, removed fields, scope, image suffix.
# Image suffixes are compatibility names, not references to an old implementation.
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

FINAL_CONFIGURATIONS = ((100006,
  394,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, false, 4>']},
  (),
  0,
  'exp_nt_seed394_sp32_u2_pm1_wide0_gm4_h65ba257a3de3'),
 (100019,
  295,
  {'stages': 3,
   'traits': {'args': ['128',
                       '128',
                       '3',
                       '2',
                       '0',
                       'true',
                       '7168',
                       '4',
                       '1',
                       '0',
                       '2',
                       '0',
                       '128',
                       'true',
                       'true',
                       'false',
                       '0',
                       '0'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_traits'}},
  (),
  1,
  'exp_nt_seed295_p3_bd2_ba4_xcd1_contig1_k7168_gm0_ca0_cb0_pinA1C0_pub0_hd852f71a6088'),
 (100024,
  392,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, false, 16>']},
  (),
  2,
  'exp2_nt_seed100001_sp64_u2_pm0_wide0_gm16_hc7f7619cdfda'),
 (100025,
  389,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>']},
  (),
  2,
  'exp2_nt_seed100003_sp64_u2_pm0_wide1_gm16_h73b09ed56a4b'),
 (100045,
  390,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, false, 4>'],
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'}},
  (),
  3,
  'pm0_stage5_seed100000_2e0280e5b12a2933'),
 (100047,
  387,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 4>'],
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'}},
  (),
  4,
  'pm0_stage5_seed100002_5b6a89dcecce3d7e'),
 (100049,
  393,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, false, 4>'],
   'traits': {'args': ['64', '4', 'true'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'}},
  (),
  5,
  'pm1_stage5_seed100004_2aa4459884833cbd'),
 (100051,
  390,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, false, 16>'],
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'}},
  (),
  6,
  'pm0_stage5_seed100024_34caecd2f810a565'),
 (100052,
  387,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>'],
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'}},
  (),
  7,
  'pm0_stage5_seed100025_79cd545bf3bb77dc'),
 (100071,
  390,
  {'body_args': ['opus_stage6_static_shape_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, false, 4>, 0, 1536>'],
   'static_m': 0,
   'static_n': 1536,
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'},
   'traits_header': 'opus_gemm_traits_stage6_gfx950.cuh'},
  (),
  8,
  'pm0_stage6_seed100045_a3088298e8ab0aaf'),
 (100072,
  387,
  {'body_args': ['opus_stage6_static_shape_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 4>, 0, 1536>'],
   'static_m': 0,
   'static_n': 1536,
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'},
   'traits_header': 'opus_gemm_traits_stage6_gfx950.cuh'},
  (),
  9,
  'pm0_stage6_seed100047_f7e380d72556369e'),
 (100073,
  387,
  {'body_args': ['opus_stage6_static_shape_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 4>, 0, 2048>'],
   'static_m': 0,
   'static_n': 2048,
   'traits': {'args': ['64', '4', 'false'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits'},
   'traits_header': 'opus_gemm_traits_stage6_gfx950.cuh'},
  (),
  10,
  'pm0_stage6_seed100047_c85d29a604ade2fa'),
 (100076,
  392,
  {'body_args': ['opus_stage6_static_shape_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits>, '
                 '0, 7168>'],
   'static_m': 0,
   'static_n': 7168,
   'traits_header': 'opus_gemm_traits_stage6_gfx950.cuh'},
  (),
  11,
  'pm0_stage6_seed392_c6c24a8c47259c42'),
 (100083,
  389,
  {'body_args': ['opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>'],
   'device_flags': 'flags_stage6_agpr_form'},
  (),
  12,
  'pm0_stage6_seed100025_24e1ffa76fc2973f'),
 (100205,
  395,
  {'body_args': ['opus_stage7_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits>, '
                 '2>'],
   'body_func': 'opus_stage7_pin_body',
   'pin_mask': 0,
   'pipeline_header': 'opus_gemm_pipeline_stage7_pin_agpr_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage7_gfx950.cuh'},
  (),
  13,
  'pm1_stage7_seed395_18d013e086edc68a'),
 (100224,
  54,
  {'body_args': ['opus_stage7_b_traits<CoTraits, true, false, -1>'],
   'body_func': 'opus_stage7_b_detail::bulk_panel_body',
   'pin_mask': 0,
   'pipeline_header': 'opus_gemm_pipeline_stage7_b_vgpr_gfx950.cuh',
   'stages': 2,
   'tile': [96, 128, 256],
   'traits': {'args': ['96',
                       '128',
                       '2',
                       '1',
                       '1',
                       'true',
                       '7168',
                       '4',
                       '1',
                       '17',
                       '2',
                       '0',
                       '256',
                       'false',
                       'false',
                       'false',
                       '0'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits'},
   'traits_header': 'opus_gemm_traits_stage7_gfx950.cuh'},
  (),
  14,
  'pm1_stage7_seed7510_02ad82dd1814986b'),
 (100231,
  216,
  {'body_args': ['opus_stage7_b_traits<CoTraits, false, true, -1>'],
   'body_func': 'opus_stage7_b_detail::bulk_panel_overlap_body',
   'launch_bounds': [128, 2],
   'pin_mask': 0,
   'pipeline_header': 'opus_gemm_pipeline_stage7_b_vgpr_gfx950.cuh',
   'tile': [64, 128, 256],
   'traits': {'args': ['64',
                       '128',
                       '4',
                       '2',
                       '17',
                       'true',
                       '7168',
                       '2',
                       '2',
                       '1',
                       '2',
                       '0',
                       '256',
                       'false',
                       'false',
                       'false',
                       '0',
                       '1'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_traits'},
   'traits_header': 'opus_gemm_traits_stage7_gfx950.cuh'},
  (),
  15,
  'pm1_stage7_seed41971_f877f9673a05ccda'),
 (100413,
  216,
  {'body_args': ['opus_stage7_b_traits<opus_stage6_b_queue_traits<CoTraits, 7>, true, true, 0>'],
   'body_func': 'opus_stage9_b_detail::bulk_panel_overlap_body',
   'pin_mask': 0,
   'pipeline_header': 'opus_gemm_pipeline_stage9_compact_b_gfx950.cuh',
   'tile': [64, 64, 256],
   'traits': {'args': ['64',
                       '64',
                       '4',
                       '2',
                       '1',
                       'true',
                       '7168',
                       '2',
                       '1',
                       '17',
                       '2',
                       '0',
                       '256',
                       'false',
                       'false',
                       'false',
                       '0',
                       '2'],
              'name': 'opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_traits'},
   'traits_header': 'opus_gemm_traits_stage7_gfx950.cuh'},
  (),
  16,
  'pm1_stage9_compact_seed41672_356eabf81d59cc3b'),
 (100606,
  389,
  {'body_args': ['opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>'],
   'body_func': 'opus_stage7_pin_body',
   'device_flags': 'flags_stage6_agpr_form',
   'pipeline_header': 'opus_gemm_pipeline_stage7_pin_agpr_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage11_unroll_gfx950.cuh'},
  (),
  17,
  'pm0_stage11_u8_seed100083_7f69bf8f7b7dd23c'),
 (100610,
  389,
  {'body_args': ['opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>'],
   'body_func': 'opus_stage7_pin_body',
   'pipeline_header': 'opus_gemm_pipeline_stage7_pin_agpr_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage11_unroll_gfx950.cuh'},
  (),
  18,
  'pm0_stage11_u8_seed100025_6069937f4a5c4b85'),
 (100808,
  389,
  {'body_args': ['opus_stage13_scale_traits<opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>, 32>'],
   'body_func': 'opus_stage13_pin_body',
   'device_flags': 'flags_stage6_agpr_form',
   'pipeline_header': 'opus_gemm_pipeline_stage13_local_scale_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage13_scale_gfx950.cuh'},
  (),
  19,
  'pm0_stage13_local1_p32_seed100606_dbf9728098bf318d'),
 (100906,
  389,
  {'body_args': ['opus_stage14_scale_traits<opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>, 64>'],
   'body_func': 'opus_stage14_pin_body',
   'device_flags': 'flags_stage6_agpr_form',
   'pipeline_header': 'opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage14_chunk_scale_gfx950.cuh'},
  (),
  20,
  'pm0_stage14_chunk1_p64_seed100606_ed127bd70f72026f'),
 (100908,
  389,
  {'body_args': ['opus_stage14_scale_traits<opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>, 32>'],
   'body_func': 'opus_stage14_pin_body',
   'device_flags': 'flags_stage6_agpr_form',
   'pipeline_header': 'opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage14_chunk_scale_gfx950.cuh'},
  (),
  21,
  'pm0_stage14_chunk1_p32_seed100606_a81c8bf13113466f'),
 (100911,
  389,
  {'body_args': ['opus_stage14_scale_traits<opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>, 64>'],
   'body_func': 'opus_stage14_pin_body',
   'pipeline_header': 'opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage14_chunk_scale_gfx950.cuh'},
  (),
  22,
  'pm0_stage14_chunk1_p64_seed100610_59dc92ef7ba1bf2f'),
 (101011,
  389,
  {'body_args': ['opus_stage15_pin_traits<opus_stage14_scale_traits<opus_stage11_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, '
                 'opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 16>, 8, 0>, 64>, '
                 '32, 2>'],
   'body_func': 'opus_stage14_pin_body',
   'device_flags': 'flags_stage6_agpr_form',
   'pipeline_header': 'opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage15_pin_unroll_gfx950.cuh'},
  (),
  23,
  'pm0_stage15_pin_u2_p32_seed100906_3d12dd3df85711de'),
 (101020,
  395,
  {'body_args': ['opus_stage15_pin_traits<opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits<CoTraits>, '
                 '64, 2>'],
   'body_func': 'opus_stage14_pin_body',
   'pipeline_header': 'opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh',
   'traits_header': 'opus_gemm_traits_stage15_pin_unroll_gfx950.cuh'},
  (),
  24,
  'pm1_stage15_pin_u2_p64_seed395_6c89e1217b6dbe16'),
 (101312,
  0,
  {'block_size': 512,
   'body_args': ['opus_stage18_interleave_traits<CoTraits, true, false>'],
   'body_func': 'opus_stage18_interleave_body',
   'launch_bounds': [512, 1],
   'pin_mask': 0,
   'pin_variant': False,
   'pipeline': 'wave_interleave',
   'pipeline_header': 'opus_gemm_pipeline_stage18_interleave_gfx950.cuh',
   'tile': [192, 256, 256],
   'traits': {'args': ['192', '0', '8', 'true', '1', '17', '4'],
              'name': 'opus_gemm_bpreshuffle_wave_interleave_traits'},
   'traits_header': 'opus_gemm_traits_stage18_remote_final_gfx950.cuh',
   'wave_layout': [2, 4]},
  ('m_align', 'partial_pin_c'),
  25,
  'stage18_remote_seed47680_82f0d5eb1d22ee90'))


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


def pin_agpr_candidates(pad_m_values=(True, False), *, tiled_layout=False):
    unroll = 2
    for pad_m, panel in itertools.product(pad_m_values, (16, 32, 64)):
        wrapper = (
            TRAITS_PREFIX + "4wave_pin_agpr_traits<CoTraits>"
            if tiled_layout else
            TRAITS_PREFIX + "4wave_pin_agpr_traits<CoTraits, "
            "opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, "
            "opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 0>"
        )
        candidate = record(
            "pin_agpr",
            (256, 256, 128),
            (2, 2),
            2,
            TRAITS_PREFIX + "4wave_pin_agpr_panel_traits",
            (panel, unroll, pad_m),
            TRAITS_PREFIX + "4wave_pin_agpr",
            "4wave_pin_agpr",
            (wrapper,),
            None if tiled_layout else 16384,
            pin_mask=3,
            pad_m=pad_m,
            wide_output=not tiled_layout,
        )
        # The padded scale load guards the start of each 16-byte M vector;
        # the unpadded path loads and stores a complete 256-row tile.
        candidate["m_align"] = 16 if pad_m else candidate["tile"][0]
        # C01 is tagged with C01_AGPR_BASE=148; its sixteen MFMA
        # accumulators occupy 64 dwords. The other C halves are not all pinned.
        candidate["partial_pin_c"] = {"base": 148, "dwords": 64}
        # Keep existing CO image names; k0/reset0 are only legacy name tokens.
        suffix = (
            ("" if pad_m else "pm0_") + f"sp{panel}_u{unroll}"
            if tiled_layout else
            f"pm{int(pad_m)}_sp{panel}_k0_reset0_u{unroll}_gm0_wide_nt0"
        )
        yield candidate, suffix


def b_vgpr_lds(bm, bn, wn, bk, nb, fixed_k, pin_a, pin_c):
    threads = wn * WARP_SIZE
    mr, nr = bm // MFMA_M, bn // (wn * MFMA_N)
    if bn % (wn * MFMA_N) or mr * nr * 4 > 256:
        return None
    if bm * bk % (threads * VECTOR_SIZE) or fixed_k % bk:
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
    if ceil_div(lds, 1280) * 1280 > LDS_LIMIT:
        return None
    return lds


def b_vgpr_candidates(k_values, *, tiled_layout=False, geometries=None):
    if geometries is None:
        # Keep each original sweep first, then cover all seven geometries with
        # that sweep's BD/XCD/Contiguous bundle.
        original = TILED_B_GEOMETRIES if tiled_layout else B_GEOMETRIES
        geometries = tuple(dict.fromkeys((*original, *B_GEOMETRIES, *TILED_B_GEOMETRIES)))
    ahead = 1 if tiled_layout else 2
    for (bm, bn, wn), fixed_k in itertools.product(geometries, k_values):
        bk, nb = 128, 4
        for pin_a, pin_c in itertools.product((False, True), repeat=2):
            if not (pin_a or pin_c):
                continue
            if b_vgpr_lds(bm, bn, wn, bk, nb, fixed_k, pin_a, pin_c) is None:
                continue
            mask = int(pin_a) + 4 * int(pin_c)
            args = (
                bm,
                bn,
                nb,
                ahead,
                0,
                not tiled_layout,  # XCD tile order.
                fixed_k,
                wn,
                1,
                0,
                2,
                0,
                bk,
                not tiled_layout,  # A/B strides specialized to K when true.
                pin_a,
                pin_c,
                0,
            )
            candidate = record(
                "prefetch_b_vgpr",
                (bm, bn, bk),
                (1, wn),
                nb,
                TRAITS_PREFIX + "prefetch_b_vgpr_traits",
                args,
                "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body",
                "b_vgpr",
                ("CoTraits",),
                fixed_k,
                fixed_k=fixed_k,
                pin_mask=mask,
                grid_layout="linear" if tiled_layout else "xcd",
            )
            suffix = (
                f"p{nb}_bd{ahead}_k{fixed_k}_pinA{int(pin_a)}B0C{int(pin_c)}"
                if tiled_layout else
                f"p{nb}_a2_k{fixed_k}_c0x0_gm0_o0_contig_"
                f"pinA{int(pin_a)}B0C{int(pin_c)}_nt0"
            )
            yield candidate, suffix


def b_vgpr_overlap_candidates(k_values, *, tiled_layout=False):
    for candidate, suffix in b_vgpr_candidates(k_values, tiled_layout=tiled_layout):
        candidate["pipeline"] = "prefetch_b_vgpr_overlap"
        candidate["traits"]["name"] = TRAITS_PREFIX + "prefetch_b_vgpr_overlap_traits"
        candidate["traits"]["args"].append("0")
        candidate["traits_header"] = (
            "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"
        )
        if tiled_layout:
            candidate["body_args"] = ["CoTraits", "CoTraits::PUBLISH_M"]
        else:
            candidate["body_func"] = (
                "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_overlap_body"
            )
            suffix = suffix.removesuffix("_nt0") + "_pub0_nt0"
        yield candidate, suffix


def _generate_parameter_document(k_values):
    k_values = sorted(set(k_values))
    local_candidates = itertools.chain(
        pin_agpr_candidates((True,)),
        b_vgpr_candidates(k_values),
        b_vgpr_overlap_candidates(k_values),
        pin_agpr_candidates((False,)),
    )
    tiled_candidates = itertools.chain(
        pin_agpr_candidates((False, True), tiled_layout=True),
        b_vgpr_candidates(k_values, tiled_layout=True),
        b_vgpr_overlap_candidates(k_values, tiled_layout=True),
    )
    records, suffixes = [], {}
    configs = set()
    sources = {}
    for source, candidates in (("local", local_candidates), ("tiled_layout", tiled_candidates)):
        first_kid, duplicates = len(records), 0
        for candidate, suffix in candidates:
            config = json.dumps(candidate, sort_keys=True)
            if config in configs:
                duplicates += 1
                continue
            configs.add(config)
            kid = len(records)
            candidate["kid"] = kid
            candidate["configKey"] = f"bpreshuffle_co_{candidate['pipeline']}_{kid}"
            records.append(candidate)
            suffixes[str(kid)] = suffix
        sources[source] = {
            "added": len(records) - first_kid,
            "duplicates_skipped": duplicates,
            "kid_range": [first_kid, len(records) - 1] if len(records) > first_kid else [],
        }
    return {
        "_generation": {"candidate_sources": sources},
        "defaults": {
            "tag": "a8w8_blockscale_mxscale_gemm_bpreshuffle",
            "dtypes": {"a": "fp8_t", "b": "fp8_t", "c": "bf16_t", "acc": "fp32_t"},
        },
        "device_flag_sets": {"flags_00": list(DEVICE_FLAGS)},
        "image_suffixes": suffixes,
        "gfx950": records,
    }


def generate_document():
    """Emit the complete parameter set even when no output JSON exists."""
    document = _generate_parameter_document(FIXED_K_VALUES)
    regular = {entry["kid"]: entry for entry in document["gfx950"]}
    if set(regular) != set(range(780)):
        raise ValueError("Regular CO parameters must preserve original IDs 0..779")
    for kid, base_kid, overrides, removed, scope, suffix in FINAL_CONFIGURATIONS:
        entry = copy.deepcopy(regular[base_kid])
        entry.pop("configKey", None)
        for field in removed:
            entry.pop(field)
        entry.update(copy.deepcopy(overrides))
        entry["kid"] = kid
        entry["supported_shapes"] = [list(shape) for shape in FINAL_SHAPE_SCOPES[scope]]
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = suffix
    option = DEVICE_FLAGS.index("--amdgpu-mfma-vgpr-form")
    document["device_flag_sets"]["flags_stage6_agpr_form"] = (
        DEVICE_FLAGS[:option - 1] + DEVICE_FLAGS[option + 1:]
    )
    document["jit_kid_base"] = 780
    document["_generation"].update(
        version=2,
        regular_parameter_co=780,
        final_configurations=len(FINAL_CONFIGURATIONS),
    )
    return document


def validate_document(document, shapes):
    records = document["gfx950"]
    kids = {record["kid"] for record in records}
    if len(kids) != len(records):
        raise ValueError("Duplicate local CO kid")
    if any(type(kid) is not int or kid < 0 for kid in kids):
        raise ValueError("Local CO kids must be nonnegative integers")
    if {kid for kid in kids if kid < 100000} != set(range(780)):
        raise ValueError("Regular CO IDs must remain 0..779; final additions are sparse")
    if document.get("jit_kid_base") != 780:
        raise ValueError("JIT IDs must remain based at 780")
    if set(document["image_suffixes"]) != {str(kid) for kid in kids}:
        raise ValueError("Image suffixes must exactly match declared CO IDs")
    symbols = set()
    for candidate in records:
        if candidate["kid"] < 780 and not candidate.get("pin_mask", 0):
            raise ValueError("CO candidates must declare their register pinning")
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
            suffix = document["image_suffixes"][str(candidate["kid"])]
            if candidate["kid"] < 780 and not (suffix.startswith(f"pm{int(pad_m)}_")
                    or pad_m and suffix.startswith("sp")):
                raise ValueError("Pin PAD_M differs between metadata and image name")
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
    document = generate_document()
    counts = validate_document(document, shapes)
    print(
        json.dumps(
            {
                "input_shapes": len(shapes),
                "fixed_k_values": list(FIXED_K_VALUES),
                "input_k_values": k_values,
                "co_counts": counts,
                "total_co": sum(counts.values()),
                "regular_parameter_co": 780,
                "final_configurations": len(FINAL_CONFIGURATIONS),
                "candidate_sources": document["_generation"]["candidate_sources"],
                "local_co_range": [
                    min(candidate["kid"] for candidate in document["gfx950"]),
                    max(candidate["kid"] for candidate in document["gfx950"]),
                ],
            },
            indent=2,
        )
    )
    if not args.count_only:
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
