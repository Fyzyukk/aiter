# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parameter tables for retained MXFP8 B-preshuffle pipeline variants.

One entry is an exact configuration, not a copied kernel body. This module is
scalar-only so selection, code generation and offline tools use the same table.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BpreshuffleVariant:
    kid: int
    label: str
    family: str
    pipeline_header: str
    traits_header: str
    kernel: str
    traits: str
    tile_m: int
    tile_n: int
    wave_m: int
    wave_n: int
    wave_k: int = 1
    m_align: int = 1
    max_m: int | None = None
    fixed_k: int = 0
    split_k: int = 1
    dynamic_lds: bool = False
    pin_agpr: bool = False
    reduce_vec: int = 16
    reduce_block: int = 128
    sfa_alignment: int = 16
    c_alignment: int = 16
    runtime_split_k: bool = False
    schedule: int = 0


# Compatibility descriptors are derived from the canonical scalar catalog.
try:
    from .opus_gemm_bpreshuffle_config import CATALOG_ROWS
except ImportError:
    from opus_gemm_bpreshuffle_config import CATALOG_ROWS

NEW_BPRESHUFFLE_VARIANTS = tuple(
    BpreshuffleVariant(**row['abi']['bpreshuffle_variant']) for row in CATALOG_ROWS
    if row['abi'].get('bpreshuffle_variant'))
NEW_BPRESHUFFLE_VARIANTS_BY_KID = {entry.kid: entry for entry in NEW_BPRESHUFFLE_VARIANTS}
assert len(NEW_BPRESHUFFLE_VARIANTS_BY_KID) == len(NEW_BPRESHUFFLE_VARIANTS) == 64
