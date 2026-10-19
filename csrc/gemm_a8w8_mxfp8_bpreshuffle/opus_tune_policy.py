# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only shape pruning and coarse/refinement selection for OPUS tuning."""

import math
import re
from collections import defaultdict


def _configuration(instance):
    entry = instance.bpreshuffle_entry
    depth, output, ahead = map(
        int, re.search(r"_p(\d+)_o(\d+)_ba(\d+)_sk", entry["config_name"]).groups()
    )
    wave_k = instance.BLOCK_SIZE // (64 * instance.T_M * instance.T_N)
    geometry = (
        entry["pipeline"], instance.B_M, instance.B_N, instance.B_K,
        instance.T_M, instance.T_N, wave_k, instance.bpreshuffle_split_k,
    )
    return geometry, depth, ahead, output


def coarse_candidates(candidates, m, n, k, cu_num=256):
    """Keep legal COs and two schedule samples per admitted tile/wave geometry.

    Like PR #6603, reject split-K when the unsplit grid already fills the device;
    callers have already checked alignment and workspace/resource legality.
    Keep wave layouts distinct, sample both output modes, and measure all fine
    schedules, output modes, cache policies and WG/CU on winning geometries.
    This is a tune policy, not a kernel support predicate or runtime dispatch.
    """
    groups = defaultdict(list)
    co = []
    for kid, instance in candidates:
        if instance.bpreshuffle_co is not None:
            co.append(kid)
            continue
        geometry, depth, ahead, output = _configuration(instance)
        base_wg = ((m + instance.B_M - 1) // instance.B_M) * (
            (n + instance.B_N - 1) // instance.B_N
        )
        if instance.bpreshuffle_split_k > 1 and base_wg >= cu_num:
            continue
        groups[geometry].append((kid, instance, depth, ahead, output))
    if not groups and not co and candidates:
        # Preserve coverage if a future pipeline has only split-K variants.
        for kid, instance in candidates:
            geometry, depth, ahead, output = _configuration(instance)
            groups[geometry].append((kid, instance, depth, ahead, output))
    coarse = set(co)
    for variants in groups.values():
        # Sample a shallow/default-cache schedule and a deeper/streaming one.
        # Each sample retains normal timing and full correctness validation.
        output_modes = sorted({item[4] for item in variants})
        for target_depth, cache, wg, output in (
            (2, (0, 0), 2, output_modes[0]),
            (4, (1, 17), 1, output_modes[-1]),
        ):
            selected = min(
                variants,
                key=lambda item: (
                    item[4] != output,
                    abs(item[2] - target_depth),
                    (item[1].cachectl_a, item[1].cachectl_b) != cache,
                    item[1].WG_PER_CU != wg,
                    item[3], item[0],
                ),
            )
            coarse.add(selected[0])
    return sorted(coarse), {
        geometry: [item[0] for item in variants]
        for geometry, variants in groups.items()
    }


def refinement_candidates(groups, measurements, already_measured, top_groups=3):
    """Expand the three fastest valid geometries independently per pipeline.

    ``measurements`` contains (kid, microseconds, error_ratio). Failed samples
    cannot win. When every sample in a pipeline fails, expand its first three
    geometries as a bounded fallback rather than dropping that pipeline.
    """
    measured = {
        int(kid): float(us) for kid, us, error in measurements
        if math.isfinite(float(us)) and float(us) > 0 and float(error) == 0
    }
    by_pipeline = defaultdict(list)
    for geometry, variants in groups.items():
        best = min((measured[kid] for kid in variants if kid in measured), default=math.inf)
        by_pipeline[geometry[0]].append((best, geometry, variants))
    selected = set()
    for geometries in by_pipeline.values():
        ranked = sorted(geometries, key=lambda item: (item[0], item[1]))
        viable = [item for item in ranked if math.isfinite(item[0])]
        for _, _, variants in (viable or ranked)[:top_groups]:
            selected.update(variants)
    return sorted(selected - set(already_measured))


def refinement_upper_bound(groups, already_measured, top_groups=3):
    by_pipeline = defaultdict(list)
    measured = set(already_measured)
    for geometry, variants in groups.items():
        by_pipeline[geometry[0]].append(len(set(variants) - measured))
    return sum(sum(sorted(sizes, reverse=True)[:top_groups]) for sizes in by_pipeline.values())
