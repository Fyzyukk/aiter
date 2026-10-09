# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scalar parameter catalog for the five gfx950 MXFP8 B-preshuffle pipelines.

Pipeline names and compile parameters are the public selection interface. The
existing numeric IDs remain private launcher ABI keys, including historical
configurations needed to replay saved CSVs. This module imports no tensor or
GPU packages; the existing registry remains the source of supported shapes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import json
from types import MappingProxyType

try:
    from . import opus_gemm_common as _common
except ImportError:  # Sibling imports used by directly executable OPUS tools.
    import opus_gemm_common as _common


@dataclass(frozen=True)
class BpreshuffleConfig:
    """Complete registered traits and launcher policy, with a stable schema.

    The five producers use different subsets of the axes below. Zero/False
    and the declared defaults are neutral values for axes unused by a given
    producer (for example, pin has no NUM_STAGES and register has no scale
    panel). Resolution accepts registered combinations only; changing an
    inactive axis does not request an invented specialization.
    """

    pipeline: str
    tile_m: int
    tile_n: int
    tile_k: int = 128
    wave_m: int = 1
    wave_n: int = 1
    wave_k: int = 1
    # Traits::SCHEDULE selects a compile-time load/wait policy in each of the
    # five producer templates. It is never an exact configuration ID.
    schedule: int = 0
    stages: int = 0
    cluster: int = 1
    prefetch: int = 0
    b_direct: bool = False
    b_ahead: int = 0
    b_direct_sets: int = 0
    c_chunk_rows: int = 0
    register_scales: bool = False
    xor_lds: bool = False
    early_scale_loads: bool = False
    prefetch_before_read: bool = False
    read_only_drain: bool = False
    fine_m_loads: bool = False
    scale_panel: int = 0
    scale_reset: bool = False
    loop_unroll: int = 2
    group_m: int = 0
    fixed_k: int = 0
    # Runtime split counts live in launch arguments / the CSV splitK column.
    # None means that this kernel does not specialize the split count.
    split_k: int | None = 1
    runtime_split_k: bool = False
    reduce_vec: int = 4
    reduce_block: int = 128
    store_cache: int = 0
    b_cache: int = 0
    output_mode: int = 0
    pad_n: bool = False
    reuse_b_scale: bool = False
    pin_agpr: bool = False
    dynamic_lds: bool = False
    m_align: int = 1
    max_m: int | None = None
    max_k: int | None = None
    max_tensor_bytes: int = 2**31 - 1
    sfa_alignment: int = 16
    c_alignment: int = 16
    # Static K-specific traits overrides, and ordered host dispatch to another
    # parameter configuration. Nested payloads also contain no numeric IDs.
    specializations: tuple[tuple[int, ...], ...] = ()
    dispatch: tuple[tuple[str, str], ...] = ()
    _legacy_kid: int = field(default=-1, repr=False, compare=False)

    @property
    def legacy_kid(self):
        """Internal ABI compatibility key; use pipeline/parameters to select."""
        return self._legacy_kid

    @property
    def compile_params(self):
        return MappingProxyType({
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "pipeline" and not item.name.startswith("_")
        })

    def to_json(self):
        """Canonical complete compile parameters, excluding pipeline and ID."""
        return json.dumps(dict(self.compile_params), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class BpreshufflePipeline:
    name: str
    description: str
    configs: tuple[BpreshuffleConfig, ...]

    @property
    def active_configs(self):
        return tuple(config for config in self.configs
                     if config.legacy_kid in _common.A8W8_BPRESHUFFLE_TUNING_KIDS)


_FAMILY_PIPELINE = {
    "pin": "pin", "padded_pin": "pin", "pin_fixed": "pin", "pad_pin_fixed": "pin",
    "main": "tiled", "small_main": "tiled", "narrow": "tiled",
    "geometry": "tiled", "tile_order": "tiled", "shortk": "tiled",
    "register": "register", "register_split": "register",
    "small_lds": "lds", "fine_lds": "lds", "small_direct_b": "lds",
    "large_output": "large_output",
}

# Existing small traits parameters, expressed without registry IDs. Tuple
# fields: stages, cluster, output, register_scales, xor, early, prefetch, drain.
_LDS_BASE = {
    "small_lds_32x64": (8, 2, 2, False, False, True, False, False),
    "small_lds_64x64": (4, 1, 2, False, False, True, True, True),
    "small_lds_96x64": (4, 1, 1, False, False, False, False, True),
    "small_lds_64x128": (6, 2, 2, False, False, False, False, False),
    "small_regscale_32x64": (4, 1, 2, True, False, False, False, False),
    "small_regscale_xor_32x64": (4, 1, 2, True, True, False, False, False),
    "small_regscale_xor_32x128": (4, 2, 2, True, True, False, True, False),
    "small_lds_deep_32x64": (12, 4, 2, False, False, True, False, False),
    "small_lds_deep_64x64": (8, 2, 2, False, False, True, False, False),
}
# Register tuple: prefetch, output, B-cache, reuse-B-scale.
_REGISTER_BASE = {
    "small_register_16x32": (6, 3, 3, False),
    "small_register_16x16": (2, 0, 0, False),
    "small_register_32x32": (3, 3, 0, True),
    "small_register_prefetch_16x16": (3, 3, 3, False),
    "small_register_prefetch_16x32": (3, 3, 3, False),
    "small_register_wavek_16x32": (2, 3, 3, False),
    "small_register_wavek_32x32": (2, 3, 3, True),
    "small_register_wavek_32x64": (2, 3, 3, True),
}


def _lds_params(key):
    stages, cluster, output, registers, xor, early, prefetch, drain = _LDS_BASE[key]
    return dict(stages=stages, cluster=cluster, output_mode=output,
                register_scales=registers, xor_lds=xor, early_scale_loads=early,
                prefetch_before_read=prefetch, read_only_drain=drain,
                reduce_vec=4, reduce_block=128, store_cache=0)


def _config(kid, instance):
    family = _common.A8W8_BPRESHUFFLE_FAMILY_BY_KID[kid]
    variant = instance.bpreshuffle_variant
    runtime = bool(variant and variant.runtime_split_k)
    params = dict(
        pipeline=_FAMILY_PIPELINE[family], tile_m=instance.B_M, tile_n=instance.B_N,
        tile_k=instance.B_K, wave_m=instance.T_M, wave_n=instance.T_N,
        wave_k=instance.BLOCK_SIZE // (64 * instance.T_M * instance.T_N),
        fixed_k=instance.bpreshuffle_fixed_k,
        split_k=None if runtime else instance.bpreshuffle_split_k,
        runtime_split_k=runtime, reduce_vec=instance.bpreshuffle_reduce_vec,
        reduce_block=instance.bpreshuffle_reduce_block,
        store_cache=instance.bpreshuffle_store_cache, pad_n=instance.bpreshuffle_pad_n,
        pin_agpr=kid in _common.A8W8_BPRESHUFFLE_PIN_AGPR_KIDS,
        m_align=instance.m_align, max_m=instance.max_m, max_k=instance.max_k,
        max_tensor_bytes=instance.max_tensor_bytes,
        specializations=instance.bpreshuffle_specializations, _legacy_kid=kid,
    )
    if variant:
        params.update(dynamic_lds=variant.dynamic_lds,
                      sfa_alignment=variant.sfa_alignment, c_alignment=variant.c_alignment,
                      schedule=variant.schedule)
    pipeline = params["pipeline"]
    tag = instance.name_tag
    if pipeline == "pin":
        params.update(scale_panel=64, loop_unroll=instance.bpreshuffle_loop_unroll,
                      scale_reset=instance.bpreshuffle_scale_reset)
        if not variant:
            params["schedule"] = 1 if instance.pad_m else 0
        if variant and family == "pin_fixed":
            fixed, panel, reset = _trait_axes(variant.traits)
            params.update(scale_panel=panel, scale_reset=reset)
        elif variant and family == "pad_pin_fixed":
            params["loop_unroll"] = 4
    elif pipeline == "tiled":
        params.update(stages=2, scale_panel=32)
        if family == "main":
            params["scale_panel"] = 128
        if not variant:
            params["schedule"] = (0 if family == "main" else
                                  1 if family == "small_main" and instance.B_M == 128 else
                                  2 if family == "small_main" else
                                  4 if instance.B_N == 64 else 3)
        if family in {"narrow", "tile_order"}:
            params["stages"] = 4 if instance.B_N == 64 else 3
        elif family == "small_main" and instance.B_M == 128:
            params["stages"] = 3
        if variant and family == "geometry":
            _, stages, panel, *_ = _trait_axes(variant.traits)
            params.update(stages=stages, scale_panel=panel)
        elif variant and family == "tile_order":
            panel, _, group = _trait_axes(variant.traits)
            params.update(scale_panel=panel, group_m=group)
        elif family == "shortk":
            params["scale_panel"] = 8
    elif pipeline == "register":
        if runtime:
            params.update(prefetch=3, output_mode=3, b_cache=3)
        elif tag.startswith("register_tail"):
            params.update(prefetch=instance.bpreshuffle_register_prefetch, output_mode=4,
                          b_cache=instance.cachectl_b, reuse_b_scale=128 % instance.B_N == 0)
        else:
            prefetch, output, cache, reuse = _REGISTER_BASE[f"{tag}_{instance.B_M}x{instance.B_N}"]
            params.update(prefetch=prefetch, output_mode=output, b_cache=cache,
                          reuse_b_scale=reuse)
    elif pipeline == "lds":
        if variant and family == "small_direct_b":
            actual, ahead = _trait_axes(variant.traits)
            baseline = _common.a8w8_mxscale_gemm_bpreshuffle_kernels_list[actual]
            params.update(_lds_params(f"{baseline.name_tag}_{baseline.B_M}x{baseline.B_N}"))
            params.update(b_direct=True, b_ahead=ahead)
        elif variant and family == "fine_lds":
            params.update(stages=4, output_mode=2, early_scale_loads=True,
                          prefetch_before_read=True, read_only_drain=True,
                          fine_m_loads=True, store_cache=2)
        elif tag.startswith("fine_lds"):
            params.update(stages=instance.bpreshuffle_stages, cluster=instance.bpreshuffle_cluster,
                          output_mode=2, early_scale_loads=True, prefetch_before_read=True,
                          read_only_drain=True, fine_m_loads=True, dynamic_lds=True)
        else:
            params.update(_lds_params(f"{tag}_{instance.B_M}x{instance.B_N}"))
            params["dynamic_lds"] = True
    else:
        params.update(stages=2, scale_panel=128)
        if variant:
            params["scale_panel"] = 16
            if variant.label == "large_output_direct_b_k1536":
                params.update(stages=3, b_direct=True, b_direct_sets=2, c_chunk_rows=96)
    return BpreshuffleConfig(**params)


def _trait_axes(traits):
    """Decode scalar template axes already present in the registry."""
    tokens = traits.split("<", 1)[1].removesuffix(">").split(",")
    return tuple(token.strip() == "true" if token.strip() in ("true", "false")
                 else int(token) for token in tokens)


_CONFIGS_BY_KID = {
    kid: _config(kid, instance)
    for kid, instance in sorted(_common.a8w8_mxscale_gemm_bpreshuffle_kernels_list.items())
}
for _kid, _instance in _common.a8w8_mxscale_gemm_bpreshuffle_kernels_list.items():
    if _instance.bpreshuffle_dispatch:
        _CONFIGS_BY_KID[_kid] = replace(_CONFIGS_BY_KID[_kid], dispatch=tuple(
            (condition, _CONFIGS_BY_KID[target].to_json())
            for condition, target in _instance.bpreshuffle_dispatch
        ))
CONFIGS_BY_LEGACY_KID = MappingProxyType(_CONFIGS_BY_KID)

_DESCRIPTIONS = {
    "pin": "Statement-pinned AGPR pipeline with fixed or runtime K and padded M variants.",
    "tiled": "Main and narrow tiles with staged matrix loads and scale panels.",
    "register": "Register operand queues with local K-wave reduction and optional runtime split-K.",
    "lds": "LDS rings with scale/layout policies, optional direct B, and runtime or fixed split-K.",
    "large_output": "Wide output addressing, scale panels, and optional chunked direct-B output.",
}
BPRESHUFFLE_PIPELINES = MappingProxyType({
    name: BpreshufflePipeline(name, description, tuple(
        config for config in _CONFIGS_BY_KID.values() if config.pipeline == name
    )) for name, description in _DESCRIPTIONS.items()
})
BPRESHUFFLE_PIPELINE_NAMES = frozenset(BPRESHUFFLE_PIPELINES)


def config_from_legacy_kid(kid):
    """Restore an exact historical configuration without redirecting aliases."""
    if isinstance(kid, bool) or not isinstance(kid, int) or kid not in _CONFIGS_BY_KID:
        raise ValueError(f"unknown MXFP8 B-preshuffle ABI ID: {kid!r}")
    return _CONFIGS_BY_KID[kid]


def _selected_pipelines(pipelines):
    if pipelines is None:
        return BPRESHUFFLE_PIPELINE_NAMES
    if isinstance(pipelines, str):
        pipelines = [part.strip() for part in pipelines.split(",")]
    selected = frozenset(item.name if isinstance(item, BpreshufflePipeline) else item
                         for item in pipelines)
    unknown = selected - BPRESHUFFLE_PIPELINE_NAMES
    if unknown or not selected:
        raise ValueError(f"invalid MXFP8 B-preshuffle pipelines: {sorted(unknown)}; "
                         f"available: {sorted(BPRESHUFFLE_PIPELINE_NAMES)}")
    return selected


def pipeline_configs(shape, pipelines=None, *, gfx="gfx950", outdtype="bf16",
                     include_legacy=False, families=None):
    """Enumerate registered compile configurations for a scalar (M, N, K)."""
    m, n, k = shape
    selected = _selected_pipelines(pipelines)
    return tuple(_CONFIGS_BY_KID[kid] for kid in
                 _common.a8w8_mxscale_bpreshuffle_candidate_kids(
                     gfx, m, n, k, outdtype, include_legacy=include_legacy, families=families
                 ) if _CONFIGS_BY_KID[kid].pipeline in selected)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate compile parameter: {key}")
        result[key] = value
    return result


def _params(config):
    if isinstance(config, BpreshuffleConfig):
        return dict(config.compile_params)
    if isinstance(config, str):
        try:
            config = json.loads(config, object_pairs_hook=_json_pairs)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("config must be a JSON object of compile parameters") from exc
    if not isinstance(config, Mapping):
        raise ValueError("config must be a JSON object or compile-parameter mapping")
    params = dict(config)
    allowed = set(next(iter(_CONFIGS_BY_KID.values())).compile_params)
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unknown compile parameters: {sorted(unknown)}")
    for key in ("specializations", "dispatch"):
        if key in params:
            if not isinstance(params[key], (tuple, list)):
                raise ValueError(f"{key} must be a sequence")
            try:
                params[key] = tuple(tuple(item) for item in params[key])
            except TypeError as exc:
                raise ValueError(f"{key} must contain sequences") from exc
    return params


def _same_type_value(expected, value):
    if type(expected) is not type(value):
        return False
    if isinstance(expected, tuple):
        return len(expected) == len(value) and all(
            _same_type_value(left, right) for left, right in zip(expected, value)
        )
    return expected == value


def _matches(config, params):
    # bool is an int subclass; strict types reject a malformed CSV silently
    # changing booleans to numeric axes or accepting float-valued tile sizes.
    return all(_same_type_value(getattr(config, key), value)
               for key, value in params.items())


def resolve_config(pipeline, config=None, **compile_params):
    """Resolve a registered configuration by complete payload or named axes.

    Partial axes must identify exactly one configuration. Unsupported values
    are rejected instead of inventing an uncompiled launcher specialization.
    """
    name = pipeline.name if isinstance(pipeline, BpreshufflePipeline) else pipeline
    _selected_pipelines([name])
    if config is not None and compile_params:
        raise ValueError("provide a config or named compile parameters, not both")
    params = _params(config if config is not None else compile_params)
    if isinstance(config, BpreshuffleConfig) and config.pipeline != name:
        raise ValueError("config belongs to a different pipeline")
    if config is not None and set(params) != set(next(iter(_CONFIGS_BY_KID.values())).compile_params):
        raise ValueError("a complete config must contain every compile parameter")
    matches = [item for item in BPRESHUFFLE_PIPELINES[name].configs if _matches(item, params)]
    if not matches:
        raise ValueError(f"unsupported {name} compile configuration")
    if len(matches) != 1:
        raise ValueError(f"ambiguous {name} compile parameters; matched {len(matches)} configurations")
    return matches[0]


def validate_saved_config(pipeline, config, kernel_id):
    """Validate new CSV metadata against its exact internal launcher ABI key."""
    expected = config_from_legacy_kid(kernel_id)
    name = pipeline.name if isinstance(pipeline, BpreshufflePipeline) else pipeline
    params = _params(config)
    if name != expected.pipeline or set(params) != set(expected.compile_params) or not _matches(expected, params):
        raise ValueError(f"pipeline/config does not match kernelId {kernel_id}")
    return expected
