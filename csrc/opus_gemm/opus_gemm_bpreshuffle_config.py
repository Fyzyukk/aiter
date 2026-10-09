# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Canonical scalar configurations for the five gfx950 B-preshuffle producers.

This module is independent of the kernel registry. Compile parameters construct
traits and launch instances; IDs only restore historical ABI names and policy.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

@dataclass(frozen=True)
class BpreshuffleConfig:
    """Canonical compile-time traits and launcher policy, with a stable schema.

    The five producers use different subsets of the axes below. Zero/False
    and the declared defaults are neutral values for axes unused by a given
    producer (for example, pin has no NUM_STAGES and register has no scale
    panel). Construction validates producer constraints; new legal combinations
    instantiate the same five producer templates without a numeric ID.
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
    tunable_axes: frozenset[str] = frozenset()

    @property
    def active_configs(self):
        return tuple(config for config in self.configs if config.legacy_kid in ACTIVE_LEGACY_KIDS)


_CATALOG = json.loads(Path(__file__).with_name('opus_gemm_bpreshuffle_catalog.json').read_text())
CATALOG_ROWS = tuple(_CATALOG['configs'])
CATALOG_BY_LEGACY_KID = MappingProxyType({row['legacy_kid']: row for row in CATALOG_ROWS})
LEGACY_ALIASES = MappingProxyType({int(k): v for k, v in _CATALOG['legacy_aliases'].items()})
ACTIVE_LEGACY_KIDS = frozenset(row['legacy_kid'] for row in CATALOG_ROWS if row['active'])


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate compile parameter: {key}')
        result[key] = value
    return result


_PUBLIC_FIELDS = {item.name: item for item in fields(BpreshuffleConfig)
                  if item.name != 'pipeline' and not item.name.startswith('_')}


def _params(config):
    if isinstance(config, BpreshuffleConfig) or (type(config).__name__ == "BpreshuffleConfig" and hasattr(config, "compile_params")):
        return dict(config.compile_params)
    if isinstance(config, str):
        try:
            config = json.loads(config, object_pairs_hook=_json_pairs)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError('config must be a JSON object of compile parameters') from exc
    if not isinstance(config, Mapping):
        raise ValueError('config must be a JSON object or compile-parameter mapping')
    params = dict(config)
    unknown = set(params) - set(_PUBLIC_FIELDS)
    if unknown:
        raise ValueError(f'unknown compile parameters: {sorted(unknown)}')
    for key in ('specializations', 'dispatch'):
        if key in params:
            if not isinstance(params[key], (tuple, list)):
                raise ValueError(f'{key} must be a sequence')
            try:
                params[key] = tuple(tuple(item) for item in params[key])
            except TypeError as exc:
                raise ValueError(f'{key} must contain sequences') from exc
    return params


_CONFIGS_BY_KID = {row['legacy_kid']: BpreshuffleConfig(
    pipeline=row['pipeline'], _legacy_kid=row['legacy_kid'], **_params(row['compile_params']))
    for row in CATALOG_ROWS}
CONFIGS_BY_LEGACY_KID = MappingProxyType(_CONFIGS_BY_KID)
_BY_IDENTITY = {(config.pipeline, config.to_json()): config for config in _CONFIGS_BY_KID.values()}
_DESCRIPTIONS = {
    'pin': 'Statement-pinned AGPR pipeline with fixed/runtime K and padded M.',
    'tiled': 'Staged matrix loads, scale panels and tile scheduling.',
    'register': 'Register operand queues, local K waves and optional runtime split-K.',
    'lds': 'LDS rings, scale/layout policies, optional direct B and split-K.',
    'large_output': 'Wide output addressing with optional chunked direct-B output.',
}
_COMMON_AXES = {'tile_m', 'tile_n', 'tile_k', 'wave_m', 'wave_n', 'wave_k', 'schedule',
                'fixed_k', 'm_align', 'max_m', 'max_k', 'max_tensor_bytes',
                'sfa_alignment', 'c_alignment'}
_TUNABLE_AXES = {
    'pin': _COMMON_AXES | {'scale_panel', 'scale_reset', 'loop_unroll', 'pin_agpr'},
    'tiled': _COMMON_AXES | {'stages', 'scale_panel', 'group_m', 'loop_unroll'},
    'register': _COMMON_AXES | {'prefetch', 'b_cache', 'output_mode', 'pad_n',
                              'reuse_b_scale', 'runtime_split_k', 'split_k', 'reduce_vec', 'reduce_block'},
    'lds': _COMMON_AXES | {'stages', 'cluster', 'output_mode', 'register_scales', 'xor_lds',
                         'early_scale_loads', 'prefetch_before_read', 'read_only_drain',
                         'fine_m_loads', 'b_direct', 'b_ahead', 'split_k', 'runtime_split_k',
                         'reduce_vec', 'reduce_block', 'store_cache', 'dynamic_lds',
                         'specializations', 'dispatch'},
    'large_output': _COMMON_AXES | {'stages', 'scale_panel', 'b_direct', 'b_direct_sets', 'c_chunk_rows'},
}
BPRESHUFFLE_PIPELINES = MappingProxyType({name: BpreshufflePipeline(
    name, description, tuple(config for config in _CONFIGS_BY_KID.values() if config.pipeline == name),
    frozenset(_TUNABLE_AXES[name])) for name, description in _DESCRIPTIONS.items()})
BPRESHUFFLE_PIPELINE_NAMES = frozenset(BPRESHUFFLE_PIPELINES)


def config_from_legacy_kid(kid):
    if type(kid) is not int or kid not in _CONFIGS_BY_KID:
        raise ValueError(f'unknown MXFP8 B-preshuffle ABI ID: {kid!r}')
    return _CONFIGS_BY_KID[kid]


def _selected_pipelines(pipelines):
    if pipelines is None:
        return BPRESHUFFLE_PIPELINE_NAMES
    if isinstance(pipelines, str):
        pipelines = [part.strip() for part in pipelines.split(',')]
    selected = frozenset(item.name if isinstance(item, BpreshufflePipeline) else item for item in pipelines)
    unknown = selected - BPRESHUFFLE_PIPELINE_NAMES
    if unknown or not selected:
        raise ValueError(f'invalid MXFP8 B-preshuffle pipelines: {sorted(unknown)}; '
                         f'available: {sorted(BPRESHUFFLE_PIPELINE_NAMES)}')
    return selected


def _same_type_value(expected, value):
    if type(expected) is not type(value):
        return False
    if isinstance(expected, tuple):
        return len(expected) == len(value) and all(_same_type_value(a, b) for a, b in zip(expected, value))
    return expected == value


def _matches(config, params):
    return all(_same_type_value(getattr(config, key), value) for key, value in params.items())


def config_identity(config):
    """Stable producer/parameter identity, independent of compatibility IDs."""
    return hashlib.sha256((config.pipeline + '\n' + config.to_json()).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    """Validate actual producer/template limits; never consult the ID registry."""
    _require(isinstance(config, BpreshuffleConfig), 'expected BpreshuffleConfig')
    c = config
    _selected_pipelines([c.pipeline])
    nullable = {'split_k', 'max_m', 'max_k'}
    for name, item in _PUBLIC_FIELDS.items():
        value = getattr(c, name)
        if name in ('specializations', 'dispatch'):
            _require(type(value) is tuple, f'{name} must be a tuple')
        elif name in nullable and value is None:
            continue
        else:
            expected = bool if isinstance(item.default, bool) else int
            _require(type(value) is expected, f'{name} must be {expected.__name__}')
    _require(c.tile_k == 128, 'tile_k must be 128')
    _require(min(c.tile_m, c.tile_n, c.wave_m, c.wave_n, c.wave_k) > 0, 'tile and wave axes must be positive')
    block = 64 * c.wave_m * c.wave_n * c.wave_k
    _require(block <= 512 and c.tile_m % (16*c.wave_m) == 0 and c.tile_n % (16*c.wave_n) == 0,
             'tile dimensions must match MFMA16 wave geometry and block_size <= 512')
    _require(c.fixed_k == 0 or 0 < c.fixed_k <= 16384 and c.fixed_k % 128 == 0, 'fixed_k must be a K128 multiple <= 16384')
    _require(c.m_align > 0 and c.m_align <= c.tile_m and c.tile_m % c.m_align == 0, 'm_align must divide tile_m')
    _require(c.max_m is None or c.max_m > 0, 'max_m must be positive or null')
    _require(c.max_k is None or c.max_k > 0 and c.max_k % 128 == 0, 'max_k must be a K128 multiple or null')
    _require(c.max_tensor_bytes > 0 and c.max_tensor_bytes <= 2**63-1, 'max_tensor_bytes must fit signed int64')
    _require(c.sfa_alignment in (1,2,4,8,16) and c.c_alignment in (8,16), 'invalid scale/output alignment')
    _require(c.runtime_split_k and c.split_k is None or not c.runtime_split_k and c.split_k in (1,2,4,8), 'runtime split_k must be null; static split_k must be 1,2,4,8')
    # Inactive axes must retain their neutral values. Historical complete payloads
    # remain valid even when old descriptors recorded a non-neutral host default.
    historical = _BY_IDENTITY.get((c.pipeline, c.to_json()))
    if historical is None:
        for name in set(_PUBLIC_FIELDS) - _TUNABLE_AXES[c.pipeline]:
            default = _PUBLIC_FIELDS[name].default
            _require(_same_type_value(default, getattr(c, name)), f'{name} is unused by {c.pipeline}')
        _require(c.max_tensor_bytes <= (2**63-1 if c.pipeline == 'large_output' else 2**31-1), 'producer addressing extent exceeds signed byte contract')
        _require(c.c_alignment == 16 or c.pipeline == 'lds', 'producer requires 16-byte C alignment')
        _require(c.sfa_alignment == 16 or c.pipeline in ('lds','register'), 'producer requires 16-byte SFA alignment')
    if c.pipeline == 'pin':
        _require((c.tile_m,c.tile_n,c.wave_m,c.wave_n,c.wave_k)==(256,256,2,2,1), 'pin requires 256x256/2x2x1')
        _require(c.schedule in (0,1,2) and c.pin_agpr, 'pin requires schedule 0..2 and pin_agpr')
        _require(c.scale_panel in (16,32,64) and c.loop_unroll in (2,4), 'pin panel must be 16/32/64 and unroll 2/4')
        _require(c.schedule != 0 or c.m_align == 256, 'unpadded pin requires M256 alignment')
        _require(c.schedule == 0 or c.m_align >= (16 if c.schedule == 2 else 64), 'padded pin M alignment does not match schedule')
        _require(c.schedule != 2 or c.loop_unroll == 4, 'pin schedule 2 preserves unroll4 traits with disabled loop unrolling')
    elif c.pipeline == 'register':
        em,en=c.tile_m//(16*c.wave_m),c.tile_n//(16*c.wave_n)
        _require(1<=em<=4 and 1<=en<=8 and 1<=c.prefetch<=8, 'register E_M<=4,E_N<=8,prefetch1..8')
        _require(c.output_mode in (0,3,4) and c.b_cache in (0,1,2,3), 'invalid register output/cache')
        _require(not c.pad_n or c.output_mode==4, 'N tails require output_mode4')
        _require(not c.reuse_b_scale or 128%c.tile_n==0, 'B scale reuse requires tile_n divides128')
        _require(c.output_mode!=4 or c.wave_k>1, 'output_mode4 requires local K waves')
        _require((c.wave_k-(0 if c.output_mode==4 else 1))*c.tile_m*c.tile_n*4<=160*1024, 'register reduction exceeds LDS')
        _require(c.schedule==(1 if c.runtime_split_k else 0), 'register schedule must match runtime split-K')
        _require(not c.runtime_split_k or c.output_mode==3 and not c.pad_n, 'runtime register requires packed output without N tails')
        _require(c.runtime_split_k or c.split_k==1, 'register static global split-K unsupported')
        _require(c.reduce_vec in (4,8,12,16) and c.reduce_block in (64,128,256), 'invalid reduction geometry')
        _require(historical is not None or c.runtime_split_k or (c.reduce_vec,c.reduce_block)==(4,128), 'register reduction axes require runtime split-K')
        _require(c.max_k is not None and c.max_k<=16384, 'register requires max_k<=16384')
    elif c.pipeline == 'lds':
        waves=c.wave_m*c.wave_n
        em=c.tile_m//(16*c.wave_m)
        _require(c.wave_k==1 and c.tile_n%(waves*16)==0, 'LDS B geometry must span all waves')
        _require(c.fine_m_loads or c.tile_m%(block//8)==0, 'LDS partial A copy requires fine_m_loads')
        _require(1<=em<=(8 if c.fine_m_loads else 4), 'LDS E_M exceeds loader capacity')
        _require(c.output_mode in (1,2) and c.stages>=2*c.cluster>0 and c.stages%c.cluster==0, 'LDS stages must contain at least2 clusters')
        split=1 if c.runtime_split_k else c.split_k
        max_loops=(128+split-1)//split
        a=c.tile_m//8*(1024+(0 if c.xor_lds else 32));b=c.tile_n//8*(1024+(0 if c.xor_lds else 32))
        groups=(c.tile_n+127)//128
        _require(c.stages<=max_loops and groups*max_loops<=block, 'LDS stages/scales exceed fixed capacity')
        _require(c.stages*(a+(0 if c.b_direct else b))+(0 if c.register_scales else (c.tile_m+groups)*max_loops)<=160*1024, 'LDS ring exceeds160KiB')
        _require(not c.xor_lds or waves==4 and a%4096==0 and b%4096==0, 'XOR LDS requires aligned four-wave layout')
        _require(c.output_mode!=1 or c.tile_m*(c.tile_n+8)*2<=a+b and c.tile_m*c.tile_n%(block*8)==0, 'LDS output1 staging geometry unsupported')
        _require(c.b_direct == (c.schedule==2) and c.schedule in (0,2), 'LDS direct B requires schedule2')
        _require(not c.b_direct or 1<=c.b_ahead<=3, 'direct B ahead must be1..3')
        _require(c.b_direct or c.b_ahead==0, 'b_ahead requires direct B')
        _require(historical is not None or not c.b_direct or (c.cluster==1 and not c.prefetch_before_read and not c.read_only_drain), 'direct B does not consume cluster/read/drain policies')
        _require(historical is not None or not c.register_scales or not c.early_scale_loads, 'early_scale_loads is unused with register scales')
        _require(not c.runtime_split_k or c.fine_m_loads and not c.b_direct, 'runtime LDS requires fine M loader and LDS B')
        _require(not c.runtime_split_k or not c.specializations and not c.dispatch, 'runtime LDS does not consume static specializations or dispatch')
        _require(c.dynamic_lds and c.max_k is not None and c.max_k<=16384, 'LDS requires dynamic_lds and max_k<=16384')
        _require(c.reduce_vec in (4,8,12,16) and c.reduce_block in (64,128,256) and c.store_cache in (0,1,2,3), 'invalid LDS reduction/store policy')
        _require(historical is not None or c.runtime_split_k or c.split_k>1 or (c.reduce_vec,c.reduce_block,c.store_cache)==(4,128,0), 'LDS reduction/store axes require global split-K')
        for item in c.specializations:
            _require(len(item)==6 and all(type(v)is int for v in item), 'specialization must contain K,stages,cluster,vec,block,cache')
            fk,st,cl,rv,rb,sc=item
            validate_config(replace(c,fixed_k=fk,stages=st,cluster=cl,reduce_vec=rv,reduce_block=rb,store_cache=sc,specializations=(),dispatch=(),_legacy_kid=-1))
        for condition,payload in c.dispatch:
            _require(type(condition)is str and type(payload)is str, 'dispatch must contain condition/config JSON')
            target=construct_config('lds',config=payload)
            _require(target.split_k==c.split_k and target.runtime_split_k==c.runtime_split_k and target.max_m==c.max_m and target.m_align==c.m_align, 'dispatch must preserve launcher shape/split contract')
    elif c.pipeline == 'tiled':
        _require(not c.runtime_split_k and c.split_k==1 and c.wave_k==1, 'tiled supports direct output only')
        _require(c.schedule in (0,1,2,3,4), 'tiled schedule must be0..4')
        _require(c.loop_unroll==2, 'tiled loop unroll is fixed at2')
        _require(c.scale_panel>=8 and c.scale_panel&(c.scale_panel-1)==0, 'scale_panel must be power of two >=8')
        _require(c.max_k is not None and c.max_k<=16384, 'tiled requires max_k<=16384')
        _require(c.schedule not in (0,2) or c.m_align>=16, 'tiled raw SFA vectors require M16 alignment')
        if c.schedule==0:
            _require(c.wave_m==4 and c.wave_n==2 and c.tile_m in (128,192) and c.tile_n in (128,256) and c.stages==2 and c.scale_panel in (8,16,32,64,128), 'main tiled geometry/panel unsupported')
            _require(c.fixed_k>0 or c.max_k<=c.scale_panel*128, 'main runtime K must fit scale panel')
            _require(c.fixed_k<=c.scale_panel*128, 'main fixed K must fit scale panel')
        elif c.schedule in (1,2):
            _require(c.wave_m==c.wave_n==2 and c.tile_n==128 and c.tile_m in (64,96,128,160) and c.stages in (2,3) and c.scale_panel<=32, 'small tiled geometry/stages/panel unsupported')
            _require(c.schedule!=1 or c.tile_m==128, 'tiled schedule1 requires BM128')
        else:
            _require(c.wave_m==c.wave_n==2 and c.tile_m==64 and c.tile_n==(128 if c.schedule==3 else 64) and 2<=c.stages<=4 and c.scale_panel<=128, 'narrow tiled geometry/stages/panel unsupported')
            _require(c.schedule!=4 or c.stages==4, 'N64 schedule requires four stages')
            _require(c.fixed_k==0 or c.fixed_k<=c.scale_panel*128, 'narrow fixed K must fit scale panel')
        _require(c.group_m>=0 and (c.group_m==0 or c.schedule==4), 'group_m is only used by N64 tiled schedule')
        matrix=c.stages*(c.tile_m+c.tile_n)//8*1056
        lds=matrix+c.tile_m*c.scale_panel+(4*c.scale_panel if c.schedule==0 else ((c.tile_n+127)//128)*c.scale_panel)
        _require(c.tile_m*(c.tile_n+8)*2<=lds<=160*1024, 'tiled LDS/output arena exceeds capacity')
    else:
        _require((c.tile_m,c.tile_n,c.wave_m,c.wave_n,c.wave_k)==(192,256,4,2,1), 'large_output requires192x256/4x2x1')
        _require(c.schedule in (0,1) and c.b_direct==(c.schedule==1), 'large direct B requires schedule1')
        _require(c.scale_panel in (16,32,64,128), 'large scale panel must be16/32/64/128')
        _require(c.max_k is not None and (c.fixed_k>0 or c.max_k<=c.scale_panel*128) and c.fixed_k<=c.scale_panel*128, 'large K must fit scale panel')
        _require(c.m_align>=16, 'large raw SFA vectors require M16 alignment')
        _require(historical is not None or c.scale_panel<=32, 'large scale loader has no repeated K-panel pass above32')
        _require(not c.b_direct or c.fixed_k>=256, 'large direct-B prologue requires at least two fixed K128 tiles')
        _require(c.stages==(3 if c.b_direct else 2), 'large stages are fixed by movement schedule')
        _require(not c.b_direct or c.b_direct_sets==2 and c.c_chunk_rows>0 and c.tile_m%c.c_chunk_rows==0 and c.c_chunk_rows%16==0 and c.c_chunk_rows*c.tile_n%(block*8)==0, 'large direct B requires2sets and valid C chunks')
        _require(c.b_direct or c.b_direct_sets==c.c_chunk_rows==0, 'large chunk axes require direct B')
        lds=c.stages*(c.tile_m+(0 if c.b_direct else c.tile_n))//8*1056+c.tile_m*c.scale_panel+4*c.scale_panel
        output=c.c_chunk_rows if c.b_direct else c.tile_m
        _require(output*(c.tile_n+8)*2<=lds<=160*1024, 'large LDS/output arena exceeds capacity')
    return historical if historical is not None else (config if config.legacy_kid == -1 else replace(config, _legacy_kid=-1))


def construct_config(pipeline, config=None, **compile_params):
    """Construct a producer configuration from legal axes without registering ID.

    Named axes inherit a geometry-compatible catalog seed. A complete payload
    retains every explicit value. Inactive axes are rejected by validation.
    """
    name=pipeline.name if isinstance(pipeline,BpreshufflePipeline) else pipeline
    _selected_pipelines([name])
    if config is not None and compile_params:
        raise ValueError('provide a config or named compile parameters, not both')
    if hasattr(config, "pipeline") and config.pipeline!=name:
        raise ValueError('config belongs to a different pipeline')
    params=_params(config if config is not None else compile_params)
    exact=[c for c in BPRESHUFFLE_PIPELINES[name].configs if _matches(c,params)]
    if len(exact)==1:
        return validate_config(exact[0])
    if set(params)==set(_PUBLIC_FIELDS):
        candidate=BpreshuffleConfig(pipeline=name,**params)
    else:
        selectors=('tile_m','tile_n','wave_m','wave_n','wave_k','runtime_split_k','b_direct','schedule','output_mode')
        seeds=list(BPRESHUFFLE_PIPELINES[name].active_configs)
        compatible=[c for c in seeds if all(getattr(c,key)==params[key] for key in selectors if key in params)]
        if not compatible:
            # New geometry uses a producer default, then inherits explicit axes.
            default_kid={'pin':9000,'tiled':92002,'register':9040,'lds':92430,'large_output':92500}[name]
            seed=config_from_legacy_kid(default_kid)
        else:
            seed=min(compatible,key=lambda c: sum(getattr(c,key)!=value for key,value in params.items()))
        candidate=replace(seed,**params,_legacy_kid=-1)
        neutral={key: _PUBLIC_FIELDS[key].default for key in set(_PUBLIC_FIELDS)-_TUNABLE_AXES[name] if key not in params}
        candidate=replace(candidate,**neutral)
        if name=='pin' and 'fixed_k' not in params:
            candidate=replace(candidate,fixed_k=0)
        if name in ('register','lds') and not candidate.runtime_split_k and candidate.split_k==1:
            inactive={'reduce_vec':4,'reduce_block':128}
            if name=='lds':
                inactive['store_cache']=0
            candidate=replace(candidate,**{key:value for key,value in inactive.items() if key not in params})
        if candidate.runtime_split_k:
            changes={}
            if 'split_k' not in params:
                changes['split_k']=None
            if 'schedule' not in params:
                changes['schedule']=1 if name=='register' else 0
            candidate=replace(candidate,**changes)
        elif name in ('register','lds') and 'runtime_split_k' in params and not params['runtime_split_k']:
            changes={}
            if 'split_k' not in params:
                changes['split_k']=1
            if 'schedule' not in params:
                changes['schedule']=2 if candidate.b_direct else 0
            candidate=replace(candidate,**changes)
        if name in ('pin','tiled','large_output','lds'):
            candidate=replace(candidate,**{key:() for key in ('specializations','dispatch') if key not in params})
        if name=='lds' and candidate.b_direct:
            candidate=replace(candidate,**{key:value for key,value in {'cluster':1,'prefetch_before_read':False,'read_only_drain':False}.items() if key not in params})
        if name=='lds' and candidate.register_scales and 'early_scale_loads' not in params:
            candidate=replace(candidate,early_scale_loads=False)
        if name=='large_output' and 'scale_panel' in params and 'max_k' not in params:
            candidate=replace(candidate,max_k=min(seed.max_k or 16384,candidate.scale_panel*128))
    candidate=validate_config(candidate)
    return _BY_IDENTITY.get((candidate.pipeline,candidate.to_json()),candidate)


def resolve_config(pipeline, config=None, **compile_params):
    """Select a unique catalog entry or construct an explicit legal tuple."""
    name=pipeline.name if isinstance(pipeline,BpreshufflePipeline) else pipeline
    _selected_pipelines([name])
    if config is not None and compile_params:
        raise ValueError('provide a config or named compile parameters, not both')
    params=_params(config if config is not None else compile_params)
    if hasattr(config, "pipeline") and config.pipeline!=name:
        raise ValueError('config belongs to a different pipeline')
    matches=[c for c in BPRESHUFFLE_PIPELINES[name].configs if _matches(c,params)]
    if len(matches)==1:
        return validate_config(matches[0])
    if len(matches)>1 and config is None and not {'tile_m','tile_n'}<=set(params):
        raise ValueError(f'ambiguous {name} compile parameters; matched {len(matches)} configurations')
    return construct_config(name,config=params)


def config_supports_shape(config,m,n,k):
    c=config
    if any(type(x)is not int or x<=0 for x in (m,n,k)):
        return False
    if (m%c.m_align or n%128 or (not c.pad_n and n%c.tile_n) or k%128
            or c.max_m is not None and m>c.max_m or c.max_k is not None and k>c.max_k
            or c.fixed_k and k!=c.fixed_k):
        return False
    if c.pipeline=='large_output':
        return max(m*k,n*k,2*((c.tile_m-1)*n+c.tile_n))<=2**31-1<2*m*n<=c.max_tensor_bytes
    return max(m*k,n*k,(4 if not c.runtime_split_k and c.split_k>1 else 2)*m*n)<=c.max_tensor_bytes


supports_shape=config_supports_shape


def pipeline_configs(shape,pipelines=None,*,gfx='gfx950',outdtype='bf16',include_legacy=False,families=None):
    selected=_selected_pipelines(pipelines)
    if gfx!='gfx950' or outdtype not in ('bf16','bf16_t'):
        return ()
    if isinstance(families,str):
        families={part.strip() for part in families.split(',')}
    known={row['family'] for row in CATALOG_ROWS}
    if families is not None and (set(families)-known or not families):
        raise ValueError(f'invalid MXFP8 B-preshuffle families: {sorted(set(families)-known)}')
    return tuple(c for kid,c in _CONFIGS_BY_KID.items() if c.pipeline in selected
                 and (include_legacy or kid in ACTIVE_LEGACY_KIDS)
                 and (families is None or CATALOG_BY_LEGACY_KID[kid]['family'] in families)
                 and config_supports_shape(c,*shape))


def validate_saved_config(pipeline,config,kernel_id):
    name=pipeline.name if isinstance(pipeline,BpreshufflePipeline) else pipeline
    params=_params(config)
    if kernel_id==-1:
        expected=construct_config(name,config=params)
        _require(set(params)==set(expected.compile_params), 'saved config must contain complete parameters')
    else:
        expected=config_from_legacy_kid(kernel_id)
    if name!=expected.pipeline or set(params)!=set(expected.compile_params) or not _matches(expected,params):
        raise ValueError(f'pipeline/config does not match kernelId {kernel_id}')
    return expected


def get_config_dependencies(config):
    """Dispatch targets embedded into the selected producer's generated TU."""
    return tuple(construct_config(config.pipeline,config=payload) for _,payload in config.dispatch)

def config_traits(config, specialization=None):
    """Emit the complete traits template from canonical producer parameters."""
    c=config
    if specialization:
        fk,st,cl,rv,rb,sc=specialization
        c=replace(c,fixed_k=fk,stages=st,cluster=cl,reduce_vec=rv,reduce_block=rb,store_cache=sc)
    def tpl(name,*args):
        return name+'<'+', '.join(str(x).lower() if type(x)is bool else str(x) for x in args)+'>'
    if c.pipeline=='pin':
        return tpl('opus_gemm_mxscale_bpreshuffle_pin_config_traits',c.scale_panel,c.fixed_k,c.scale_reset,c.loop_unroll)
    if c.pipeline=='register':
        base=tpl('opus_gemm_small_register_traits_gfx950',c.tile_m,c.tile_n,c.wave_m,c.wave_n,c.prefetch,c.wave_k,c.output_mode,c.b_cache,c.fixed_k,c.pad_n,c.reuse_b_scale)
        if c.runtime_split_k:
            base=tpl('opus_gemm_mxscale_bpreshuffle_runtime_config_traits',base,c.reduce_vec,c.reduce_block)
        return base
    if c.pipeline=='lds':
        base=tpl('opus_gemm_small_lds_traits_gfx950',c.tile_m,c.tile_n,c.wave_m,c.wave_n,c.stages,c.cluster,c.output_mode,c.register_scales,c.xor_lds,c.early_scale_loads,c.prefetch_before_read,c.read_only_drain,1 if c.runtime_split_k else c.split_k,c.reduce_vec,c.reduce_block,c.store_cache,c.fixed_k,c.fine_m_loads)
        if c.b_direct:
            base=tpl('opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits',base,c.b_ahead)
        if c.runtime_split_k:
            base=tpl('opus_gemm_mxscale_bpreshuffle_runtime_config_traits',base,c.reduce_vec,c.reduce_block)
        return base
    if c.pipeline=='large_output':
        return tpl('opus_gemm_mxscale_bpreshuffle_large_config_traits',c.scale_panel,c.fixed_k,c.b_direct,c.c_chunk_rows)
    if c.schedule==0:
        return tpl('opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950',c.tile_m,c.tile_n,c.scale_panel,c.fixed_k)
    if c.schedule in (1,2):
        row=CATALOG_BY_LEGACY_KID.get(c.legacy_kid)
        if row and row['family']=='shortk':
            return tpl('opus_gemm_mxscale_bpreshuffle_shortk_traits',c.fixed_k)
        return tpl('opus_gemm_mxscale_bpreshuffle_small_config_traits',c.tile_m,c.stages,c.scale_panel,c.fixed_k,c.schedule)
    if c.schedule==3:
        return tpl('opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950',c.stages,c.scale_panel,c.fixed_k)
    return tpl('opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950',c.scale_panel,c.fixed_k,c.group_m)


def kernel_instance_from_config(config, *, canonical_codegen=True):
    """Create a launcher instance directly from scalar parameters.

    Legacy configurations retain their ABI metadata/name. Unregistered tuples
    receive a stable parameter hash name and use a module-local ABI key.
    """
    c=validate_config(config)
    try:
        from .opus_gemm_common import OpusGemmInstance
        from .opus_gemm_bpreshuffle_variants import BpreshuffleVariant, NEW_BPRESHUFFLE_VARIANTS_BY_KID
    except ImportError:
        from opus_gemm_common import OpusGemmInstance
        from opus_gemm_bpreshuffle_variants import BpreshuffleVariant, NEW_BPRESHUFFLE_VARIANTS_BY_KID
    row=CATALOG_BY_LEGACY_KID.get(c.legacy_kid)
    if row is not None and _matches(c,_params(row['compile_params'])):
        abi=dict(row['abi'])
        for item in fields(OpusGemmInstance):
            if isinstance(item.default, tuple) and item.name in abi:
                abi[item.name] = tuple(abi[item.name])
        if abi.get('bpreshuffle_variant'):
            abi['bpreshuffle_variant']=NEW_BPRESHUFFLE_VARIANTS_BY_KID[c.legacy_kid]
        abi['bpreshuffle_specializations']=tuple(tuple(x) for x in abi.get('bpreshuffle_specializations',()))
        abi['bpreshuffle_dispatch']=tuple(tuple(x) for x in abi.get('bpreshuffle_dispatch',()))
        instance=OpusGemmInstance(**abi)
    else:
        split=4 if c.runtime_split_k else c.split_k
        family='fine_lds' if c.pipeline=='lds' else 'large_output' if c.pipeline=='large_output' else c.pipeline
        variant=BpreshuffleVariant(
            -1,'config_'+config_identity(c)[:20],family,
            f'gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{c.pipeline}_gfx950.cuh',
            'gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh',
            f'opus_gemm_mxscale_bpreshuffle_{c.pipeline}_kernel',config_traits(c),
            c.tile_m,c.tile_n,c.wave_m,c.wave_n,c.wave_k,c.m_align,c.max_m,c.fixed_k,
            split,c.dynamic_lds,c.pin_agpr,c.reduce_vec,c.reduce_block,c.sfa_alignment,
            c.c_alignment,c.runtime_split_k,c.schedule)
        instance=OpusGemmInstance(
            64*c.wave_m*c.wave_n*c.wave_k,c.tile_m,c.tile_n,128,c.wave_m,c.wave_n,
            16,16,128,16,16,4,1,128,128,'a8w8_mxscale_gemm_bpreshuffle',['bf16_t'],
            WG_PER_CU=1,has_oob=c.m_align<c.tile_m,arch_prefix='gfx950',
            direct_only=not c.runtime_split_k and c.split_k==1,scale_dtype='e8m0',
            max_tensor_bytes=c.max_tensor_bytes,pad_m=c.m_align<c.tile_m,max_m=c.max_m,max_k=c.max_k,
            bpreshuffle_split_k=split,bpreshuffle_stages=c.stages,bpreshuffle_cluster=c.cluster,
            bpreshuffle_reduce_vec=c.reduce_vec,bpreshuffle_reduce_block=c.reduce_block,
            bpreshuffle_store_cache=c.store_cache,bpreshuffle_fixed_k=c.fixed_k,bpreshuffle_pad_n=c.pad_n,
            bpreshuffle_loop_unroll=c.loop_unroll,bpreshuffle_scale_reset=c.scale_reset,
            bpreshuffle_specializations=c.specializations,bpreshuffle_variant=variant,
            splitk_workspace_dtype='fp32_t' if c.runtime_split_k or c.split_k>1 else None)
    instance.bpreshuffle_config=c
    instance.bpreshuffle_config_codegen=canonical_codegen
    return instance
