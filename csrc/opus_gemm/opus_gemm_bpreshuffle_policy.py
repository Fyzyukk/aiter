# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tuned lookup and legal fallback for gfx950 B-preshuffle GEMM.

The shape key is exactly ``(gfx, cu_num, M, N, K)``. The default production
CSV and dedicated ``OPUS_BPRESHUFFLE_TUNED_CONFIG`` override both accept old
kernelId rows and pipeline/config rows. Other backends never select an OPUS
configuration. A missing or stale default row falls back to the catalog;
malformed explicit files and stale matching rows in an explicit file raise
``TunedConfigError`` with the file and line number.

Fallback scores describe grid coverage and tile padding, not measured speed.
Compile-time splits remain part of the config; their launcher argument is
zero. Automatic runtime splits are validated literal positive counts. Saved
or explicitly requested runtime zero preserves the historical default ABI.
This module never imports torch, aiter, HIP, or HSA or queries a device.
"""

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path

try:
    from . import opus_gemm_bpreshuffle_config as _catalog
    from . import opus_gemm_common as _common
except ImportError:  # Directly executable scalar OPUS tools.
    import opus_gemm_bpreshuffle_config as _catalog
    import opus_gemm_common as _common


TUNED_CONFIG_ENV = "OPUS_BPRESHUFFLE_TUNED_CONFIG"
DEFAULT_TUNED_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv"
)


class TunedConfigError(ValueError):
    """An explicitly selected tuning file cannot be used as requested."""


@dataclass(frozen=True)
class BpreshuffleSelection:
    config: _catalog.BpreshuffleConfig
    split_k: int
    source: str


@dataclass(frozen=True)
class _TunedRow:
    line: int
    values: dict


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _shape(shape):
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must contain three positive integers (M, N, K)") from exc
    if len(values) != 3:
        raise ValueError("shape must contain three positive integers (M, N, K)")
    return tuple(_positive_integer(value, name) for name, value in zip("MNK", values))


def _csv_integer(value, label, *, minimum=0):
    # Accept decimal integer spelling, including whitespace, and reject floats,
    # exponent notation and empty cells instead of truncating them.
    token = str(value).strip()
    digits = token[1:] if token.startswith("-") and minimum < 0 else token
    if not digits or not digits.isascii() or not digits.isdecimal():
        raise ValueError(f"{label} must be an integer >= {minimum}")
    result = int(token)
    if result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _row_error(path, line, exc):
    return TunedConfigError(f"invalid OPUS tuned config {path}:{line}: {exc}")


@lru_cache(maxsize=16)
def _load_tuned_rows(path, mtime_ns, size, inode, strict):
    """Cache scalar rows while noticing replacements and ordinary CSV edits."""
    del mtime_ns, size, inode
    rows = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = reader.fieldnames or []
            required = {"gfx", "cu_num", "M", "N", "K", "libtype", "splitK"}
            missing = required - set(columns)
            if missing:
                raise _row_error(path, 1, f"missing columns {sorted(missing)}")
            if len(columns) != len(set(columns)):
                raise _row_error(path, 1, "duplicate CSV columns")
            if "kernelId" not in columns and not {"pipeline", "config"} <= set(columns):
                raise _row_error(path, 1, "requires kernelId or both pipeline and config")
            if ("pipeline" in columns) != ("config" in columns):
                raise _row_error(path, 1, "requires both pipeline and config columns")
            for row in reader:
                if str(row.get("libtype", "")).strip() != "opus":
                    continue
                line = reader.line_num
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("CSV row has the wrong number of fields")
                    gfx = row["gfx"].strip()
                    if not gfx:
                        raise ValueError("gfx must be nonempty")
                    key = (gfx, *(_csv_integer(row[name], name, minimum=1)
                                  for name in ("cu_num", "M", "N", "K")))
                    rows.setdefault(key, []).append(_TunedRow(line, row))
                except ValueError as exc:
                    if strict:
                        raise _row_error(path, line, exc) from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _row_error(path, getattr(locals().get("reader"), "line_num", 1), exc) from exc
    return {key: tuple(sorted(value, key=_timing_key)) for key, value in rows.items()}


def _timing_key(row):
    """Prefer measured duplicate rows, preserving file order for absent timing."""
    try:
        us = float(row.values.get("us", ""))
    except (TypeError, ValueError):
        return math.inf
    return us if math.isfinite(us) and us > 0 else math.inf


def _tuned_path(tuned_file):
    if tuned_file is not None:
        return Path(tuned_file), True
    override = os.environ.get(TUNED_CONFIG_ENV)
    return (Path(override), True) if override else (DEFAULT_TUNED_CONFIG, False)


def _instance(config):
    constructor = getattr(_catalog, "kernel_instance_from_config", None)
    if constructor is not None:
        return constructor(config)
    return _common.a8w8_mxscale_gemm_bpreshuffle_kernels_list[config.legacy_kid]


def _launch_split(config, shape, requested, cu_num):
    instance = _instance(config)
    if not config.runtime_split_k:
        if requested not in (None, 0):
            raise ValueError("this compile configuration has fixed split-K; use split_k=0")
        _common.bpreshuffle_launch_plan(instance, *shape, 0, cu_num)
        return 0
    automatic = requested in (None, -1)
    if requested == 0:
        # Zero is an explicit historical ABI request. In particular, old
        # short-K register rows can intentionally contain empty partitions.
        _common.bpreshuffle_launch_plan(instance, *shape, 0, cu_num)
        return 0
    try:
        plan = _common.bpreshuffle_launch_plan(
            instance, *shape, -1 if automatic else requested, cu_num
        )
        _common.bpreshuffle_launch_plan(instance, *shape, plan.split_k, cu_num)
        return plan.split_k
    except ValueError:
        if not automatic:
            raise
    # A grid-based split may exceed dynamic LDS. Choose the nearest legal
    # literal count when another partition count makes that config launchable.
    m, n, k = shape
    grid = ((m + config.tile_m - 1) // config.tile_m
            * ((n + config.tile_n - 1) // config.tile_n))
    desired = min(16, k // 128, max(1, (cu_num + grid - 1) // grid))
    legal = _common.bpreshuffle_candidate_split_k(instance, *shape, cu_num)
    if not legal:
        raise ValueError("this compile configuration has no legal runtime split-K")
    return min(legal, key=lambda split: (abs(split - desired), split))


def _selected_pipelines(pipeline):
    # The catalog owns validation and accepts names, pipeline objects, and
    # comma-separated filters. A pipeline-only request still uses tuned rows.
    if isinstance(pipeline, _catalog.BpreshufflePipeline):
        pipeline = pipeline.name
    return _catalog._selected_pipelines(pipeline)


def _parameter_saved_config(pipeline, payload):
    config = _catalog.resolve_config(pipeline, payload)
    # Selection accepts named partial axes, while a saved config contains the
    # complete canonical tuple so future seed changes cannot alter its replay.
    return _catalog.validate_saved_config(pipeline, payload, config.legacy_kid)


def _saved_config(row):
    pipeline = row.get("pipeline", "").strip()
    payload = row.get("config", "").strip()
    token = row.get("kernelId", "").strip()
    if pipeline or payload:
        if not pipeline or not payload:
            raise ValueError("OPUS rows require both pipeline and config when either is present")
        if token:
            kernel_id = _csv_integer(token, "kernelId", minimum=-1)
            if kernel_id == -1:
                # -1 denotes omitted compatibility metadata. The canonical
                # tuple can also coincide with an existing legacy config.
                return _parameter_saved_config(pipeline, payload)
            return _catalog.validate_saved_config(
                pipeline, payload, kernel_id
            )
        return _parameter_saved_config(pipeline, payload)
    if not token:
        raise ValueError("OPUS rows require kernelId or pipeline/config")
    return _catalog.config_from_legacy_kid(_csv_integer(token, "kernelId"))


def _lookup_tuned(shape, selected, split_k, gfx, cu_num, tuned_file):
    path, strict = _tuned_path(tuned_file)
    try:
        stat = path.stat()
        rows = _load_tuned_rows(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino, strict)
    except (OSError, TunedConfigError) as exc:
        if strict:
            if isinstance(exc, TunedConfigError):
                raise
            raise TunedConfigError(f"cannot read OPUS tuned config {path}: {exc}") from exc
        return None
    candidates = rows.get((gfx, cu_num, *shape), ())
    for saved in candidates:
        row = saved.values
        # An unrelated pipeline's stale payload cannot affect a filtered call.
        row_pipeline = row.get("pipeline", "").strip()
        if row_pipeline in _catalog.BPRESHUFFLE_PIPELINE_NAMES and row_pipeline not in selected:
            continue
        try:
            config = _saved_config(row)
            if config.pipeline not in selected:
                continue
            literal = _launch_split(config, shape, _csv_integer(row["splitK"], "splitK"), cu_num)
            if split_k is not None:
                try:
                    requested = _launch_split(config, shape, split_k, cu_num)
                except ValueError:
                    continue
                if literal != requested:
                    continue
            return BpreshuffleSelection(config, literal, "tuned")
        except (ValueError, KeyError) as exc:
            if strict:
                raise _row_error(path, saved.line, exc) from exc
    return None


def _fallback_score(config, shape, literal, cu_num):
    m, n, _ = shape
    order = (("register", "lds", "tiled", "pin", "large_output") if m <= 32 else
             ("lds", "tiled", "register", "pin", "large_output") if m <= 256 else
             ("pin", "tiled", "lds", "register", "large_output"))
    grid_m = (m + config.tile_m - 1) // config.tile_m
    grid_n = (n + config.tile_n - 1) // config.tile_n
    partitions = config.split_k
    if config.runtime_split_k:
        partitions = literal or _common.bpreshuffle_launch_plan(
            _instance(config), *shape, 0, cu_num).split_k
    total_grid = grid_m * grid_n * partitions
    padding = grid_m * config.tile_m * grid_n * config.tile_n - m * n
    return (order.index(config.pipeline), abs(total_grid - cu_num), padding,
            partitions, -bool(config.fixed_k), config.to_json())


def _explicit_config(pipeline, config, compile_params):
    if isinstance(config, _catalog.BpreshuffleConfig) and pipeline is None:
        pipeline = config.pipeline
    selected = _selected_pipelines(pipeline)
    if len(selected) != 1:
        raise ValueError("an explicit config or compile parameters require one pipeline")
    if config is not None and compile_params:
        raise ValueError("provide a config or compile parameters, not both")
    if compile_params is not None and not isinstance(compile_params, Mapping):
        raise ValueError("compile_params must be a mapping")
    return _catalog.resolve_config(next(iter(selected)), config, **(compile_params or {}))


def select_config(shape, pipeline=None, compile_params=None, config=None, split_k=None,
                  *, gfx="gfx950", cu_num=256, tuned_file=None):
    """Select one legal compile configuration and an exact launch split.

    A config object can supply its own pipeline. A JSON config or named axes
    require one pipeline. A pipeline without axes filters tuned lookup and
    fallback. ``split_k=None`` uses the saved tuned split when present and a
    grid/CU heuristic otherwise; an explicit split filters tuned rows before
    selecting a compatible fallback. Explicit configurations bypass tuning.
    Runtime ``split_k=0`` explicitly preserves its historical default, while
    ``None`` and ``-1`` fallback requests always produce a positive literal.
    """
    shape = _shape(shape)
    _positive_integer(cu_num, "cu_num")
    if type(gfx) is not str or not gfx:
        raise ValueError("gfx must be a nonempty architecture string")
    if gfx != "gfx950":
        raise ValueError("MXFP8 B-preshuffle compile configurations require gfx950")
    if split_k is not None and (type(split_k) is not int or split_k < -1):
        raise ValueError("split_k must be an integer >= -1 or None")
    if compile_params is not None and not isinstance(compile_params, Mapping):
        raise ValueError("compile_params must be a mapping")
    if config is not None or compile_params:
        chosen = _explicit_config(pipeline, config, compile_params)
        return BpreshuffleSelection(chosen, _launch_split(chosen, shape, split_k, cu_num), "explicit")
    selected = _selected_pipelines(pipeline)
    saved = _lookup_tuned(shape, selected, split_k, gfx, cu_num, tuned_file)
    if saved is not None:
        return saved
    choices = []
    for candidate in _catalog.pipeline_configs(shape, selected, gfx=gfx):
        try:
            literal = _launch_split(candidate, shape, split_k, cu_num)
        except ValueError:
            continue
        choices.append((candidate, literal))
    if not choices:
        raise ValueError(f"no legal OPUS B-preshuffle configuration for shape {shape}, "
                         f"gfx={gfx}, pipelines={sorted(selected)}, split_k={split_k}")
    chosen, literal = min(choices, key=lambda item:
                          _fallback_score(item[0], shape, item[1], cu_num))
    return BpreshuffleSelection(chosen, literal, "heuristic")
