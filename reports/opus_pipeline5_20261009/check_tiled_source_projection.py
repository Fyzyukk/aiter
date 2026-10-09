"""Read-only source projection checks; no imports from GPU-facing packages.

Project the new shared main/geometry bodies onto their frozen schedules.
Normalization is deliberately limited to the consolidation's approved changes:
comment/whitespace removal, compile-time schedule selection and unused lambdas,
fixed/runtime loop expression, equivalent per-pass layout tuple declarations,
and movement of the independent read_scales lambda definition.
The narrow schedules are reviewed separately in tiled_source_review.md.
"""

from pathlib import Path
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
OLD = REPORT / "before/csrc/opus_gemm/include/gfx950"
NEW = ROOT / "csrc/opus_gemm/include/gfx950"
PIPELINE = NEW / "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_tiled_gfx950.cuh"
LAYOUT = NEW / "opus_gemm_mxscale_bpreshuffle_tiled_layout_gfx950.cuh"


def end_block(source, opening):
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return end


def span(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    return start, opening, end_block(source, opening)


def body(source, marker):
    _, opening, end = span(source, marker)
    return source[opening + 1:end - 1]


def without_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def tokens(source):
    return re.findall(r"\w+|[^\s]", without_comments(source))


def drop_lambda(source, name):
    start, _, end = span(source, "auto " + name + " =")
    while source[end].isspace():
        end += 1
    assert source[end] == ";"
    return source[:start] + source[end + 1:]


def select_schedule(source, schedule_one):
    marker = "if constexpr (T::SCHEDULE == 1)"
    while marker in source:
        start, opening, end = span(source, marker)
        chain_end = end
        # Account for the internal issue lambda's else-if/else chain too.
        while match := re.match(r"\s*else\b", source[chain_end:]):
            next_opening = source.index("{", chain_end + match.end())
            chain_end = end_block(source, next_opening)
        match = re.match(r"\s*else\s*", source[end:])
        other = source[end + match.end():chain_end] if match else ""
        if other.startswith("{"):
            other = other[1:-1]
        selected = source[opening + 1:end - 1] if schedule_one else other
        source = source[:start] + selected + source[chain_end:]
    return source


def canonical_geometry(source):
    source = without_comments(source)
    source = re.sub(r"(?:const|constexpr) int loops = [^;]+;", "LOOPS_EXPRESSION;", source)
    # Frozen 128 uses one layout; 160 uses an explicit one/two-pass tuple;
    # geometry and the new shared body use the same transform_tuple expression.
    source = re.sub(r"^\s*const auto u_[gs]sfa(?:_\d)? =[^\n]+", "", source, flags=re.M)
    source = re.sub(r"^\s*const auto sfa_gmem_offsets =[^\n]+", "SFA_GMEM_LAYOUTS;", source, flags=re.M)
    source = re.sub(r"^\s*const auto sfa_smem_offsets =[^\n]+", "SFA_SMEM_LAYOUTS;", source, flags=re.M)
    source = re.sub(r"sfa_(gmem|smem)_offsets\[pass\]", r"opus::get<pass>(sfa_\1_offsets)[0]", source)
    start, _, end = span(source, "auto read_scales =")
    while source[end].isspace():
        end += 1
    assert source[end] == ";"
    definition = source[start:end + 1]
    source = source[:start] + source[end + 1:]
    start = source.index("auto load_a_fragment =")
    source = source[:start] + definition + source[start:]
    return tokens(source)


checks = []


def compare(name, frozen, current):
    checks.append({"name": name, "equal": frozen == current,
                   "frozen_tokens": len(frozen), "current_tokens": len(current)})
    assert frozen == current, name


pipeline = PIPELINE.read_text()
layout = LAYOUT.read_text()
for tag in ("8wave_192x256", "4wave_128x128", "4wave_64x128"):
    frozen = (OLD / f"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{tag}_gfx950.cuh").read_text()
    namespace = "namespace opus_gemm_" + tag + "_layout"
    compare("layout_" + tag, tokens(body(frozen, namespace)), tokens(body(layout, namespace)))

frozen = (OLD / "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh").read_text()
compare("schedule0_main", tokens(body(frozen, "void gemm_a8w8_mxfp8_scale_8wave_192x256_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs)")),
        tokens(body(pipeline, "void main(")))

for tag, symbol, schedule_one in (
    ("4wave_128x128", "gemm_a8w8_mxfp8_scale_4wave_128x128_kernel", True),
    ("geometry", "opus_gemm_mxscale_bpreshuffle_geometry_kernel", False),
    ("4wave_160x128", "gemm_a8w8_mxfp8_scale_4wave_160x128_kernel", False),
    ("shortk", "opus_gemm_mxscale_bpreshuffle_shortk_kernel", False),
):
    current = body(pipeline, "void geometry(")
    unused = ("load_sfa_panel", "load_sfb_panel") if schedule_one else (
        "issue_sfa_panel", "issue_sfb_panel", "publish_sfa_panel", "publish_sfb_panel")
    for name in unused:
        current = drop_lambda(current, name)
    if not schedule_one:
        current = re.sub(r"^\s*(?:array<vector_t<D_SF, T::VEC_SCALE_A>, T::SFA_PASSES> raw_sfa_panel|vector_t<D_SF, 1> raw_sfb_panel);", "", current, flags=re.M)
    current = select_schedule(current, schedule_one)
    frozen = (OLD / f"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{tag}_gfx950.cuh").read_text()
    frozen = body(frozen, f"void {symbol}(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs)")
    compare("schedule1_" + tag if schedule_one else "schedule2_" + tag,
            canonical_geometry(frozen), canonical_geometry(current))

print(json.dumps({"scope": "CPU source projection only; no GPU import, query, compile, or execution",
                  "pipeline_sha256": hashlib.sha256(PIPELINE.read_bytes()).hexdigest(),
                  "layout_sha256": hashlib.sha256(LAYOUT.read_bytes()).hexdigest(),
                  "checks": checks}, indent=2))
